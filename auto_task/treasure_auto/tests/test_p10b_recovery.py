import json
import sqlite3
from dataclasses import replace
from unittest.mock import patch
from test_p10b_support import StoreCase, c, m, s, stop_report


class RecoveryTests(StoreCase):
    def timer(self, now=280, sequence=1):
        return m.TimerRecord(1, 'sim-run-1', s.stamp(100), s.stamp(700), s.stamp(now),
                             max(0, 700 - now), 1.0, sequence, m.RunState.OFFLINE, False)

    def test_T01_180_then_10_seconds_restores_410_using_injected_policy(self):
        self.clock.advance(170)
        self.assertEqual(self.store.refresh_timer(self.timer()).status, m.CommitStatus.COMMITTED)
        self.clock.advance(10)
        self.reopen()
        got = self.store.register_recovery(self.recovery())
        self.assertEqual(got.remaining_s, 410)
        self.assertEqual(got.snapshot.run.deadline, s.stamp(700))
        self.assertTrue(got.snapshot.run.ever_restarted)
        self.assertEqual(got.snapshot.run.state, m.RunState.RECOVERING)
        self.assertTrue(self.store.actions_blocked)
        record = c.loads((self.root / 'runs/sim-run-1/timer.txt').read_text(encoding='utf-8'))
        self.assertTrue(record.ever_restarted)
        self.assertEqual(record.remaining_s, 410)

    def test_T01_delayed_timer_and_sequence_cannot_extend_deadline(self):
        self.clock.advance(180)
        self.assertEqual(self.store.refresh_timer(self.timer()).status, m.CommitStatus.COMMITTED)
        self.assertNotEqual(self.store.refresh_timer(self.timer(270, 0)).status, m.CommitStatus.COMMITTED)
        self.reopen()
        got = self.store.register_recovery(self.recovery())
        self.assertEqual(got.remaining_s, 410)

    def test_T03_corrupt_timer_blocks_instead_of_starting_600(self):
        self.store.close()
        path = self.root / 'runs/sim-run-1/timer.txt'
        path.write_text('420\ntruncated', encoding='utf-8')
        self.store = self.open_store()
        got = self.store.register_recovery(self.recovery())
        self.assertNotEqual(got.status, c.Outcome.SUCCESS)
        self.assertIsNone(got.remaining_s)
        self.assertTrue(self.store.actions_blocked)
        self.assertEqual(path.read_text(encoding='utf-8'), '420\ntruncated')

    def test_T03_missing_snapshot_is_not_recreated(self):
        self.store.close()
        path = self.root / 'runs/sim-run-1/state.sqlite3'
        path.unlink()
        self.store = self.open_store()
        self.assertNotEqual(self.store.register_recovery(self.recovery()).status, c.Outcome.SUCCESS)
        self.assertNotEqual(self.store.initialize_run(self.initial).status, m.CommitStatus.COMMITTED)
        self.assertFalse(path.exists())

    def test_T03_cross_boot_old_run_and_wrong_original_deadline_fail_closed(self):
        self.reopen()
        request = self.recovery()
        wrong_run = replace(request, envelope=replace(request.envelope, run_id='old-run'))
        self.assertNotEqual(self.store.register_recovery(wrong_run).status, c.Outcome.SUCCESS)
        wrong_deadline = replace(request, original_deadline=s.stamp(800))
        self.assertNotEqual(self.store.register_recovery(wrong_deadline).status, c.Outcome.SUCCESS)
        self.clock._now = s.stamp(110, 'other-boot')
        self.assertNotEqual(self.store.register_recovery(request).status, c.Outcome.SUCCESS)
        self.assertTrue(self.store.actions_blocked)

    def test_T02_each_interruption_boundary_demands_review_never_replays(self):
        for index, phase in enumerate(('intent', 'submitted', 'echoed', 'observed')):
            intent = s.arm_intent()
            if index == 0:
                self.store.commit_intent(intent)
            else:
                result = replace(s.timeout_result(), phase=phase,
                                 physical=c.Fact.CONFIRMED if phase == 'observed' else c.Fact.UNKNOWN,
                                 evidence=(s.evidence(),) if phase == 'observed' else ())
                self.store.append_event(m.ActionEvent(s.envelope(event_id=phase, sequence=index + 1), result))
            self.reopen()
            got = self.store.register_recovery(self.recovery('instance-' + str(index), index + 2))
            self.assertEqual(got.snapshot.run.state, m.RunState.RECOVERING)
            self.assertEqual(got.snapshot.holding.state, c.Holding.HELD)
            self.assertNotEqual(self.store.commit_intent(intent).commit.status, m.CommitStatus.COMMITTED)

    def test_T04_recovery_keeps_arrival_budget_and_layout(self):
        self.store.commit(s.release_transaction())
        before = self.store.read('sim-run-1').snapshot
        self.reopen()
        req = self.recovery()
        got = self.store.register_recovery(req)
        self.assertEqual(got.snapshot.deliveries, before.deliveries)
        self.assertEqual(got.snapshot.layouts, before.layouts)
        self.assertEqual(got.snapshot.budgets, before.budgets)
        self.assertEqual(self.store.register_recovery(req), got)
        self.assertTrue(got.snapshot.run.ever_restarted)

    def test_T02_expired_recovery_only_ends_without_requiring_movement(self):
        self.clock.advance(600)
        self.reopen()
        req = replace(self.recovery(), old_stop=stop_report(710, False))
        got = self.store.register_recovery(req)
        self.assertEqual(got.remaining_s, 0)
        self.assertEqual(got.snapshot.run.state, m.RunState.ENDED)
        self.assertNotEqual(got.status, c.Outcome.FAILED)
        self.assertTrue(self.store.actions_blocked)

    def test_T03_version_or_snapshot_hash_corruption_rejected(self):
        self.store.close()
        db = sqlite3.connect(str(self.root / 'runs/sim-run-1/state.sqlite3'))
        db.execute("UPDATE snapshot SET payload = replace(payload, '1.0.0', '0.0.0')")
        db.commit()
        db.close()
        self.store = self.open_store()
        self.assertNotEqual(self.store.register_recovery(self.recovery()).status, c.Outcome.SUCCESS)

    def test_T05_atomic_timer_failure_keeps_old_complete_text_and_blocks_actions(self):
        old = (self.root / 'runs/sim-run-1/timer.txt').read_bytes()
        self.clock.advance(170)
        with patch('treasure_auto.persistence.filesystem.os.replace', side_effect=OSError('disk full')):
            got = self.store.refresh_timer(self.timer())
        self.assertNotEqual(got.status, m.CommitStatus.COMMITTED)
        self.assertEqual((self.root / 'runs/sim-run-1/timer.txt').read_bytes(), old)
        self.assertTrue(self.store.actions_blocked)
