import json
import os
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from test_p10b_support import StoreCase, c, m, r, s, stop_report


class BoundaryTests(StoreCase):
    def test_T05_closing_writer_cannot_reopen_gate_when_last_commit_finishes(self):
        entered, release = threading.Event(), threading.Event()
        original = self.store._write_snapshot
        def pause(snapshot):
            entered.set()
            release.wait(2)
            original(snapshot)
        with patch.object(self.store, '_write_snapshot', side_effect=pause):
            commit = self.store.submit('commit', s.release_transaction())
            self.assertTrue(entered.wait(1))
            done = []
            thread = threading.Thread(target=lambda: done.append(self.store.close()))
            thread.start()
            # close's admission guard is independent of the paused SQLite writer.
            for _ in range(100):
                if not self.store._accepting:
                    break
                threading.Event().wait(0.001)
            self.assertFalse(self.store._accepting)
            release.set()
            thread.join(3)
            self.assertEqual(done, [True])
            self.assertEqual(commit.result().status, m.CommitStatus.COMMITTED)
            self.assertTrue(self.store.actions_blocked)

    def test_T05_rejected_keyframe_keeps_metadata_without_raw_content(self):
        reference = s.evidence(evidence_id='oversized-frame', uri='pending:oversized')
        with self.assertRaises(c.ContractError):
            self.store.save_keyframe(reference, b'x' * 4097)
        history = self.store.history('sim-run-1')
        rows = [x for x in history if x['kind'] == 'rejected:save_keyframe']
        self.assertEqual(len(rows), 1)
        self.assertEqual(c.loads(rows[0]['payload']), reference)
        self.assertIsNotNone(c.loads(rows[0]['result']))

    def test_T06_missing_stop_keyframe_cannot_authorize_automatic_cleanup(self):
        self.clock.advance(590)
        stop = stop_report(700)
        missing = replace(stop.evidence[0], uri=(self.root / 'runs/sim-run-1/keyframes/missing.bin').as_uri(),
                          sha256='0' * 64)
        stop = replace(stop, evidence=(missing,))
        e = replace(self.initial.run.envelope, event_id='end', emitted_at=s.stamp(700), expires_at=s.stamp(720))
        run = replace(self.initial.run, envelope=e, state=m.RunState.ENDED)
        marker = self.store.seal(m.SealRequest(e, run, stop, c.Reason(c.ErrorCode.DEADLINE, 'run', 'expired')))
        self.assertNotEqual(marker.stop_confirmed, c.Fact.CONFIRMED)
        self.assertNotEqual(self.store.cleanup(self.cleanup_request(marker)).status, c.Outcome.SUCCESS)
        self.assertTrue((self.root / 'runs/sim-run-1/state.sqlite3').exists())

    def test_T01_advancing_clock_initialization_uses_one_timer_sample(self):
        class AdvancingClock:
            def __init__(self):
                self.value = 110.0
            def now(self):
                self.value += 0.001
                return s.stamp(self.value)
        store = self.p.SQLitePersistence(replace(self.store.config, runtime_root=str(self.root.parent / 'advancing')),
                                         AdvancingClock(), remaining=r.remaining_seconds)
        self.stores.append(store)
        result = store.initialize_run(self.initial)
        self.assertEqual(result.status, m.CommitStatus.COMMITTED)
        timer = c.loads((store.root / 'runs/sim-run-1/timer.txt').read_text(encoding='utf-8'))
        self.assertEqual(timer.remaining_s, 700 - timer.refreshed_at.seconds)

    def test_T07_directory_empty_but_pending_marker_never_reopens_match(self):
        self.clock.advance(590)
        marker = self.seal()
        original = self.store._write_marker
        def fail_completion(value):
            if value.cleanup_state is m.CleanupState.COMPLETE:
                raise OSError('power-loss stand-in after deletion before marker completion')
            return original(value)
        with patch.object(self.store, '_write_marker', side_effect=fail_completion):
            result = self.store.cleanup(self.cleanup_request(marker))
        self.assertEqual(result.marker.cleanup_state, m.CleanupState.PENDING)
        self.assertFalse((self.root / 'runs/sim-run-1').exists())
        self.reopen()
        self.assertEqual(self.store.register_recovery(self.recovery()).remaining_s, 0)
        self.assertNotEqual(self.store.initialize_run(self.initial).status, m.CommitStatus.COMMITTED)
        result = self.store.cleanup(self.cleanup_request(result.marker))
        self.assertEqual(result.status, c.Outcome.SUCCESS)

    def test_T03_recovery_rejects_missing_previously_committed_frame(self):
        from urllib.parse import urlparse
        from urllib.request import url2pathname
        ref = self.store.save_keyframe(s.evidence(uri='pending:test'), b'keyframe')
        self.store.append_event(m.ActionEvent(s.envelope(event_id='with-frame'),
                                replace(s.timeout_result(), evidence=(ref,))))
        Path(url2pathname(urlparse(ref.uri).path)).unlink()
        self.reopen()
        self.assertNotEqual(self.store.register_recovery(self.recovery()).status, c.Outcome.SUCCESS)

    def test_T03_storage_format_version_is_checked(self):
        import sqlite3
        self.store.close()
        db = sqlite3.connect(str(self.root / 'runs/sim-run-1/state.sqlite3'))
        db.execute('PRAGMA user_version=999')
        db.close()
        self.store = self.open_store()
        self.assertNotEqual(self.store.register_recovery(self.recovery()).status, c.Outcome.SUCCESS)

    def test_T02_real_process_exit_after_intent_commit_keeps_intent(self):
        self.store.close()
        source = str(Path(__file__).resolve().parents[1] / 'src')
        tests = str(Path(__file__).resolve().parent)
        fresh_root = self.root.parent / 'crash-intent'
        code = """
import os,sys
sys.path[:0]=[sys.argv[2],sys.argv[3]]
from test_p10b_support import c,m,r,s,initial_request
from contract_fixtures.fakes import FakeClock
from treasure_auto.persistence import SQLitePersistence,PersistenceConfig
store=SQLitePersistence(PersistenceConfig(sys.argv[1],8,2,1,4096),FakeClock(s.stamp(110)),remaining=r.remaining_seconds)
assert store.initialize_run(initial_request()).status is m.CommitStatus.COMMITTED
assert store.commit_intent(s.arm_intent()).commit.status is m.CommitStatus.COMMITTED
os._exit(28)
"""
        result = subprocess.run([sys.executable, '-B', '-c', code, str(fresh_root), source, tests],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 28, result.stderr.decode(errors='replace'))
        self.root = fresh_root
        self.store = self.open_store()
        history = self.store.history('sim-run-1')
        self.assertEqual(sum(row['kind'] == 'intent' for row in history), 1)
        recovered = self.store.register_recovery(self.recovery())
        self.assertEqual(recovered.snapshot.run.state, m.RunState.RECOVERING)
        self.assertNotEqual(self.store.commit_intent(s.arm_intent()).commit.status, m.CommitStatus.COMMITTED)

    def test_T03_retried_registration_recomputes_time_without_new_revision(self):
        self.reopen()
        request = self.recovery()
        first = self.store.register_recovery(request)
        self.clock.advance(10)
        again = self.store.register_recovery(request)
        self.assertEqual(again.registration, first.registration)
        self.assertEqual(again.remaining_s, 580)

    def test_T03_old_recovery_id_cannot_retake_newer_controller(self):
        self.reopen()
        first_request = self.recovery()
        self.store.register_recovery(first_request)
        second = self.store.register_recovery(self.recovery('third-instance', 3))
        self.assertEqual(second.status, c.Outcome.SUCCESS)
        stale = self.store.register_recovery(first_request)
        self.assertNotEqual(stale.status, c.Outcome.SUCCESS)

    def test_T07_terminal_snapshot_without_marker_never_recovers_running(self):
        self.clock.advance(590)
        with patch.object(self.store, '_write_marker', side_effect=OSError('marker unavailable')):
            with self.assertRaises(c.ContractError):
                self.seal()
        self.reopen()
        result = self.store.register_recovery(self.recovery())
        self.assertNotEqual(result.status, c.Outcome.SUCCESS)
        self.assertTrue(self.store.actions_blocked)
        self.assertTrue((self.root / 'runs/sim-run-1').exists())

    def test_T06_sealed_timer_contains_terminal_state(self):
        self.clock.advance(590)
        self.seal()
        timer = c.loads((self.root / 'runs/sim-run-1/timer.txt').read_text(encoding='utf-8'))
        self.assertEqual(timer.state, m.RunState.ENDED)
        self.assertEqual(timer.remaining_s, 0)

    def test_T05_slow_commit_returns_unknown_then_preserves_actual_commit(self):
        from treasure_auto.persistence.config import PersistenceConfig
        self.store.config = replace(self.store.config, operation_timeout_s=0.05)
        entered, release = threading.Event(), threading.Event()
        original = self.store._write_snapshot
        def pause(snapshot):
            entered.set()
            release.wait(2)
            original(snapshot)
        try:
            with patch.object(self.store, '_write_snapshot', side_effect=pause):
                receipt = self.store.commit(s.release_transaction())
                self.assertTrue(entered.is_set())
                self.assertEqual(receipt.status, m.CommitStatus.UNKNOWN)
                self.assertTrue(self.store.actions_blocked)
                release.set()
                self.store.config = replace(self.store.config, operation_timeout_s=2)
                self.assertEqual(len(self.store.read('sim-run-1').snapshot.deliveries), 1)
                self.assertNotEqual(self.store.commit_intent(s.arm_intent()).commit.status, m.CommitStatus.COMMITTED)
        finally:
            release.set()
            self.store.config = replace(self.store.config, operation_timeout_s=2)

    def test_T05_cancelled_future_does_not_kill_writer_or_leave_queued_jobs(self):
        entered, release = threading.Event(), threading.Event()
        original = self.store._append_event
        def pause(event):
            entered.set()
            release.wait(2)
            return original(event)
        event = m.ActionEvent(s.envelope(event_id='pause'), s.timeout_result())
        with patch.object(self.store, '_append_event', side_effect=pause):
            first = self.store.submit('append_event', event)
            self.assertTrue(entered.wait(1))
            pending = self.store.submit('read', 'sim-run-1')
            pending.cancel()
            release.set()
            first.result(2)
            self.assertEqual(self.store.read('sim-run-1').status, c.Outcome.SUCCESS)

    def test_T03_missing_time_policy_fails_before_creating_data(self):
        from treasure_auto.persistence import SQLitePersistence
        with self.assertRaises(c.ContractError):
            SQLitePersistence(replace(self.store.config, runtime_root=str(self.root.parent / 'new')), self.clock, remaining=None)
        self.assertFalse((self.root.parent / 'new').exists())

    def test_T08_linked_sqlite_sidecar_never_opened(self):
        self.store.close()
        from treasure_auto.persistence import filesystem as fs
        sidecar = self.root / 'runs/sim-run-1/state.sqlite3-wal'
        sidecar.write_bytes(b'not an actual symlink; link flag injected')
        original = fs.is_link
        with patch.object(fs, 'is_link', side_effect=lambda path: str(path) == str(sidecar) or original(path)):
            self.store = self.open_store()
            self.assertNotEqual(self.store.read('sim-run-1').status, c.Outcome.SUCCESS)

    def test_T01_generic_c0_result_journal_and_rejected_transaction_are_auditable(self):
        observation = s.all_samples()['localization']
        self.assertEqual(self.store.append_record(observation).status, m.CommitStatus.COMMITTED)
        bad = replace(s.release_transaction(), holding_update=None)
        self.store.commit(bad)
        history = self.store.history('sim-run-1')
        self.assertTrue(any(c.loads(x['payload']) == observation for x in history))
        self.assertTrue(any(x['kind'] == 'rejected:commit' and x['result'] is not None for x in history))

    def test_T06_old_controller_stop_proof_does_not_enable_cleanup(self):
        self.clock.advance(590)
        old_stop = replace(stop_report(700), envelope=replace(stop_report(700).envelope,
                           instance_id='obsolete', control_epoch=0))
        run = self.initial.run
        e = replace(run.envelope, event_id='end', emitted_at=s.stamp(700), expires_at=s.stamp(720))
        marker = self.store.seal(m.SealRequest(e, replace(run, envelope=e, state=m.RunState.ENDED), old_stop,
                                 c.Reason(c.ErrorCode.DEADLINE, 'run', 'time expired')))
        self.assertNotEqual(marker.stop_confirmed, c.Fact.CONFIRMED)
        self.assertNotEqual(self.store.cleanup(self.cleanup_request(marker)).status, c.Outcome.SUCCESS)

    def test_T02_real_process_exit_after_uncommitted_snapshot_rolls_back(self):
        self.store.close()
        # A real child process dies without finally/close; only its temporary
        # SQLite transaction is affected. No ROS/robot process is involved.
        source = str(Path(__file__).resolve().parents[1] / 'src')
        tests = str(Path(__file__).resolve().parent)
        code = """
import os,sys
sys.path[:0]=[sys.argv[2],sys.argv[3]]
from test_p10b_support import c,m,r,s,initial_request
from contract_fixtures.fakes import FakeClock
from treasure_auto.persistence import SQLitePersistence,PersistenceConfig
store=SQLitePersistence(PersistenceConfig(sys.argv[1],8,2,1,4096),FakeClock(s.stamp(110)),remaining=r.remaining_seconds)
from dataclasses import replace
env=s.envelope(c.Owner.P10A,instance_id='child',control_epoch=2,event_id='child-recovery')
from test_p10b_support import stop_report
store.register_recovery(m.RecoveryRequest(env,s.stamp(100),s.stamp(700),stop_report()))
original=store._write_snapshot
def crash(value):
    original(value)
    os._exit(27)
store._write_snapshot=crash
req=m.RecoveryRequest(replace(env,instance_id='child-next',control_epoch=3,event_id='child-next'),s.stamp(100),s.stamp(700),stop_report())
store.register_recovery(req)
"""
        result = subprocess.run([sys.executable, '-B', '-c', code, str(self.root), source, tests],
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 27, result.stderr.decode(errors='replace'))
        self.store = self.open_store()
        snap = self.store.read('sim-run-1').snapshot
        self.assertEqual(snap.run.envelope.instance_id, 'child')
        self.assertTrue(snap.run.ever_restarted)
        self.assertTrue(self.store.actions_blocked)
