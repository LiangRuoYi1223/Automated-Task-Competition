import os
import threading
from dataclasses import replace
from unittest.mock import patch
from test_p10b_support import StoreCase, c, m


class CleanupTests(StoreCase):
    def test_T06_deadline_closes_then_cleans_only_named_run(self):
        other = self.root / 'runs/other-run'
        other.mkdir()
        (other / 'keep.txt').write_text('keep', encoding='utf-8')
        fixed = self.root / 'map.yaml'
        fixed.write_text('keep', encoding='utf-8')
        self.clock.advance(590)
        marker = self.seal()
        self.assertEqual(marker.writers_closed, c.Fact.CONFIRMED)
        result = self.store.cleanup(self.cleanup_request(marker))
        self.assertEqual(result.status, c.Outcome.SUCCESS)
        self.assertEqual(result.marker.cleanup_state, m.CleanupState.COMPLETE)
        self.assertFalse((self.root / 'runs/sim-run-1').exists())
        self.assertTrue((other / 'keep.txt').exists())
        self.assertTrue(fixed.exists())
        self.assertNotEqual(self.store.initialize_run(self.initial).status, m.CommitStatus.COMMITTED)
        self.reopen()
        self.assertNotEqual(self.store.initialize_run(self.initial).status, m.CommitStatus.COMMITTED)
        recovered = self.store.register_recovery(self.recovery())
        self.assertEqual(recovered.remaining_s, 0)
        self.assertIsNone(recovered.snapshot)
        self.assertFalse((self.root / 'runs/sim-run-1').exists())

    def test_T06_early_emergency_and_unknown_stop_are_retained(self):
        marker = self.seal(c.ErrorCode.CANCELLED)
        got = self.store.cleanup(self.cleanup_request(marker))
        self.assertNotEqual(got.status, c.Outcome.SUCCESS)
        self.assertTrue((self.root / 'runs/sim-run-1/state.sqlite3').exists())

    def test_T06_deadline_with_unknown_stop_retained(self):
        self.clock.advance(590)
        marker = self.seal(confirmed=False)
        got = self.store.cleanup(self.cleanup_request(marker))
        self.assertNotEqual(got.status, c.Outcome.SUCCESS)

    def test_T06_restarted_run_retained_but_manual_closed_run_can_be_removed(self):
        self.reopen()
        self.store.register_recovery(self.recovery())
        self.clock.advance(590)
        marker = self.seal()
        self.assertTrue(marker.ever_restarted)
        self.assertNotEqual(self.store.cleanup(self.cleanup_request(marker)).status, c.Outcome.SUCCESS)
        self.assertEqual(self.store.cleanup(self.cleanup_request(marker, False)).status, c.Outcome.SUCCESS)

    def test_T08_active_or_forged_marker_and_wrong_root_rejected(self):
        fake = m.EndMarker('sim-run-1', self.initial.run.deadline, self.clock.now(), False,
                           c.Fact.CONFIRMED, c.Fact.CONFIRMED, c.Fact.CONFIRMED, m.CleanupState.PENDING)
        self.assertNotEqual(self.store.cleanup(self.cleanup_request(fake, False)).status, c.Outcome.SUCCESS)
        self.clock.advance(590)
        marker = self.seal()
        wrong = replace(self.cleanup_request(marker, False), runtime_root=str(self.root.parent))
        self.assertNotEqual(self.store.cleanup(wrong).status, c.Outcome.SUCCESS)
        traversal = replace(self.cleanup_request(marker, False), run_id='../map')
        self.assertNotEqual(self.store.cleanup(traversal).status, c.Outcome.SUCCESS)

    def test_T08_hardlink_rejected_and_outside_file_preserved(self):
        outside = self.root / 'fixed.txt'
        outside.write_text('fixed', encoding='utf-8')
        link = self.root / 'runs/sim-run-1/linked.txt'
        link.write_text('link stand-in; metadata injected below', encoding='utf-8')
        self.clock.advance(590)
        marker = self.seal()
        original = os.lstat
        def metadata(path, *args, **kwargs):
            info = original(path, *args, **kwargs)
            if str(path) == str(link):
                from types import SimpleNamespace
                return SimpleNamespace(st_mode=info.st_mode, st_nlink=2,
                                       st_file_attributes=getattr(info, 'st_file_attributes', 0))
            return info
        # Native hardlink creation is denied on this host. Exercise the exact
        # metadata guard without claiming a native-filesystem integration test.
        with patch('treasure_auto.persistence.filesystem.os.lstat', side_effect=metadata):
            got = self.store.cleanup(self.cleanup_request(marker, False))
        self.assertNotEqual(got.status, c.Outcome.SUCCESS)
        self.assertEqual(outside.read_text(encoding='utf-8'), 'fixed')

    def test_T08_reparse_point_refused_before_deletion(self):
        self.clock.advance(590)
        marker = self.seal()
        from treasure_auto.persistence import filesystem as fs
        original = fs.is_link
        target = self.root / 'runs/sim-run-1/timer.txt'
        with patch.object(fs, 'is_link', side_effect=lambda p: str(p) == str(target) or original(p)):
            got = self.store.cleanup(self.cleanup_request(marker, False))
        self.assertNotEqual(got.status, c.Outcome.SUCCESS)
        self.assertTrue(target.exists())

    def test_T07_interrupted_cleanup_resumes_from_external_marker(self):
        self.clock.advance(590)
        marker = self.seal()
        with patch('treasure_auto.persistence.lifecycle.remove_run', side_effect=OSError('simulated interrupted deletion')):
            got = self.store.cleanup(self.cleanup_request(marker))
        self.assertNotEqual(got.status, c.Outcome.SUCCESS)
        self.assertEqual(got.marker.cleanup_state, m.CleanupState.PENDING)
        self.reopen()
        self.assertNotEqual(self.store.initialize_run(self.initial).status, m.CommitStatus.COMMITTED)
        got = self.store.cleanup(self.cleanup_request(got.marker))
        self.assertEqual(got.status, c.Outcome.SUCCESS)

    def test_T07_recovery_before_cleanup_permanently_prevents_auto_deletion(self):
        self.reopen()
        recovered = self.store.register_recovery(self.recovery())
        self.assertTrue(recovered.snapshot.run.ever_restarted)
        self.clock.advance(590)
        marker = self.seal()
        jobs = [self.store.submit('register_recovery', self.recovery('sim-instance-3', 3)),
                self.store.submit('cleanup', self.cleanup_request(marker))]
        for job in jobs:
            job.result(2)
        self.assertTrue((self.root / 'runs/sim-run-1').exists())

    def test_T06_writes_after_seal_never_recreate_run_directory(self):
        self.clock.advance(590)
        marker = self.seal()
        self.store.cleanup(self.cleanup_request(marker))
        from test_p10b_support import s
        receipt = self.store.commit(s.release_transaction())
        self.assertNotEqual(receipt.status, m.CommitStatus.COMMITTED)
        self.assertFalse((self.root / 'runs/sim-run-1').exists())
