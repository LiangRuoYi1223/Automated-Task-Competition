from dataclasses import replace
from test_p05_support import P05Case, Harness, frame, c, m, s


class RecoveryTests(P05Case):
    def require_recovery(self,h):
        self.assertTrue(hasattr(h.service,'restore'),'P05 bounded same-run recovery missing')

    def test_T06_frozen_context_can_restore_but_does_not_restore_permission(self):
        h=Harness(); self.require_recovery(h)
        original=h.send((frame(0),frame(1))).mapping
        saved=h.service.export_progress()
        h.clock.advance(20)
        recovered=h.new_service()
        self.assertEqual(recovered.restore(saved),original)
        self.assertFalse(recovered.dispatch_allowed)
        self.assertEqual(len(h.calls),1)

    def test_T06_partial_confirmation_is_rebuilt_and_replay_does_not_count(self):
        h=Harness(); self.require_recovery(h)
        h.send((frame(0),))
        saved=h.service.export_progress()
        restored=h.new_service(); self.assertIsNone(restored.restore(saved))
        h.service=restored
        self.assertIsNone(h.send((frame(0),frame(1))).mapping)
        self.assertEqual(h.send((frame(2),)).result.status,c.Outcome.SUCCESS)

    def test_T06_progress_roundtrip_retains_budget_and_original_deadline(self):
        h=Harness(); self.require_recovery(h)
        h.send()
        encoded=h.service.export_progress().to_json()
        saved=self.p.ReadProgress.from_json(encoded)
        restored=h.new_service(); restored.restore(saved); h.service=restored
        self.assertEqual(restored.views_used,1)
        self.assertIsNotNone(h.send(deadline=300).additional_view)
        self.assertEqual(h.send(deadline=300).result.status,c.Outcome.FAILED)

    def test_T06_expired_read_progress_cannot_reopen_budget(self):
        h=Harness(); self.require_recovery(h)
        h.send(deadline=111)
        saved=h.service.export_progress(); h.clock.advance(2)
        recovered=h.new_service(); recovered.restore(saved); h.service=recovered
        self.assertEqual(h.send(deadline=300).result.status,c.Outcome.TIMED_OUT)

    def test_T06_unknown_commit_progress_is_never_resubmitted(self):
        h=Harness(); self.require_recovery(h); h.mode='unknown_before'
        h.send((frame(0),frame(1)))
        saved=self.p.ReadProgress.from_json(h.service.export_progress().to_json())
        recovered=h.new_service(); recovered.restore(saved); h.service=recovered
        self.assertIsNone(h.send((frame(2),frame(3))).mapping)
        self.assertEqual(len(h.calls),1)

    def test_T06_missing_progress_without_frozen_context_is_blocked(self):
        h=Harness(); self.require_recovery(h)
        with self.assertRaises(c.ContractError): h.new_service().restore()

    def test_T05_wrong_run_template_deadline_and_epoch_are_not_restored(self):
        h=Harness(); self.require_recovery(h); h.send((frame(0),frame(1)))
        saved=h.service.export_progress()
        for fields in ({'run_id':'previous'},{'fingerprint':'untrusted'},{'run_deadline':s.stamp(701)},
                       {'views_used':-1}):
            with self.subTest(fields=fields),self.assertRaises(c.ContractError):
                h.new_service().restore(replace(saved,**fields))
        old=h.store._snapshot
        h.store._snapshot=replace(old,run=replace(old.run,envelope=replace(old.run.envelope,control_epoch=0)))
        with self.assertRaises(c.ContractError): h.new_service().restore(saved)

    def test_T05_new_run_cannot_read_previous_frozen_mapping(self):
        h=Harness(); self.require_recovery(h); h.send((frame(0),frame(1)))
        new=h.new_service()
        new.run=replace(h.run,envelope=replace(h.run.envelope,run_id='next'))
        new.context=replace(new.context,run_id='next')
        with self.assertRaises(c.ContractError): new.restore()

