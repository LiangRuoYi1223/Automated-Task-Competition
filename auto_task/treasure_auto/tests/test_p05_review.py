"""Regression cases from independent review and final boundary audit."""
import threading
from dataclasses import replace
from test_p05_support import P05Case, Harness, frame, c, m


class ReviewTests(P05Case):
    def test_R05_conflict_during_unknown_commit_survives_later_readback(self):
        h=Harness(); h.mode='unknown_before'
        h.send((frame(0),frame(1)))
        h.send((frame(2,categories=(m.Category.CLOCK,m.Category.DOLL,m.Category.BANANA)),))
        h.mode='normal'
        proposal,audit=h.calls[0]
        self.assertEqual(h.commit(proposal,audit).status,m.CommitStatus.COMMITTED)
        result=h.send()
        self.assertEqual(result.mapping,proposal)
        self.assertFalse(h.service.dispatch_allowed)
        self.assertEqual(result.result.reason.code,c.ErrorCode.CONFLICT)

    def test_R05_no_dispatch_or_intermediate_success_before_batch_conflicts_checked(self):
        h=Harness(); h.mode='unknown_after'; h.send((frame(0),frame(1)))
        entered=threading.Event(); release=threading.Event(); original=h.service.resolve_frame
        def blocked(obs):
            entered.set(); release.wait(2); return original(obs)
        h.service.resolve_frame=blocked
        request=h.request((frame(2,categories=(m.Category.CLOCK,m.Category.DOLL,m.Category.BANANA)),))
        output=[]; start=len(h.service.audit)
        worker=threading.Thread(target=lambda:output.append(h.service.read_mapping(request)))
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertFalse(h.service.dispatch_allowed)
        finally:
            release.set(); worker.join(3)
        self.assertFalse(worker.is_alive())
        statuses=[e['value']['result']['status'] for e in h.service.audit[start:] if e['type']=='result']
        self.assertNotIn('success',statuses)
        self.assertFalse(h.service.dispatch_allowed)

    def test_T06_restore_serializes_with_read_to_prevent_budget_rollback(self):
        h=Harness(); h.send(); progress=h.service.export_progress()
        restored=h.new_service(); entered=threading.Event(); release=threading.Event()
        original=h.store.read
        def blocked(run_id):
            entered.set(); release.wait(2); return original(run_id)
        h.store.read=blocked
        errors=[]
        def work():
            try: restored.restore(progress)
            except Exception as error: errors.append(error)
        worker=threading.Thread(target=work); worker.start()
        try:
            self.assertTrue(entered.wait(2))
            for _ in range(2):
                result=restored.read_mapping(h.request())
                self.assertIsNone(result.additional_view)
        finally:
            release.set(); worker.join(3); h.store.read=original
        self.assertFalse(worker.is_alive()); self.assertEqual(errors,[])
        h.service=restored
        self.assertIsNotNone(h.send().additional_view)
        self.assertEqual(h.send().result.status,c.Outcome.FAILED)

    def test_R06_tightening_budget_below_spent_keeps_valid_terminal_progress(self):
        h=Harness(); h.send()
        self.assertEqual(h.send(views=0).result.status,c.Outcome.FAILED)
        progress=h.service.export_progress()
        self.assertEqual(progress.views_used,1)
        self.assertEqual(progress.view_limit,1)

    def test_R01_live_is_disabled_even_when_parameter_labels_are_verified(self):
        h=Harness(); request=h.request((frame(0),frame(1)))
        # Source checks precede the explicit LIVE adapter gate; neither can grant success.
        h.service.context=replace(h.service.context,origin=c.Origin.LIVE)
        result=h.service.read_mapping(replace(request,envelope=replace(request.envelope,origin=c.Origin.LIVE)))
        self.assertIsNone(result.mapping)
        self.assertFalse(h.service.dispatch_allowed)

