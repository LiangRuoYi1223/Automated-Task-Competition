from dataclasses import replace
from test_p05_support import P05Case, Harness, frame, c, m


class CommitTests(P05Case):
    def test_R04_failure_unknown_exception_and_wrong_receipt_never_publish(self):
        for mode in ('reject','unknown_before','raise','wrong_receipt'):
            with self.subTest(mode=mode):
                h=Harness(); h.mode=mode
                result=h.send((frame(0),frame(1)))
                self.assertIsNone(result.mapping)
                self.assertFalse(h.service.dispatch_allowed)
                self.assertNotEqual(result.result.status,c.Outcome.SUCCESS)

    def test_R04_unknown_after_commit_reconciles_without_resubmission(self):
        h=Harness(); h.mode='unknown_after'
        self.assertIsNone(h.send((frame(0),frame(1))).mapping)
        result=h.send()
        self.assertEqual(result.result.status,c.Outcome.SUCCESS,result.result.reason)
        self.assertEqual(len(h.calls),1)

    def test_R04_unknown_before_commit_does_not_retry_blindly(self):
        h=Harness(); h.mode='unknown_before'
        h.send((frame(0),frame(1)))
        self.assertIsNone(h.send((frame(2),frame(3))).mapping)
        self.assertEqual(len(h.calls),1)

    def test_R04_readback_missing_or_mismatched_never_publishes(self):
        h=Harness(); real=h.store.read
        h.store.read=lambda run_id: m.ReadResult(c.Outcome.UNKNOWN,None,c.Reason(c.ErrorCode.STORAGE_FAILURE,'read','offline fault'))
        # The coordinator also uses read; failure here remains a storage unknown.
        self.assertIsNone(h.send((frame(0),frame(1))).mapping)
        self.assertFalse(h.service.dispatch_allowed)
        h.store.read=real

    def test_R04_missing_coordinator_does_not_impersonate_P09(self):
        h=Harness(); h.service.submit_mapping=None
        result=h.send((frame(0),frame(1)))
        self.assertIsNone(result.mapping)
        self.assertEqual(result.result.reason.code,c.ErrorCode.UNAVAILABLE)
        self.assertIsNone(h.store.read(h.run.envelope.run_id).snapshot.mapping)

    def test_R04_commit_returning_after_read_deadline_never_allows_dispatch(self):
        h=Harness(); h.mode='late'
        result=h.send((frame(0),frame(1)))
        self.assertEqual(result.result.status,c.Outcome.TIMED_OUT)
        self.assertFalse(h.service.dispatch_allowed)

