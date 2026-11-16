import unittest
from dataclasses import replace
from test_p10a_support import *


def recovery_setup():
    original, clock = started()
    saved = original.run
    rt, _ = make_runtime(instance_id='sim-instance-2',control_epoch=2)
    rt.clock=clock
    clock.advance(190)
    rt.begin_recovery(saved)
    snap=initial_snapshot(replace(rt.run,ever_restarted=True))
    result=m.RecoveryResult(c.Outcome.SUCCESS,snap,410,
                            m.CommitReceipt('sim-recovery',m.CommitStatus.COMMITTED,0,0,None),None)
    return rt,clock,result


class RecoveryTests(unittest.TestCase):
    def test_T06_registration_does_not_replace_old_action_stop(self):
        rt,clock,result=recovery_setup()
        self.assertIsNone(rt.recover(result).permit)
        self.assertEqual(rt.remaining_s,410)
        req=request(rt,clock)
        bad=replace(req,old_stop=replace(req.old_stop,old_queue_empty=c.Fact.UNKNOWN))
        self.assertIsNone(rt.resume(bad).permit)
        clock.advance(.01)
        self.assertIsNotNone(rt.resume(request(rt,clock,seq=20)).permit)
        self.assertTrue(rt.run.ever_restarted)
        self.assertEqual(rt.run.deadline.seconds,700)

    def test_T06_wrong_boot_deadline_instance_or_failed_registration_keeps_stop(self):
        for bad in ('boot','deadline','instance','registration','restart'):
            rt,clock,result=recovery_setup()
            run=result.snapshot.run
            if bad=='boot': run=replace(run,started_at=s.stamp(100,'other'),deadline=s.stamp(700,'other'))
            if bad=='deadline': run=replace(run,started_at=s.stamp(101),deadline=s.stamp(701))
            if bad=='instance': run=replace(run,envelope=replace(run.envelope,instance_id='old'))
            if bad=='restart': run=replace(run,ever_restarted=False)
            result=replace(result,snapshot=replace(result.snapshot,run=run))
            if bad=='registration': result=replace(result,registration=None)
            with self.subTest(bad=bad):
                self.assertIsNotNone(rt.recover(result).reason)
                self.assertIsNone(rt.resume(request(rt,clock)).permit)

    def test_T06_late_recovery_ack_cannot_clear_deadline(self):
        rt,clock,result=recovery_setup()
        clock.advance(410)
        rt.recover(result)
        self.assertIsNone(rt.resume(request(rt,clock)).permit)
        self.assertEqual(rt.run.deadline.seconds,700)

    def test_T07_referee_limit_first_lift_and_duplicate_events(self):
        rt,clock=started()
        unauthorized=signal(rt,clock,m.SignalKind.MANUAL_DEBUG,'signal-2',c.Fact.UNKNOWN)
        self.assertIsNotNone(rt.signal(unauthorized).reason)
        self.assertEqual(rt.manual_debug_count,0)
        for n in range(3,6):
            sig=signal(rt,clock,m.SignalKind.MANUAL_DEBUG,'signal-{}'.format(n))
            rt.signal(sig); rt.signal(sig)
        self.assertEqual(rt.manual_debug_count,3)
        self.assertEqual(rt.signal(signal(rt,clock,m.SignalKind.MANUAL_DEBUG,'signal-6')).reason.code,c.ErrorCode.LIMIT_EXCEEDED)
        self.assertEqual(rt.run.deadline.seconds,700)
        rt2,clock2=started()
        req=request(rt2,clock2)
        rt2.confirm_first_lift(req.readiness.evidence[0])
        self.assertIsNotNone(rt2.signal(signal(rt2,clock2,m.SignalKind.MANUAL_DEBUG,'signal-2')).reason)
        self.assertEqual(rt2.manual_debug_count,0)

    def test_T07_debug_resume_needs_relocalization_reread_and_new_control_epoch(self):
        rt,clock=started(); req=request(rt,clock)
        rt.signal(signal(rt,clock,m.SignalKind.MANUAL_DEBUG,'signal-2'))
        clock.advance(.5)
        self.assertIsNone(rt.resume(req).permit)
        self.assertEqual(rt.remaining_s,599.5)

    def test_restart_does_not_reset_debug_or_lift_history_to_permissive_defaults(self):
        rt,clock,result=recovery_setup(); rt.recover(result)
        decision=rt.signal(signal(rt,clock,m.SignalKind.MANUAL_DEBUG,'signal-9'))
        self.assertIsNotNone(decision.reason)


if __name__=='__main__': unittest.main()
