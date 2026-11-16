"""Failure regressions found while checking state transitions and re-entry."""
import ast
import subprocess
import threading
import unittest
from dataclasses import replace
from test_p10a_support import *
from test_p10a_recovery import recovery_setup


class BoundaryTests(unittest.TestCase):
    def test_prestart_emergency_latches_even_without_public_run_decision(self):
        rt,clock=make_runtime(); rt.arm()
        with self.assertRaises(c.ContractError):
            rt.signal(signal(rt,clock,m.SignalKind.EMERGENCY_STOP,'signal-1'))
        with self.assertRaises(c.ContractError): rt.signal(signal(rt,clock,event='signal-2'))
        self.assertIsNone(rt.run)

    def test_unknown_initialization_cannot_blindly_retry(self):
        rt,clock=make_runtime(); rt.arm(); rt.signal(signal(rt,clock))
        snap=initial_snapshot(rt.run); store=Store(m.CommitStatus.UNKNOWN)
        rt.initialize(store,snap)
        rt.initialize(store,snap)
        self.assertEqual(store.calls,1)

    def test_late_initialization_cannot_release_deadline(self):
        rt,clock=make_runtime(); rt.arm(); rt.signal(signal(rt,clock))
        store=Store(block=True); errors=[]
        def work():
            try: rt.initialize(store,initial_snapshot(rt.run))
            except Exception as error: errors.append(error)
        worker=threading.Thread(target=work); worker.start()
        try:
            self.assertTrue(store.entered.wait(1)); clock.advance(600)
            rt.supervise(clock.now(),None)
        finally:
            store.release.set(); worker.join(1)
        self.assertFalse(errors)
        self.assertEqual(rt.state,m.RunState.STOPPING)
        self.assertIsNone(rt.resume(request(rt,clock)).permit)

    def test_recovery_cannot_revive_an_ended_or_terminally_stopped_original(self):
        for state in (m.RunState.ENDED,m.RunState.STOPPING):
            old,clock=started(); old_run=replace(old.run,state=state)
            rt,_=make_runtime(instance_id='next',control_epoch=2); rt.clock=clock
            rt.begin_recovery(old_run)
            snap=initial_snapshot(rt.run)
            result=m.RecoveryResult(c.Outcome.SUCCESS,snap,600,m.CommitReceipt('r',m.CommitStatus.COMMITTED,0,0,None),None)
            rt.recover(result)
            self.assertIsNone(rt.resume(request(rt,clock)).permit)

    def test_failed_registration_after_success_still_blocks_recovery(self):
        rt,clock,result=recovery_setup(); rt.recover(result)
        rt.recover(replace(result,registration=None))
        self.assertIsNone(rt.resume(request(rt,clock)).permit)

    def test_end_requires_stop_seal_and_closed_writers_and_stays_ended(self):
        rt,clock=started(); req=request(rt,clock)
        rt.signal(signal(rt,clock,m.SignalKind.FINISH,'signal-2'))
        marker=m.EndMarker(rt.run.envelope.run_id,rt.run.deadline,clock.now(),False,
                           c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,m.CleanupState.RETAINED)
        unknown=replace(req.old_stop,physically_stopped=c.Fact.UNKNOWN)
        self.assertIsNotNone(rt.accept_end(marker,unknown).reason)
        self.assertNotEqual(rt.state,m.RunState.ENDED)
        stop=rt.stop_requests[-1]
        fresh=replace(req.old_stop,envelope=replace(stop.envelope,producer=c.Owner.P01),
                      evidence=(replace(req.old_stop.evidence[0],action_id=stop.envelope.action_id,captured_at=clock.now()),))
        rt.accept_end(marker,fresh)
        self.assertEqual(rt.state,m.RunState.ENDED)
        rt.signal(signal(rt,clock,event='signal-3'))
        self.assertEqual(rt.state,m.RunState.ENDED)

    def test_stale_health_and_clock_rollback_fail_closed(self):
        rt,clock=started(); req=request(rt,clock); permit=rt.request_permission(req).permit
        clock._now=s.stamp(99)
        self.assertFalse(rt.authorize_output(permit,req.envelope,req.source,real_output=False))
        clock._now=s.stamp(100)
        self.assertIsNone(rt.resume(req).permit)

    def test_import_is_ros_free_and_all_runtime_python_has_38_syntax(self):
        src=PACKAGE/'src'
        code="import sys; sys.path.insert(0,sys.argv[1]); import treasure_auto.runtime; assert 'rospy' not in sys.modules; assert 'serial' not in sys.modules"
        result=subprocess.run([sys.executable,'-B','-c',code,str(src)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        for path in (src/'treasure_auto/runtime').rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,8))


if __name__=='__main__': unittest.main()
