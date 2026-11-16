import unittest
from unittest.mock import patch
from dataclasses import replace
from test_p10a_support import *
from test_p10a_recovery import recovery_setup


class ReviewTests(unittest.TestCase):
    def test_new_unknown_permission_facts_revoke_old_active_permit(self):
        rt,clock=started(); req=request(rt,clock); permit=rt.request_permission(req).permit
        bad=replace(req,readiness=replace(req.readiness,posture_confirmed=c.Fact.UNKNOWN))
        self.assertIsNone(rt.request_permission(bad).permit)
        self.assertFalse(rt.authorize_output(permit,req.envelope,req.source,real_output=False))

    def test_explicit_resume_cannot_reuse_facts_from_before_invalidation(self):
        rt,clock=started(); req=request(rt,clock); rt.request_permission(req)
        unknown=replace(req.readiness,holding=c.HoldingSnapshot(c.Holding.UNKNOWN,None,None,()))
        rt.observe_phase(m.Phase.CONTACT,unknown)
        self.assertIsNone(rt.resume(req).permit)
        clock.advance(.01)
        self.assertIsNotNone(rt.resume(request(rt,clock,seq=20)).permit)

    def test_authoritative_recovery_snapshot_ended_cannot_resume(self):
        rt,clock,result=recovery_setup()
        snap=replace(result.snapshot,run=replace(result.snapshot.run,state=m.RunState.ENDED))
        rt.recover(replace(result,snapshot=snap))
        self.assertIsNone(rt.resume(request(rt,clock)).permit)

    def test_repeated_stop_request_refreshes_expiry_and_retains_first_request(self):
        rt,clock=started(); clock.advance(600)
        first=rt.supervise(clock.now(),None).stop
        clock.advance(3)
        second=rt.supervise(clock.now(),None).stop
        self.assertGreater(second.envelope.expires_at-clock.now(),0)
        self.assertNotEqual(first.envelope.event_id,second.envelope.event_id)
        self.assertIn(first,rt.stop_requests)

    def test_out_of_order_thread_clock_samples_are_not_os_clock_rollback(self):
        rt,clock=started(); req=request(rt,clock); permit=rt.request_permission(req).permit
        pending_sample=clock.now()
        clock.advance(.01)
        self.assertTrue(rt.authorize_output(permit,req.envelope,req.source,real_output=False))
        decision=rt.supervise(pending_sample,req.health)
        self.assertIsNone(decision.stop)
        self.assertTrue(rt.authorize_output(permit,req.envelope,req.source,real_output=False))

    def test_mapping_version_string_without_frozen_mapping_cannot_enable_central(self):
        rt,clock=make_runtime();rt.arm();rt.signal(signal(rt,clock))
        snapshot=replace(initial_snapshot(rt.run),mapping=None)
        rt.initialize(Store(),snapshot)
        self.assertIsNone(rt.request_permission(request(rt,clock,purpose='central_pick')).permit)

    def test_lost_localization_does_not_reuse_old_positive_snapshot(self):
        rt,clock=started();req=request(rt,clock);rt.request_permission(req)
        rt.observe_localization(replace(req.localization,trusted=c.Fact.UNKNOWN))
        self.assertIsNone(rt.resume(req).permit)

    def test_terminal_recovery_result_cannot_be_replaced_by_older_running_result(self):
        rt,clock,result=recovery_setup()
        rt.recover(replace(result,snapshot=replace(result.snapshot,run=replace(result.snapshot.run,state=m.RunState.ENDED))))
        rt.recover(result)
        self.assertIsNone(rt.resume(request(rt,clock)).permit)

    def test_terminal_committed_snapshot_latches_stop(self):
        rt,clock=started();req=request(rt,clock);rt.request_permission(req)
        snap=replace(initial_snapshot(replace(rt.run,state=m.RunState.ENDED)),revision=1)
        rt.accept_snapshot(snap,m.CommitReceipt('end',m.CommitStatus.COMMITTED,1,0,None))
        self.assertIsNone(rt.resume(req).permit)

    def test_end_rejects_stop_report_observed_before_this_stop(self):
        rt,clock=started();req=request(rt,clock);rt.request_permission(req)
        clock.advance(.1);rt.signal(signal(rt,clock,m.SignalKind.FINISH,'signal-2'))
        marker=m.EndMarker(rt.run.envelope.run_id,rt.run.deadline,clock.now(),False,
                           c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,m.CleanupState.RETAINED)
        rt.accept_end(marker,req.old_stop)
        self.assertNotEqual(rt.state,m.RunState.ENDED)

    def test_delayed_old_capture_is_not_fresh_after_unknown(self):
        rt,clock=started();req=request(rt,clock);rt.request_permission(req)
        clock.advance(.1)
        rt.observe_phase(m.Phase.CONTACT,replace(req.readiness,holding=c.HoldingSnapshot(c.Holding.UNKNOWN,None,None,())))
        clock.advance(.1);new=request(rt,clock,seq=20)
        for capture in (s.stamp(99),s.sensor_stamp(19.0)):
            ev=replace(new.readiness.evidence[0],captured_at=capture)
            bad=replace(new,readiness=replace(new.readiness,evidence=(ev,),holding=replace(new.readiness.holding,evidence=(ev,))))
            self.assertIsNone(rt.resume(bad).permit)

    def test_same_reason_new_invalidation_during_resume_changes_authority_ticket(self):
        rt,clock=started();req=request(rt,clock);rt.request_permission(req)
        bad=replace(req.health,capabilities=(replace(req.health.capabilities[0],status=c.Fact.UNKNOWN),))
        rt.supervise(clock.now(),bad)
        clock.advance(.1);fresh=request(rt,clock,seq=20)
        release=rt.authority.release
        def race(ticket,now):
            rt.supervise(clock.now(),bad)
            return release(ticket,now)
        with patch.object(rt.authority,'release',side_effect=race):
            self.assertIsNone(rt.resume(fresh).permit)

    def test_invalidation_barrier_is_published_before_authority_revocation(self):
        rt,clock=started();req=request(rt,clock);rt.request_permission(req)
        bad=replace(req.health,capabilities=(replace(req.health.capabilities[0],status=c.Fact.UNKNOWN),))
        rt.supervise(clock.now(),bad)
        clock.advance(.1);pending=request(rt,clock,seq=20)
        clock.advance(.1)
        revoke=rt.authority.revoke;decisions=[]
        def interleave(*args,**kwargs):
            result=revoke(*args,**kwargs)
            decisions.append(rt.resume(pending))
            return result
        with patch.object(rt.authority,'revoke',side_effect=interleave):
            rt.supervise(clock.now(),bad)
        self.assertIsNone(decisions[0].permit)

    def test_unknown_observation_itself_advances_capture_watermark(self):
        rt,clock=started();req=request(rt,clock);rt.request_permission(req)
        clock.advance(.1)
        unknown_ev=replace(req.readiness.evidence[0],captured_at=s.sensor_stamp(20.1),received_at=clock.now())
        unknown=replace(req.readiness,holding=c.HoldingSnapshot(c.Holding.UNKNOWN,None,None,(unknown_ev,)),evidence=(unknown_ev,))
        rt.observe_phase(m.Phase.CONTACT,unknown)
        clock.advance(.1);fresh=request(rt,clock,seq=20)
        old_ev=replace(fresh.readiness.evidence[0],captured_at=s.sensor_stamp(20.05))
        fresh=replace(fresh,readiness=replace(fresh.readiness,evidence=(old_ev,),holding=replace(fresh.readiness.holding,evidence=(old_ev,))))
        self.assertIsNone(rt.resume(fresh).permit)

    def test_stop_identity_does_not_make_old_capture_physical_stop_proof(self):
        rt,clock=started();req=request(rt,clock);rt.request_permission(req)
        clock.advance(.1);decision=rt.signal(signal(rt,clock,m.SignalKind.FINISH,'signal-2'))
        clock.advance(.1)
        stop=decision.stop
        ev=replace(req.old_stop.evidence[0],action_id=stop.envelope.action_id,captured_at=s.stamp(99),received_at=clock.now())
        report=replace(req.old_stop,envelope=replace(stop.envelope,producer=c.Owner.P01,emitted_at=clock.now()),evidence=(ev,))
        marker=m.EndMarker(rt.run.envelope.run_id,rt.run.deadline,clock.now(),False,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,m.CleanupState.RETAINED)
        rt.accept_end(marker,report)
        self.assertNotEqual(rt.state,m.RunState.ENDED)


if __name__=='__main__':unittest.main()
