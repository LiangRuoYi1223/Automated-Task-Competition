import unittest
from test_p03_support import *
from treasure_auto.navigation.observations import StopWindow,MotionSample

class BoundaryTests(unittest.TestCase):
    def test_stop_unknown_malformed_future_and_gap_cannot_confirm(self):
        for bad in (True,float('nan'),float('inf')):
            nav,clock,_=service();start(nav);nav.coarse_event(arrived(nav,clock));nav.observe_stop(stop_report(nav,clock))
            nav.observe_motion(motion(nav,clock,v=bad));clock.advance(.21);nav.observe_motion(motion(nav,clock))
            self.assertEqual(nav.phase,'handoff')
        nav,clock,_=service();start(nav);nav.coarse_event(arrived(nav,clock));nav.observe_stop(stop_report(nav,clock))
        nav.observe_motion(motion(nav,clock));clock.advance(.3);nav.observe_motion(motion(nav,clock))
        self.assertEqual(nav.phase,'handoff')

    def test_stop_report_unknown_cannot_turn_into_cancellation_confirmation(self):
        nav,clock,_=service();start(nav);nav.coarse_event(arrived(nav,clock))
        nav.observe_stop(stop_report(nav,clock,requested=c.Fact.UNKNOWN))
        nav.observe_motion(motion(nav,clock));clock.advance(.21);nav.observe_motion(motion(nav,clock))
        self.assertEqual(nav.phase,'handoff')

    def test_expired_permit_is_revoked_by_tick_without_output_request(self):
        nav,clock,gate=service();req=request()
        req=replace(req,permit=replace(req.permit,envelope=replace(req.permit.envelope,expires_at=stamp(110.2))))
        start(nav,req);clock.advance(.21);nav.tick()
        self.assertTrue(nav.terminal)
        self.assertTrue(gate.revocations)

    def test_delayed_dispatch_cannot_extend_stale_geometry_lease(self):
        nav,clock,_=service();start(nav);clock.advance(.8)
        intent=nav.prepare_velocity(.1,0,m.MotionSource.NAVIGATION)
        self.assertLessEqual(intent.envelope.expires_at.seconds,111.)

    def test_precision_second_command_needs_observation_after_first_motion(self):
        nav,clock,_=service();to_precision(nav,clock);clock.advance(.1)
        old=feedback(nav,clock,forward_m=.2);intent=nav.precision(old)
        self.assertIsNotNone(nav.dispatch_velocity(intent,receipt(intent),precision_permit(nav,Clock(110.21))))
        clock.advance(.01)
        self.assertIsNone(nav.precision(old))

    def test_success_revokes_current_work_on_invalid_readiness_or_pose_drift(self):
        for bad in ('holding','pose'):
            nav,clock,_=service();to_precision(nav,clock);nav.precision(feedback(nav,clock));stopped(nav,clock)
            self.assertEqual(nav.current_result().work_ready,c.Fact.CONFIRMED)
            clock.advance(.01)
            nav.refresh(loc(clock.t,p=pose(2.1,2) if bad=='pose' else pose()),
                        readiness(clock.t,state=c.Holding.UNKNOWN if bad=='holding' else c.Holding.EMPTY))
            self.assertEqual(nav.current_result().work_ready,c.Fact.UNKNOWN)
            self.assertEqual(sum(e.kind=='terminal' for e in nav.effects),1)

    def test_out_of_order_positive_readiness_does_not_replace_newer_context(self):
        nav,clock,_=service();start(nav);clock.advance(.2)
        nav.refresh(loc(clock.t),readiness(clock.t));clock.advance(.1)
        result=nav.refresh(loc(110.1),readiness(110.1))
        self.assertFalse(result.accepted)
        self.assertTrue(nav.terminal)

    def test_recovery_route_cannot_silently_change_named_destination(self):
        nav,clock,_=service();start(nav);nav.blocked('test');stopped(nav,clock)
        nav.planner=lambda req:(req.localization.pose,pose(4,2))
        self.assertFalse(nav.recover(request(clock.t).permit).accepted)

    def test_result_rejection_identifies_new_request_not_the_running_one(self):
        nav,_,_=service();start(nav)
        second=replace(request(),envelope=replace(request().envelope,request_id='nav2'))
        self.assertEqual(nav.navigate(second).envelope.request_id,'nav2')
        self.assertEqual(nav.active.envelope.request_id,'nav1')

    def test_sequential_request_retains_run_recovery_budget(self):
        nav,clock,_=service();to_precision(nav,clock);nav.precision(feedback(nav,clock));stopped(nav,clock)
        req=request(clock.t,request_id='nav2')
        result=nav.navigate(req)
        self.assertEqual(result.result.accepted,c.Fact.CONFIRMED)
        self.assertEqual(nav.active.envelope.request_id,'nav2')
        self.assertFalse(nav.terminal)

    def test_exit_rejects_wrong_config_and_map_origin_uncertainty(self):
        from treasure_auto.navigation.exit import ExitTracker
        tracker=ExitTracker(run(),bundle().metadata,profile(),1.)
        bad=loc(p=pose(2,.5));bad=replace(bad,envelope=replace(bad.envelope,versions=replace(bad.envelope.versions,config='other')))
        rd=replace(readiness(),envelope=replace(readiness().envelope,versions=bad.envelope.versions))
        self.assertIsNone(tracker.observe(bad,rd,stamp()))

    def test_raw_sensor_capture_cannot_masquerade_as_monotonic_freshness(self):
        nav,_,_=service();req=request()
        evidence=replace(req.localization.evidence[0],captured_at=c.ClockStamp(110,c.ClockBasis.SENSOR,'camera',None))
        req=replace(req,localization=replace(req.localization,evidence=(evidence,)))
        self.assertEqual(nav.navigate(req).result.status,c.Outcome.REJECTED)
