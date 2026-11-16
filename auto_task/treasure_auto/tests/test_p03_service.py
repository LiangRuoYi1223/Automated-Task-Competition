import unittest
from test_p03_support import *
try:
    from treasure_auto.navigation.service import NavigationService
except ImportError:NavigationService=None

class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(NavigationService,'P03 state machine required')
        self.nav,self.clock,self.gateway=service()

    def test_R01_invalid_goal_localization_holding_posture_permit_no_motion(self):
        req=request()
        cases=[replace(req,goal=replace(req.goal,valid=c.Fact.UNKNOWN)),
               replace(req,localization=replace(req.localization,trusted=c.Fact.UNKNOWN)),
               replace(req,readiness=readiness(state=c.Holding.UNKNOWN)),
               replace(req,readiness=replace(readiness(),posture_confirmed=c.Fact.UNKNOWN)),
               replace(req,permit=replace(req.permit,source=m.MotionSource.DEBUG)),
               replace(req,envelope=replace(req.envelope,instance_id='old'))]
        for bad in cases:
            with self.subTest(bad=bad):
                nav,_,gate=service();result=nav.navigate(bad)
                self.assertEqual(result.result.status,c.Outcome.REJECTED)
                self.assertEqual(gate.commands,[])

    def test_R01_duplicate_and_busy_requests_do_not_replace_active(self):
        req=request();first=self.nav.navigate(req)
        self.assertEqual(self.nav.navigate(req),first)
        other=replace(req,envelope=replace(req.envelope,request_id='nav2'))
        self.assertEqual(self.nav.navigate(other).result.status,c.Outcome.REJECTED)
        changed=replace(req,goal=replace(req.goal,name='different'))
        self.assertEqual(self.nav.navigate(changed).result.reason.code,c.ErrorCode.ID_CONFLICT)
        self.assertEqual(self.nav.active,req)

    def test_goal_requires_exact_committed_intent_before_backend_start(self):
        self.nav.navigate(request());intent=self.nav.goal_intent
        bad=replace(receipt(intent),commit=m.CommitReceipt('bad',m.CommitStatus.UNKNOWN,None,None,
                    c.Reason(c.ErrorCode.STORAGE_FAILURE,'intent','unknown')))
        self.assertFalse(self.nav.submit_goal(bad).accepted)
        self.assertFalse(any(e.kind=='start_goal' for e in self.nav.effects))
        self.assertTrue(self.nav.submit_goal(receipt(intent)).accepted)
        self.assertEqual(sum(e.kind=='start_goal' for e in self.nav.effects),1)
        self.assertTrue(self.nav.submit_goal(receipt(intent)).accepted)
        self.assertEqual(sum(e.kind=='start_goal' for e in self.nav.effects),1)

    def test_T01_arrived_but_moving_never_opens_work(self):
        start(self.nav);self.nav.coarse_event(arrived(self.nav,self.clock))
        self.nav.observe_stop(stop_report(self.nav,self.clock))
        self.nav.observe_motion(motion(self.nav,self.clock,v=.1))
        self.clock.advance(.21);self.nav.observe_motion(motion(self.nav,self.clock,v=.1))
        self.assertEqual(self.nav.phase,'handoff')
        self.assertEqual(self.nav.current_result().work_ready,c.Fact.UNKNOWN)
        self.assertFalse(self.nav.install_precision(precision_permit(self.nav,self.clock)).accepted)

    def test_T02_revoke_cancel_then_new_permission_late_success_ignored(self):
        start(self.nav);event=arrived(self.nav,self.clock)
        self.nav.coarse_event(event)
        kinds=[e.kind for e in self.nav.effects]
        self.assertLess(kinds.index('revoke'),kinds.index('cancel_goal'))
        stopped(self.nav,self.clock)
        self.assertEqual(self.nav.phase,'precision_ready')
        self.assertTrue(self.nav.install_precision(precision_permit(self.nav,self.clock)).accepted)
        self.nav.coarse_event(replace(event,envelope=replace(event.envelope,event_id='late-success')))
        self.assertEqual(self.nav.phase,'precision')
        self.assertIsNone(self.nav.prepare_velocity(.1,0,m.MotionSource.NAVIGATION))

    def test_handoff_requires_queue_confirmation_and_new_samples(self):
        start(self.nav);self.nav.coarse_event(arrived(self.nav,self.clock))
        self.nav.observe_stop(stop_report(self.nav,self.clock,old_queue_empty=c.Fact.UNKNOWN))
        sample=motion(self.nav,self.clock)
        self.nav.observe_motion(sample);self.clock.advance(.21);self.nav.observe_motion(sample)
        self.assertEqual(self.nav.phase,'handoff')
        self.nav.observe_motion(motion(self.nav,self.clock));self.nav.observe_stop(stop_report(self.nav,self.clock))
        self.assertEqual(self.nav.phase,'precision_ready')

    def test_R04_missing_distance_no_forward_self_occlusion_requests_P07(self):
        for values in ({'forward_m':None},{'distance_basis':m.PlaneBasis.UNKNOWN},{'self_occluded':True}):
            nav,clock,gate=service();to_precision(nav,clock)
            self.assertIsNone(nav.precision(feedback(nav,clock,**values)))
            self.assertEqual(gate.commands,[])
            self.assertNotEqual(nav.current_result().work_ready,c.Fact.CONFIRMED)
            if 'self_occluded' in values:self.assertTrue(any(e.kind=='request_P07_observation' for e in nav.effects))

    def test_R04_differential_turn_then_forward_no_lateral_command(self):
        to_precision(self.nav,self.clock)
        self.clock.advance(.1)
        turn=self.nav.precision(feedback(self.nav,self.clock,forward_m=.2,lateral_m=.2))
        self.assertIsNotNone(turn)
        self.assertEqual(turn.base_velocity_m_s_rad_s[0],0.)
        self.assertGreater(turn.base_velocity_m_s_rad_s[1],0.)
        self.assertEqual(len(turn.base_velocity_m_s_rad_s),2)
        self.clock.advance(.1)
        forward=self.nav.precision(feedback(self.nav,self.clock,forward_m=.2))
        self.assertGreater(forward.base_velocity_m_s_rad_s[0],0.)
        self.assertLessEqual(forward.base_velocity_m_s_rad_s[0],policy().max_linear_m_s)

    def test_R05_success_requires_final_fresh_stop_and_conditions(self):
        to_precision(self.nav,self.clock)
        self.nav.precision(feedback(self.nav,self.clock))
        self.assertFalse(self.nav.terminal)
        self.assertEqual(self.nav.phase,'final_stop')
        stopped(self.nav,self.clock)
        result=self.nav.current_result()
        self.assertTrue(self.nav.terminal)
        self.assertEqual(result.result.status,c.Outcome.SUCCESS)
        self.assertEqual(result.work_ready,c.Fact.CONFIRMED)
        self.assertEqual(result.residual_pose.x,0.)
        self.assertIsNone(result.exit_evidence)
        self.clock.advance(1.1)
        self.assertEqual(self.nav.current_result().work_ready,c.Fact.UNKNOWN)

    def test_T03_space_change_cancels_current_path_and_pending_intent(self):
        start(self.nav)
        intent=self.nav.prepare_velocity(.1,0,m.MotionSource.NAVIGATION)
        changed=space(version=1,items=(obstacle(),))
        self.assertTrue(self.nav.invalidate_space(changed).accepted)
        self.assertEqual(self.nav.current_result().result.reason.code,c.ErrorCode.VERSION_MISMATCH)
        self.assertTrue(self.nav.terminal)
        self.assertIsNone(self.nav.dispatch_velocity(intent,receipt(intent),request().permit))
        self.assertEqual(self.gateway.commands,[])
        self.assertEqual(changed.occupancies[0].state,m.OccupancyState.UNKNOWN)

    def test_wrong_version_update_and_same_version_conflict_are_not_accepted(self):
        start(self.nav)
        self.assertTrue(self.nav.invalidate_space(request().space).accepted)
        self.assertFalse(self.nav.invalidate_space(space(items=(obstacle(),))).accepted)
        self.assertTrue(self.nav.terminal)

    def test_cancel_timeout_and_late_callbacks_have_one_terminal_result(self):
        start(self.nav);old=arrived(self.nav,self.clock)
        self.clock.advance(8.)
        self.nav.tick();result=self.nav.current_result()
        self.assertEqual(result.result.status,c.Outcome.TIMED_OUT)
        self.nav.coarse_event(old);self.nav.tick()
        self.assertEqual(self.nav.current_result().result,result.result)
        self.assertEqual(sum(e.kind=='terminal' for e in self.nav.effects),1)

    def test_unknown_send_result_stops_and_no_retry_of_same_intent(self):
        start(self.nav)
        intent=self.nav.prepare_velocity(.1,0,m.MotionSource.NAVIGATION)
        def fail(cmd):raise OSError('unresolved send')
        self.gateway.submit_base=fail
        self.assertIsNone(self.nav.dispatch_velocity(intent,receipt(intent),request().permit))
        self.assertTrue(self.nav.terminal)
        self.assertEqual(self.nav.current_result().result.status,c.Outcome.UNKNOWN)

    def test_motion_epoch_clears_old_geometry_and_pending_commands(self):
        to_precision(self.nav,self.clock)
        intent=self.nav.precision(feedback(self.nav,self.clock,forward_m=.2))
        self.clock.advance(.1)
        l=loc(self.clock.t,p=pose(2.02,2),base_motion_epoch=1)
        rd=readiness(self.clock.t,base_motion_epoch=1)
        self.nav.refresh(l,rd)
        self.assertIsNone(self.nav.dispatch_velocity(intent,receipt(intent),precision_permit(self.nav,self.clock)))
        self.assertTrue(any(e.kind=='invalidate_geometry' for e in self.nav.effects))

    def test_original_deadline_clock_rollback_and_expired_evidence_stop(self):
        for change in ('deadline','backward','stale'):
            nav,clock,gate=service();start(nav)
            clock.t={'deadline':700,'backward':109,'stale':111.1}[change]
            nav.tick()
            self.assertTrue(nav.terminal)
            self.assertNotEqual(nav.current_result().work_ready,c.Fact.CONFIRMED)
            self.assertTrue(gate.revocations)
