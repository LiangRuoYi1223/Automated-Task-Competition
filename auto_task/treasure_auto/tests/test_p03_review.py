import unittest
from test_p03_support import *
from test_p03_recovery import certificate

class ReviewTests(unittest.TestCase):
    def test_goal_effect_is_not_backend_submission_acknowledgement(self):
        nav,clock,_=service();start(nav)
        self.assertEqual(nav.current_result().result.submitted,c.Fact.UNKNOWN)
        nav.coarse_event(arrived(nav,clock))
        self.assertEqual(nav.current_result().result.submitted,c.Fact.CONFIRMED)

    def test_new_unknown_queue_cannot_be_overwritten_by_old_success(self):
        nav,clock,_=service();start(nav);nav.coarse_event(arrived(nav,clock))
        old=stop_report(nav,clock);stopped(nav,clock)
        clock.advance(.01)
        self.assertTrue(nav.observe_stop(stop_report(nav,clock,old_queue_empty=c.Fact.UNKNOWN)).accepted)
        self.assertFalse(nav.observe_stop(old).accepted)
        self.assertFalse(nav.install_precision(precision_permit(nav,clock)).accepted)

    def test_new_stop_report_missing_proof_invalidates_old_positive(self):
        nav,clock,_=service();start(nav);nav.coarse_event(arrived(nav,clock));stopped(nav,clock)
        clock.advance(.01)
        self.assertFalse(nav.observe_stop(stop_report(nav,clock,evidence=())).accepted)
        self.assertFalse(nav.install_precision(precision_permit(nav,clock)).accepted)

    def test_uncommitted_space_producer_cannot_start_navigation(self):
        nav,_,gate=service();req=request()
        req=replace(req,space=replace(req.space,envelope=replace(req.space.envelope,producer=c.Owner.P09)))
        self.assertEqual(nav.navigate(req).result.status,c.Outcome.REJECTED)
        self.assertEqual(gate.commands,[])

    def test_movement_high_water_prevents_replaying_previously_stable_frames(self):
        nav,clock,_=service();start(nav);nav.coarse_event(arrived(nav,clock));nav.observe_stop(stop_report(nav,clock))
        first=motion(nav,clock);nav.observe_motion(first)
        clock.advance(.21);second=motion(nav,clock);nav.observe_motion(second)
        clock.advance(.01);nav.observe_motion(motion(nav,clock,v=.3))
        nav.observe_motion(first);nav.observe_motion(second)
        self.assertFalse(nav.install_precision(precision_permit(nav,clock)).accepted)

    def test_collapse_motion_requires_new_stop_anchor_and_static_window(self):
        nav,clock,_=service();start(nav);nav.collapse(('low-brick',),poly(0,0,6,6),1);stopped(nav,clock)
        clock.advance(.1);nav.observe_motion(motion(nav,clock,v=.3))
        clock.advance(.1);nav.observe_motion(motion(nav,clock))
        clock.advance(.21);nav.observe_motion(motion(nav,clock))
        clock.advance(.1);nav.observe_motion(motion(nav,clock))
        self.assertFalse(nav.confirm_stillness(certificate(nav,clock,.5)).accepted)

    def test_actual_precision_motion_then_new_geometry_can_complete(self):
        nav,clock,_=service();to_precision(nav,clock);clock.advance(.1)
        permit=nav._permit
        intent=nav.precision(feedback(nav,clock,forward_m=.1))
        self.assertIsNotNone(nav.dispatch_velocity(intent,receipt(intent),permit))
        clock.advance(.05)
        nav.precision(feedback(nav,clock));stopped(nav,clock)
        self.assertTrue(nav.terminal)
        self.assertEqual(nav.current_result().result.status,c.Outcome.SUCCESS)
        self.assertEqual(nav.current_result().work_ready,c.Fact.CONFIRMED)

    def test_T05_valid_collapse_window_and_current_stop_can_resume(self):
        nav,clock,_=service();start(nav)
        nav.collapse(('low-brick',),poly(0,0,6,6),1);stopped(nav,clock)
        anchor=clock.t
        for i in range(3):
            clock.advance(.2);nav.observe_motion(motion(nav,clock))
        cert=certificate(nav,clock,clock.t-anchor)
        self.assertTrue(nav.confirm_stillness(cert).accepted)
        nav.refresh(loc(clock.t),readiness(clock.t))
        self.assertTrue(nav.recover(request(clock.t).permit).accepted)
        self.assertEqual(nav.phase,'planned')
        self.assertEqual(nav.budget.attempts_used,1)

    def test_same_motion_id_changed_payload_revokes_previous_stable_window(self):
        nav,clock,_=service();start(nav);nav.coarse_event(arrived(nav,clock));stopped(nav,clock)
        self.assertEqual(nav.phase,'precision_ready')
        moving=motion(nav,clock,v=.3)
        nav.observe_motion(moving)
        self.assertFalse(nav.install_precision(precision_permit(nav,clock)).accepted)

    def test_blocked_recovery_time_survives_checkpoint_and_original_request_deadline(self):
        nav,clock,_=service();start(nav);nav.blocked('blocked');clock.advance(2.)
        nav.tick()
        self.assertTrue(nav.terminal)
        self.assertEqual(nav.current_result().result.status,c.Outcome.TIMED_OUT)
        self.assertGreaterEqual(nav.budget.seconds_used,2.)

    def test_unresolved_cancel_send_cannot_be_hidden_by_unrelated_stop_report(self):
        nav,clock,gate=service();start(nav)
        gate.revoke=lambda source:False
        nav.coarse_event(arrived(nav,clock))
        # A later physical proof and queue ACK may resolve a previously unknown
        # send, but a report for another request never can.
        bad=stop_report(nav,clock)
        bad=replace(bad,envelope=replace(bad.envelope,request_id='other'))
        self.assertFalse(nav.observe_stop(bad).accepted)
        nav.observe_motion(motion(nav,clock));clock.advance(.21);nav.observe_motion(motion(nav,clock))
        self.assertEqual(nav.phase,'handoff')

    def test_precision_dispatch_rechecks_current_geometry_and_preserves_exact_receipt(self):
        nav,clock,gate=service();to_precision(nav,clock);clock.advance(.1)
        intent=nav.precision(feedback(nav,clock,forward_m=.1))
        wrong=replace(intent,base_velocity_m_s_rad_s=(.2,0.))
        self.assertIsNone(nav.dispatch_velocity(wrong,receipt(wrong),nav._permit))
        self.assertEqual(gate.commands,[])
        clock.advance(1.1)
        self.assertIsNone(nav.dispatch_velocity(intent,receipt(intent),nav._permit))
        self.assertEqual(gate.commands,[])
