"""Synthetic metres/seconds only: these values are not hardware calibration."""
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c, models as m
try:
    from treasure_auto.navigation.field_regions import FieldRegions
    from treasure_auto.navigation.controlled_push import (
        ControlledPushService, PushLimits, PushPlan, PushObservation)
except ImportError:
    FieldRegions = ControlledPushService = None


def box(x, y, w=.1, h=.1):
    return tuple(m.Point2(a, b) for a, b in ((x,y),(x+w,y),(x+w,y+h),(x,y+h)))


def regions():
    return FieldRegions('map', box(0,0,10,10), box(3,3,4,4),
                        {a:box(x,y,1,1) for a,x,y in [('A',1,1),('B',8,1),('C',1,8),('D',8,8)]},
                        {a:box(x,y,.45,.45) for a,x,y in [('A',1,1),('B',8,1),('C',1,8),('D',8,8)]},
                        (box(6,6,.3,.3),), (box(7,5,.1,1),))


def limits(**changes):
    values=dict(max_step_distance_m=.2,max_step_duration_s=2.,max_speed_m_s=.1,
                max_observation_age_s=.5,stall_window_s=.5,min_fixed_progress_m=.005,
                stop_linear_m_s=.001,stop_angular_rad_s=.001,stop_window_s=.2,
                stability_window_s=.5)
    values.update(changes)
    return PushLimits(**values)


def plan(**changes):
    values=dict(group_id='pile',action_id='push-1',frame_id='map',affected_ids=('brick-1','brick-2'),
                target_footprint=box(4,4,.3,.3),body_sweep=box(3.5,4,.5,.3),
                brick_sweep=box(4,4,.6,.3),exit_sweep=box(3.3,4,.7,.3),error_m=.01,
                direction_xy=(1.,0.),distance_m=.2,duration_s=2.,speed_m_s=.1)
    values.update(changes)
    return PushPlan(**values)


def obs(t=10., **changes):
    values=dict(sample_id='frame-'+str(t),captured_at=t,frame_id='map',group_id='pile',
                affected_ids=('brick-1','brick-2'),fixed_position_xy=(3.5,4.),wheel_distance_m=0.,
                linear_m_s=0.,angular_rad_s=0.,body_footprint=box(3.5,4,.2,.3),
                brick_footprint=box(4,4,.3,.3),localization_valid=True,observation_valid=True,
                space_confirmed=True,affected_visible=True,affected_stable=True,graspable_count=0,
                old_action_stopped=True,queue_empty=True,stable_since=t-1)
    values.update(changes)
    return PushObservation(**values)


class FieldRegionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(FieldRegions, 'field-region policy must be implemented')
        self.regions=regions()

    def test_opening_and_region_priorities(self):
        priority=self.regions.recovery_priority
        self.assertEqual(priority(box(4,4),normal_known_layout=True,abnormal=True,push_safe=True),'grasp')
        self.assertEqual(priority(box(4,4),abnormal=True,reliable_grasp=True,push_safe=True),'push')
        self.assertEqual(priority(box(2.5,2.5),reliable_grasp=True,push_safe=True),'grasp')
        self.assertEqual(priority(box(2.5,2.5),reliable_grasp=False,push_safe=True),'push')
        self.assertEqual(priority(box(1.7,1.7),abnormal=True,push_safe=True),'arm_recovery')
        self.assertEqual(priority(box(4,4),abnormal=True,push_safe=False),'wait')

    def test_all_four_full_areas_prohibit_push_including_transport(self):
        for x,y in ((1.7,1.7),(8.7,1.7),(1.7,8.7),(8.7,8.7)):
            self.assertTrue(self.regions.push_prohibited(box(x,y)))
        self.assertTrue(self.regions.push_prohibited(box(2.01,1.5),error_m=.02))
        self.assertTrue(self.regions.push_prohibited(box(6.1,6.1)))
        self.assertTrue(self.regions.push_prohibited(box(7,5.2)))

    def test_whole_footprint_and_uncertainty_not_center_lookup(self):
        self.assertEqual(self.regions.area_of(box(1.8,1.8,.3,.3)),'outside')
        self.assertEqual(self.regions.area_of(box(1.8,1.8),error_m=.11),'outside')
        self.assertEqual(self.regions.area_of(box(1.8,1.8)),'A')
        self.assertEqual(self.regions.area_of(box(0,4)), 'boundary')
        self.assertTrue(self.regions.push_prohibited(box(4,4),frame_id='camera'))

    def test_transport_temporary_ground_includes_a_and_requires_confirmed_clearance(self):
        check=self.regions.temporary_ground_ok
        for area,x,y in (('A',1.7,1.7),('B',8.7,1.7),('C',1.7,8.7),('D',8.7,8.7)):
            self.assertTrue(check(area,box(x,y),.01,True,True,True))
        self.assertFalse(check('A',box(1.2,1.2),.01,True,True,True))
        self.assertFalse(check('A',box(1.7,1.7)))
        self.assertFalse(check('A',box(1.7,1.7),.01,True,True,True,(box(1.75,1.75),)))


class ControlledPushTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ControlledPushService,'bounded push state machine must be implemented')
        self.service=ControlledPushService(regions(),limits())

    def start(self, **changes):
        result=self.service.start(plan(**changes),obs(),10.)
        self.assertEqual(result.action,'push_step',result)
        return result

    def settle(self, t, graspable=0):
        self.service.observe(obs(t,stable_since=t),t)
        result=self.service.observe(obs(t+.21,stable_since=t),t+.21)
        self.assertEqual(result.action,'reobserve',result)
        return self.service.observe(obs(t+.8,stable_since=t,graspable_count=graspable),t+.8)

    def test_missing_measurements_and_live_never_start(self):
        for value in (None,float('nan'),0.):
            service=ControlledPushService(regions(),limits(max_step_distance_m=value))
            self.assertEqual(service.start(plan(),obs(),10.).action,'denied')
        service=ControlledPushService(regions(),limits(),origin=c.Origin.LIVE)
        self.assertEqual(service.start(plan(),obs(),10.).action,'denied')
        self.assertEqual(self.service.start(plan(),obs(origin=c.Origin.LIVE),10.).action,'denied')

    def test_plan_all_sweeps_and_dynamic_obstacles_checked(self):
        for changes in ({'target_footprint':box(1.7,1.7)}, {'body_sweep':box(7,5)},
                        {'brick_sweep':box(6,6)}, {'exit_sweep':box(-.1,4)},
                        {'speed_m_s':.11},{'distance_m':.21},{'duration_s':2.1},
                        {'error_m':float('nan')},{'frame_id':'camera'}):
            with self.subTest(changes=changes):
                self.assertEqual(self.service.start(plan(**changes),obs(),10.).action,'denied')
        self.assertEqual(self.service.start(plan(),obs(other_obstacles=(box(3.7,4),)),10.).action,'denied')

    def test_unknown_or_stale_start_evidence_denies(self):
        for changes in ({'localization_valid':False},{'space_confirmed':False},{'affected_visible':False},
                        {'old_action_stopped':False},{'queue_empty':False},{'fixed_position_xy':None},
                        {'captured_at':9.},{'linear_m_s':float('nan')},{'affected_ids':('brick-1',)}):
            self.assertEqual(self.service.start(plan(),obs(**changes),10.).action,'denied',changes)

    def test_wheel_motion_without_fixed_environment_motion_stops_and_cannot_retry(self):
        self.start()
        result=self.service.observe(obs(10.5,wheel_distance_m=.1,linear_m_s=.1),10.5)
        self.assertEqual(result.action,'stop')
        self.assertIn('stall',result.reason)
        self.assertEqual(self.service.start(plan(action_id='push-2'),obs(10.6),10.6).action,'stop')
        self.assertEqual(self.settle(10.7).action,'ended')

    def test_time_and_fixed_distance_each_stop_step(self):
        for sample in (obs(10.2,fixed_position_xy=(3.71,4.)),obs(12.,fixed_position_xy=(3.6,4.))):
            self.service=ControlledPushService(regions(),limits())
            self.start()
            self.assertEqual(self.service.observe(sample,sample.captured_at).action,'stop')

    def test_observation_loss_and_actual_boundary_crossing_stop(self):
        for changes in ({'body_footprint':box(6,6)}, {'brick_footprint':box(1.7,1.7)},
                        {'localization_valid':False},{'observation_valid':False},
                        {'affected_visible':False},{'abnormal_motion':True},
                        {'fixed_position_xy':(float('nan'),4.)}):
            self.service=ControlledPushService(regions(),limits());self.start()
            self.assertEqual(self.service.observe(obs(10.1,**changes),10.1).action,'stop',changes)

    def test_duplicate_observation_cannot_refresh_progress_or_emit_motion(self):
        self.start()
        duplicate=self.service.start(plan(),obs(),10.)
        self.assertNotEqual(duplicate.action,'push_step')
        self.assertEqual(duplicate.attempts,1)
        self.assertEqual(self.service.observe(obs(),10.2).action,'wait')
        self.assertEqual(self.service.observe(obs(),10.6).action,'stop')

    def test_stopping_requires_actual_stop_queue_empty_then_separate_stable_frame(self):
        self.start();self.service.observe(obs(10.2,fixed_position_xy=(3.71,4.)),10.2)
        self.assertEqual(self.service.observe(obs(10.3,old_action_stopped=False),10.3).action,'stop')
        self.assertEqual(self.service.observe(obs(10.4,queue_empty=False),10.4).action,'stop')
        self.assertEqual(self.service.start(plan(action_id='push-2'),obs(10.45),10.45).action,'stop')
        self.service.observe(obs(10.5),10.5)
        result=self.service.observe(obs(10.71),10.71)
        self.assertEqual(result.action,'reobserve')
        self.assertEqual(self.service.observe(obs(10.71),10.72).action,'wait')
        self.assertEqual(self.service.observe(obs(10.8,affected_stable=False),10.8).action,'reobserve')
        self.assertEqual(self.service.observe(obs(11.,stable_since=10.4),11.).action,'ready')

    def test_one_graspable_stable_brick_ends_push_without_flattening_group(self):
        self.start();self.service.observe(obs(10.2,fixed_position_xy=(3.71,4.)),10.2)
        result=self.settle(10.3,graspable=1)
        self.assertEqual(result.action,'grasp')
        self.assertEqual(result.attempts,1)
        self.assertNotEqual(self.service.start(plan(action_id='push-2'),obs(11.2),11.2).action,'push_step')

    def test_graspable_neighbor_cannot_finish_while_other_affected_bricks_move(self):
        self.start();self.service.observe(obs(10.2,fixed_position_xy=(3.71,4.)),10.2)
        self.service.observe(obs(10.3),10.3);self.service.observe(obs(10.51),10.51)
        self.assertEqual(self.service.observe(obs(11.,graspable_count=1,affected_stable=False),11.).action,'reobserve')

    def test_three_attempts_independent_of_thirty_second_limit(self):
        self.start()
        for attempt,t in ((1,10.),(2,12.),(3,14.)):
            if attempt>1:
                self.assertEqual(self.service.start(plan(action_id='push-'+str(attempt)),obs(t),t).action,'push_step')
            self.service.observe(obs(t+.2,fixed_position_xy=(3.71,4.)),t+.2)
            result=self.settle(t+.3)
            self.assertEqual(result.action,'ready' if attempt<3 else 'ended')
        self.assertEqual(result.attempts,3)
        self.assertLess(result.elapsed_s,30)

    def test_stop_and_observation_wait_count_toward_thirty_seconds(self):
        self.start();self.service.observe(obs(10.2,fixed_position_xy=(3.71,4.)),10.2)
        self.assertEqual(self.service.tick(40.).action,'stop')
        result=self.settle(40.1)
        self.assertEqual(result.action,'ended')
        self.assertEqual(result.attempts,1)
        self.assertGreaterEqual(result.elapsed_s,30.)

    def test_restored_unresolved_action_stops_without_resend_or_budget_refund(self):
        self.start()
        state=json.loads(json.dumps(self.service.checkpoint(10.2)))
        restored=ControlledPushService(regions(),limits())
        result=restored.restore(state,20.)
        self.assertEqual(result.action,'stop')
        self.assertEqual(result.attempts,1)
        self.assertGreaterEqual(result.elapsed_s,10.)
        self.assertNotEqual(restored.start(plan(),obs(20.1),20.1).action,'push_step')
        self.service=restored
        self.assertEqual(self.settle(20.2).action,'ended')

    def test_group_renaming_does_not_refund_budget_for_same_physical_bricks(self):
        self.start();self.service.observe(obs(10.5,wheel_distance_m=.1),10.5);self.settle(10.6)
        result=self.service.start(plan(group_id='renamed',action_id='next'),obs(12.,group_id='renamed'),12.)
        self.assertNotEqual(result.action,'push_step')
        self.assertEqual(result.attempts,1)

    def test_older_clock_or_changed_sample_payload_cannot_move_again(self):
        self.start()
        self.assertEqual(self.service.observe(obs(10.,fixed_position_xy=(3.7,4.)),10.1).action,'stop')
        self.assertNotEqual(self.service.start(plan(action_id='next'),obs(9.9),9.9).action,'push_step')

    def test_sweep_must_cover_translated_body_bricks_and_exit(self):
        for change in ({'body_sweep':box(3.5,4,.2,.3)},
                       {'brick_sweep':box(4,4,.3,.3)},
                       {'exit_sweep':box(3.3,4,.4,.3)}):
            self.assertEqual(self.service.start(plan(**change),obs(),10.).action,'denied',change)

    def test_fresh_frame_after_unobserved_gap_cannot_rescue_motion(self):
        self.start()
        result=self.service.observe(obs(10.6,fixed_position_xy=(3.6,4.)),10.6)
        self.assertEqual(result.action,'stop')
        self.assertIn('gap',result.reason)

    def test_excess_actual_speed_and_wrong_direction_stop_immediately(self):
        for changes in ({'linear_m_s':.11},{'fixed_position_xy':(3.49,4.)}):
            self.service=ControlledPushService(regions(),limits());self.start()
            self.assertEqual(self.service.observe(obs(10.1,**changes),10.1).action,'stop')

    def test_gap_in_stop_samples_does_not_prove_continuous_stop(self):
        self.start();self.service.observe(obs(10.2,fixed_position_xy=(3.71,4.)),10.2)
        self.service.observe(obs(10.3),10.3)
        self.assertEqual(self.service.observe(obs(11.),11.).action,'stop')
        self.assertEqual(self.service.observe(obs(11.21),11.21).action,'reobserve')

    def test_invalid_observation_after_stop_requires_new_stop_window(self):
        self.start();self.service.observe(obs(10.2,fixed_position_xy=(3.71,4.)),10.2)
        self.service.observe(obs(10.3),10.3);self.service.observe(obs(10.51),10.51)
        self.assertEqual(self.service.observe(obs(10.6,localization_valid=False),10.6).action,'stop')
        self.assertEqual(self.service.observe(obs(10.7,graspable_count=1),10.7).action,'stop')

    def test_distance_budget_counts_fixed_environment_path_not_just_displacement(self):
        self.start()
        self.assertEqual(self.service.observe(obs(10.1,fixed_position_xy=(3.64,4.)),10.1).action,'wait')
        self.assertEqual(self.service.observe(obs(10.2,fixed_position_xy=(3.56,4.)),10.2).action,'stop')

    def test_pre_step_captured_observation_cannot_establish_progress(self):
        self.assertEqual(self.service.start(plan(),obs(9.9),10.).action,'push_step')
        result=self.service.observe(obs(9.95,fixed_position_xy=(3.6,4.)),10.05)
        self.assertEqual(result.action,'stop')

    def test_reused_action_identity_across_groups_never_issues_again(self):
        self.start();self.service.observe(obs(10.5),10.5);self.settle(10.6)
        result=self.service.start(plan(group_id='other',affected_ids=('new-brick',)),
                                  obs(12.,group_id='other',affected_ids=('new-brick',)),12.)
        self.assertEqual(result.action,'denied')

    def test_restore_does_not_change_limits_or_accept_live_origin(self):
        self.start();state=self.service.checkpoint(10.2)
        live=ControlledPushService(regions(),limits(),c.Origin.LIVE)
        self.assertEqual(live.restore(state,11.).action,'denied')
        unmeasured=ControlledPushService(regions(),PushLimits())
        self.assertEqual(unmeasured.restore(state,11.).action,'denied')

    def test_expired_budget_wins_over_new_fresh_progress(self):
        self.start()
        self.assertEqual(self.service.observe(obs(40.,fixed_position_xy=(3.6,4.)),40.).action,'stop')
        self.assertEqual(self.service.start(plan(action_id='next'),obs(40.1),40.1).action,'stop')

    def test_completion_publishes_grasp_once_not_from_ticks_or_replayed_frame(self):
        self.start();self.service.observe(obs(10.2,fixed_position_xy=(3.71,4.)),10.2)
        self.assertEqual(self.settle(10.3,graspable=1).action,'grasp')
        self.assertEqual(self.service.tick(11.2).action,'ended')
        self.assertEqual(self.service.observe(obs(11.1,stable_since=10.3,graspable_count=1),11.2).action,'ended')
        self.assertEqual(self.service.tick(100.).action,'ended')

    def test_completed_push_still_stops_on_invalid_or_moving_current_feedback(self):
        for changes in ({'localization_valid':False},{'observation_valid':False},
                        {'affected_visible':False},{'linear_m_s':.1},
                        {'old_action_stopped':False},{'queue_empty':False},
                        {'captured_at':10.}):
            with self.subTest(changes=changes):
                self.service=ControlledPushService(regions(),limits());self.start()
                self.service.observe(obs(10.2,fixed_position_xy=(3.71,4.)),10.2)
                self.assertEqual(self.settle(10.3,graspable=1).action,'grasp')
                result=self.service.observe(obs(11.2,stable_since=10.3,graspable_count=1,**changes),11.2)
                self.assertEqual(result.action,'stop')
                self.assertNotEqual(self.service.tick(11.3).action,'grasp')

    def test_completed_push_waits_if_affected_bricks_no_longer_stable(self):
        self.start();self.service.observe(obs(10.2,fixed_position_xy=(3.71,4.)),10.2)
        self.assertEqual(self.settle(10.3,graspable=1).action,'grasp')
        result=self.service.observe(obs(11.2,affected_stable=False,graspable_count=1),11.2)
        self.assertEqual(result.action,'reobserve')

    def test_restored_completed_push_never_republishes_old_grasp_permission(self):
        self.start();self.service.observe(obs(10.2,fixed_position_xy=(3.71,4.)),10.2)
        self.assertEqual(self.settle(10.3,graspable=1).action,'grasp')
        state=json.loads(json.dumps(self.service.checkpoint(11.2)))
        restored=ControlledPushService(regions(),limits())
        result=restored.restore(state,100.)
        self.assertEqual(result.action,'ended')
        self.assertEqual(result.attempts,1)
        self.assertEqual(restored.tick(100.1).action,'ended')
        self.assertEqual(restored.observe(obs(100.2,graspable_count=1),100.2).action,'ended')

    def test_invalid_sample_inside_stop_window_breaks_continuous_stop_evidence(self):
        for changes in ({'localization_valid':False},{'observation_valid':False},
                        {'affected_visible':False},{'linear_m_s':float('nan')},
                        {'linear_m_s':.1},{'queue_empty':False}):
            with self.subTest(changes=changes):
                self.service=ControlledPushService(regions(),limits());self.start()
                self.service.observe(obs(10.2,fixed_position_xy=(3.71,4.)),10.2)
                self.assertEqual(self.service.observe(obs(10.3),10.3).action,'stop')
                self.assertEqual(self.service.observe(obs(10.4,**changes),10.4).action,'stop')
                self.assertEqual(self.service.observe(obs(10.51),10.51).action,'stop')
                self.assertEqual(self.service.observe(obs(10.72),10.72).action,'reobserve')


if __name__ == '__main__':
    unittest.main()
