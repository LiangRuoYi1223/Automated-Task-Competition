"""Approved field policies consumed by the real mission boundary, offline only."""
import unittest
from dataclasses import replace

from test_p09_engine import Flow
from test_p09_support import c, m, s
from treasure_auto.mission.inputs import Stage


class FieldIntegrationTests(unittest.TestCase):
    def flow(self):
        h = Flow(mapping=True)
        h.engine.stage = Stage.SELECT
        h.add_candidate()
        return h

    def report(self, h, **changes):
        from treasure_auto.perception.strategy import PerceptionPolicyReport
        result = m.PerceptionResult(h.env(c.Owner.P04, target_id=None, action_id=None),
                                   m.ObservationState.DETECTED, h.candidates, None, None)
        report = PerceptionPolicyReport(
            envelope=result.envelope, observation_id='field-image', image_sha256='a' * 64,
            camera_frame='camera', capture_pose=c.MapPose('map', 0, 0, 0, 'm', 'rad', .01, .01), current=True,
            pattern_count=3, independent_count=2, geometry_count=1,
            exploratory_candidates=(), review_targets=())
        return result, replace(report, **changes)

    def test_report_counts_reach_mission_without_replacing_existing_grasp_checks(self):
        h = self.flow()
        result, report = self.report(h)
        out = h.step(perception_result=result, perception_policy=report)
        self.assertEqual(out.perception_counts, (3, 2, 1))
        self.assertEqual(out.directive.stage, Stage.GRASP)

    def test_expired_report_cannot_dispatch_using_fresh_manually_supplied_candidates(self):
        h = self.flow()
        result, report = self.report(h)
        old = replace(report.envelope, emitted_at=s.stamp(108), expires_at=s.stamp(109))
        report = replace(report, envelope=old)
        result = replace(result, envelope=old)
        out = h.step(perception_result=result, perception_policy=report)
        self.assertIsNone(out.directive)
        self.assertIsNotNone(out.stop_request)

    def test_report_requires_exact_result_and_not_an_unrelated_candidate_list(self):
        h = self.flow()
        result, report = self.report(h)
        out = h.step(perception_result=replace(result, targets=()), perception_policy=report)
        self.assertIsNone(out.directive)

    def test_policy_conflict_withholds_ordinary_candidate(self):
        from treasure_auto.perception.strategy import TargetReview
        h = self.flow()
        result, report = self.report(h)
        review = TargetReview('body', h.candidates[0].observation_id,
                              h.candidates[0].target_id, ('map_correspondence_conflict',))
        out = h.step(perception_result=result, perception_policy=replace(report, review_targets=(review,)))
        self.assertNotEqual(out.stage, Stage.GRASP)
        self.assertIsNone(out.decision.order)

    def test_current_false_report_does_not_publish_old_counts_or_grasp(self):
        h = self.flow()
        result, report = self.report(h, current=False)
        out = h.step(perception_result=result, perception_policy=report)
        self.assertIsNone(out.directive)
        self.assertEqual(out.perception_counts, (0, 0, 0))

    def field_flow(self):
        from test_field_push import regions, limits
        from treasure_auto.mission.field_strategy import FieldStrategyController
        h = self.flow()
        h.engine.field_strategy = FieldStrategyController(regions(), limits())
        return h

    def push_input(self, h):
        from test_field_push import plan, obs, box
        from treasure_auto.mission.field_strategy import FieldInput
        return FieldInput('select', 'brick-1', 'map', box(4,4,.3,.3), .01,
                          abnormal=True, reliable_grasp=True,
                          push_plan=plan(), push_observation=obs(h.clock.now().seconds))

    def test_central_push_preempts_grasp_and_missing_feedback_stops(self):
        h = self.field_flow()
        out = h.step(field_input=self.push_input(h))
        self.assertEqual(out.field_decision.action, 'push_step')
        self.assertIsNone(out.directive)
        self.assertFalse(out.field_decision.authorizes_motion)
        h.clock.advance(.6)
        out = h.step()
        self.assertEqual(out.field_decision.action, 'stop')
        self.assertIsNotNone(out.stop_request)
        self.assertIsNone(out.directive)

    def test_abcd_transport_space_never_pushes(self):
        from test_field_push import box
        h = self.field_flow()
        out = h.step(field_input=replace(self.push_input(h), footprint=box(1.7,1.7)))
        self.assertEqual(out.field_decision.action, 'arm_recovery')
        self.assertIsNone(out.directive)

    def test_navigation_completion_routes_field_policy_before_ordinary_grasp(self):
        from test_field_push import regions, limits
        from treasure_auto.mission.field_strategy import FieldStrategyController
        from test_p09_support import proof
        h = Flow()
        h.board()
        h.add_candidate()
        h.engine.field_strategy = FieldStrategyController(regions(), limits())
        d = h.engine.pending
        event = m.ActionEvent(replace(d.envelope, producer=d.owner, event_id='navigation-finished'),
                              c.ActionResult(c.Outcome.SUCCESS, d.stage.value, None,
                                             c.Fact.CONFIRMED, c.Fact.CONFIRMED, c.Fact.CONFIRMED,
                                             c.Fact.CONFIRMED, (proof(action=d.envelope.action_id),), .1))
        h.events = (event,)
        out = h.step(field_input=self.push_input(h))
        self.assertIsNotNone(out.field_decision, out.issues)
        self.assertEqual(out.field_decision.action, 'push_step')
        self.assertIsNone(out.directive)

    def test_unknown_checkpoint_withholds_push_intent(self):
        h = self.field_flow()
        h.journal.fail = True
        out = h.step(field_input=self.push_input(h))
        self.assertIsNone(out.field_decision)
        self.assertIsNotNone(out.stop_request)

    def test_slow_checkpoint_cannot_publish_expired_push_decision(self):
        h = self.field_flow()
        save = h.journal.save
        def slow_save(payload):
            h.clock.advance(.6)
            return save(payload)
        h.journal.save = slow_save
        out = h.step(field_input=self.push_input(h))
        self.assertIsNone(out.field_decision)
        self.assertIsNotNone(out.stop_request)

    def test_unfinished_push_restart_keeps_attempt_and_never_reissues(self):
        from test_field_push import regions, limits
        from treasure_auto.mission.field_strategy import FieldStrategyController
        from treasure_auto.mission.engine import MissionEngine
        from treasure_auto.mission.checkpoint import SimulationJournal
        h = self.field_flow()
        h.step(field_input=self.push_input(h))
        state = h.journal.load()
        h.clock.advance(.1)
        controller = FieldStrategyController(regions(), limits())
        engine = MissionEngine(h.engine.config, h.clock, SimulationJournal(state), field_strategy=controller)
        out = engine.advance(h.frame())
        self.assertIsNone(out.directive)
        self.assertIsNone(out.field_decision)
        self.assertEqual(controller.push.checkpoint(h.clock.now().seconds)['groups']['pile']['attempts'], 1)

    def test_real_perception_observe_report_reaches_mission_without_faking_grasp_geometry(self):
        from test_p04_service import ServiceTests
        fixture = ServiceTests()
        fixture.setUp()
        service = fixture.service(lambda image, raw: fixture.batch(patterns=(fixture.pattern(complete=c.Fact.CONFIRMED),)))
        service.observe(fixture.request())
        fixture.advance()
        observed = service.observe(fixture.request())
        h = Flow(mapping=True)
        h.clock.advance(1.)
        h.engine.stage = Stage.SELECT
        h.candidates = observed.targets
        frame = h.frame()
        # The actual P04 result is consumed unchanged; no guessed P06 plan is supplied.
        out = h.engine.advance(replace(frame, perception_result=observed, perception_policy=service.policy_report))
        self.assertEqual(out.perception_counts, (1,1,1))
        self.assertNotEqual(out.stage, Stage.GRASP)

    def test_offline_entry_dispatches_real_field_policies(self):
        from treasure_auto.integration.replay import run_replay, validate_replay
        for scenario in ('field_push', 'field_stall', 'field_budget', 'field_regions', 'field_recenter'):
            with self.subTest(scenario=scenario):
                result = run_replay('V1', scenario)
                self.assertEqual(validate_replay(result), ())
                self.assertEqual(result['hardware_outputs'], 0)
                self.assertTrue(result['field_actions'])

    def test_shipped_unmeasured_config_cannot_start_push(self):
        from pathlib import Path
        from test_field_push import regions, plan, obs
        from treasure_auto.mission.field_strategy import FieldStrategyController
        path=Path(__file__).resolve().parents[1]/'config/mission/field_strategy.unmeasured.json'
        controller=FieldStrategyController.from_config(path, regions())
        self.assertEqual(controller.push.start(plan(),obs(),10.).action,'denied')

    def completed_push(self, graspable=1):
        from test_field_push import obs
        from treasure_auto.mission.field_strategy import FieldInput
        h = self.field_flow()
        request = self.push_input(h)
        request = replace(request, target_id='cake',
                          push_plan=replace(request.push_plan, affected_ids=('cake',), distance_m=.02),
                          push_observation=replace(request.push_observation, affected_ids=('cake',)))
        self.assertEqual(h.step(field_input=request).field_decision.action, 'push_step')
        for t, expected in ((110.21, 'stop'), (110.30, 'stop'), (110.51, 'reobserve'),
                            (110.81, 'grasp' if graspable else 'ready')):
            h.clock.advance(t-h.clock.now().seconds)
            feedback = obs(t, affected_ids=('cake',), fixed_position_xy=(3.53,4.),
                           stable_since=110.30, graspable_count=graspable)
            out = h.step(field_input=FieldInput('push_observation','cake','map',push_observation=feedback))
            self.assertEqual(out.field_decision.action, expected, out.issues)
        h.clock.advance(.01)
        return h

    def test_completed_push_cannot_reuse_pre_push_grasp_geometry(self):
        for graspable in (0, 1):
            with self.subTest(graspable=graspable):
                h = self.completed_push(graspable)
                occupancy = h.snapshot.space.occupancies
                out = h.step()
                self.assertIsNone(out.decision.order)
                self.assertNotEqual(out.stage, Stage.GRASP)
                self.assertEqual(h.snapshot.space.occupancies, occupancy)
                self.assertEqual(h.snapshot.space.envelope.base_motion_epoch, 0)

    def test_post_push_grasp_requires_new_epoch_report_and_plan_together(self):
        for missing in ('epoch', 'report', 'plan', 'post_stop_report', 'post_report_plan', None):
            with self.subTest(missing=missing):
                h = self.completed_push()
                if missing != 'epoch':
                    h.bump(space=replace(h.snapshot.space,
                                        envelope=replace(h.snapshot.space.envelope, base_motion_epoch=1)))
                result, report = self.report(h)
                if missing == 'post_stop_report':
                    old = replace(report.envelope, emitted_at=s.stamp(110.))
                    result, report = replace(result, envelope=old), replace(report, envelope=old)
                if missing != 'plan':
                    h.plans = (replace(h.plans[0], envelope=h.env(c.Owner.P06, target_id='cake', action_id=None)),)
                if missing == 'post_report_plan':
                    h.plans = (replace(h.plans[0], envelope=replace(h.plans[0].envelope, emitted_at=s.stamp(110.81))),)
                kwargs = {} if missing == 'report' else dict(perception_result=result, perception_policy=report)
                out = h.step(**kwargs)
                if missing is None:
                    self.assertEqual(out.directive.stage, Stage.GRASP, out.issues)
                else:
                    self.assertIsNone(out.decision.order)

    def handling_input(self, h, phase='probe_start', target='cake', action='probe-1', **changes):
        from treasure_auto.mission.field_strategy import FieldInput
        from treasure_auto.manipulation.strategy import HandlingObservation
        observation = HandlingObservation(target, phase+'-'+str(h.clock.now().seconds), h.clock.now().seconds,
                                          phase, c.Holding.EMPTY, visible=c.Fact.CONFIRMED,
                                          neighbors_stable=c.Fact.CONFIRMED, action_id=action, **changes)
        return FieldInput('handling', target, 'map', handling_observation=observation,
                          exploratory_entity_id='body', observation_id='field-image')

    def start_probe(self, h):
        from treasure_auto.perception.strategy import ExploratoryCandidate
        exploratory = ExploratoryCandidate('body','field-image','cake',m.Category.CAKE,
                                           m.PixelPoint(10,10),100.,True,())
        result, report = self.report(h, exploratory_candidates=(exploratory,))
        out = h.step(field_input=self.handling_input(h), perception_result=result, perception_policy=report)
        self.assertEqual(out.field_decision.action, 'probe', out.issues)

    def test_pending_probe_serializes_explicit_selection_and_other_handling(self):
        for kind in ('select', 'other_probe', 'same_probe'):
            with self.subTest(kind=kind):
                h = self.field_flow()
                self.start_probe(h)
                h.clock.advance(.01)
                request = (self.push_input(h) if kind == 'select' else
                           self.handling_input(h, target='brick-1' if kind == 'other_probe' else 'cake', action='new-probe'))
                # Also cover the controller API independently of P09's optional input path.
                out = h.engine.field_strategy.advance(request, h.snapshot, h.clock.now().seconds)
                self.assertIn(out.action, ('await_observation','stop_reobserve'))
                self.assertEqual(h.engine.field_strategy.push.groups, {})
                self.assertEqual(h.engine.field_strategy.handling.decision_for('cake').probe_attempts, 1)

    def test_explicit_selection_during_pending_probe_never_starts_push(self):
        h = self.field_flow()
        self.start_probe(h)
        h.clock.advance(.01)
        out = h.step(field_input=self.push_input(h))
        self.assertIn(out.field_decision.action, ('await_observation','stop_reobserve'))
        self.assertEqual(h.engine.field_strategy.push.groups, {})
        self.assertIsNone(out.directive)

    def test_pending_probe_accepts_only_matching_result_or_stop_resolution(self):
        h = self.field_flow()
        self.start_probe(h)
        h.clock.advance(.01)
        out = h.step(field_input=self.handling_input(h, 'probe_result', independent_lift=c.Fact.REJECTED))
        self.assertEqual(out.field_decision.action, 'await_observation')
        self.assertEqual(h.engine.field_strategy.handling.pending_targets(), ())
        self.assertEqual(h.engine.field_strategy.handling.decision_for('cake').probe_attempts, 1)

    def test_checkpoint_with_field_state_requires_controller_at_construction(self):
        from treasure_auto.mission.engine import MissionEngine
        from treasure_auto.mission.checkpoint import SimulationJournal
        h = self.field_flow()
        self.start_probe(h)
        for n in range(3):
            h.clock.advance(.01)
            h.step(field_input=self.handling_input(h, 'probe_result', action='probe-'+str(n+1),
                                                 independent_lift=c.Fact.REJECTED))
            if n < 2:
                h.clock.advance(.01)
                h.engine.field_strategy.advance(self.handling_input(h, action='probe-'+str(n+2)),
                                               h.snapshot, h.clock.now().seconds)
        h.step()
        self.assertTrue(h.engine.field_strategy.handling.decision_for('cake').skip_this_run)
        with self.assertRaises(c.ContractError):
            MissionEngine(h.engine.config, h.clock, SimulationJournal(h.journal.load()))

    def test_restored_controller_cannot_be_replaced_by_fresh_controller(self):
        from test_field_push import regions, limits
        from treasure_auto.mission.field_strategy import FieldStrategyController
        from treasure_auto.mission.engine import MissionEngine
        from treasure_auto.mission.checkpoint import SimulationJournal
        h = self.field_flow()
        self.start_probe(h)
        controller = FieldStrategyController(regions(), limits())
        engine = MissionEngine(h.engine.config, h.clock, SimulationJournal(h.journal.load()), field_strategy=controller)
        engine.field_strategy = FieldStrategyController(regions(), limits())
        # Even an otherwise reconciled instance cannot replace its restored budgets.
        engine.stage = Stage.SELECT
        engine.blocked = None
        h.bump()
        out = engine.advance(h.frame())
        self.assertIsNone(out.directive)
        self.assertIsNotNone(out.stop_request)

    def test_invalid_saved_push_budget_rejects_entire_restore(self):
        from test_field_push import regions, limits
        from treasure_auto.mission.field_strategy import FieldStrategyController
        h = self.field_flow()
        h.step(field_input=self.push_input(h))
        state = h.engine.field_strategy.checkpoint(h.clock.now().seconds)
        state['push']['groups']['pile']['attempts'] = 4
        controller = FieldStrategyController(regions(), limits())
        with self.assertRaises(c.ContractError):
            controller.restore(state, h.clock.now().seconds)

    def test_unmeasured_empty_push_restore_preserves_pending_handling(self):
        from test_field_push import regions
        from treasure_auto.navigation.controlled_push import PushLimits
        from treasure_auto.mission.field_strategy import FieldStrategyController
        from treasure_auto.mission.engine import MissionEngine
        from treasure_auto.mission.checkpoint import SimulationJournal
        h = self.field_flow()
        h.engine.field_strategy = FieldStrategyController(regions(), PushLimits())
        self.start_probe(h)
        controller = FieldStrategyController(regions(), PushLimits())
        engine = MissionEngine(h.engine.config,h.clock,SimulationJournal(h.journal.load()),field_strategy=controller)
        self.assertEqual(controller.handling.pending_targets(), ('cake',))
        self.assertEqual(controller.handling.decision_for('cake').probe_attempts, 1)
        self.assertIsNone(engine.advance(h.frame()).field_decision)

    def test_completed_push_geometry_barrier_survives_controller_restore(self):
        from test_field_push import regions, limits
        from treasure_auto.mission.field_strategy import FieldStrategyController
        from treasure_auto.mission.engine import MissionEngine
        from treasure_auto.mission.checkpoint import SimulationJournal
        h = self.completed_push()
        controller = FieldStrategyController(regions(), limits())
        engine = MissionEngine(h.engine.config,h.clock,SimulationJournal(h.journal.load()),field_strategy=controller)
        out = self.resume_engine(h, engine)
        self.assertEqual(out.directive.stage, Stage.NAV_CENTRAL, out.issues)
        h.plans = (replace(h.plans[0], envelope=h.env(c.Owner.P06,target_id='cake',action_id=None)),)
        out = h.finish()
        self.assertIsNone(out.decision.order)
        self.assertNotEqual(out.stage, Stage.GRASP)
        self.assertTrue(any('candidate.geometry' in rejection['reason']
                            for rejection in engine.trace[-1]['candidate_rejections']))

    def resume_engine(self, h, engine):
        from test_p09_support import proof
        h.clock.advance(.01)
        registered = dict(instance_id='sim-instance-2', control_epoch=2)
        h.bump(run=replace(h.snapshot.run, envelope=replace(h.snapshot.run.envelope, **registered),
                           ever_restarted=True),
               space=replace(h.snapshot.space, envelope=replace(h.snapshot.space.envelope, **registered)))
        review = m.ActionEvent(h.env(c.Owner.P10A,target_id=None,action_id=None),
                              c.ActionResult(c.Outcome.SUCCESS,'mission_reconciled',None,
                                             c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,
                                             c.Fact.CONFIRMED,(proof(h.clock.now().seconds),),.01))
        h.engine = engine
        return engine.resume(h.frame(),review)

    def test_valid_restore_and_physical_resume_never_reissue_pending_probe(self):
        from test_field_push import regions, limits
        from treasure_auto.mission.field_strategy import FieldStrategyController
        from treasure_auto.mission.engine import MissionEngine
        from treasure_auto.mission.checkpoint import SimulationJournal
        h = self.field_flow()
        self.start_probe(h)
        controller = FieldStrategyController(regions(),limits())
        engine = MissionEngine(h.engine.config,h.clock,SimulationJournal(h.journal.load()),field_strategy=controller)
        out = self.resume_engine(h,engine)
        self.assertEqual(out.field_decision.action,'stop_reobserve',out.issues)
        self.assertIsNotNone(out.stop_request)
        self.assertIsNone(out.directive)
        self.assertEqual(controller.handling.decision_for('cake').probe_attempts,1)
        h.clock.advance(.01)
        out = h.step(field_input=self.handling_input(h,'resolve_stop',stopped=c.Fact.CONFIRMED))
        self.assertEqual(out.field_decision.action,'await_observation')
        h.clock.advance(.01)
        out = h.step(field_input=self.handling_input(h,'probe_result',independent_lift=c.Fact.REJECTED))
        self.assertEqual(out.field_decision.action,'await_observation')
        self.assertEqual(controller.handling.pending_targets(),())
        self.assertEqual(controller.handling.decision_for('cake').probe_attempts,1)

    def test_saved_probe_skip_survives_valid_controller_and_physical_resume(self):
        from test_field_push import regions, limits
        from treasure_auto.mission.field_strategy import FieldStrategyController
        from treasure_auto.mission.engine import MissionEngine
        from treasure_auto.mission.checkpoint import SimulationJournal
        h = self.field_flow()
        self.start_probe(h)
        for n in range(3):
            h.clock.advance(.01)
            h.step(field_input=self.handling_input(h,'probe_result',action='probe-'+str(n+1),
                                                 independent_lift=c.Fact.REJECTED))
            if n < 2:
                h.clock.advance(.01)
                h.engine.field_strategy.advance(self.handling_input(h,action='probe-'+str(n+2)),
                                               h.snapshot,h.clock.now().seconds)
        controller = FieldStrategyController(regions(),limits())
        engine = MissionEngine(h.engine.config,h.clock,SimulationJournal(h.journal.load()),field_strategy=controller)
        out = self.resume_engine(h,engine)
        self.assertEqual(out.directive.stage,Stage.NAV_CENTRAL,out.issues)
        h.plans = (replace(h.plans[0],envelope=h.env(c.Owner.P06,target_id='cake',action_id=None)),)
        out = h.finish()
        self.assertIsNone(out.decision.order)
        self.assertNotEqual(out.stage,Stage.GRASP)
        self.assertEqual(controller.handling.decision_for('cake').probe_attempts,3)
        self.assertTrue(controller.handling.decision_for('cake').skip_this_run)


if __name__ == '__main__':
    unittest.main()
