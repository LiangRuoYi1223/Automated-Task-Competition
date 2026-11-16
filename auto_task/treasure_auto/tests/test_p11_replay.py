"""P11-T02/T04: observe the real P09 scheduler through explicit fake inputs."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c
from treasure_auto.integration.replay import run_replay, validate_replay


class ReplayTests(unittest.TestCase):
    def test_T02_cake_nine_bricks_once_then_original_600_second_deadline(self):
        for layout, points in [('V1', 21), ('V2', 27), ('V3', 42)]:
            with self.subTest(layout=layout):
                result = run_replay(layout)
                self.assertEqual(validate_replay(result), ())
                self.assertEqual(result['arrivals'], 10)
                self.assertEqual(result['counts'], {'cake': 1, 'banana': 3, 'doll': 3, 'clock': 3})
                self.assertEqual(len(set(result['delivery_keys'])), 10)
                self.assertEqual(result['conditional_points'], points)
                self.assertEqual(result['elapsed_s'], 600)
                self.assertEqual(result['original_deadline_s'], 700)
                self.assertFalse(result['deadline_probe']['issued'])
                self.assertEqual(result['origin'], 'simulated')
                issued = result['issued']
                grabs = [x for x in issued if x['stage'] == 'grasp_and_transport_ready']
                self.assertEqual([g['category'] for g in grabs[:4]], ['cake', 'banana', 'doll', 'clock'])
                first_ordinary = issued.index(grabs[1])
                self.assertIn('confirm_empty_ready', [x['stage'] for x in issued[:first_ordinary]])
                self.assertEqual(result['hardware_outputs'], 0)

    def test_T04_unknown_trial_transport_release_never_dispatch_second_grasp(self):
        for scenario in ('unknown_grasp', 'unknown_transport', 'unknown_release', 'echo_only'):
            with self.subTest(scenario=scenario):
                result = run_replay(scenario=scenario)
                self.assertEqual(validate_replay(result), ())
                self.assertEqual(result['arrivals'], 0)
                self.assertEqual(result['flow_phase'], 'recovery')
                self.assertEqual(result['holding'], 'unknown')
                self.assertEqual(len([x for x in result['issued'] if x['stage'] == 'grasp_and_transport_ready']), 1)
                self.assertGreater(result['stop_requests'], 0)

    def test_dropped_frames_alone_block_carry_and_audit_matches_submitted_events(self):
        result = run_replay(scenario='dropped_frames')
        self.assertEqual(result['holding'], 'held')
        self.assertIn('readiness', ' '.join(x for row in result['trace'] for x in row['issues']))
        self.assertEqual(result['arrivals'], 0)
        self.assertEqual(result['flow_phase'], 'recovery')
        event_ids = [c.loads(value).envelope.event_id for row in result['trace'] for value in row['action_events']]
        self.assertEqual([x['event_id'] for x in result['events']], event_ids)

    def test_conflicting_callback_trace_records_every_returned_directive(self):
        result = run_replay(scenario='conflicting_duplicate')
        dispatched = [row['directive'] for row in result['trace'] if row['directive'] is not None]
        self.assertEqual([x['action_id'] for x in result['issued']], dispatched)
        self.assertEqual(len(dispatched), 2)

    def test_restart_trace_retains_precrash_history(self):
        result = run_replay(scenario='crash_after_release')
        dispatched = [row['directive'] for row in result['trace'] if row['directive'] is not None]
        self.assertEqual([x['action_id'] for x in result['issued']], dispatched)
        self.assertGreater(len(dispatched), 4)

    def test_failed_and_timed_out_callbacks_retain_work_without_retry(self):
        for scenario in ('failed_grasp', 'timed_out_grasp'):
            result = run_replay(scenario=scenario)
            self.assertEqual(validate_replay(result), ())
            self.assertEqual(result['arrivals'], 0)
            self.assertEqual(result['flow_phase'], 'recovery')
            self.assertEqual(sum(x['stage'] == 'grasp_and_transport_ready' for x in result['issued']), 1)

    def test_duplicate_and_old_callbacks_do_not_multiply_dispatch_or_arrivals(self):
        for scenario in ('duplicate', 'late_event', 'old_instance'):
            result = run_replay(scenario=scenario)
            self.assertEqual(validate_replay(result), ())
            self.assertEqual(result['arrivals'], 10)
            self.assertEqual(len(result['issued']), 63)
            self.assertEqual(len({x['action_id'] for x in result['issued']}), 63)
            self.assertTrue(result['injections'])

    def test_conflicting_duplicate_and_unknown_checkpoint_lock_progress(self):
        for scenario in ('conflicting_duplicate', 'checkpoint_unknown'):
            result = run_replay(scenario=scenario)
            self.assertEqual(validate_replay(result), ())
            self.assertEqual(result['arrivals'], 0)
            self.assertEqual(result['flow_phase'], 'recovery')
            self.assertGreater(result['stop_requests'], 0)

    def test_crashes_before_and_after_physical_change_do_not_replay_pending(self):
        for scenario, arrivals in [('crash_before_grasp', 0), ('crash_after_grasp', 0),
                                   ('crash_before_release', 0), ('crash_after_release', 1)]:
            result = run_replay(scenario=scenario)
            self.assertEqual(validate_replay(result), ())
            self.assertEqual(result['arrivals'], arrivals)
            self.assertEqual(result['flow_stage'], 'revalidate_after_restart')
            self.assertFalse(result['restart']['reissued'])
            self.assertEqual(result['restart']['pending_before'], result['restart']['pending_after'])

    def test_restart_subtracts_real_downtime_without_new_600_seconds(self):
        result = run_replay(scenario='restart_410')
        self.assertEqual(validate_replay(result), ())
        self.assertEqual(result['restart']['saved_remaining_s'], 420)
        self.assertEqual(result['restart']['remaining_s'], 410)
        self.assertEqual(result['original_deadline_s'], 700)
        self.assertEqual(result['arrivals'], 0)

    def test_deadline_during_carry_keeps_held_object_and_issues_no_cleanup_action(self):
        result = run_replay(scenario='deadline')
        self.assertEqual(validate_replay(result), ())
        self.assertEqual(result['holding'], 'held')
        self.assertEqual(result['arrivals'], 0)
        self.assertNotIn('release_in_destination', [x['stage'] for x in result['issued']])

    def test_invalid_scenario_layout_and_unmeasured_config_are_rejected(self):
        for kwargs in ({'scenario': 'invented'}, {'layout': 'V4'}, {'scenario': ''}):
            with self.assertRaises((c.ContractError, ValueError)):
                run_replay(**kwargs)
        from treasure_auto.mission.config import MissionConfig
        with self.assertRaises(c.ContractError):
            MissionConfig.load(Path(__file__).resolve().parents[1] / 'config/mission/unmeasured.json')

    def test_result_validation_does_not_pass_without_actual_injection_or_complete_flow(self):
        from copy import deepcopy
        result = run_replay(scenario='duplicate')
        for field, value in [('injections', []), ('arrivals', 9), ('elapsed_s', 0), ('hardware_outputs', 1)]:
            broken = deepcopy(result); broken[field] = value
            self.assertTrue(validate_replay(broken), field)


if __name__ == '__main__':
    unittest.main()
