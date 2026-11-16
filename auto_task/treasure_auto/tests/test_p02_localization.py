"""P02-T02/T04/T05/T06: continuous evidence, invalidation and restart."""
import math
import unittest
from dataclasses import replace
from test_p02_support import c, m, env, proof, pose, policy, transforms, Clock

try:
    from treasure_auto.mapping import localization as loc
except ImportError:
    loc = None


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(loc, 'P02 localization implementation missing')
        self.clock = Clock()
        self.tracker = loc.LocalizationTracker(env(), policy(), self.clock, 'base_link')

    def observe(self, t, **kw):
        self.clock.t = t
        return self.tracker.observe(*transforms(t, **kw))

    def ready(self):
        self.tracker.initialize(pose())
        for t in (10, 10.2, 10.4):
            self.observe(t)
        self.assertEqual(self.tracker.snapshot().trusted, c.Fact.CONFIRMED)

    def test_T02_seed_alone_never_confirms(self):
        self.tracker.initialize(pose())
        self.assertEqual(self.tracker.state, loc.LocalizationState.CONFIRMING)
        self.assertEqual(self.tracker.snapshot().trusted, c.Fact.UNKNOWN)
        self.assertIsNone(self.tracker.snapshot().pose)

    def test_T02_three_distinct_continuous_observations_confirm(self):
        self.tracker.initialize(pose())
        self.observe(10)
        self.observe(10.2)
        self.assertEqual(self.tracker.snapshot().trusted, c.Fact.UNKNOWN)
        self.observe(10.4)
        self.assertEqual(self.tracker.snapshot().pose, pose())

    def test_T02_duplicate_events_do_not_fill_window(self):
        self.tracker.initialize(pose())
        for _ in range(10):
            self.observe(10)
        self.assertEqual(self.tracker.snapshot().trusted, c.Fact.UNKNOWN)
        self.assertEqual(len(self.tracker.samples), 1)

    def test_T02_duplicate_with_changed_payload_revokes(self):
        self.ready()
        epoch = self.tracker.base_motion_epoch
        self.observe(10.4, x=2.1)
        self.assertEqual(self.tracker.snapshot().reason.code, c.ErrorCode.ID_CONFLICT)
        self.assertGreater(self.tracker.base_motion_epoch, epoch)

    def test_T02_old_capture_under_new_event_is_not_new_evidence(self):
        self.ready()
        a, b = transforms(10.5)
        a = replace(a, evidence=replace(a.evidence, captured_at=proof(10.4).captured_at))
        b = replace(b, evidence=replace(b.evidence, captured_at=proof(10.4).captured_at))
        self.clock.t = 10.5
        result = self.tracker.observe(a, b)
        self.assertEqual(result.trusted, c.Fact.UNKNOWN)

    def test_T02_wrong_start_or_heading_never_confirms(self):
        for changes in (dict(x=3), dict(yaw=math.pi)):
            with self.subTest(changes=changes):
                self.tracker.initialize(pose())
                self.assertEqual(self.observe(self.clock.t + .1, **changes).trusted, c.Fact.UNKNOWN)

    def test_T02_gap_and_high_uncertainty_revoke(self):
        self.ready()
        self.observe(11)
        self.assertEqual(self.tracker.snapshot().trusted, c.Fact.UNKNOWN)
        a, b = transforms(11.2)
        a = replace(a, pose=replace(a.pose, error_m=.2))
        self.clock.t = 11.2
        self.assertEqual(self.tracker.observe(a, b).trusted, c.Fact.UNKNOWN)

    def test_T05_map_correction_jump_invalidates_old_geometry(self):
        self.ready()
        epoch = self.tracker.base_motion_epoch
        result = self.observe(10.6, correction=pose(.6, 0))
        self.assertEqual(result.trusted, c.Fact.UNKNOWN)
        self.assertGreater(result.base_motion_epoch, epoch)
        self.assertTrue(self.tracker.take_events())

    def test_T05_odometry_motion_updates_epoch_without_false_localization_jump(self):
        self.ready()
        epoch = self.tracker.base_motion_epoch
        result = self.observe(10.6, x=3)
        self.assertEqual(result.trusted, c.Fact.CONFIRMED)
        self.assertGreater(result.base_motion_epoch, epoch)
        before = self.tracker.base_motion_epoch
        self.observe(10.6, x=3)
        self.assertEqual(self.tracker.base_motion_epoch, before)

    def test_T06_restart_history_never_restores_current_or_exit(self):
        self.ready()
        old = self.tracker.snapshot()
        self.tracker.restore_history(old)
        result = self.tracker.snapshot()
        self.assertEqual(self.tracker.historical_pose, pose())
        self.assertIsNone(result.pose)
        self.assertEqual(result.trusted, c.Fact.UNKNOWN)
        self.assertGreater(result.base_motion_epoch, old.base_motion_epoch)

    def test_T06_fresh_data_after_restore_still_requires_initialization(self):
        self.ready()
        self.tracker.restore_history(self.tracker.snapshot())
        self.observe(10.6)
        self.assertEqual(self.tracker.snapshot().trusted, c.Fact.UNKNOWN)

    def test_T05_expiry_and_reinitialization_revoke(self):
        self.ready()
        self.clock.t = 12
        self.assertEqual(self.tracker.snapshot().trusted, c.Fact.UNKNOWN)
        self.tracker.initialize(pose())
        self.assertEqual(self.tracker.state, loc.LocalizationState.RELOCALIZING)
        self.assertEqual(self.tracker.snapshot().trusted, c.Fact.UNKNOWN)

    def test_R03_rotated_chain_and_uncertainty(self):
        a, b = transforms(correction=pose(1, 2, math.pi/2), x=1, y=0, yaw=math.pi/2)
        a = replace(a, pose=replace(a.pose, error_m=.01, error_rad=.02))
        result = loc.resolve_chain(a, b, env(), self.clock(), policy(), 'base_link')
        self.assertIsNone(result.reason)
        self.assertAlmostEqual(result.pose.x, 1)
        self.assertAlmostEqual(result.pose.y, 3)
        self.assertAlmostEqual(abs(result.pose.yaw), math.pi)
        self.assertGreater(result.pose.error_m, .029)
        self.assertEqual(len(result.evidence), 2)

    def test_R03_wrong_frames_versions_clocks_echo_and_provenance_rejected(self):
        a, b = transforms()
        bad = (
            replace(a, child_frame='wrong'),
            replace(a, envelope=replace(a.envelope, versions=replace(a.envelope.versions,map='old'))),
            replace(a, evidence=replace(a.evidence, captured_at=c.ClockStamp(10,c.ClockBasis.ROS,'other','sim-boot'))),
            replace(a, evidence=replace(a.evidence, kind=c.EvidenceKind.ECHO)),
            replace(a, evidence=replace(a.evidence, origin=c.Origin.REPLAY)),
            replace(a, envelope=replace(a.envelope, instance_id='old')),
        )
        for wrong in bad:
            with self.subTest(wrong=wrong):
                result = loc.resolve_chain(wrong, b, env(), self.clock(), policy(), 'base_link')
                self.assertIsNone(result.pose)
                self.assertIsNotNone(result.reason)

    def test_R03_missing_expired_and_unsynchronized_transform(self):
        a, b = transforms()
        for first, second, now in ((None,b,10), (a,b,12), (a,transforms(10.1)[1],10.1)):
            with self.subTest(now=now):
                result = loc.resolve_chain(first, second, env(), c.ClockStamp(now,c.ClockBasis.MONOTONIC,'sim-clock','sim-boot'), policy(), 'base_link')
                self.assertIsNone(result.pose)
                self.assertIsNotNone(result.reason)

    def test_R04_live_tracker_rejects_unmeasured_or_simulated_policy(self):
        for measurement in (m.Measurement.UNMEASURED, m.Measurement.SIMULATED):
            with self.subTest(measurement=measurement), self.assertRaises(c.ContractError):
                loc.LocalizationTracker(replace(env(),origin=c.Origin.LIVE),
                                        replace(policy(),measurement=measurement), self.clock, 'base_link')

    def test_R04_invalid_policy_values_rejected(self):
        for values in (dict(min_samples=1),dict(max_gap_s=0),dict(max_age_s=float('nan')),
                       dict(jump_m=True),dict(confirmation_s=0)):
            with self.subTest(values=values), self.assertRaises(c.ContractError):
                replace(policy(),**values)
