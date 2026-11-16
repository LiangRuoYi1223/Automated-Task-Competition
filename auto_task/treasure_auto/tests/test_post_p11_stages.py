"""Regressions for real inter-package failures found after P11 delivery."""
import unittest
from dataclasses import replace
from test_p07_support import Harness as GraspHarness
from post_p11_support import (AdvancingClock, PlacementHarness, wire_placement,
                             place_permission, c, m, proof)


class StageIntegrationTests(unittest.TestCase):
    def test_grasp_accepts_normal_monotonic_clock_progress(self):
        h = GraspHarness()
        h.clock = AdvancingClock(h.clock.now())
        h.service.clock = h.clock
        result = h.service.grasp(h.request())
        self.assertEqual(result.result.submitted, c.Fact.CONFIRMED, result.result.reason)
        self.assertEqual(len(h.commands), 1)

    def test_grasp_context_from_another_clock_still_rejected(self):
        h = GraspHarness()
        h.service.context_provider = lambda: replace(h.context,
            now=replace(h.clock.now(), clock_id='foreign-clock'))
        result = h.service.grasp(h.request())
        self.assertEqual(result.result.status, c.Outcome.REJECTED)
        self.assertEqual(result.result.reason.code, c.ErrorCode.CLOCK_MISMATCH)
        self.assertFalse(h.commands)

    def test_placement_receipt_uses_time_after_actual_gateway_send(self):
        h = PlacementHarness()
        h.clock = AdvancingClock(h.clock.now())
        h.executor.clock = h.clock
        gateway, transport = wire_placement(self, h)
        h.executor.start(h.req)
        self.assertEqual(h.executor.state, 'waiting', h.executor.reason)
        self.assertEqual(len(transport.arm), 1)

    def test_second_placement_segment_reaches_actual_gateway(self):
        h = PlacementHarness()
        gateway, transport = wire_placement(self, h)
        self.assertEqual(h.executor.start(h.req), 'waiting')
        first = h.executor.command
        h.clock.advance(.05)
        event = gateway.record_feedback(first.envelope.action_id,
            proof(h.clock.now().seconds, action=first.envelope.action_id))
        self.assertTrue(h.executor.confirm(event))
        h.executor.tick()
        self.assertEqual(len(transport.arm), 2, h.executor.reason)
        self.assertGreater(h.executor.command.envelope.sequence, first.envelope.sequence)
        self.assertEqual(h.executor.command.phase, m.Phase.DESCEND)

    def test_runtime_allows_same_object_held_placement_approach(self):
        runtime, clock, permission = place_permission()
        decision = runtime.request_permission(permission)
        self.assertIsNotNone(decision.permit, decision.reason)

    def test_place_label_cannot_disguise_second_grasp_or_unknown_holding(self):
        for owner, purpose, holding in (
            (c.Owner.P07, 'place', c.Holding.HELD),
            (c.Owner.P07, 'grasp', c.Holding.HELD),
            (c.Owner.P08, 'place', c.Holding.UNKNOWN),
        ):
            with self.subTest(owner=owner, purpose=purpose, holding=holding):
                runtime, clock, permission = place_permission(owner, purpose, holding)
                self.assertIsNone(runtime.request_permission(permission).permit)

    def test_placement_must_bind_the_actually_held_target(self):
        runtime, clock, permission = place_permission()
        permission = replace(permission, readiness=replace(permission.readiness,
            holding=replace(permission.readiness.holding, target_id='different-brick')))
        self.assertIsNone(runtime.request_permission(permission).permit)


if __name__ == '__main__':
    unittest.main()
