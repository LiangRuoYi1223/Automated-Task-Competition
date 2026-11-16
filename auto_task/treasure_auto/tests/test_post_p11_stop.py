"""Real P10A revocation and P01 cancellation; physical I/O remains simulated."""
import unittest
from dataclasses import replace
from contract_fixtures import samples as s
from post_p11_support import PlacementHarness, wire_placement, c, m


def runtime_for(clock):
    from test_p10a_support import config_data, signal, initial_snapshot, Store
    from treasure_auto.runtime import RuntimeConfig, RuntimeController
    ids = iter(['sim-run-1'] + ['stop-test-' + str(i) for i in range(200)])
    rt = RuntimeController(RuntimeConfig.from_dict(config_data()), clock, s.versions(),
        'sim-instance-1', 1, id_factory=lambda: next(ids))
    rt.arm(); rt.signal(signal(rt, clock)); rt.initialize(Store(), initial_snapshot(rt.run))
    rt.heartbeat(replace(rt.run.envelope, producer=c.Owner.P09))
    return rt


class StopIntegrationTests(unittest.TestCase):
    def active(self):
        h = PlacementHarness()
        gateway, transport = wire_placement(self, h)
        runtime = runtime_for(h.clock)
        from treasure_auto.integration.stop_router import StopRouter
        router = StopRouter(runtime, gateway)
        h.executor.stop_sink = router
        self.assertEqual(h.executor.start(h.req), 'waiting')
        self.assertTrue(gateway._active)
        return h, runtime, gateway, transport, router

    def domain(self, h, owner=c.Owner.P09, **changes):
        now = h.clock.now()
        envelope = replace(h.req.envelope, producer=owner, event_id='domain-stop',
            emitted_at=now, expires_at=replace(now, seconds=now.seconds+1), **changes)
        return m.StopRequest(envelope, c.Reason(c.ErrorCode.CANCELLED, 'test', 'stop requested'), h.req.holding)

    def test_placement_cancel_reaches_gateway_without_claiming_physical_stop(self):
        h, rt, gateway, transport, router = self.active()
        h.executor.cancel('integration cancellation')
        self.assertFalse(gateway._active)
        self.assertIn('cancel_requested', transport.stops)
        self.assertIsNotNone(rt.authority.reason)
        self.assertEqual(rt.stop_requests[-1].envelope.producer, c.Owner.P10A)
        self.assertEqual(h.executor.stop_report, rt.stop_report)
        self.assertEqual(rt.stop_report.submitted, c.Fact.CONFIRMED)
        self.assertEqual(rt.stop_report.physically_stopped, c.Fact.UNKNOWN)
        self.assertEqual(rt.stop_report.old_queue_empty, c.Fact.UNKNOWN)
        self.assertNotIn(False, transport.pump)

    def test_expired_root_action_can_issue_fresh_stop(self):
        h, rt, gateway, transport, router = self.active()
        h.clock.advance(200)
        h.executor.cancel('root request expired')
        self.assertFalse(gateway._active)
        self.assertIsNotNone(h.executor.stop_report)

    def test_failed_cancel_preserves_unknown_and_revokes_runtime_permission_first(self):
        h, rt, gateway, transport, router = self.active()
        from test_p10a_support import request
        req = request(rt, h.clock)
        permit = rt.request_permission(req).permit
        self.assertIsNotNone(permit)
        self.assertTrue(rt.authorize_output(permit, req.envelope, req.source, real_output=False))
        transport.fail = 'cancel'
        report = router(self.domain(h))
        self.assertFalse(rt.authorize_output(permit, req.envelope, req.source, real_output=False))
        self.assertEqual(report.submitted, c.Fact.UNKNOWN)
        self.assertEqual(report.physically_stopped, c.Fact.UNKNOWN)

    def test_foreign_controller_or_unowned_domain_cannot_stop_current_run(self):
        h, rt, gateway, transport, router = self.active()
        for owner, changes in [(c.Owner.P04, {}), (c.Owner.P09, {'instance_id':'old'}),
                               (c.Owner.P07, {'control_epoch':2}), (c.Owner.P09, {'run_id':'old'})]:
            with self.subTest(owner=owner, changes=changes), self.assertRaises(c.ContractError):
                router(self.domain(h, owner, **changes))
        self.assertTrue(gateway._active)
        self.assertFalse(transport.stops)

    def test_current_controller_can_stop_with_obsolete_geometry_version(self):
        h, rt, gateway, transport, router = self.active()
        request = self.domain(h, versions=replace(h.req.envelope.versions, space=77))
        router(request)
        self.assertFalse(gateway._active)
        self.assertEqual(rt.stop_requests[-1].envelope.versions, rt.run.envelope.versions)

    def test_new_revocation_with_same_reason_gets_current_stop_identity(self):
        h, rt, gateway, transport, router = self.active()
        request = self.domain(h)
        first = rt.request_domain_stop(request).stop
        again = replace(request, envelope=replace(request.envelope, event_id='new-stop',
                                                 sequence=request.envelope.sequence+1))
        second = rt.request_domain_stop(again).stop
        self.assertNotEqual(first.envelope.event_id, second.envelope.event_id)
        self.assertEqual(rt.request_domain_stop(again).stop, second)

    def test_grasp_cancel_reaches_actual_gateway(self):
        from test_p07_support import Harness
        from treasure_auto.integration.stop_router import StopRouter
        grasp = Harness()
        placement = PlacementHarness(); placement.clock = grasp.clock
        gateway, transport = wire_placement(self, placement)
        gateway.config = replace(gateway.config, arm_frame_id='sim-arm')
        rt = runtime_for(grasp.clock)
        grasp.service.stop_sink = StopRouter(rt, gateway)
        grasp.service.device = gateway
        def prepare(command, intent, receipt):
            gateway.register_intent(command, intent, receipt)
            gateway.install_permit(command.permit)
        grasp.service.prepare = prepare
        result = grasp.service.grasp(grasp.request())
        self.assertEqual(result.result.submitted, c.Fact.CONFIRMED, result.result.reason)
        request = m.StopRequest(grasp.env(c.Owner.P09),
            c.Reason(c.ErrorCode.CANCELLED, 'test', 'cancel grasp'), grasp.service.holding)
        grasp.service.cancel(request)
        self.assertFalse(gateway._active)
        self.assertIn('cancel_requested', transport.stops)
        self.assertEqual(grasp.service.stop_report, rt.stop_report)

    def test_mission_anomaly_routes_stop_before_checkpoint(self):
        from test_p09_engine import Flow
        h, rt, gateway, transport, router = self.active()
        mission = Flow()
        mission.engine.stop_sink = router
        order = []
        save = mission.journal.save
        def record(payload):
            order.append(bool(rt.stop_report) and not gateway._active)
            return save(payload)
        mission.journal.save = record
        out = mission.step(affected_targets=('fallen',))
        self.assertIsNotNone(out.stop_request)
        self.assertTrue(order[0])
        self.assertIsNotNone(rt.stop_report)


if __name__ == '__main__':
    unittest.main()
