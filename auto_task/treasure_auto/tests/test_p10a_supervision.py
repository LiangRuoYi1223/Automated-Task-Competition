import threading
import unittest
from dataclasses import replace
from test_p10a_support import *


class SupervisionTests(unittest.TestCase):
    def test_T02_blocked_business_and_store_do_not_block_deadline_stop(self):
        from treasure_auto.runtime import IndependentSupervisor
        rt, clock = started()
        store = Store(block=True)
        gateway = ShadowGateway(rt, clock)
        req = request(rt, clock)
        permit = rt.request_permission(req).permit
        self.assertTrue(gateway.send(permit, req.envelope, req.source, .1))
        errors = []
        def initialize():
            try: store.initialize_run(m.InitializeRunRequest(rt.run,initial_snapshot(rt.run),signal(rt,clock),None))
            except Exception as error: errors.append(error)
        writer = threading.Thread(target=initialize)
        business_release = threading.Event()
        business = threading.Thread(target=lambda: business_release.wait(3))
        supervisor = IndependentSupervisor(rt, gateway.stop)
        writer.start(); business.start()
        try:
            self.assertTrue(store.entered.wait(1))
            supervisor.start()
            clock.advance(600)
            self.assertTrue(gateway.stopped.wait(1), 'watchdog waited for store/business')
            self.assertTrue(writer.is_alive())
            self.assertTrue(business.is_alive())
            self.assertEqual(gateway.output, 0)
            self.assertEqual(gateway.stops[-1].reason.code, c.ErrorCode.DEADLINE)
        finally:
            store.release.set(); business_release.set()
            writer.join(1); business.join(1); supervisor.close()
        self.assertFalse(errors)
        self.assertIsNone(rt.request_permission(request(rt, clock, seq=20)).permit)

    def test_T02_late_velocity_after_deadline_cannot_restart_shadow_output(self):
        rt, clock = started()
        req = request(rt, clock)
        permit = rt.request_permission(req).permit
        gateway = ShadowGateway(rt, clock)
        self.assertTrue(gateway.send(permit, req.envelope, req.source, 0.1))
        clock.advance(600)
        self.assertFalse(gateway.send(permit, req.envelope, req.source, 0.1))
        self.assertEqual(gateway.output, 0)

    def test_T03_lease_expiration_and_heartbeat_loss_are_latched(self):
        for elapsed, expected in ((1.0, c.ErrorCode.EXPIRED), (2.0, c.ErrorCode.TIMEOUT)):
            rt, clock = started()
            req = request(rt, clock); permit = rt.request_permission(req).permit
            clock.advance(elapsed)
            decision = rt.supervise(clock.now(), req.health)
            self.assertEqual(decision.reason.code, expected)
            rt.heartbeat(replace(req.envelope, producer=c.Owner.P09, sequence=100,
                                 emitted_at=clock.now(), expires_at=s.stamp(120)))
            self.assertFalse(rt.authorize_output(permit, req.envelope, req.source, real_output=False))
            self.assertIsNone(rt.request_permission(request(rt, clock, seq=20)).permit)

    def test_T03_old_instance_wrong_source_and_unissued_permit_denied(self):
        rt, clock = started()
        req = request(rt, clock); permit = rt.request_permission(req).permit
        for p, e, source in ((permit, replace(req.envelope, instance_id='old'), req.source),
                              (permit, req.envelope, m.MotionSource.DEBUG),
                              (replace(permit, permit_id='invented'), req.envelope, req.source)):
            self.assertFalse(rt.authorize_output(p,e,source,real_output=False))

    def test_T03_concurrent_motion_sources_are_mutually_exclusive(self):
        rt, clock = started()
        nav = request(rt, clock); first = rt.request_permission(nav).permit
        precision = request(rt, clock, m.MotionSource.PRECISION, seq=20)
        self.assertIsNone(rt.request_permission(precision).permit)
        self.assertTrue(rt.authorize_output(first, nav.envelope, nav.source, real_output=False))

    def test_T04_held_stop_preserves_holding_and_unknown_physical_stop(self):
        rt, clock = started()
        req = request(rt, clock)
        held = c.HoldingSnapshot(c.Holding.HELD, 'brick-1', None,
                               (replace(req.readiness.holding.evidence[0], target_id='brick-1'),))
        ready = replace(req.readiness, holding=held)
        rt.observe_phase(m.Phase.TRANSPORT, ready)
        decision = rt.signal(signal(rt, clock, m.SignalKind.EMERGENCY_STOP, 'signal-2'))
        self.assertEqual(decision.stop.holding.state, c.Holding.HELD)
        self.assertIn('transport', decision.stop.reason.detail)
        self.assertIsNone(rt.stop_report)
        echo = replace(req.old_stop, physically_stopped=c.Fact.UNKNOWN, old_queue_empty=c.Fact.UNKNOWN)
        rt.record_stop_report(echo)
        self.assertEqual(rt.stop_report.physically_stopped, c.Fact.UNKNOWN)
        for index, source in enumerate(m.MotionSource):
            self.assertIsNone(rt.request_permission(request(rt, clock, source, seq=30+index)).permit)

    def test_health_recovery_does_not_automatically_clear_stop(self):
        rt, clock = started()
        req = request(rt, clock); rt.request_permission(req)
        bad = replace(req.health, capabilities=(replace(req.health.capabilities[0], status=c.Fact.UNKNOWN),))
        rt.supervise(clock.now(), bad)
        rt.supervise(clock.now(), req.health)
        self.assertIsNone(rt.request_permission(req).permit)
        clock.advance(.01)
        self.assertIsNotNone(rt.resume(request(rt, clock, seq=20)).permit)

    def test_storage_fault_notification_revokes_without_calling_store(self):
        rt,clock=started();req=request(rt,clock);permit=rt.request_permission(req).permit
        rt.notify_storage_fault(c.Reason(c.ErrorCode.STORAGE_FAILURE,'queue','simulated queue failure'))
        self.assertFalse(rt.authorize_output(permit,req.envelope,req.source,real_output=False))
        clock.advance(.01)
        self.assertIsNone(rt.resume(request(rt,clock,seq=20)).permit)
        clock.advance(600)
        self.assertIsNotNone(rt.supervise(clock.now(),None).stop)


if __name__ == '__main__': unittest.main()
