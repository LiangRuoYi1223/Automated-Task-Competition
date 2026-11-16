"""P01-T03..T06. Boundary fakes only; authorization/dispatch/evidence logic is real."""
import dataclasses
import io
from pathlib import Path
import sys
import threading
import unittest
from uuid import uuid4

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from contract_fixtures import samples as s
from contract_fixtures.fakes import FakeClock
from treasure_auto.contracts import core as c, models as m
try:
    from treasure_auto.devices.gateway import DeviceGateway, SafetyWorker
    from treasure_auto.devices.events import EventOutbox
    from treasure_auto.devices.config import ControlConfig
    from treasure_auto.devices.drivers import SerialLease
except ImportError:
    DeviceGateway = SafetyWorker = EventOutbox = ControlConfig = SerialLease = None


class BoundaryTransport:
    origin = c.Origin.SIMULATED

    def __init__(self):
        self.base, self.arm, self.pump, self.stops = [], [], [], []
        self.fail = None

    def send_base(self, linear, angular):
        if self.fail == 'base':
            raise OSError('simulated unknown base send result')
        self.base.append((linear, angular))

    def send_arm(self, dispatch):
        if self.fail == 'arm':
            raise OSError('simulated unknown arm send result')
        self.arm.append(dispatch)

    def send_pump(self, enabled):
        self.pump.append(enabled)

    def cancel_arm(self):
        self.stops.append('cancel_requested')
        if self.fail == 'cancel':
            raise OSError('cancel unavailable')


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(DeviceGateway, 'P01 final gateway required')
        self.clock = FakeClock(s.stamp(111))
        self.transport = BoundaryTransport()
        self.config = ControlConfig('sim-config-v1', .5, 1., 'sim-arm',
                                    ((-200., 200.), (-200., 200.), (0., 200.)),
                                    tuple((phase, (.5, 10., 5.)) for phase in m.Phase))
        self.outbox = EventOutbox(1000)
        self.lease = SerialLease('sim://' + str(uuid4())).acquire()
        self.addCleanup(self.lease.release)
        self.health_status = {'base': c.Fact.CONFIRMED, 'arm': c.Fact.CONFIRMED, 'pump': c.Fact.CONFIRMED}
        self.gateway = DeviceGateway(self.clock, s.run_context(), self.config, self.transport,
                                     self.health_snapshot, self.outbox, self.lease)

    def health_snapshot(self):
        return m.HealthSnapshot(s.envelope(c.Owner.P01, emitted_at=self.clock.now(),
                                           expires_at=dataclasses.replace(self.clock.now(), seconds=self.clock.now().seconds + .5)),
                                tuple(m.Capability(k, v, None, ()) for k, v in self.health_status.items()))

    def base(self, source=m.MotionSource.NAVIGATION, sequence=1, **changes):
        env = s.envelope(c.Owner.P03, sequence=sequence, event_id='command-' + str(sequence),
                         action_id='action-' + str(sequence))
        permit = m.ControlPermit(dataclasses.replace(env, producer=c.Owner.P10A, action_id=None), 'permit-' + str(sequence),
                                  source, source.value, ())
        values = dict(envelope=env, permit=permit, intent_receipt_id='receipt-' + str(sequence),
                      linear_m_s=.2, angular_rad_s=.1, source=source)
        values.update(changes)
        return m.BaseCommand(**values)

    def intent(self, command):
        if isinstance(command, m.BaseCommand):
            return m.ActionIntent(command.envelope, m.IntentOperation.BASE_VELOCITY, None, None, None,
                                  (command.linear_m_s, command.angular_rad_s), None)
        if isinstance(command, m.ArmCommand):
            plan = m.GeometryPlan(dataclasses.replace(command.envelope, producer=c.Owner.P06), 'sim-path', command.target,
                                  (m.PathSegment(command.phase, command.target, command.speed_mm_s, command.timeout_s),),
                                  c.Fact.CONFIRMED, c.Fact.CONFIRMED, ())
            return m.ActionIntent(command.envelope, m.IntentOperation.ARM, None, plan, None, None, None)
        return m.ActionIntent(command.envelope, m.IntentOperation.PUMP, None, None, None, None, command.enabled)

    def register(self, command, intent=None):
        intent = intent or self.intent(command)
        receipt = s.intent_receipt(intent)
        receipt = dataclasses.replace(receipt, commit=dataclasses.replace(receipt.commit, transaction_id=command.intent_receipt_id))
        self.gateway.register_intent(command, intent, receipt)
        return receipt

    def prepare(self, command):
        self.register(command)
        self.gateway.install_permit(command.permit)

    def arm(self, sequence=1, speed=2., x=100.):
        cmd = s.arm_command()
        env = dataclasses.replace(cmd.envelope, sequence=sequence, event_id='arm-' + str(sequence), action_id='arm-action-' + str(sequence))
        permit = m.ControlPermit(dataclasses.replace(env, producer=c.Owner.P10A), 'arm-permit-' + str(sequence),
                                  m.MotionSource.ARM, m.Phase.TRIAL_LIFT.value, ())
        return dataclasses.replace(cmd, envelope=env, permit=permit, speed_mm_s=speed,
                                   target=dataclasses.replace(cmd.target, x=x), intent_receipt_id='arm-receipt-' + str(sequence))

    def stop_request(self):
        return m.StopRequest(s.envelope(c.Owner.P10A), c.Reason(c.ErrorCode.CANCELLED, 'stop', 'sim stop'),
                             c.HoldingSnapshot(c.Holding.UNKNOWN, None, None, ()))

    def test_T03_revoked_buffer_and_old_instance_cannot_resume(self):
        cmd = self.base(); self.prepare(cmd)
        self.assertEqual(self.gateway.submit_base(cmd).result.submitted, c.Fact.CONFIRMED)
        self.gateway.revoke(m.MotionSource.NAVIGATION)
        self.assertEqual(self.gateway.submit_base(cmd).result.accepted, c.Fact.REJECTED)
        stale = dataclasses.replace(cmd, envelope=dataclasses.replace(cmd.envelope, instance_id='old'))
        self.assertEqual(self.gateway.submit_base(stale).result.accepted, c.Fact.REJECTED)
        self.assertEqual(self.transport.base, [(.2, .1), (0., 0.)])

    def test_T04_competing_sources_require_explicit_revocation(self):
        cmd = self.base(); self.prepare(cmd); self.gateway.submit_base(cmd)
        other = self.base(m.MotionSource.PRECISION, 2); self.register(other)
        with self.assertRaises(c.ContractError):
            self.gateway.install_permit(other.permit)
        self.assertEqual(self.gateway.submit_base(other).result.accepted, c.Fact.REJECTED)
        self.assertEqual(self.transport.base[-1], (.2, .1))

    def test_T04_expired_permit_stops_without_new_business_command(self):
        cmd = self.base(); self.prepare(cmd); self.gateway.submit_base(cmd)
        self.clock.advance(9.)
        self.gateway.tick()
        self.assertEqual(self.transport.base[-1], (0., 0.))
        self.assertEqual(self.gateway.submit_base(cmd).result.accepted, c.Fact.REJECTED)

    def test_T03_uninstalled_permit_or_missing_intent_never_outputs(self):
        cmd = self.base()
        self.assertEqual(self.gateway.submit_base(cmd).result.accepted, c.Fact.REJECTED)
        self.gateway.install_permit(cmd.permit)
        self.assertEqual(self.gateway.submit_base(cmd).result.accepted, c.Fact.REJECTED)
        self.assertEqual(self.transport.base, [])

    def test_T03_duplicate_is_idempotent_and_conflict_rejected(self):
        cmd = self.base(); self.prepare(cmd)
        first = self.gateway.submit_base(cmd)
        self.assertEqual(self.gateway.submit_base(cmd), first)
        changed = dataclasses.replace(cmd, linear_m_s=.3)
        self.assertEqual(self.gateway.submit_base(changed).result.accepted, c.Fact.REJECTED)
        self.assertEqual(self.transport.base, [(.2, .1)])

    def test_T03_old_sequence_is_rejected_even_with_new_action_id(self):
        cmd = self.base(sequence=3); self.prepare(cmd); self.gateway.submit_base(cmd)
        old = self.base(sequence=2)
        old = dataclasses.replace(old, permit=cmd.permit)
        self.register(old)
        self.assertEqual(self.gateway.submit_base(old).result.accepted, c.Fact.REJECTED)
        self.assertEqual(len(self.transport.base), 1)

    def test_T03_future_wrong_version_and_forged_producer_reject(self):
        cmd = self.base(); self.prepare(cmd)
        for env in (dataclasses.replace(cmd.envelope, emitted_at=s.stamp(112)),
                    dataclasses.replace(cmd.envelope, versions=dataclasses.replace(cmd.envelope.versions, config='old')),
                    dataclasses.replace(cmd.envelope, producer=c.Owner.P04)):
            with self.subTest(env=env):
                self.assertEqual(self.gateway.submit_base(dataclasses.replace(cmd, envelope=env)).result.accepted, c.Fact.REJECTED)
        self.assertEqual(self.transport.base, [])

    def test_T03_intent_must_bind_exact_payload_and_committed_receipt(self):
        cmd = self.base(); intent = self.intent(cmd)
        with self.assertRaises(c.ContractError):
            self.register(cmd, dataclasses.replace(intent, base_velocity_m_s_rad_s=(.3, .1)))
        receipt = dataclasses.replace(s.intent_receipt(intent), commit=m.CommitReceipt('receipt-1', m.CommitStatus.UNKNOWN,
                          None, None, c.Reason(c.ErrorCode.STORAGE_FAILURE, 'commit', 'unknown')))
        with self.assertRaises(c.ContractError):
            self.gateway.register_intent(cmd, intent, receipt)
        self.gateway.install_permit(cmd.permit)
        self.assertEqual(self.gateway.submit_base(cmd).result.submitted, c.Fact.REJECTED)

    def test_T05_position_stage_speed_and_timeout_remain_bound(self):
        one = self.arm(speed=2., x=100.); self.prepare(one); self.gateway.submit_arm(one)
        self.gateway.revoke(m.MotionSource.ARM)
        two = self.arm(sequence=2, speed=3., x=150.); self.prepare(two); self.gateway.submit_arm(two)
        self.assertEqual([(d.x_mm, d.feed_mm_min, d.timeout_s, d.phase) for d in self.transport.arm],
                         [(100., 120., 2., m.Phase.TRIAL_LIFT), (150., 180., 2., m.Phase.TRIAL_LIFT)])

    def test_T05_missing_speed_outside_bounds_or_unmeasured_limits_reject(self):
        with self.assertRaises(c.ContractError):
            dataclasses.replace(self.arm(), speed_mm_s=None)
        for cmd in (self.arm(sequence=1, speed=11.), self.arm(sequence=2, x=201.),
                    dataclasses.replace(self.arm(sequence=3), timeout_s=6.),
                    dataclasses.replace(self.arm(sequence=4), target=dataclasses.replace(self.arm().target, frame_id='wrong'))):
            with self.subTest(cmd=cmd):
                self.prepare(cmd)
                self.assertEqual(self.gateway.submit_arm(cmd).result.accepted, c.Fact.REJECTED)
                self.gateway.revoke(m.MotionSource.ARM)
        self.assertEqual(self.transport.arm, [])

    def test_T06_echo_never_proves_arrival_or_grip(self):
        cmd = self.arm(); self.prepare(cmd); self.gateway.submit_arm(cmd)
        ev = s.evidence(kind=c.EvidenceKind.ECHO, action_id=cmd.envelope.action_id, received_at=self.clock.now())
        result = self.gateway.record_feedback(cmd.envelope.action_id, ev)
        self.assertEqual(result.result.echoed, c.Fact.CONFIRMED)
        self.assertEqual(result.result.physical, c.Fact.UNKNOWN)
        self.assertEqual(result.result.status, c.Outcome.UNKNOWN)
        self.assertEqual(len(self.transport.arm), 1)

    def test_T06_physical_ref_needs_domain_validator_and_fresh_action(self):
        cmd = self.arm(); self.prepare(cmd); self.gateway.submit_arm(cmd)
        ev = s.evidence(action_id=cmd.envelope.action_id, received_at=self.clock.now())
        self.assertEqual(self.gateway.record_feedback(cmd.envelope.action_id, ev).result.physical, c.Fact.UNKNOWN)
        self.gateway.revoke(m.MotionSource.ARM)
        with self.assertRaises(c.ContractError):
            self.gateway.record_feedback(cmd.envelope.action_id, ev)

    def test_T06_health_recovery_does_not_reinstall_old_permit(self):
        cmd = self.arm(); self.prepare(cmd); self.gateway.submit_arm(cmd)
        self.health_status['arm'] = c.Fact.REJECTED; self.gateway.tick()
        self.health_status['arm'] = c.Fact.CONFIRMED; self.gateway.tick()
        self.assertEqual(self.gateway.submit_arm(cmd).result.accepted, c.Fact.REJECTED)
        self.assertEqual(len(self.transport.arm), 1)
        with self.assertRaises(c.ContractError):
            self.gateway.install_permit(cmd.permit)

    def test_T06_stop_reports_submission_not_physical_and_never_toggles_pump(self):
        cmd = self.arm(); self.prepare(cmd); self.gateway.submit_arm(cmd)
        report = self.gateway.stop(self.stop_request())
        self.assertEqual(report.submitted, c.Fact.CONFIRMED)
        self.assertEqual(report.physically_stopped, c.Fact.UNKNOWN)
        self.assertEqual(report.old_queue_empty, c.Fact.UNKNOWN)
        self.assertEqual(self.transport.pump, [])
        self.assertEqual(self.transport.base[-1], (0., 0.))

    def test_T06_send_error_is_unknown_and_same_action_not_retried(self):
        cmd = self.arm(); self.prepare(cmd); self.transport.fail = 'arm'
        result = self.gateway.submit_arm(cmd)
        self.assertEqual(result.result.status, c.Outcome.UNKNOWN)
        self.assertEqual(result.result.submitted, c.Fact.UNKNOWN)
        self.transport.fail = None
        self.gateway.submit_arm(cmd)
        self.assertEqual(self.transport.arm, [])

    def test_T06_arm_timeout_does_not_infer_physical_failure(self):
        cmd = self.arm(); self.prepare(cmd); self.gateway.submit_arm(cmd)
        self.clock.advance(2); self.gateway.tick()
        events = self.outbox.events
        timed = [e for e in events if e.result.status is c.Outcome.TIMED_OUT]
        self.assertEqual(len(timed), 1)
        self.assertEqual(timed[0].result.physical, c.Fact.UNKNOWN)

    def test_T06_blocked_storage_does_not_block_stop(self):
        cmd = self.base(); self.prepare(cmd); self.gateway.submit_base(cmd)
        entered, release = threading.Event(), threading.Event()
        class Store:
            def append_event(self, event):
                entered.set(); release.wait(2)
                return m.CommitReceipt('sim-event-save', m.CommitStatus.COMMITTED, 1, 0, None)
        worker = threading.Thread(target=self.outbox.flush_one, args=(Store(),), daemon=True)
        worker.start()
        try:
            self.assertTrue(entered.wait(1))
            report = self.gateway.stop(self.stop_request())
            self.assertEqual(report.submitted, c.Fact.CONFIRMED)
            self.assertTrue(worker.is_alive())
        finally:
            release.set(); worker.join(1)

    def test_T06_storage_failure_revokes_on_independent_tick(self):
        cmd = self.base(); self.prepare(cmd); self.gateway.submit_base(cmd)
        class Store:
            def append_event(self, event):
                raise OSError('disk full')
        self.outbox.flush_one(Store()); self.gateway.tick()
        self.assertEqual(self.transport.base[-1], (0., 0.))
        self.assertIsNotNone(self.outbox.failure)

    def test_T06_monitor_worker_stops_without_business_poll(self):
        cmd = self.base(); self.prepare(cmd); self.gateway.submit_base(cmd)
        signal = threading.Event()
        send = self.transport.send_base
        def observe(linear, angular):
            send(linear, angular)
            if linear == angular == 0:
                signal.set()
        self.transport.send_base = observe
        with SafetyWorker(self.gateway, .005):
            self.clock.advance(10)
            self.assertTrue(signal.wait(1), 'independent worker must stop expired output')

    def test_T01_live_transport_is_blocked_even_with_simulated_commands(self):
        self.transport.origin = c.Origin.LIVE
        with self.assertRaises(ValueError):
            DeviceGateway(self.clock, s.run_context(), self.config, self.transport,
                          self.health_snapshot, self.outbox, self.lease)
        self.assertEqual(self.transport.base, [])

    def test_T04_action_scoped_permit_cannot_authorize_another_action(self):
        cmd = self.base()
        permit = dataclasses.replace(cmd.permit, envelope=dataclasses.replace(cmd.permit.envelope, action_id='different-action'))
        cmd = dataclasses.replace(cmd, permit=permit); self.prepare(cmd)
        self.assertEqual(self.gateway.submit_base(cmd).result.accepted, c.Fact.REJECTED)
        self.assertEqual(self.transport.base, [])

    def test_T05_unmeasured_limits_reject_prepared_commands(self):
        from treasure_auto.devices.config import load_control_config
        config = load_control_config(PACKAGE / 'config/devices/device_control.json')
        config = dataclasses.replace(config, config_version='sim-config-v1')
        self.gateway.config = config
        cmd = self.arm(); self.prepare(cmd)
        result = self.gateway.submit_arm(cmd)
        self.assertEqual(result.result.reason.code, c.ErrorCode.UNCONFIGURED)
        base = self.base(); self.prepare(base)
        self.assertEqual(self.gateway.submit_base(base).result.reason.code, c.ErrorCode.UNCONFIGURED)
        self.assertEqual(self.transport.arm + self.transport.base, [])

    def test_T05_arm_intent_cannot_borrow_another_plan_identity(self):
        cmd = self.arm(); intent = self.intent(cmd)
        plan = dataclasses.replace(intent.arm_plan, envelope=dataclasses.replace(intent.arm_plan.envelope, action_id='other'))
        with self.assertRaises(c.ContractError):
            self.register(cmd, dataclasses.replace(intent, arm_plan=plan))

    def test_T05_geometry_expiring_after_registration_blocks_dispatch(self):
        cmd = self.arm(); intent = self.intent(cmd)
        plan = dataclasses.replace(intent.arm_plan, envelope=dataclasses.replace(intent.arm_plan.envelope, expires_at=s.stamp(111.1)))
        self.register(cmd, dataclasses.replace(intent, arm_plan=plan))
        self.gateway.install_permit(cmd.permit)
        self.clock.advance(.2)
        result = self.gateway.submit_arm(cmd)
        self.assertEqual(result.result.accepted, c.Fact.REJECTED)
        self.assertEqual(self.transport.arm, [])

    def test_T05_precondition_expiring_after_registration_blocks_dispatch(self):
        cmd = self.arm()
        cmd = dataclasses.replace(cmd, preconditions=(s.evidence(valid_until=s.stamp(111.1)),))
        self.prepare(cmd); self.clock.advance(.2)
        self.assertEqual(self.gateway.submit_arm(cmd).result.accepted, c.Fact.REJECTED)
        self.assertEqual(self.transport.arm, [])

    def test_T06_pump_echo_is_not_holding_or_release(self):
        env = s.envelope(c.Owner.P07, action_id='pump-action')
        permit = m.ControlPermit(dataclasses.replace(env, producer=c.Owner.P10A), 'pump-permit', m.MotionSource.PUMP,
                                  m.Phase.CONTACT.value, ())
        cmd = m.PumpCommand(env, permit, 'pump-receipt', True, m.Phase.CONTACT,
                            c.HoldingSnapshot(c.Holding.UNKNOWN, None, None, ()))
        self.prepare(cmd); result = self.gateway.submit_pump(cmd)
        self.assertEqual(result.result.submitted, c.Fact.CONFIRMED)
        echo = s.evidence(kind=c.EvidenceKind.ECHO, action_id='pump-action', received_at=self.clock.now())
        self.assertEqual(self.gateway.record_feedback('pump-action', echo).result.physical, c.Fact.UNKNOWN)
        self.gateway.stop(self.stop_request())
        self.assertEqual(self.transport.pump, [True])

    def test_T06_old_physical_observation_cannot_confirm_new_dispatch(self):
        self.gateway._physical_validator = lambda cmd, ev: True
        cmd = self.arm(); self.prepare(cmd); self.gateway.submit_arm(cmd)
        ev = s.evidence(action_id=cmd.envelope.action_id, received_at=s.stamp(110))
        with self.assertRaises(c.ContractError):
            self.gateway.record_feedback(cmd.envelope.action_id, ev)

    def test_T06_new_validated_physical_evidence_can_confirm_without_holding_claim(self):
        self.gateway._physical_validator = lambda cmd, ev: ev.uri == 'sim://independent-observer/at-target'
        cmd = self.arm(); self.prepare(cmd); self.gateway.submit_arm(cmd)
        self.clock.advance(.1)
        ev = s.evidence(action_id=cmd.envelope.action_id, received_at=self.clock.now(), uri='sim://independent-observer/at-target')
        result = self.gateway.record_feedback(cmd.envelope.action_id, ev)
        self.assertEqual(result.result.physical, c.Fact.CONFIRMED)
        self.assertEqual(result.result.status, c.Outcome.SUCCESS)
        self.assertEqual(len(self.transport.arm), 1)

    def test_T06_same_serial_lease_cannot_be_shared_by_two_gateways(self):
        with self.assertRaises(RuntimeError):
            DeviceGateway(self.clock, s.run_context(), self.config, BoundaryTransport(),
                          self.health_snapshot, EventOutbox(10), self.lease)

    def test_T06_lease_release_and_reacquire_does_not_revive_old_controller(self):
        cmd = self.arm(); self.prepare(cmd)
        self.lease.release(); self.lease.acquire()
        self.assertEqual(self.gateway.submit_arm(cmd).result.accepted, c.Fact.REJECTED)
        self.assertEqual(self.transport.arm, [])

    def test_T06_blocked_evidence_validation_does_not_delay_stop(self):
        entered, release = threading.Event(), threading.Event()
        def validate(cmd, ev):
            entered.set(); release.wait(2); return True
        self.gateway._physical_validator = validate
        cmd = self.arm(); self.prepare(cmd); self.gateway.submit_arm(cmd); self.clock.advance(.1)
        ev = s.evidence(action_id=cmd.envelope.action_id, received_at=self.clock.now())
        errors = []
        def observe():
            try: self.gateway.record_feedback(cmd.envelope.action_id, ev)
            except c.ContractError as error: errors.append(error.code)
        worker = threading.Thread(target=observe, daemon=True); worker.start()
        try:
            self.assertTrue(entered.wait(1))
            report = self.gateway.stop(self.stop_request())
            self.assertTrue(worker.is_alive(), 'stop must finish before a blocked validator')
            self.assertEqual(report.physically_stopped, c.Fact.UNKNOWN)
        finally:
            release.set(); worker.join(1)
        self.assertEqual(errors, [c.ErrorCode.STALE_REQUEST])

    def test_T06_stop_without_active_commands_is_logged_and_duplicate_is_idempotent(self):
        request = self.stop_request()
        first = self.gateway.stop(request)
        self.assertEqual(self.gateway.stop(request), first)
        self.assertEqual(self.transport.base, [(0., 0.)])
        self.assertEqual(len([e for e in self.outbox.events if e.result.phase == 'stop']), 1)

    def test_T03_clock_regression_stops_and_latches(self):
        cmd = self.base(); self.prepare(cmd); self.gateway.submit_base(cmd)
        self.clock._now = s.stamp(110)
        self.gateway.tick()
        self.assertEqual(self.transport.base[-1], (0., 0.))
        self.clock._now = s.stamp(112)
        new = self.base(sequence=2)
        with self.assertRaises(c.ContractError): self.prepare(new)

    def test_T06_outbox_full_rejects_dispatch_and_preserves_pending(self):
        self.outbox.capacity = 1
        cmd = self.base(); self.prepare(cmd)
        self.gateway.submit_base(cmd)
        self.assertEqual(self.transport.base, [(0., 0.)])
        self.assertEqual(self.outbox.pending_count, 1)
        self.assertIsNotNone(self.outbox.failure)

    def test_T06_storage_unknown_is_not_retried_and_is_not_acknowledged(self):
        cmd = self.base(); self.prepare(cmd); self.gateway.submit_base(cmd)
        calls = []
        class Store:
            def append_event(self, event):
                calls.append(event)
                return m.CommitReceipt('unknown', m.CommitStatus.UNKNOWN, None, None,
                                       c.Reason(c.ErrorCode.STORAGE_FAILURE, 'disk', 'uncertain commit'))
        count = self.outbox.pending_count
        self.outbox.flush_one(Store()); self.outbox.flush_one(Store())
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.outbox.pending_count, count)

    def test_T04_long_original_deadline_cannot_be_renewed_by_permit(self):
        cmd = self.base()
        permit = dataclasses.replace(cmd.permit, envelope=dataclasses.replace(cmd.permit.envelope, expires_at=s.stamp(701)))
        with self.assertRaises(c.ContractError): self.gateway.install_permit(permit)

    def test_T04_business_purpose_read_board_keeps_exact_registered_scope(self):
        cmd = self.base()
        cmd = dataclasses.replace(cmd, permit=dataclasses.replace(cmd.permit, purpose='read_board'))
        self.prepare(cmd)
        self.assertEqual(self.gateway.submit_base(cmd).result.submitted, c.Fact.CONFIRMED)
        changed = dataclasses.replace(cmd, permit=dataclasses.replace(cmd.permit, purpose='central_pickup'))
        self.assertEqual(self.gateway.submit_base(changed).result.accepted, c.Fact.REJECTED)
        self.assertEqual(self.transport.base, [(.2, .1)])

    def test_T06_base_health_loss_does_not_toggle_pump_or_restart_arm(self):
        base = self.base(); arm = self.arm()
        self.prepare(base); self.prepare(arm)
        self.gateway.submit_base(base); self.gateway.submit_arm(arm)
        self.health_status['base'] = c.Fact.REJECTED; self.gateway.tick()
        self.assertEqual(self.transport.base[-1], (0., 0.))
        self.assertEqual(self.transport.stops, [])
        self.assertEqual(len(self.transport.arm), 1)

    def test_T06_cancel_failure_cannot_claim_stop_submitted(self):
        self.transport.fail = 'cancel'
        report = self.gateway.stop(self.stop_request())
        self.assertEqual(report.submitted, c.Fact.UNKNOWN)
        self.assertEqual(report.physically_stopped, c.Fact.UNKNOWN)

    def test_T02_real_clock_progress_during_health_sampling_does_not_latch_stop(self):
        from treasure_auto.devices.health import DeviceRule, DeviceObservation, HealthMonitor
        class AdvancingClock(FakeClock):
            def now(self):
                self.advance(.001)
                return super().now()
        self.clock = AdvancingClock(s.stamp(111))
        rule = DeviceRule('odom', 'nav_msgs/Odometry', '/base', 'odom', 'sim-v1', .5, ('odom->base',), 2, .01)
        health = HealthMonitor(self.clock, s.envelope(c.Owner.P01), (rule,), {'base': ('odom',)})
        for n in (1, 2):
            self.clock.advance(.02)
            health.update('odom', DeviceObservation(str(n), self.clock.now(), s.sensor_stamp(20+n),
                           True, '/base', 'nav_msgs/Odometry', 'nav_msgs/Odometry', n, 'odom', 'sim-v1', ('odom->base',), True))
        self.gateway.clock = self.clock
        self.gateway._health_provider = health.health
        cmd = self.base(); self.prepare(cmd)
        self.assertEqual(self.gateway.submit_base(cmd).result.submitted, c.Fact.CONFIRMED)
        self.assertEqual(self.transport.base, [(.2, .1)])

    def test_T04_base_physical_confirmation_cannot_disable_command_watchdog(self):
        cmd = self.base(envelope=s.envelope(c.Owner.P03, action_id='action-1', expires_at=s.stamp(112)))
        self.gateway._physical_validator = lambda command, ev: True
        self.prepare(cmd); self.gateway.submit_base(cmd)
        self.clock.advance(.1)
        self.gateway.record_feedback(cmd.envelope.action_id,
                                     s.evidence(action_id=cmd.envelope.action_id, received_at=self.clock.now()))
        self.clock.advance(1.1); self.gateway.tick()
        self.assertEqual(self.transport.base[-1], (0., 0.))

    def test_T06_stop_clock_fault_still_attempts_zero_output(self):
        cmd = self.base(); self.prepare(cmd); self.gateway.submit_base(cmd)
        self.clock._now = s.stamp(110.5)
        with self.assertRaises(c.ContractError): self.gateway.stop(self.stop_request())
        self.assertEqual(self.transport.base[-1], (0., 0.))

    def test_T06_failed_stop_is_retried_by_watchdog_without_replaying_motion(self):
        cmd = self.base(); self.prepare(cmd); self.gateway.submit_base(cmd)
        self.transport.fail = 'base'
        request = self.stop_request()
        self.assertEqual(self.gateway.stop(request).submitted, c.Fact.UNKNOWN)
        with self.assertRaises(c.ContractError):
            self.gateway.install_permit(self.base(sequence=2).permit)
        self.transport.fail = None
        self.clock.advance(.1); self.gateway.tick()
        report = self.gateway.stop(request)
        self.assertEqual(report.submitted, c.Fact.CONFIRMED)
        self.assertEqual(report.physically_stopped, c.Fact.UNKNOWN)
        self.assertEqual(self.transport.base, [(.2, .1), (0., 0.)])
        self.assertEqual(self.gateway.submit_base(cmd).result.accepted, c.Fact.REJECTED)

    def test_T06_deadline_latch_still_retries_failed_stop(self):
        cmd = self.base(); self.prepare(cmd); self.gateway.submit_base(cmd)
        self.transport.fail = 'base'
        self.clock._now = s.stamp(700); self.gateway.tick()
        self.transport.fail = None; self.gateway.tick()
        self.assertEqual(self.transport.base[-1], (0., 0.))
        with self.assertRaises(c.ContractError): self.gateway.install_permit(self.base(sequence=2).permit)

    def test_T06_failed_base_stop_cannot_hide_arm_timeout(self):
        base, arm = self.base(), self.arm()
        self.prepare(base); self.prepare(arm)
        self.gateway.submit_base(base); self.gateway.submit_arm(arm)
        self.transport.fail = 'base'; self.gateway.revoke(m.MotionSource.NAVIGATION)
        self.clock.advance(2); self.gateway.tick()
        self.assertEqual(self.transport.stops, ['cancel_requested'])


if __name__ == '__main__':
    unittest.main()
