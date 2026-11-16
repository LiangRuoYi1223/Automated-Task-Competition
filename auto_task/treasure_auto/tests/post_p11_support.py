"""Simulated sensor/transport boundaries; domain and gateway classes stay real."""
from dataclasses import replace
from uuid import uuid4
from test_p08_execution import Harness as PlacementHarness, c, m, proof, s
from test_p01_gateway import BoundaryTransport
from treasure_auto.devices.gateway import DeviceGateway
from treasure_auto.devices.config import ControlConfig
from treasure_auto.devices.events import EventOutbox
from treasure_auto.devices.drivers import SerialLease


class AdvancingClock:
    def __init__(self, stamp):
        self.current = stamp

    def now(self):
        self.current = replace(self.current, seconds=self.current.seconds + .001)
        return self.current


def wire_placement(case, h):
    transport = BoundaryTransport()
    lease = SerialLease('sim://post-p11-' + uuid4().hex).acquire()
    case.addCleanup(lease.release)
    config = ControlConfig('sim-config-v1', .5, 1., 'arm',
        ((-200., 200.), (-200., 200.), (0., 300.)),
        tuple((phase, (.5, 10., 5.)) for phase in m.Phase))
    health = m.HealthSnapshot(replace(h.req.envelope, producer=c.Owner.P01),
        tuple(m.Capability(k, c.Fact.CONFIRMED, None, ()) for k in ('base', 'arm', 'pump')))
    gateway = DeviceGateway(h.clock, s.run_context(), config, transport,
        lambda: health, EventOutbox(100), lease,
        physical_validator=lambda command, evidence: True)
    original_authorize = h.authorize
    h.executor.authorize = lambda intent, receipt: replace(
        original_authorize(intent, receipt), permit_id='permit:' + intent.envelope.action_id)

    def commit_intent(intent):
        receipt = s.intent_receipt(intent)
        return replace(receipt, commit=replace(receipt.commit,
            transaction_id='intent:' + intent.envelope.action_id))

    def prepare(command, intent, receipt):
        if gateway._permits:
            gateway.revoke(command.permit.source)
        gateway.register_intent(command, intent, receipt)
        gateway.install_permit(command.permit)

    h.commit_intent = commit_intent
    h.executor.prepare = prepare
    h.executor.device = gateway
    return gateway, transport


def place_permission(owner=c.Owner.P08, purpose='place', holding_state=c.Holding.HELD):
    from test_p10a_support import started, request
    runtime, clock = started()
    base = request(runtime, clock)
    env = replace(base.envelope, producer=owner, action_id='place:0', target_id='brick-1')
    evidence = replace(base.readiness.evidence[0], target_id='brick-1', action_id=env.action_id)
    point = c.ArmPoint('arm', 0., 0., 100., 'mm', .1)
    plan = m.GeometryPlan(replace(env, producer=c.Owner.P06), 'place-approach', point,
        (m.PathSegment(m.Phase.APPROACH, point, 1., 2.),),
        c.Fact.CONFIRMED, c.Fact.CONFIRMED, (evidence,))
    intent = m.ActionIntent(env, m.IntentOperation.ARM, None, plan, None, None, None)
    receipt = s.intent_receipt(intent)
    held = c.HoldingSnapshot(holding_state, 'brick-1', point, (evidence,))
    permission = replace(base, envelope=env, source=m.MotionSource.ARM, purpose=purpose,
        readiness=replace(base.readiness, holding=held), intent_receipt=receipt)
    runtime.remember_intent(intent, receipt, m.MotionSource.ARM)
    return runtime, clock, permission
