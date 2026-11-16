"""P10A SIMULATED fixtures; never a hardware adapter or installed package."""
from dataclasses import replace
from pathlib import Path
import sys
import threading

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from treasure_auto.contracts import core as c, models as m, rules as r
from contract_fixtures import samples as s
from contract_fixtures.fakes import FakeClock


def config_data():
    return dict(mode='offline', supervision_period_s=0.01, lease_s=1.0,
                heartbeat_timeout_s=2.0, required_capabilities=['base', 'unique_control'],
                signal_adapter=None)


def make_runtime(mapping=True, **kwargs):
    from treasure_auto.runtime import RuntimeConfig, RuntimeController
    clock = FakeClock(s.stamp(100))
    ids = iter('sim-id-{}'.format(i) for i in range(10000))
    versions = replace(s.versions(), mapping='sim-mapping-v1' if mapping else None)
    rt = RuntimeController(RuntimeConfig.from_dict(config_data()), clock, versions,
                           kwargs.get('instance_id', 'sim-instance-1'),
                           kwargs.get('control_epoch', 1), id_factory=lambda: next(ids))
    return rt, clock


def signal(rt, clock, kind=m.SignalKind.START, event='signal-1', auth=c.Fact.CONFIRMED):
    env = rt.admission if rt.run is None else rt.run.envelope
    env = replace(env, event_id=event, sequence=int(event.rsplit('-', 1)[-1]),
                  emitted_at=clock.now(), expires_at=replace(clock.now(), seconds=clock.now().seconds+10))
    return m.RuntimeSignal(env, kind, auth)


def initial_snapshot(run):
    env = replace(run.envelope, producer=c.Owner.P08)
    mapping=None
    if env.versions.mapping is not None:
        ev=s.evidence(run_id=env.run_id,target_id=None,action_id=None,
                      received_at=env.emitted_at,valid_until=env.expires_at)
        mapping=replace(s.mapping_samples()[0],envelope=replace(env,producer=c.Owner.P05),
                        mapping_version=env.versions.mapping,frozen_at=env.emitted_at,evidence=(ev,))
    return m.StateSnapshot(run, 0, m.SpaceSnapshot(env, env.versions.space, ()),
                           c.HoldingSnapshot(c.Holding.UNKNOWN, None, None, ()), (), (), (), mapping, ())


class Store:
    """Models only initialization boundaries, including genuine blocking waits."""
    def __init__(self, status=m.CommitStatus.COMMITTED, block=False):
        self.status, self.block, self.calls = status, block, 0
        self.entered, self.release = threading.Event(), threading.Event()
        self.snapshot = None

    def initialize_run(self, request):
        self.calls += 1
        self.entered.set()
        if self.block and not self.release.wait(3):
            raise TimeoutError('test failed to release blocked store')
        self.snapshot = request.initial_snapshot
        reason = None if self.status is m.CommitStatus.COMMITTED else c.Reason(c.ErrorCode.STORAGE_FAILURE, 'commit', 'simulated uncertainty')
        return m.CommitReceipt('sim-init', self.status, 0 if reason is None else None,
                               0 if reason is None else None, reason)

    def read(self, run_id):
        return m.ReadResult(c.Outcome.SUCCESS, self.snapshot, None)


def started(mapping=True):
    rt, clock = make_runtime(mapping)
    rt.arm()
    rt.signal(signal(rt, clock))
    rt.initialize(Store(), initial_snapshot(rt.run))
    rt.heartbeat(replace(rt.run.envelope, producer=c.Owner.P09))
    return rt, clock


def request(rt, clock, source=m.MotionSource.NAVIGATION, purpose='read_board', seq=10):
    now = clock.now()
    env = replace(rt.run.envelope, producer=c.Owner.P09, event_id='req-{}'.format(seq),
                  sequence=seq, request_id='sim-work-{}'.format(seq), action_id='sim-action-{}'.format(seq),
                  target_id=None, emitted_at=now, expires_at=replace(now, seconds=now.seconds+10),
                  base_motion_epoch=0, layout_recovery_round=0, rebuild_generation=0)
    def owner(o):
        return replace(env, producer=o, request_id=None, action_id=None, target_id=None)
    ev = s.evidence(run_id=env.run_id, target_id=None, action_id=None,
                    captured_at=s.sensor_stamp(now.seconds-80),
                    received_at=now, valid_until=replace(now, seconds=now.seconds+10))
    holding = c.HoldingSnapshot(c.Holding.EMPTY, None, None, (ev,))
    health = m.HealthSnapshot(owner(c.Owner.P01), tuple(m.Capability(n, c.Fact.CONFIRMED, None, (ev,)) for n in ('base', 'unique_control')))
    loc = m.LocalizationSnapshot(owner(c.Owner.P02), c.MapPose('map', 0, 0, 0, 'm', 'rad', 0.01, 0.01), c.Fact.CONFIRMED, 0, (ev,), None)
    ready = m.DriveReadiness(owner(c.Owner.P07), holding, c.Fact.CONFIRMED, (ev,))
    space = m.SpaceSnapshot(owner(c.Owner.P08), env.versions.space, ())
    stop = m.StopReport(owner(c.Owner.P01), c.Fact.CONFIRMED, c.Fact.CONFIRMED, c.Fact.UNKNOWN,
                        c.Fact.CONFIRMED, c.Fact.CONFIRMED, (ev,))
    intent = m.ActionIntent(env, m.IntentOperation.BASE_VELOCITY, None, None, None, (0.1, 0), None)
    receipt = s.intent_receipt(intent)
    rt.remember_intent(intent, receipt, source=source)
    req = m.PermissionRequest(env, source, purpose, health, loc, ready, space, stop, None, receipt)
    return req


class ShadowGateway:
    """Test sink asks the real P10A authority; no copied P01 gateway algorithm."""
    origin = c.Origin.SIMULATED
    def __init__(self, runtime, clock):
        self.runtime, self.clock, self.output = runtime, clock, 0
        self.stops, self.stopped = [], threading.Event()

    def send(self, permit, envelope, source, value):
        accepted = self.runtime.authorize_output(permit, envelope, source, real_output=False)
        self.output = value if accepted else 0
        return accepted

    def stop(self, request):
        self.stops.append(request)
        self.output = 0
        self.stopped.set()
