"""P05 simulated vectors only; no robot parameters or production adapters."""
import importlib
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c, models as m
from contract_fixtures import samples as s


def api():
    module = importlib.import_module('treasure_auto.classification')
    assert hasattr(module, 'ClassificationConfig'), 'P05 configuration/geometry implementation missing'
    return module


def config():
    return api().ClassificationConfig('sim-config-v1', m.Measurement.SIMULATED,
        2, 0.1, 1.0, 2.0, 100.0, 0.05, 0.8, 0.01, 2,
        'board_observe', 'sim-waypoint-v1')


def template():
    p = api()
    return p.BoardTemplate('sim-template-v1', 'verified-top', ('border:top',),
        tuple(p.SlotRegion(a, i / 3, 0.0, (i + 1) / 3, 1.0)
              for i, a in enumerate((m.Area.B, m.Area.C, m.Area.D))), m.Measurement.SIMULATED)


def env(time=110.0, event='read-1', **changes):
    values = dict(event_id=event, target_id=None, action_id=None, request_id='read-board',
                  emitted_at=s.stamp(time), expires_at=s.stamp(time+10),
                  versions=replace(s.versions(), mapping=None))
    values.update(changes)
    return s.envelope(c.Owner.P09, **values)


def ev(name, time=110.0, sensor=20.0):
    import hashlib
    return s.evidence(evidence_id=name, target_id=None, action_id=None,
                      captured_at=s.sensor_stamp(sensor), received_at=s.stamp(time),
                      valid_until=s.stamp(time+10), uri='sim://p05/'+name,
                      sha256=hashlib.sha256(name.encode()).hexdigest())


def frame(index=0, categories=(m.Category.DOLL, m.Category.CLOCK, m.Category.BANANA), project=None):
    p = api()
    project = project or (lambda x,y: (100+600*x, 100+200*y))
    time, sensor = 110+index*0.2, 20+index*0.2
    evidence = ev('image-'+str(index), time, sensor)
    image = m.ImageObservation(replace(env(time, 'frame-'+str(index)), producer=c.Owner.P04),
        'obs-'+str(index), evidence, 1000, 1000, 'rgb8', 'camera', m.Distortion.CORRECTED,
        c.MapPose('map',0,0,0,'m','rad',0,0), m.PixelTransform(0,0,1,1,0,0))
    points = lambda coords: tuple(m.PixelPoint(*project(x,y)) for x,y in coords)
    direction = p.DirectionHint('border:top', (0,1,2,3), (evidence,))
    board = p.BoardCandidate(points(((0,0),(1,0),(1,1),(0,1))), c.Fact.CONFIRMED, (direction,))
    patterns = tuple(p.PatternInput(cat, 0.99,
        points(((i/3+0.06,0.2),(i/3+0.26,0.2),(i/3+0.26,0.8),(i/3+0.06,0.8))),
        (evidence,)) for i,cat in enumerate(categories))
    return p.FrameInput(image,'capture-'+str(index),s.sensor_stamp(sensor+0.01),s.sensor_stamp(19.9),
                        'stable-1','sim-waypoint-v1',c.Fact.CONFIRMED,(evidence,),(board,),patterns)


def observation(value):
    return m.BoardObservation(value.image,'sim-template-v1',c.Fact.CONFIRMED,(),(value.image.image,))


class P05Case(unittest.TestCase):
    def setUp(self):
        self.p = api()


class Harness:
    """Fake P03/P04 inputs and fake P08/P09 coordinator, real C0 validation/store oracle."""
    def __init__(self, settings=None):
        from contract_fixtures.fakes import FakeClock, FakeStateStore
        self.clock=FakeClock(s.stamp(110.01))
        self.run=replace(s.run_context(),envelope=replace(env(),producer=c.Owner.P10A))
        initial=replace(s.state_snapshot(),run=self.run,mapping=None,inventory=(),deliveries=(),
                        holding=c.HoldingSnapshot(c.Holding.EMPTY,None,None,()),
                        space=replace(s.space_snapshot(),occupancies=(),
                            envelope=replace(s.space_snapshot().envelope,versions=env().versions)))
        self.store=FakeStateStore(initial,self.clock)
        self.frames={}; self.calls=[]; self.mode='normal'; self.events=0
        self.settings=settings or config()
        p=api()
        assert hasattr(p,'ClassificationService'),'P05 read/confirm/freeze implementation missing'
        self.service=self.new_service()

    def new_service(self):
        e=env()
        context=m.ConsumerContext(e.run_id,e.instance_id,e.control_epoch,self.clock.now(),e.versions,
                                  e.request_id,None,None,0,0,0,e.origin)
        return api().ClassificationService(self.settings,template(),self.run,context,self.clock,
                   resolve_frame=self.resolve,store=self.store,submit_mapping=self.commit)

    def resolve(self, obs):
        return self.frames[obs.image.observation_id]

    def commit(self, mapping, audit):
        self.calls.append((mapping,audit))
        if self.mode=='raise': raise OSError('simulated storage outage')
        reason=c.Reason(c.ErrorCode.STORAGE_FAILURE,'commit','simulated ambiguity/failure')
        if self.mode in ('reject','unknown_before'):
            return m.CommitReceipt(mapping.mapping_version,
                m.CommitStatus.REJECTED if self.mode=='reject' else m.CommitStatus.UNKNOWN,None,None,reason)
        previous=self.store.read(self.run.envelope.run_id).snapshot
        header=replace(mapping.envelope,producer=c.Owner.P09,versions=previous.space.envelope.versions)
        space=replace(previous.space,version=previous.space.version+1,
                      envelope=replace(mapping.envelope,producer=c.Owner.P08,
                          versions=replace(mapping.envelope.versions,space=previous.space.version+1)))
        tx=m.StateTransaction(header,mapping.mapping_version,previous.revision,previous.space.version,
            space,previous.holding,previous.inventory,previous.layouts,mapping,(),(),None,previous.budgets,None)
        receipt=self.store.commit(tx)
        if self.mode=='unknown_after': return m.CommitReceipt(mapping.mapping_version,m.CommitStatus.UNKNOWN,None,None,reason)
        if self.mode=='wrong_receipt': return replace(receipt,transaction_id='unrelated')
        if self.mode=='late': self.clock.advance(10)
        return receipt

    def request(self, frames=(), deadline=115, views=2):
        self.events+=1
        for f in frames: self.frames[f.image.observation_id]=f
        if frames:
            latest=max(f.image.image.received_at.seconds for f in frames)+0.01
            if latest>self.clock.now().seconds: self.clock.advance(latest-self.clock.now().seconds)
        return m.ClassificationRequest(env(self.clock.now().seconds,'read-'+str(self.events)),s.stamp(deadline),
                                       tuple(observation(f) for f in frames),views)

    def send(self, frames=(), **changes):
        return self.service.read_mapping(self.request(frames,**changes))

