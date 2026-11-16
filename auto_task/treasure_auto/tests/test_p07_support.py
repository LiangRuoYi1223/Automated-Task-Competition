"""P07-only SIMULATED vectors and bounded dependency doubles; never robot values."""
import importlib
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c, models as m, rules as r
from contract_fixtures import samples as s
from contract_fixtures.fakes import FakeClock


def api():
    try:
        return importlib.import_module('treasure_auto.manipulation.service')
    except ModuleNotFoundError as exc:
        raise AssertionError('P07 executable service has not been delivered') from exc


def point(z=10, x=0):
    return c.ArmPoint('sim-arm', x, 0, z, 'mm', .1)


class Checkpoints:
    """Durable-file fake only when path supplied; not a P10B implementation."""
    def __init__(self, path=None):
        self.path, self.value, self.fail = path, None, False

    def load(self, run_id):
        if self.fail:
            raise OSError('simulated checkpoint read/write unavailable')
        if self.path and self.path.exists():
            return json.loads(self.path.read_text(encoding='utf-8'))
        return json.loads(json.dumps(self.value))

    def save(self, run_id, transaction_id, value):
        if self.fail:
            return m.CommitReceipt(transaction_id, m.CommitStatus.UNKNOWN, None, None,
                                   c.Reason(c.ErrorCode.STORAGE_FAILURE, 'checkpoint', 'unknown write'))
        self.value = json.loads(json.dumps(value))
        if self.path:
            self.path.write_text(json.dumps(value), encoding='utf-8')
        return m.CommitReceipt(transaction_id, m.CommitStatus.COMMITTED, 0, 0, None)


class Harness:
    def __init__(self, checkpoints=None):
        self.p = api()
        self.clock = FakeClock(s.stamp(110))
        self.context = replace(s.consumer_context(), now=self.clock.now(), request_id=None,
                               action_id=None, target_id=None)
        self.config = self.p.ManipulationConfig('sim-config-v1', 2., 2, .1, .5, 1., 1., 5., 30.,
                                               'sim-transport-42', 2., 1.)
        self.checkpoints = checkpoints or Checkpoints()
        self.commands, self.stops, self.intents, self.events = [], [], [], []
        self.intent_status = m.CommitStatus.COMMITTED
        self.send_raises = False
        self.stop_confirmed = True
        self.expire_during_commit = False
        self.plan_available = True
        self.prepare_callback = None
        self.sequence = 0
        self.store = self
        self.service = self.new_service()
        if not self.service.reconciliation_required:
            self.service.observe(self.observation(empty=True, action_id=None, target_id=None))

    def new_service(self):
        return self.p.ManipulationService(self.config, s.run_context(), self.clock,
            context_provider=lambda: replace(self.context, now=self.clock.now()),
            capture_now=self.clock.now, stop_provider=self.stop_report,
            stage_planner=self.stage_plan, authorize=self.authorize, prepare=self.prepare,
            device=self, persistence=self.store, checkpoints=self.checkpoints)

    def env(self, owner=c.Owner.P07, **changes):
        now = self.clock.now()
        self.sequence+=1
        changes.setdefault('event_id','fixture-'+str(self.sequence))
        changes.setdefault('sequence',self.sequence)
        changes.setdefault('versions',self.context.versions)
        return s.envelope(owner, emitted_at=now, expires_at=replace(now, seconds=now.seconds+10),
                          **changes)

    def ev(self, action_id=None, target_id=None, **changes):
        now = self.clock.now()
        data = dict(evidence_id='ev-'+str(now.seconds), action_id=action_id, target_id=target_id,
                    captured_at=now, received_at=now, valid_until=replace(now, seconds=now.seconds+2),
                    uri='sim://p07/'+str(now.seconds))
        data.update(changes)
        return s.evidence(**data)

    def stop_report(self):
        fact = c.Fact.CONFIRMED if self.stop_confirmed else c.Fact.UNKNOWN
        return m.StopReport(self.env(c.Owner.P01, action_id=None, target_id=None), fact, fact, fact,
                            fact, fact, (self.ev(),))

    def request(self, action_id='grasp-1', attempts=0):
        e = self.env(c.Owner.P09, action_id=action_id)
        order = m.WorkOrder(e, m.Category.CAKE, m.Area.A, 'sim-mapping-v1', 'inventory-1',
                           m.Budget('p07-grasp:inventory-1', 2, 30., attempts, 0.))
        phases = (m.Phase.APPROACH,m.Phase.DESCEND,m.Phase.CONTACT,m.Phase.TRIAL_LIFT,m.Phase.WITHDRAW)
        plan = m.GeometryPlan(replace(e, producer=c.Owner.P06), 'source-path', point(),
                  tuple(m.PathSegment(p, point(z), 1., 2.) for p,z in zip(phases,(20,12,10,30,30))),
                  c.Fact.CONFIRMED,c.Fact.CONFIRMED,(self.ev(action_id,'brick-1'),))
        permit = m.ControlPermit(replace(e,producer=c.Owner.P10A), 'admission',m.MotionSource.ARM,
                                 'grasp', (self.ev(action_id,'brick-1'),))
        return m.GraspRequest(e,order,plan,self.stop_report(),self.service.holding,permit)

    def stage_plan(self, phase, envelope, order, holding):
        if not self.plan_available:
            return None
        z = {m.Phase.APPROACH:20,m.Phase.DESCEND:12,m.Phase.CONTACT:10,
             m.Phase.TRIAL_LIFT:30,m.Phase.WITHDRAW:30,m.Phase.TRANSPORT:40}[phase]
        e=replace(envelope, producer=c.Owner.P06)
        return m.GeometryPlan(e,'stage-'+e.action_id,point(z),
                 (m.PathSegment(phase,point(z),1.,2.),),c.Fact.CONFIRMED,c.Fact.CONFIRMED,
                 (self.ev(e.action_id,e.target_id),))

    def commit_intent(self,intent):
        self.intents.append(intent)
        if self.expire_during_commit:
            self.clock.advance(600)
        if self.intent_status is m.CommitStatus.COMMITTED:
            return s.intent_receipt(intent)
        return m.IntentReceipt(intent.envelope.run_id,intent.envelope.action_id,r.intent_digest(intent),
            m.CommitReceipt('bad',self.intent_status,None,None,c.Reason(c.ErrorCode.STORAGE_FAILURE,'intent','write failed')))

    def append_event(self,event):
        self.events.append(event)
        return m.CommitReceipt('event',m.CommitStatus.COMMITTED,0,0,None)

    def authorize(self,intent,receipt,holding):
        return m.ControlPermit(replace(intent.envelope,producer=c.Owner.P10A),
              'permit-'+intent.envelope.action_id,
              m.MotionSource.PUMP if intent.operation is m.IntentOperation.PUMP else m.MotionSource.ARM,
              'grasp', (self.ev(intent.envelope.action_id,intent.envelope.target_id),))

    def prepare(self,command,intent,receipt):
        if self.prepare_callback:
            self.prepare_callback()

    def submit_arm(self,command):
        self.commands.append(command)
        if self.send_raises:
            raise OSError('simulated acknowledgement lost after send')
        return m.ActionEvent(replace(command.envelope,producer=c.Owner.P01),
            c.ActionResult(c.Outcome.UNKNOWN,command.phase.value,c.Reason(c.ErrorCode.EVIDENCE_MISSING,'physical','not observed'),
              c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.UNKNOWN,(),0.))

    submit_pump=submit_arm

    def stop(self,request):
        self.stops.append(request)
        return self.stop_report()

    def observation(self, empty=False, action_id='active', target_id='brick-1', z=None, **changes):
        self.clock.advance(.15)
        service=self.service
        if action_id=='active':
            action_id=service.active_action_id
        if z is None:
            z=service.current_target.z if service.current_target else 10
        tool=m.ToolObservation(self.env(c.Owner.P04,action_id=action_id,target_id=target_id),
                  None if empty else target_id, (m.PixelPoint(1,1),m.PixelPoint(2,1),m.PixelPoint(2,2)),
                  m.PixelPoint(1.5,1.5),c.Fact.CONFIRMED,(self.ev(action_id,target_id),))
        data=dict(tool=tool,arm=point(z),cup=point(z),brick=None if empty else point(z-10),
                  support_z_mm=0.,support_error_mm=.1,support_visible=c.Fact.CONFIRMED,
                  tool_visible=c.Fact.CONFIRMED,empty=c.Fact.CONFIRMED if empty else c.Fact.UNKNOWN,
                  dropped=c.Fact.UNKNOWN,drop_region=None,posture_id='sim-transport-42')
        data.update(changes)
        return self.p.MetricObservation(**data)

    def confirm_stage(self):
        for _ in range(2):
            self.service.observe(self.observation())

    def reach(self,stage):
        if self.service.active_action_id is None:
            self.service.grasp(self.request())
        for _ in range(10):
            if self.service.stage==stage:
                return
            self.confirm_stage()
            self.service.tick()
        raise AssertionError('stage not reached: '+stage)


class Case(unittest.TestCase):
    def setUp(self):
        self.h=Harness()
