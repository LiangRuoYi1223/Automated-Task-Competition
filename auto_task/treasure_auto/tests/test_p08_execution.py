import unittest
from dataclasses import replace
from test_p08_support import *
from contract_fixtures.fakes import FakeClock
from treasure_auto.contracts import rules as r
from treasure_auto.placement.execution import PlacementExecutor


class Journal:
    def __init__(self):self.value=None;self.unknown=False
    def load(self,run_id):return self.value
    def save(self,run_id,transaction_id,value):
        if self.unknown:return m.CommitReceipt(transaction_id,m.CommitStatus.UNKNOWN,None,None,c.Reason(c.ErrorCode.STORAGE_FAILURE,'journal','unknown'))
        self.value=value
        return m.CommitReceipt(transaction_id,m.CommitStatus.COMMITTED,1,0,None)


class Harness:
    def __init__(self):
        self.clock=FakeClock(s.stamp(110));self.ctx=context();self.req=request();self.commands=[];self.intents=[];self.stops=[]
        self.journal=Journal();self.unknown=False;self.on_commit=None;self.on_stop=None;self.on_prepare=None
        self.executor=self.build()
    def current(self):return replace(self.ctx,now=self.clock.now())
    def stop(self):
        if self.on_stop:self.on_stop()
        return self.req.stop
    def plan(self,phase,envelope,order,holding):
        point=self.req.plan.target
        return m.GeometryPlan(replace(envelope,producer=c.Owner.P06),'p08-'+phase.value,point,
            (m.PathSegment(phase,point,1.,2.),),c.Fact.CONFIRMED,c.Fact.CONFIRMED,
            (proof(self.clock.now().seconds,action=envelope.action_id),))
    def commit_intent(self,intent):
        self.intents.append(intent)
        if self.on_commit:self.on_commit()
        receipt=s.intent_receipt(intent)
        if self.unknown:return replace(receipt,commit=m.CommitReceipt('intent',m.CommitStatus.UNKNOWN,None,None,c.Reason(c.ErrorCode.STORAGE_FAILURE,'intent','unknown')))
        return receipt
    def append_event(self,event):return m.CommitReceipt('event',m.CommitStatus.COMMITTED,1,0,None)
    def authorize(self,intent,receipt):
        return m.ControlPermit(replace(intent.envelope,producer=c.Owner.P10A),'stage-permit',
             m.MotionSource.PUMP if intent.operation is m.IntentOperation.PUMP else m.MotionSource.ARM,'place',intent.arm_plan.evidence if intent.arm_plan else (proof(action=intent.envelope.action_id),))
    def prepare(self,*args):
        if self.on_prepare:self.on_prepare()
    def submit_arm(self,command):return self.submit(command)
    def submit_pump(self,command):return self.submit(command)
    def submit(self,command):
        self.commands.append(command)
        return m.ActionEvent(command.envelope,c.ActionResult(c.Outcome.UNKNOWN,'submitted',c.Reason(c.ErrorCode.EVIDENCE_MISSING,'physical','await observation'),
            c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.UNKNOWN,c.Fact.UNKNOWN,(),0))
    def stop_device(self,request):self.stops.append(request);return self.req.stop
    def build(self):
        class Device:
            submit_arm=self.submit_arm;submit_pump=self.submit_pump;stop=self.stop_device
        return PlacementExecutor(config(),s.run_context(),self.clock,context_provider=self.current,stop_provider=self.stop,
            stage_planner=self.plan,persistence=self,authorize=self.authorize,prepare=self.prepare,device=Device(),checkpoints=self.journal)
    def complete(self,physical=c.Fact.CONFIRMED):
        command=self.commands[-1];self.clock.advance(.05)
        event=m.ActionEvent(replace(command.envelope,producer=c.Owner.P01,emitted_at=self.clock.now()),
            c.ActionResult(c.Outcome.SUCCESS,'observed',None,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,physical,
                           (proof(self.clock.now().seconds,action=command.envelope.action_id),),.05))
        return self.executor.confirm(event)


class ExecutionTests(unittest.TestCase):
    def test_R04_stages_wait_for_physical_feedback_then_pump_and_withdraw(self):
        h=Harness();h.executor.start(h.req)
        self.assertEqual(len(h.commands),1)
        h.executor.tick();self.assertEqual(len(h.commands),1)
        h.complete(c.Fact.UNKNOWN);h.executor.tick();self.assertEqual(len(h.commands),1)
        for _ in range(5):
            h.complete();h.executor.tick()
        self.assertEqual([x.phase if isinstance(x,m.ArmCommand) else None for x in h.commands],
            [m.Phase.APPROACH,m.Phase.DESCEND,m.Phase.PLACE,None,m.Phase.WITHDRAW])
        self.assertFalse(h.commands[3].enabled)
        self.assertEqual(h.executor.state,'await_release')
        self.assertEqual(h.executor.holding.state,c.Holding.HELD)

    def test_repeat_start_conflicting_id_and_duplicate_feedback(self):
        h=Harness();h.executor.start(h.req);h.executor.start(h.req)
        self.assertEqual(len(h.commands),1)
        h.complete();h.executor.tick();self.assertEqual(len(h.commands),2)
        h.executor.start(h.req);self.assertEqual(len(h.commands),2)
        with self.assertRaises(c.ContractError):h.executor.start(replace(h.req,plan=replace(h.req.plan,path_id='altered')))

    def test_unknown_intent_or_checkpoint_never_dispatch(self):
        for location in ('intent','checkpoint'):
            h=Harness();h.unknown=location=='intent';h.journal.unknown=location=='checkpoint'
            h.executor.start(h.req)
            self.assertEqual(h.commands,[]);self.assertEqual(h.executor.state,'unresolved')

    def test_deadline_and_cancel_during_callbacks_prevent_late_output(self):
        for where in ('commit','stop','prepare'):
            h=Harness();setattr(h,'on_'+where,lambda:h.clock.advance(600))
            h.executor.start(h.req)
            self.assertEqual(h.commands,[])
        h=Harness();h.on_prepare=lambda:h.executor.cancel('simulated emergency')
        h.executor.start(h.req);self.assertEqual(h.commands,[])

    def test_timeout_cancels_without_additional_pump_off_or_withdraw(self):
        h=Harness();h.executor.start(h.req);h.clock.advance(3);h.executor.tick()
        self.assertEqual(h.executor.state,'unresolved');self.assertEqual(len(h.commands),1)
        self.assertEqual(len(h.stops),1);self.assertEqual(h.executor.holding.state,c.Holding.HELD)

    def test_restart_and_old_context_never_replays_commands(self):
        h=Harness();h.executor.start(h.req);new=h.build();new.start(h.req)
        self.assertEqual(len(h.commands),1);self.assertEqual(new.state,'recovery_required')
        h.ctx=replace(h.ctx,base_motion_epoch=1);h.complete();h.executor.tick()
        self.assertEqual(len(h.commands),1)

    def test_missing_per_stage_plan_and_exhausted_budget_block(self):
        h=Harness();h.executor.stage_planner=lambda *a:None;h.executor.start(h.req)
        self.assertEqual(h.commands,[])
        h=Harness();h.req=replace(h.req,order=replace(h.req.order,budget=m.Budget('recovery-B',1,10,1,1)))
        h.executor.start(h.req);self.assertEqual(h.commands,[])


if __name__=='__main__':unittest.main()
