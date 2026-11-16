"""Bounded SIMULATED stage execution. Missing production adapters fail closed.

This does not split or rename an existing P06 plan. Each arm stage asks the
planner for a new one-stage certification bound to its own immutable action ID.
"""
import hashlib
from dataclasses import replace
from uuid import uuid4
from treasure_auto.contracts import core as c, models as m, rules as r
from . import guards


class PlacementExecutor:
    STAGES=(m.Phase.APPROACH,m.Phase.DESCEND,m.Phase.PLACE,None,m.Phase.WITHDRAW)

    def __init__(self,config,run,clock,*,context_provider,stop_provider,stage_planner,persistence,authorize,prepare,device,checkpoints,stop_sink=None):
        c.require(run.envelope.origin is c.Origin.SIMULATED,'origin','only offline simulated execution is commissioned',c.ErrorCode.NOT_AUTHORIZED)
        self.config=config;self.run=run;self.clock=clock;self.context_provider=context_provider;self.stop_provider=stop_provider
        self.stage_planner=stage_planner;self.persistence=persistence;self.authorize=authorize;self.prepare=prepare
        self.device=device;self.checkpoints=checkpoints;self.state='idle';self.reason=None;self.request=None
        self.stop_sink=stop_sink
        self.holding=None;self.index=0;self.command=None;self.started=None;self.sent_at=None;self.timeout=None;self.stop_report=None
        self.completed_phases=[];self._fingerprints={};self._feedback={};self._cancelled=False;self._busy=False;self._sequence=0;self._budgets={};self._limits={};self._area_names={}
        self._command_sequence=0
        try:
            previous=checkpoints.load(run.envelope.run_id)
            if previous is not None:
                c.require(type(previous) is dict and previous.get('format')=='p08-execution-1' and
                          previous.get('run_id')==run.envelope.run_id and previous.get('deadline')==c.to_data(run.deadline),
                          'checkpoint','invalid or foreign checkpoint')
                # Any old intent may have reached the actuator. Loading history
                # never replays output or treats a missing success as no action.
                self.state='recovery_required'
                self.reason=c.Reason(c.ErrorCode.UNAVAILABLE,'recovery','P09/P10B recovery rebinding required; no automatic resume')
        except Exception as error:
            self.state='recovery_required';self.reason=c.Reason(c.ErrorCode.STORAGE_FAILURE,'checkpoint',str(error))

    def _context(self):
        ctx=replace(self.context_provider(),now=self.clock.now())
        c.require(not self._cancelled,'cancel','execution cancelled',c.ErrorCode.CANCELLED)
        c.require(ctx.origin is c.Origin.SIMULATED and ctx.run_id==self.run.envelope.run_id,'run','wrong run/provenance')
        c.require(r.remaining_seconds(self.run,ctx.now)>0,'deadline','original cutoff reached',c.ErrorCode.DEADLINE)
        if self.request:
            r.validate_envelope(self.request.envelope,ctx)
            budget=self.request.order.budget
            used=self._budgets.get(budget.name,(budget.attempts_used,budget.seconds_used))[1]
            c.require(budget.seconds_limit is not None and used+(ctx.now-self.started)<budget.seconds_limit,
                      'budget','cumulative recovery time exhausted',c.ErrorCode.LIMIT_EXCEEDED)
        return ctx

    def _save(self):
        self._sequence+=1
        key='p08-checkpoint:{}:{}'.format(self.request.envelope.action_id,self._sequence)
        value=dict(format='p08-execution-1',run_id=self.run.envelope.run_id,deadline=c.to_data(self.run.deadline),
            root=c.dumps(self.request),state=self.state,index=self.index,
            command=None if self.command is None else c.dumps(self.command),
            budgets={k:list(v) for k,v in self._budgets.items()},started=c.to_data(self.started))
        receipt=self.checkpoints.save(self.run.envelope.run_id,key,value)
        c.require(type(receipt) is m.CommitReceipt and receipt.transaction_id==key and receipt.status is m.CommitStatus.COMMITTED,
                  'checkpoint','exact durable checkpoint required',c.ErrorCode.STORAGE_FAILURE)

    def _stop(self,error):
        already_stopping=self._cancelled
        self.state='unresolved';self.reason=c.Reason(error.code,error.field,error.detail);self._cancelled=True
        if self.request is not None and not already_stopping:
            try:
                now=self.clock.now()
                self._command_sequence=max(self._command_sequence,self.request.envelope.sequence)+1
                envelope=replace(self.request.envelope,producer=c.Owner.P08,event_id='p08-stop:'+uuid4().hex,
                    sequence=self._command_sequence,emitted_at=now,
                    expires_at=replace(now,seconds=now.seconds+self.config.certificate_ttl_s))
                report=(self.stop_sink or self.device.stop)(m.StopRequest(envelope,self.reason,self.holding))
                if type(report) is m.StopReport:self.stop_report=report
            except Exception:self.stop_report=None  # Stop failure is never physical stopped.

    def start(self,request):
        fingerprint=hashlib.sha256(c.dumps(request).encode('utf-8')).hexdigest();key=request.envelope.action_id
        if key in self._fingerprints:
            c.require(self._fingerprints[key]==fingerprint,'action_id','same action, different payload',c.ErrorCode.ID_CONFLICT)
            return self.state
        if self.state=='recovery_required':return self.state
        c.require(self.state=='idle','action','one placement at a time; release transaction must finish before next action')
        try:
            ctx=self._context();r.validate_envelope(request.envelope,ctx)
            c.require(request.holding.state is c.Holding.HELD,'holding','known held object required')
            b=request.order.budget
            c.require(request.order.destination not in self._area_names or self._area_names[request.order.destination]==b.name,
                      'budget.name','cannot rename regional budget')
            c.require(b.name not in self._limits or self._limits[b.name]==(b.attempts_limit,b.seconds_limit),
                      'budget.limits','cannot change cumulative budget limits')
            old=self._budgets.get(b.name,(0,0.))
            used_attempts=max(old[0],b.attempts_used);used_seconds=max(old[1],b.seconds_used)
            c.require(b.attempts_limit is not None and b.seconds_limit is not None and b.attempts_used<b.attempts_limit and
                      used_attempts<b.attempts_limit and used_seconds<b.seconds_limit,'budget','explicit remaining attempts/time required',c.ErrorCode.LIMIT_EXCEEDED)
            self.request=request;self.holding=request.holding;self.started=ctx.now
            self._budgets[b.name]=(used_attempts+1,used_seconds);self._limits[b.name]=(b.attempts_limit,b.seconds_limit)
            self._area_names[request.order.destination]=b.name
            self._fingerprints[key]=fingerprint;self.state='ready';self._save();self._context()
        except c.ContractError as error:self._stop(error);return self.state
        except Exception as error:self._stop(c.ContractError(c.ErrorCode.STORAGE_FAILURE,'checkpoint',str(error)));return self.state
        return self.tick()

    def tick(self):
        if self.state not in ('ready','waiting') or self._busy:return self.state
        self._busy=True
        try:
            ctx=self._context()
            if self.state=='waiting':
                c.require(ctx.now-self.sent_at<self.timeout,'timeout','stage completion unknown',c.ErrorCode.TIMEOUT)
                return self.state
            if self.index==len(self.STAGES):self.state='await_release';self._save();return self.state
            phase=self.STAGES[self.index]
            stop=self.stop_provider();ctx=self._context();guards.stopped(stop,ctx,self.config)
            self._command_sequence=max(self._command_sequence,self.request.envelope.sequence)+1
            e=replace(self.request.envelope,producer=c.Owner.P08,action_id=self.request.envelope.action_id+':place:'+str(self.index),
                      event_id=self.request.envelope.event_id+':place:'+str(self.index),emitted_at=ctx.now,
                      sequence=self._command_sequence)
            plan=None
            if phase is not None:
                plan=self.stage_planner(phase,e,self.request.order,self.holding);ctx=self._context()
                c.require(type(plan) is m.GeometryPlan,'stage_plan','P06 independently certified one-stage plan missing',c.ErrorCode.UNAVAILABLE)
                r.validate_envelope(plan.envelope,replace(ctx,action_id=e.action_id))
                c.require(plan.envelope.producer is c.Owner.P06 and len(plan.segments)==1 and plan.segments[0].phase is phase and
                          plan.reachable is c.Fact.CONFIRMED and plan.collision_free is c.Fact.CONFIRMED,
                          'stage_plan','not a certified current single stage')
                guards.physical(plan.evidence,e,ctx.now,self.config)
                self.timeout=plan.segments[0].timeout_s
            else:
                # Pump acknowledgement is followed by physical release observation;
                # it never proves separation. Bound its wait by the PLACE timeout.
                self.timeout=next(s.timeout_s for s in self.request.plan.segments if s.phase is m.Phase.PLACE)
            c.require(self.run.deadline-ctx.now>self.timeout,'deadline','not enough time for bounded stage',c.ErrorCode.DEADLINE)
            budget=self.request.order.budget
            c.require(budget.seconds_limit-self._budgets[budget.name][1]-(ctx.now-self.started)>self.timeout,
                      'budget','not enough remaining regional time for bounded stage',c.ErrorCode.LIMIT_EXCEEDED)
            intent=m.ActionIntent(e,m.IntentOperation.PUMP if phase is None else m.IntentOperation.ARM,self.request.order,plan,None,None,False if phase is None else None)
            receipt=self.persistence.commit_intent(intent);ctx=self._context();r.validate_intent_receipt(intent,receipt)
            permit=self.authorize(intent,receipt);ctx=self._context()
            c.require(type(permit) is m.ControlPermit,'permit','missing authorization',c.ErrorCode.NOT_AUTHORIZED)
            r.validate_envelope(permit.envelope,replace(ctx,action_id=e.action_id))
            c.require(permit.purpose=='place' and permit.source is (m.MotionSource.PUMP if phase is None else m.MotionSource.ARM),
                      'permit','wrong phase/source/purpose',c.ErrorCode.NOT_AUTHORIZED)
            if phase is None:command=m.PumpCommand(e,permit,receipt.commit.transaction_id,False,m.Phase.PLACE,self.holding)
            else:
                seg=plan.segments[0]
                command=m.ArmCommand(e,permit,receipt.commit.transaction_id,phase,seg.target,seg.speed_mm_s,'mm/s',seg.timeout_s,plan.evidence+stop.evidence)
            self.prepare(command,intent,receipt);self._context()
            self.command=command;self.sent_at=self.clock.now();self.state='waiting'
            self._save();ctx=self._context()
            # Re-check after all arbitrary callbacks, immediately before output.
            r.validate_envelope(permit.envelope,replace(ctx,action_id=e.action_id));guards.stopped(stop,ctx,self.config)
            if plan is not None:
                r.validate_envelope(plan.envelope,replace(ctx,action_id=e.action_id))
                guards.physical(plan.evidence,e,ctx.now,self.config)
            result=self.device.submit_pump(command) if phase is None else self.device.submit_arm(command)
            ctx=self._context()
            r.validate_envelope(result.envelope,replace(ctx,action_id=e.action_id))
            c.require(result.result.submitted is c.Fact.CONFIRMED,'submit','device did not confirm submission',c.ErrorCode.UNAVAILABLE)
        except c.ContractError as error:self._stop(error)
        except Exception as error:self._stop(c.ContractError(c.ErrorCode.UNAVAILABLE,'adapter',str(error)))
        finally:self._busy=False
        return self.state

    def confirm(self,event):
        key=(event.envelope.run_id,event.envelope.producer,event.envelope.event_id)
        fingerprint=c.dumps(event)
        if key in self._feedback:
            c.require(self._feedback[key]==fingerprint,'event_id','feedback identity reused with different payload',c.ErrorCode.ID_CONFLICT)
            return True
        if self.state!='waiting':return False
        try:
            ctx=self._context()
            r.validate_envelope(event.envelope,replace(ctx,action_id=self.command.envelope.action_id))
            c.require(event.envelope.producer is c.Owner.P01,'feedback','P01 stage feedback required')
            c.require(ctx.now-self.sent_at<self.timeout,'timeout','late physical result',c.ErrorCode.TIMEOUT)
            if event.result.status is not c.Outcome.SUCCESS or event.result.physical is not c.Fact.CONFIRMED:return False
            guards.physical(event.result.evidence,self.command.envelope,ctx.now,self.config)
            c.require(all(ev.captured_at-self.sent_at>0 for ev in event.result.evidence),'feedback','pre-command frame cannot confirm completion')
            receipt=self.persistence.append_event(event);self._context()
            c.require(receipt.status is m.CommitStatus.COMMITTED,'event','completion journal not committed',c.ErrorCode.STORAGE_FAILURE)
            self.completed_phases.append(self.STAGES[self.index]);self.index+=1;self.state='ready';self._save();self._context()
            self._feedback[key]=fingerprint
            return True
        except c.ContractError as error:self._stop(error);return False
        except Exception as error:self._stop(c.ContractError(c.ErrorCode.STORAGE_FAILURE,'feedback',str(error)));return False

    def cancel(self,detail):
        self._stop(c.ContractError(c.ErrorCode.CANCELLED,'cancel',detail))
        return self.state

    def finish_committed(self,receipt,snapshot):
        c.require(self.state=='await_release' and self.request is not None,'state','all physical placement stages must complete')
        c.require(receipt.status is m.CommitStatus.COMMITTED and receipt.revision==snapshot.revision and
                  receipt.space_version==snapshot.space.version and snapshot.holding.state is c.Holding.EMPTY,
                  'commit','only linked committed empty state ends cycle',c.ErrorCode.STORAGE_FAILURE)
        order=self.request.order
        c.require(snapshot.run.started_at==self.run.started_at and snapshot.run.deadline==self.run.deadline and
                  snapshot.run.envelope.run_id==self.run.envelope.run_id and
                  any(d.key==m.DeliveryKey(self.run.envelope.run_id,order.inventory_id) and d.area is order.destination for d in snapshot.deliveries),
                  'commit','original run and linked delivery required')
        used=self._budgets[order.budget.name]
        self._budgets[order.budget.name]=(used[0],used[1]+(self.clock.now()-self.started))
        self.state='cycle_committed';self._save()
        self.request=None;self.command=None;self.started=None;self.index=0;self.completed_phases=[];self.holding=snapshot.holding;self.state='idle'
