"""Serial P07 offline state machine implementing the C0 ManipulationPort.

Integration hooks are P07-local, not frozen public ports. No default hooks,
hardware adapter, timer thread, or automatic recovery motion is supplied.
P10A/P01 independent stop supervision must never wait for this business loop.
"""
from dataclasses import replace
import math
from uuid import uuid4
from treasure_auto.contracts import core as c, models as m, rules as r
from .config import ManipulationConfig
from .evidence import MetricObservation, Observer, offset
from . import guards


STAGES=('approach','descend','contact','pump','trial_lift','withdraw','transport')


def reason(error):
    if isinstance(error,c.ContractError):
        return c.Reason(error.code,error.field,error.detail)
    return c.Reason(c.ErrorCode.UNAVAILABLE,'dependency',str(error) or type(error).__name__)


class ManipulationService:
    def __init__(self,config,run,clock,*,context_provider,capture_now,stop_provider,
                 stage_planner,authorize,prepare,device,persistence,checkpoints,stop_sink=None):
        c.require(type(config) is ManipulationConfig,'config','validated config required')
        c.require(run.envelope.origin is c.Origin.SIMULATED,'origin',
                  'only offline simulated execution delivered',c.ErrorCode.NOT_AUTHORIZED)
        c.require(config.config_version==run.envelope.versions.config,'config_version','config differs from run')
        self.config,self.run,self.clock=config,run,clock
        self.context_provider,self.capture_now=context_provider,capture_now
        self.stop_provider,self.stage_planner=stop_provider,stage_planner
        self.authorize,self.prepare,self.device=authorize,prepare,device
        self.persistence,self.checkpoints=persistence,checkpoints
        self.stop_sink=stop_sink
        self.observer=Observer(config)
        self.holding=c.HoldingSnapshot(c.Holding.UNKNOWN,None,None,())
        self.readiness=None
        self.recovery_request=None
        self.drop_region=None
        self.pending_update=None
        self.pending_release=None
        self.pump_required=False
        self.stop_report=None
        self.reconciliation_required=False
        self._records,self._budgets={},{}
        self._request=None; self._index=0; self._sequence=0; self._save_sequence=0
        self._terminal=False; self._contact=False; self._stage_done=False; self._sent=False
        self._closed=False; self._completed_refs=()
        self._ever_held=False
        self._empty_session=None
        self._status=c.Outcome.UNKNOWN
        self._reason=c.Reason(c.ErrorCode.EVIDENCE_MISSING,'stage','waiting for observation')
        self._submitted=c.Fact.UNKNOWN; self._echoed=c.Fact.UNKNOWN
        self._plan=None; self._command=None; self._baseline=None
        self._floor=None; self._started=None; self._attempt_started=None
        self._last_now=clock.now()
        self._recovery_floor=capture_now()
        self._history_valid=True
        self._load()
        if self.reconciliation_required:
            self._stop(c.Reason(c.ErrorCode.NOT_AUTHORIZED,'recovery','old commands and scene need reconciliation'))

    @property
    def stage(self):
        return STAGES[self._index] if self._request else 'idle'

    @property
    def active_action_id(self):
        return self._command.envelope.action_id if self._command else None

    @property
    def current_target(self):
        return self._plan.segments[0].target if self._plan else None

    def _now(self):
        now=self.clock.now()
        c.require(now-self._last_now>=0,'clock','clock rollback',c.ErrorCode.CLOCK_MISMATCH)
        c.require(now-self.run.started_at>=0,'clock','clock before original start',c.ErrorCode.CLOCK_MISMATCH)
        self._last_now=now
        return now

    def _context(self):
        before=self._now(); ctx=self.context_provider(); now=self._now()
        e=self.run.envelope
        c.require((ctx.run_id,ctx.instance_id,ctx.control_epoch,ctx.origin)==
                  (e.run_id,e.instance_id,e.control_epoch,e.origin),'context','old run/control instance',c.ErrorCode.STALE_CONTROL)
        # A provider samples between two clock reads, not at an identical float.
        # Subtraction also rejects a foreign clock/boot or a clock rollback.
        c.require(ctx.now-before>=0 and now-ctx.now>=0,'context.now',
                  'context time must be sampled during this call',c.ErrorCode.CLOCK_MISMATCH)
        return replace(ctx,now=now)

    def _env(self,base=None,**changes):
        base=base or (self._request.envelope if self._request else self.run.envelope)
        now=self._last_now; self._sequence+=1
        values=dict(producer=c.Owner.P07,event_id='p07:'+uuid4().hex,sequence=self._sequence,
                    emitted_at=now,expires_at=replace(now,seconds=now.seconds+self.config.result_ttl_s))
        values.update(changes)
        return replace(base,**values)

    @property
    def result(self):
        env=self._env(self._command.envelope if self._command else None)
        evidence=tuple(ev for ev in self.holding.evidence if ev.action_id==env.action_id and
                       ev.target_id==env.target_id and ev.valid_until-self._last_now>0)
        elapsed=max(0.,self._last_now-self._attempt_started) if self._attempt_started else 0.
        result=c.ActionResult(self._status,self.stage,None if self._status is c.Outcome.SUCCESS else self._reason,
              c.Fact.CONFIRMED if self._request else c.Fact.UNKNOWN,self._submitted,self._echoed,
              c.Fact.CONFIRMED if self._status is c.Outcome.SUCCESS and evidence else c.Fact.UNKNOWN,evidence,elapsed)
        return m.ManipulationResult(env,result,self.holding,self.readiness,None)

    def _rejected(self,req,error,status=c.Outcome.REJECTED):
        result=c.ActionResult(status,'admission',reason(error),c.Fact.REJECTED,c.Fact.REJECTED,
                              c.Fact.UNKNOWN,c.Fact.UNKNOWN,(),0.)
        return m.ManipulationResult(self._env(req.envelope),result,self.holding,None,None)

    def _load(self):
        try:
            state=self.checkpoints.load(self.run.envelope.run_id)
            if state is None:
                return
            c.require(type(state) is dict and state.get('format')=='p07-local-1','checkpoint','unknown checkpoint format',c.ErrorCode.STORAGE_FAILURE)
            c.require(state['run_id']==self.run.envelope.run_id and c.loads(state['start'])==self.run.started_at and
                      c.loads(state['deadline'])==self.run.deadline,'checkpoint','original run/time identity differs',c.ErrorCode.CLOCK_MISMATCH)
            self._records=state['records']; self._budgets=state['budgets']
            c.require(type(self._records) is dict and type(self._budgets) is dict,'checkpoint','invalid records')
            for entry in self._records.values():
                c.require(type(c.loads(entry['request'])) is m.GraspRequest and type(c.loads(entry['result'])) is m.ManipulationResult,
                          'checkpoint','invalid C0 payload')
            counts={}
            for action_id,entry in self._records.items():
                req=c.loads(entry['request']); inventory=req.order.inventory_id
                c.require(action_id==req.envelope.action_id and inventory in self._budgets,
                          'checkpoint.budget','action history without cumulative budget',c.ErrorCode.STORAGE_FAILURE)
                b=self._budgets[inventory]
                c.require((b['target'],b['request_id'],b['attempts_limit'],b['seconds_limit'])==
                          (req.envelope.target_id,req.envelope.request_id,req.order.budget.attempts_limit,req.order.budget.seconds_limit),
                          'checkpoint.budget','budget identity or limits disagree with action history')
                counts[inventory]=counts.get(inventory,0)+1
            for b in self._budgets.values():
                c.require(type(b['attempts']) is int and b['attempts']>=0 and type(b['seconds']) in (float,int) and
                          math.isfinite(b['seconds']) and b['seconds']>=0 and
                          type(b['attempts_limit']) is int and b['attempts_limit']>0 and
                          type(b['seconds_limit']) in (float,int) and math.isfinite(b['seconds_limit']) and b['seconds_limit']>0,
                          'checkpoint.budget','invalid cumulative budget')
            c.require(all(self._budgets[k]['attempts']>=count for k,count in counts.items()),
                      'checkpoint.budget','attempt count below recorded actions',c.ErrorCode.STORAGE_FAILURE)
            old=c.loads(state['holding'])
            c.require(type(old) is c.HoldingSnapshot,'checkpoint.holding','invalid holding')
            self.holding=replace(old,state=c.Holding.DROPPED if old.state is c.Holding.DROPPED else c.Holding.UNKNOWN)
            self.drop_region=c.loads(state['drop_region']) if state.get('drop_region') else None
            c.require(type(state['ever_held']) is bool and type(state['pump_required']) is bool,
                      'checkpoint','boolean lock facts required')
            self._ever_held=state['ever_held'] or old.state in (c.Holding.HELD,c.Holding.DROPPED)
            self.pump_required=state['pump_required'] or old.state is not c.Holding.EMPTY
            self.reconciliation_required=True
            self._save_sequence=state['save_sequence']
            # Active attempt includes real downtime; it is never silently reset.
            active=state.get('active')
            if active:
                start=c.loads(active['started'])
                elapsed=self._last_now-start
                c.require(elapsed>=0,'checkpoint.started','clock continuity lost',c.ErrorCode.CLOCK_MISMATCH)
                b=self._budgets[active['budget_key']]
                b['seconds']+=elapsed
        except Exception as error:
            self._history_valid=False
            self.reconciliation_required=True
            self._reason=reason(error)
            self.holding=c.HoldingSnapshot(c.Holding.UNKNOWN,None,None,())
            self.pump_required=True

    def _save(self):
        self._save_sequence+=1
        if self._request:
            self._records[self._request.envelope.action_id]['result']=c.dumps(self.result)
        active=None
        if self._request and not self._terminal:
            active=dict(started=c.dumps(self._attempt_started),budget_key=self._request.order.inventory_id)
        state=dict(format='p07-local-1',run_id=self.run.envelope.run_id,start=c.dumps(self.run.started_at),
                   deadline=c.dumps(self.run.deadline),records=self._records,budgets=self._budgets,
                   holding=c.dumps(self.holding),pump_required=self.pump_required,
                   save_sequence=self._save_sequence,active=active,
                   drop_region=c.dumps(self.drop_region) if self.drop_region else None,ever_held=self._ever_held)
        tx='p07:checkpoint:{}:{}'.format(self.run.envelope.run_id,self._save_sequence)
        try:
            receipt=self.checkpoints.save(self.run.envelope.run_id,tx,state)
            c.require(type(receipt) is m.CommitReceipt and receipt.status is m.CommitStatus.COMMITTED and receipt.transaction_id==tx,
                      'checkpoint','unconfirmed checkpoint; reconcile before any further action',c.ErrorCode.STORAGE_FAILURE)
        except Exception as error:
            self.reconciliation_required=True; self._history_valid=False; self._closed=True
            self.readiness=None
            raise c.ContractError(c.ErrorCode.STORAGE_FAILURE,'checkpoint',str(error)) from error

    def _budget(self):
        return self._budgets[self._request.order.inventory_id]

    def _guard_active(self):
        c.require(not self._terminal and not self.reconciliation_required,'state','motion closed',c.ErrorCode.NOT_AUTHORIZED)
        ctx=self._context()
        c.require(self.run.deadline-ctx.now>0,'deadline','original deadline reached',c.ErrorCode.DEADLINE)
        report=self.stop_provider()
        # An external stop query may block or deliver cancellation reentrantly.
        # Never authorize dispatch using the pre-query timestamp or latch state.
        ctx=self._context()
        c.require(not self._terminal and not self._closed and not self.reconciliation_required,
                  'state','motion closed during stop query',c.ErrorCode.NOT_AUTHORIZED)
        c.require(self.run.deadline-ctx.now>0,'deadline','original deadline reached',c.ErrorCode.DEADLINE)
        r.validate_envelope(self._request.envelope,guards.context_for(ctx,self._request.envelope))
        b=self._budget()
        c.require(b['seconds']+(ctx.now-self._attempt_started)<b['seconds_limit'],
                  'budget','cumulative time exhausted',c.ErrorCode.LIMIT_EXCEEDED)
        guards.stopped(report,ctx,firmware=True)
        return ctx

    def grasp(self,request):
        try:
            self._now()
            key=request.envelope.action_id
            known=self._records.get(key)
            if known:
                c.require(known['request']==c.dumps(request),'action_id','same action id with different content',c.ErrorCode.ID_CONFLICT)
                if self._request and self._request.envelope.action_id==key:
                    return self.result
                historical=c.loads(known['result'])
                return replace(historical,holding=self.holding,readiness=None)
            c.require(not self.reconciliation_required,'recovery','site/old queues need explicit reconciliation',c.ErrorCode.NOT_AUTHORIZED)
            c.require(self._request is None or self._terminal,'work_order','one active work order only',c.ErrorCode.CONFLICT)
            ctx=self._context()
            c.require(self.run.deadline-ctx.now>0,'deadline','original deadline reached',c.ErrorCode.DEADLINE)
            guards.request(request,ctx,self.holding)
            source=request.order.budget; key=request.order.inventory_id
            c.require(all(value['target']!=request.envelope.target_id or inventory==key
                          for inventory,value in self._budgets.items()),'inventory_id',
                      'same physical target cannot acquire a fresh retry budget',c.ErrorCode.CONFLICT)
            b=self._budgets.get(key)
            if b:
                c.require((b['target'],b['request_id'],b['attempts_limit'],b['seconds_limit'])==
                          (request.envelope.target_id,request.envelope.request_id,source.attempts_limit,source.seconds_limit),
                          'budget','target/work/limits changed across retry',c.ErrorCode.CONFLICT)
                c.require(source.attempts_used>=b['attempts'] and source.seconds_used>=b['seconds'],
                          'budget','cumulative budget rolled back',c.ErrorCode.STALE_RECOVERY)
            else:
                b=dict(target=request.envelope.target_id,request_id=request.envelope.request_id,
                       attempts_limit=source.attempts_limit,seconds_limit=source.seconds_limit,
                       attempts=0,seconds=0.)
            b['attempts']=max(b['attempts'],source.attempts_used)
            b['seconds']=max(b['seconds'],source.seconds_used)
            c.require(b['attempts']<b['attempts_limit'] and b['seconds']<b['seconds_limit'],
                      'budget','same-target budget exhausted; hand to mission',c.ErrorCode.LIMIT_EXCEEDED)
            self._budgets[key]=b; b['attempts']+=1
            self._request=request; self._index=0; self._terminal=False; self._stage_done=False
            self._contact=False; self._sent=False; self._baseline=None; self._command=None
            self._closed=False; self._completed_refs=()
            self._submitted=c.Fact.UNKNOWN; self._echoed=c.Fact.UNKNOWN
            self._attempt_started=ctx.now; self._status=c.Outcome.UNKNOWN
            self._reason=c.Reason(c.ErrorCode.EVIDENCE_MISSING,'stage','awaiting physical observation')
            self._records[request.envelope.action_id]=dict(request=c.dumps(request),result=c.dumps(self.result))
            self._save()
            self._dispatch()
            return self.result
        except Exception as error:
            if self._request is request:
                self._finish(c.Outcome.UNKNOWN,reason(error))
                return self.result
            status=c.Outcome.FAILED if reason(error).code is c.ErrorCode.LIMIT_EXCEEDED else c.Outcome.REJECTED
            return self._rejected(request,error,status)

    def _dispatch(self):
        self._submitted=c.Fact.UNKNOWN; self._echoed=c.Fact.UNKNOWN; self._command=None
        ctx=self._guard_active()
        phase=m.Phase.CONTACT if self.stage=='pump' else m.Phase(self.stage)
        action_id=self._request.envelope.action_id+':'+str(self._index)
        env=self._env(action_id=action_id,expires_at=self._request.envelope.expires_at)
        expected=guards.context_for(ctx,env)
        plan=None
        if self.stage!='pump':
            plan=self.stage_planner(phase,env,self._request.order,self.holding)
            ctx=self._guard_active(); expected=guards.context_for(ctx,env)
            guards.plan(plan,expected,phase)
            timeout=plan.segments[0].timeout_s
        else:
            timeout=self.config.pump_timeout_s
        c.require(self.run.deadline-ctx.now>timeout,'deadline','insufficient time for bounded stage',c.ErrorCode.DEADLINE)
        c.require(self._budget()['seconds']+(ctx.now-self._attempt_started)+timeout<=self._budget()['seconds_limit'],
                  'budget','insufficient cumulative stage time',c.ErrorCode.LIMIT_EXCEEDED)
        intent=m.ActionIntent(env,m.IntentOperation.PUMP if self.stage=='pump' else m.IntentOperation.ARM,
                              self._request.order,plan,None,None,True if self.stage=='pump' else None)
        receipt=self.persistence.commit_intent(intent)
        r.validate_intent_receipt(intent,receipt)
        self._guard_active()
        permit=self.authorize(intent,receipt,self.holding)
        ctx=self._guard_active(); expected=guards.context_for(ctx,env)
        source=m.MotionSource.PUMP if self.stage=='pump' else m.MotionSource.ARM
        guards.permit(permit,expected,source)
        if plan:
            guards.plan(plan,expected,phase)
            segment=plan.segments[0]
            command=m.ArmCommand(env,permit,receipt.commit.transaction_id,phase,segment.target,
                                  segment.speed_mm_s,'mm/s',segment.timeout_s,plan.evidence)
        else:
            command=m.PumpCommand(env,permit,receipt.commit.transaction_id,True,phase,self.holding)
        self.prepare(command,intent,receipt)
        ctx=self._guard_active(); expected=guards.context_for(ctx,env)
        guards.permit(permit,expected,source)
        if plan:
            guards.plan(plan,expected,phase)
        self._plan=plan if plan else self._plan
        self._command=command; self._started=self._last_now; self._timeout=timeout
        self._floor=self.capture_now(); self._stage_done=False; self.observer.reset()
        self._completed_refs=()
        self._contact=self._contact or self.stage in STAGES[1:]
        if self._contact:
            self.pump_required=True
            self.holding=replace(self.holding,state=c.Holding.UNKNOWN,target_id=self._request.envelope.target_id)
        self._sent=True  # Persist a may-have-sent barrier BEFORE touching the gateway.
        self._save()
        ctx=self._guard_active(); expected=guards.context_for(ctx,env)
        guards.permit(permit,expected,source)
        if plan:
            guards.plan(plan,expected,phase)
        event=self.device.submit_pump(command) if self.stage=='pump' else self.device.submit_arm(command)
        c.require(type(event) is m.ActionEvent and event.envelope.producer is c.Owner.P01,'device','P01 ActionEvent required')
        r.validate_envelope(event.envelope,guards.context_for(self._context(),env))
        self._submitted,self._echoed=event.result.submitted,event.result.echoed
        committed=self.persistence.append_event(event)
        c.require(committed.status is m.CommitStatus.COMMITTED,'event','device result not durably recorded',c.ErrorCode.STORAGE_FAILURE)
        if event.result.accepted is c.Fact.REJECTED or event.result.status in (c.Outcome.FAILED,c.Outcome.CANCELLED,c.Outcome.TIMED_OUT,c.Outcome.REJECTED):
            self._finish(event.result.status,event.result.reason)
        self._save()

    def _stop(self,why):
        self.readiness=None
        try:
            self._now()
            request=m.StopRequest(self._env(),why,self.holding)
            self.stop_report=(self.stop_sink or self.device.stop)(request)
        except Exception:
            self.stop_report=None  # stop REQUESTED is never silently upgraded to stopped.

    def _journal(self,event):
        try:
            receipt=self.persistence.append_event(event)
            c.require(type(receipt) is m.CommitReceipt and receipt.status is m.CommitStatus.COMMITTED,
                      'event','phase observation/termination journal not committed',c.ErrorCode.STORAGE_FAILURE)
        except Exception as error:
            self.reconciliation_required=True; self._history_valid=False; self._closed=True
            self.readiness=None
            raise c.ContractError(c.ErrorCode.STORAGE_FAILURE,'event',str(error)) from error

    def _finish(self,status,why):
        if self._terminal:
            return
        self._terminal=True; self._status=status; self._reason=why
        self._closed=True
        if self._contact:
            self.pump_required=True
            if self.holding.state not in (c.Holding.HELD,c.Holding.DROPPED):
                self.holding=replace(self.holding,state=c.Holding.UNKNOWN)
        self._stop(why)  # never wait for checkpoint/DB before requesting stop
        if self._request and self._attempt_started:
            self._budget()['seconds']+=max(0.,self._last_now-self._attempt_started)
        try:
            result=self.result
            self._journal(m.ActionEvent(result.envelope,result.result))
            self._save()
        except Exception as error:
            self.reconciliation_required=True
            self._reason=reason(error)

    def cancel(self,request):
        try:
            ctx=self._context()
            r.validate_envelope(request.envelope,replace(ctx,request_id=None,action_id=None,target_id=None))
            c.require(request.envelope.producer in (c.Owner.P09,c.Owner.P10A),'stop.producer','unauthorized stop origin')
        except Exception as error:
            return self._rejected(request,error)
        self._closed=True
        if self._terminal:
            self._stop(request.reason)
        else:
            self._finish(c.Outcome.CANCELLED,request.reason)
        return self.result

    def tick(self):
        try:
            now=self._now()
            if self.run.deadline-now<=0:
                if not self._terminal:
                    self._finish(c.Outcome.TIMED_OUT,c.Reason(c.ErrorCode.DEADLINE,'deadline','original deadline reached'))
                else:
                    self._stop(c.Reason(c.ErrorCode.DEADLINE,'deadline','original deadline reached'))
                return self.result
            if self.holding.state is c.Holding.HELD:
                if not self.holding.evidence or any(ev.valid_until-now<=0 or now-ev.received_at>self.config.evidence_age_s for ev in self.holding.evidence):
                    self._unknown(c.Reason(c.ErrorCode.EXPIRED,'holding','holding evidence expired'))
            if self._request and not self._terminal:
                self._guard_active()
                if now-self._started>=self._timeout:
                    self._finish(c.Outcome.TIMED_OUT,c.Reason(c.ErrorCode.TIMEOUT,'stage','physical completion timed out'))
                elif self._stage_done:
                    if not self._completed_refs or any(ev.valid_until-now<=0 or now-ev.received_at>self.config.evidence_age_s
                                                       for ev in self._completed_refs):
                        self._stage_done=False; self.observer.reset()
                        self._stop(c.Reason(c.ErrorCode.EXPIRED,'stage','completion evidence expired before next stage'))
                        return self.result
                    if self.stage=='transport':
                        self._terminal=True; self._status=c.Outcome.SUCCESS; self._reason=None
                        self._budget()['seconds']+=now-self._attempt_started
                        self._save()
                    else:
                        self._index+=1
                        self._dispatch()
        except Exception as error:
            if not self._terminal:
                self._finish(c.Outcome.UNKNOWN,reason(error))
            else:
                self._unknown(reason(error))
        return self.result

    def _unknown(self,why):
        if self.holding.state is not c.Holding.DROPPED:
            self.holding=replace(self.holding,state=c.Holding.UNKNOWN)
        self.recovery_request='bounded_arm_observation'
        self._stage_done=False
        self._stop(why)
        self.observer.reset()

    def observe_holding(self,observation):
        # C0 v1 has no lift/support, explicit empty, or posture measurements.
        if self._request:
            self._unknown(c.Reason(c.ErrorCode.EVIDENCE_MISSING,'tool','C0 ToolObservation lacks metric holding proof'))
        return self.holding

    def observe(self,sample):
        try:
            ctx=self._context()
            if self.run.deadline-ctx.now<=0:
                self._finish(c.Outcome.TIMED_OUT,c.Reason(c.ErrorCode.DEADLINE,'deadline','original deadline reached'))
                self.readiness=None
                return self.holding
            if self._request and not self._terminal and self._started and ctx.now-self._started>=self._timeout:
                self._finish(c.Outcome.TIMED_OUT,c.Reason(c.ErrorCode.TIMEOUT,'stage','physical completion timed out'))
                return self.holding
            if self._request:
                e=self._command.envelope if self._command else self._request.envelope
                ctx=guards.context_for(ctx,e)
            stamp=self.observer.validate(sample,ctx,self.capture_now(),self._floor)
            if sample.dropped is c.Fact.CONFIRMED:
                c.require(sample.drop_region is not None and self._request is not None,'drop_region','record possible fall range before confirmation')
                self.drop_region=sample.drop_region
                self.holding=c.HoldingSnapshot(c.Holding.DROPPED,self._request.envelope.target_id,None,sample.tool.evidence)
                self._unknown(c.Reason(c.ErrorCode.OCCUPANCY_UNRESOLVED,'drop','confirmed dropped; range retained'))
                if not self._terminal:
                    self._finish(c.Outcome.FAILED,c.Reason(c.ErrorCode.OCCUPANCY_UNRESOLVED,'drop','drop unresolved'))
                else:
                    self._closed=True
                    self._save()
                return self.holding
            if sample.empty is c.Fact.CONFIRMED and sample.tool_visible is c.Fact.CONFIRMED:
                empty=c.HoldingSnapshot(c.Holding.EMPTY,None,None,sample.tool.evidence)
                if self._request is None and not self.reconciliation_required:
                    self.holding=empty
                elif self.holding.state is c.Holding.DROPPED:
                    self.pending_update=m.HoldingUpdate(self._env(),c.Holding.DROPPED,empty,m.HoldingCause.DROP_RECONCILED)
                elif self._request:
                    self._unknown(c.Reason(c.ErrorCode.OCCUPANCY_UNRESOLVED,'empty','unexpected empty view; reconcile target before unlocking'))
                return self.holding
            if self._closed or (self._terminal and self._status is not c.Outcome.SUCCESS):
                return self.holding  # late evidence cannot revive cancelled/timed-out action
            if self._request is None:
                return self.holding
            target=self._request.envelope.target_id
            valid=self.observer.at_target(sample,self.current_target)
            if self.stage in ('contact','pump'):
                valid=valid and self.observer.supported(sample,target)
            elif self.stage=='trial_lift':
                valid=valid and self.observer.lifted(sample,self._baseline,target)
            elif self.stage in ('withdraw','transport'):
                expected=offset(self._baseline) if self._baseline else None
                valid=valid and self.observer.held(sample,expected,target)
                if self.stage=='transport':
                    valid=valid and sample.posture_id==self.config.transport_pose_id
            if not valid:
                self._stage_done=False
                if self.holding.state is c.Holding.HELD:
                    self._unknown(c.Reason(c.ErrorCode.EVIDENCE_MISSING,'holding','new observation cannot confirm holding/posture'))
            if self.observer.window_passes(sample,stamp,valid):
                completion=c.ActionResult(c.Outcome.SUCCESS,'contact_observation' if self.stage=='pump' else self.stage,
                    None,c.Fact.CONFIRMED,self._submitted,self._echoed,c.Fact.CONFIRMED,
                    self.observer.refs(),max(0.,ctx.now-self._started))
                self._journal(m.ActionEvent(self._env(sample.tool.envelope),completion))
                ctx=self._context()
                c.require(not self._closed,'state','stopped while recording observation',c.ErrorCode.NOT_AUTHORIZED)
                c.require(self.run.deadline-ctx.now>0,'deadline','deadline while recording observation',c.ErrorCode.DEADLINE)
                if not self._terminal:
                    c.require(ctx.now-self._started<self._timeout,'timeout','stage expired during observation commit',c.ErrorCode.TIMEOUT)
                if self.stage=='pump':
                    self._baseline=sample
                if self.stage in ('trial_lift','withdraw','transport'):
                    self.holding=c.HoldingSnapshot(c.Holding.HELD,target,offset(sample),self.observer.refs())
                    self._ever_held=True
                self._stage_done=True
                self._completed_refs=self.observer.refs()
                if self.stage=='transport':
                    if self.recovery_request is not None:
                        guards.stopped(self.stop_provider(),ctx,firmware=True)
                        self.recovery_request=None
                    refs=self.observer.refs()
                    expiry=self._evidence_expiry(refs)
                    self.readiness=m.DriveReadiness(self._env(sample.tool.envelope,expires_at=expiry),self.holding,c.Fact.CONFIRMED,refs)
                self._save()
        except Exception as error:
            self.observer.reset()
            if self._request and reason(error).code is not c.ErrorCode.DUPLICATE:
                self._unknown(reason(error))
        return self.holding

    def budget(self,inventory_id):
        """Cumulative debit for P09; attempts and elapsed downtime never reset."""
        b=self._budgets[inventory_id]
        elapsed=0.
        if self._request and not self._terminal and self._request.order.inventory_id==inventory_id:
            elapsed=self._now()-self._attempt_started
        return m.Budget('p07-grasp:'+inventory_id,b['attempts_limit'],b['seconds_limit'],b['attempts'],b['seconds']+elapsed)

    def _evidence_expiry(self,refs):
        now=self._last_now; capture_now=self.capture_now()
        remaining=min([self.run.deadline-now]+[min(ev.valid_until-now,
                      self.config.evidence_age_s-(now-ev.received_at),
                      self.config.evidence_age_s-(capture_now-ev.captured_at)) for ev in refs])
        c.require(remaining>0,'evidence','no current readiness interval',c.ErrorCode.EXPIRED)
        return replace(now,seconds=now.seconds+remaining)

    def reconcile(self,sample,stop):
        """Explicit post-stop empty-tool review; never resumes an old action.

        P10A must establish the current control epoch before constructing this
        service. Corrupt/unknown history needs storage repair, not an empty frame.
        Drop reconciliation instead requires a committed HoldingUpdate.
        """
        c.require(self._history_valid,'checkpoint','history unavailable; cannot reset budgets',c.ErrorCode.STORAGE_FAILURE)
        c.require(self._request is None or self._terminal,'recovery','stop old action first')
        c.require(self.holding.state is not c.Holding.DROPPED,'drop','drop needs transactional clearance',c.ErrorCode.OCCUPANCY_UNRESOLVED)
        c.require(not self._ever_held,'holding','previously held object requires release/drop transaction',c.ErrorCode.OCCUPANCY_UNRESOLVED)
        ctx=self._context()
        c.require(self.run.deadline-ctx.now>0,'deadline','cannot recover after original deadline',c.ErrorCode.DEADLINE)
        guards.stopped(stop,ctx,firmware=True)
        self.observer.validate(sample,replace(ctx,request_id=None,action_id=None,target_id=None),
                               self.capture_now(),self._recovery_floor)
        c.require(sample.empty is c.Fact.CONFIRMED and sample.tool_visible is c.Fact.CONFIRMED,
                  'empty','reliable complete empty tool view required',c.ErrorCode.EVIDENCE_MISSING)
        self.holding=c.HoldingSnapshot(c.Holding.EMPTY,None,None,sample.tool.evidence)
        self.pump_required=False; self.readiness=None; self._terminal=True
        self._stage_done=False; self.reconciliation_required=True; self._closed=False
        self._save()
        self.reconciliation_required=False
        return self.holding

    def confirm_empty_readiness(self,sample,plan,posture_id):
        """Observe an independently certified empty-arm posture; does not move.

        posture_id and the P06 WITHDRAW plan must come from verified configuration
        in the future integration adapter, never inferred from release or pump OFF.
        """
        ctx=self._context()
        c.require(not self.reconciliation_required and self.holding.state is c.Holding.EMPTY and self.pending_update is None,
                  'holding','committed/reconciled empty required')
        c.require(type(posture_id) is str and bool(posture_id.strip()),'posture_id','explicit empty-arm pose required')
        c.require(self.run.deadline-ctx.now>0,'deadline','no readiness after deadline',c.ErrorCode.DEADLINE)
        expected=guards.context_for(ctx,plan.envelope)
        guards.plan(plan,expected,m.Phase.WITHDRAW); guards.stopped(self.stop_provider(),ctx,firmware=True)
        session=(c.dumps(plan),posture_id)
        if self._empty_session!=session:
            self.observer.reset()
            self._empty_session=session
        stamp=self.observer.validate(sample,expected,self.capture_now(),plan.envelope.emitted_at)
        valid=(sample.empty is c.Fact.CONFIRMED and sample.tool_visible is c.Fact.CONFIRMED and
               sample.posture_id==posture_id and self.observer.at_target(sample,plan.target))
        self.readiness=None
        if self.observer.window_passes(sample,stamp,valid):
            refs=self.observer.refs()
            self.holding=c.HoldingSnapshot(c.Holding.EMPTY,None,None,refs)
            expiry=self._evidence_expiry(refs)
            self.readiness=m.DriveReadiness(self._env(sample.tool.envelope,expires_at=expiry),self.holding,c.Fact.CONFIRMED,refs)
        return self.readiness

    def propose_release(self,release,holding):
        ctx=self._context()
        c.require(self._request is not None and holding==self.holding and holding.state in (c.Holding.HELD,c.Holding.UNKNOWN),
                  'holding','release must reconcile current P07 object')
        c.require(release.envelope.producer is c.Owner.P08,'release.producer','only P08 release evidence')
        r.validate_envelope(release.envelope,guards.context_for(ctx,self._request.envelope,action=False))
        c.require(release.separated is c.Fact.CONFIRMED and release.correct_region is c.Fact.CONFIRMED,
                  'release','physical separation/correct region required',c.ErrorCode.EVIDENCE_MISSING)
        c.require(release.delivery_key.run_id==ctx.run_id and release.delivery_key.inventory_id==self._request.order.inventory_id and
                  release.category is self._request.order.category and release.area is self._request.order.destination and
                  release.mapping_version==self._request.order.mapping_version,'release','wrong work item/region/mapping')
        c.require(ctx.now-release.released_at>=0 and release.released_at-self.run.started_at>=0 and self.run.deadline-release.released_at>0,
                  'released_at','release time outside original run',c.ErrorCode.DEADLINE)
        c.require(release.released_at-self._attempt_started>=0,'released_at','release predates this grasp',c.ErrorCode.STALE_REQUEST)
        guards.physical(release.evidence,release.envelope,ctx)
        empty=c.HoldingSnapshot(c.Holding.EMPTY,None,None,release.evidence)
        self.pending_update=m.HoldingUpdate(self._env(release.envelope),holding.state,empty,m.HoldingCause.RELEASE)
        self.pending_release=release
        self.readiness=None
        self.observer.reset(); self._empty_session=None
        return self.pending_update

    def accept_holding_commit(self,receipt):
        """Apply only a verified P10B published state; never unlock on a proposal."""
        c.require(self.pending_update is not None,'holding_update','no pending change')
        c.require(type(receipt) is m.CommitReceipt and receipt.status is m.CommitStatus.COMMITTED,
                  'receipt','transaction not committed',c.ErrorCode.STORAGE_FAILURE)
        read=self.persistence.read(self.run.envelope.run_id)
        c.require(read.status is c.Outcome.SUCCESS and read.snapshot is not None,'snapshot','P10B state unavailable',c.ErrorCode.STORAGE_FAILURE)
        snap=read.snapshot
        c.require(snap.revision==receipt.revision and snap.space.version==receipt.space_version and
                  snap.holding==self.pending_update.current,'snapshot','commit/snapshot/holding mismatch',c.ErrorCode.CONFLICT)
        c.require(snap.run.started_at==self.run.started_at and snap.run.deadline==self.run.deadline and
                  snap.run.envelope.run_id==self.run.envelope.run_id,'snapshot','wrong original run')
        if self.pending_release:
            c.require(any(d.key==self.pending_release.delivery_key for d in snap.deliveries),
                      'deliveries','release and arrival must commit together',c.ErrorCode.CONFLICT)
        else:
            c.require(any(o.target_id==self.holding.target_id for o in snap.space.occupancies),
                      'drop','drop occupancy must remain recorded',c.ErrorCode.OCCUPANCY_UNRESOLVED)
        self.holding=snap.holding; self.pump_required=False; self.readiness=None
        self._ever_held=False
        self.observer.reset(); self._empty_session=None
        self.pending_update=None; self.pending_release=None
        self._save()
        return self.holding
