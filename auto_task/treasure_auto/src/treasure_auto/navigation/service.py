"""Nonblocking P03 offline state machine and explicit adapter hooks.

The service never starts ROS, publishes a goal, grants a permit, or writes a
score. P11 must consume effects and provide fresh C0 evidence. Velocity output
is possible only through the injected simulated P01 final gateway, after an
exact durable intent receipt and an externally installed P10A permit.
"""
import math
from dataclasses import dataclass, replace
from uuid import uuid4
from treasure_auto.contracts import core as c, models as m, rules as r
from treasure_auto.mapping.geometry import angle,covered
from .config import number
from .geometry import check_route
from .observations import (StopWindow,validate_env,fresh,validate_localization,
                           validate_readiness,reason)


@dataclass(frozen=True)
class NavigationEffect:
    kind: str
    envelope: c.Envelope
    payload: object


class NavigationService:
    """Caller serializes hooks; independent P01/P10A supervision is mandatory.

    ``navigate`` returns an accepted UNKNOWN while in progress. ``terminal``
    disambiguates UNKNOWN terminal send/stop failures. The current result never
    reopens expired work conditions, although the historical terminal is kept.
    """
    def __init__(self,run,clock,policy,profile,bundle,gateway,planner,restored_budget=None, *, space_readback=None):
        policy.validate()
        c.require(run.envelope.origin is c.Origin.SIMULATED and run.state is m.RunState.OFFLINE,
                  'run','only explicit offline simulation delivered',c.ErrorCode.NOT_AUTHORIZED)
        c.require(policy.config_version==run.envelope.versions.config and
                  bundle.metadata.map_version==run.envelope.versions.map and
                  bundle.metadata.measurement is m.Measurement.SIMULATED,'config','map/config mismatch')
        self.run,self.clock,self.policy,self.profile,self.bundle=run,clock,policy,profile,bundle
        self.gateway,self.planner=gateway,planner
        self.space_readback=space_readback
        self.active=None;self.phase='idle';self.terminal=False
        self.context=run.envelope;self.effects=[];self.goal_intent=None;self.route=()
        self._last_now=clock.now();self._sequence=0;self._cache={};self._events={}
        self._pending={};self._dispatched={};self._terminal_result=None
        self._progress_result=None;self._history={};self._geometry_after=None;self._work_invalid=False
        self._submitted=c.Fact.UNKNOWN
        self._stop=None;self._stopped=StopWindow(policy);self._stop_since=None;self._actual_stop_at=None;self._stop_input=None
        self._permit=None;self._feedback=None;self._residual=None
        self._began=None;self._stage_at=None;self._progress_pose=None;self._progress_at=None
        self._last_velocity=(0.,0.);self._last_output_at=self._last_now
        self._collapse=None;self._certificate=None
        self._recovery_at=None;self._recovery_used=0.;self._attempts=0
        self._budget_missing=run.ever_restarted and restored_budget is None
        if restored_budget is not None:
            c.require(restored_budget.name=='p03_recovery' and
                      restored_budget.attempts_limit==policy.retry_limit and
                      restored_budget.seconds_limit==policy.recovery_seconds,'budget','restore exact original limits')
            self._attempts=restored_budget.attempts_used;self._recovery_used=restored_budget.seconds_used

    def _now(self):
        now=self.clock.now();c.monotonic(now,'now')
        c.require(now-self._last_now>=0,'clock','clock regressed',c.ErrorCode.CLOCK_MISMATCH)
        self._last_now=now
        return now

    def _env(self,now=None,**kw):
        now=now or self._last_now;self._sequence+=1
        d=dict(producer=c.Owner.P03,event_id=str(uuid4()),sequence=self._sequence,emitted_at=now,
               expires_at=replace(now,seconds=now.seconds+self.policy.max_age_s))
        d.update(kw)
        return replace(self.context,**d)

    def _effect(self,kind,payload=None):
        self.effects.append(NavigationEffect(kind,self._env(),payload))

    def _check(self,accepted=True,error=None):
        return m.ContractCheck(accepted,reason(error) if error else None)

    def _gate(self,now,permit=None,source=None):
        c.require(self.active is not None and not self.terminal,'request','no active movable request',c.ErrorCode.CANCELLED)
        r.require_committed_space(self.active.space,self.space_readback)
        now=self._now()
        c.require(now-self.run.started_at>=0 and self.run.deadline-now>0,'deadline','original deadline reached',c.ErrorCode.DEADLINE)
        c.require(self.active.envelope.expires_at-now>0,'request.deadline','original request expired',c.ErrorCode.TIMEOUT)
        validate_localization(self.active.localization,self.context,now,self.policy.max_age_s)
        validate_readiness(self.active.readiness,self.context,now,self.policy.max_age_s)
        validate_env(self.active.space.envelope,self.context,now,motion=False)
        r.require_space_version(self.context,self.active.space)
        health=self.gateway.health();validate_env(health.envelope,self.context,now,motion=False)
        c.require(health.envelope.producer is c.Owner.P01 and
                  any(cap.name=='base' and cap.status is c.Fact.CONFIRMED for cap in health.capabilities),
                  'health.base','base capability unavailable',c.ErrorCode.UNAVAILABLE)
        if self._collapse is not None:
            c.require(self._certificate is not None,'collapse','continuous 500ms static certificate missing',c.ErrorCode.NOT_AUTHORIZED)
            self._validate_certificate(self._certificate,now)
        if permit is not None:
            validate_env(permit.envelope,self.context,now,identity=True)
            c.require(permit.source is source and permit.purpose==self.active.goal.purpose,
                      'permit','source/purpose mismatch',c.ErrorCode.NOT_AUTHORIZED)
            c.require(permit.envelope.expires_at-self.run.deadline<=0,'permit','extends original deadline',c.ErrorCode.DEADLINE)
            fresh(permit.prerequisite_evidence,replace(permit.envelope,target_id=None),now,self.policy.max_age_s)

    def _result(self,status=c.Outcome.UNKNOWN,why=None,accepted=c.Fact.CONFIRMED,physical=c.Fact.UNKNOWN):
        why=why or (None if status is c.Outcome.SUCCESS else c.Reason(c.ErrorCode.EVIDENCE_MISSING,self.phase,'awaiting explicit evidence'))
        evidence=self._stopped.evidence if physical is c.Fact.CONFIRMED else ()
        if physical is c.Fact.CONFIRMED and self._feedback:evidence+=self._feedback.evidence
        action=c.ActionResult(status,self.phase,why,accepted,self._submitted,
            c.Fact.UNKNOWN,physical,evidence,max(0.,self._last_now-self._began) if self._began else 0.)
        return m.NavigationResult(self._env(),action,self._stop,self._residual,
                    c.Fact.CONFIRMED if status is c.Outcome.SUCCESS else c.Fact.UNKNOWN,None)

    def navigate(self,request):
        try:
            now=self._now();key=request.envelope.request_id
            c.require(key is not None,'request_id','navigation identity required')
            if key in self._cache:
                c.require(self._cache[key]==request,'request_id','same request ID with changed payload',c.ErrorCode.ID_CONFLICT)
                if self.active and key==self.active.envelope.request_id:return self.current_result()
                return replace(self._history[key],work_ready=c.Fact.UNKNOWN)
            if self.active is not None and self.terminal:
                c.require(self._stop_confirmed(now),'stop','successor requires current actual stop/queue confirmation',c.ErrorCode.NOT_AUTHORIZED)
                self._history[self.active.envelope.request_id]=self.current_result()
                self.active=None;self.terminal=False;self._terminal_result=None;self._progress_result=None
                self._submitted=c.Fact.UNKNOWN
                self._feedback=None;self._geometry_after=None;self._work_invalid=False
                self._collapse=None;self._certificate=None;self._stop=None;self._stopped.reset()
            c.require(self.active is None,'request','another request is active',c.ErrorCode.CONFLICT)
            c.require(not self._budget_missing,'budget','restart needs persisted spent recovery budget',c.ErrorCode.UNAVAILABLE)
            validate_env(request.envelope,self.run.envelope,now)
            c.require(request.envelope.producer is c.Owner.P09,'request.producer','P09 owns mission intent')
            validate_env(request.goal.envelope,request.envelope,now,identity=True)
            c.require(request.goal.envelope.producer is c.Owner.P02 and request.goal.valid is c.Fact.CONFIRMED and
                      request.goal.pose is not None,'goal','valid P02 named goal required',c.ErrorCode.UNCONFIGURED)
            self.policy.dock(request.goal.purpose)
            self.context=request.envelope;self.active=request
            self._gate(now,request.permit,m.MotionSource.NAVIGATION)
            route=self.planner(request)
            c.require(bool(route) and route[0]==request.localization.pose and route[-1]==request.goal.pose,
                      'route','planner must bind current start and named goal')
            check=check_route(self.bundle,route,self.profile,request.readiness.holding.state,request.space)
            if not check.accepted:raise c.ContractError(check.reason.code,check.reason.field,check.reason.detail)
            self.route=route;self._permit=request.permit;self._began=now;self._stage_at=now
            self._progress_at=now;self._progress_pose=request.localization.pose
            self._last_output_at=now;self.phase='planned';self._cache[key]=request
            self._new_goal()
            return self.current_result()
        except (c.ContractError,OSError) as error:
            err=error if isinstance(error,c.ContractError) else c.ContractError(c.ErrorCode.UNAVAILABLE,'planner',str(error))
            if self.active is request and request.envelope.request_id not in self._cache:self.active=None
            result=self._result(c.Outcome.REJECTED,reason(err),c.Fact.REJECTED)
            return replace(result,envelope=replace(result.envelope,request_id=request.envelope.request_id,
                                                    target_id=request.envelope.target_id,action_id=None))

    def _new_goal(self):
        e=self._env(action_id='goal-'+str(uuid4()),expires_at=self.active.envelope.expires_at)
        self.goal_intent=m.ActionIntent(e,m.IntentOperation.BASE_GOAL,None,None,self.active.goal.pose,None,None)

    def submit_goal(self,receipt):
        try:
            now=self._now();self._gate(now,self._permit,m.MotionSource.NAVIGATION)
            r.validate_intent_receipt(self.goal_intent,receipt)
            if self.phase=='coarse':return self._check()
            c.require(self.phase=='planned','phase','goal is not awaiting dispatch',c.ErrorCode.CONFLICT)
            self.phase='coarse';self._stage_at=now
            self._effect('start_goal',(self.goal_intent,self.route,receipt))
            return self._check()
        except c.ContractError as error:return self._check(False,error)

    def _halt(self,next_phase):
        old_source=self._permit.source if self._permit else (m.MotionSource.PRECISION if self.phase=='precision' else m.MotionSource.NAVIGATION)
        self._pending.clear();self._permit=None;self._feedback=None
        self._stopped.reset();self._stop_since=self._last_now;self._actual_stop_at=None;self._stop_input=None
        self._last_velocity=(0.,0.);self._last_output_at=self._last_now
        submitted=c.Fact.UNKNOWN
        try:
            if self.gateway.revoke(old_source):submitted=c.Fact.CONFIRMED
        except Exception:pass
        self._effect('revoke',old_source)
        if self.goal_intent is not None:self._effect('cancel_goal',self.goal_intent.envelope)
        self._stop=m.StopReport(self._env(),c.Fact.CONFIRMED,submitted,c.Fact.UNKNOWN,c.Fact.UNKNOWN,c.Fact.UNKNOWN,())
        self.phase=next_phase;self._stage_at=self._last_now

    def _finish(self,status,why):
        if self.terminal:return
        stage=self.phase
        if status is not c.Outcome.SUCCESS:self._halt(stage)
        self.terminal=True
        self._terminal_result=self._result(status,why,physical=c.Fact.CONFIRMED if status is c.Outcome.SUCCESS else c.Fact.UNKNOWN)
        if self._recovery_at is not None:
            self._recovery_used=self.budget.seconds_used;self._recovery_at=None
        self._effect('terminal',self._terminal_result)

    def coarse_event(self,event):
        try:
            now=self._now()
            c.require(self.phase=='coarse' and not self.terminal,'phase','late coarse callback ignored',c.ErrorCode.STALE_REQUEST)
            validate_env(event.envelope,self.context,now,identity=True)
            c.require(event.envelope.producer is c.Owner.P01 and event.envelope.action_id==self.goal_intent.envelope.action_id,
                      'action_id','callback from old/wrong goal',c.ErrorCode.STALE_REQUEST)
            key=event.envelope.event_id
            if key in self._events:
                c.require(self._events[key]==event,'event_id','same ID with changed event',c.ErrorCode.ID_CONFLICT)
                return self._check()
            self._gate(now,self._permit,m.MotionSource.NAVIGATION);self._events[key]=event
            self._submitted=event.result.submitted
            if event.result.status is c.Outcome.SUCCESS:self._halt('handoff')
            elif event.result.status in (c.Outcome.FAILED,c.Outcome.REJECTED):self.blocked('coarse planner failure')
            elif event.result.status in (c.Outcome.CANCELLED,c.Outcome.TIMED_OUT):self._finish(event.result.status,event.result.reason)
            elif event.result.submitted is c.Fact.UNKNOWN:self._finish(c.Outcome.UNKNOWN,event.result.reason)
            return self._check()
        except c.ContractError as error:return self._check(False,error)

    def observe_stop(self,report):
        try:
            now=self._now();validate_env(report.envelope,self.context,now,identity=True)
            c.require(report.envelope.producer is c.Owner.P01 and self._stop_since is not None,
                      'stop','P01 report for an existing stop required')
            if self._stop_input is not None:
                if report==self._stop_input:return self._check()
                c.require(report.envelope.event_id!=self._stop_input.envelope.event_id and
                          report.envelope.emitted_at-self._stop_input.envelope.emitted_at>0,
                          'stop','older/conflicting stop report',c.ErrorCode.STALE_REQUEST)
                if self._stop_input.evidence and report.evidence:
                    c.require(min(e.captured_at.seconds for e in report.evidence)>=
                              max(e.captured_at.seconds for e in self._stop_input.evidence),
                              'stop','capture regressed',c.ErrorCode.STALE_REQUEST)
            self._stop_input=report
            # A new same-request UNKNOWN/invalid report removes usability before
            # checking its physical references; it cannot leave old success live.
            self._stop=replace(report,physically_stopped=c.Fact.UNKNOWN,old_queue_empty=c.Fact.UNKNOWN)
            fresh(report.evidence,report.envelope,now,self.policy.max_age_s)
            c.require(all(e.captured_at-self._stop_since>=0 for e in report.evidence),'stop','report predates this stop')
            self._stop=report;self._advance_stop(now)
            return self._check()
        except c.ContractError as error:return self._check(False,error)

    def observe_motion(self,sample):
        try:
            now=self._now()
            c.require(self._stop_since is not None,'stop','stop not requested')
            self._stopped.observe(sample,self.context,now,self._stop_since)
            if not self._stopped.confirmed(now):
                self._actual_stop_at=None;self._certificate=None
            self._advance_stop(now)
            return self._check()
        except c.ContractError as error:return self._check(False,error)

    def _stop_confirmed(self,now):
        try:
            c.require(self._stop is not None and self._stop_input is not None,'stop','no current cancellation acknowledgement')
            validate_env(self._stop_input.envelope,self.context,now,identity=True)
            fresh(self._stop_input.evidence,self._stop_input.envelope,now,self.policy.max_age_s)
            return (self._stop.requested is c.Fact.CONFIRMED and self._stop.submitted is c.Fact.CONFIRMED and
                    self._stop.old_queue_empty is c.Fact.CONFIRMED and self._stopped.confirmed(now))
        except c.ContractError:return False

    def _advance_stop(self,now):
        if not self._stop_confirmed(now) or self.terminal:return
        if self._actual_stop_at is None:self._actual_stop_at=self._stopped.evidence[-1].captured_at
        self._stop=replace(self._stop,physically_stopped=c.Fact.CONFIRMED,evidence=self._stopped.evidence)
        if self.phase=='handoff':self.phase='precision_ready'
        elif self.phase=='final_stop':
            try:
                self._gate(now)
                self._validate_feedback(self._feedback,now)
                self._work_invalid=False
                self._finish(c.Outcome.SUCCESS,None)
            except c.ContractError as error:self._finish(c.Outcome.FAILED,reason(error))

    def install_precision(self,permit):
        try:
            now=self._now();self._gate(now,permit,m.MotionSource.PRECISION)
            c.require(self.phase=='precision_ready' and self._stop_confirmed(now),'handoff','not confirmed stopped/cleared',c.ErrorCode.NOT_AUTHORIZED)
            self._permit=permit;self.phase='precision';self._stage_at=now
            return self._check()
        except c.ContractError as error:return self._check(False,error)

    def _validate_feedback(self,value,now):
        c.require(value is not None,'geometry','feedback missing',c.ErrorCode.EVIDENCE_MISSING)
        validate_env(value.envelope,self.context,now,identity=True)
        c.require(value.envelope.producer in (c.Owner.P04,c.Owner.P06) and value.goal_name==self.active.goal.name and
                  value.purpose==self.active.goal.purpose,'geometry','wrong purpose/goal/producer')
        fresh(value.evidence,value.envelope,now,self.policy.max_age_s,self.context.target_id)
        if self._geometry_after is not None:
            c.require(all(e.captured_at-self._geometry_after>0 for e in value.evidence),'geometry',
                      'new observation after movement required',c.ErrorCode.STALE_GEOMETRY)
        c.require(not value.self_occluded,'holding_occlusion','P07 limited arm observation required',c.ErrorCode.UNAVAILABLE)
        c.require(value.geometry_valid is c.Fact.CONFIRMED and value.distance_basis in
                  (m.PlaneBasis.CALIBRATION,m.PlaneBasis.VERIFIED_DEPTH,m.PlaneBasis.SUPPORT),
                  'distance','verified distance basis required',c.ErrorCode.EVIDENCE_MISSING)
        for key in ('forward_m','lateral_m','yaw_rad'):
            v=getattr(value,key)
            c.require(type(v) in (int,float) and math.isfinite(v),key,'finite geometric residual required',c.ErrorCode.EVIDENCE_MISSING)

    def precision(self,value):
        try:
            now=self._now();self._gate(now,self._permit,m.MotionSource.PRECISION)
            c.require(self.phase=='precision','phase','precision is not active',c.ErrorCode.NOT_AUTHORIZED)
            self._validate_feedback(value,now)
            window=self.policy.dock(value.purpose)
            c.require(math.hypot(value.forward_m,value.lateral_m)<=window.max_correction_m,
                      'correction','outside measured precision range',c.ErrorCode.LIMIT_EXCEEDED)
            self._residual=c.MapPose('base_residual',value.forward_m,value.lateral_m,angle(value.yaw_rad),'m','rad',0.,0.)
            self._feedback=value
            close=(abs(value.forward_m)<=window.forward_tolerance_m and abs(value.lateral_m)<=window.lateral_tolerance_m)
            if close and abs(angle(value.yaw_rad))<=window.yaw_tolerance_rad:
                c.require(value.work_condition is c.Fact.CONFIRMED,'work_condition','purpose conditions missing',c.ErrorCode.EVIDENCE_MISSING)
                self._halt('final_stop');self._feedback=value
                return None
            steering=angle(value.yaw_rad) if close else math.atan2(value.lateral_m,value.forward_m)
            if abs(steering)>window.yaw_tolerance_rad:
                return self.prepare_velocity(0.,math.copysign(self.policy.max_angular_rad_s,steering),m.MotionSource.PRECISION)
            c.require(value.forward_m>0,'forward','backward correction needs separately verified recovery',c.ErrorCode.NOT_AUTHORIZED)
            return self.prepare_velocity(min(value.forward_m,self.policy.max_linear_m_s),0.,m.MotionSource.PRECISION)
        except c.ContractError as error:
            if error.field=='holding_occlusion':self._effect('request_P07_observation',value)
            self._finish(c.Outcome.FAILED,reason(error));return None

    def prepare_velocity(self,linear,angular,source):
        try:
            now=self._now()
            c.require((self.phase,source) in (('coarse',m.MotionSource.NAVIGATION),('precision',m.MotionSource.PRECISION)),
                      'source','inactive source; recovery micro-motions not commissioned',c.ErrorCode.NOT_AUTHORIZED)
            c.require(self._permit is not None,'permit','current permission missing',c.ErrorCode.NOT_AUTHORIZED)
            self._gate(now,self._permit,source)
            for v,key in ((linear,'linear'),(angular,'angular')):
                c.require(type(v) in (int,float) and math.isfinite(v),key,'finite differential speed required')
            c.require(abs(linear)<=self.policy.max_linear_m_s and abs(angular)<=self.policy.max_angular_rad_s,
                      'velocity','outside measured speed envelope',c.ErrorCode.LIMIT_EXCEEDED)
            if source is m.MotionSource.PRECISION:self._validate_feedback(self._feedback,now)
            dt=max(0.,now-self._last_output_at)
            limits=(self.policy.max_linear_accel_m_s2*dt,self.policy.max_angular_accel_rad_s2*dt)
            # A stop/turn axis stays zero; acceleration clipping must not carry
            # previous forward speed into an in-place turn.
            values=tuple(0. if wanted==0 else max(old-lim,min(old+lim,wanted))
                         for wanted,old,lim in zip((linear,angular),self._last_velocity,limits))
            p=self.active.localization.pose
            # All possible arcs during this command lie in a disk expanded by
            # path length; checking a straight endpoint chord would miss turns.
            swept=replace(self.profile,error_m=self.profile.error_m+abs(values[0])*self.policy.max_age_s)
            check=check_route(self.bundle,(p,p),swept,self.active.readiness.holding.state,self.active.space)
            if not check.accepted:raise c.ContractError(check.reason.code,check.reason.field,check.reason.detail)
            refs=(self.active.localization.evidence+self.active.readiness.evidence+
                  self.active.readiness.holding.evidence+self._permit.prerequisite_evidence)
            if self._feedback is not None:refs+=self._feedback.evidence
            limits=[self.active.envelope.expires_at,self.run.deadline,self._permit.envelope.expires_at,
                    self.active.space.envelope.expires_at,replace(now,seconds=now.seconds+self.policy.max_age_s)]
            for evidence in refs:
                limits.extend((evidence.valid_until,replace(evidence.captured_at,seconds=evidence.captured_at.seconds+self.policy.max_age_s)))
            expires=min(limits,key=lambda t:t.seconds)
            c.require(expires-now>0,'evidence','motion evidence lease exhausted',c.ErrorCode.EXPIRED)
            e=self._env(action_id='velocity-'+str(uuid4()),expires_at=expires)
            intent=m.ActionIntent(e,m.IntentOperation.BASE_VELOCITY,None,None,None,values,None)
            self._pending.clear();self._pending[e.action_id]=(intent,source)
            return intent
        except c.ContractError:return None

    def dispatch_velocity(self,intent,receipt,permit):
        try:
            now=self._now()
            c.require(intent is not None and intent.envelope.action_id in self._pending,'intent','obsolete/unprepared velocity',c.ErrorCode.STALE_REQUEST)
            expected,source=self._pending[intent.envelope.action_id]
            c.require(intent==expected,'intent','payload differs',c.ErrorCode.ID_CONFLICT)
            self._gate(now,permit,source);validate_env(intent.envelope,self.context,now,identity=True)
            c.require(permit==self._permit,'permit','not active P03 lease',c.ErrorCode.NOT_AUTHORIZED)
            r.validate_intent_receipt(intent,receipt)
            command=m.BaseCommand(intent.envelope,permit,receipt.commit.transaction_id,
                                  *intent.base_velocity_m_s_rad_s,source)
            self.gateway.register_intent(command,intent,receipt)
            self._pending.clear()
            result=self.gateway.submit_base(command)
            self._submitted=result.result.submitted
            c.require(result.result.submitted is c.Fact.CONFIRMED,'submitted','output not confirmed',c.ErrorCode.UNAVAILABLE)
            self._last_output_at=now;self._last_velocity=intent.base_velocity_m_s_rad_s
            self._dispatched[intent.envelope.action_id]=result
            self._geometry_after=now;self._feedback=None;self._work_invalid=True
            self._effect('invalidate_geometry',intent.envelope.base_motion_epoch)
            return result
        except c.ContractError as error:
            if error.field=='submitted':self._finish(c.Outcome.UNKNOWN,reason(error))
            return None
        except Exception as error:
            self._finish(c.Outcome.UNKNOWN,c.Reason(c.ErrorCode.UNAVAILABLE,'output',str(error)));return None

    def refresh(self,localization,readiness):
        try:
            now=self._now()
            c.require(self.active is not None,'request','no active request')
            c.require(localization.base_motion_epoch>=self.context.base_motion_epoch,'epoch','old movement epoch',c.ErrorCode.STALE_GEOMETRY)
            reference=replace(self.context,base_motion_epoch=localization.base_motion_epoch)
            validate_localization(localization,reference,now,self.policy.max_age_s)
            validate_readiness(readiness,reference,now,self.policy.max_age_s)
            for current,old in ((localization,self.active.localization),(readiness,self.active.readiness)):
                c.require(current.envelope.emitted_at-old.envelope.emitted_at>=0 and
                          min(e.captured_at.seconds for e in current.evidence)>=max(e.captured_at.seconds for e in old.evidence),
                          'observation','older capture cannot overwrite current facts',c.ErrorCode.STALE_REQUEST)
            old=self._progress_pose;p=localization.pose
            if old is None or math.hypot(p.x-old.x,p.y-old.y)>=self.policy.progress_m or abs(angle(p.yaw-old.yaw))>=self.policy.progress_rad:
                self._progress_at=now;self._progress_pose=p
            if reference.base_motion_epoch!=self.context.base_motion_epoch or p!=self.active.localization.pose:
                self._pending.clear();self._feedback=None;self._certificate=None
                self._work_invalid=True;self._geometry_after=now
                self._stopped.reset()
                self._effect('invalidate_geometry',reference.base_motion_epoch)
            self.context=reference;self.active=replace(self.active,localization=localization,readiness=readiness)
            return self._check()
        except c.ContractError as error:
            self._work_invalid=True;self._feedback=None
            self._finish(c.Outcome.FAILED,reason(error));return self._check(False,error)

    def adopt_committed_space(self,space):
        """Explicit same-run spatial publication; never reset spent budgets.

        Assembly must separately refresh gateway/health/localization adapters.
        An active path is cancelled and requires actual stop evidence before a
        successor request. This method grants no motion or control lease.
        """
        try:
            published=r.require_committed_space(space,self.space_readback)
            now=self._now();old=self.run;new=published.run
            c.require(new.envelope.producer is c.Owner.P10A and new.state is m.RunState.OFFLINE,
                      'run','P10A offline run publication required',c.ErrorCode.NOT_AUTHORIZED)
            c.require((new.started_at,new.deadline,new.start_signal_id,new.ever_restarted)==
                      (old.started_at,old.deadline,old.start_signal_id,old.ever_restarted),
                      'run','original run/deadline cannot change',c.ErrorCode.STALE_RUN)
            for name in ('config','map','calibration','model','mapping','layout'):
                c.require(getattr(new.envelope.versions,name)==getattr(old.envelope.versions,name),
                          'versions.'+name,'only spatial version may change',c.ErrorCode.VERSION_MISMATCH)
            reference=replace(old.envelope,versions=new.envelope.versions)
            validate_env(new.envelope,reference,now)
            validate_env(space.envelope,reference,now)
            c.require(space.version>=old.envelope.versions.space,'space','publication cannot regress',c.ErrorCode.VERSION_MISMATCH)
            if self.active is not None:
                check=self.invalidate_space(space)
                if not check.accepted:return check
            self.run=new
            if self.active is None:self.context=new.envelope
            return self._check()
        except c.ContractError as error:
            return self._check(False,error)

    def invalidate_space(self,snapshot):
        try:
            now=self._now();reference=replace(self.context,versions=replace(self.context.versions,space=snapshot.version))
            validate_env(snapshot.envelope,reference,now,motion=False)
            r.require_committed_space(snapshot,self.space_readback)
            now=self._now();validate_env(snapshot.envelope,reference,now,motion=False)
            if self.active is None:return self._check()
            c.require(snapshot.version>=self.active.space.version,'space','space version regressed',c.ErrorCode.VERSION_MISMATCH)
            if snapshot.version==self.active.space.version:
                c.require(snapshot==self.active.space,'space','same version with different occupancy',c.ErrorCode.ID_CONFLICT)
                return self._check()
            self._feedback=None;self._certificate=None
            self._finish(c.Outcome.CANCELLED,c.Reason(c.ErrorCode.VERSION_MISMATCH,'space','active path invalidated; obtain new request/permit'))
            self._effect('path_invalidated',snapshot)
            return self._check()
        except c.ContractError as error:
            self._feedback=None;self._finish(c.Outcome.FAILED,reason(error));return self._check(False,error)

    def blocked(self,detail):
        if self.terminal:return
        self._now()
        if self._recovery_at is None:self._recovery_at=self._last_now
        self._halt('recovery_wait');self._effect('recovery',detail)

    @property
    def budget(self):
        used=self._recovery_used+(max(0.,self._last_now-self._recovery_at) if self._recovery_at else 0.)
        return m.Budget('p03_recovery',self.policy.retry_limit,self.policy.recovery_seconds,self._attempts,used)

    def recover(self,permit):
        try:
            now=self._now();self._gate(now,permit,m.MotionSource.NAVIGATION)
            c.require(self.phase in ('recovery_wait','collapse') and self._stop_confirmed(now),
                      'recovery','old action/queue not stopped',c.ErrorCode.NOT_AUTHORIZED)
            c.require(self._attempts<self.policy.retry_limit and self.budget.seconds_used<self.policy.recovery_seconds,
                      'recovery_budget','original recovery budget exhausted',c.ErrorCode.LIMIT_EXCEEDED)
            route=self.planner(self.active)
            c.require(bool(route) and route[0]==self.active.localization.pose and route[-1]==self.active.goal.pose,
                      'route','recovery must preserve current start and original named destination')
            check=check_route(self.bundle,route,self.profile,self.active.readiness.holding.state,self.active.space)
            if not check.accepted:raise c.ContractError(check.reason.code,check.reason.field,check.reason.detail)
            self._attempts+=1;self.route=route;self._permit=permit;self.phase='planned';self._stage_at=now
            self._new_goal();return self._check()
        except c.ContractError as error:
            self._finish(c.Outcome.FAILED,reason(error));return self._check(False,error)

    def collapse(self,affected_targets,coverage,scene_epoch):
        c.require(type(affected_targets) is tuple and bool(affected_targets),'collapse','affected identities required')
        self.blocked('suspected/confirmed collapse')
        self._collapse=(affected_targets,coverage,scene_epoch);self._certificate=None;self.phase='collapse'

    def _validate_certificate(self,cert,now):
        targets,coverage,epoch=self._collapse
        c.require(cert.envelope.producer is c.Owner.P08,'certificate','P08 owns stillness')
        r.validate_stillness(cert,self.context,now,targets,epoch)
        fresh(cert.evidence,cert.envelope,now,self.policy.max_age_s)
        c.require(self._stop is not None and self._stop.physically_stopped is c.Fact.CONFIRMED and
                  self._actual_stop_at is not None and cert.capture_start-self._actual_stop_at>=0,
                  'capture_start','static window must start after actual stop')
        c.require(cert.frame_id==self.active.localization.pose.frame_id and covered(coverage,cert.coverage_polygon_m),
                  'coverage','scattered range not covered')
        c.require(all(b.captured_at-a.captured_at<=self.policy.max_gap_s for a,b in zip(cert.evidence,cert.evidence[1:])),
                  'continuous','frame gap exceeds configured continuity')

    def confirm_stillness(self,cert):
        try:
            now=self._now();c.require(self._collapse is not None,'collapse','no collapse context')
            self._validate_certificate(cert,now);self._certificate=cert;return self._check()
        except c.ContractError as error:self._certificate=None;return self._check(False,error)

    def tick(self):
        if self.active is None or self.terminal:return self.current_result()
        try:
            now=self._now()
            c.require(self.run.deadline-now>0 and self.active.envelope.expires_at-now>0,'deadline','original deadline expired',c.ErrorCode.DEADLINE)
            limit=(self.policy.coarse_timeout_s if self.phase in ('planned','coarse') else
                   self.policy.precision_timeout_s if self.phase=='precision' else self.policy.handoff_timeout_s)
            c.require(now-self._stage_at<limit,'phase_timeout','phase budget exhausted',c.ErrorCode.TIMEOUT)
            if self._recovery_at:
                c.require(self.budget.seconds_used<self.policy.recovery_seconds,'recovery_budget','total recovery time exhausted',c.ErrorCode.TIMEOUT)
            if self.phase in ('coarse','planned','precision'):
                source=m.MotionSource.PRECISION if self.phase=='precision' else m.MotionSource.NAVIGATION
                c.require(self._permit is not None,'permit','current permit missing',c.ErrorCode.NOT_AUTHORIZED)
                self._gate(now,self._permit,source)
            elif self.phase not in ('collapse','recovery_wait'):self._gate(now)
            if self.phase=='coarse' and now-self._progress_at>=self.policy.recovery_seconds:self.blocked('no observed displacement/rotation progress')
        except c.ContractError as error:
            self._finish(c.Outcome.TIMED_OUT if error.code in (c.ErrorCode.TIMEOUT,c.ErrorCode.DEADLINE) else c.Outcome.FAILED,reason(error))
        return self.current_result()

    def cancel(self,request):
        now=self._now();validate_env(request.envelope,self.context,now)
        c.require(request.envelope.producer is c.Owner.P10A,'cancel','P10A stop required')
        self._finish(c.Outcome.CANCELLED,request.reason)
        return self._stop

    def current_result(self):
        if self._terminal_result is None:
            old=self._progress_result
            if old is None or (old.result.phase,old.stop,old.residual_pose,old.result.submitted)!=(self.phase,self._stop,self._residual,self._submitted):
                self._progress_result=self._result()
            return self._progress_result
        result=self._terminal_result
        if result.work_ready is c.Fact.CONFIRMED:
            try:
                now=self.clock.now()
                c.require(not self._work_invalid and now-self._last_now>=0 and self.run.deadline-now>0 and
                          self.active.envelope.expires_at-now>0,'clock','expired/invalid work conditions')
                validate_localization(self.active.localization,self.context,now,self.policy.max_age_s)
                validate_readiness(self.active.readiness,self.context,now,self.policy.max_age_s)
                self._validate_feedback(self._feedback,now)
                c.require(self._stop_confirmed(now),'stop','stop evidence no longer fresh')
            except c.ContractError:return replace(result,work_ready=c.Fact.UNKNOWN)
        return result
