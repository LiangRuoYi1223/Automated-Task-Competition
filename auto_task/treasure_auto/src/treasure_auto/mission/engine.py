"""One-work-order offline scheduler. Directives never authorize device output."""
from dataclasses import replace
from treasure_auto.contracts import core as c,models as m,rules as r
from .inputs import Stage,Directive,MissionFrame,MissionOutput,OWNERS
from .accounting import arrival_counts
from . import guards


class MissionEngine:
    def __init__(self,config,clock,journal,*,stop_sink=None,field_strategy=None):
        c.require(getattr(journal,'origin',None) is c.Origin.SIMULATED,'checkpoint','C0 production checkpoint adapter not frozen',c.ErrorCode.UNAVAILABLE)
        self.config,self.clock,self.journal=config,clock,journal
        self.stop_sink=stop_sink
        self.stage=Stage.NAV_BOARD;self.order=None;self.pending=None;self.run=None
        self.cake_closed=False;self.exit_evidence=None;self.blocked=None;self.end_reason=None
        self.sequence=0;self.events={};self.trace=[];self.observations=0
        self.recovery_attempts=0;self.recovery_seconds=0.;self.last_now=None
        self.last_revision=-1;self.last_snapshot=None;self.mapping=None
        self.field_strategy=field_strategy;self._field_state=None
        self._bound_field_strategy=None
        self._field_decision=None;self._perception_counts=None;self._perception_policy=None
        self._field_expires_s=None
        self.restored=journal.load()
        if self.restored is not None:self._restore(self.restored)

    @property
    def exit_points(self):return 1 if self.exit_evidence is not None else 0

    def _reason(self,code,field,detail):return c.Reason(code,field,detail)

    def _require_field_strategy(self):
        if self._bound_field_strategy is not None or self._field_state is not None:
            c.require(self.field_strategy is not None and self.field_strategy is self._bound_field_strategy,
                      'field_strategy','persisted field policy must remain bound to its restored controller',c.ErrorCode.UNAVAILABLE)
        elif self.field_strategy is not None:
            self._bound_field_strategy=self.field_strategy

    def _envelope(self,snapshot,*,action=None,target=None,request=None):
        now=self.clock.now();e=snapshot.space.envelope
        return replace(e,producer=c.Owner.P09,event_id='p09-'+str(self.sequence),sequence=self.sequence,
                       emitted_at=now,expires_at=replace(now,seconds=now.seconds+self.config.evidence_age_s),
                       action_id=action,target_id=target,request_id=request)

    def _snapshot(self,frame):
        snap=frame.inputs.snapshot;now=self.clock.now();e=frame.inputs.envelope
        r.require_simulated(frame.inputs)
        r.validate_envelope(e,guards.context(replace(snap.run.envelope,versions=snap.space.envelope.versions,
            base_motion_epoch=e.base_motion_epoch,layout_recovery_round=e.layout_recovery_round,rebuild_generation=e.rebuild_generation),now))
        c.require(e.versions.config==self.config.version,'config','frozen configuration differs')
        c.require(self.last_now is None or now-self.last_now>=0,'clock','clock moved backwards',c.ErrorCode.CLOCK_MISMATCH)
        self.last_now=now
        if self.run is None:
            self.run=snap.run
            if snap.run.ever_restarted or snap.deliveries or snap.layouts:
                self.stage=Stage.REVALIDATE
                self.blocked=self._reason(c.ErrorCode.UNAVAILABLE,'checkpoint','existing run requires restored context and physical reconciliation')
        else:
            c.require((snap.run.envelope.run_id,snap.run.started_at,snap.run.deadline,snap.run.start_signal_id)==
                      (self.run.envelope.run_id,self.run.started_at,self.run.deadline,self.run.start_signal_id),
                      'run','original run/start/deadline are immutable',c.ErrorCode.STALE_RUN)
            if self.stage is not Stage.REVALIDATE:
                c.require((snap.run.envelope.instance_id,snap.run.envelope.control_epoch)==(self.run.envelope.instance_id,self.run.envelope.control_epoch),
                          'control','new instance requires recovery',c.ErrorCode.STALE_CONTROL)
        c.require(snap.revision>=self.last_revision,'revision','stale state snapshot',c.ErrorCode.CONFLICT)
        if snap.revision==self.last_revision:
            c.require(snap==self.last_snapshot,'revision','same revision changed content',c.ErrorCode.ID_CONFLICT)
        if self.mapping is not None:c.require(snap.mapping==self.mapping,'mapping','same-run mapping is frozen',c.ErrorCode.INVALID_MAPPING)
        elif snap.mapping is not None:
            c.require(snap.mapping.envelope.producer is c.Owner.P05 and snap.mapping.envelope.run_id==e.run_id and snap.mapping.mapping_version==e.versions.mapping,
                      'mapping','P05 committed mapping required',c.ErrorCode.INVALID_MAPPING)
            self.mapping=snap.mapping
        self.last_revision=snap.revision;self.last_snapshot=snap
        return snap,now,e

    def _exit(self,ex,snap,now,e):
        r.validate_envelope(ex.envelope,guards.context(e,now))
        c.require(ex.envelope.producer is c.Owner.P03 and ex.area is m.Area.A and ex.entire_envelope_outside is c.Fact.CONFIRMED,
                  'exit','whole-envelope P03 proof required')
        c.require(snap.run.deadline-ex.envelope.emitted_at>0 and ex.envelope.emitted_at-snap.run.started_at>=0,'exit','out of original time interval')
        guards.physical(ex.evidence,ex.envelope,now,self.config.evidence_age_s)
        if self.exit_evidence is None:self.exit_evidence=ex

    def _consume(self,event,frame,snap,now,e):
        key=event.envelope.producer.value+':'+event.envelope.event_id;payload=c.dumps(event)
        if key in self.events:
            c.require(self.events[key]==payload,'event_id','conflicting repeated event',c.ErrorCode.ID_CONFLICT)
            return
        c.require(self.pending is not None,'callback','no active directive',c.ErrorCode.STALE_REQUEST)
        d=self.pending
        r.validate_envelope(event.envelope,guards.context(d.envelope,now,identities=True))
        c.require(event.envelope.producer is d.owner and event.result.phase==d.stage.value,'callback','wrong stage owner or result phase',c.ErrorCode.STALE_REQUEST)
        self.events[key]=payload
        if event.result.status is not c.Outcome.SUCCESS or event.result.physical is not c.Fact.CONFIRMED:
            self.blocked=event.result.reason or self._reason(c.ErrorCode.EVIDENCE_MISSING,'physical','success/echo does not prove physical completion')
            return
        guards.physical(event.result.evidence,event.envelope,now,self.config.evidence_age_s)
        if d.stage in (Stage.NAV_BOARD,Stage.NAV_CENTRAL,Stage.CARRY,Stage.RETURN):
            guards.stopped(frame.stop,e,now,self.config.evidence_age_s)
        if d.stage is Stage.READ_BOARD:c.require(snap.mapping is not None,'mapping','await committed complete mapping')
        if d.stage is Stage.GRASP:
            c.require(snap.holding.target_id==self.order.envelope.target_id,'holding','wrong object held')
            guards.ready(frame.readiness,snap,e,now,self.config.evidence_age_s,c.Holding.HELD)
        if d.stage is Stage.PLACE:
            key=m.DeliveryKey(e.run_id,self.order.inventory_id)
            c.require(any(x.key==key and x.category is self.order.category and x.area is self.order.destination for x in snap.deliveries) and snap.holding.state is c.Holding.EMPTY,
                      'release','await linked P10B release transaction; no inferred arrival')
        if d.stage is Stage.LAYOUT:
            layout=next((x for x in snap.layouts if x.area is self.order.destination),None)
            guards.layout_prefix(layout,snap,e,now,self.config.evidence_age_s,self.order.envelope.target_id)
        if d.stage is Stage.EMPTY_READY:
            guards.ready(frame.readiness,snap,e,now,self.config.evidence_age_s,c.Holding.EMPTY)
            if self.order.category is m.Category.CAKE:self.cake_closed=True
        next_stage={Stage.NAV_BOARD:Stage.READ_BOARD,Stage.READ_BOARD:Stage.NAV_CENTRAL,Stage.NAV_CENTRAL:Stage.SELECT,
            Stage.OBSERVE:Stage.SELECT,Stage.GRASP:Stage.CARRY,Stage.CARRY:Stage.PLACE,Stage.PLACE:Stage.LAYOUT,
            Stage.LAYOUT:Stage.EMPTY_READY,Stage.EMPTY_READY:Stage.RETURN,Stage.RETURN:Stage.SELECT}
        self.pending=None;self.stage=next_stage[d.stage]
        if d.stage is Stage.RETURN:self.order=None;self.observations=0

    def _candidate(self,target,frame,snap,now,e):
        """The same gates apply to first selection and every grasp retry."""
        self._require_field_strategy()
        if self._perception_policy is not None:
            c.require(not any((v.target_id is not None and v.target_id==target.target_id) or
                              v.observation_id==target.observation_id
                              for v in self._perception_policy.review_targets),
                      'candidate.review','target requires new localization/camera observation')
        if self.field_strategy is not None:
            policy=self.field_strategy.handling.decision_for(target.target_id)
            c.require(policy is None or not policy.skip_this_run,'candidate.probe','target exhausted independent probe attempts')
        c.require(target.target_id is not None and target.identity_confirmed is c.Fact.CONFIRMED and
                  target.visibility is m.Visibility.VISIBLE and target.top is not None and
                  target.top.visibility is m.Visibility.VISIBLE and target.top.plane_basis is not m.PlaneBasis.UNKNOWN and
                  target.top.height_mm is not None,'candidate','identity and visible real top with height required')
        c.require((target.category is m.Category.CAKE)==(not self.cake_closed),'candidate','cake priority')
        item=next((i for i in snap.inventory if target.target_id in i.target_ids),None)
        delivered={x.key.inventory_id for x in snap.deliveries}
        c.require(item is not None and item.identity_confirmed is c.Fact.CONFIRMED and item.category is target.category and
                  item.location is m.InventoryLocation.CENTRAL and item.inventory_id not in delivered,
                  'inventory','not an undelivered central physical target')
        env=replace(e,target_id=target.target_id,action_id=None)
        guards.physical(target.evidence,env,now,self.config.evidence_age_s)
        guards.physical(target.top.evidence,env,now,self.config.evidence_age_s)
        plan=next((p for p in frame.plans if p.envelope.target_id==target.target_id),None)
        c.require(plan is not None and plan.envelope.producer is c.Owner.P06 and plan.reachable is c.Fact.CONFIRMED and
                  plan.collision_free is c.Fact.CONFIRMED and bool(plan.segments),
                  'plan','current reachable and collision-free P06 geometry required')
        r.validate_envelope(plan.envelope,guards.context(e,now))
        guards.physical(plan.evidence,plan.envelope,now,self.config.evidence_age_s)
        barrier=self.field_strategy.geometry_barrier if self.field_strategy is not None else None
        if barrier is not None:
            report=self._perception_policy
            c.require(not self.field_strategy.push_active and barrier['settled_at_s'] is not None and
                      e.base_motion_epoch>barrier['base_motion_epoch'],
                      'candidate.geometry','push requires confirmed stop and a new public base-motion epoch',c.ErrorCode.STALE_GEOMETRY)
            # P04's envelope is already validated in the run's monotonic clock.
            # The image capture stamp may belong to an unrelated sensor clock.
            c.require(report is not None and report.envelope.emitted_at.seconds>barrier['settled_at_s'] and
                      plan.envelope.emitted_at-report.envelope.emitted_at>=0,
                      'candidate.geometry','post-stop current P04 report and recomputed P06 plan required',c.ErrorCode.STALE_GEOMETRY)
        return item

    def _select(self,frame,snap,now,e):
        c.require(self.config.can_start(r.remaining_seconds(snap.run,now)),'budget','insufficient measured/simulated full-order reserve',c.ErrorCode.LIMIT_EXCEEDED)
        c.require(snap.mapping is not None,'mapping','mapping required')
        c.require(snap.holding.state is c.Holding.EMPTY,'holding','resolve held/unknown/dropped object before another grasp')
        c.require(not any(i.identity_confirmed is not c.Fact.CONFIRMED or i.location in
            (m.InventoryLocation.HELD,m.InventoryLocation.UNKNOWN,m.InventoryLocation.TEMPORARY) for i in snap.inventory),
            'inventory','resolve uncertain/held/temporary inventory before new grasp')
        c.require(not any(o.state is not m.OccupancyState.ACTUAL for o in snap.space.occupancies),'space','unknown or reserved occupancy remains')
        c.require(not any(l.quality is c.Fact.REJECTED or l.layout_recovery_round>0 and l.confirmed_count<3 for l in snap.layouts),'layout','pending regional repair must finish first')
        guards.stopped(frame.stop,e,now,self.config.evidence_age_s)
        guards.ready(frame.readiness,snap,e,now,self.config.evidence_age_s,c.Holding.EMPTY)
        for target in frame.inputs.candidates:
            try:
                item=self._candidate(target,frame,snap,now,e)
                self.sequence+=1
                order_env=self._envelope(snap,target=target.target_id,request='p09-work-'+str(self.sequence))
                area=snap.mapping.destination(target.category)
                budget=next((b for b in snap.budgets if b.name=='recovery-'+area.value),
                    m.Budget('recovery-'+area.value,self.config.recovery_attempts,self.config.recovery_total_s,0,0.))
                self.order=m.WorkOrder(order_env,target.category,area,snap.mapping.mapping_version,item.inventory_id,budget)
                self.stage=Stage.GRASP;return
            except c.ContractError as error:
                self._candidate_rejections.append(dict(target=target.target_id,reason=str(error)))
                continue
        if self.observations>=self.config.observation_attempts:
            self.stage=Stage.STOPPED;self.end_reason=self._reason(c.ErrorCode.UNAVAILABLE,'candidate','bounded observations found no valid candidate; partial finish')
        else:self.observations+=1;self.stage=Stage.OBSERVE

    def _dispatch(self,frame,snap,now,e):
        if self.stage in (Stage.SELECT,Stage.REVALIDATE,Stage.STOPPED):return None
        if self.stage in (Stage.NAV_BOARD,Stage.NAV_CENTRAL,Stage.RETURN):
            guards.ready(frame.readiness,snap,e,now,self.config.evidence_age_s,c.Holding.EMPTY)
        if self.stage in (Stage.CARRY,Stage.PLACE):
            c.require(self.order is not None and snap.holding.state is c.Holding.HELD and snap.holding.target_id==self.order.envelope.target_id,
                      'holding','the same work-order brick must still be held')
        if self.stage is Stage.CARRY:guards.ready(frame.readiness,snap,e,now,self.config.evidence_age_s,c.Holding.HELD)
        if self.stage is Stage.PLACE:guards.stopped(frame.stop,e,now,self.config.evidence_age_s)
        if frame.affected_targets:
            c.require(frame.stillness is not None,'stillness','collapse revokes all movement before 500ms new-frame proof')
            guards.stopped(frame.stop,e,now,self.config.evidence_age_s)
            r.validate_stillness(frame.stillness,e,now,frame.affected_targets,frame.scene_epoch)
        self.sequence+=1
        request=self.order.envelope.request_id if self.order else 'p09-stage-'+str(self.sequence)
        target=self.order.envelope.target_id if self.order else None
        env=self._envelope(snap,action='p09-action-'+str(self.sequence),target=target,request=request)
        if self.order:self.order=replace(self.order,envelope=replace(env,action_id=None))
        self.pending=Directive(self.stage,env,self.order,OWNERS[self.stage])
        return self.pending

    def _output(self,snap,issues,directive,stop=None):
        now=self.clock.now();e=self._envelope(snap)
        arrivals=arrival_counts(snap)
        layout=guards.all_layouts(snap,e,now,self.config.evidence_age_s)
        phase=m.MissionPhase.STOPPED if self.stage is Stage.STOPPED else (m.MissionPhase.RECOVERY if self.blocked else
            m.MissionPhase.WAIT_MAPPING if self.stage in (Stage.NAV_BOARD,Stage.READ_BOARD) else
            m.MissionPhase.ORDINARY if self.cake_closed else m.MissionPhase.CAKE)
        unknown=tuple(i.inventory_id for i in snap.inventory)+('match:physical-score-facts',)
        score=m.ScoreEstimate(float(self.exit_points) if self.exit_evidence else None,unknown,
            self.exit_evidence.evidence if self.exit_evidence else (),'P09-conditional-v1')
        decision=m.MissionDecision(e,phase,self.order,arrivals.complete,layout,score,
            m.Budget('mission-recovery',self.config.recovery_attempts,self.config.recovery_total_s,self.recovery_attempts,self.recovery_seconds),
            self.end_reason or self.blocked)
        return MissionOutput(decision,directive,self.stage,tuple(issues),stop,
                             self._perception_counts,self._field_decision)

    def _stop_request(self,snapshot):
        self.sequence+=1
        stop=m.StopRequest(self._envelope(snapshot),self.end_reason or self.blocked,snapshot.holding)
        if self.stop_sink is not None:
            try:self.stop_sink(stop)
            except Exception as error:
                self.blocked=self._reason(c.ErrorCode.UNAVAILABLE,'stop_sink',str(error))
        return stop

    def _field_step(self,frame,snap,now,e):
        from .field_strategy import FieldInput,FieldDecision
        field_input=frame.field_input
        if self.field_strategy is not None:
            gate=self.field_strategy.pending_handling_gate(field_input,now.seconds,self.config.evidence_age_s)
            if gate is not None:
                self._field_decision=gate
                if not gate.requires_stop:return None
                stop=m.StopRequest(self._envelope(snap),self._reason(c.ErrorCode.EXPIRED,'handling',gate.reason),snap.holding)
                if self.stop_sink is not None:self.stop_sink(stop)
                return stop
            if field_input is None and self.field_strategy.push_active:
                field_input=FieldInput('tick',self.field_strategy.active_target,self.field_strategy.regions.frame_id)
        if field_input is None:return None
        c.require(self.field_strategy is not None,'field_strategy','configured offline strategy adapter required',c.ErrorCode.UNAVAILABLE)
        c.require(self.pending is None,'field_strategy','normal action must finish before field recovery')
        if field_input.kind=='select':
            c.require(self.stage is Stage.SELECT,'field_strategy','selection only at a stopped selection boundary')
            guards.stopped(frame.stop,e,now,self.config.evidence_age_s)
            guards.ready(frame.readiness,snap,e,now,self.config.evidence_age_s,c.Holding.EMPTY)
        if field_input.kind=='handling':
            observation=field_input.handling_observation
            c.require(observation is not None,'handling','explicit physical observation required')
            if observation.phase=='probe_start':
                c.require(self._perception_policy is not None and field_input.observation_id==self._perception_policy.observation_id and
                          any(v.eligible and v.entity_id==field_input.exploratory_entity_id and
                              (v.target_id is None or v.target_id==field_input.target_id)
                              for v in self._perception_policy.exploratory_candidates),
                          'probe','current P04 local contact/height/approach evidence required')
                guards.stopped(frame.stop,e,now,self.config.evidence_age_s)
            if observation.phase=='temporary_release':
                matches=[i for i in snap.inventory if field_input.target_id in i.target_ids]
                c.require(len(matches)==1 and matches[0].identity_confirmed is c.Fact.CONFIRMED and
                          matches[0].location is m.InventoryLocation.REGION and snap.mapping is not None and
                          matches[0].area is snap.mapping.destination(matches[0].category),
                          'temporary_release','committed correctly located physical inventory required')
                item=matches[0]
                c.require(any(d.key==m.DeliveryKey(e.run_id,item.inventory_id) and d.area is item.area for d in snap.deliveries),
                          'temporary_release','actual linked release receipt required')
                places=[o for o in snap.space.occupancies if o.target_id==field_input.target_id]
                c.require(len(places)==1 and places[0].state is m.OccupancyState.ACTUAL and places[0].area is item.area,
                          'temporary_release','actual released occupancy required')
                c.require(self.field_strategy.regions.temporary_ground_ok(item.area.value,places[0].footprint_m,places[0].error_m,
                          True,True,True,frame_id=places[0].frame_id),
                          'temporary_release','committed occupancy must be in transport temporary ground')
        self._field_decision=self.field_strategy.advance(field_input,snap,now.seconds)
        if self._field_decision.action=='push_step':
            self.field_strategy.geometry_barrier['base_motion_epoch']=e.base_motion_epoch
        self._field_expires_s=min(e.expires_at.seconds,now.seconds+self.config.evidence_age_s)
        if field_input.push_observation is not None and self.field_strategy.push.limits.valid():
            self._field_expires_s=min(self._field_expires_s,field_input.push_observation.captured_at+
                                     self.field_strategy.push.limits.max_observation_age_s)
        if field_input.handling_observation is not None:
            self._field_expires_s=min(self._field_expires_s,field_input.handling_observation.observed_at_s+self.config.evidence_age_s)
        if not self._field_decision.requires_stop:return None
        stop=m.StopRequest(self._envelope(snap),self._reason(c.ErrorCode.OCCUPANCY_UNRESOLVED,
            'field_recovery',self._field_decision.reason),snap.holding)
        if self.stop_sink is not None:self.stop_sink(stop)
        return stop

    def advance(self,frame):
        snap=frame.inputs.snapshot;issues=[];directive=None;stop=None;self._candidate_rejections=[]
        self._field_decision=None;self._perception_policy=None
        self._field_expires_s=None
        self._perception_counts=(0,0,0) if frame.perception_policy is not None else None
        try:
            self._require_field_strategy()
            snap,now,e=self._snapshot(frame)
            if frame.perception_policy is not None or frame.perception_result is not None:
                from .field_strategy import consume_perception
                self._perception_counts=consume_perception(frame,now)
                self._perception_policy=frame.perception_policy
            if snap.run.state in (m.RunState.STOPPING,m.RunState.ENDED) or r.remaining_seconds(snap.run,now)<=0:
                self.stage=Stage.STOPPED;self.pending=None
                self.end_reason=self._reason(c.ErrorCode.DEADLINE,'run','original deadline or runtime stop; no further action')
            elif snap.run.state not in (m.RunState.RUNNING,m.RunState.OFFLINE):
                self.blocked=self._reason(c.ErrorCode.NOT_AUTHORIZED,'runtime','runtime not ready')
            if frame.affected_targets or any(l.quality is c.Fact.REJECTED for l in snap.layouts):
                self.blocked=self._reason(c.ErrorCode.OCCUPANCY_UNRESOLVED,'anomaly','revoke all sources; current stop/stillness and P10A reconciliation required')
            if self.pending and self.pending.stage in (Stage.NAV_BOARD,Stage.NAV_CENTRAL,Stage.CARRY,Stage.RETURN):
                wanted=c.Holding.HELD if self.pending.stage is Stage.CARRY else c.Holding.EMPTY
                if snap.holding.state is not wanted:
                    self.blocked=self._reason(c.ErrorCode.OCCUPANCY_UNRESOLVED,'holding','holding changed during navigation; revoke movement')
                else:
                    guards.ready(frame.readiness,snap,e,now,self.config.evidence_age_s,wanted)
                    if wanted is c.Holding.HELD:
                        c.require(self.order is not None and snap.holding.target_id==self.order.envelope.target_id,
                                  'holding','held identity changed during transport',c.ErrorCode.UNKNOWN_IDENTITY)
            for ex in frame.inputs.exits:
                try:self._exit(ex,snap,now,e)
                except c.ContractError as error:issues.append(str(error))
            report=arrival_counts(snap)
            if report.review:self.blocked=self._reason(c.ErrorCode.LIMIT_EXCEEDED,'arrival','; '.join(report.review))
            if not self.blocked and self.stage is not Stage.STOPPED and self._field_decision is None:
                for event in frame.inputs.action_events:
                    try:self._consume(event,frame,snap,now,e)
                    except c.ContractError as error:
                        issues.append(str(error))
                        if error.code not in (c.ErrorCode.STALE_REQUEST,c.ErrorCode.STALE_CONTROL,c.ErrorCode.STALE_RUN,c.ErrorCode.EXPIRED,c.ErrorCode.VERSION_MISMATCH):
                            self.blocked=self._reason(error.code,error.field,error.detail)
                    if self.blocked:break
                if not self.blocked and self.pending is None:
                    if self.stage is not Stage.REVALIDATE:stop=self._field_step(frame,snap,now,e)
                if not self.blocked and self.pending is None and self._field_decision is None:
                    if self.order is None and self.stage is Stage.SELECT and self.cake_closed and report.complete is c.Fact.CONFIRMED and guards.all_layouts(snap,e,now,self.config.evidence_age_s) is c.Fact.CONFIRMED:
                        self.stage=Stage.STOPPED;self.end_reason=self._reason(c.ErrorCode.CANCELLED,'finish','all deliveries and current layouts complete')
                    elif self.stage is Stage.SELECT:self._select(frame,snap,now,e)
                    directive=self._dispatch(frame,snap,now,e)
        except c.ContractError as error:
            self.blocked=self._reason(error.code,error.field,error.detail);issues.append(str(error));directive=None
        self.trace.append(dict(stage=self.stage.value,request=self.order.envelope.request_id if self.order else None,
            directive=directive.envelope.action_id if directive else None,issues=issues,blocked=self.blocked.detail if self.blocked else None,
            snapshot_revision=snap.revision,action_events=[c.dumps(x) for x in frame.inputs.action_events],
            exits=[c.dumps(x) for x in frame.inputs.exits],candidate_rejections=self._candidate_rejections))
        self.trace[-1]['perception_counts']=self._perception_counts
        self.trace[-1]['field_action']=self._field_decision.action if self._field_decision else None
        if self.blocked or self.end_reason:stop=self._stop_request(snap);directive=None
        try:
            c.require(self.journal.save(self.checkpoint()) is c.Fact.CONFIRMED,'checkpoint','checkpoint commit unknown; dispatch withheld',c.ErrorCode.STORAGE_FAILURE)
        except Exception as error:
            self.blocked=self._reason(c.ErrorCode.STORAGE_FAILURE,'checkpoint',str(error));directive=None
        # A slow journal/adapter cannot make a previously valid directive current.
        try:
            now=self.clock.now()
            if r.remaining_seconds(snap.run,now)<=0:
                self.stage=Stage.STOPPED;self.pending=None;directive=None
                self._field_decision=None
                self.end_reason=self._reason(c.ErrorCode.DEADLINE,'run','deadline elapsed during processing')
            elif directive is not None and directive.envelope.expires_at-now<=0:
                self.blocked=self._reason(c.ErrorCode.EXPIRED,'directive','dispatch expired during checkpoint');directive=None
            elif self._field_decision is not None and self._field_expires_s is not None and now.seconds>=self._field_expires_s:
                self.blocked=self._reason(c.ErrorCode.EXPIRED,'field_decision','observation expired during checkpoint')
                self._field_decision=None
        except c.ContractError as error:
            self.blocked=self._reason(error.code,error.field,error.detail);directive=None
        if (self.blocked or self.end_reason) and stop is None:stop=self._stop_request(snap)
        if self.blocked:self._field_decision=None
        return self._output(snap,issues,directive,stop)

    def step(self,inputs):
        """C0 facade fails closed without the missing assembly facts."""
        return self.advance(MissionFrame(inputs)).decision

    def checkpoint(self):
        controller=self._bound_field_strategy or self.field_strategy
        return dict(schema=1,run=c.dumps(self.run) if self.run else None,stage=self.stage.value,
            order=c.dumps(self.order) if self.order else None,
            pending=dict(stage=self.pending.stage.value,envelope=c.dumps(self.pending.envelope),owner=self.pending.owner.value) if self.pending else None,
            cake_closed=self.cake_closed,exit=c.dumps(self.exit_evidence) if self.exit_evidence else None,
            sequence=self.sequence,events=dict(self.events),observations=self.observations,
            recovery_attempts=self.recovery_attempts,recovery_seconds=self.recovery_seconds,
            snapshot_revision=self.last_revision,
            last_now=c.dumps(self.last_now) if self.last_now else None,mapping=c.dumps(self.mapping) if self.mapping else None,
            trace=list(self.trace),field_strategy=controller.checkpoint(self.clock.now().seconds)
            if controller is not None else self._field_state)

    def _restore(self,data):
        c.require(type(data) is dict and data.get('schema')==1,'checkpoint','unsupported checkpoint')
        if 'field_strategy' not in data:data=dict(data,field_strategy=None)
        expected=set(self.checkpoint())
        c.require(set(data)==expected and data['run'] is not None,'checkpoint','missing or unknown checkpoint fields')
        c.require(type(data['cake_closed']) is bool and all(type(data[k]) is int and data[k]>=0 for k in
            ('sequence','observations','recovery_attempts')),'checkpoint','invalid bool/cumulative counters')
        c.require(type(data['snapshot_revision']) is int and data['snapshot_revision']>=0,'checkpoint','saved revision required')
        c.require(data['stage'] in {s.value for s in Stage} and type(data['events']) is dict and type(data['trace']) is list,
                  'checkpoint','invalid stage/audit container')
        import math
        c.require(type(data['recovery_seconds']) in (int,float) and math.isfinite(data['recovery_seconds']) and data['recovery_seconds']>=0,
                  'checkpoint','invalid cumulative duration')
        self.run=c.loads(data['run'])
        c.require(type(self.run) is m.RunContext,'checkpoint.run','expected original RunContext')
        r.require_simulated(self.run)
        self._field_state=data['field_strategy']
        if self._field_state is not None:
            c.require(self.field_strategy is not None,'field_strategy','saved field policy requires a compatible controller',c.ErrorCode.UNAVAILABLE)
            self.field_strategy.restore(self._field_state,self.clock.now().seconds)
            self.field_strategy.bind_run(self.run)
            self._bound_field_strategy=self.field_strategy
        self.order=c.loads(data['order']) if data['order'] else None
        self.exit_evidence=c.loads(data['exit']) if data['exit'] else None
        self.mapping=c.loads(data['mapping']) if data['mapping'] else None
        self.cake_closed=data['cake_closed'];self.sequence=data['sequence'];self.events=dict(data['events'])
        self.observations=data['observations'];self.recovery_attempts=data['recovery_attempts'];self.recovery_seconds=data['recovery_seconds']
        self.last_now=c.loads(data['last_now']) if data['last_now'] else None;self.trace=list(data['trace'])
        self.last_revision=data['snapshot_revision']
        c.require(type(self.last_now) is c.ClockStamp and self.last_now-self.run.started_at>=0,'checkpoint.clock','original trusted clock required')
        for record,typ in ((self.order,m.WorkOrder),(self.exit_evidence,m.ExitEvidence),(self.mapping,m.MappingSnapshot)):
            if record is not None:
                c.require(type(record) is typ,'checkpoint.type','unexpected public record type')
                r.require_simulated(record)
                c.require(record.envelope.run_id==self.run.envelope.run_id,'checkpoint.run','foreign-run context')
        if self.exit_evidence is not None:
            ex=self.exit_evidence
            c.require(ex.envelope.producer is c.Owner.P03 and ex.area is m.Area.A and ex.entire_envelope_outside is c.Fact.CONFIRMED and
                      bool(ex.evidence) and self.run.deadline-ex.envelope.emitted_at>0 and ex.envelope.emitted_at-self.run.started_at>=0,
                      'checkpoint.exit','invalid historical exit')
            c.require(all(ev.kind is c.EvidenceKind.PHYSICAL and ev.run_id==self.run.envelope.run_id for ev in ex.evidence),
                      'checkpoint.exit','historical physical proof required')
        for key,value in self.events.items():
            c.require(type(key) is str and type(value) is str,'checkpoint.events','invalid event index')
            event=c.loads(value)
            c.require(type(event) is m.ActionEvent and event.envelope.run_id==self.run.envelope.run_id and
                      key==event.envelope.producer.value+':'+event.envelope.event_id,'checkpoint.events','wrong event identity')
        if data['pending']:
            p=data['pending']
            c.require(type(p) is dict and set(p)=={'stage','envelope','owner'} and p['stage'] in {s.value for s in OWNERS},'checkpoint.pending','invalid pending action')
            self.pending=Directive(Stage(p['stage']),c.loads(p['envelope']),self.order,c.Owner(p['owner']))
            c.require(type(self.pending.envelope) is c.Envelope and self.pending.owner is OWNERS[self.pending.stage] and
                      self.pending.envelope.run_id==self.run.envelope.run_id,'checkpoint.pending','pending identity/type mismatch')
        self.stage=Stage.REVALIDATE
        self.blocked=self._reason(c.ErrorCode.UNAVAILABLE,'recovery','read disk; stop/fence and fresh physical reconciliation required; never replay pending action')

    def resume(self,frame,review):
        """Explicit new-instance reconciliation. Never infer an action did not happen."""
        c.require(self.stage is Stage.REVALIDATE and self.restored is not None,'recovery','restored checkpoint required')
        self._require_field_strategy()
        snap,now,e=self._snapshot(frame)
        c.require(snap.run.envelope.instance_id!=self.run.envelope.instance_id and
                  snap.run.envelope.control_epoch>self.run.envelope.control_epoch and snap.run.ever_restarted,
                  'control','P10A/P10B new controller registration required')
        c.require(snap.run.state in (m.RunState.RUNNING,m.RunState.OFFLINE) and r.remaining_seconds(self.run,now)>0,
                  'runtime','runtime resume not authorized or original deadline elapsed')
        r.validate_envelope(review.envelope,guards.context(e,now))
        c.require(review.envelope.producer is c.Owner.P10A and review.result.phase=='mission_reconciled' and
                  review.result.status is c.Outcome.SUCCESS and review.result.physical is c.Fact.CONFIRMED,
                  'review','P10A fresh physical reconciliation required')
        guards.physical(review.result.evidence,review.envelope,now,self.config.evidence_age_s)
        guards.stopped(frame.stop,e,now,self.config.evidence_age_s)
        if frame.affected_targets:
            c.require(frame.stillness is not None,'stillness','500ms certificate required before retry')
            r.validate_stillness(frame.stillness,e,now,frame.affected_targets,frame.scene_epoch)
        guards.ready(frame.readiness,snap,e,now,self.config.evidence_age_s,c.Holding.EMPTY)
        for ev in review.result.evidence:
            c.require(self.restored['last_now'] is not None and ev.captured_at-c.loads(self.restored['last_now'])>0,
                      'review','observation predates interruption')
        if self.order:
            key=m.DeliveryKey(e.run_id,self.order.inventory_id)
            c.require(any(d.key==key for d in snap.deliveries),'order','unresolved interrupted order needs P07/P08 rebind; absence of receipt is not failure')
            lay=next((l for l in snap.layouts if l.area is self.order.destination),None)
            guards.layout_prefix(lay,snap,e,now,self.config.evidence_age_s,self.order.envelope.target_id)
            if self.order.category is m.Category.CAKE:self.cake_closed=True
            self.order=replace(self.order,envelope=replace(e,target_id=self.order.envelope.target_id,request_id='p09-resume-'+str(self.sequence),action_id=None))
            self.stage=Stage.RETURN
        else:self.stage=Stage.NAV_CENTRAL if snap.mapping else Stage.NAV_BOARD
        self.run=snap.run;self.pending=None;self.blocked=None;self.restored=None
        return self.advance(replace(frame,inputs=replace(frame.inputs,action_events=(),exits=())))

    def retry_current(self,frame,review):
        """Bounded retry only after a distinct current P10A reconciliation event."""
        c.require(self.blocked is not None and self.pending is not None and self.stage is not Stage.REVALIDATE,
                  'retry','pending interrupted action required')
        c.require(self.blocked.code is not c.ErrorCode.STORAGE_FAILURE,'retry','unknown storage requires readback/recovery, never action retry',c.ErrorCode.STORAGE_FAILURE)
        snap,now,e=self._snapshot(frame)
        c.require(self.recovery_attempts<self.config.recovery_attempts and
                  self.recovery_seconds+self.config.recovery_s<=self.config.recovery_total_s and
                  r.remaining_seconds(snap.run,now)>=self.config.recovery_s+self.config.stop_margin_s,
                  'budget','cumulative retry budget exhausted',c.ErrorCode.LIMIT_EXCEEDED)
        r.validate_envelope(review.envelope,guards.context(self.pending.envelope,now,identities=True))
        c.require(review.envelope.producer is c.Owner.P10A and review.result.phase=='action_reconciled' and
                  review.result.status is c.Outcome.SUCCESS and review.result.physical is c.Fact.CONFIRMED,
                  'retry','explicit physical reconciliation required')
        guards.physical(review.result.evidence,review.envelope,now,self.config.evidence_age_s)
        c.require(all(ev.captured_at-self.pending.envelope.emitted_at>0 for ev in review.result.evidence),'retry','new review required')
        cost=max(review.result.elapsed_s,now-self.pending.envelope.emitted_at)
        c.require(self.recovery_seconds+cost+self.config.recovery_s<=self.config.recovery_total_s,
                  'budget','elapsed recovery plus next reserve exceeds cumulative limit',c.ErrorCode.LIMIT_EXCEEDED)
        guards.stopped(frame.stop,e,now,self.config.evidence_age_s)
        if frame.affected_targets:
            c.require(frame.stillness is not None,'stillness','500ms current stillness certificate required')
            r.validate_stillness(frame.stillness,e,now,frame.affected_targets,frame.scene_epoch)
        if self.pending.stage in (Stage.GRASP,Stage.NAV_BOARD,Stage.NAV_CENTRAL,Stage.RETURN,Stage.OBSERVE):
            guards.ready(frame.readiness,snap,e,now,self.config.evidence_age_s,c.Holding.EMPTY)
        # A retry of grasp also requires the same brick positively observed in CENTRAL.
        if self.pending.stage is Stage.GRASP:
            target=self.order.envelope.target_id
            candidate=next((t for t in frame.inputs.candidates if t.target_id==target),None)
            c.require(candidate is not None,'retry','same target positively reconciled at source required')
            item=self._candidate(candidate,frame,snap,now,e)
            c.require(item.inventory_id==self.order.inventory_id,'retry','original inventory identity required')
            c.require(self.config.can_start(r.remaining_seconds(snap.run,now)),'budget','retry must reserve complete order and stop',c.ErrorCode.LIMIT_EXCEEDED)
        self.recovery_attempts+=1;self.recovery_seconds+=cost
        self.stage=self.pending.stage;self.pending=None;self.blocked=None
        return self.advance(replace(frame,inputs=replace(frame.inputs,action_events=(),exits=())))
