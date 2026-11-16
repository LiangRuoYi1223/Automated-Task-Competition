"""C0 PlacementPort facade plus explicit local metric/adaptation entrypoints."""
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m, rules as r
from . import guards
from .targets import validate_request,near_center,target_for_slot
from .inputs import PlacementBinding
from .space import SpaceProposals,accept_committed
from .evidence import ReleaseWindow
from .layout import LayoutTracker


class PlacementService:
    def __init__(self,config,run,clock,*,context_provider,state_provider,binding_provider,manipulation,executor=None,persistence=None):
        c.require(run.envelope.origin is c.Origin.SIMULATED,'origin','P08 field adapters are not commissioned',c.ErrorCode.NOT_AUTHORIZED)
        self.config=config;self.run=run;self.clock=clock;self.context_provider=context_provider;self.state_provider=state_provider
        self.binding_provider=binding_provider;self.manipulation=manipulation;self.executor=executor;self.persistence=persistence
        self.spaces=SpaceProposals(config);self.active_request=None;self.window=None;self.pending_holding=None;self.pending_result=None
        self.stop_request=None;self.return_permitted=False;self._fingerprints={};self._results={};self.last_reason=None
        self._temporary_regions=None

    def _context(self):return replace(self.context_provider(),now=self.clock.now())

    def _layout(self,req,state):
        existing=next((l for l in state.layouts if l.area is req.region.area),None)
        if existing is not None:return existing
        return m.LayoutSnapshot(replace(req.envelope,producer=c.Owner.P08),req.region.area,m.LayoutKind(req.envelope.versions.layout),
            req.envelope.layout_recovery_round or 0,req.envelope.rebuild_generation or 0,0,(),c.Fact.UNKNOWN,c.Fact.UNKNOWN)

    def _result(self,req,state,status,phase,why=None,release=None,proposed=None,holding=None,layout=None):
        result=c.ActionResult(status,phase,why,c.Fact.REJECTED if status is c.Outcome.REJECTED else c.Fact.CONFIRMED,
            c.Fact.UNKNOWN,c.Fact.UNKNOWN,c.Fact.CONFIRMED if release is not None else c.Fact.UNKNOWN,
            release.evidence if release is not None else (),0.)
        return m.PlacementResult(replace(req.envelope,producer=c.Owner.P08),result,release,layout or self._layout(req,state),
            proposed or state.space,holding or state.holding,None)

    def place(self,request):
        return self._place(request)

    def place_temporary(self,request,field_regions,*,free_observed=False,approach_clear=False,regrasp_clear=False):
        """Explicit transport-ground route using the same release/commit guards.

        Current metric adapters must establish clear ground and access. The
        complete correct area is used for arrival evidence; a temporary release
        never advances collection layout quality.
        """
        return self._place(request,(field_regions,free_observed,approach_clear,regrasp_clear))

    def _temporary_target(self,request,target,state,temporary):
        regions,free,approach,regrasp=temporary
        from treasure_auto.navigation.field_regions import FieldRegions
        c.require(type(regions) is FieldRegions,'regions','configured full scoring/collection regions required')
        c.require(request.region.frame_id==regions.frame_id and
            request.region.polygon_m==regions.full_areas[request.region.area.value] and
            request.region.error_m>=regions.error_m,
            'region','temporary release requires exact configured full destination area')
        c.require(not target.support_ids and abs(target.z_min_m-self.config.ground_z_m)+target.error_m<=self.config.height_tolerance_m,
                  'ground','temporary brick must rest on verified ground')
        others=tuple(o for o in state.space.occupancies if o.target_id!=target.target_id)
        c.require(all(o.frame_id==regions.frame_id for o in others),'space.frame','all obstacles require same map frame')
        margin=target.error_m+max([o.error_m for o in others]+[0.])
        c.require(regions.temporary_ground_ok(request.region.area,target.footprint_m,margin,free,approach,regrasp,
                  tuple(o.footprint_m for o in others),target.frame_id),
                  'temporary','transport ground/approach/regrasp clearance unconfirmed',c.ErrorCode.OCCUPANCY_UNRESOLVED)

    def _place(self,request,temporary=None):
        state=self.state_provider();key=request.envelope.action_id
        fingerprint=c.dumps(request)+(':collection' if temporary is None else ':temporary:'+repr(temporary))
        if key in self._fingerprints:
            c.require(self._fingerprints[key]==fingerprint,'action_id','same action, changed payload',c.ErrorCode.ID_CONFLICT)
            return self._results[key]
        try:
            ctx=self._context()
            c.require(self.active_request is None,'action','previous placement/commit unresolved',c.ErrorCode.OCCUPANCY_UNRESOLVED)
            c.require(r.remaining_seconds(self.run,ctx.now)>0,'deadline','original cutoff reached',c.ErrorCode.DEADLINE)
            c.require(state.run.started_at==self.run.started_at and state.run.deadline==self.run.deadline and
                state.run.envelope.run_id==ctx.run_id and state.run.envelope.instance_id==ctx.instance_id and state.run.envelope.control_epoch==ctx.control_epoch,
                'snapshot','current committed run required')
            c.require(state.mapping==request.mapping and state.holding==request.holding,'snapshot','mapping and P07 holding must match committed state',c.ErrorCode.CONFLICT)
            c.require(self.binding_provider is not None,'binding','P06/P02 geometry binding adapter missing',c.ErrorCode.UNAVAILABLE)
            binding=self.binding_provider(request,state);ctx=self._context()
            c.require(type(binding) is PlacementBinding and binding.path_id==request.plan.path_id and binding.held_offset==request.holding.tool_offset,
                      'binding','plan must bind actual brick target and measured held offset')
            c.require(binding.base_pose.frame_id==request.region.frame_id,'base_pose','current P02 worksite pose required')
            guards.physical(binding.localization_evidence,replace(request.envelope,target_id=None,action_id=None),ctx.now,self.config)
            validate_request(request,ctx,state.space,binding.target,self.config)
            layout=self._layout(request,state)
            if temporary is None:
                expected=target_for_slot(layout,state.space,request.region,binding.target,self.config,ctx.now).occupancy
                c.require(all(getattr(expected,f)==getattr(binding.target,f) for f in
                    ('footprint_m','z_min_m','z_max_m','support_ids','frame_id','area')),
                    'target.geometry','plan target must match this slot and current actual supports')
                if layout.confirmed_count==0:c.require(near_center(binding.target,request.region,self.config),'target.center','first brick/cake uses region center')
            else:self._temporary_target(request,binding.target,state,temporary)
            existing=next((o for o in state.space.occupancies if o.target_id==binding.target.target_id),None)
            if existing is None:
                proposal=self.spaces.reserve(state.space,request.envelope,binding.target,ctx.now)
                return self._result(request,state,c.Outcome.UNKNOWN,'await_reservation_commit',
                    c.Reason(c.ErrorCode.UNAVAILABLE,'reservation','P09/P10B commit and P06 replan against new version required'),proposed=proposal)
            c.require(existing==binding.target and existing.state is m.OccupancyState.RESERVED,'reservation','exact committed reservation required')
            c.require(self.executor is not None,'executor','P06 single-stage/P10A/P01/P10B adapters missing',c.ErrorCode.UNAVAILABLE)
            self._fingerprints[key]=fingerprint;self.active_request=request;self.window=ReleaseWindow(request,self.config,ctx.now)
            self._temporary_regions=temporary
            execution_state=self.executor.start(request)
            out=self._result(request,state,c.Outcome.UNKNOWN,execution_state,
                getattr(self.executor,'reason',None) or c.Reason(c.ErrorCode.EVIDENCE_MISSING,'release','actual separation not yet observed'))
            self._results[key]=out;return out
        except c.ContractError as error:
            return self._result(request,state,c.Outcome.REJECTED,'rejected',c.Reason(error.code,error.field,error.detail))

    def observe_metrics(self,sample):
        c.require(self.active_request is not None,'request','no active placement')
        req=self.active_request;ctx=self._context();state=self.state_provider()
        try:
            r.validate_envelope(req.envelope,ctx);r.require_space_version(req.envelope,state.space)
            c.require(self.executor.state not in ('unresolved','recovery_required'),'executor','execution interrupted')
            c.require(r.remaining_seconds(self.run,ctx.now)>0,'deadline','original cutoff reached',c.ErrorCode.DEADLINE)
            if self.pending_result is not None:return self.pending_result
            c.require(m.Phase.PLACE in self.executor.completed_phases,'stage','supported placement must physically complete')
            if self._temporary_regions is not None:
                actual=next((o for o in sample.objects if o.target_id==req.envelope.target_id),None)
                c.require(actual is not None,'temporary','actual released target must be observed',c.ErrorCode.EVIDENCE_MISSING)
                self._temporary_target(req,actual,state,self._temporary_regions)
            event=self.window.push(sample,ctx,self.run)
            if event is None or m.Phase.WITHDRAW not in self.executor.completed_phases:
                out=self._result(req,state,c.Outcome.UNKNOWN,'await_release',self.window.last_reason or
                    c.Reason(c.ErrorCode.EVIDENCE_MISSING,'release','new cup/brick separation window incomplete'))
                self._results[req.envelope.action_id]=out;return out
            proposed=self.spaces.observe(state.space,sample,ctx)
            update=self.manipulation.propose_release(event,state.holding);ctx=self._context()
            c.require(self.executor.state not in ('unresolved','recovery_required'),'executor','cancelled during P07 callback')
            c.require(r.remaining_seconds(self.run,ctx.now)>0,'deadline','cutoff during P07 callback',c.ErrorCode.DEADLINE)
            refreshed=self.state_provider();ctx=self._context()
            c.require(self.executor.state not in ('unresolved','recovery_required'),'executor','cancelled during state readback')
            c.require(refreshed==state,'snapshot','state changed during P07 callback',c.ErrorCode.CONFLICT)
            c.require(r.remaining_seconds(self.run,ctx.now)>0,'deadline','cutoff during state readback',c.ErrorCode.DEADLINE)
            r.validate_envelope(event.envelope,ctx)
            c.require(type(update) is m.HoldingUpdate and update.envelope.producer is c.Owner.P07 and update.current.state is c.Holding.EMPTY and
                      update.previous is state.holding.state and update.cause is m.HoldingCause.RELEASE,'holding_update','P07 release proposal required')
            r.validate_envelope(update.envelope,ctx)
            self.pending_holding=update
            self.pending_result=self._result(req,state,c.Outcome.SUCCESS,'release_observed_pending_commit',release=event,
                proposed=proposed,holding=update.current)
            self._results[req.envelope.action_id]=self.pending_result
            return self.pending_result
        except c.ContractError as error:
            self.last_reason=c.Reason(error.code,error.field,error.detail)
            # Never clear or shrink physical extent on missing observations.
            proposed=self.spaces.mark_unknown(state.space,req.envelope,req.envelope.target_id,ctx.now) if req.envelope.versions.space==state.space.version and req.envelope.expires_at-ctx.now>0 else state.space
            out=self._result(req,state,c.Outcome.UNKNOWN,'unresolved',self.last_reason,proposed=proposed)
            self._results[req.envelope.action_id]=out;return out

    def accept_commit(self,transaction_id,receipt):
        c.require(self.pending_result is not None and self.pending_holding is not None and self.persistence is not None,
                  'commit','release proposal and real readback adapter required')
        read=self.persistence.read(self.run.envelope.run_id)
        c.require(read.status is c.Outcome.SUCCESS and read.snapshot is not None,'read','cannot resolve commit outcome',c.ErrorCode.STORAGE_FAILURE)
        snap=read.snapshot;release=self.pending_result.release
        accept_committed(transaction_id,receipt,self.pending_result.proposed_space,snap)
        c.require(snap.holding==self.pending_holding.current and any(d.key==release.delivery_key for d in snap.deliveries),
                  'transaction','release, occupancy, holding and arrival must publish atomically')
        self.manipulation.accept_holding_commit(receipt)
        if self.executor is not None:self.executor.finish_committed(receipt,snap)
        result=replace(self.pending_result,result=replace(self.pending_result.result,phase='release_committed'))
        self._results[self.active_request.envelope.action_id]=result
        self.active_request=None;self.pending_result=None;self.pending_holding=None;self.window=None;self._temporary_regions=None
        # No automatic arm motion or driving. The next P09 work order still
        # requires newly committed P07 held state and current geometry.
        return result

    def propose_space(self,observations,previous):
        raise c.ContractError(c.ErrorCode.UNAVAILABLE,'metric_geometry',
            'C0 PerceptionResult has pixels, not map footprint/height/orientation/coverage; P04/P06 adapter required')

    def assess_layout(self,snapshot,previous):
        # Occupancy alone does not prove release, contact or stability.
        return replace(previous,quality=c.Fact.UNKNOWN)

    def assess_stillness(self,observations,stop,snapshot):
        self.last_reason=c.Reason(c.ErrorCode.UNAVAILABLE,'metric_frames','C0 lacks normalized capture time, coverage and metric scene history')
        return None

    def assess_metric_layout(self,event,snapshot,facts,region,previous):
        tracker=LayoutTracker(previous,self.config)
        return tracker.record_placement(event,snapshot,facts,region,self._context())

    def notify_anomaly(self,event,previous,confirmed,full_collapse):
        self.return_permitted=False
        if self.executor is not None:self.executor.cancel('suspected/confirmed collapse')
        holding=self.active_request.holding if self.active_request is not None else c.HoldingSnapshot(c.Holding.UNKNOWN,None,None,())
        self.stop_request=m.StopRequest(replace(event.envelope,producer=c.Owner.P08),
            c.Reason(c.ErrorCode.OCCUPANCY_UNRESOLVED,'collapse','revoke navigation, precision, recovery/debug motion and old queues'),holding)
        tracker=LayoutTracker(previous,self.config)
        return tracker.failure(event,confirmed,full_collapse,self._context())
