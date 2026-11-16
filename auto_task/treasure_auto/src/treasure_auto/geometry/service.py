"""C0 GeometryPort implementation. Calculations only; no device commands.

Single worker per instance. Revalidate returned plans immediately before every
stage via validate_plan; P10A/P07/P08 remain the execution and evidence owners.
"""
import math
import uuid
from dataclasses import asdict, replace
from treasure_auto.contracts import core as c, models as m, rules
from . import guards, planar
from .math3d import (map_to_arm, arm_to_map, apply, inverse, angle_distance, finite)
from .surface import resolve_plane, solve_top, tool_target
from .support import compute_support
from .collision import check_path


class GeometryService:
    def __init__(self,config,inputs_provider,motion_verifier,audit_sink, *, space_readback=None):
        self.config=config;self.inputs_provider=inputs_provider;self.motion_verifier=motion_verifier;self.audit_sink=audit_sink
        self.space_readback=space_readback
        self._seen={};self._cache={};self._plans={};self._contexts={};self._sequence=0

    def _inputs(self,env):
        before=self.inputs_provider()
        rules.require_committed_space(before.space,self.space_readback)
        current=self.inputs_provider()
        c.require(current.space==before.space,'space','space changed during publication read',c.ErrorCode.STALE_GEOMETRY)
        return guards.current_inputs(env,current,self.config)

    def _refs(self,ctx,env,refs,field,target=None):
        return guards.evidence_refs(refs,env,ctx.consumer.now,field,target)

    def _common(self,request,ctx):
        c.require(request.space==ctx.space,'space','request is not the current committed snapshot',c.ErrorCode.CONFLICT)
        if hasattr(request,'tool_transform'):
            c.require(request.tool_transform==ctx.tool_transform,'tool_transform','tool changed; entire plan invalid',c.ErrorCode.STALE_GEOMETRY)

    def _grasp(self,req,ctx):
        cfg=self.config;env=req.envelope;obs=req.observation;top=obs.top
        c.require(ctx.holding.state is c.Holding.EMPTY and req.held_offset is None,'holding','second pick prohibited unless P07 confirms EMPTY',c.ErrorCode.EVIDENCE_MISSING)
        self._refs(ctx,env,ctx.holding.evidence,'holding.evidence')
        c.require(obs.identity_confirmed is c.Fact.CONFIRMED and obs.target_id==env.target_id and top is not None,
                  'observation','known physical target and real top required',c.ErrorCode.UNKNOWN_IDENTITY)
        c.require(ctx.image is not None and ctx.pixels is not None,'image','capture/processing history missing',c.ErrorCode.UNCONFIGURED)
        image=ctx.image
        rules.validate_envelope(image.envelope,guards.generic_context(ctx.consumer))
        c.require(obs.observation_id==image.observation_id==ctx.pixels.observation_id and
                  req.capture_pose==image.capture_pose and req.current_pose==ctx.localization.pose and
                  ctx.pixels.preprocessing==image.preprocessing and image.distortion==top.distortion,
                  'image.provenance','capture pose, observation or processing chain mismatch',c.ErrorCode.STALE_GEOMETRY)
        c.require(ctx.capture_now is not None and ctx.capture_floor is not None,'capture.clock','sensor clock/barrier missing',c.ErrorCode.UNCONFIGURED)
        self._refs(ctx,env,(image.image,),'image.evidence',env.target_id)
        age=ctx.capture_now-image.image.captured_at
        c.require(0<=age<cfg.max_pose_age_s and image.image.captured_at-ctx.capture_floor>0,
                  'capture','stale or pre-motion frame',c.ErrorCode.EXPIRED)
        c.require(top.frame_id==image.camera_frame and (top.image_width,top.image_height)==(image.width,image.height),
                  'image','top does not belong to this camera resolution')
        self._refs(ctx,env,obs.evidence+top.evidence,'observation.evidence',env.target_id)
        c.require(image.image in top.evidence and image.image in obs.evidence,'observation.evidence','real corners must reference exact source image')
        plane=resolve_plane(ctx.planes,cfg);self._refs(ctx,env,plane.evidence,'plane.evidence')
        report=guards.calibration_report(ctx,env,'image',cfg)
        c.require(req.calibration==report.binding.camera,'calibration.binding',
                  'camera parameter contents differ from independently validated binding',c.ErrorCode.VERSION_MISMATCH)
        c.require(req.calibration.version==report.version and req.calibration.validation_error_mm>=report.max_error_mm,
                  'calibration','camera calibration/report mismatch',c.ErrorCode.VERSION_MISMATCH)
        required={s.evidence.evidence_id for s in report.validation_samples}
        c.require(required<=set(e.evidence_id for e in req.calibration.independent_validation),
                  'calibration.independent_validation','camera report validation refs missing',c.ErrorCode.EVIDENCE_MISSING)
        geometry=solve_top(top,req.calibration,ctx.camera_to_base,ctx.base_to_arm,plane,ctx.pixels,cfg,report.max_error_mm)
        for point in geometry.corners+(geometry.center,):guards.within_domain(point,report)
        # Recover capture-base coordinates, rotate/translate through map, then current base.
        capture_xyz=apply(inverse(ctx.base_to_arm),(geometry.center.x,geometry.center.y,geometry.center.z))
        co,si=math.cos(req.capture_pose.yaw),math.sin(req.capture_pose.yaw)
        map_pose=replace(req.capture_pose,x=req.capture_pose.x+(co*capture_xyz[0]-si*capture_xyz[1])/1000.,
                         y=req.capture_pose.y+(si*capture_xyz[0]+co*capture_xyz[1])/1000.)
        xyz=map_to_arm(map_pose,capture_xyz[2],req.current_pose,ctx.base_to_arm)
        lever=math.hypot(*capture_xyz[:2]);err=geometry.center.error_mm+1000*(req.capture_pose.error_m+req.current_pose.error_m)
        err+=lever*(req.capture_pose.error_rad+req.current_pose.error_rad)+cfg.tool_error_mm
        center=c.ArmPoint(cfg.arm_frame,*xyz,'mm',err)
        guards.within_domain(center,report)
        target=tool_target(center,req.tool_transform,None,cfg.fixed_tool_yaw_rad)
        return target,report,dict(surface=asdict(geometry),plane=asdict(plane),chain='image->capture_base->map->current_base->arm->tool'),tuple(dict.fromkeys(obs.evidence+top.evidence+plane.evidence))

    def _placement(self,req,ctx):
        cfg=self.config;env=req.envelope;h=ctx.holding
        c.require(type(ctx.approximate_center) is bool,'approximate_center','explicit boolean control assumption required')
        c.require(h.state is c.Holding.HELD and h.target_id==env.target_id and h.tool_offset==req.held_offset,
                  'holding','current P07 actual held offset required',c.ErrorCode.EVIDENCE_MISSING)
        self._refs(ctx,env,h.evidence,'holding.evidence',env.target_id)
        c.require(ctx.held_yaw_rad is not None,'held_yaw','transported orientation unknown',c.ErrorCode.EVIDENCE_MISSING)
        finite(ctx.held_yaw_rad,'held_yaw')
        c.require(req.current_base_pose==ctx.localization.pose and req.desired_brick_pose.frame_id==cfg.map_frame and req.region.frame_id==cfg.map_frame,
                  'placement.frame','current map/base pose required',c.ErrorCode.STALE_GEOMETRY)
        plane=resolve_plane(ctx.planes,cfg);self._refs(ctx,env,plane.evidence,'plane.evidence')
        c.require(set(plane.support_ids)==set(req.actual_support_ids) and len(req.actual_support_ids)==len(set(req.actual_support_ids)),
                  'supports','placement plane must bind exact current support set')
        for name in req.actual_support_ids:
            matches=[o for o in ctx.space.occupancies if o.target_id==name]
            c.require(len(matches)==1 and matches[0].state is m.OccupancyState.ACTUAL and
                      abs(matches[0].z_max_m*1000.-plane.height_mm)<=cfg.plane_tolerance_mm,
                      'supports','missing/unknown/current support height conflict',c.ErrorCode.EVIDENCE_MISSING)
            # A matching top height elsewhere on the field is not a support here.
            co,si=math.cos(req.desired_brick_pose.yaw),math.sin(req.desired_brick_pose.yaw)
            half=cfg.brick_size_mm[0]/2000.
            footprint=tuple((req.desired_brick_pose.x+co*x-si*y,req.desired_brick_pose.y+si*x+co*y)
                            for x,y in ((-half,-half),(half,-half),(half,half),(-half,half)))
            lower=planar.convex(tuple((p.x,p.y) for p in matches[0].footprint_m))
            c.require(planar.area(planar.intersection(footprint,lower))>1e-12,
                      'supports.footprint','named support does not overlap planned bottom',c.ErrorCode.EVIDENCE_MISSING)
        report=guards.calibration_report(ctx,env,'placement',cfg)
        mount_yaw=math.atan2(ctx.tool_transform.matrix[4],ctx.tool_transform.matrix[0])
        arm_yaw=math.atan2(ctx.base_to_arm.matrix[4],ctx.base_to_arm.matrix[0])
        needed=req.desired_brick_pose.yaw-req.current_base_pose.yaw+arm_yaw-mount_yaw-ctx.held_yaw_rad
        c.require(angle_distance(needed,cfg.fixed_tool_yaw_rad,math.pi/2)<=cfg.yaw_tolerance_rad,
                  'placement.yaw','C0 point-only path cannot command an unverified wrist rotation',c.ErrorCode.UNCONFIGURED)
        xyz=map_to_arm(req.desired_brick_pose,plane.height_mm+cfg.brick_size_mm[2],req.current_base_pose,ctx.base_to_arm)
        lever=math.hypot(xyz[0],xyz[1]);err=report.max_error_mm+plane.error_mm+cfg.tool_error_mm
        err+=1000*(req.current_base_pose.error_m+req.desired_brick_pose.error_m)+lever*(req.current_base_pose.error_rad+req.desired_brick_pose.error_rad)
        center=c.ArmPoint(cfg.arm_frame,*xyz,'mm',err)
        guards.within_domain(center,report)
        polygon=planar.convex(tuple((p.x,p.y) for p in req.region.polygon_m))
        # Circumscribed radius contains every orientation of the square and the error.
        discrepancy=math.hypot(req.held_offset.x,req.held_offset.y) if ctx.approximate_center else 0.
        radius=math.hypot(*cfg.brick_size_mm[:2])/2000.+(err+req.held_offset.error_mm+discrepancy)/1000.+req.region.error_m
        c.require(planar.clearance(polygon,(req.desired_brick_pose.x,req.desired_brick_pose.y))>radius,
                  'placement.region','whole brick/error envelope must remain in destination')
        # Approximate the planar control center, never the physical extent.
        # Keep measured vertical offset so support contact height is not guessed.
        # check_path still receives ctx.holding.tool_offset (the actual offset).
        control_offset=replace(req.held_offset,x=0.,y=0.,error_mm=req.held_offset.error_mm+discrepancy) if ctx.approximate_center else req.held_offset
        target=tool_target(center,req.tool_transform,control_offset,cfg.fixed_tool_yaw_rad)
        return target,report,dict(plane=asdict(plane),desired_brick_pose=c.to_data(req.desired_brick_pose),held_yaw_rad=ctx.held_yaw_rad,
                                 approximate_center=ctx.approximate_center,actual_held_offset=c.to_data(req.held_offset),
                                 chain='map bottom center->current_base->arm brick top->actual held offset->tool'),h.evidence+plane.evidence

    def _segments(self,target,placement):
        cfg=self.config
        c.require(target.error_mm<=cfg.max_error_mm,'target.error','accumulated uncertainty exceeds contact limit')
        specs=((m.Phase.APPROACH,0.,cfg.approach_mm),(m.Phase.DESCEND,0.,cfg.descend_mm),
               (m.Phase.PLACE if placement else m.Phase.CONTACT,0.,0.),
               (m.Phase.WITHDRAW if placement else m.Phase.TRIAL_LIFT,0.,cfg.lift_mm),
               (m.Phase.WITHDRAW,-cfg.withdraw_mm,cfg.lift_mm))
        return tuple(m.PathSegment(phase,replace(target,x=target.x+dx,z=target.z+dz),speed,timeout)
                     for (phase,dx,dz),speed,timeout in zip(specs,cfg.speeds_mm_s,cfg.timeouts_s))

    def _envelope(self,env,ctx,evidence=(),use_image=False):
        now=ctx.consumer.now;expiry=env.expires_at
        dependencies=(ctx.localization.envelope,ctx.space.envelope,ctx.holding_envelope)
        if use_image:dependencies+=(ctx.image.envelope,)
        for source in dependencies:
            if source.expires_at-expiry<0:expiry=source.expires_at
        for e in evidence:
            if e.valid_until-expiry<0:expiry=e.valid_until
        for ev in ctx.localization.evidence+ctx.stop_evidence+ctx.coverage_evidence+ctx.start_evidence:
            bound=replace(ev.received_at,seconds=ev.received_at.seconds+self.config.max_pose_age_s)
            if bound-expiry<0:expiry=bound
        if use_image:
            bound=replace(now,seconds=now.seconds+self.config.max_pose_age_s-(ctx.capture_now-ctx.image.image.captured_at))
            if bound-expiry<0:expiry=bound
        c.require(expiry-now>0,'expires_at','inputs expired before publication',c.ErrorCode.EXPIRED)
        self._sequence+=1
        return replace(env,producer=c.Owner.P06,event_id='p06:'+uuid.uuid4().hex,sequence=self._sequence,emitted_at=now,expires_at=expiry)

    def _failure(self,req,error,support=False):
        reason=c.Reason(error.code,error.field,error.detail) if isinstance(error,c.ContractError) else c.Reason(c.ErrorCode.UNAVAILABLE,'geometry',str(error))
        env=replace(req.envelope,producer=c.Owner.P06,event_id='p06:rejected:'+uuid.uuid4().hex)
        if support:return m.SupportGeometryResult(env,req.upper.target_id,tuple(o.target_id for o in req.lowers),None,None,None,c.Fact.UNKNOWN,c.Fact.UNKNOWN,(),reason)
        status=c.Outcome.UNKNOWN if reason.code in (c.ErrorCode.EVIDENCE_MISSING,c.ErrorCode.UNAVAILABLE,c.ErrorCode.UNCONFIGURED) else c.Outcome.REJECTED
        return m.GeometryResult(env,c.ActionResult(status,'geometry',reason,c.Fact.REJECTED,c.Fact.UNKNOWN,c.Fact.UNKNOWN,c.Fact.UNKNOWN,(),0.),None)

    def _audit(self,req,result,diagnostics):
        try:
            c.require(self.audit_sink is not None,'audit','durable diagnostic sink required',c.ErrorCode.STORAGE_FAILURE)
            self.audit_sink(dict(contract_version=c.CONTRACT_VERSION,request=c.to_data(req),result=c.to_data(result),diagnostics=diagnostics))
        except Exception as exc:
            raise c.ContractError(c.ErrorCode.STORAGE_FAILURE,'audit','result not published: '+str(exc)) from exc

    def _run(self,req,placement=False,support=False):
        try:
            ctx=self._inputs(req.envelope);self._common(req,ctx)
            key=(req.envelope.run_id,req.envelope.instance_id,req.envelope.control_epoch,req.envelope.event_id)
            signature=c.dumps(req)
            c.require(key not in self._seen or self._seen[key]==signature,'event_id','same ID carries different request',c.ErrorCode.ID_CONFLICT)
            self._seen[key]=signature
            if key in self._cache:
                result=self._cache[key]
                if support:
                    rules.validate_envelope(result.envelope,ctx.consumer)
                    c.require(guards.unchanged(self._contexts[key],ctx),'support','support facts changed',c.ErrorCode.STALE_GEOMETRY)
                else:
                    check=self.validate_plan(result.plan)
                    if not check.accepted:raise c.ContractError(check.reason.code,check.reason.field,check.reason.detail)
                return result
            if support:
                proof=ctx.contact_proof
                c.require(proof is not None,'contact','contact/horizontal proof port unavailable',c.ErrorCode.EVIDENCE_MISSING)
                self._refs(ctx,req.envelope,req.contact_evidence+proof.evidence,'contact.evidence',req.upper.target_id)
                result=compute_support(req,proof,self.config.max_contact_gap_mm/1000.)
                env=self._envelope(req.envelope,ctx,result.evidence)
                result=replace(result,envelope=env);diagnostics=dict(chain='shared actual support polygons')
            else:
                target,report,diagnostics,refs=self._placement(req,ctx) if placement else self._grasp(req,ctx)
                segments=self._segments(target,placement)
                for segment in segments:guards.within_domain(segment.target,report)
                refs+=check_path(segments,ctx,self.config,self.motion_verifier,placement,
                                lambda es,f:self._refs(ctx,req.envelope,es,f))
                refs=tuple(dict.fromkeys(refs+ctx.stop_evidence+ctx.localization.evidence+ctx.coverage_evidence+ctx.holding.evidence+ctx.start_evidence))
                env=self._envelope(req.envelope,ctx,refs,use_image=not placement)
                plan=m.GeometryPlan(env,'p06-path:'+uuid.uuid4().hex,target,segments,c.Fact.CONFIRMED,c.Fact.CONFIRMED,refs)
                result=m.GeometryResult(env,c.ActionResult(c.Outcome.SUCCESS,'geometry',None,c.Fact.CONFIRMED,c.Fact.UNKNOWN,c.Fact.UNKNOWN,c.Fact.UNKNOWN,refs,0.),plan)
                diagnostics.update(calibration=asdict(report),tool=c.to_data(ctx.tool_transform),arm_geometry_version=ctx.arm_geometry_version)
            latest=self._inputs(req.envelope)
            c.require(guards.unchanged(ctx,latest),'inputs','context changed during geometry calculation',c.ErrorCode.STALE_GEOMETRY)
            self._audit(req,result,diagnostics)
            latest=self._inputs(req.envelope)
            c.require(guards.unchanged(ctx,latest),'inputs','context changed during diagnostic commit',c.ErrorCode.STALE_GEOMETRY)
            rules.validate_envelope(result.envelope,latest.consumer)
            if not support:
                self._plans[result.plan.path_id]=(result.plan,ctx,req,self.config)
            self._cache[key]=result;self._contexts[key]=ctx
            return result
        except Exception as error:
            result=self._failure(req,error,support)
            try:self._audit(req,result,dict(rejected=True))
            except c.ContractError as exc:result=self._failure(req,exc,support)
            return result

    def plan(self,request):return self._run(request)
    def plan_placement(self,request):return self._run(request,placement=True)
    def support_geometry(self,request):return self._run(request,support=True)

    def validate_plan(self,plan,stage_index=0):
        try:
            c.require(plan is not None and plan.path_id in self._plans,'plan','unknown/restored plan cannot recover motion authority',c.ErrorCode.STALE_GEOMETRY)
            original,ctx,req,config=self._plans[plan.path_id]
            c.require(plan==original and self.config==config,'plan','plan/config altered',c.ErrorCode.CONFLICT)
            latest=self._inputs(req.envelope)
            rules.validate_envelope(plan.envelope,latest.consumer)
            if type(req) is m.GeometryRequest:
                rules.validate_envelope(latest.image.envelope,guards.generic_context(latest.consumer))
            c.require(type(stage_index) is int and 0<=stage_index<len(plan.segments),'stage_index','valid next stage required')
            expected=ctx.start if stage_index==0 else plan.segments[stage_index-1].target
            distance=math.sqrt(sum((getattr(latest.start,a)-getattr(expected,a))**2 for a in ('x','y','z')))
            certified_error=max(expected.error_mm,plan.segments[stage_index].target.error_mm)
            c.require(latest.start.error_mm<=self.config.max_error_mm and distance+latest.start.error_mm<=certified_error+1e-9,
                      'start','observed start is outside the certified stage envelope',c.ErrorCode.STALE_GEOMETRY)
            normalized=replace(latest,start=ctx.start,start_evidence=ctx.start_evidence)
            c.require(guards.unchanged(ctx,normalized),'plan','base/camera/tool/space/request facts changed; all stages invalid',c.ErrorCode.STALE_GEOMETRY)
            return m.ContractCheck(True,None)
        except Exception as error:
            reason=c.Reason(error.code,error.field,error.detail) if isinstance(error,c.ContractError) else c.Reason(c.ErrorCode.UNAVAILABLE,'plan',str(error))
            return m.ContractCheck(False,reason)
