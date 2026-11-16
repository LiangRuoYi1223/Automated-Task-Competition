"""P06 consumer gates; callers cannot refresh an old result by re-wrapping it."""
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m, rules
from .inputs import GeometryInputs
from .math3d import validate_transform, rotate
from .calibration import CalibrationReport, make_report


def generic_context(context,epoch=True):
    return replace(context,request_id=None,action_id=None,target_id=None,
                   base_motion_epoch=context.base_motion_epoch if epoch else None)


def evidence_refs(items,env,now,field,target=None):
    c.require(bool(items),field,'fresh physical evidence required',c.ErrorCode.EVIDENCE_MISSING)
    for ev in items:
        rules.require_physical_evidence(ev,replace(env,target_id=target,action_id=None),now,env.origin)
    return tuple(items)


def current_inputs(env,ctx,config):
    c.require(type(ctx) is GeometryInputs,'inputs','explicit P06 wiring context required',c.ErrorCode.UNCONFIGURED)
    config.require_ready(env.origin)
    rules.validate_envelope(env,ctx.consumer)
    c.require(env.request_id is not None and env.base_motion_epoch is not None and env.target_id is not None,
              'request','active request, target and base epoch required')
    c.require(config.version==env.versions.config,'config.version','configured version differs',c.ErrorCode.VERSION_MISMATCH)
    loc=ctx.localization
    rules.validate_envelope(loc.envelope,generic_context(ctx.consumer))
    c.require(loc.envelope.producer is c.Owner.P02 and loc.trusted is c.Fact.CONFIRMED and
              loc.pose is not None and loc.base_motion_epoch==env.base_motion_epoch and
              loc.pose.frame_id==config.map_frame,'localization','current trusted P02 pose/epoch required',c.ErrorCode.STALE_GEOMETRY)
    c.require(ctx.stopped is c.Fact.CONFIRMED,'stopped','actual base stop unconfirmed',c.ErrorCode.EVIDENCE_MISSING)
    for refs,field in ((loc.evidence,'localization.evidence'),(ctx.stop_evidence,'stop.evidence'),(ctx.coverage_evidence,'space.coverage'),(ctx.start_evidence,'start.evidence')):
        evidence_refs(refs,env,ctx.consumer.now,field)
        for ev in refs:c.require(ctx.consumer.now-ev.received_at<=config.max_pose_age_s,field,'old observation',c.ErrorCode.EXPIRED)
    rules.require_space_version(env,ctx.space)
    rules.validate_envelope(ctx.space.envelope,generic_context(ctx.consumer,False))
    c.require(ctx.space.envelope.producer is c.Owner.P08,'space.producer','P08 owns spatial facts')
    validate_transform(ctx.camera_to_base,target=config.base_frame)
    validate_transform(ctx.base_to_arm,config.base_frame,config.arm_frame)
    validate_transform(ctx.tool_transform,'cup','arm_reference')
    c.require(rotate(ctx.base_to_arm,(0.,0.,1.))==(0.,0.,1.) and rotate(ctx.tool_transform,(0.,0.,1.))==(0.,0.,1.),
              'normal','finite horizontal paths require aligned vertical axes; tilt unsupported')
    c.require(ctx.start.frame_id==config.arm_frame,'start.frame','measured starting point in arm frame required')
    c.require(type(ctx.arm_geometry_version) is str and bool(ctx.arm_geometry_version.strip()),'arm_geometry','versioned robot geometry required')
    rules.validate_envelope(ctx.holding_envelope,generic_context(ctx.consumer))
    c.require(ctx.holding_envelope.producer is c.Owner.P07,'holding.producer','P07 owns holding facts')
    c.require(ctx.holding_envelope.target_id==ctx.holding.target_id,'holding.target','P07 envelope and holding target differ',c.ErrorCode.STALE_REQUEST)
    return ctx


def calibration_report(ctx,env,chain,config):
    report=ctx.image_report if chain=='image' else ctx.placement_report
    c.require(type(report) is CalibrationReport and report.accepted,'calibration.report','independent validated report missing/failed',c.ErrorCode.UNCONFIGURED)
    c.require(report.chain==chain and report.version==env.versions.calibration and
              report.camera_version==ctx.camera_to_base.version and report.tool_version==ctx.tool_transform.version,
              'calibration.version','calibration camera/tool/chain version mismatch',c.ErrorCode.VERSION_MISMATCH)
    binding=report.binding
    c.require(binding is not None,'calibration.binding','report must bind actual parameters, not only their labels',c.ErrorCode.UNCONFIGURED)
    c.require((binding.camera_to_base,binding.base_to_arm,binding.tool_transform,binding.config)==
              (ctx.camera_to_base,ctx.base_to_arm,ctx.tool_transform,config),
              'calibration.binding','transform/config contents differ from independently validated binding',c.ErrorCode.VERSION_MISMATCH)
    checked=make_report(report.version,report.chain,report.camera_version,report.tool_version,report.domain_mm,
                        report.fit_samples,report.validation_samples,report.limit_mm,binding)
    c.require(checked==report,'calibration.report','report contents inconsistent',c.ErrorCode.CONFLICT)
    for s in report.fit_samples+report.validation_samples:
        c.require(s.evidence.origin is env.origin,'calibration.origin','calibration provenance mismatch',c.ErrorCode.SIMULATED_EVIDENCE)
    return report


def within_domain(point,report):
    c.require(all(lo+point.error_mm<=v<=hi-point.error_mm for lo,v,hi in zip(report.domain_mm[0],(point.x,point.y,point.z),report.domain_mm[1])),
              'calibration.domain','point/error envelope outside independently validated domain')


def unchanged(before,after):
    # Time may advance; any physical, versioned, request or provenance field change
    # forces recomputation. Occupancy is never mutated or cleared here.
    return replace(before,consumer=replace(before.consumer,now=after.consumer.now))==after
