"""Trust, identity, clock and numeric checks shared by P08 algorithms."""
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m, rules as r
from treasure_auto.geometry import planar
from .config import finite


def generic(ctx):
    return replace(ctx,request_id=None,action_id=None,target_id=None)


def context(env,now):
    return m.ConsumerContext(env.run_id,env.instance_id,env.control_epoch,now,env.versions,
        env.request_id,env.action_id,env.target_id,env.layout_recovery_round,env.rebuild_generation,env.base_motion_epoch,env.origin)


def physical(refs,env,now,config):
    c.require(bool(refs),'evidence','fresh physical observations required',c.ErrorCode.EVIDENCE_MISSING)
    for ev in refs:
        r.require_physical_evidence(ev,env,now,env.origin)
        c.monotonic(ev.captured_at,'captured_at')
        c.require(0<=now-ev.captured_at<=config.evidence_max_age_s,'captured_at','stale/future capture',c.ErrorCode.EXPIRED)
        c.require(ev.received_at-ev.captured_at>=0,'captured_at','capture after receipt')


def polygon(points):
    return planar.convex(tuple((p.x,p.y) for p in points))


def actual(item,env,now,config):
    c.require(item.state is m.OccupancyState.ACTUAL and item.visibility is m.Visibility.VISIBLE,
              'occupancy','fresh visible actual geometry required',c.ErrorCode.OCCUPANCY_UNRESOLVED)
    physical(item.evidence,replace(env,target_id=item.target_id,action_id=None),now,config)
    c.require(0<=now-item.observed_at<=config.evidence_max_age_s,'observed_at','old/future occupancy',c.ErrorCode.EXPIRED)
    polygon(item.footprint_m)


def frame(sample,ctx,config):
    c.require(ctx.versions.config==config.version,'config.version','different configuration',c.ErrorCode.VERSION_MISMATCH)
    c.require(ctx.origin is c.Origin.SIMULATED,'origin','P08 field adapters uncommissioned',c.ErrorCode.NOT_AUTHORIZED)
    r.validate_envelope(sample.envelope,generic(ctx))
    c.require(sample.envelope.producer in (c.Owner.P04,c.Owner.P06),'producer','metric source must be perception/geometry')
    c.require(type(sample.observation_id) is str and bool(sample.observation_id),'observation_id','required')
    c.require(type(sample.scene_epoch) is int and sample.scene_epoch>=0,'scene_epoch','nonnegative integer required')
    c.require(sample.identity_complete is c.Fact.CONFIRMED,'identity','ambiguous or missing identity',c.ErrorCode.UNKNOWN_IDENTITY)
    physical((sample.evidence,),replace(sample.envelope,target_id=None,action_id=None),ctx.now,config)
    c.require(sample.envelope.emitted_at-sample.evidence.received_at>=0,'emitted_at','message precedes observation')
    polygon(sample.coverage_polygon_m)
    ids=tuple(o.target_id for o in sample.objects)
    c.require(len(ids)==len(set(ids)) and len(sample.orientations)==len(ids),'objects','unique objects and complete orientations required')
    c.require({o[0] for o in sample.orientations}==set(ids),'orientations','each object orientation required')
    for _,yaw,tilt in sample.orientations:
        for x in (yaw,tilt):
            c.require(type(x) in (float,int) and __import__('math').isfinite(x),'orientation','finite radians required')
    for item in sample.objects:
        c.require(item.frame_id==sample.frame_id,'frame_id','mixed frames')
        actual(item,sample.envelope,ctx.now,config)
        c.require(item.observed_at==sample.evidence.captured_at,'observed_at','geometry must belong to this frame')
        c.require(all(e.captured_at==sample.evidence.captured_at for e in item.evidence),'objects.evidence','mixed exposure geometry')
    finite(sample.cup_error_m,'cup_error_m')
    if sample.cup_xyz_m is not None:
        c.require(type(sample.cup_xyz_m) is tuple and len(sample.cup_xyz_m)==3,'cup','three metric coordinates required')
        for v in sample.cup_xyz_m:
            c.require(type(v) in (float,int) and __import__('math').isfinite(v),'cup','finite coordinates required')


def stopped(stop,ctx,config):
    r.validate_envelope(stop.envelope,generic(ctx))
    c.require(stop.envelope.producer is c.Owner.P01,'stop.producer','P01 must certify old actuator queue')
    c.require(stop.physically_stopped is c.Fact.CONFIRMED and stop.old_queue_empty is c.Fact.CONFIRMED,
              'stop','physical stop and old queue clearance required',c.ErrorCode.EVIDENCE_MISSING)
    physical(stop.evidence,replace(stop.envelope,target_id=None,action_id=None),ctx.now,config)


def next_envelope(env,now,suffix,version=None):
    return replace(env,producer=c.Owner.P08,event_id=env.event_id+':'+suffix,emitted_at=now,
        versions=env.versions if version is None else replace(env.versions,space=version))
