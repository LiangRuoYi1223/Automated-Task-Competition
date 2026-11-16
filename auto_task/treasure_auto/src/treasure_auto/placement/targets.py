"""Layout targets in map metres; P06 alone certifies arm paths and offsets."""
import math
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m, rules as r
from treasure_auto.geometry import planar
from . import guards
from .inputs import TargetProposal


def whole_in_region(item,region):
    if item.frame_id!=region.frame_id:return False
    poly=guards.polygon(region.polygon_m)
    guards.polygon(item.footprint_m)
    return all(planar.clearance(poly,(p.x,p.y))>=item.error_m+region.error_m for p in item.footprint_m)


def center(item):
    return (sum(p.x for p in item.footprint_m)/len(item.footprint_m),sum(p.y for p in item.footprint_m)/len(item.footprint_m))


def near_center(item,region,config):
    x,y=center(item)
    return math.hypot(x-region.placement_center.x,y-region.placement_center.y)+item.error_m+region.placement_center.error_m<=config.center_tolerance_m


def target_for_slot(layout,space,region,footprint,config,now):
    c.require(layout.area is region.area,'area','layout/region mismatch')
    n=layout.confirmed_count+1
    c.require(n<= (1 if region.area is m.Area.A else 3),'slot','region layout already complete')
    c.require(footprint.frame_id==region.frame_id,'frame','brick metric frame differs')
    c.require(len(footprint.footprint_m)==4,'footprint','first version requires actual quadrilateral')
    guards.polygon(footprint.footprint_m)
    x,y=region.placement_center.x,region.placement_center.y
    z=config.ground_z_m;supports=()
    items={o.target_id:o for o in space.occupancies}
    bricks=sorted(layout.bricks,key=lambda b:b.slot)
    if n>1:
        c.require(len(bricks)==n-1 and [b.slot for b in bricks]==list(range(1,n)),'layout','continuous confirmed prefix required')
        required=bricks if layout.kind is m.LayoutKind.V2 and n==3 else (bricks[-1] if layout.kind is m.LayoutKind.V3 else bricks[0],)
        lowers=[]
        for b in required:
            c.require(b.slot==1 or b.placed_recovery_round==layout.layout_recovery_round,'recovery','old second cannot support new third',c.ErrorCode.STALE_RECOVERY)
            c.require(b.target_id in items,'supports','actual support missing',c.ErrorCode.OCCUPANCY_UNRESOLVED)
            item=items[b.target_id];guards.actual(item,space.envelope,now,config)
            c.require(item.frame_id==region.frame_id,'supports.frame','mixed map frames')
            lowers.append(item)
        if layout.kind is m.LayoutKind.V3 or (layout.kind is m.LayoutKind.V2 and n==3):
            x=sum(center(o)[0] for o in lowers)/len(lowers);y=sum(center(o)[1] for o in lowers)/len(lowers)
            c.require(max(o.z_max_m for o in lowers)-min(o.z_max_m for o in lowers)<=config.height_tolerance_m,
                      'supports.height','incompatible actual support heights')
            z=max(o.z_max_m for o in lowers);supports=tuple(o.target_id for o in lowers)
        else:
            first=lowers[0];x,y=center(first)
            span=max(p.x for p in first.footprint_m)-min(p.x for p in first.footprint_m)
            width=max(p.x for p in footprint.footprint_m)-min(p.x for p in footprint.footprint_m)
            x+=(1 if n==2 else -1)*(span/2+width/2+config.adjacent_gap_m)
    ox,oy=center(footprint)
    proposed=replace(footprint,state=m.OccupancyState.RESERVED,footprint_m=tuple(m.Point2(p.x+x-ox,p.y+y-oy) for p in footprint.footprint_m),
        z_min_m=z,z_max_m=z+footprint.z_max_m-footprint.z_min_m,support_ids=supports,area=region.area,observed_at=now,evidence=())
    c.require(whole_in_region(proposed,region),'region','entire brick plus uncertainty must fit')
    c.require(n!=1 or near_center(proposed,region,config),'center','first brick must fit center tolerance')
    return TargetProposal(n,c.MapPose(region.frame_id,x,y,0,'m','rad',proposed.error_m,0),proposed)


def validate_request(req,ctx,space,target,config):
    c.require(ctx.versions.config==config.version,'config.version','different configuration',c.ErrorCode.VERSION_MISMATCH)
    c.require(ctx.origin is c.Origin.SIMULATED,'origin','live operation not commissioned',c.ErrorCode.NOT_AUTHORIZED)
    r.validate_envelope(req.envelope,ctx);r.require_space_version(req.envelope,space)
    c.require(req.order.envelope.producer is c.Owner.P09,'order.producer','P09 work order required')
    r.validate_envelope(req.order.envelope,ctx)
    c.require(req.mapping.envelope.run_id==ctx.run_id and req.mapping.envelope.origin is ctx.origin and
        req.mapping.mapping_version==ctx.versions.mapping==req.order.mapping_version,'mapping','wrong or old frozen mapping',c.ErrorCode.INVALID_MAPPING)
    c.require(req.order.destination is req.mapping.destination(req.order.category) and req.region.area is req.order.destination,
        'region','wrong region',c.ErrorCode.INVALID_MAPPING)
    c.require(req.holding.state is c.Holding.HELD and req.holding.target_id==req.order.envelope.target_id and req.holding.tool_offset is not None,
        'holding','known same object and measured offset required',c.ErrorCode.OCCUPANCY_UNRESOLVED)
    guards.physical(req.holding.evidence,replace(req.envelope,action_id=None),ctx.now,config)
    guards.stopped(req.stop,ctx,config)
    r.validate_envelope(req.plan.envelope,ctx)
    c.require(req.plan.envelope.producer is c.Owner.P06 and req.plan.reachable is c.Fact.CONFIRMED and req.plan.collision_free is c.Fact.CONFIRMED,
        'plan','P06 reachability and full swept path required')
    phases=tuple(seg.phase for seg in req.plan.segments)
    c.require(phases==(m.Phase.APPROACH,m.Phase.DESCEND,m.Phase.PLACE,m.Phase.WITHDRAW),
              'plan.segments','approach, descent, supported placement and withdrawal required')
    guards.physical(req.plan.evidence,req.envelope,ctx.now,config)
    r.validate_envelope(req.permit.envelope,ctx)
    c.require(req.permit.source is m.MotionSource.ARM and req.permit.purpose=='place','permit','placement arm permit required')
    c.require(target.target_id==req.order.envelope.target_id and target.area is req.region.area and whole_in_region(target,req.region),
              'target','wrong target or entire brick outside region')
    return True
