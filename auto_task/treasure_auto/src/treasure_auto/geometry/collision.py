"""Conservative continuous AABB sweeps, including payload and uncertainty.

The injected robot verifier must certify joint limits, continuous full-arm sweep,
self collision and static body/camera/fence clearance for the exact MotionQuery.
It is not supplied by this package. These checks cover the shared dynamic space.
"""
import math
from itertools import product
from treasure_auto.contracts import core as c, models as m
from .inputs import MotionQuery, MotionCertificate, SweepBox
from .math3d import finite, map_to_arm, apply


def box(points,margin=0.):
    return SweepBox(tuple(min(p[i] for p in points)-margin for i in range(3)),
                    tuple(max(p[i] for p in points)+margin for i in range(3)))


def validate_box(b):
    c.require(type(b) is SweepBox and len(b.minimum_mm)==len(b.maximum_mm)==3,'sweep','explicit 3D box required')
    for lo,hi in zip(b.minimum_mm,b.maximum_mm):
        finite(lo,'sweep');finite(hi,'sweep');c.require(lo<=hi,'sweep','inverted box')


def expanded(b,error):
    validate_box(b)
    return SweepBox(tuple(v-error for v in b.minimum_mm),tuple(v+error for v in b.maximum_mm))


def overlaps(a,b):
    return all(a.minimum_mm[i]<=b.maximum_mm[i] and b.minimum_mm[i]<=a.maximum_mm[i] for i in range(3))


def _tool_point(point,ctx,config,offset=None):
    local=apply(ctx.tool_transform,(0.,0.,0.) if offset is None else (offset.x,offset.y,offset.z))
    co,si=math.cos(config.fixed_tool_yaw_rad),math.sin(config.fixed_tool_yaw_rad)
    return (point.x+co*local[0]-si*local[1],point.y+si*local[0]+co*local[1],point.z+local[2])


def check_path(segments,ctx,config,verifier,placement,check_evidence):
    c.require(verifier is not None,'robot_geometry','full-path robot verifier unavailable',c.ErrorCode.UNAVAILABLE)
    c.require(ctx.coverage is c.Fact.CONFIRMED and bool(ctx.coverage_evidence),'space.coverage',
              'empty occupancy list does not establish known free space',c.ErrorCode.EVIDENCE_MISSING)
    obstacles=[]
    for o in ctx.space.occupancies:
        c.require(o.frame_id==config.map_frame,'space.frame','occupancy is not in configured map')
        vertices=[map_to_arm(c.MapPose(o.frame_id,p.x,p.y,0.,'m','rad',0.,0.),z*1000.,ctx.localization.pose,ctx.base_to_arm)
                  for p,z in product(o.footprint_m,(o.z_min_m,o.z_max_m))]
        distance=max(math.hypot(p[0],p[1]) for p in vertices)
        err=o.error_m*1000.+ctx.localization.pose.error_m*1000.+distance*ctx.localization.pose.error_rad
        obstacles.append((o,box(vertices,err)))
    released=None
    if placement:
        top=_tool_point(segments[2].target,ctx,config,ctx.holding.tool_offset)
        radius=math.hypot(*config.brick_size_mm[:2])/2
        released=expanded(SweepBox((top[0]-radius,top[1]-radius,top[2]-config.brick_size_mm[2]),
                                   (top[0]+radius,top[1]+radius,top[2])),segments[2].target.error_mm)
    current=ctx.start;evidence=[]
    for n,segment in enumerate(segments):
        distance=math.sqrt(sum((getattr(current,a)-getattr(segment.target,a))**2 for a in ('x','y','z')))
        c.require(segment.timeout_s>=distance/segment.speed_mm_s,'segment.timeout','timeout shorter than minimum travel time')
        carrying=(n<3) if placement else (n>=3)
        offset=ctx.holding.tool_offset if placement else None
        error=max(current.error_mm,segment.target.error_mm)
        query=MotionQuery(current,segment,carrying,config.brick_size_mm,offset,error,
                          ctx.tool_transform,ctx.arm_geometry_version,ctx.space.version)
        cert=verifier(query)
        c.require(type(cert) is MotionCertificate and cert.query==query,'robot_geometry.certificate',
                  'certificate does not bind the exact stage, payload and versions',c.ErrorCode.ID_CONFLICT)
        c.require(cert.reachable is c.Fact.CONFIRMED,'reachable','joint/continuous path reachability not confirmed',c.ErrorCode.UNAVAILABLE)
        c.require(cert.full_arm_covered is c.Fact.CONFIRMED and cert.static_clearance is c.Fact.CONFIRMED and bool(cert.swept_boxes),
                  'sweep','full-arm and fixed robot/field clearance unconfirmed',c.ErrorCode.UNAVAILABLE)
        check_evidence(cert.evidence,'robot_geometry.evidence');evidence.extend(cert.evidence)
        components=[('arm',expanded(b,error)) for b in cert.swept_boxes]
        endpoints=(current,segment.target);cup=[_tool_point(p,ctx,config) for p in endpoints]
        shaft=cup+[(p.x,p.y,p.z) for p in endpoints]
        components.append(('tool',box(shaft,config.cup_radius_mm+error)))
        if carrying:
            tops=[_tool_point(p,ctx,config,offset) for p in endpoints]
            radius=math.hypot(*config.brick_size_mm[:2])/2
            payload=SweepBox((min(p[0] for p in tops)-radius-error,min(p[1] for p in tops)-radius-error,
                              min(p[2] for p in tops)-config.brick_size_mm[2]-error),
                             (max(p[0] for p in tops)+radius+error,max(p[1] for p in tops)+radius+error,max(p[2] for p in tops)+error))
            components.append(('payload',payload))
        # Hypothetical just-placed geometry is a path obstacle, not a committed
        # occupancy fact. No P08/P10B record is added, deleted or renamed.
        if released is not None and n>=3:
            for kind,sweep in components:
                if overlaps(sweep,released):
                    contact=(kind=='tool' and n==3 and
                             sweep.minimum_mm[2]>=released.maximum_mm[2]-2*error-config.cup_radius_mm-1e-8)
                    c.require(contact,'collision.release_exit','withdrawal intersects expected released brick',c.ErrorCode.OCCUPANCY_UNRESOLVED)
        for occupancy,obstacle in obstacles:
            for kind,sweep in components:
                if not overlaps(sweep,obstacle):continue
                # The same held brick is not a second obstacle; its old occupancy is
                # kept in the snapshot. Only the explicitly identified actual object.
                own=(occupancy.target_id==ctx.consumer.target_id and occupancy.state is m.OccupancyState.ACTUAL)
                if own and carrying and kind=='payload':continue
                contacts=ctx.planes[0].support_ids if placement else (ctx.consumer.target_id,)
                # Only intended tool/payload top contact, never the arm, unknown
                # objects, or arbitrary penetration. P07/P08 still prove real contact.
                permitted=(occupancy.target_id in contacts and occupancy.state is m.OccupancyState.ACTUAL and
                    kind!='arm' and n in (1,2,3) and
                    sweep.minimum_mm[2]>=obstacle.maximum_mm[2]-error-config.cup_radius_mm-2*occupancy.error_m*1000.-1e-8)
                if not permitted:
                    raise c.ContractError(c.ErrorCode.OCCUPANCY_UNRESOLVED,'collision.'+segment.phase.value,
                                          '{} sweep intersects {} ({})'.format(kind,occupancy.target_id,occupancy.state.value))
        current=segment.target
    return tuple(dict.fromkeys(evidence))
