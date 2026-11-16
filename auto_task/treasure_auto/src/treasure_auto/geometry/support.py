"""Actual convex support union with conservative geometric uncertainty bounds."""
import math
from treasure_auto.contracts import core as c, models as m, rules
from . import planar
from .math3d import finite


def compute_support(request,proof,max_contact_gap_m):
    rules.require_space_version(request.envelope,request.space)
    finite(max_contact_gap_m,'contact_gap',0.)
    upper,lowers=request.upper,request.lowers
    ids=tuple(x.target_id for x in lowers)
    c.require(bool(lowers) and len(ids)==len(set(ids)) and upper.target_id not in ids,'supports','distinct actual lower objects required')
    snapshot={o.target_id:o for o in request.space.occupancies}
    c.require(all(snapshot.get(o.target_id)==o for o in (upper,)+lowers),'space','support geometry differs from shared snapshot',c.ErrorCode.CONFLICT)
    c.require(all(o.frame_id==upper.frame_id for o in lowers),'frame','support coordinates must share map frame')
    p=planar.convex(tuple((x.x,x.y) for x in upper.footprint_m));total=planar.area(p)
    polygons=[planar.convex(tuple((p.x,p.y) for p in o.footprint_m)) for o in lowers]
    supported=min(total,planar.union_area([planar.intersection(p,q) for q in polygons]))
    valid=(proof is not None and proof.upper_target_id==upper.target_id and
           set(proof.lower_target_ids)==set(ids) and len(proof.lower_target_ids)==len(ids) and proof.space_version==request.space.version)
    c.require(valid,'contact_proof','proof must bind this exact upper/support set and space',c.ErrorCode.EVIDENCE_MISSING)
    for ev in request.contact_evidence+proof.evidence:
        rules.require_physical_evidence(ev,request.envelope,request.envelope.emitted_at,request.envelope.origin)
    contact=proof.contact;horizontal=proof.horizontal;reason=None;ratio=None
    has_contact=(contact is c.Fact.CONFIRMED and horizontal is c.Fact.CONFIRMED and
                 bool(proof.evidence) and bool(request.contact_evidence) and
                 all(o.state is m.OccupancyState.ACTUAL for o in (upper,)+lowers) and
                 all(abs(upper.z_min_m-o.z_max_m)<=max_contact_gap_m+upper.error_m+o.error_m for o in lowers))
    if has_contact:
        guaranteed=planar.erode(p,upper.error_m)
        actual_lower=[planar.erode(q,o.error_m) for q,o in zip(polygons,lowers)]
        numerator=planar.union_area([planar.intersection(guaranteed,q) if q else () for q in actual_lower])
        perimeter=sum(math.hypot(a[0]-b[0],a[1]-b[1]) for a,b in zip(p,p[1:]+p[:1]))
        denominator=total+perimeter*upper.error_m+math.pi*upper.error_m**2
        # The public bound itself must be conservative; downstream C0 consumers
        # compare directly with .8 and need not call this module's helper.
        ratio=max(0.,min(1.,numerator/denominator)-1e-12)
    else:
        reason=c.Reason(c.ErrorCode.EVIDENCE_MISSING,'contact','projected overlap lacks compatible height/contact/horizontal evidence')
        if contact is c.Fact.CONFIRMED:contact=c.Fact.UNKNOWN
    return m.SupportGeometryResult(request.envelope,upper.target_id,ids,total,supported,ratio,
        contact,horizontal,tuple(dict.fromkeys(request.contact_evidence+proof.evidence)),reason)


def passes_quality(result):
    # Roundoff must not promote an exact 0.80 boundary to a success.
    return (result.contact is c.Fact.CONFIRMED and result.horizontal is c.Fact.CONFIRMED and
            result.ratio_lower_bound is not None and result.ratio_lower_bound>.8)
