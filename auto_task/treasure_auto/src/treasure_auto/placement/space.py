"""Pure proposals over the one committed snapshot. P10B publishes, never P08."""
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m, rules as r
from treasure_auto.geometry import planar
from . import guards


class SpaceProposals:
    def __init__(self,config):self.config=config

    def _next(self,previous,env,items,now,tag):
        r.require_space_version(env,previous)
        r.validate_envelope(env,guards.context(env,now))
        c.require(env.origin is c.Origin.SIMULATED,'origin','only simulation commissioned')
        e=guards.next_envelope(env,now,tag,previous.version+1)
        return m.SpaceSnapshot(e,previous.version+1,tuple(items))

    def reserve(self,previous,env,target,now):
        c.require(target.target_id==env.target_id and target.state is m.OccupancyState.RESERVED,'reservation','target-bound reservation required')
        guards.polygon(target.footprint_m)
        for old in previous.occupancies:
            if old.target_id==target.target_id:
                c.require(old==target and old.state is m.OccupancyState.RESERVED,'reservation','cannot replace actual/unknown occupancy',c.ErrorCode.CONFLICT)
                return previous
            c.require(old.frame_id==target.frame_id,'space.frame','mixed coordinate frames')
            if min(old.z_max_m,target.z_max_m)>max(old.z_min_m,target.z_min_m):
                # A conservative expanded AABB also includes uncertainty outside the polygon.
                def bounds(o):
                    return (min(p.x for p in o.footprint_m)-o.error_m,max(p.x for p in o.footprint_m)+o.error_m,
                            min(p.y for p in o.footprint_m)-o.error_m,max(p.y for p in o.footprint_m)+o.error_m)
                a,b=bounds(old),bounds(target)
                c.require(a[1]<=b[0] or b[1]<=a[0] or a[3]<=b[2] or b[3]<=a[2],
                          'reservation','intersects known/reserved/unknown space',c.ErrorCode.OCCUPANCY_UNRESOLVED)
        return self._next(previous,env,previous.occupancies+(target,),now,'reserve')

    def mark_unknown(self,previous,env,target_id,now):
        c.require(any(o.target_id==target_id for o in previous.occupancies),'target','cannot discard unknown physical extent')
        items=tuple(replace(o,state=m.OccupancyState.UNKNOWN,visibility=m.Visibility.UNKNOWN) if o.target_id==target_id else o for o in previous.occupancies)
        # No invented camera evidence. C0 1.0.0 rejects conservative downgrades
        # without newer physical proof; this proposal remains pending in that case.
        return self._next(previous,env,items,now,'unknown')

    def observe(self,previous,sample,ctx):
        guards.frame(sample,ctx,self.config);r.require_space_version(sample.envelope,previous)
        items={o.target_id:o for o in previous.occupancies}
        for item in sample.objects:
            old=items.get(item.target_id)
            c.require(old is None or item.observed_at-old.observed_at>0,'observed_at','occupancy update requires newer exposure')
            items[item.target_id]=item
        # Missing objects remain obstacles; no silent EMPTY interpretation.
        return self._next(previous,sample.envelope,tuple(items.values()),ctx.now,'observe')

    def clear(self,previous,env,target_id,clearance,now):
        old=next((o for o in previous.occupancies if o.target_id==target_id),None)
        c.require(old is not None and clearance is not None and clearance.target_id==target_id and
                  clearance.confirmed is c.Fact.CONFIRMED,'clearance','explicit moved/empty evidence required',c.ErrorCode.OCCUPANCY_UNRESOLVED)
        guards.physical(clearance.evidence,replace(env,target_id=target_id),now,self.config)
        c.require(all(e.captured_at-old.observed_at>0 for e in clearance.evidence),'clearance','new observation required')
        c.require(not any(target_id in o.support_ids for o in previous.occupancies),'supports','remove upper loads first')
        return self._next(previous,env,tuple(o for o in previous.occupancies if o.target_id!=target_id),now,'clear')


def accept_committed(transaction_id,receipt,expected,readback):
    c.require(receipt.status is m.CommitStatus.COMMITTED and receipt.transaction_id==transaction_id,
              'receipt','unknown/failed/unrelated commit cannot publish',c.ErrorCode.STORAGE_FAILURE)
    c.require(readback.revision==receipt.revision and readback.space.version==receipt.space_version and
              readback.space==expected,'readback','exact committed proposal required',c.ErrorCode.CONFLICT)
    return readback.space
