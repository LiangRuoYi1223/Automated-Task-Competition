"""Independent layout quality and sequential progress proposals."""
import math
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m, rules as r
from treasure_auto.geometry import planar
from treasure_auto.geometry.inputs import ContactProof
from treasure_auto.geometry.support import compute_support
from . import guards
from .targets import whole_in_region,near_center


def _segment_distance(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    t=max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/(dx*dx+dy*dy)))
    return math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)


def adjacent(a,b,config):
    pa,pb=guards.polygon(a.footprint_m),guards.polygon(b.footprint_m)
    if planar.area(planar.intersection(pa,pb))>1e-12:return False
    gap=min(_segment_distance(p,x,y) for poly,other in ((pa,pb),(pb,pa)) for p in poly for x,y in zip(other,other[1:]+other[:1]))
    return (gap+a.error_m+b.error_m<=config.adjacent_gap_m+1e-12 and gap-a.error_m-b.error_m>=-1e-12 and
            abs(a.z_max_m-b.z_max_m)+a.error_m+b.error_m<=config.height_tolerance_m)


def _current_prefix(previous,facts,space,region,config,now):
    prior=sorted(previous.bricks,key=lambda b:b.slot)
    current={f.target_id:f for f in facts.prior_facts};items={o.target_id:o for o in space.occupancies}
    c.require(len(current)==len(facts.prior_facts) and set(current)=={b.target_id for b in prior},
              'prior_facts','fresh complete prefix quality evidence required',c.ErrorCode.EVIDENCE_MISSING)
    refreshed=[]
    for i,b in enumerate(prior):
        c.require(b.slot==1 or b.placed_recovery_round==previous.layout_recovery_round,'prior_facts',
                  'old second/third cannot be inherited into current recovery',c.ErrorCode.STALE_RECOVERY)
        c.require(b.target_id in items,'prior_facts','previous object missing')
        partial=replace(previous,confirmed_count=i,bricks=tuple(refreshed))
        pf=replace(current[b.target_id],prior_facts=tuple(current[x.target_id] for x in prior[:i]))
        checked=assess_brick(partial,items[b.target_id],pf,space,region,config,now)
        c.require((checked.slot,checked.layer,checked.support_ids)==(b.slot,b.layer,b.support_ids),
                  'prior_facts','previous placement structure changed')
        # Re-observation is not re-placement. Preserve the actual placement round.
        refreshed.append(replace(checked,placed_recovery_round=b.placed_recovery_round))
    return tuple(refreshed)


def assess_brick(previous,item,facts,space,region,config,now):
    n=previous.confirmed_count+1
    c.require(n<= (1 if region.area is m.Area.A else 3),'slot','layout already complete')
    c.require(facts.target_id==item.target_id and previous.area is region.area and item.area is region.area,'target','wrong region or object')
    r.validate_envelope(facts.envelope,guards.context(replace(space.envelope,target_id=item.target_id,
        action_id=facts.envelope.action_id,request_id=facts.envelope.request_id,
        layout_recovery_round=previous.layout_recovery_round,rebuild_generation=previous.rebuild_generation),now))
    c.require(facts.envelope.producer in (c.Owner.P04,c.Owner.P06),'facts','P04/P06 physical source required')
    c.require(facts.stable is c.Fact.CONFIRMED and facts.horizontal is c.Fact.CONFIRMED and facts.released is c.Fact.CONFIRMED,
              'quality','stable, released and horizontal observations required',c.ErrorCode.INVALID_LAYOUT)
    guards.actual(item,facts.envelope,now,config);guards.physical(facts.evidence,facts.envelope,now,config)
    c.require(whole_in_region(item,region),'region','whole brick must fit')
    if n==1:c.require(near_center(item,region,config),'center','first brick outside measured center tolerance')
    items={o.target_id:o for o in space.occupancies}
    c.require(items.get(item.target_id)==item,'space','quality must use exact shared geometry')
    prior=_current_prefix(previous,facts,space,region,config,now)
    c.require(len(prior)==n-1,'layout','confirmed prefix inconsistent')
    layer=1;ratio=None;supports=()
    if previous.kind is m.LayoutKind.V3 and n>1:
        layer=n;supports=(prior[-1].target_id,)
    elif previous.kind is m.LayoutKind.V2 and n==3:
        layer=2;supports=tuple(b.target_id for b in prior)
    c.require(item.support_ids==supports,'supports','wrong actual support relationship',c.ErrorCode.INVALID_LAYOUT)
    if not supports:
        c.require(abs(item.z_min_m-config.ground_z_m)+item.error_m<=config.height_tolerance_m,'ground','not on verified ground plane')
        for b in prior:
            c.require(b.target_id in items,'space','previous base missing')
            c.require(planar.area(planar.intersection(guards.polygon(item.footprint_m),guards.polygon(items[b.target_id].footprint_m)))<=1e-12,
                      'footprint','ground bricks overlap')
    else:
        c.require(all(i in items for i in supports),'supports','missing actual lower surface')
        lowers=tuple(items[i] for i in supports)
        for lower in lowers:guards.actual(lower,facts.envelope,now,config)
        if previous.kind is m.LayoutKind.V3:
            c.require(n==2 or prior[-1].placed_recovery_round==previous.layout_recovery_round,
                      'recovery','third requires this round second placement',c.ErrorCode.STALE_RECOVERY)
        claimed=facts.support
        c.require(claimed is not None and claimed.upper_target_id==item.target_id and claimed.support_ids==supports,
                  'support','contact evidence for exact supports required',c.ErrorCode.EVIDENCE_MISSING)
        r.validate_envelope(claimed.envelope,guards.context(facts.envelope,now))
        guards.physical(claimed.evidence,facts.envelope,now,config)
        proof=ContactProof(item.target_id,supports,space.version,claimed.contact,claimed.horizontal,claimed.evidence)
        req=m.SupportGeometryRequest(replace(facts.envelope,emitted_at=now),item,lowers,space,claimed.evidence)
        result=compute_support(req,proof,config.height_tolerance_m)
        c.require(result.contact is c.Fact.CONFIRMED and result.horizontal is c.Fact.CONFIRMED,'contact','actual horizontal contact required')
        ratio=result.ratio_lower_bound
        if previous.kind is m.LayoutKind.V3:
            c.require(ratio is not None and ratio>.8,'support_ratio_lower','actual bottom support bound must be strictly >80%',c.ErrorCode.INVALID_LAYOUT)
        else:
            c.require(adjacent(lowers[0],lowers[1],config),'bases','two bases must be adjacent and height-compatible',c.ErrorCode.INVALID_LAYOUT)
            upper_poly=planar.erode(guards.polygon(item.footprint_m),item.error_m)
            c.require(all(planar.area(planar.intersection(upper_poly,planar.erode(guards.polygon(o.footprint_m),o.error_m)))>1e-12 for o in lowers),
                      'supports','top must have conservative positive support from both bases',c.ErrorCode.INVALID_LAYOUT)
    if previous.kind is m.LayoutKind.V2 and n==2:
        c.require(adjacent(items[prior[0].target_id],item,config),'bases','second base must sit adjacent',c.ErrorCode.INVALID_LAYOUT)
    return m.LayoutBrick(item.target_id,n,layer,supports,ratio,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,
        c.Fact.CONFIRMED if n==1 else c.Fact.UNKNOWN,previous.layout_recovery_round,facts.evidence)


class LayoutTracker:
    """Tracks proposals, not published state. Reconstruct from committed layout.

    Old-round envelopes are rejected after restart, so prior failure/release
    events cannot advance a new round. Current-round replay IDs can be supplied
    from the P10B journal using accepted_events; no private database is opened.
    """
    def __init__(self,layout,config,accepted_events=()):
        self.layout=layout;self.config=config;self.events=dict(accepted_events);self.requires_stop=False

    def _seen(self,event,extra=''):
        key=(event.envelope.run_id,event.envelope.producer,event.envelope.event_id)
        digest=c.dumps(event)+extra
        if key in self.events:
            c.require(self.events[key]==digest,'event_id','same event identity, different payload',c.ErrorCode.ID_CONFLICT)
            return True,key,digest
        return False,key,digest

    def failure(self,event,confirmed,full_collapse,ctx):
        seen,key,digest=self._seen(event,str((confirmed,full_collapse)))
        if seen:return self.layout
        r.validate_envelope(event.envelope,ctx)
        c.require(event.envelope.layout_recovery_round==self.layout.layout_recovery_round and
                  event.envelope.rebuild_generation==self.layout.rebuild_generation,'recovery','old failure cannot reset current round',c.ErrorCode.STALE_RECOVERY)
        c.require(type(confirmed) is c.Fact and type(full_collapse) is c.Fact,'failure','tri-state facts required')
        self.requires_stop=True
        if confirmed is c.Fact.CONFIRMED:
            c.require(event.envelope.producer in (c.Owner.P04,c.Owner.P06) and event.result.physical is c.Fact.CONFIRMED,
                      'failure','physical geometry/motion confirmation required')
            guards.physical(event.result.evidence,event.envelope,ctx.now,self.config)
            round_=self.layout.layout_recovery_round+1;gen=self.layout.rebuild_generation+(full_collapse is c.Fact.CONFIRMED)
            e=replace(guards.next_envelope(event.envelope,ctx.now,'failed'),layout_recovery_round=round_,rebuild_generation=gen)
            self.layout=replace(self.layout,envelope=e,layout_recovery_round=round_,rebuild_generation=gen,
                                confirmed_count=0,bricks=(),quality=c.Fact.REJECTED,bases_adjacent=c.Fact.UNKNOWN)
        else:self.layout=replace(self.layout,quality=c.Fact.UNKNOWN)
        self.events[key]=digest
        return self.layout

    def keep_first(self,target_id,space,facts,region,holding,inventory,ctx):
        c.require(self.layout.confirmed_count==0,'count','only first brick may be retained')
        c.require(holding.state is c.Holding.EMPTY,'holding','resolve held/unknown object before recovery')
        items=[i for i in inventory if target_id in i.target_ids]
        c.require(len(items)==1 and items[0].identity_confirmed is c.Fact.CONFIRMED and
                  items[0].location is m.InventoryLocation.REGION and items[0].area is region.area,'inventory','known released regional inventory required')
        c.require(not any(target_id in o.support_ids for o in space.occupancies),'support','remove upper load before keeping/reusing base')
        item=next((o for o in space.occupancies if o.target_id==target_id),None)
        c.require(item is not None,'target','first brick missing')
        result=assess_brick(self.layout,item,facts,space,region,self.config,ctx.now)
        self.layout=replace(self.layout,envelope=guards.next_envelope(facts.envelope,ctx.now,'retain-first'),confirmed_count=1,bricks=(result,),quality=c.Fact.UNKNOWN)
        return self.layout

    def correct_top(self,event,target_id,space,facts,region,holding,ctx):
        """Propose top-only correction without erasing actual occupancy.

        This revokes progress, not a removal command. P07/P06 still own grasp
        eligibility. Missing evidence pauses; confirmed bad lower geometry uses
        the existing whole-layout recovery transition.
        """
        extra='top:'+target_id+':'+repr(facts)+':'+c.dumps(space)
        seen,key,digest=self._seen(event,extra)
        if seen:return self.layout
        r.validate_envelope(event.envelope,ctx)
        c.require(event.envelope.target_id==target_id,'target','anomaly must identify the corrected physical target')
        c.require(event.envelope.layout_recovery_round==self.layout.layout_recovery_round and
            event.envelope.rebuild_generation==self.layout.rebuild_generation,
            'recovery','old correction cannot revoke current progress',c.ErrorCode.STALE_RECOVERY)
        c.require(holding.state is c.Holding.EMPTY,'holding','resolve held/unknown object before correction')
        c.require(not any(target_id in o.support_ids for o in space.occupancies),
                  'support','loaded lower brick cannot be extracted',c.ErrorCode.INVALID_LAYOUT)
        prior=sorted(self.layout.bricks,key=lambda b:b.slot)
        c.require(bool(prior) and prior[-1].target_id==target_id and facts.target_id==target_id,
                  'target','only the current final top may be locally corrected')
        c.require(event.envelope.producer in (c.Owner.P04,c.Owner.P06) and event.result.physical is c.Fact.CONFIRMED,
                  'failure','physical current top anomaly required')
        guards.physical(event.result.evidence,event.envelope,ctx.now,self.config)
        self.requires_stop=True
        prefix=replace(self.layout,confirmed_count=len(prior)-1,bricks=tuple(prior[:-1]),quality=c.Fact.UNKNOWN)
        try:
            items={o.target_id:o for o in space.occupancies}
            # Validate freshness before interpreting even a negative fact.
            # A stale "tilted" observation cannot erase a current good prefix.
            for fact in (facts,)+facts.prior_facts:
                r.validate_envelope(fact.envelope,replace(ctx,target_id=fact.target_id))
                c.require(fact.envelope.producer in (c.Owner.P04,c.Owner.P06),
                          'prior_facts','P04/P06 physical source required',c.ErrorCode.EVIDENCE_MISSING)
                guards.physical(fact.evidence,fact.envelope,ctx.now,self.config)
                c.require(fact.target_id in items,'prior_facts','current occupancy missing',c.ErrorCode.EVIDENCE_MISSING)
                guards.actual(items[fact.target_id],fact.envelope,ctx.now,self.config)
            c.require(c.Fact.REJECTED in (facts.horizontal,facts.stable),
                      'prior_facts','actual top anomaly not yet confirmed',c.ErrorCode.EVIDENCE_MISSING)
            c.require(not any(c.Fact.UNKNOWN in (f.stable,f.horizontal,f.released) for f in facts.prior_facts),
                      'prior_facts','unknown lower quality requires reobservation',c.ErrorCode.EVIDENCE_MISSING)
            c.require(not any(f.support is not None and c.Fact.UNKNOWN in
                      (f.support.contact,f.support.horizontal) for f in facts.prior_facts),
                      'prior_facts','unknown lower contact/orientation requires reobservation',c.ErrorCode.EVIDENCE_MISSING)
            refreshed=_current_prefix(prefix,facts,space,region,self.config,ctx.now)
        except c.ContractError as error:
            if error.code is c.ErrorCode.INVALID_LAYOUT or error.field in (
                    'center','ground','supports','footprint','bases','contact','support_ratio_lower','region'):
                # Actual invalid lower support/pose: preserve the established
                # whole-layout reset semantics rather than inheriting it.
                result=self.failure(event,c.Fact.CONFIRMED,c.Fact.REJECTED,ctx)
                self.events[key]=digest
                return result
            self.layout=replace(self.layout,quality=c.Fact.UNKNOWN)
            self.events[key]=digest
            return self.layout
        self.layout=replace(prefix,envelope=guards.next_envelope(event.envelope,ctx.now,'top-correction'),bricks=refreshed)
        self.events[key]=digest
        return self.layout

    def record_placement(self,event,space,facts,region,ctx):
        seen,key,digest=self._seen(event)
        if seen:return self.layout
        r.validate_envelope(event.envelope,replace(ctx,target_id=event.envelope.target_id))
        c.require(event.envelope.producer is c.Owner.P08 and event.envelope.layout_recovery_round==self.layout.layout_recovery_round and
            event.envelope.rebuild_generation==self.layout.rebuild_generation,'recovery','this round actual release required',c.ErrorCode.STALE_RECOVERY)
        c.require(event.separated is c.Fact.CONFIRMED and event.correct_region is c.Fact.CONFIRMED and event.area is region.area and
                  event.mapping_version==ctx.versions.mapping,'release','correct current regional release required')
        guards.physical(event.evidence,event.envelope,ctx.now,self.config)
        c.require(ctx.now-event.released_at>=0 and event.released_at-self.layout.envelope.emitted_at>=0,
                  'released_at','old placement cannot advance recovered layout')
        if self.layout.confirmed_count>0 or self.layout.layout_recovery_round>0:
            c.require(event.released_at-self.layout.envelope.emitted_at>0,'released_at',
                      'next placement must occur strictly after last confirmed progress/failure')
        c.require(event.envelope.target_id not in {b.target_id for b in self.layout.bricks},'target','one physical brick cannot occupy two slots')
        item=next((o for o in space.occupancies if o.target_id==event.envelope.target_id),None)
        c.require(item is not None,'target','physical placement missing')
        b=assess_brick(self.layout,item,facts,space,region,self.config,ctx.now)
        refreshed=_current_prefix(self.layout,facts,space,region,self.config,ctx.now)
        n=self.layout.confirmed_count+1
        e=guards.next_envelope(facts.envelope,ctx.now,'layout-'+str(n))
        candidate=replace(self.layout,envelope=e,confirmed_count=n,bricks=refreshed+(b,),
            bases_adjacent=c.Fact.CONFIRMED if self.layout.kind is m.LayoutKind.V2 and n>=2 else self.layout.bases_adjacent,
            quality=c.Fact.CONFIRMED if n==(1 if region.area is m.Area.A else 3) else c.Fact.UNKNOWN)
        if n==3:r.require_layout_quality(candidate)
        self.layout=candidate
        self.events[key]=digest
        return self.layout
