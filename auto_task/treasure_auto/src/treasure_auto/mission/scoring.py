"""Conditional score graph, independent of arrival count and layout progress.

Rule source: automatic competition rules PDF pp.20-21 and v2 section 9.5.
Supplemental ScoreBrick is simulation input, not a new C0 wire record.
"""
import math
from dataclasses import dataclass,replace
from enum import Enum
from typing import Optional,Tuple
from treasure_auto.contracts import core as c,models as m,rules as r
from . import guards

POLICY_VERSION='P09-automatic-rules-p20-21-v1'


class ScoreZone(str,Enum):
    COLLECTION='collection'
    TRANSPORT='transport'
    OUTSIDE='outside'


@dataclass(frozen=True)
class ScoreBrick:
    envelope:c.Envelope
    target_id:str
    inventory_id:str
    category:m.Category
    area:Optional[m.Area]
    zone:Optional[ScoreZone]
    physical_layer:Optional[int]
    support_ids:Optional[Tuple[str,...]]
    horizontal:c.Fact
    released:c.Fact
    released_at:Optional[c.ClockStamp]
    height_top_m:Optional[float]
    whole_in_scoring:c.Fact
    evidence:Tuple[c.EvidenceRef,...]


@dataclass(frozen=True)
class ScoreScene:
    mapping:m.MappingSnapshot
    bricks:Tuple[ScoreBrick,...]
    exit:Optional[m.ExitEvidence]
    cake_priority:c.Fact
    deductions:Optional[float]


@dataclass(frozen=True)
class ScoreTerm:
    target_id:str
    points:Optional[float]
    effective_layer:Optional[int]
    zone:Optional[ScoreZone]
    reason:str


@dataclass(frozen=True)
class ScoreReport:
    estimate:m.ScoreEstimate
    terms:Tuple[ScoreTerm,...]


def estimate(scene,run,now):
    r.require_simulated(run)
    c.require(scene.mapping.envelope.run_id==run.envelope.run_id and
              scene.mapping.envelope.origin is run.envelope.origin,'mapping','same-run mapping required')
    ids=[b.target_id for b in scene.bricks];inventory=[b.inventory_id for b in scene.bricks]
    c.require(len(ids)==len(set(ids)) and len(inventory)==len(set(inventory)),'identity','one physical inventory per score object',c.ErrorCode.ID_CONFLICT)
    items={b.target_id:b for b in scene.bricks};cache={};proofs=[]

    def physical(b):
        c.require(b.envelope.target_id==b.target_id and b.envelope.producer in (c.Owner.P08,c.Owner.P06),'producer','P08/P06 current facts required')
        ctx=m.ConsumerContext(run.envelope.run_id,run.envelope.instance_id,run.envelope.control_epoch,now,
            b.envelope.versions,None,None,b.target_id,None,None,None,run.envelope.origin)
        r.validate_envelope(b.envelope,ctx)
        c.require(all(getattr(b.envelope.versions,k)==getattr(run.envelope.versions,k) for k in
                      ('config','map','calibration','model','layout')),'versions','score facts use another frozen version')
        c.require(b.envelope.versions.mapping==scene.mapping.mapping_version,'mapping','score mapping changed')
        c.require(bool(b.evidence),'evidence','current physical score evidence missing')
        for ev in b.evidence:
            guards.physical((ev,),b.envelope,now,ev.valid_until-ev.received_at)
        c.require(b.category in m.Category,'category','unknown category')

    def support(target,visiting):
        if target in cache:return cache[target]
        c.require(target in items and target not in visiting,'support','missing or cyclic support')
        b=items[target];physical(b)
        c.require(b.released is c.Fact.CONFIRMED and b.released_at is not None and
                  b.released_at-run.started_at>=0 and now-b.released_at>=0 and b.whole_in_scoring is c.Fact.CONFIRMED,
                  'support.release','support ancestor is not confirmed released in the scoring area')
        c.require(type(b.physical_layer) is int and 1<=b.physical_layer<=4 and b.support_ids is not None,
                  'layer','physical layer and complete support identities required')
        c.require(b.zone in (ScoreZone.COLLECTION,ScoreZone.TRANSPORT) and b.area in m.Area,'area','unknown scoring area')
        c.require(len(b.support_ids)==len(set(b.support_ids)),'support','duplicate support')
        if not b.support_ids:
            c.require(b.physical_layer==1,'support','non-ground brick has no support')
            area,zone,wrong=b.area,b.zone,set()
        else:
            lowers=[support(t,visiting|{target}) for t in b.support_ids]
            c.require(all(items[t].physical_layer==b.physical_layer-1 for t in b.support_ids),'support','non-adjacent physical layers')
            areas={v[0] for v in lowers}
            c.require(len(areas)==1,'support','cross-area attribution ambiguous')
            area=next(iter(areas));zone=ScoreZone.TRANSPORT if any(v[1] is ScoreZone.TRANSPORT for v in lowers) else ScoreZone.COLLECTION
            wrong=set().union(*(v[2] for v in lowers))
        own_wrong=scene.mapping.destination(b.category) is not area
        ancestry=set(wrong)
        if own_wrong:wrong.add(b.physical_layer)
        cache[target]=(area,zone,wrong,ancestry,own_wrong)
        return cache[target]

    terms=[]
    for b in scene.bricks:
        layer=None;zone=b.zone
        try:
            physical(b)
            c.require(b.height_top_m is not None and type(b.height_top_m) in (int,float) and math.isfinite(b.height_top_m) and b.height_top_m>=0,
                      'height','actual measured top height required')
            c.require(b.released_at is not None,'released_at','physical release time required')
            c.require(b.released_at-run.started_at>=0,'released_at','predates start')
            # A future release assertion is not evidence of an excluded physical brick.
            c.require(now-b.released_at>=0,'released_at','future release')
            if (b.released is c.Fact.REJECTED or b.whole_in_scoring is c.Fact.REJECTED or b.zone is ScoreZone.OUTSIDE or
                b.height_top_m>.4 or run.deadline-b.released_at<=0):
                terms.append(ScoreTerm(b.target_id,0,None,zone,'excluded by release/region/height/deadline'))
            else:
                c.require(b.released is c.Fact.CONFIRMED and b.whole_in_scoring is c.Fact.CONFIRMED and b.horizontal is not c.Fact.UNKNOWN,
                          'facts','release/region/horizontal contact unresolved')
                area,zone,wrong,ancestry,own_wrong=support(b.target_id,set())
                layer=1 if b.horizontal is c.Fact.REJECTED else b.physical_layer-len(ancestry)
                c.require(layer>=1,'layer','invalid effective layer')
                if b.category is m.Category.CAKE:
                    c.require(scene.cake_priority is not c.Fact.UNKNOWN,'cake_priority','cake priority unresolved')
                excluded=own_wrong or (b.category is m.Category.CAKE and scene.cake_priority is c.Fact.REJECTED)
                points=0 if excluded else ((layer if zone is ScoreZone.COLLECTION else 0)+2**(layer-1))
                terms.append(ScoreTerm(b.target_id,points,layer,zone,'wrong area/priority' if excluded else 'conditional rule score'))
            proofs.extend(b.evidence)
        except (c.ContractError,TypeError,ValueError) as error:
            terms.append(ScoreTerm(b.target_id,None,None,zone,str(error)))
    unresolved=[t.target_id for t in terms if t.points is None]
    known=[t.points for t in terms if t.points is not None]
    if scene.exit is not None:
        ex=scene.exit
        try:
            c.require(ex.area is m.Area.A and ex.entire_envelope_outside is c.Fact.CONFIRMED and ex.envelope.producer is c.Owner.P03,
                      'exit','entire envelope outside A required')
            ctx=m.ConsumerContext(run.envelope.run_id,run.envelope.instance_id,run.envelope.control_epoch,now,ex.envelope.versions,None,None,None,None,None,None,run.envelope.origin)
            r.validate_envelope(ex.envelope,ctx)
            c.require(run.deadline-ex.envelope.emitted_at>0 and ex.envelope.emitted_at-run.started_at>=0 and bool(ex.evidence),'exit','before-deadline physical proof required')
            for ev in ex.evidence:r.require_physical_evidence(ev,ex.envelope,now,run.envelope.origin)
            known.append(1);proofs.extend(ex.evidence)
        except c.ContractError:unresolved.append('run:exit')
    if scene.deductions is None:unresolved.append('match:deductions')
    else:c.require(type(scene.deductions) in (int,float) and math.isfinite(scene.deductions) and scene.deductions>=0,'deductions','nonnegative explicit deduction required')
    total=None if not known else max(0.,sum(known)-(scene.deductions or 0.))
    unique={p.evidence_id:p for p in proofs}
    return ScoreReport(m.ScoreEstimate(total,tuple(unresolved),tuple(unique.values()),POLICY_VERSION),tuple(terms))
