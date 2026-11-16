"""P09 acceptance/finite-budget rules for P08-owned layout proposals."""
import math
from dataclasses import replace
from treasure_auto.contracts import core as c,models as m,rules as r
from . import guards


class LayoutRecovery:
    def __init__(self,layout,budget,evidence_age_s):
        self.layout,self.budget,self.evidence_age_s=layout,budget,evidence_age_s
        self._events={};self._charges={}

    def charge(self,event_id,elapsed_s):
        c.require(type(event_id) is str and bool(event_id) and type(elapsed_s) in (int,float) and math.isfinite(elapsed_s) and elapsed_s>=0,
                  'budget','valid attempt identity and measured duration required')
        if event_id in self._charges:
            c.require(self._charges[event_id]==elapsed_s,'budget','same attempt changed cost',c.ErrorCode.ID_CONFLICT)
            return self.budget
        c.require(self.budget.attempts_limit is not None and self.budget.seconds_limit is not None and
                  self.budget.attempts_used<self.budget.attempts_limit and self.budget.seconds_used<self.budget.seconds_limit,
                  'budget','cumulative recovery budget exhausted/unconfigured',c.ErrorCode.LIMIT_EXCEEDED)
        self.budget=replace(self.budget,attempts_used=self.budget.attempts_used+1,seconds_used=self.budget.seconds_used+elapsed_s)
        self._charges[event_id]=elapsed_s
        return self.budget

    def can_retry(self,remaining_s,required_s=0.):
        b=self.budget
        return (b.attempts_limit is not None and b.seconds_limit is not None and b.attempts_used<b.attempts_limit and
                b.seconds_used+required_s<b.seconds_limit and remaining_s>required_s)

    def accept(self,proposed,now,release=None):
        old=self.layout
        if proposed==old:return old
        c.require(proposed.envelope.producer is c.Owner.P08 and proposed.area is old.area and proposed.kind is old.kind,
                  'layout','P08 owns unchanged region/layout strategy')
        c.require((proposed.layout_recovery_round,proposed.rebuild_generation)==(old.layout_recovery_round,old.rebuild_generation),
                  'recovery','progress cannot change failure counters')
        r.validate_envelope(proposed.envelope,guards.context(old.envelope,now))
        c.require(proposed.confirmed_count==old.confirmed_count+1 and len(proposed.bricks)==proposed.confirmed_count,
                  'count','only one physical placement per transition',c.ErrorCode.INVALID_LAYOUT)
        bs=sorted(proposed.bricks,key=lambda b:b.slot)
        c.require([b.slot for b in bs]==list(range(1,len(bs)+1)) and
                  [b.target_id for b in bs[:-1]]==[b.target_id for b in sorted(old.bricks,key=lambda b:b.slot)],'prefix','prior prefix changed')
        for b in bs:
            c.require(b.released is c.Fact.CONFIRMED and b.in_region is c.Fact.CONFIRMED and b.horizontal is c.Fact.CONFIRMED,
                      'layout','current released horizontal in-region prefix required')
            c.require(b.slot==1 or b.placed_recovery_round==old.layout_recovery_round,'round','old second/third cannot be inherited')
            guards.physical(b.evidence,replace(proposed.envelope,target_id=b.target_id,action_id=None),now,self.evidence_age_s)
        c.require(bs[0].near_center is c.Fact.CONFIRMED,'center','center-qualified first brick required')
        new=bs[-1]
        if new.slot>1 or release is not None:
            c.require(release is not None and release.envelope.producer is c.Owner.P08 and release.envelope.target_id==new.target_id and
                      release.envelope.action_id is not None and release.area is old.area and release.separated is c.Fact.CONFIRMED and release.correct_region is c.Fact.CONFIRMED,
                      'release','actual current-round re-placement required')
            r.validate_envelope(release.envelope,guards.context(proposed.envelope,now))
            c.require(release.released_at-old.envelope.emitted_at>0 and now-release.released_at>=0,'released_at','new physical placement must follow prior progress/failure')
            guards.physical(release.evidence,release.envelope,now,self.evidence_age_s)
            key=release.envelope.event_id;digest=c.dumps(release)
            c.require(key not in self._events,'release','same physical event cannot advance twice',c.ErrorCode.DUPLICATE)
        if proposed.kind is m.LayoutKind.V3 and new.slot>1:
            c.require(new.support_ratio_lower is not None and new.support_ratio_lower>.8 and new.support_ids==(bs[-2].target_id,),
                      'support','V3 current actual support lower bound must exceed 80%')
        if proposed.confirmed_count==3:r.require_layout_quality(proposed)
        self.layout=proposed
        if release is not None:self._events[release.envelope.event_id]=c.dumps(release)
        return proposed

    def failed(self,proposed,event,full_collapse,now):
        key=event.envelope.event_id;digest=(c.dumps(event),c.dumps(proposed),full_collapse)
        if key in self._events:
            c.require(self._events[key]==digest,'failure','event changed',c.ErrorCode.ID_CONFLICT)
            return self.layout
        old=self.layout
        c.require(type(full_collapse) is bool and event.envelope.producer is c.Owner.P08 and event.result.status is c.Outcome.FAILED and
                  event.result.physical is c.Fact.CONFIRMED,'failure','suspected/unknown is not confirmed physical failure')
        r.validate_envelope(event.envelope,guards.context(replace(old.envelope,layout_recovery_round=event.envelope.layout_recovery_round),now))
        guards.physical(event.result.evidence,event.envelope,now,self.evidence_age_s)
        c.require(proposed.area is old.area and proposed.kind is old.kind and proposed.envelope.producer is c.Owner.P08 and
                  proposed.confirmed_count==0 and not proposed.bricks and proposed.layout_recovery_round==old.layout_recovery_round+1 and
                  proposed.rebuild_generation==old.rebuild_generation+int(full_collapse),'failure','exact 0 reset/round/generation transition required')
        c.require(proposed.envelope.layout_recovery_round==proposed.layout_recovery_round and proposed.envelope.rebuild_generation==proposed.rebuild_generation,
                  'envelope','failure counters must match')
        self.charge(key,event.result.elapsed_s)
        self.layout=proposed;self._events[key]=digest
        return proposed

    def propose_operation(self,target_id,snapshot,temporary,context,placement_config,deadline,*,stop,stillness=None,affected_targets=(),scene_epoch=0):
        """Use the delivered P08 planner; this result is not permission to move."""
        from treasure_auto.placement.recovery import recovery_step
        guards.stopped(stop,context,context.emitted_at,self.evidence_age_s)
        if affected_targets:
            c.require(stillness is not None,'stillness','500ms current proof required')
            r.validate_stillness(stillness,context,context.emitted_at,affected_targets,scene_epoch)
        return recovery_step(target_id,snapshot.inventory,snapshot.holding,snapshot.space,temporary,
            guards.context(context,context.emitted_at),placement_config,self.budget,deadline)
