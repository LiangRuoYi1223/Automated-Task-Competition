"""Bounded field handling policy. Decisions do not command hardware or edit C0 facts.

The adapter supplies freshly verified observations in one monotonic time domain.
Target IDs are physical identities, never frame-local detection IDs. Attempt
budgets belong to that identity for this run and survive local layout repair.
"""
import copy
import json
import math
from dataclasses import asdict, dataclass
from typing import Optional
from treasure_auto.contracts import core as c


@dataclass(frozen=True)
class HandlingObservation:
    target_id: str
    event_id: str
    observed_at_s: float
    phase: str
    holding: c.Holding
    visible: c.Fact = c.Fact.UNKNOWN
    independent_lift: c.Fact = c.Fact.UNKNOWN
    following: c.Fact = c.Fact.UNKNOWN
    neighbors_stable: c.Fact = c.Fact.UNKNOWN
    offset_verified: c.Fact = c.Fact.UNKNOWN
    measured_offset: Optional[c.ArmPoint] = None
    action_id: Optional[str] = None
    stopped: c.Fact = c.Fact.UNKNOWN


@dataclass(frozen=True)
class HandlingDecision:
    action: str
    target_id: str
    probe_attempts: int
    recenter_attempts: int
    bias: bool
    skip_this_run: bool
    measured_offset: Optional[c.ArmPoint]
    reason: str
    independent_confirmed: bool = False


class HandlingPolicy:
    def __init__(self):
        self.targets = {}
        self.events = {}

    @staticmethod
    def _new():
        return dict(probe_attempts=0,recenter_attempts=0,bias=True,skip_this_run=False,
                    independent_confirmed=False,measured_offset=None,pending=None,
                    temporary_empty=False,last_time=None,used_actions=[],restart_stop=False,
                    last_action='await_observation',last_reason='no observation')

    def _decision(self,ident,state,action,reason):
        offset=state['measured_offset']
        return HandlingDecision(action,ident,state['probe_attempts'],state['recenter_attempts'],
            state['bias'],state['skip_this_run'],c.ArmPoint(**offset) if offset else None,
            reason,state['independent_confirmed'])

    def observe(self,observation):
        o=observation
        c.require(type(o) is HandlingObservation,'observation','explicit handling observation required')
        c.require(bool(o.target_id) and bool(o.event_id),'identity','stable physical target and event IDs required')
        c.require(type(o.observed_at_s) in (float,int) and math.isfinite(o.observed_at_s) and o.observed_at_s>=0,
                  'observed_at_s','finite monotonic observation time required')
        c.require(type(o.holding) is c.Holding and all(type(getattr(o,k)) is c.Fact for k in
            ('visible','independent_lift','following','neighbors_stable','offset_verified','stopped')),
            'facts','explicit tri-state observations required')
        c.require(o.phase in ('probe_start','probe_result','temporary_release','recenter_start','recenter_result','resolve_stop'),
                  'phase','unknown handling phase')
        if o.measured_offset is not None:
            p=o.measured_offset
            c.require(type(p) is c.ArmPoint and p.unit=='mm' and p.frame_id in ('tool','cup') and
                all(math.isfinite(v) for v in (p.x,p.y,p.z,p.error_mm)) and p.error_mm>=0,
                'measured_offset','bounded actual tool-frame offset required')
        signature=json.dumps(asdict(o),sort_keys=True,separators=(',',':'))
        state=self.targets.get(o.target_id,self._new())
        if o.event_id in self.events:
            c.require(self.events[o.event_id]==signature,'event_id','same event changed payload',c.ErrorCode.ID_CONFLICT)
            return self._decision(o.target_id,state,'stop_reobserve' if state['restart_stop'] else 'await_observation',
                                  'duplicate observation never reissues an action')
        c.require(state['last_time'] is None or o.observed_at_s>state['last_time'],
                  'observed_at_s','stale or duplicate exposure',c.ErrorCode.EXPIRED)
        # Work on a copy: rejecting a conflicting action cannot partially consume a budget.
        state=copy.deepcopy(state)
        action,reason=self._advance(o,state)
        state['last_time']=o.observed_at_s
        state['last_action']=action;state['last_reason']=reason
        self.targets[o.target_id]=state
        self.events[o.event_id]=signature
        return self._decision(o.target_id,state,action,reason)

    def _advance(self,o,state):
        if o.neighbors_stable is c.Fact.REJECTED:
            state['skip_this_run']=True
            state['restart_stop']=True
            return 'stop_reobserve','neighbor/board motion: stop lift before resolving actual holding'
        if o.phase=='resolve_stop':
            if o.stopped is not c.Fact.CONFIRMED or o.holding not in (c.Holding.HELD,c.Holding.EMPTY) or o.visible is not c.Fact.CONFIRMED:
                return 'stop_reobserve','actual stop, known holding and fresh visibility required'
            state['restart_stop']=False
            if state['skip_this_run'] and o.holding is c.Holding.EMPTY:
                state['pending']=None
                return 'skip_target','actual stop and EMPTY resolved; skip target while retaining occupancy and identity'
            return 'await_observation','stopped; unresolved action still needs physical result'
        if state['restart_stop']:
            return 'stop_reobserve','restart/interruption requires actual stopped acknowledgement'
        if state['skip_this_run']:
            return ('stop_reobserve' if state['pending'] else 'skip_target'),'target skipped; identity and occupancy are retained'
        if o.holding not in (c.Holding.EMPTY,c.Holding.HELD) or o.visible is not c.Fact.CONFIRMED or o.neighbors_stable is not c.Fact.CONFIRMED:
            return 'stop_reobserve','holding, visibility or neighboring motion unresolved'
        if o.phase=='temporary_release':
            if o.holding is not c.Holding.EMPTY or state['pending']:
                return 'stop_reobserve','committed real release and EMPTY required before another pick'
            state['temporary_empty']=True
            return 'await_observation','temporary release confirmed; require a new independent top observation'
        if o.phase.endswith('_start'):
            kind=o.phase.split('_')[0]
            if state['pending'] or o.holding is not c.Holding.EMPTY:
                return 'stop_reobserve','unresolved action/held object blocks another pick'
            if kind=='recenter' and not state['temporary_empty']:
                return 'stop_reobserve','recenter requires actual temporary release followed by EMPTY'
            c.require(bool(o.action_id),'action_id','stable action identity required')
            c.require(o.action_id not in state['used_actions'],'action_id','action identity already spent',c.ErrorCode.ID_CONFLICT)
            counter=kind+'_attempts'
            if state[counter]>=3:
                return ('skip_target' if kind=='probe' else 'await_observation'),'attempt budget exhausted; no extra regrasp'
            state[counter]+=1
            state['used_actions'].append(o.action_id)
            state['pending']={'kind':kind,'action_id':o.action_id}
            state['temporary_empty']=False
            return kind,'bounded attempt accepted; physical result still required'
        kind=o.phase.split('_')[0]
        c.require(state['pending']=={'kind':kind,'action_id':o.action_id},'action_id',
                  'result must match unresolved physical attempt',c.ErrorCode.STALE_REQUEST)
        if o.holding is c.Holding.EMPTY and o.independent_lift is c.Fact.REJECTED:
            state['pending']=None
            if kind=='recenter':
                # This attempt never lifted the released brick. Keep the
                # established temporary-ground prerequisite for a fresh top
                # observation and another bounded attempt; no new release is
                # claimed and the consumed attempt is never refunded.
                state['temporary_empty']=True
            if kind=='probe' and state['probe_attempts']>=3:
                state['skip_this_run']=True
                return 'skip_target','three unsuccessful identity probes; preserve identity and occupancy'
            return 'await_observation','empty after unsuccessful attempt; require fresh observation'
        if not (o.holding is c.Holding.HELD and o.independent_lift is c.Fact.CONFIRMED and
                o.following is c.Fact.CONFIRMED and o.measured_offset is not None):
            return 'stop_reobserve','independent lifted following brick and bounded actual offset not established'
        state['pending']=None
        state['independent_confirmed']=True
        state['measured_offset']=asdict(o.measured_offset)
        if o.offset_verified is c.Fact.CONFIRMED:
            state['bias']=False
            return 'transport','actual relative position and orientation verified'
        state['bias']=True
        if state['recenter_attempts']>=3:
            return 'approximate_place','use current suction point as nominal center; retain measured offset and physical quality checks'
        return 'temporary_place','independent brick confirmed; actual offset still requires recentering'

    def checkpoint(self):
        return copy.deepcopy(dict(version=1,targets=self.targets,events=self.events))

    def decision_for(self,target_id):
        state=self.targets.get(target_id)
        if state is None:return None
        return self._decision(target_id,state,'stop_reobserve' if state['restart_stop'] else state['last_action'],
            'restart requires actual stop and observation' if state['restart_stop'] else state['last_reason'])

    def pending_targets(self):
        """Targets whose unanswered action/stop blocks ordinary dispatch."""
        return tuple(sorted(ident for ident,state in self.targets.items() if state['pending'] or state['restart_stop']))

    @classmethod
    def restore(cls,data):
        c.require(type(data) is dict and data.get('version')==1,'checkpoint','unsupported handling state')
        policy=cls()
        policy.targets=copy.deepcopy(data['targets']);policy.events=copy.deepcopy(data['events'])
        c.require(type(policy.targets) is dict and type(policy.events) is dict,'checkpoint','invalid handling dictionaries')
        for state in policy.targets.values():
            c.require(set(state)==set(cls._new()),'checkpoint','incomplete handling state')
            c.require(all(type(state[k]) is int and 0<=state[k]<=3 for k in ('probe_attempts','recenter_attempts')),
                      'checkpoint','invalid attempt budget')
            c.require(type(state['used_actions']) is list and len(set(state['used_actions']))==len(state['used_actions']) and
                      len(state['used_actions'])==state['probe_attempts']+state['recenter_attempts'],
                      'checkpoint','attempt journal and budget disagree')
            if state['pending'] is not None:
                c.require(type(state['pending']) is dict and set(state['pending'])=={'kind','action_id'} and
                    state['pending']['kind'] in ('probe','recenter') and state['pending']['action_id'] in state['used_actions'],
                    'checkpoint','unresolved attempt is not journaled')
            if state['pending'] is not None:state['restart_stop']=True
        return policy
