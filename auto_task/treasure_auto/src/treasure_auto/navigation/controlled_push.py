"""SIMULATED-only bounded contact policy; emits no device commands or permits.

Sweeps are conservative whole-body/whole-group/exit envelopes supplied by the
planner in the field frame, including selected contact geometry. Other obstacles
remain forbidden. No physical thresholds are defaulted: missing measurements
deny start. Limits here are synthetic policy inputs, never hardware validation.
"""
import math
from dataclasses import asdict, dataclass, field
from typing import Optional
from treasure_auto.contracts import core as c, models as m
from treasure_auto.mapping.geometry import validate_polygon, collides, covered
from treasure_auto.geometry.planar import convex


def _number(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def _point(value):
    return (type(value) in (tuple, list) and len(value) == 2 and
            all(type(v) in (int, float) and math.isfinite(v) for v in value))


@dataclass(frozen=True)
class PushLimits:
    max_step_distance_m: Optional[float] = None
    max_step_duration_s: Optional[float] = None
    max_speed_m_s: Optional[float] = None
    max_observation_age_s: Optional[float] = None
    stall_window_s: Optional[float] = None
    min_fixed_progress_m: Optional[float] = None
    stop_linear_m_s: Optional[float] = None
    stop_angular_rad_s: Optional[float] = None
    stop_window_s: Optional[float] = None
    stability_window_s: Optional[float] = None

    def valid(self):
        return all(_number(v, True) for v in asdict(self).values())


@dataclass(frozen=True)
class PushPlan:
    group_id: str
    action_id: str
    frame_id: str
    affected_ids: tuple
    target_footprint: tuple
    body_sweep: tuple
    brick_sweep: tuple
    exit_sweep: tuple
    error_m: float
    direction_xy: tuple
    distance_m: float
    duration_s: float
    speed_m_s: float


@dataclass(frozen=True)
class PushObservation:
    sample_id: str
    captured_at: float
    frame_id: str
    group_id: str
    affected_ids: tuple
    fixed_position_xy: tuple
    wheel_distance_m: float
    linear_m_s: float
    angular_rad_s: float
    body_footprint: tuple
    brick_footprint: tuple
    localization_valid: bool
    observation_valid: bool
    space_confirmed: bool
    affected_visible: bool
    affected_stable: bool
    graspable_count: int
    old_action_stopped: bool
    queue_empty: bool
    stable_since: Optional[float] = None
    other_obstacles: tuple = ()
    abnormal_motion: bool = False
    origin: c.Origin = c.Origin.SIMULATED


@dataclass(frozen=True)
class PushDecision:
    action: str
    phase: str
    group_id: Optional[str]
    action_id: Optional[str]
    attempts: int
    elapsed_s: float
    reason: str


@dataclass
class _Group:
    group_id: str
    affected_ids: tuple
    first_started_at: float
    attempts: int = 0
    elapsed_s: float = 0.
    phase: str = 'ready'
    terminal_reason: str = ''
    plan: Optional[PushPlan] = None
    step_started_at: float = 0.
    start_position: tuple = ()
    last_position: tuple = ()
    fixed_distance_m: float = 0.
    progress_position: tuple = ()
    progress_at: float = 0.
    last_sample_at: float = 0.
    stop_requested_at: Optional[float] = None
    stop_samples_since: Optional[float] = None
    stop_confirmed_at: Optional[float] = None
    seen: dict = field(default_factory=dict)
    action_ids: list = field(default_factory=list)


class ControlledPushService:
    """Independent group budgets persist through stop/observe phases and restart.

    Caller must checkpoint before publishing any accepted action intent. Repeated
    starts never emit ``push_step`` again; restored unresolved intent is stopped
    and retired after fresh physical stop and stability observations.
    ``grasp`` is emitted only on the fresh completion transition. Completion
    retained in a checkpoint is historical, never a renewable grasp permit.
    """
    MAX_ATTEMPTS = 3
    MAX_TOTAL_S = 30.

    def __init__(self, regions, limits, origin=c.Origin.SIMULATED):
        self.regions = regions
        self.limits = limits
        self.origin = origin
        self.groups = {}
        self.active_group = None
        self._members = {}
        self._last_now = None

    def _group(self):
        return self.groups.get(self.active_group)

    def _decision(self, action, reason, group=None):
        group = group or self._group()
        return PushDecision(action, group.phase if group else 'idle',
                            group.group_id if group else None,
                            group.plan.action_id if group and group.plan else None,
                            group.attempts if group else 0, group.elapsed_s if group else 0., reason)

    def _clock(self, now):
        if not _number(now) or (self._last_now is not None and now < self._last_now):
            return False
        self._last_now = now
        for group in self.groups.values():
            if group.phase not in ('ended', 'complete'):
                group.elapsed_s = max(group.elapsed_s, now - group.first_started_at)
        return True

    def _stop(self, group, now, reason, terminal=False):
        if terminal:
            group.terminal_reason = group.terminal_reason or reason
            # A missing/invalid sample breaks continuity even when a prior stop
            # request is already outstanding. Start the evidence window again
            # after this fault, rather than joining samples across the gap.
            group.stop_requested_at = now
            group.stop_samples_since = None
            group.stop_confirmed_at = None
        if group.phase not in ('stopping', 'ended', 'complete'):
            group.phase = 'stopping'
            group.stop_requested_at = now
            group.stop_samples_since = None
            group.stop_confirmed_at = None
        return self._decision('stop', reason, group)

    def _geometry_safe(self, polygons, error, obstacles):
        for polygon in polygons:
            if self.regions.push_prohibited(polygon, error, self.regions.frame_id):
                return False
            for obstacle in obstacles:
                validate_polygon(obstacle)
                if collides(polygon, obstacle, error + self.regions.error_m):
                    return False
        return True

    def _observation_error(self, observation, now, plan):
        if type(observation) is not PushObservation:
            return 'missing observation'
        if self.origin is not c.Origin.SIMULATED or observation.origin is not c.Origin.SIMULATED:
            return 'SIMULATED policy only'
        if (type(observation.sample_id) is not str or not observation.sample_id or
                not _number(observation.captured_at) or not 0 <= now-observation.captured_at <= self.limits.max_observation_age_s):
            return 'stale or invalid observation'
        if (observation.frame_id != self.regions.frame_id or observation.frame_id != plan.frame_id or
                observation.group_id != plan.group_id or observation.affected_ids != plan.affected_ids):
            return 'frame or affected group mismatch'
        if not all(v is True for v in (observation.localization_valid, observation.observation_valid,
                                       observation.space_confirmed, observation.affected_visible)):
            return 'observation or localization unavailable'
        if (not _point(observation.fixed_position_xy) or not _number(observation.wheel_distance_m) or
                not all(type(v) in (float,int) and math.isfinite(v) for v in (observation.linear_m_s, observation.angular_rad_s)) or
                type(observation.graspable_count) is not int or not 0 <= observation.graspable_count <= len(plan.affected_ids)):
            return 'missing finite measurements'
        if observation.abnormal_motion is not False:
            return 'abnormal affected-object motion'
        if not self._geometry_safe((observation.body_footprint, observation.brick_footprint), plan.error_m, observation.other_obstacles):
            return 'observed footprint crosses forbidden region or obstacle'
        return ''

    def _stopped(self, observation):
        return (observation.old_action_stopped is True and observation.queue_empty is True and
                abs(observation.linear_m_s) <= self.limits.stop_linear_m_s and
                abs(observation.angular_rad_s) <= self.limits.stop_angular_rad_s)

    def _stable(self, observation, since=None):
        return (observation.affected_stable is True and _number(observation.stable_since) and
                observation.captured_at - observation.stable_since >= self.limits.stability_window_s and
                (since is None or observation.stable_since >= since))

    def start(self, plan, observation, now):
        group = self._group()
        if not self._clock(now):
            return self._stop(group, self._last_now, 'clock rollback', True) if group and group.phase in ('pushing','stopping','reobserving') else self._decision('denied','invalid clock')
        if group and group.phase in ('pushing','stopping','reobserving'):
            if group.phase == 'pushing' and type(plan) is PushPlan and plan == group.plan and repr(observation) == group.seen.get(getattr(observation,'sample_id',None)):
                return self.tick(now)
            return self._stop(group, now, 'unfinished action requires stop and fresh reobservation')
        if self.origin is not c.Origin.SIMULATED or not self.limits.valid():
            return self._decision('denied', 'SIMULATED origin and explicit finite limits required')
        try:
            if (type(plan) is not PushPlan or not plan.group_id or not plan.action_id or
                    type(plan.affected_ids) is not tuple or not plan.affected_ids or
                    any(type(v) is not str or not v for v in plan.affected_ids) or
                    len(set(plan.affected_ids)) != len(plan.affected_ids)):
                return self._decision('denied','explicit physical group and action identity required')
            aliases = {self._members[v] for v in plan.affected_ids if v in self._members}
            if len(aliases) > 1:
                return self._decision('denied','merged prior groups require explicit recovery')
            canonical = next(iter(aliases)) if aliases else plan.group_id
            group = self.groups.get(canonical)
            if any(plan.action_id in previous.action_ids for key,previous in self.groups.items() if key != canonical):
                return self._decision('denied','action identity already consumed by another group',group)
            if group:
                self.active_group = canonical
                if group.phase in ('ended','complete') or group.terminal_reason or group.attempts >= self.MAX_ATTEMPTS or group.elapsed_s >= self.MAX_TOTAL_S:
                    group.phase = 'ended' if group.phase != 'complete' else 'complete'
                    return self._decision('ended','group budget or recovery already ended',group)
                if plan.action_id in group.action_ids:
                    return self._decision('denied','action identity already consumed',group)
            if (not _number(plan.error_m) or plan.frame_id != self.regions.frame_id or not _point(plan.direction_xy) or
                    not math.isclose(math.hypot(*plan.direction_xy),1.,abs_tol=1e-8) or
                    not all(_number(v,True) for v in (plan.distance_m,plan.duration_s,plan.speed_m_s)) or
                    plan.distance_m > self.limits.max_step_distance_m or plan.duration_s > self.limits.max_step_duration_s or
                    plan.speed_m_s > self.limits.max_speed_m_s):
                return self._decision('denied','step geometry or motion limits invalid',group)
            problem = self._observation_error(observation,now,plan)
            if problem or not self._stopped(observation) or not self._stable(observation):
                return self._decision('denied',problem or 'fresh stationary base and stable group required',group)
            if group and (observation.sample_id in group.seen or observation.captured_at <= group.last_sample_at):
                return self._decision('denied','new observation required before next attempt',group)
            if not self._geometry_safe((plan.target_footprint,plan.body_sweep,plan.brick_sweep,plan.exit_sweep),plan.error_m,observation.other_obstacles):
                return self._decision('denied','planned sweeps intersect forbidden space',group)
            # A convex envelope containing both endpoints contains every translated
            # footprint between them. Merely checking endpoint centres is unsafe.
            for sweep in (plan.body_sweep,plan.brick_sweep,plan.exit_sweep):
                convex(tuple((p.x,p.y) for p in sweep))
            def translated(polygon):
                return tuple(m.Point2(p.x+plan.direction_xy[0]*plan.distance_m,
                                      p.y+plan.direction_xy[1]*plan.distance_m) for p in polygon)
            for shape,sweep in ((observation.body_footprint,plan.body_sweep),
                                (observation.body_footprint,plan.exit_sweep),
                                (observation.brick_footprint,plan.brick_sweep),
                                (plan.target_footprint,plan.brick_sweep)):
                if not covered(shape,sweep) or not covered(translated(shape),sweep):
                    return self._decision('denied','sweeps do not include complete translated geometry and retreat',group)
            if group is None:
                group = _Group(plan.group_id,plan.affected_ids,now)
                self.groups[canonical] = group
            self.active_group = canonical
            for member in plan.affected_ids:
                self._members[member] = canonical
            group.affected_ids = tuple(sorted(set(group.affected_ids) | set(plan.affected_ids)))
            group.attempts += 1
            group.action_ids.append(plan.action_id)
            group.phase = 'pushing'
            group.plan = plan
            group.step_started_at = now
            group.start_position = tuple(observation.fixed_position_xy)
            group.last_position = group.start_position
            group.fixed_distance_m = 0.
            group.progress_position = group.start_position
            group.progress_at = observation.captured_at
            group.last_sample_at = observation.captured_at
            group.seen[observation.sample_id] = repr(observation)
            group.stop_requested_at = group.stop_samples_since = group.stop_confirmed_at = None
            return self._decision('push_step','bounded simulated step accepted',group)
        except (c.ContractError, TypeError, ValueError, AttributeError, OverflowError):
            return self._decision('denied','invalid geometry or observation',group)

    def tick(self, now):
        group = self._group()
        if not self._clock(now):
            return self._stop(group,self._last_now,'clock rollback',True) if group else self._decision('denied','invalid clock')
        if not group:
            return self._decision('wait','no active group')
        if group.phase in ('ended','complete'):
            return self._decision('ended',group.terminal_reason or 'historical group outcome; no new action authorized',group)
        if group.elapsed_s >= self.MAX_TOTAL_S:
            group.terminal_reason = 'total time budget exhausted'
            if group.phase == 'ready':
                group.phase = 'ended'
                return self._decision('ended',group.terminal_reason,group)
            if group.phase == 'pushing':
                return self._stop(group,now,group.terminal_reason,True)
        if group.phase == 'pushing':
            if now-group.step_started_at >= group.plan.duration_s:
                return self._stop(group,now,'step duration limit')
            if now-group.last_sample_at > self.limits.max_observation_age_s:
                return self._stop(group,now,'observation timeout',True)
            if now-group.progress_at >= self.limits.stall_window_s:
                return self._stop(group,now,'fixed-environment stall',True)
            return self._decision('wait','step already issued; no repeated command',group)
        return self._decision({'stopping':'stop','reobserving':'reobserve','ready':'ready'}[group.phase],group.terminal_reason or 'await physical evidence',group)

    def observe(self, observation, now):
        group = self._group()
        if not group:
            return self._decision('denied','no active group')
        if not self._clock(now):
            return self._stop(group,self._last_now,'clock rollback',True)
        if group.phase in ('ended','ready'):
            return self.tick(now)
        try:
            problem = self._observation_error(observation,now,group.plan)
            if problem:
                return self._stop(group,now,problem,True)
            if group.phase == 'complete':
                if not self._stopped(observation):
                    return self._stop(group,now,'motion or queue clearance unconfirmed after completion',True)
                if not self._stable(observation):
                    return self._decision('reobserve','completed group requires current stability review',group)
                return self._decision('ended','historical group outcome; reacquire through normal grasp guards',group)
            previous = group.seen.get(observation.sample_id)
            if previous is not None:
                if previous != repr(observation):
                    return self._stop(group,now,'same sample identity changed',True)
                result = self.tick(now)
                return self._decision('wait','duplicate observation',group) if result.action in ('reobserve','ready') else result
            if observation.captured_at <= group.last_sample_at:
                return self._stop(group,now,'non-increasing sample timestamp',True)
            sample_gap = observation.captured_at-group.last_sample_at
            group.seen[observation.sample_id] = repr(observation)
            group.last_sample_at = observation.captured_at
            if group.elapsed_s >= self.MAX_TOTAL_S:
                group.terminal_reason = 'total time budget exhausted'
            if group.phase == 'pushing':
                if observation.captured_at < group.step_started_at:
                    return self._stop(group,now,'observation predates step intent',True)
                if sample_gap > self.limits.max_observation_age_s:
                    return self._stop(group,now,'observation gap during motion',True)
                if abs(observation.linear_m_s) > group.plan.speed_m_s:
                    return self._stop(group,now,'observed speed exceeds step bound',True)
                if not covered(observation.body_footprint,group.plan.body_sweep) or not covered(observation.brick_footprint,group.plan.brick_sweep):
                    return self._stop(group,now,'observed geometry left planned sweep',True)
                dx = observation.fixed_position_xy[0]-group.start_position[0]
                dy = observation.fixed_position_xy[1]-group.start_position[1]
                group.fixed_distance_m += math.hypot(observation.fixed_position_xy[0]-group.last_position[0],
                                                    observation.fixed_position_xy[1]-group.last_position[1])
                group.last_position = tuple(observation.fixed_position_xy)
                if dx*group.plan.direction_xy[0]+dy*group.plan.direction_xy[1] < -self.limits.min_fixed_progress_m:
                    return self._stop(group,now,'fixed-environment motion opposes planned direction',True)
                if group.fixed_distance_m >= group.plan.distance_m:
                    return self._stop(group,now,'step distance limit')
                progress = ((observation.fixed_position_xy[0]-group.progress_position[0])*group.plan.direction_xy[0] +
                            (observation.fixed_position_xy[1]-group.progress_position[1])*group.plan.direction_xy[1])
                if progress >= self.limits.min_fixed_progress_m:
                    group.progress_at = observation.captured_at
                    group.progress_position = tuple(observation.fixed_position_xy)
                if observation.graspable_count >= 1 and self._stable(observation):
                    return self._stop(group,now,'graspable target found; verify actual stop')
                return self.tick(now)
            if group.phase == 'stopping':
                if sample_gap > self.limits.max_observation_age_s:
                    group.stop_samples_since = None
                if observation.captured_at < group.stop_requested_at or not self._stopped(observation):
                    group.stop_samples_since = None
                    return self._decision('stop','actual stop and old queue clearance unconfirmed',group)
                if group.stop_samples_since is None:
                    group.stop_samples_since = observation.captured_at
                if observation.captured_at-group.stop_samples_since < self.limits.stop_window_s:
                    return self._decision('stop','collecting actual stop window',group)
                group.stop_confirmed_at = observation.captured_at
                group.phase = 'reobserving'
                return self._decision('reobserve','actual stop confirmed; require a later stable frame',group)
            if not self._stopped(observation):
                group.phase = 'pushing'
                return self._stop(group,now,'motion after stop confirmation',True)
            if observation.captured_at <= group.stop_confirmed_at or not self._stable(observation,group.stop_requested_at):
                return self._decision('reobserve','all affected bricks must be visible and stable',group)
            if group.terminal_reason or group.attempts >= self.MAX_ATTEMPTS:
                group.phase = 'ended'
                return self._decision('ended',group.terminal_reason or 'attempt budget exhausted',group)
            if observation.graspable_count >= 1:
                group.phase = 'complete'
                return self._decision('grasp','at least one graspable brick; other affected bricks stable',group)
            group.phase = 'ready'
            return self._decision('ready','fresh stable observation permits replanning',group)
        except (c.ContractError,TypeError,ValueError,AttributeError,OverflowError):
            return self._stop(group,now,'invalid observation',True)

    def checkpoint(self, now):
        """JSON-safe state; timing is the caller's persistent monotonic domain."""
        self.tick(now)
        return {'version':1,'last_now':self._last_now,'active_group':self.active_group,
                'members':dict(self._members),'groups':{key:asdict(value) for key,value in self.groups.items()}}

    def restore(self, state, now):
        """Load once; no intent replay. Same-clock downtime consumes the budget."""
        if (self.origin is not c.Origin.SIMULATED or not self.limits.valid() or
                self.groups or type(state) is not dict or not _number(now) or
                state.get('version') != 1 or not _number(state.get('last_now'))):
            return self._decision('denied','invalid checkpoint or already initialized service')
        try:
            restored = {}
            for key, value in state['groups'].items():
                data = dict(value)
                if (not _number(data['attempts']) or type(data['attempts']) is not int or
                        not 0 < data['attempts'] <= self.MAX_ATTEMPTS or not _number(data['elapsed_s']) or
                        not _number(data['first_started_at']) or data['first_started_at'] > state['last_now']):
                    raise ValueError('invalid budget')
                plan_data = dict(data['plan'])
                for name in ('target_footprint','body_sweep','brick_sweep','exit_sweep'):
                    plan_data[name] = tuple(m.Point2(**p) for p in plan_data[name])
                for name in ('affected_ids','direction_xy'):
                    plan_data[name] = tuple(plan_data[name])
                data['plan'] = PushPlan(**plan_data)
                for name in ('affected_ids','start_position','last_position','progress_position'):
                    data[name] = tuple(data[name])
                group = _Group(**data)
                if group.phase not in ('pushing','stopping','reobserving','ready','complete','ended'):
                    raise ValueError('invalid phase')
                if group.phase not in ('complete','ended'):
                    group.elapsed_s = max(group.elapsed_s,now-group.first_started_at)
                    group.terminal_reason = 'restored unresolved action; require fresh recovery'
                    group.phase = 'stopping'
                    group.stop_requested_at = max(now,state['last_now'])
                    group.stop_samples_since = group.stop_confirmed_at = None
                restored[key] = group
            if state['active_group'] is not None and state['active_group'] not in restored:
                raise ValueError('missing active group')
            if any(value not in restored for value in state['members'].values()):
                raise ValueError('missing member group')
            self.groups = restored
            self._members = dict(state['members'])
            self.active_group = state['active_group']
            self._last_now = max(now,state['last_now'])
            return self.tick(now)
        except (KeyError,TypeError,ValueError,AttributeError,c.ContractError):
            return self._decision('denied','malformed checkpoint')
