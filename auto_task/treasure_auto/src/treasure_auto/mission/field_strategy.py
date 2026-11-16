"""Field-policy orchestration used by MissionEngine, never a device permission.

P04/P06/P07 adapters supply observations; this module cannot manufacture sensor
truth. Controlled contact and handling remain owned by their domain services.
"""
import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from treasure_auto.contracts import core as c, models as m, rules as r
from . import guards


def consume_perception(frame, now):
    """Validate one exact P04 report/result pair and return current-frame counts."""
    from treasure_auto.perception.strategy import PerceptionPolicyReport
    report, result = frame.perception_policy, frame.perception_result
    if report is None and result is None:
        return None
    c.require(type(report) is PerceptionPolicyReport and type(result) is m.PerceptionResult,
              'perception_policy', 'paired typed P04 report/result required')
    c.require(report.envelope == result.envelope and result.envelope.producer is c.Owner.P04,
              'perception_policy', 'report must bind exact P04 result')
    r.validate_envelope(report.envelope, guards.context(frame.inputs.envelope, now))
    c.require(report.current is True and result.state in (m.ObservationState.DETECTED, m.ObservationState.EMPTY,
                                                         m.ObservationState.UNKNOWN_CLASS),
              'perception_policy', 'current successful frame required', c.ErrorCode.EXPIRED)
    c.require(result.targets == frame.inputs.candidates,
              'perception_policy', 'candidate list is not the paired P04 result', c.ErrorCode.CONFLICT)
    c.require(bool(report.observation_id) and bool(report.camera_frame) and len(report.image_sha256) == 64,
              'perception_policy', 'image identity required')
    counts = (report.pattern_count, report.independent_count, report.geometry_count)
    c.require(all(type(n) is int and n >= 0 for n in counts), 'counts', 'nonnegative current counts required')
    return counts


@dataclass(frozen=True)
class FieldInput:
    """Local adapter input; footprint is in the configured field map frame.

    The target ID is a physical tracking key, not a frame-local detector index.
    Domain observations supply their own immutable observation/action IDs.
    """
    kind: str
    target_id: str
    frame_id: str
    footprint: Tuple[m.Point2, ...] = ()
    error_m: float = 0.
    abnormal: bool = False
    normal_known_layout: bool = False
    reliable_grasp: bool = False
    push_plan: object = None
    push_observation: object = None
    handling_observation: object = None
    exploratory_entity_id: Optional[str] = None
    observation_id: Optional[str] = None


@dataclass(frozen=True)
class FieldDecision:
    action: str
    target_id: str
    reason: str
    domain_result: object = None
    requires_stop: bool = False
    # Explicitly prevents consumers treating a strategy choice as motor authority.
    authorizes_motion: bool = False


class FieldStrategyController:
    @classmethod
    def from_config(cls, path, regions):
        import json
        from pathlib import Path
        from dataclasses import fields
        from treasure_auto.navigation.controlled_push import PushLimits
        data=json.loads(Path(path).read_text(encoding='utf-8-sig'))
        c.require(type(data) is dict and set(data)=={'schema_version','origin','measurement','policy','push_limits'},
                  'field_config','exact field configuration required')
        c.require(type(data['schema_version']) is int and data['schema_version']==1 and
                  data['origin']=='simulated' and data['measurement'] in ('unmeasured','simulated'),
                  'field_config','only explicit unmeasured/simulated offline configuration is supported')
        expected={'probe_attempts':3,'recenter_attempts':3,'push_attempts':3,'push_total_s':30}
        c.require(data['policy']==expected and all(type(v) is int for v in data['policy'].values()),
                  'field_config.policy','approved 3/3/3 attempts and 30 second policy required')
        c.require(type(data['push_limits']) is dict and set(data['push_limits'])=={f.name for f in fields(PushLimits)},
                  'field_config.push_limits','all unmeasured physical settings must be explicit')
        if data['measurement']=='unmeasured':
            c.require(all(v is None for v in data['push_limits'].values()),
                      'field_config.measurement','unmeasured settings cannot masquerade as physical measurements')
        return cls(regions,PushLimits(**data['push_limits']))

    def __init__(self, regions, push_limits, *, origin=c.Origin.SIMULATED):
        from treasure_auto.navigation.controlled_push import ControlledPushService
        from treasure_auto.manipulation.strategy import HandlingPolicy
        c.require(origin is c.Origin.SIMULATED, 'origin', 'field adapters are not commissioned', c.ErrorCode.NOT_AUTHORIZED)
        self.regions = regions
        self.push = ControlledPushService(regions, push_limits, origin=origin)
        self.handling = HandlingPolicy()
        self.active_target = None
        self.push_active = False
        self.run_id = None
        self.deadline = None
        self.last_now = None
        self.restored = False
        # An accepted base contact invalidates all pre-contact arm geometry.
        # This is a local fence, never an invented public motion epoch.
        self.geometry_barrier = None

    def bind_run(self, run):
        c.require(run.envelope.origin is c.Origin.SIMULATED, 'origin', 'simulated field workflow only')
        if self.run_id is None:
            self.run_id, self.deadline = run.envelope.run_id, run.deadline.seconds
        c.require((self.run_id, self.deadline) == (run.envelope.run_id, run.deadline.seconds),
                  'run', 'field recovery cannot reset the original run/deadline')

    def _push_decision(self, result, target):
        action = result.action
        if action == 'push_step':
            self.push_active = True
        elif action in ('grasp', 'ready', 'ended', 'denied'):
            self.push_active = False
        return FieldDecision(action, target, result.reason, result, action == 'stop')

    def pending_handling_gate(self, request, now, max_age=2.):
        """Only the unresolved action's result/stop acknowledgement may proceed."""
        pending = self.handling.pending_targets()
        if not pending:
            return None
        target = pending[0]
        state = self.handling.checkpoint()['targets'][target]
        observation = request.handling_observation if request is not None and request.kind == 'handling' else None
        matches = (observation is not None and request.target_id == target and observation.target_id == target and
                   (observation.phase == 'resolve_stop' or
                    state['pending'] is not None and
                    observation.phase == state['pending']['kind'] + '_result' and
                    observation.action_id == state['pending']['action_id']))
        if matches:
            return None
        expired = state['restart_stop'] or now - state['last_time'] >= max_age
        return FieldDecision('stop_reobserve' if expired else 'await_observation', target,
                             'unresolved handling action; matching physical result or stop required',
                             requires_stop=expired)

    def advance(self, request, snapshot, now):
        c.require(type(request) is FieldInput, 'field_input', 'typed field observation required')
        c.require(type(now) in (int, float) and math.isfinite(now), 'clock', 'finite monotonic time required')
        c.require(self.last_now is None or now >= self.last_now, 'clock', 'field clock regressed')
        self.last_now = now
        self.bind_run(snapshot.run)
        if now >= self.deadline:
            return FieldDecision('stop', request.target_id, 'original_deadline', requires_stop=True)
        gate = self.pending_handling_gate(request, now)
        if gate is not None:
            return gate
        c.require(bool(request.target_id) and request.frame_id == self.regions.frame_id,
                  'target', 'physical tracking identity and matching map frame required')
        c.require(type(request.error_m) in (int, float) and math.isfinite(request.error_m) and request.error_m >= 0,
                  'error_m', 'finite nonnegative footprint error required')
        c.require(all(type(value) is bool for value in (request.abnormal, request.normal_known_layout, request.reliable_grasp)),
                  'facts', 'explicit booleans required')
        if self.push_active:
            c.require(request.target_id == self.active_target and request.kind in ('push_observation', 'tick'),
                      'field_action', 'unresolved push must stop/reobserve before changing target')
        if request.kind == 'select':
            c.require(snapshot.holding.state is c.Holding.EMPTY, 'holding', 'resolve held or unknown object before target selection')
            priority = self.regions.recovery_priority(
                request.footprint, request.error_m, normal_known_layout=request.normal_known_layout,
                reliable_grasp=request.reliable_grasp, abnormal=request.abnormal,
                push_safe=request.push_plan is not None, frame_id=request.frame_id)
            self.active_target = request.target_id
            if priority == 'push':
                c.require(request.push_observation is not None, 'push', 'fresh initial observation required')
                c.require(request.target_id in request.push_plan.affected_ids,
                          'push.target', 'selected target must belong to affected physical group')
                self.restored = False
                result = self.push.start(request.push_plan, request.push_observation, now)
                if result.action == 'push_step':
                    self.geometry_barrier = dict(base_motion_epoch=snapshot.space.envelope.base_motion_epoch,
                                                 accepted_at_s=now, settled_at_s=None)
                return self._push_decision(result, request.target_id)
            return FieldDecision(priority, request.target_id, 'region_recovery_priority')
        if request.kind in ('push_observation', 'tick'):
            c.require(self.active_target == request.target_id, 'push.target', 'feedback belongs to active physical target')
            result = (self.push.observe(request.push_observation, now) if request.kind == 'push_observation'
                      else self.push.tick(now))
            self.restored = False
            if self.geometry_barrier is not None:
                if result.action == 'stop':
                    self.geometry_barrier['settled_at_s'] = None
                elif result.action in ('grasp', 'ready', 'ended'):
                    group = self.push.groups.get(result.group_id)
                    if group is not None and group.stop_confirmed_at is not None:
                        self.geometry_barrier['settled_at_s'] = now
            return self._push_decision(result, request.target_id)
        if request.kind == 'handling':
            observation = request.handling_observation
            c.require(observation is not None and observation.target_id == request.target_id,
                      'handling', 'same physical target observation required')
            c.require(observation.holding is snapshot.holding.state,
                      'holding', 'handling observation must agree with current committed holding')
            if snapshot.holding.state is c.Holding.HELD:
                c.require(snapshot.holding.target_id == request.target_id,
                          'holding', 'another object is held')
            c.require(0 <= now - observation.observed_at_s <= 2.,
                      'handling.time', 'fresh normalized observation required')
            result = self.handling.observe(observation)
            return FieldDecision(result.action, request.target_id, result.reason, result,
                                 result.action == 'stop_reobserve')
        raise c.ContractError(c.ErrorCode.INVALID_FIELD, 'field_input.kind', 'unknown field operation')

    def checkpoint(self, now):
        return dict(schema=1, run_id=self.run_id, deadline=self.deadline,
                    active_target=self.active_target, push_active=self.push_active,
                    last_now=self.last_now, push=self.push.checkpoint(now), handling=self.handling.checkpoint(),
                    geometry_barrier=copy.deepcopy(self.geometry_barrier))

    def restore(self, data, now):
        c.require(type(data) is dict and data.get('schema') == 1,
                  'field_checkpoint', 'unsupported checkpoint')
        # Legacy handling-only state has no invalidated geometry to reconstruct.
        if 'geometry_barrier' not in data:
            c.require(type(data.get('push')) is dict and data['push'].get('groups') == {} and
                      data['push'].get('members') == {} and data['push'].get('active_group') is None,
                      'field_checkpoint', 'legacy push state lacks the persistent geometry fence')
            data = dict(data, geometry_barrier=None)
        c.require(set(data) == {'schema', 'run_id', 'deadline', 'active_target', 'push_active', 'last_now', 'push', 'handling', 'geometry_barrier'},
                  'field_checkpoint', 'missing or extra checkpoint fields')
        c.require(type(data['push_active']) is bool, 'field_checkpoint', 'invalid pending state')
        number = lambda value: type(value) in (int, float) and math.isfinite(value) and value >= 0
        c.require(number(now) and (data['last_now'] is None or number(data['last_now']) and now >= data['last_now']),
                  'field_checkpoint', 'invalid or regressed checkpoint clock')
        saved_push = data['push']
        c.require(type(saved_push) is dict and set(saved_push) == {'version','last_now','active_group','members','groups'} and
                  type(saved_push['version']) is int and saved_push['version'] == 1 and
                  number(saved_push['last_now']) and now >= saved_push['last_now'] and
                  type(saved_push['groups']) is dict and type(saved_push['members']) is dict,
                  'field_checkpoint.push', 'invalid saved push state')
        barrier = data['geometry_barrier']
        c.require(barrier is None or type(barrier) is dict and
                  set(barrier) == {'base_motion_epoch','accepted_at_s','settled_at_s'} and
                  type(barrier['base_motion_epoch']) is int and barrier['base_motion_epoch'] >= 0 and
                  number(barrier['accepted_at_s']) and barrier['accepted_at_s'] <= now and
                  (barrier['settled_at_s'] is None or number(barrier['settled_at_s']) and
                   barrier['accepted_at_s'] <= barrier['settled_at_s'] <= now),
                  'field_checkpoint.geometry', 'invalid saved geometry barrier')
        c.require(not saved_push['groups'] or barrier is not None,
                  'field_checkpoint.geometry', 'push history requires a geometry barrier')
        from treasure_auto.navigation.controlled_push import ControlledPushService
        restored_push = ControlledPushService(self.regions, self.push.limits, origin=self.push.origin)
        empty_push = saved_push['groups'] == {} and saved_push['members'] == {} and saved_push['active_group'] is None
        if empty_push:
            c.require(not data['push_active'] and barrier is None,
                      'field_checkpoint.push', 'empty push history cannot contain pending motion')
            # No physical push settings are needed to preserve an exactly empty
            # push journal alongside a nonempty handling journal.
            restored_push.tick(now)
        else:
            result = restored_push.restore(saved_push, now)
            c.require(result.action != 'denied', 'field_checkpoint.push', result.reason)
        restored_handling = self.handling.restore(data['handling'])
        c.require(not self.restored and not self.push.groups and not self.handling.targets,
                  'field_checkpoint', 'restore requires a fresh controller')
        self.run_id, self.deadline = data['run_id'], data['deadline']
        self.active_target, self.push_active = data['active_target'], data['push_active']
        self.last_now = data['last_now']
        self.push, self.handling = restored_push, restored_handling
        self.geometry_barrier = copy.deepcopy(barrier)
        if not empty_push and result.action in ('stop', 'reobserve'):
            self.push_active = True
            self.geometry_barrier['settled_at_s'] = None
        self.restored = True
