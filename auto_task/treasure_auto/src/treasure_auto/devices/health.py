"""Cached health, distinct fresh samples and dependency-specific capabilities.

Callbacks only cache validated summaries. ROS acquisition is a separate boundary;
the caller must verify actual driver subscriptions, payloads and TF before updating.
"""
from dataclasses import dataclass, replace
import threading
from typing import Optional, Tuple
from uuid import uuid4
from urllib.parse import quote
from treasure_auto.contracts import core as c, models as m
from .config import positive


@dataclass(frozen=True)
class DeviceRule:
    name: str
    message_type: str
    driver: str
    frame_id: str
    driver_version: str
    max_age_s: float
    required_tf: Tuple[str, ...]
    recovery_samples: int
    recovery_s: float

    def __post_init__(self):
        for name in ('name', 'message_type', 'driver', 'frame_id', 'driver_version'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError('{} required'.format(name))
        positive(self.max_age_s, 'max_age_s')
        positive(self.recovery_s, 'recovery_s')
        if type(self.recovery_samples) is not int or self.recovery_samples < 2:
            raise ValueError('recovery requires at least two distinct observations')
        if type(self.required_tf) is not tuple or any(not isinstance(v, str) or not v for v in self.required_tf):
            raise ValueError('explicit TF pairs required')


@dataclass(frozen=True)
class DeviceObservation:
    sample_id: str
    received_at: c.ClockStamp
    captured_at: Optional[c.ClockStamp]
    process_running: bool
    driver: str
    actual_type: str
    subscriber_type: str
    message_count: int
    frame_id: str
    driver_version: str
    transforms: Tuple[str, ...]
    valid_content: bool


class HealthMonitor:
    def __init__(self, clock, envelope, rules, dependencies):
        self.clock, self.envelope = clock, envelope
        self.rules = {rule.name: rule for rule in rules}
        if not self.rules or len(self.rules) != len(rules) or not dependencies:
            raise ValueError('nonempty unique rules and dependencies required')
        self.dependencies = {name: tuple(names) for name, names in dependencies.items()}
        if any(not name or not names or any(n not in self.rules for n in names)
               for name, names in self.dependencies.items()):
            raise ValueError('capability references unknown/empty dependencies')
        self._lock = threading.RLock()
        self._state = {name: [None, 0, None, None] for name in self.rules}
        self._sequence = 0

    def _validate(self, rule, item, now):
        c.require(type(item) is DeviceObservation, 'observation', 'expected device summary')
        c.require(item.process_running is True, 'process', 'process not observed', c.ErrorCode.UNAVAILABLE)
        c.require(item.driver == rule.driver, 'driver', 'wrong driver', c.ErrorCode.UNAVAILABLE)
        c.require(item.actual_type == rule.message_type and item.subscriber_type == rule.message_type,
                  'type', 'actual subscription/message mismatch', c.ErrorCode.INVALID_FIELD)
        c.require(type(item.message_count) is int and item.message_count > 0,
                  'messages', 'no actual messages', c.ErrorCode.UNAVAILABLE)
        c.require(isinstance(item.sample_id, str) and bool(item.sample_id), 'sample_id', 'identity missing')
        c.require(type(item.received_at) is c.ClockStamp, 'received_at', 'clock stamp required')
        c.monotonic(item.received_at, 'received_at')
        c.require(0 <= now - item.received_at < rule.max_age_s,
                  'received_at', 'future or stale data', c.ErrorCode.EXPIRED)
        c.require(item.frame_id == rule.frame_id, 'frame_id', 'wrong coordinate frame')
        c.require(item.driver_version == rule.driver_version, 'driver_version', 'wrong driver version', c.ErrorCode.VERSION_MISMATCH)
        c.require(type(item.transforms) is tuple and all(v in item.transforms for v in rule.required_tf),
                  'transforms', 'required current TF missing', c.ErrorCode.UNAVAILABLE)
        c.require(item.valid_content is True, 'content', 'invalid or unchecked payload')

    def update(self, name, observation):
        with self._lock:
            rule = self.rules[name]
            previous, count, first, reason = self._state[name]
            now = self.clock.now()
            try:
                self._validate(rule, observation, now)
                if previous is not None:
                    # No observer polling is needed to detect an invalid gap.
                    if now - previous.received_at >= rule.max_age_s:
                        count, first = 0, None
                    if observation.sample_id == previous.sample_id:
                        return  # Never refresh age or recovery with a duplicate frame.
                    c.require(observation.message_count > previous.message_count,
                              'messages', 'out of order observation')
                    c.require(observation.received_at - previous.received_at > 0,
                              'received_at', 'non-increasing receipt time')
                    if observation.captured_at is not None and previous.captured_at is not None:
                        c.require(observation.captured_at - previous.captured_at > 0,
                                  'captured_at', 'replayed capture time')
                self._state[name] = [observation, count + 1, first or observation.received_at, None]
            except (c.ContractError, TypeError, AttributeError) as error:
                why = (c.Reason(error.code, error.field, error.detail) if isinstance(error, c.ContractError)
                       else c.Reason(c.ErrorCode.INVALID_FIELD, 'observation', str(error)))
                self._state[name] = [previous, 0, None, why]

    def health(self):
        with self._lock:
            now = self.clock.now()
            device_states = {}
            for name, rule in self.rules.items():
                item, count, first, reason = self._state[name]
                if reason is None and item is not None:
                    try:
                        self._validate(rule, item, now)
                    except c.ContractError as error:
                        reason = c.Reason(error.code, error.field, error.detail)
                        self._state[name] = [item, 0, None, reason]
                if reason:
                    device_states[name] = (c.Fact.REJECTED, reason)
                elif item is None or count < rule.recovery_samples or item.received_at - first < rule.recovery_s:
                    device_states[name] = (c.Fact.UNKNOWN, c.Reason(c.ErrorCode.UNAVAILABLE, name, 'waiting for continuous new observations'))
                else:
                    device_states[name] = (c.Fact.CONFIRMED, None)
            capabilities = []
            for name, dependencies in self.dependencies.items():
                status, reason = c.Fact.CONFIRMED, None
                for dependency in dependencies:
                    state, why = device_states[dependency]
                    if state is c.Fact.REJECTED or (state is c.Fact.UNKNOWN and status is c.Fact.CONFIRMED):
                        status, reason = state, why
                evidence = []
                if status is c.Fact.CONFIRMED:
                    for dependency in dependencies:
                        item = self._state[dependency][0]
                        rule = self.rules[dependency]
                        evidence.append(c.EvidenceRef(
                            dependency + ':' + item.sample_id, self.envelope.run_id, None, None,
                            c.EvidenceKind.INTERFACE, self.envelope.origin, item.captured_at, item.received_at,
                            replace(item.received_at, seconds=item.received_at.seconds + rule.max_age_s),
                            'p01://device/{}/{}'.format(quote(dependency, safe=''), quote(item.sample_id, safe='')), None))
                capabilities.append(m.Capability(name, status, reason, tuple(evidence)))
            # Polling does not renew the underlying observations' lifetime.
            expiry = now.seconds + min(rule.max_age_s for rule in self.rules.values())
            for name, (status, _) in device_states.items():
                if status is c.Fact.CONFIRMED:
                    expiry = min(expiry, self._state[name][0].received_at.seconds + self.rules[name].max_age_s)
            self._sequence += 1
            env = replace(self.envelope, producer=c.Owner.P01, sequence=self._sequence,
                          event_id=str(uuid4()), emitted_at=now,
                          expires_at=replace(now, seconds=expiry))
            return m.HealthSnapshot(env, tuple(capabilities))
