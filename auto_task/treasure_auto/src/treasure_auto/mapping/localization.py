"""Evidence-driven localization, independent of ROS and actuator submission.

The input dataclasses are P02 algorithm inputs, not substitute public C0 wire
contracts. P01's timestamped data adapter remains an integration prerequisite.
"""
import math
from dataclasses import dataclass, replace
from enum import Enum
from treasure_auto.contracts import core as c, models as m
from treasure_auto.contracts.rules import require_physical_evidence, validate_envelope
from .geometry import angle, finite, reason


class LocalizationState(str, Enum):
    UNINITIALIZED = 'uninitialized'
    CONFIRMING = 'confirming'
    USABLE = 'usable'
    ABNORMAL = 'abnormal'
    RELOCALIZING = 'relocalizing'


@dataclass(frozen=True)
class LocalizationPolicy:
    min_samples: int
    confirmation_s: float
    max_gap_s: float
    max_age_s: float
    max_skew_s: float
    max_error_m: float
    max_error_rad: float
    jump_m: float
    jump_rad: float
    initial_m: float
    initial_rad: float
    stable_m: float
    stable_rad: float
    measurement: m.Measurement
    evidence: tuple = ()

    def __post_init__(self):
        c.require(type(self.min_samples) is int and self.min_samples >= 2, 'min_samples', 'at least two observations')
        for name in ('confirmation_s','max_gap_s','max_age_s','max_skew_s','max_error_m',
                     'max_error_rad','jump_m','jump_rad','initial_m','initial_rad','stable_m','stable_rad'):
            finite(getattr(self,name), name)
            c.require(getattr(self,name) > 0, name, 'explicit positive measured limit required')
        c.require(type(self.measurement) is m.Measurement and type(self.evidence) is tuple,
                  'measurement', 'explicit measurement status and evidence tuple required')


@dataclass(frozen=True)
class TransformSample:
    envelope: c.Envelope
    pose: c.MapPose  # child origin expressed in parent frame
    child_frame: str
    evidence: c.EvidenceRef
    source_now: object = None  # trusted current stamp in the acquisition clock domain


@dataclass(frozen=True)
class ChainResult:
    pose: object
    odometry: object
    correction: object
    evidence: tuple
    envelope: object
    reason: object
    valid_until: object = None


def validate_input(envelope, context, now):
    c.require(envelope.versions.config == context.versions.config and
              envelope.versions.map == context.versions.map, 'versions',
              'configuration/map changed', c.ErrorCode.VERSION_MISMATCH)
    current = m.ConsumerContext(context.run_id, context.instance_id, context.control_epoch,
                                now, envelope.versions, None, None, None, None, None, None, context.origin)
    validate_envelope(envelope, current)


def fresh(evidence, envelope, now, max_age_s):
    require_physical_evidence(evidence, envelope, now, envelope.origin)
    c.require(now-evidence.received_at < max_age_s-1e-9, 'received_at', 'observation too old', c.ErrorCode.EXPIRED)
    capture = evidence.captured_at
    # Do not mix ROS/sensor and monotonic time. Same-domain future captures can be rejected directly.
    if (capture.basis,capture.clock_id,capture.boot_id) == (now.basis,now.clock_id,now.boot_id):
        c.require(0 <= now-capture <= max_age_s, 'captured_at', 'stale or future capture', c.ErrorCode.EXPIRED)


def resolve_chain(map_to_odom, odom_to_base, context, now, policy, base_frame):
    try:
        c.require(type(map_to_odom) is TransformSample and type(odom_to_base) is TransformSample,
                  'transform', 'map->odom and odom->base observations both required', c.ErrorCode.UNAVAILABLE)
        first, second = map_to_odom, odom_to_base
        c.require(all(type(t.pose) is c.MapPose and type(t.envelope) is c.Envelope and
                      type(t.evidence) is c.EvidenceRef and type(t.source_now) is c.ClockStamp
                      for t in (first,second)), 'transform', 'complete typed pose, evidence and source clock required')
        c.require(first.pose.frame_id == 'map' and first.child_frame == 'odom' and
                  second.pose.frame_id == 'odom' and second.child_frame == base_frame,
                  'transform.frame_id', 'expected map->odom->configured base chain')
        remaining=[]
        for transform in (first, second):
            c.require(transform.envelope.producer is c.Owner.P01, 'producer', 'P01 data adapter required')
            validate_input(transform.envelope, context, now)
            fresh(transform.evidence, transform.envelope, now, policy.max_age_s)
            c.require(0 <= transform.source_now-transform.evidence.captured_at < policy.max_age_s,
                      'captured_at','stale/future source capture',c.ErrorCode.EXPIRED)
            # source_now is sampled at monotonic received_at. Carry the already
            # consumed capture lifetime into the receiver clock without mixing epochs.
            source_age=transform.source_now-transform.evidence.captured_at
            remaining.extend((transform.envelope.expires_at-now,transform.evidence.valid_until-now,
                              policy.max_age_s-source_age-(now-transform.evidence.received_at)))
        c.require(min(remaining)>1e-9,'captured_at','source capture lifetime exhausted',c.ErrorCode.EXPIRED)
        c.require(abs(first.evidence.captured_at-second.evidence.captured_at) <= policy.max_skew_s,
                  'captured_at', 'TF chain observations are not synchronized', c.ErrorCode.UNAVAILABLE)
        a, b = first.pose, second.pose
        cs, sn = math.cos(a.yaw), math.sin(a.yaw)
        error_m = a.error_m+b.error_m+2*math.hypot(b.x,b.y)*math.sin(min(math.pi,a.error_rad)/2)
        result = c.MapPose('map', a.x+cs*b.x-sn*b.y, a.y+sn*b.x+cs*b.y,
                           angle(a.yaw+b.yaw), 'm', 'rad', error_m, a.error_rad+b.error_rad)
        return ChainResult(result, b, a, (first.evidence,second.evidence), second.envelope, None,
                           replace(now,seconds=now.seconds+min(remaining)))
    except c.ContractError as error:
        return ChainResult(None, None, None, (), None, reason(error))


@dataclass(frozen=True)
class Observation:
    chain: ChainResult
    stopped: c.Fact
    stop_evidence: object


class LocalizationTracker:
    def __init__(self, context, policy, clock, base_frame):
        c.require(context.versions.map is not None, 'map', 'select a map before localization')
        c.require(type(base_frame) is str and bool(base_frame.strip()) and base_frame not in ('map','odom'),
                  'base_frame', 'explicit distinct body frame required')
        if context.origin is c.Origin.LIVE:
            c.require(policy.measurement is m.Measurement.VERIFIED and bool(policy.evidence) and
                      all(type(e) is c.EvidenceRef and e.origin is c.Origin.LIVE and
                          e.kind is c.EvidenceKind.PHYSICAL for e in policy.evidence),
                      'policy', 'live localization requires measured limits with evidence', c.ErrorCode.UNCONFIGURED)
        self.context, self.policy, self.clock, self.base_frame = context, policy, clock, base_frame
        self.state = LocalizationState.UNINITIALIZED
        self.base_motion_epoch = context.base_motion_epoch or 0
        self.historical_pose = None
        self.seed, self.last = None, None
        self.samples = []
        self._seen, self._events = {}, []
        self._capture = None
        self._odom_capture = None
        self._source_seen,self._evidence_seen={},{}
        self._reset_at = clock()
        self._serial = 0
        self._reason = c.Reason(c.ErrorCode.UNAVAILABLE, 'localization', 'not initialized')

    def envelope(self, **changes):
        now = self.clock()
        self._serial += 1
        return replace(self.context, producer=c.Owner.P02,
                       event_id='{}:P02:{}'.format(self.context.instance_id,self._serial),
                       sequence=self._serial, emitted_at=now,
                       expires_at=replace(now, seconds=now.seconds+self.policy.max_age_s),
                       base_motion_epoch=self.base_motion_epoch, **changes)

    def _snapshot(self):
        trusted = self.state is LocalizationState.USABLE
        envelope=self.envelope()
        if trusted:
            until=min([envelope.expires_at.seconds,self.last.valid_until.seconds]+
                      [min(e.valid_until.seconds,e.received_at.seconds+self.policy.max_age_s) for e in self.last.evidence])
            envelope=replace(envelope,expires_at=replace(envelope.expires_at,seconds=until))
        return m.LocalizationSnapshot(envelope, self.last.pose if trusted else None,
                                      c.Fact.CONFIRMED if trusted else c.Fact.UNKNOWN,
                                      self.base_motion_epoch, self.last.evidence if trusted else (), self._reason)

    def invalidate(self, why, state=LocalizationState.ABNORMAL):
        self.base_motion_epoch += 1
        self.state, self.seed, self._reason = state, None, why
        self.samples.clear()
        self._reset_at = self.clock()
        self._events.append(self._snapshot())

    def initialize(self, seed):
        # Records a requested estimate; it never sends initialpose or any movement.
        c.require(type(seed) is c.MapPose and seed.frame_id == 'map', 'seed', 'map-frame estimate required')
        state = LocalizationState.CONFIRMING if self.state is LocalizationState.UNINITIALIZED else LocalizationState.RELOCALIZING
        self.invalidate(c.Reason(c.ErrorCode.EVIDENCE_MISSING,'localization','initial estimate awaits continuous observations'), state)
        self.seed = seed

    def restore_history(self, snapshot):
        c.require(type(snapshot) is m.LocalizationSnapshot, 'history', 'C0 localization snapshot required')
        self.historical_pose = snapshot.pose
        self.base_motion_epoch = max(self.base_motion_epoch, snapshot.base_motion_epoch)
        self.last = None
        self.invalidate(c.Reason(c.ErrorCode.EVIDENCE_MISSING,'history','disk pose is context only'),
                        LocalizationState.UNINITIALIZED)

    def snapshot(self):
        if self.last is not None and self.state is LocalizationState.USABLE:
            try:
                validate_input(self.last.envelope,self.context,self.clock())
                c.require(self.last.valid_until-self.clock()>1e-9,'localization',
                          'TF chain lifetime exhausted',c.ErrorCode.EXPIRED)
                for e in self.last.evidence:
                    fresh(e, self.last.envelope, self.clock(), self.policy.max_age_s)
            except c.ContractError as error:
                self.invalidate(reason(error))
        return self._snapshot()

    def take_events(self):
        result = tuple(self._events)
        self._events.clear()
        return result

    def observe(self, map_to_odom, odom_to_base, stopped=c.Fact.UNKNOWN, stop_evidence=None):
        now, p = self.clock(), self.policy
        result = resolve_chain(map_to_odom, odom_to_base, self.context, now, p, self.base_frame)
        if result.reason:
            self.invalidate(result.reason)
            return self._snapshot()
        try:
            key = (map_to_odom.envelope.event_id, odom_to_base.envelope.event_id)
            value = (map_to_odom, odom_to_base, stopped, stop_evidence)
            if key in self._seen:
                c.require(self._seen[key] == value, 'event_id', 'same identity with different payload', c.ErrorCode.ID_CONFLICT)
                return self.snapshot()
            for transform in (map_to_odom,odom_to_base):
                identity=(transform.child_frame,transform.envelope.event_id)
                c.require(identity not in self._source_seen or self._source_seen[identity]==transform,
                          'event_id','individual TF event payload changed',c.ErrorCode.ID_CONFLICT)
                ref=transform.evidence
                c.require(ref.evidence_id not in self._evidence_seen or self._evidence_seen[ref.evidence_id]==ref,
                          'evidence_id','evidence identity payload changed',c.ErrorCode.ID_CONFLICT)
            for transform in (map_to_odom,odom_to_base):
                self._source_seen[(transform.child_frame,transform.envelope.event_id)]=transform
                self._evidence_seen[transform.evidence.evidence_id]=transform.evidence
            capture = result.evidence[0].captured_at
            c.require(all(e.received_at-self._reset_at >= 0 for e in result.evidence),
                      'received_at', 'observation predates initialization', c.ErrorCode.EXPIRED)
            for transform in (map_to_odom,odom_to_base):
                capture_age=transform.source_now-transform.evidence.captured_at
                c.require(transform.evidence.received_at-self._reset_at >= capture_age-1e-9,
                          'captured_at','capture predates initialization/revalidation fence',c.ErrorCode.EXPIRED)
            if self._capture is not None:
                gap = capture-self._capture
                c.require(gap > 0, 'captured_at', 'replayed or out of order observation', c.ErrorCode.DUPLICATE)
                if self.samples:
                    c.require(gap <= p.max_gap_s+1e-9, 'captured_at', 'continuous evidence interrupted', c.ErrorCode.EXPIRED)
            if self._odom_capture is not None:
                c.require(result.evidence[1].captured_at-self._odom_capture > 0,'captured_at',
                          'odometry capture is repeated/out of order',c.ErrorCode.DUPLICATE)
            c.require(result.pose.error_m <= p.max_error_m and result.pose.error_rad <= p.max_error_rad,
                      'uncertainty', 'localization uncertainty exceeds configured limit', c.ErrorCode.UNAVAILABLE)
            if self.last is not None and self.samples:
                a, b = self.last.correction, result.correction
                c.require(math.hypot(a.x-b.x,a.y-b.y) <= p.jump_m and abs(angle(a.yaw-b.yaw)) <= p.jump_rad,
                          'localization', 'map-to-odom correction jumped', c.ErrorCode.STALE_GEOMETRY)
            if self.seed is not None and self.state is not LocalizationState.USABLE:
                c.require(math.hypot(result.pose.x-self.seed.x,result.pose.y-self.seed.y) <= p.initial_m and
                          abs(angle(result.pose.yaw-self.seed.yaw)) <= p.initial_rad,
                          'initial_pose', 'observation disagrees with initial area or heading', c.ErrorCode.UNAVAILABLE)
            c.require(type(stopped) is c.Fact, 'stopped', 'explicit stop evidence state required')
            if stopped is c.Fact.CONFIRMED:
                c.require(type(stop_evidence) is c.EvidenceRef, 'stopped', 'physical stop proof required', c.ErrorCode.EVIDENCE_MISSING)
                fresh(stop_evidence, result.envelope, now, p.max_age_s)
                c.require(abs(stop_evidence.captured_at-capture) <= p.max_skew_s,
                          'stopped', 'stop evidence must cover this sample')
            moved = self.last is not None and (math.hypot(result.odometry.x-self.last.odometry.x,
                     result.odometry.y-self.last.odometry.y) > 1e-10 or
                     abs(angle(result.odometry.yaw-self.last.odometry.yaw)) > 1e-10 or
                     math.hypot(result.pose.x-self.last.pose.x,result.pose.y-self.last.pose.y) > 1e-10 or
                     abs(angle(result.pose.yaw-self.last.pose.yaw)) > 1e-10)
            self._seen[key], self._capture = value, capture
            self._odom_capture=result.evidence[1].captured_at
            self.last, self.historical_pose = result, result.pose
            self.samples.append(Observation(result, stopped, stop_evidence))
            # Retain enough recent samples for a full confirmation/recording window.
            while len(self.samples) > p.min_samples and capture-self.samples[1].chain.evidence[0].captured_at >= p.confirmation_s:
                self.samples.pop(0)
            if self.seed is not None and len(self.samples) >= p.min_samples and (
                    capture-self.samples[0].chain.evidence[0].captured_at >= p.confirmation_s-1e-9):
                self.state, self._reason = LocalizationState.USABLE, None
            if moved:
                self.base_motion_epoch += 1
                event = self._snapshot()
                self._events.append(replace(event, reason=c.Reason(c.ErrorCode.STALE_GEOMETRY,
                    'base_motion_epoch','base moved; requery all targets and paths; occupancies unchanged')))
            return self._snapshot()
        except c.ContractError as error:
            self.invalidate(reason(error))
            return self._snapshot()

    def stable_pose(self):
        """Average only a currently trusted, continuously stopped recent window."""
        c.require(self.snapshot().trusted is c.Fact.CONFIRMED and len(self.samples) >= self.policy.min_samples,
                  'localization', 'trusted stable observation window required', c.ErrorCode.EVIDENCE_MISSING)
        now = self.clock()
        for sample in self.samples:
            c.require(sample.chain.valid_until-now>1e-9,'stability','old TF chain in recording window',c.ErrorCode.EXPIRED)
            c.require(sample.stopped is c.Fact.CONFIRMED, 'stopped', 'actual stopped evidence required', c.ErrorCode.EVIDENCE_MISSING)
            fresh(sample.stop_evidence, sample.chain.envelope, now, self.policy.max_age_s)
            for e in sample.chain.evidence:
                fresh(e, sample.chain.envelope, now, self.policy.max_age_s)
        poses = [sample.chain.pose for sample in self.samples]
        x, y = sum(p.x for p in poses)/len(poses), sum(p.y for p in poses)/len(poses)
        sn, cs = sum(math.sin(p.yaw) for p in poses), sum(math.cos(p.yaw) for p in poses)
        c.require(math.hypot(sn,cs) > 1e-8, 'yaw', 'ambiguous circular average')
        yaw = math.atan2(sn,cs)
        spread = max(math.hypot(p.x-x,p.y-y) for p in poses)
        angular = max(abs(angle(p.yaw-yaw)) for p in poses)
        c.require(spread <= self.policy.stable_m and angular <= self.policy.stable_rad,
                  'stability', 'samples are not stationary enough', c.ErrorCode.UNAVAILABLE)
        return c.MapPose('map',x,y,yaw,'m','rad',max(p.error_m for p in poses)+spread,
                         max(p.error_rad for p in poses)+angular)
