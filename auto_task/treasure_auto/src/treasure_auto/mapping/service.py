"""C0 MappingPort implementation plus local preparation/integration hooks.

No ROS processes, movement, initialpose publishing, occupancy writes, or durable
runtime transactions occur here. P11 serializes calls; callbacks are pull queues.
"""
import math
from dataclasses import dataclass, replace
from enum import Enum
from treasure_auto.contracts import core as c, models as m
from treasure_auto.contracts.rules import require_space_version, require_committed_space
from .geometry import check_parking, angle, finite, reason, validate_polygon
from .localization import LocalizationState, validate_input, fresh
from .storage import MapBundle, with_version


class Mode(str, Enum):
    PREPARATION = 'preparation_mapping'
    COMPETITION = 'competition_localization'
    STOPPED = 'stopped'


@dataclass(frozen=True)
class FootprintProfile:
    body: object
    carried: object
    margin_m: object
    measurement: m.Measurement
    evidence: tuple = ()

    def validate(self, origin):
        c.require(self.body is not None and self.margin_m is not None, 'footprint',
                  'body and margin must be configured', c.ErrorCode.UNCONFIGURED)
        validate_polygon(self.body,'body')
        if self.carried is not None:
            validate_polygon(self.carried,'carried')
        finite(self.margin_m,'margin_m')
        c.require(self.measurement is not m.Measurement.UNMEASURED, 'footprint',
                  'footprint is unmeasured', c.ErrorCode.UNCONFIGURED)
        if origin is c.Origin.LIVE:
            c.require(self.measurement is m.Measurement.VERIFIED and bool(self.evidence) and
                      all(type(e) is c.EvidenceRef and e.origin is c.Origin.LIVE and
                          e.kind is c.EvidenceKind.PHYSICAL for e in self.evidence),
                      'footprint', 'live body envelope needs measured evidence', c.ErrorCode.UNCONFIGURED)


@dataclass(frozen=True)
class PurposeEvidence:
    name: str
    purpose: str
    map_version: str
    space_version: int
    base_motion_epoch: int
    state: c.Fact
    evidence: tuple


class MappingService:
    def __init__(self, bundle, tracker, footprint, mapping_source, localization_source, *, space_readback=None):
        c.require(type(bundle) is MapBundle and bundle.metadata.map_version == tracker.context.versions.map and
                  bundle.metadata.envelope.versions.config == tracker.context.versions.config and
                  bundle.metadata.envelope.origin is tracker.context.origin,
                  'bundle', 'context and bundle must agree', c.ErrorCode.VERSION_MISMATCH)
        c.require(all(type(s) is str and s.strip() for s in (mapping_source,localization_source)) and
                  mapping_source != localization_source, 'sources', 'two explicit distinct source identities required')
        self.bundle, self.tracker, self.footprint = bundle, tracker, footprint
        self.mapping_source, self.localization_source = mapping_source, localization_source
        self.space_readback = space_readback
        self.mode = Mode.PREPARATION
        self.holding = c.Holding.UNKNOWN
        self._holding=None
        self._holding_seen={}
        self._holding_latest=None
        self._holding_sequences={}
        self._purpose, self._recorded, self._draft = {}, {}, {}
        self._draft_evidence = {}
        self.record_history = []
        self._events = []
        self._sources_at = None

    def observe_holding(self, snapshot, envelope):
        """Consume a P07-owned C0 fact; stale/invalid input clears local usability."""
        try:
            c.require(type(snapshot) is c.HoldingSnapshot,'holding','C0 HoldingSnapshot required')
            validate_input(envelope,self.tracker.context,self.tracker.clock())
            c.require(envelope.producer is c.Owner.P07,'holding.producer','P07 owns holding facts')
            key=envelope.event_id
            if key in self._holding_seen:
                c.require(self._holding_seen[key]==(snapshot,envelope),'holding.event_id',
                          'same P07 event with changed payload',c.ErrorCode.ID_CONFLICT)
                return m.ContractCheck(True,None)
            stream=(envelope.producer,envelope.instance_id,envelope.control_epoch,envelope.request_id,envelope.action_id)
            c.require(stream not in self._holding_sequences or envelope.sequence>self._holding_sequences[stream],
                      'holding.sequence','sequence did not advance within the C0 producer/action stream',c.ErrorCode.STALE_REQUEST)
            if self._holding_latest is not None:
                c.require(envelope.emitted_at-self._holding_latest.emitted_at>=0,
                          'holding.sequence','old P07 observation cannot replace newer fact',c.ErrorCode.STALE_REQUEST)
            if snapshot.state in (c.Holding.EMPTY,c.Holding.HELD):
                c.require(bool(snapshot.evidence),'holding.evidence','physical evidence required',c.ErrorCode.EVIDENCE_MISSING)
                for e in snapshot.evidence:
                    fresh(e,envelope,self.tracker.clock(),self.tracker.policy.max_age_s)
            if snapshot.state is not self.holding:
                self._purpose.clear()
            self.holding,self._holding=snapshot.state,(snapshot,envelope)
            self._holding_seen[key]=(snapshot,envelope)
            self._holding_latest=envelope
            self._holding_sequences[stream]=envelope.sequence
            return m.ContractCheck(True,None)
        except c.ContractError as error:
            self.holding,self._holding=c.Holding.UNKNOWN,None
            self._purpose.clear()
            return m.ContractCheck(False,reason(error))

    def set_mode(self, mode, publishers, deadline):
        try:
            c.require(type(mode) is Mode, 'mode', 'explicit operating mode required')
            c.require(deadline-self.tracker.clock() > 0, 'deadline', 'mode transition timed out', c.ErrorCode.TIMEOUT)
            expected = () if mode is Mode.STOPPED else (self.mapping_source if mode is Mode.PREPARATION else self.localization_source,)
            c.require(type(publishers) is tuple and publishers == expected, 'publishers',
                      'exactly the selected map-to-odom publisher must be observed', c.ErrorCode.CONFLICT)
            if mode is Mode.COMPETITION:
                status = self.bundle.metadata.measurement
                c.require(status is not m.Measurement.UNMEASURED and
                          (self.tracker.context.origin is not c.Origin.LIVE or status is m.Measurement.VERIFIED),
                          'map.measurement', 'unmeasured map/regions/points cannot be competition data', c.ErrorCode.UNCONFIGURED)
                c.require(not self._draft, 'waypoints', 'save, reload and install draft points before competition', c.ErrorCode.CONFLICT)
            if self.mode is not mode:
                self._purpose.clear()
                # Moving between publishers invalidates any prior localization solution.
                self.tracker.invalidate(c.Reason(c.ErrorCode.STALE_GEOMETRY,'mode','source mode changed'))
            self.mode, self._sources_at = mode, self.tracker.clock()
            return m.ContractCheck(True,None)
        except c.ContractError as error:
            self.mode = Mode.STOPPED
            self._purpose.clear()
            self.tracker.invalidate(reason(error))
            return m.ContractCheck(False,reason(error))

    def map_metadata(self):
        return replace(self.bundle.metadata, envelope=self.tracker.envelope())

    def localization(self):
        return self.tracker.snapshot()

    def names_for(self, purpose):
        return tuple(p.name for p in self.bundle.waypoints if p.purpose == purpose)

    def region(self, area):
        c.require(type(area) is m.Area,'area','A/B/C/D required')
        return next(r for r in self.bundle.metadata.regions if r.area is area)

    def set_purpose_evidence(self, evidence):
        c.require(type(evidence) is PurposeEvidence, 'purpose', 'P02 purpose assessment required')
        self._purpose[(evidence.name,evidence.purpose)] = evidence

    def _query_envelope(self, request):
        return self.tracker.envelope(versions=replace(request.envelope.versions,
                 config=self.tracker.context.versions.config,map=self.bundle.metadata.map_version),
                 request_id=request.envelope.request_id,action_id=request.envelope.action_id,
                 target_id=request.envelope.target_id)

    def query(self, request, space):
        try:
            now = self.tracker.clock()
            validate_input(request.envelope,self.tracker.context,now)
            c.require(request.envelope.base_motion_epoch == self.tracker.base_motion_epoch,
                      'base_motion_epoch','requery after base motion',c.ErrorCode.STALE_GEOMETRY)
            c.require(self.mode is Mode.COMPETITION, 'mode', 'competition localization not active', c.ErrorCode.UNAVAILABLE)
            c.require(self._sources_at is not None and now-self._sources_at <= self.tracker.policy.max_age_s,
                      'publishers','source inventory needs refresh',c.ErrorCode.EXPIRED)
            c.require(self.localization().trusted is c.Fact.CONFIRMED, 'localization',
                      'current localization is not trusted', c.ErrorCode.EVIDENCE_MISSING)
            point = next((p for p in self.bundle.waypoints if p.name == request.name and p.purpose == request.purpose),None)
            c.require(point is not None and point.pose is not None,'waypoint','unknown/unconfigured name and purpose',c.ErrorCode.UNCONFIGURED)
            require_space_version(request.envelope,space)
            validate_input(space.envelope,self.tracker.context,now)
            require_committed_space(space, self.space_readback)
            now = self.tracker.clock()
            validate_input(request.envelope,self.tracker.context,now)
            validate_input(space.envelope,self.tracker.context,now)
            c.require(self.localization().trusted is c.Fact.CONFIRMED, 'localization',
                      'localization expired during publication read', c.ErrorCode.EVIDENCE_MISSING)
            self.footprint.validate(self.tracker.context.origin)
            c.require(self._holding is not None,'holding','no current P07 holding fact',c.ErrorCode.EVIDENCE_MISSING)
            holding,holding_envelope=self._holding
            validate_input(holding_envelope,self.tracker.context,now)
            for e in holding.evidence:
                fresh(e,holding_envelope,now,self.tracker.policy.max_age_s)
            assessment = self._purpose.get((point.name,point.purpose))
            c.require(assessment is not None and assessment.state is c.Fact.CONFIRMED and bool(assessment.evidence),
                      'purpose','view/workspace/placement suitability lacks physical evidence',c.ErrorCode.EVIDENCE_MISSING)
            c.require((assessment.map_version,assessment.space_version,assessment.base_motion_epoch) ==
                      (self.bundle.metadata.map_version,space.version,self.tracker.base_motion_epoch),
                      'purpose','purpose evidence uses obsolete map/space/base pose',c.ErrorCode.STALE_GEOMETRY)
            for evidence in assessment.evidence:
                fresh(evidence,request.envelope,now,self.tracker.policy.max_age_s)
            check = check_parking(self.bundle,point.pose,self.footprint.body,space,self.holding,
                                  self.footprint.carried,self.footprint.margin_m)
            c.require(check.accepted,check.reason.field if check.reason else 'geometry',
                      check.reason.detail if check.reason else 'geometry rejected',
                      check.reason.code if check.reason else c.ErrorCode.CONFLICT)
            output=self._query_envelope(request)
            dependencies=(request.envelope,space.envelope,holding_envelope,self.localization().envelope)
            expiries=[e.expires_at.seconds for e in dependencies]+[output.expires_at.seconds,
                      self._sources_at.seconds+self.tracker.policy.max_age_s]
            for e in assessment.evidence+holding.evidence:
                expiries.extend((e.valid_until.seconds,e.received_at.seconds+self.tracker.policy.max_age_s))
            output=replace(output,expires_at=replace(output.expires_at,seconds=min(expiries)))
            return m.WaypointResult(output,point.name,point.purpose,point.pose,c.Fact.CONFIRMED,None)
        except c.ContractError as error:
            return m.WaypointResult(self._query_envelope(request),request.name,request.purpose,None,c.Fact.UNKNOWN,reason(error))

    def record(self, waypoint):
        return self._record(waypoint,False)

    def replace_waypoint(self, waypoint):
        """Explicit preparation-only replacement; old entries remain in record_history."""
        return self._record(waypoint,True)

    def _record(self, waypoint, replace_existing):
        try:
            c.require(self.mode is Mode.PREPARATION,'mode','recording is preparation-only',c.ErrorCode.CONFLICT)
            validate_input(waypoint.envelope,self.tracker.context,self.tracker.clock())
            c.require(waypoint.envelope.base_motion_epoch == self.tracker.base_motion_epoch,
                      'base_motion_epoch','record request uses old base pose',c.ErrorCode.STALE_GEOMETRY)
            key = waypoint.envelope.event_id
            if key in self._recorded:
                c.require(self._recorded[key] == (waypoint,replace_existing), 'event_id',
                          'same recording event has changed payload',c.ErrorCode.ID_CONFLICT)
                return m.ContractCheck(True,None)
            c.require(waypoint.purpose in ('observe','grasp','place','start','transit'),
                      'purpose','unsupported waypoint purpose')
            exists = waypoint.name in self._draft or any(p.name == waypoint.name for p in self.bundle.waypoints)
            c.require(not exists or replace_existing,'name','explicit replacement required',c.ErrorCode.CONFLICT)
            stable = self.tracker.stable_pose()
            c.require(waypoint.pose is not None and waypoint.pose.frame_id == stable.frame_id and
                      math.hypot(waypoint.pose.x-stable.x,waypoint.pose.y-stable.y) <= self.tracker.policy.stable_m and
                      abs(angle(waypoint.pose.yaw-stable.yaw)) <= self.tracker.policy.stable_rad,
                      'pose','record proposal disagrees with stopped observation window')
            recorded = replace(waypoint,pose=stable,envelope=self.tracker.envelope(),valid=c.Fact.UNKNOWN,
                               reason=c.Reason(c.ErrorCode.EVIDENCE_MISSING,'geometry','recorded reference still needs query and route checks'))
            refs = tuple(e for sample in self.tracker.samples for e in sample.chain.evidence+(sample.stop_evidence,))
            self._draft[waypoint.name],self._draft_evidence[waypoint.name] = recorded,refs
            self.record_history.append(recorded)
            self._recorded[key] = (waypoint,replace_existing)
            return m.ContractCheck(True,None)
        except c.ContractError as error:
            return m.ContractCheck(False,reason(error))

    def export_draft(self, version):
        c.require(self.mode is Mode.PREPARATION,'mode','static versions frozen in competition',c.ErrorCode.CONFLICT)
        c.require(version != self.bundle.metadata.map_version,'version','new point records require a new bundle version',c.ErrorCode.VERSION_MISMATCH)
        points={p.name:p for p in self.bundle.waypoints}
        points.update(self._draft)
        refs=dict(self.bundle.waypoint_evidence)
        refs.update(self._draft_evidence)
        return with_version(replace(self.bundle,waypoints=tuple(points.values()),waypoint_evidence=tuple(refs.items())),version)

    def install(self, bundle):
        try:
            c.require(self.mode is Mode.PREPARATION,'mode','competition static versions are frozen',c.ErrorCode.CONFLICT)
            c.require(type(bundle) is MapBundle and bundle.metadata.envelope.origin is self.tracker.context.origin and
                      bundle.metadata.envelope.versions.config == self.tracker.context.versions.config,
                      'bundle','configuration/provenance mismatch',c.ErrorCode.VERSION_MISMATCH)
            c.require(bundle.metadata.map_version != self.bundle.metadata.map_version or bundle == self.bundle,
                      'map_version','changed static content needs a new version',c.ErrorCode.VERSION_MISMATCH)
            self.bundle=bundle
            self.tracker.context=replace(self.tracker.context,versions=replace(self.tracker.context.versions,map=bundle.metadata.map_version))
            self._draft.clear()
            self._draft_evidence.clear()
            self._purpose.clear()
            self.tracker.last=None
            self.tracker.invalidate(c.Reason(c.ErrorCode.STALE_GEOMETRY,'map_version','static bundle loaded; reinitialize localization'),
                                    LocalizationState.UNINITIALIZED)
            self._events.append(self.map_metadata())
            return m.ContractCheck(True,None)
        except c.ContractError as error:
            return m.ContractCheck(False,reason(error))

    def take_events(self):
        """Pull immutable C0 MapMetadata/LocalizationSnapshot notifications."""
        events=tuple(self._events)+self.tracker.take_events()
        self._events.clear()
        return events
