"""C0 PerceptionPort implementation with explicit, injected offline/live boundaries.

The caller supplies image bytes, a current acquisition/pose context, an existing
backend and an audit sink. There are no robot, file-store or ROS side effects here.
"""
import copy
import math
import time
import uuid
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from typing import Optional, Tuple
from treasure_auto.contracts import core as c, models as m, rules
from treasure_auto.geometry.surface import known_layer_plane
from .association import TargetTracker
from .backend import BackendBatch
from .geometry import contains, fit_top, overlaps, require_convex_polygon
from .pixels import FrameGate, TransformPlan, RawImage
from .tool import observe_tool
from .strategy import PerceptionPolicyReport, build_policy_report, entity_outline_groups, unavailable_report


@dataclass(frozen=True)
class ObservationContext:
    """Private wiring arguments. Provider must bind ROIs/barriers to the capture pose.

    capture_floor and sensor_now use the SAME sensor clock as image.captured_at;
    consumer.now is the authoritative monotonic receipt/expiration clock.
    """
    consumer: m.ConsumerContext
    capture_floor: c.ClockStamp
    sensor_now: c.ClockStamp
    allowed_polygon: Tuple[m.PixelPoint, ...]
    board_polygons: Tuple[Tuple[m.PixelPoint, ...], ...]
    camera_frame: str
    capture_pose: c.MapPose
    # These IDs refer to this batch only. Compatible lidar absence never vetoes vision.
    correspondence_conflicts: Tuple[str, ...] = ()
    lidar_entity_ids: Tuple[str, ...] = ()
    base_frame: Optional[str] = None


def _plain(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, (tuple, list)):
        return [_plain(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def _reason(error, field):
    if isinstance(error, c.ContractError):
        return c.Reason(error.code, error.field, error.detail)
    return c.Reason(c.ErrorCode.UNAVAILABLE, field, '{}: {}'.format(type(error).__name__, error))


class PerceptionService:
    """One active run/controller per service; serialized calls (one observer worker).

    audit_sink receives JSON-compatible diagnostics and must raise on failed storage.
    An in-memory sink is suitable only for offline tests. P10B's general result-store
    contract is pending; installing this class does not claim durable persistence.
    """
    def __init__(self, config, image_reader, context_provider, backend, audit_sink):
        self.config, self.image_reader, self.context_provider = config, image_reader, context_provider
        self.backend, self.audit_sink = backend, audit_sink
        self._gate = FrameGate(config.frame_size, config.max_age_s)
        self._tracker = TargetTracker(config)
        self._scope = None
        self._sequence = 0
        self.latest = None
        self.last_audit = None
        self.policy_report = None  # type: Optional[PerceptionPolicyReport]

    @property
    def accepted_frames(self):
        return self._gate.accepted_count

    def _envelope(self, image, now, sensor_now=None):
        self._sequence += 1
        # Expired failure results keep their expired validity, never mint new evidence life.
        emitted = now if image.envelope.expires_at - now > 0 else image.envelope.emitted_at
        expiry = image.envelope.expires_at
        if sensor_now is not None:
            remaining = min(expiry - now, image.image.valid_until - now,
                            self.config.max_age_s - (now - image.image.received_at),
                            self.config.max_age_s - (sensor_now - image.image.captured_at))
            c.require(remaining > 0, 'result.expiry', 'image expired during processing', c.ErrorCode.EXPIRED)
            expiry = replace(now, seconds=now.seconds + remaining)
        return replace(image.envelope, producer=c.Owner.P04, event_id='p04:' + uuid.uuid4().hex,
                       sequence=self._sequence, emitted_at=emitted, expires_at=expiry)

    def _current(self, image):
        ctx = self.context_provider()
        c.require(type(ctx) is ObservationContext, 'context', 'P04 observation wiring context required')
        rules.validate_envelope(image.envelope, ctx.consumer)
        c.require(ctx.consumer.request_id is not None and
                  (image.envelope.request_id, image.envelope.action_id, image.envelope.target_id) ==
                  (ctx.consumer.request_id, ctx.consumer.action_id, ctx.consumer.target_id),
                  'request_id', 'exact active observation/action/target required', c.ErrorCode.STALE_REQUEST)
        c.require((image.camera_frame, image.capture_pose) == (ctx.camera_frame, ctx.capture_pose),
                  'capture_pose', 'source or pose does not match capture-aligned region context', c.ErrorCode.STALE_GEOMETRY)
        for ids in (ctx.correspondence_conflicts, ctx.lidar_entity_ids):
            c.require(type(ids) is tuple and all(type(value) is str and bool(value.strip()) for value in ids),
                      'correspondence', 'frame-local correspondence identities required')
        return ctx

    def _quality(self, raw):
        import numpy as np
        import cv2
        pixels = raw.array()
        gray = pixels if raw.encoding == 'mono8' else cv2.cvtColor(
            pixels, cv2.COLOR_RGB2GRAY if raw.encoding == 'rgb8' else cv2.COLOR_BGR2GRAY)
        contrast = float(np.std(gray))
        saturation = float(np.mean((gray == 0) | (gray == 255)))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        c.require(contrast >= self.config.min_contrast and saturation <= self.config.max_saturation_fraction and
                  sharpness >= self.config.min_sharpness, 'image.quality', 'image fails measured quality thresholds')
        return dict(contrast=contrast, saturation_fraction=saturation, sharpness=sharpness)

    @staticmethod
    def _same_alignment(before, after):
        names = ('allowed_polygon', 'board_polygons', 'correspondence_conflicts', 'base_frame')
        c.require(all(getattr(before, name) == getattr(after, name) for name in names),
                  'capture_alignment', 'capture-aligned region or correspondence changed during processing',
                  c.ErrorCode.STALE_GEOMETRY)

    def _batch(self, image, raw):
        c.require(self.backend is not None, 'backend', 'no existing competition recognition backend/model available', c.ErrorCode.UNAVAILABLE)
        batch = self.backend.infer(image, raw)
        c.require(type(batch) is BackendBatch, 'backend.output', 'typed output required')
        c.require((batch.observation_id, batch.image_sha256, batch.origin) ==
                  (image.observation_id, image.image.sha256, image.envelope.origin),
                  'backend.frame', 'output is not bound to this exact frame/provenance', c.ErrorCode.ID_CONFLICT)
        c.require((batch.model_version, batch.class_map_version) ==
                  (self.config.model_version, self.config.class_map_version), 'backend.version',
                  'config/model/class map versions differ', c.ErrorCode.VERSION_MISMATCH)
        t = batch.transform
        c.require(type(t) is TransformPlan and (t.full_width, t.full_height, t.transform, t.distortion) ==
                  (image.width, image.height, image.preprocessing, image.distortion), 'preprocessing',
                  'image and backend disagree on original pixels/distortion/transform', c.ErrorCode.VERSION_MISMATCH)
        return batch

    def _candidates(self, request, batch, ctx, audit):
        image = request.image
        entity_by_id, fitted, rejected, known_planes = {}, {}, {}, {}
        outline_groups = entity_outline_groups(batch, image, self.config)
        aliases = {entity_id for group in outline_groups if len(group) > 1 for entity_id in group}
        for entity in batch.entities:
            c.require(type(entity.entity_id) is str and bool(entity.entity_id.strip()) and entity.entity_id not in entity_by_id,
                      'entity_id', 'missing or duplicate backend entity identity')
            entity_by_id[entity.entity_id] = entity
            if request.scene is m.Scene.BOARD:
                continue
            try:
                c.require(entity.entity_id not in aliases, 'entity.identity',
                          'duplicate body outline requires physical identity association', c.ErrorCode.ID_CONFLICT)
                c.require(entity.entity_id not in ctx.correspondence_conflicts, 'correspondence',
                          'map/time/frame/lidar correspondence conflict; reobserve before use', c.ErrorCode.CONFLICT)
                height, basis = entity.height_mm, entity.plane_basis
                if entity.known_layer is not None:
                    plane = known_layer_plane(entity.known_layer, ctx.base_frame)
                    c.require(all(e == image.image for e in plane.evidence), 'known_layer.evidence',
                              'layer support must be verified in this exact image', c.ErrorCode.EVIDENCE_MISSING)
                    c.require(height is None or (type(height) in (int, float) and math.isfinite(height) and
                              abs(height - plane.height_mm) <= plane.error_mm), 'known_layer.height',
                              'independent plane conflicts with known layer', c.ErrorCode.CONFLICT)
                    known_planes[entity.entity_id] = plane
                    height, basis = plane.height_mm, plane.basis
                c.require(entity.kind == 'brick', 'entity.kind', 'board/shadow/unknown is not a physical brick')
                c.require(entity.body_confirmed is c.Fact.CONFIRMED and entity.member_ids == (entity.entity_id,),
                          'entity.body', 'independent single-body proof missing or adjacent bricks merged')
                c.require(bool(entity.evidence) and all(e == image.image for e in entity.evidence),
                          'entity.evidence', 'body/edge evidence must reference this exact new image')
                contour = batch.transform.restore(entity.contour).points
                edges = tuple(batch.transform.restore(edge).points for edge in entity.edges)
                surface = fit_top(image, contour, edges, self.config, basis, height, entity.evidence)
                c.require(all(contains(ctx.allowed_polygon, p) for p in surface.top.corners),
                          'region', 'top lies outside current scene region')
                c.require(not any(overlaps(poly, surface.top.corners) for poly in ctx.board_polygons),
                          'region', 'classification board excluded from physical candidates')
                fitted[entity.entity_id] = surface
                audit['surfaces'].append(dict(entity_id=entity.entity_id, **_plain(surface)))
            except c.ContractError as error:
                rejected[entity.entity_id] = _reason(error, 'entity')
                audit['rejections'].append(dict(entity_id=entity.entity_id, reason=_plain(rejected[entity.entity_id])))
        grouped = {}
        for pattern in batch.patterns:
            c.require(type(pattern.raw_class) is str and type(pattern.confidence) in (int, float) and
                      math.isfinite(pattern.confidence) and 0 <= pattern.confidence <= 1,
                      'backend.pattern', 'invalid class id or confidence')
            c.require(type(pattern.complete) is c.Fact, 'backend.pattern.complete', 'explicit completeness fact required')
            box = batch.transform.restore(pattern.box).points
            c.require(len(box) == 2 and box[0].u < box[1].u and box[0].v < box[1].v, 'box', 'positive axis-aligned pattern box required')
            category = self.config.category(pattern.raw_class) if pattern.confidence >= self.config.min_confidence else None
            record = dict(raw_class=pattern.raw_class, category=None if category is None else category.value,
                          confidence=pattern.confidence, entity_id=pattern.entity_id,
                          complete=pattern.complete.value,
                          full_box=[[p.u, p.v] for p in box], distortion=image.distortion.value)
            audit['patterns'].append(record)
            # Duplicate detections of one entity never allocate multiple physical objects.
            grouped.setdefault(pattern.entity_id, []).append((category, box))
        if request.scene is m.Scene.BOARD:
            return (), any(p['category'] is None for p in audit['patterns']), (), outline_groups
        candidates, unknown = [], False
        for entity_id, patterns in grouped.items():
            entity = entity_by_id.get(entity_id)
            if entity is not None and entity.kind == 'board':
                continue
            classes = {cat for cat, _ in patterns}
            category = next(iter(classes)) if len(classes) == 1 else None
            unknown = unknown or category is None
            top = fitted[entity_id].top if entity_id in fitted else None
            reasons = []
            if top is not None:
                for _, box in patterns:
                    polygon = (box[0], m.PixelPoint(box[1].u, box[0].v), box[1], m.PixelPoint(box[0].u, box[1].v))
                    matches = [key for key, surface in fitted.items() if
                               all(contains(surface.top.corners, p, -surface.top.error_px) for p in polygon)]
                    if matches != [entity_id] or not all(contains(top.corners, p, top.error_px) for p in polygon):
                        reasons.append('pattern not uniquely contained in the declared single body')
                    if any(overlaps(board, polygon) for board in ctx.board_polygons):
                        reasons.append('pattern overlaps excluded board region')
            elif entity_id not in rejected:
                reasons.append('no independent top/body evidence')
            if reasons:
                top = None
                audit['rejections'].append(dict(entity_id=entity_id, reason='; '.join(reasons)))
            observation = m.TargetObservation(image.observation_id + ':entity:' + str(entity_id), None, c.Fact.UNKNOWN,
                        category, top, top.visibility if top else m.Visibility.UNKNOWN, (image.image,))
            candidates.append((entity_id, observation))
        return tuple(candidates), unknown, tuple(known_planes.items()), outline_groups

    def observe(self, request: m.PerceptionRequest) -> m.PerceptionResult:
        image, started = request.image, time.monotonic()
        # In-flight/failed calls cannot expose the previous successful policy as current.
        self.policy_report = unavailable_report(image, image.envelope)
        candidate_report = None
        audit = dict(p04_version='0.1.0', contract_version=c.CONTRACT_VERSION, input=_plain(request),
                     patterns=[], surfaces=[], rejections=[], fresh_for_temporal_evidence=False,
                     published_current=False, stillness_certificate=False)
        phase, candidate_tracker, raw = 'image', None, None
        try:
            ctx = self._current(image)
            capture_context = ctx
            self.config.require_ready(ctx.consumer.origin)
            c.require((image.envelope.versions.config, image.envelope.versions.model) ==
                      (self.config.config_version, self.config.model_version), 'config.version',
                      'image references another configuration/model', c.ErrorCode.VERSION_MISMATCH)
            c.require(image.envelope.versions.calibration is not None, 'calibration', 'explicit camera/plane calibration version required', c.ErrorCode.UNCONFIGURED)
            c.require(len(ctx.allowed_polygon) >= 3, 'allowed_polygon', 'scene/tool region must be configured', c.ErrorCode.UNCONFIGURED)
            for polygon in (ctx.allowed_polygon,) + ctx.board_polygons:
                require_convex_polygon(polygon, image.width, image.height)
            scope = (image.envelope.run_id, image.envelope.instance_id, image.envelope.control_epoch)
            if self._scope is None:
                self._scope = scope
            c.require(self._scope == scope, 'service.scope', 'new controller/run requires new service; history needs revalidation', c.ErrorCode.STALE_CONTROL)
            raw = self.image_reader(image.image.uri)
            c.require(type(raw) is RawImage, 'image_reader', 'decoded RawImage required')
            self._gate.accept(image, raw, ctx.consumer, ctx.capture_floor, ctx.sensor_now)
            audit['quality'] = self._quality(raw)
            phase = 'backend'
            batch = self._batch(image, raw)
            audit['preprocessing'] = _plain(batch.transform)
            audit['model_version'], audit['class_map_version'] = batch.model_version, batch.class_map_version
            candidates, unknown, known_planes, outline_groups = self._candidates(request, batch, ctx, audit)
            candidate_tracker = copy.deepcopy(self._tracker)
            if request.scene is m.Scene.BOARD:
                targets, by_entity = (), {}
            else:
                targets, by_entity = candidate_tracker.update(candidates, image, request.scene, request.previous_targets)
            phase = 'image'
            ctx = self._current(image)
            self._same_alignment(capture_context, ctx)
            self._gate.validate(image, raw, ctx.consumer, ctx.capture_floor, ctx.sensor_now)
            envelope = self._envelope(image, ctx.consumer.now, ctx.sensor_now)
            tool = None
            if request.scene is m.Scene.TOOL:
                tool, audit['tool'] = observe_tool(batch, image, envelope, self.config, by_entity, ctx.allowed_polygon)
            state = m.ObservationState.UNKNOWN_CLASS if unknown else (m.ObservationState.DETECTED if batch.patterns or batch.tools else m.ObservationState.EMPTY)
            reason = c.Reason(c.ErrorCode.UNAVAILABLE, 'category', 'unknown, conflicting or low-confidence class') if unknown else None
            if audit['rejections'] and reason is None:
                reason = c.Reason(c.ErrorCode.EVIDENCE_MISSING, 'candidate', 'some entity geometry rejected; see P04 audit')
            result = m.PerceptionResult(envelope, state, targets, tool, reason)
            candidate_report = build_policy_report(image, envelope, batch, ctx, self.config,
                                                   by_entity, audit['rejections'], known_planes, outline_groups)
            audit['fresh_for_temporal_evidence'] = (not audit['rejections'] and not unknown and bool(targets) and
                all(t.top is not None and t.identity_confirmed is c.Fact.CONFIRMED and
                    t.visibility is m.Visibility.VISIBLE for t in targets))
            audit['published_current'] = True
        except Exception as error:
            candidate_tracker = None
            candidate_report = None
            # A stale result is historical only. Current failures replace success with explicit unknown.
            try:
                ctx = self._current(image)
                current = True
            except Exception:
                ctx, current = None, False
            reason = _reason(error, phase)
            state = m.ObservationState.BACKEND_FAILURE if (phase == 'backend' or isinstance(error, ImportError) or
                    reason.code is c.ErrorCode.UNCONFIGURED) else m.ObservationState.BAD_IMAGE
            now = ctx.consumer.now if ctx is not None else image.envelope.emitted_at
            envelope = self._envelope(image, now)
            tool = m.ToolObservation(envelope, None, (), None, c.Fact.UNKNOWN, ()) if request.scene is m.Scene.TOOL else None
            result = m.PerceptionResult(envelope, state, self._tracker.invisible() if current else (), tool, reason)
            audit['published_current'] = current
            if current:
                self._tracker.break_continuity()
        audit['result'] = _plain(result)
        self.policy_report = unavailable_report(image, result.envelope)
        audit['policy_report'] = _plain(candidate_report or self.policy_report)
        audit['elapsed_s'] = time.monotonic() - started
        audit['accepted_frames'] = self.accepted_frames
        audit['track_history'] = _plain([t.last for t in (candidate_tracker or self._tracker).tracks])
        self.last_audit = copy.deepcopy(audit)
        try:
            c.require(callable(self.audit_sink), 'audit_sink', 'result journal required', c.ErrorCode.STORAGE_FAILURE)
            self.audit_sink(copy.deepcopy(audit))
        except Exception as error:
            self._tracker.break_continuity()
            self.last_audit['published_current'] = False
            self.last_audit['fresh_for_temporal_evidence'] = False
            failed = m.PerceptionResult(result.envelope, m.ObservationState.BACKEND_FAILURE,
                self._tracker.invisible(), None,
                c.Reason(c.ErrorCode.STORAGE_FAILURE, 'audit_sink', 'result not published: ' + str(error)))
            self.last_audit['result'] = _plain(failed)
            self.policy_report = unavailable_report(image, failed.envelope)
            self.last_audit['policy_report'] = _plain(self.policy_report)
            return failed
        if audit['published_current']:
            try:
                ctx = self._current(image)
                if candidate_tracker is not None:
                    self._same_alignment(capture_context, ctx)
                    self._gate.validate(image, raw, ctx.consumer, ctx.capture_floor, ctx.sensor_now)
                    rules.validate_envelope(result.envelope, ctx.consumer)
            except Exception as error:
                self.last_audit['published_current'] = False
                self.last_audit['fresh_for_temporal_evidence'] = False
                late = m.PerceptionResult(result.envelope, m.ObservationState.BAD_IMAGE, (), None, _reason(error, 'publish'))
                self.last_audit['result'] = _plain(late)
                self.policy_report = unavailable_report(image, late.envelope)
                self.last_audit['policy_report'] = _plain(self.policy_report)
                try:
                    self.audit_sink(copy.deepcopy(self.last_audit))
                except Exception:
                    pass  # failure remains explicit and cannot publish; retained in last_audit
                return late
            if candidate_tracker is not None:
                self._tracker = candidate_tracker
            self.latest = result
            if candidate_report is not None:
                self.policy_report = candidate_report
        return result
