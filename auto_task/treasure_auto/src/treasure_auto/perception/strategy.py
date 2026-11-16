"""Frame-bound field-policy facts, separate from C0 wire objects and inventory.

These records never authorize motion. An exploratory center is a pattern center,
not a fabricated top corner or an arm target. Frame-local entity IDs must not be
used as physical identity or as keys that refund manipulation attempt budgets.
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from treasure_auto.contracts import core as c, models as m
from treasure_auto.geometry.inputs import PlaneSolution
from .geometry import contains, distance, segment_distance


@dataclass(frozen=True)
class ExploratoryCandidate:
    entity_id: str
    observation_id: str
    target_id: Optional[str]
    category: Optional[m.Category]
    center: m.PixelPoint
    contact_height_mm: Optional[float]
    eligible: bool
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class TargetReview:
    entity_id: Optional[str]
    observation_id: str
    target_id: Optional[str]
    reasons: Tuple[str, ...]


@dataclass(frozen=True)
class PerceptionPolicyReport:
    envelope: c.Envelope
    observation_id: str
    image_sha256: str
    camera_frame: str
    capture_pose: c.MapPose
    current: bool
    pattern_count: int
    independent_count: int
    geometry_count: int
    exploratory_candidates: Tuple[ExploratoryCandidate, ...]
    review_targets: Tuple[TargetReview, ...]
    known_planes: Tuple[Tuple[str, PlaneSolution], ...] = ()


def unavailable_report(image, envelope):
    return PerceptionPolicyReport(envelope, image.observation_id, image.image.sha256,
                                  image.camera_frame, image.capture_pose, False, 0, 0, 0, (), ())


def _same_detection(a, b):
    """Box overlap is only for deduplicating pattern detections, never real edges."""
    width = max(0., min(a[1].u, b[1].u) - max(a[0].u, b[0].u))
    height = max(0., min(a[1].v, b[1].v) - max(a[0].v, b[0].v))
    intersection = width * height
    area_a = (a[1].u - a[0].u) * (a[1].v - a[0].v)
    area_b = (b[1].u - b[0].u) * (b[1].v - b[0].v)
    return intersection / (area_a + area_b - intersection) >= .5


def _groups(records):
    # Connected duplicate detections form one group, independent of input order.
    groups = []
    for record in records:
        linked = [group for group in groups if any(_same_detection(record[2], other[2]) for other in group)]
        merged = [record]
        for group in linked:
            merged.extend(group)
            groups.remove(group)
        groups.append(merged)
    return groups


def _outline_center(contour):
    """Area centroid of the observed outline, independent of edge resampling."""
    pairs = tuple(zip(contour, contour[1:] + contour[:1]))
    cross = tuple(a.u * b.v - b.u * a.v for a, b in pairs)
    area2 = sum(cross)
    if abs(area2) < 1e-9:
        return None
    return m.PixelPoint(sum((a.u + b.u) * w for (a, b), w in zip(pairs, cross)) / (3 * area2),
                        sum((a.v + b.v) * w for (a, b), w in zip(pairs, cross)) / (3 * area2))


def _same_outline(a, b, tolerance):
    """Detect ambiguous aliases from whole boundaries, not shared adjacent edges.

    This comparison only withholds duplicate identities; it never supplies top
    corners. Tolerance is the configured measured edge error, not an invented
    object size. Comparing vertices to segments is independent of sampling rate.
    """
    if len(a) < 3 or len(b) < 3:
        return False
    bounds = lambda points: (min(p.u for p in points), min(p.v for p in points),
                             max(p.u for p in points), max(p.v for p in points))
    if any(abs(x - y) > tolerance for x, y in zip(bounds(a), bounds(b))):
        return False
    center_a, center_b = _outline_center(a), _outline_center(b)
    # Require actual interior overlap, so thin neighbors sharing an edge cannot
    # merge merely because their widths are smaller than the measured error.
    if (center_a is None or center_b is None or
            not contains(b, center_a, 1e-9) or not contains(a, center_b, 1e-9)):
        return False
    for source, destination in ((a, b), (b, a)):
        edges = tuple(zip(destination, destination[1:] + destination[:1]))
        if any(min(segment_distance(point, x, y) for x, y in edges) > tolerance + 1e-9 for point in source):
            return False
    return True


def entity_outline_groups(batch, image, config):
    """Frame-local ambiguity sets, never new persistent physical identities."""
    groups = []
    for entity in batch.entities:
        if (entity.kind not in ('brick', 'unknown') or not entity.evidence or
                not all(e == image.image for e in entity.evidence)):
            continue
        contour = batch.transform.restore(entity.contour).points
        # Consecutive edge sample sets often repeat their shared endpoint. Those
        # zero-length segments must not make an interior-overlap check fail.
        contour = tuple(point for i, point in enumerate(contour)
                        if i == 0 or distance(point, contour[i - 1]) > 1e-9)
        if len(contour) > 1 and distance(contour[0], contour[-1]) <= 1e-9:
            contour = contour[:-1]
        record = (entity.entity_id, contour)
        linked = [group for group in groups if any(
            _same_outline(contour, other[1], config.max_line_error_px) for other in group)]
        merged = [record]
        for group in linked:
            merged.extend(group)
            groups.remove(group)
        groups.append(merged)
    return tuple(tuple(sorted(record[0] for record in group)) for group in groups)


def build_policy_report(image, envelope, batch, ctx, config, by_entity, rejections, known_planes, outline_groups):
    """Derive counts only from this batch; tracked occluded targets are excluded."""
    reviews = {}

    def review(entity_id, reason):
        reviews.setdefault(entity_id, []).append(reason)

    for rejected in rejections:
        reason = rejected['reason']
        review(rejected.get('entity_id'), reason.get('detail', str(reason)) if isinstance(reason, dict) else reason)
    records = []
    for pattern in batch.patterns:
        category = config.category(pattern.raw_class) if pattern.confidence >= config.min_confidence else None
        box = batch.transform.restore(pattern.box).points
        records.append((pattern, category, box))
        observation = by_entity.get(pattern.entity_id)
        ordinary = observation is not None and observation.top is not None and observation.category is not None
        if pattern.complete is c.Fact.REJECTED or (pattern.complete is c.Fact.UNKNOWN and not ordinary):
            review(pattern.entity_id, 'pattern completeness unconfirmed')
        if category is None:
            review(pattern.entity_id, 'pattern class unresolved')
    count = 0
    for group in _groups(records):
        complete = [record for record in group if record[0].complete is c.Fact.CONFIRMED]
        categories = {record[1] for record in complete}
        if complete and len(categories) == 1 and None not in categories:
            count += 1
        if len(categories) > 1:
            for pattern, _, _ in group:
                review(pattern.entity_id, 'duplicate pattern classifications conflict')

    independent, geometry, exploratory, body_outlines = 0, 0, [], set()
    planes = dict(known_planes)
    outline_by_id = {entity_id: group for group in outline_groups for entity_id in group}
    for entity in batch.entities:
        if entity.kind not in ('brick', 'unknown'):
            continue
        evidence_current = bool(entity.evidence) and all(e == image.image for e in entity.evidence)
        contour = batch.transform.restore(entity.contour).points
        in_region = bool(contour) and all(contains(ctx.allowed_polygon, point) for point in contour)
        body = (entity.kind == 'brick' and entity.body_confirmed is c.Fact.CONFIRMED and
                entity.member_ids == (entity.entity_id,) and evidence_current and in_region)
        outline_key = outline_by_id.get(entity.entity_id, (entity.entity_id,))
        ambiguous = len(outline_key) > 1
        if body and outline_key not in body_outlines:
            independent += 1
            body_outlines.add(outline_key)
        if ambiguous:
            review(entity.entity_id, 'duplicate body outline requires physical identity association')
        observation = by_entity.get(entity.entity_id)
        usable = (body and not ambiguous and observation is not None and
                  observation.top is not None and observation.category is not None)
        geometry += int(usable)
        if usable or not (body or entity.suspicious_small_object is c.Fact.CONFIRMED):
            continue
        patterns = [record for record in records if record[0].entity_id == entity.entity_id]
        complete = [group for group in _groups(patterns) if
                    any(record[0].complete is c.Fact.CONFIRMED and record[1] is not None for record in group)]
        if not complete:
            review(entity.entity_id, 'no complete identifiable pattern for exploratory contact')
            continue
        selected = next(record for record in complete[0] if record[0].complete is c.Fact.CONFIRMED and record[1] is not None)
        _, category, box = selected
        center = m.PixelPoint((box[0].u + box[1].u) / 2., (box[0].v + box[1].v) / 2.)
        plane = planes.get(entity.entity_id)
        height = plane.height_mm if plane is not None else entity.height_mm
        basis = plane.basis if plane is not None else entity.plane_basis
        reasons = []
        if ambiguous:
            reasons.append('duplicate body outline requires physical identity association')
        if not evidence_current:
            reasons.append('small-object/contact evidence does not match current image')
        if not in_region or len(contour) < 3 or not contains(contour, center):
            reasons.append('pattern contact is outside the observed local object/scene')
        if len(complete) != 1 or len({record[1] for record in patterns}) != 1:
            reasons.append('exploratory pattern association is ambiguous')
        if entity.member_ids != (entity.entity_id,):
            reasons.append('multiple or unresolved physical members')
        if entity.entity_id in ctx.correspondence_conflicts:
            reasons.append('map/time/frame/lidar correspondence requires review')
        if entity.known_layer is not None and plane is None:
            reasons.append('known layer support or identity requires review')
        if entity.local_contact_confirmed is not c.Fact.CONFIRMED:
            reasons.append('local contact surface unconfirmed')
        if entity.approach_clear is not c.Fact.CONFIRMED:
            reasons.append('bounded approach path unconfirmed')
        if (type(height) not in (int, float) or not math.isfinite(height) or
                type(basis) is not m.PlaneBasis or basis is m.PlaneBasis.UNKNOWN):
            reasons.append('contact height basis unavailable')
            height = None
        exploratory.append(ExploratoryCandidate(entity.entity_id, image.observation_id,
            observation.target_id if observation is not None and not ambiguous else None,
            category, center, height, not reasons, tuple(reasons)))
        review(entity.entity_id, 'exploratory verification required; not an ordinary grasp target')
        for reason in reasons:
            review(entity.entity_id, reason)
    # Keep every alias in the review list but expose a single blocked candidate
    # per ambiguous outline. No alias can obtain a separate attempt budget.
    unique_exploratory = {}
    for candidate in sorted(exploratory, key=lambda item: item.entity_id):
        group = outline_by_id.get(candidate.entity_id, (candidate.entity_id,))
        unique_exploratory.setdefault(group, candidate)
    review_targets = tuple(TargetReview(entity_id, image.observation_id,
        by_entity[entity_id].target_id if entity_id in by_entity else None,
        tuple(dict.fromkeys(reasons))) for entity_id, reasons in reviews.items())
    return PerceptionPolicyReport(envelope, image.observation_id, image.image.sha256, image.camera_frame,
        image.capture_pose, True, count, independent, geometry, tuple(unique_exploratory.values()), review_targets, tuple(known_planes))
