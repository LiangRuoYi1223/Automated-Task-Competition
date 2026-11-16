"""Fresh visible rim geometry; a same-object relation is NOT a holding/release fact."""
import math
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m
from .geometry import contains


def observe_tool(batch, image, envelope, config, targets, allowed_polygon):
    unknown = m.ToolObservation(envelope, None, (), None, c.Fact.UNKNOWN, ())
    try:
        c.require(len(batch.tools) == 1, 'tool.candidates', 'missing or ambiguous suction rim')
        tool = batch.tools[0]
        c.require(tool.source == 'visual_edge' and tool.visibility is m.Visibility.VISIBLE,
                  'tool.source', 'prediction/occlusion cannot provide observed rim')
        c.require(bool(tool.evidence) and all(e == image.image and e.kind is c.EvidenceKind.PHYSICAL
                                             for e in tool.evidence), 'tool.evidence', 'new image evidence required')
        rim = batch.transform.restore(tool.rim).points
        c.require(len(set(rim)) >= 12, 'tool.rim', 'insufficient rim support')
        c.require(all(contains(allowed_polygon, p) for p in rim), 'tool.rim', 'outside calibrated tool search region')
        import cv2
        import numpy as np
        xy = np.array([(p.u, p.v) for p in rim], dtype=np.float32)
        (u, v), (diameter_a, diameter_b), angle = cv2.fitEllipse(xy)
        a, b = diameter_a / 2., diameter_b / 2.
        c.require(config.tool_min_axis_px <= min(a, b) <= max(a, b) <= config.tool_max_axis_px,
                  'tool.axes', 'rim size outside measured range')
        center = m.PixelPoint(float(u), float(v))
        co, si = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        angles, errors = [], []
        for p in rim:
            x, y = (p.u - u) * co + (p.v - v) * si, -(p.u - u) * si + (p.v - v) * co
            radius = math.hypot(x / a, y / b)
            errors.append(abs(radius - 1) * max(a, b))
            angles.append(math.atan2(y / b, x / a) % (2 * math.pi))
        angles.sort()
        gap = max(y - x for x, y in zip(angles, angles[1:] + [angles[0] + 2 * math.pi]))
        c.require(gap <= math.pi / 2 and max(errors) <= config.tool_max_error_px,
                  'tool.rim', 'partial or inconsistent rim cannot establish geometric center')
        target = targets.get(tool.entity_id)
        possible_bodies = [entity_id for entity_id, item in targets.items() if item.top is not None and
                           all(contains(item.top.corners, p, -item.top.error_px) for p in rim)]
        relation = (target is not None and target.target_id is not None and
                    target.identity_confirmed is c.Fact.CONFIRMED and target.top is not None and
                    possible_bodies == [tool.entity_id] and
                    all(contains(target.top.corners, p, target.top.error_px) for p in rim))
        if envelope.target_id is not None:
            relation = relation and target.target_id == envelope.target_id
        target_id = target.target_id if relation else None
        evidence = (replace(image.image, evidence_id=image.image.evidence_id + ':tool-relation', target_id=target_id),)
        observed = m.ToolObservation(envelope, target_id, rim, center,
                                    c.Fact.CONFIRMED if relation else c.Fact.UNKNOWN, evidence)
        return observed, dict(visibility=tool.visibility.value, error_px=max(errors) + .5,
                              semi_axes_px=[a, b], source='visual_edge',
                              relation='same_visible_body' if relation else 'unknown',
                              holding_fact='owned_by_P07', capture_evidence_id=image.image.evidence_id)
    except (c.ContractError, ValueError, ImportError) as error:
        return unknown, dict(visibility='unknown', error_px=None, reason=str(error))
