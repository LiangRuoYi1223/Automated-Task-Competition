"""Private offline adapter inputs and diagnostics, NOT replacements for C0 records.

P00 must freeze these missing semantics before a production P04/P03 adapter exists.
All coordinates are corrected full-image u/v; corners are never sorted by raw u.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
from treasure_auto.contracts import core as c, models as m


@dataclass(frozen=True)
class DirectionHint:
    marker_key: str  # verified family/id/mount version, or verified border landmark
    corner_order: Tuple[int, int, int, int]  # template TL, TR, BR, BL in candidate corners
    evidence: Tuple[c.EvidenceRef, ...]


@dataclass(frozen=True)
class BoardCandidate:
    corners: Tuple[m.PixelPoint, m.PixelPoint, m.PixelPoint, m.PixelPoint]
    complete: c.Fact
    directions: Tuple[DirectionHint, ...]


@dataclass(frozen=True)
class PatternInput:
    category: Optional[m.Category]
    confidence: float
    polygon: Tuple[m.PixelPoint, ...]
    evidence: Tuple[c.EvidenceRef, ...]


@dataclass(frozen=True)
class FrameInput:
    image: m.ImageObservation
    acquisition_id: str
    sensor_now: c.ClockStamp
    capture_not_before: c.ClockStamp
    segment_id: str
    waypoint_version: str
    stationary: c.Fact
    stationary_evidence: Tuple[c.EvidenceRef, ...]
    boards: Tuple[BoardCandidate, ...]
    patterns: Tuple[PatternInput, ...]


@dataclass(frozen=True)
class FrameAnalysis:
    valid: bool
    assignments: Tuple[m.SlotAssignment, ...]
    reason: Optional[c.Reason]
    conflict: bool
    raw_corners: Tuple[m.PixelPoint, ...]
    homography: Tuple[float, ...]
    rectified_patterns: Tuple[Tuple[m.PixelPoint, ...], ...]
    direction_sources: Tuple[str, ...]


def frame_evidence(frame):
    refs=[frame.image.image]+list(frame.stationary_evidence)
    for board in frame.boards:
        for direction in board.directions: refs.extend(direction.evidence)
    for pattern in frame.patterns: refs.extend(pattern.evidence)
    by_id={}
    for ref in refs:
        c.require(type(ref) is c.EvidenceRef,'evidence','C0 evidence required')
        c.require(ref.evidence_id not in by_id or by_id[ref.evidence_id] == ref,
                  'evidence_id','same id has different evidence',c.ErrorCode.ID_CONFLICT)
        by_id[ref.evidence_id]=ref
    return tuple(by_id.values())
