"""P06-local algorithm/wiring inputs, NOT new or renamed C0 wire contracts.

Providers must derive these from independently validated P02/P04/P07/P08 facts.
Missing fields in C0 are listed in P06_interface_change.md; no provider is implied.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
from treasure_auto.contracts import core as c, models as m


@dataclass(frozen=True)
class PlaneSolution:
    solution_id: str
    height_mm: float
    error_mm: float
    normal: Tuple[float, float, float]
    basis: m.PlaneBasis
    support_ids: Tuple[str, ...]
    evidence: Tuple[c.EvidenceRef, ...]


@dataclass(frozen=True)
class PixelHistory:
    observation_id: str
    coordinate_order: str
    pixel_space: str
    preprocessing: m.PixelTransform
    roi_restorations: int
    input_distortion: m.Distortion
    distortion_passes: int
    inverse_error_gain: Optional[float]


@dataclass(frozen=True)
class ContactProof:
    upper_target_id: str
    lower_target_ids: Tuple[str, ...]
    space_version: int
    contact: c.Fact
    horizontal: c.Fact
    evidence: Tuple[c.EvidenceRef, ...]


@dataclass(frozen=True)
class GeometryInputs:
    consumer: m.ConsumerContext
    localization: m.LocalizationSnapshot
    stopped: c.Fact
    stop_evidence: Tuple[c.EvidenceRef, ...]
    image: Optional[m.ImageObservation]
    pixels: Optional[PixelHistory]
    capture_now: Optional[c.ClockStamp]
    capture_floor: Optional[c.ClockStamp]
    planes: Tuple[PlaneSolution, ...]
    camera_to_base: m.RigidTransform
    base_to_arm: m.RigidTransform
    tool_transform: m.RigidTransform
    holding: c.HoldingSnapshot
    holding_envelope: c.Envelope
    held_yaw_rad: Optional[float]
    start: c.ArmPoint
    start_evidence: Tuple[c.EvidenceRef, ...]
    arm_geometry_version: str
    space: m.SpaceSnapshot
    coverage: c.Fact
    coverage_evidence: Tuple[c.EvidenceRef, ...]
    image_report: object
    placement_report: object
    contact_proof: Optional[ContactProof]
    # Explicit local control assumption only; holding.tool_offset remains actual.
    approximate_center: bool = False


@dataclass(frozen=True)
class SweepBox:
    minimum_mm: Tuple[float, float, float]
    maximum_mm: Tuple[float, float, float]


@dataclass(frozen=True)
class MotionQuery:
    start: c.ArmPoint
    segment: m.PathSegment
    carrying: bool
    payload_size_mm: Tuple[float, float, float]
    held_offset: Optional[c.ArmPoint]
    error_mm: float
    tool_transform: m.RigidTransform
    geometry_version: str
    space_version: int


@dataclass(frozen=True)
class MotionCertificate:
    query: MotionQuery
    reachable: c.Fact
    full_arm_covered: c.Fact
    static_clearance: c.Fact
    swept_boxes: Tuple[SweepBox, ...]
    evidence: Tuple[c.EvidenceRef, ...]
    reason: Optional[c.Reason]
