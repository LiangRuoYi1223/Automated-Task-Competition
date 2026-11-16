"""Local algorithm inputs, deliberately not C0 Records or wire contracts.

MetricFrame requires normalized, trusted monotonic exposure timestamps. A P04/P06
adapter must establish those; sensor-clock subtraction is never guessed here.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
from treasure_auto.contracts import core as c, models as m


@dataclass(frozen=True)
class MetricFrame:
    envelope: c.Envelope
    observation_id: str
    evidence: c.EvidenceRef
    objects: Tuple[m.Occupancy, ...]
    coverage_polygon_m: Tuple[m.Point2, ...]
    frame_id: str
    scene_epoch: int
    identity_complete: c.Fact
    cup_xyz_m: Optional[Tuple[float, float, float]]
    cup_error_m: float
    # (physical target ID, yaw rad, tilt rad); P04/P06 supply actual orientation.
    orientations: Tuple[Tuple[str, float, float], ...]


@dataclass(frozen=True)
class TargetProposal:
    slot: int
    desired_pose: c.MapPose
    occupancy: m.Occupancy


@dataclass(frozen=True)
class PlacementBinding:
    """P06/P02 adapter attestation binding planned brick geometry and held offset."""
    target: m.Occupancy
    held_offset: c.ArmPoint
    path_id: str
    base_pose: c.MapPose
    localization_evidence: Tuple[c.EvidenceRef, ...]


@dataclass(frozen=True)
class LayoutFacts:
    envelope: c.Envelope
    target_id: str
    stable: c.Fact
    horizontal: c.Fact
    released: c.Fact
    evidence: Tuple[c.EvidenceRef, ...]
    support: Optional[m.SupportGeometryResult] = None
    prior_facts: Tuple['LayoutFacts', ...] = ()


@dataclass(frozen=True)
class TemporaryPlacement:
    reservation: m.Occupancy
    plan: m.GeometryPlan
    binding: PlacementBinding
    region: m.RegionBoundary


@dataclass(frozen=True)
class RecoveryDecision:
    operation: str
    target_id: Optional[str]
    reason: Optional[c.Reason]
