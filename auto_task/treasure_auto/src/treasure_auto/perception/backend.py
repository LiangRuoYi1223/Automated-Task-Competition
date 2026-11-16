"""Private adapter input records, NOT new public C0 contracts or a recognition model.

An existing interface supplies actual image detections and independent body/edge
evidence. Annotated replay may supply the same inputs with REPLAY/SIMULATED origin.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
from treasure_auto.contracts import core as c, models as m
from treasure_auto.geometry.surface import KnownLayerEvidence
from .pixels import PixelSet, TransformPlan


@dataclass(frozen=True)
class Pattern:
    raw_class: str
    confidence: float
    entity_id: Optional[str]
    box: PixelSet
    complete: c.Fact = c.Fact.UNKNOWN


@dataclass(frozen=True)
class EntityEvidence:
    entity_id: str
    kind: str  # brick, board, shadow, unknown; never inferred from category alone
    body_confirmed: c.Fact
    member_ids: Tuple[str, ...]
    contour: PixelSet
    edges: Tuple[PixelSet, ...]
    plane_basis: m.PlaneBasis
    height_mm: Optional[float]
    evidence: Tuple[c.EvidenceRef, ...]
    suspicious_small_object: c.Fact = c.Fact.UNKNOWN
    local_contact_confirmed: c.Fact = c.Fact.UNKNOWN
    approach_clear: c.Fact = c.Fact.UNKNOWN
    known_layer: Optional[KnownLayerEvidence] = None


@dataclass(frozen=True)
class ToolEdges:
    rim: PixelSet
    entity_id: Optional[str]
    visibility: m.Visibility
    evidence: Tuple[c.EvidenceRef, ...]
    source: str


@dataclass(frozen=True)
class BackendBatch:
    observation_id: str
    image_sha256: str
    origin: c.Origin
    model_version: str
    class_map_version: str
    transform: TransformPlan
    patterns: Tuple[Pattern, ...]
    entities: Tuple[EntityEvidence, ...]
    tools: Tuple[ToolEdges, ...]


class ExistingInterfaceBackend:
    """No auto-download, model discovery, training, ROS import or generic class aliasing."""
    def __init__(self, infer, model_version, class_map_version):
        c.require(callable(infer), 'infer', 'existing inference callable required', c.ErrorCode.UNAVAILABLE)
        self._infer = infer
        self.model_version, self.class_map_version = model_version, class_map_version

    def infer(self, image, raw):
        batch = self._infer(image, raw)
        c.require(type(batch) is BackendBatch, 'backend.output', 'malformed backend output')
        c.require((batch.model_version, batch.class_map_version) == (self.model_version, self.class_map_version),
                  'backend.version', 'backend changed model/class map', c.ErrorCode.VERSION_MISMATCH)
        for name, expected in (('patterns', Pattern), ('entities', EntityEvidence), ('tools', ToolEdges)):
            values = getattr(batch, name)
            c.require(type(values) is tuple and all(type(v) is expected for v in values),
                      'backend.' + name, 'typed immutable output tuple required')
        return batch

