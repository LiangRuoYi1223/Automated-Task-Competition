"""P04-owned parameters; null means unmeasured, never a permissive default."""
import math
from dataclasses import dataclass, fields
from typing import Optional, Tuple
from treasure_auto.contracts.core import (Origin, ErrorCode, EvidenceKind, EvidenceRef,
                                         from_data, require)
from treasure_auto.contracts.models import Category, Measurement


@dataclass(frozen=True)
class PerceptionConfig:
    config_version: str
    measurement: str
    model_version: Optional[str]
    class_map_version: Optional[str]
    classes: Tuple[Tuple[str, str], ...]
    frame_size: Optional[Tuple[int, int]]
    max_age_s: Optional[float]
    min_confidence: Optional[float]
    min_edge_points: Optional[int]
    min_edge_span_px: Optional[float]
    max_line_error_px: Optional[float]
    max_extrapolation_px: Optional[float]
    min_area_px2: Optional[float]
    min_corner_sine: Optional[float]
    association_distance_px: Optional[float]
    min_confirmations: Optional[int]
    tool_min_axis_px: Optional[float]
    tool_max_axis_px: Optional[float]
    tool_max_error_px: Optional[float]
    min_contrast: Optional[float]
    max_saturation_fraction: Optional[float]
    min_sharpness: Optional[float]
    parameter_evidence: Tuple[EvidenceRef, ...] = ()

    def __post_init__(self):
        require(type(self.config_version) is str and bool(self.config_version.strip()), 'config_version', 'required')
        require(self.measurement in {v.value for v in Measurement}, 'measurement', 'unknown measurement status')
        for name in ('model_version', 'class_map_version'):
            value = getattr(self, name)
            require(value is None or (type(value) is str and bool(value.strip())), name, 'version or null required')
        require(type(self.classes) is tuple, 'classes', 'immutable pairs required')
        keys = []
        for key, value in self.classes:
            require(type(key) is str and bool(key.strip()) and value in {c.value for c in Category},
                    'classes', 'explicit competition class mapping required; no generic aliases')
            keys.append(key)
        require(len(keys) == len(set(keys)), 'classes', 'duplicate class id')
        if self.frame_size is not None:
            require(type(self.frame_size) is tuple and len(self.frame_size) == 2 and
                    all(type(n) is int and n > 0 for n in self.frame_size), 'frame_size', 'positive width/height')
        ignored = {'config_version', 'measurement', 'model_version', 'class_map_version',
                   'classes', 'frame_size', 'parameter_evidence'}
        for f in fields(self):
            if f.name in ignored:
                continue
            value = getattr(self, f.name)
            if value is None:
                continue
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    f.name, 'finite nonnegative number required, not bool')
            if f.name in {'max_age_s', 'min_edge_span_px', 'min_area_px2', 'min_corner_sine',
                          'association_distance_px', 'tool_min_axis_px', 'tool_max_axis_px'}:
                require(value > 0, f.name, 'must be positive')
            if f.name in {'min_confidence', 'min_corner_sine', 'max_saturation_fraction'}:
                require(value <= 1, f.name, 'must be at most 1')
            if f.name in {'min_edge_points', 'min_confirmations'}:
                require(type(value) is int and value >= (3 if f.name == 'min_edge_points' else 2),
                        f.name, 'integer support/confirmation threshold required')
        if self.tool_min_axis_px is not None and self.tool_max_axis_px is not None:
            require(self.tool_min_axis_px < self.tool_max_axis_px, 'tool_max_axis_px', 'inverted tool size interval')
        require(type(self.parameter_evidence) is tuple and all(type(e) is EvidenceRef for e in self.parameter_evidence),
                'parameter_evidence', 'C0 evidence tuple required')

    @classmethod
    def from_dict(cls, data):
        names = {f.name for f in fields(cls)}
        require(type(data) is dict, 'config', 'object required')
        require(not (set(data) - names), 'config', 'unknown configuration keys')
        require(not (names - {'parameter_evidence'} - set(data)), 'config', 'explicit fields including null required')
        values = dict(data)
        require(type(values['classes']) is dict, 'classes', 'id-to-name map required')
        values['classes'] = tuple(values['classes'].items())
        if values['frame_size'] is not None:
            require(type(values['frame_size']) is list, 'frame_size', 'array or null required')
            values['frame_size'] = tuple(values['frame_size'])
        values['parameter_evidence'] = tuple(from_data(EvidenceRef, e) for e in values.get('parameter_evidence', []))
        return cls(**values)

    def require_ready(self, origin):
        for f in fields(self):
            require(getattr(self, f.name) is not None, f.name, 'parameter not measured/configured', ErrorCode.UNCONFIGURED)
        require(bool(self.classes), 'classes', 'no existing competition model/class map', ErrorCode.UNCONFIGURED)
        require(self.measurement != Measurement.UNMEASURED.value, 'measurement', 'unmeasured configuration', ErrorCode.UNCONFIGURED)
        if origin is Origin.LIVE:
            require(self.measurement == Measurement.VERIFIED.value and bool(self.parameter_evidence),
                    'measurement', 'LIVE needs measured parameters and supporting evidence', ErrorCode.UNCONFIGURED)
            require(all(e.origin is Origin.LIVE and e.kind is EvidenceKind.PHYSICAL for e in self.parameter_evidence),
                    'parameter_evidence', 'simulated or echoed parameters cannot enable LIVE', ErrorCode.UNCONFIGURED)

    def category(self, raw_id):
        value = dict(self.classes).get(raw_id)
        return None if value is None else Category(value)

