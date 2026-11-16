"""Private P01 configuration; null values are unmeasured, never fallback speeds."""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from treasure_auto.contracts import CONTRACT_VERSION
from treasure_auto.contracts.models import Phase


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('{} must be finite and positive'.format(name))
    return value


@dataclass(frozen=True)
class ControlConfig:
    config_version: str
    max_linear_m_s: Optional[float]
    max_angular_rad_s: Optional[float]
    arm_frame_id: Optional[str]
    arm_bounds_mm: Optional[Tuple[Tuple[float, float], ...]]
    stage_limits: Tuple[Tuple[Phase, Optional[Tuple[float, float, float]]], ...]
    real_output_enabled: bool = False

    def __post_init__(self):
        if not isinstance(self.config_version, str) or not self.config_version.strip():
            raise ValueError('config_version required')
        if self.real_output_enabled is not False:
            raise ValueError('live outputs have not been verified')
        for name in ('max_linear_m_s', 'max_angular_rad_s'):
            if getattr(self, name) is not None:
                positive(getattr(self, name), name)
        if self.arm_frame_id is not None and (not isinstance(self.arm_frame_id, str) or not self.arm_frame_id.strip()):
            raise ValueError('arm_frame_id required')
        if self.arm_bounds_mm is not None:
            if type(self.arm_bounds_mm) is not tuple or len(self.arm_bounds_mm) != 3:
                raise ValueError('arm_bounds_mm requires three intervals')
            for bounds in self.arm_bounds_mm:
                if (type(bounds) is not tuple or len(bounds) != 2 or
                        any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds) or bounds[0] > bounds[1]):
                    raise ValueError('invalid arm bounds')
        if type(self.stage_limits) is not tuple or len(self.stage_limits) != len(Phase):
            raise ValueError('explicit stage limit or null required for every phase')
        if {phase for phase, _ in self.stage_limits} != set(Phase):
            raise ValueError('unknown or duplicate phase')
        for phase, limits in self.stage_limits:
            if type(phase) is not Phase:
                raise ValueError('invalid phase')
            if limits is not None:
                if type(limits) is not tuple or len(limits) != 3:
                    raise ValueError('stage requires minimum speed, maximum speed, timeout')
                for value in limits:
                    positive(value, phase.value)
                if limits[0] > limits[1]:
                    raise ValueError('speed range inverted')


def validate_control_config(data):
    keys = {'contract_version', 'config_version', 'real_output_enabled', 'arm_adapter',
            'max_linear_m_s', 'max_angular_rad_s', 'arm_frame_id', 'arm_bounds_mm', 'stage_limits'}
    if type(data) is not dict or set(data) != keys:
        raise ValueError('control configuration fields must match P01 schema')
    if data['contract_version'] != CONTRACT_VERSION:
        raise ValueError('C0 version mismatch')
    if data['arm_adapter'] != 'sdk_mm_min':
        raise ValueError('only stage-speed SDK conversion is implemented; fixed F10000 is forbidden')
    stages = data['stage_limits']
    if type(stages) is not dict or set(stages) != {p.value for p in Phase}:
        raise ValueError('stage_limits must explicitly list every phase')
    try:
        bounds = data['arm_bounds_mm']
        return ControlConfig(data['config_version'], data['max_linear_m_s'], data['max_angular_rad_s'],
                             data['arm_frame_id'], None if bounds is None else tuple(tuple(v) for v in bounds),
                             tuple((p, None if stages[p.value] is None else tuple(stages[p.value])) for p in Phase),
                             data['real_output_enabled'])
    except TypeError as error:
        raise ValueError('malformed control ranges') from error


def load_control_config(path):
    return validate_control_config(json.loads(Path(path).read_text(encoding='utf-8-sig')))
