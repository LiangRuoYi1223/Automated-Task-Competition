"""Explicit simulation settings. No measured robot defaults."""
import math
from dataclasses import dataclass, fields
from treasure_auto.contracts import core as c


def finite(value, field, minimum=0., positive=False):
    c.require(type(value) in (int,float) and math.isfinite(value),field,'finite configured number required',c.ErrorCode.UNCONFIGURED)
    c.require(value>minimum if positive else value>=minimum,field,'outside allowed range')
    return value


@dataclass(frozen=True)
class PlacementConfig:
    version: str
    origin: str
    measurement: str
    center_tolerance_m: float
    adjacent_gap_m: float
    height_tolerance_m: float
    ground_z_m: float
    evidence_max_age_s: float
    frame_gap_s: float
    release_window_s: float
    brick_motion_m: float
    cup_retreat_m: float
    still_translation_m: float
    still_angle_rad: float
    certificate_ttl_s: float

    def __post_init__(self):
        c.require(type(self.version) is str and bool(self.version.strip()),'version','required')
        c.require(self.origin=='simulated' and self.measurement=='simulated','origin',
                  'only explicit simulation is implemented; field adapters are not commissioned',c.ErrorCode.NOT_AUTHORIZED)
        for f in fields(self):
            if f.name not in ('version','origin','measurement'):
                finite(getattr(self,f.name),f.name,positive=f.name!='ground_z_m')
        c.require(self.frame_gap_s<=self.evidence_max_age_s,'frame_gap_s','frame gap exceeds freshness limit')

    @classmethod
    def from_dict(cls,data):
        expected={f.name for f in fields(cls)}
        c.require(type(data) is dict and set(data)==expected,'config','missing or unknown field',c.ErrorCode.UNCONFIGURED)
        return cls(**data)
