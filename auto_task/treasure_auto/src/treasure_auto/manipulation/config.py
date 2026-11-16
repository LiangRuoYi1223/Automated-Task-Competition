"""Explicit offline parameters. None is unmeasured, never a default."""
import json
import math
from dataclasses import dataclass, fields
from treasure_auto.contracts import core as c


@dataclass(frozen=True)
class ManipulationConfig:
    config_version: str
    evidence_age_s: float
    confirmation_frames: int
    confirmation_span_s: float
    max_frame_gap_s: float
    pose_tolerance_mm: float
    relative_tolerance_mm: float
    min_lift_mm: float
    max_lift_mm: float
    transport_pose_id: str
    pump_timeout_s: float
    result_ttl_s: float

    def __post_init__(self):
        for f in fields(self):
            value=getattr(self,f.name)
            if f.name in ('config_version','transport_pose_id'):
                c.require(type(value) is str and bool(value.strip()),f.name,'explicit identity required',c.ErrorCode.UNCONFIGURED)
            elif f.name=='confirmation_frames':
                c.require(type(value) is int and value>=2,f.name,'at least two independent exposures required',c.ErrorCode.UNCONFIGURED)
            else:
                c.require(type(value) in (int,float) and math.isfinite(value) and value>0,
                          f.name,'explicit finite positive measured/test value required',c.ErrorCode.UNCONFIGURED)
        c.require(self.min_lift_mm<self.max_lift_mm,'min_lift_mm','invalid short-lift interval')
        c.require(self.confirmation_span_s<=self.evidence_age_s and self.max_frame_gap_s<=self.evidence_age_s,
                  'confirmation_span_s','confirmation cannot outlive evidence')

    @classmethod
    def load(cls,path):
        def pairs(values):
            result={}
            for key,value in values:
                c.require(key not in result,key,'duplicate config key')
                result[key]=value
            return result
        with open(path,encoding='utf-8') as stream:
            data=json.load(stream,object_pairs_hook=pairs)
        c.require(set(data)=={f.name for f in fields(cls)},'config','exact fields required')
        return cls(**data)
