"""Explicit simulation budgets. No guessed field timing defaults."""
import json
import math
from dataclasses import dataclass, fields
from treasure_auto.contracts import core as c


@dataclass(frozen=True)
class MissionConfig:
    version: str
    origin: str
    measurement: str
    navigation_s: float
    observation_s: float
    grasp_s: float
    placement_s: float
    recovery_s: float
    stop_margin_s: float
    recovery_attempts: int
    recovery_total_s: float
    observation_attempts: int
    evidence_age_s: float

    def __post_init__(self):
        c.require(type(self.version) is str and bool(self.version.strip()),'version','required')
        c.require(self.origin=='simulated' and self.measurement=='simulated','origin',
                  'formal timings and adapters have not been verified',c.ErrorCode.NOT_AUTHORIZED)
        for f in fields(self):
            v=getattr(self,f.name)
            if f.name.endswith('_s'):
                c.require(type(v) in (int,float) and math.isfinite(v) and v>0,f.name,
                          'explicit positive finite timing required',c.ErrorCode.UNCONFIGURED)
            elif f.name.endswith('_attempts'):
                c.require(type(v) is int and v>0,f.name,'positive integer required',c.ErrorCode.UNCONFIGURED)

    @classmethod
    def from_dict(cls,data):
        c.require(type(data) is dict and set(data)=={f.name for f in fields(cls)},'config','exact P09 fields required')
        return cls(**data)

    @classmethod
    def load(cls,path):
        with open(path,encoding='utf-8') as stream:return cls.from_dict(json.load(stream))

    @property
    def next_order_seconds(self):
        return 2*self.navigation_s+self.observation_s+self.grasp_s+self.placement_s+self.recovery_s+self.stop_margin_s

    def can_start(self,remaining):
        c.require(type(remaining) in (int,float) and math.isfinite(remaining) and remaining>=0,'remaining','invalid remaining time')
        return remaining>=self.next_order_seconds
