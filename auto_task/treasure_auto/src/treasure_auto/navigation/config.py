"""Private P03 configuration. All numeric examples belong in simulated tests."""
import json
import math
from dataclasses import dataclass, fields
from treasure_auto.contracts import core as c, models as m
from treasure_auto.mapping.geometry import validate_polygon


def number(value, name, positive=True):
    c.require(type(value) in (int, float) and math.isfinite(value) and
              (value > 0 if positive else value >= 0), name, 'finite configured number required', c.ErrorCode.UNCONFIGURED)


@dataclass(frozen=True)
class DockPolicy:
    purpose: str
    forward_tolerance_m: float
    lateral_tolerance_m: float
    yaw_tolerance_rad: float
    max_correction_m: float

    def validate(self):
        c.require(type(self.purpose) is str and bool(self.purpose.strip()),'purpose','explicit purpose required')
        for f in fields(self):
            if f.name != 'purpose':number(getattr(self,f.name),f.name)


@dataclass(frozen=True)
class EnvelopeProfile:
    body: tuple
    arm: tuple
    carried: object
    error_m: float
    measurement: m.Measurement

    def shapes(self, holding):
        c.require(holding in (c.Holding.EMPTY,c.Holding.HELD),'holding','unknown/dropped is not empty',c.ErrorCode.UNAVAILABLE)
        c.require(self.measurement is m.Measurement.SIMULATED,'profile','only simulated envelope delivered',c.ErrorCode.UNCONFIGURED)
        number(self.error_m,'profile.error_m',False)
        shapes=(self.body,self.arm)
        if holding is c.Holding.HELD:shapes+=(self.carried,)
        for shape in shapes:validate_polygon(shape,'complete_envelope')
        return shapes


@dataclass(frozen=True)
class NavigationPolicy:
    config_version: str
    measurement: m.Measurement
    max_age_s: float
    max_gap_s: float
    stop_window_s: float
    stop_linear_m_s: float
    stop_angular_rad_s: float
    coarse_timeout_s: float
    handoff_timeout_s: float
    precision_timeout_s: float
    retry_limit: int
    recovery_seconds: float
    progress_m: float
    progress_rad: float
    max_linear_m_s: float
    max_angular_rad_s: float
    max_linear_accel_m_s2: float
    max_angular_accel_rad_s2: float
    docks: tuple

    def validate(self):
        c.require(self.measurement is m.Measurement.SIMULATED,'measurement',
                  'unmeasured/live parameters cannot enable this offline implementation',c.ErrorCode.UNCONFIGURED)
        c.require(type(self.config_version) is str and bool(self.config_version.strip()),'config_version','required')
        for f in fields(self):
            if f.name not in ('config_version','measurement','docks','retry_limit'):number(getattr(self,f.name),f.name)
        c.require(type(self.retry_limit) is int and self.retry_limit>=0,'retry_limit','nonnegative integer required')
        c.require(type(self.docks) is tuple and bool(self.docks) and all(type(p) is DockPolicy for p in self.docks),
                  'docks','purpose-specific measured windows required',c.ErrorCode.UNCONFIGURED)
        c.require(len({p.purpose for p in self.docks})==len(self.docks),'docks','duplicate purpose')
        for p in self.docks:p.validate()

    def dock(self,purpose):
        self.validate()
        value=next((p for p in self.docks if p.purpose==purpose),None)
        c.require(value is not None,'purpose','no configured docking window',c.ErrorCode.UNCONFIGURED)
        return value


def load_policy(path):
    def pairs(values):
        data={}
        for key,value in values:
            c.require(key not in data,key,'duplicate configuration key')
            data[key]=value
        return data
    with open(path,encoding='utf-8') as stream:data=json.load(stream,object_pairs_hook=pairs)
    c.require(set(data)=={f.name for f in fields(NavigationPolicy)},'config','missing/unknown configuration field')
    data['measurement']=m.Measurement(data['measurement'])
    data['docks']=tuple(DockPolicy(**p) for p in data['docks'])
    policy=NavigationPolicy(**data);policy.validate()
    return policy
