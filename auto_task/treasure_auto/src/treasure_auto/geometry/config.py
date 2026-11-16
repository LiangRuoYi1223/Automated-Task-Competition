"""Strict local configuration. Null means unmeasured, never zero or a default layer."""
from dataclasses import dataclass, fields
from treasure_auto.contracts import core as c, models as m
from .math3d import finite


@dataclass(frozen=True)
class GeometryConfig:
    version: str
    measurement: str
    map_frame: str
    base_frame: str
    arm_frame: str
    image_size: tuple
    approach_mm: float
    descend_mm: float
    lift_mm: float
    withdraw_mm: float
    speeds_mm_s: tuple
    timeouts_s: tuple
    cup_radius_mm: float
    tool_error_mm: float
    brick_size_mm: tuple
    max_error_mm: float
    plane_tolerance_mm: float
    max_pose_age_s: float
    fixed_tool_yaw_rad: float
    yaw_tolerance_rad: float
    max_contact_gap_mm: float
    max_pixel_error_px: float

    @classmethod
    def from_dict(cls,data):
        names={f.name for f in fields(cls)}
        c.require(type(data) is dict and set(data)==names,'config','exact field set required')
        d=dict(data)
        for key in ('version','map_frame','base_frame','arm_frame'):
            c.require(type(d[key]) is str and bool(d[key].strip()),key,'explicit string required')
        c.require(d['measurement'] in tuple(x.value for x in m.Measurement),'measurement','unknown measurement state')
        for key,n in (('image_size',2),('speeds_mm_s',5),('timeouts_s',5),('brick_size_mm',3)):
            if d[key] is not None:
                c.require(type(d[key]) in (tuple,list) and len(d[key])==n,key,'wrong array length')
                d[key]=tuple(d[key])
                for value in d[key]:
                    finite(value,key,0.);c.require(value>0,key,'positive values required')
                    if key=='image_size':c.require(type(value) is int,key,'integer dimensions required')
        for key in names-{'version','measurement','map_frame','base_frame','arm_frame','image_size','speeds_mm_s','timeouts_s','brick_size_mm'}:
            if d[key] is not None:finite(d[key],key,None if key=='fixed_tool_yaw_rad' else 0.)
        return cls(**d)

    def require_ready(self,origin):
        # Also validate callers using the constructor/dataclasses.replace directly.
        type(self).from_dict({f.name:getattr(self,f.name) for f in fields(self)})
        c.require(all(getattr(self,f.name) is not None for f in fields(self)) and self.measurement!='unmeasured',
                  'config','geometry parameters unmeasured',c.ErrorCode.UNCONFIGURED)
        c.require(origin is not c.Origin.LIVE or self.measurement=='verified','config.measurement',
                  'simulation cannot enable live geometry',c.ErrorCode.SIMULATED_EVIDENCE)
        c.require(self.approach_mm>self.descend_mm>0 and self.lift_mm>0 and self.withdraw_mm>0 and
                  self.cup_radius_mm>0 and self.max_pose_age_s>0,'config','positive ordered approach/descend and timing required')
        c.require(self.brick_size_mm[0]==self.brick_size_mm[1],'brick_size_mm','90 degree equivalence requires a square top')
