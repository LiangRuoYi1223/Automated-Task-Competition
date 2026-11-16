"""P05-owned configuration, with explicit unmeasured values and no field defaults."""
#定义板面模板、识别门槛，并检查配置
import hashlib
import json
import math
from dataclasses import dataclass, fields
from typing import Optional, Tuple
from treasure_auto.contracts import core as c, models as m

#数值检查
def number(value, name, minimum=0, maximum=None):
    c.require(type(value) in (int, float) and math.isfinite(value), name, 'finite numeric value required')
    c.require(value >= minimum and (maximum is None or value <= maximum), name, 'outside allowed range')

#字符串检查
def text(value, name):
    c.require(type(value) is str and bool(value.strip()), name, 'nonempty value required')


def readiness(measurement, evidence, origin):
    c.require(measurement is not m.Measurement.UNMEASURED, 'measurement', 'not measured', c.ErrorCode.UNCONFIGURED)
    if origin is c.Origin.LIVE:
        c.require(measurement is m.Measurement.VERIFIED and bool(evidence) and
                  all(e.origin is c.Origin.LIVE and e.kind is c.EvidenceKind.PHYSICAL for e in evidence),
                  'measurement', 'LIVE needs verified parameters and physical sources', c.ErrorCode.UNCONFIGURED)


@dataclass(frozen=True)
class SlotRegion:
    #描述B、C、D区域的矩形边界，左上角为(0,0)，右下角为(1,1)
    area: m.Area
    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self):
        c.require(self.area in (m.Area.B, m.Area.C, m.Area.D) and type(self.area) is m.Area,
                  'slot.area', 'only B/C/D belong to the board')
        for name in ('left','top','right','bottom'):
            number(getattr(self,name), 'slot.'+name, 0, 1)
        c.require(self.left < self.right and self.top < self.bottom, 'slot', 'positive rectangle required')


@dataclass(frozen=True)
class BoardTemplate:
    version: str
    orientation_key: str
    marker_keys: Tuple[str, ...]
    slots: Tuple[SlotRegion, ...]
    measurement: m.Measurement
    evidence: Tuple[c.EvidenceRef, ...] = ()

    def __post_init__(self):
        text(self.version,'template.version'); text(self.orientation_key,'template.orientation_key')
        c.require(type(self.measurement) is m.Measurement, 'measurement', 'C0 Measurement required')
        c.require(type(self.marker_keys) is tuple and bool(self.marker_keys) and
                  len(set(self.marker_keys)) == len(self.marker_keys), 'marker_keys', 'explicit unique sources required')
        for key in self.marker_keys: text(key,'marker_keys')
        c.require(type(self.slots) is tuple and len(self.slots) == 3 and
                  all(type(s) is SlotRegion for s in self.slots) and
                  {s.area for s in self.slots} == {m.Area.B,m.Area.C,m.Area.D},
                  'slots', 'exactly one B/C/D region required')
        for i,a in enumerate(self.slots):
            for b in self.slots[i+1:]:
                c.require(min(a.right,b.right) <= max(a.left,b.left) or
                          min(a.bottom,b.bottom) <= max(a.top,b.top), 'slots', 'overlapping slots are ambiguous')
        c.require(type(self.evidence) is tuple and all(type(e) is c.EvidenceRef for e in self.evidence),
                  'template.evidence', 'C0 evidence tuple required')

    def require_ready(self, origin):
        readiness(self.measurement,self.evidence,origin)


@dataclass(frozen=True)
class ClassificationConfig:
    version: str
    measurement: m.Measurement
    min_confirmations: Optional[int]
    min_confirmation_span_s: Optional[float]
    max_confirmation_gap_s: Optional[float]
    max_frame_age_s: Optional[float]
    min_board_area_px2: Optional[float]
    min_corner_sine: Optional[float]
    min_pattern_confidence: Optional[float]
    slot_margin: Optional[float]
    max_additional_views: Optional[int]
    observation_name: Optional[str]
    waypoint_version: Optional[str]
    evidence: Tuple[c.EvidenceRef, ...] = ()

    def __post_init__(self):
        text(self.version,'config.version')
        c.require(type(self.measurement) is m.Measurement,'measurement','C0 Measurement required')
        for f in fields(self):
            value=getattr(self,f.name)
            if f.name in ('version','measurement','evidence') or value is None: continue
            if f.name in ('observation_name','waypoint_version'):
                text(value,f.name); continue
            number(value,f.name)
            if f.name in ('min_confirmations','max_additional_views'):
                c.require(type(value) is int and value >= (2 if f.name == 'min_confirmations' else 0),f.name,'integer threshold required')
            elif f.name in ('min_corner_sine','min_pattern_confidence'):
                c.require(0 < value <= 1,f.name,'must be in (0,1]')
            elif f.name == 'slot_margin':
                c.require(value < 0.5,f.name,'normalized inward margin must be below 0.5')
            else:
                c.require(value > 0,f.name,'positive threshold required')
        c.require(type(self.evidence) is tuple and all(type(e) is c.EvidenceRef for e in self.evidence),
                  'config.evidence','C0 evidence tuple required')

    @classmethod
    def from_dict(cls, data):
        names={f.name for f in fields(cls)}
        c.require(type(data) is dict and not (set(data)-names) and
                  not (names-{'evidence'}-set(data)), 'config','explicit known fields required')
        values=dict(data)
        c.require(values['measurement'] in {v.value for v in m.Measurement},'measurement','unknown measurement')
        values['measurement']=m.Measurement(values['measurement'])
        values['evidence']=tuple(c.from_data(c.EvidenceRef,e) for e in values.get('evidence',()))
        return cls(**values)

    def require_ready(self, origin):
        for f in fields(self):
            c.require(getattr(self,f.name) is not None, f.name,'parameter not configured',c.ErrorCode.UNCONFIGURED)
        readiness(self.measurement,self.evidence,origin)


def fingerprint(config, template):
    """Bind the entire private configuration, not just caller-chosen version labels."""
    def encode(value):
        if hasattr(value,'__dataclass_fields__'):
            return {f.name:encode(getattr(value,f.name)) for f in fields(value)}
        if isinstance(value,tuple): return [encode(v) for v in value]
        if hasattr(value,'value'): return value.value
        return value
    payload=json.dumps([encode(config),encode(template)],sort_keys=True,separators=(',',':'),allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
