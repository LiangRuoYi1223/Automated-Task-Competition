"""P07-local numeric inputs; NOT C0 messages or delivered perception adapters.

brick denotes the same object's measured bottom centre, cup its measured contact
centre, arm the observed command reference point. All are in one arm frame/mm.
Each sample must have fresh independent PHYSICAL evidence; pixels alone do not
provide these measurements. No source disappearance or pump echo is interpreted.
"""
import math
from dataclasses import dataclass
from typing import Optional
from treasure_auto.contracts import core as c, models as m, rules as r


@dataclass(frozen=True)
class MetricObservation:
    tool: m.ToolObservation
    arm: Optional[c.ArmPoint]
    cup: Optional[c.ArmPoint]
    brick: Optional[c.ArmPoint]
    support_z_mm: Optional[float]
    support_error_mm: Optional[float]
    support_visible: c.Fact
    tool_visible: c.Fact
    empty: c.Fact
    dropped: c.Fact
    drop_region: Optional[m.RegionBoundary]
    posture_id: Optional[str]


def distance(a,b):
    c.require(a.frame_id==b.frame_id,'frame_id','measurements must share frame',c.ErrorCode.INVALID_UNIT)
    return math.sqrt(sum((getattr(a,k)-getattr(b,k))**2 for k in ('x','y','z')))


def offset(sample):
    cup,brick=sample.cup,sample.brick
    distance(cup,brick)
    return c.ArmPoint(cup.frame_id,brick.x-cup.x,brick.y-cup.y,brick.z-cup.z,'mm',
                      cup.error_mm+brick.error_mm)


class Observer:
    def __init__(self,config):
        self.config=config
        self.seen=set()
        self.window=[]
        self.last_capture=None
        self.events={}
        self.sequences={}

    def reset(self):
        self.window=[]

    def validate(self,sample,context,capture_now,floor=None):
        c.require(type(sample) is MetricObservation,'sample','metric adapter input required')
        tool=sample.tool
        c.require(tool.envelope.producer is c.Owner.P04,'tool.producer','only P04 observations accepted')
        r.validate_envelope(tool.envelope,context)
        event_key=(tool.envelope.run_id,tool.envelope.producer,tool.envelope.event_id)
        old=self.events.get(event_key)
        if old is not None:
            c.require(old==c.dumps(tool),'event_id','same event id with different content',c.ErrorCode.ID_CONFLICT)
            raise c.ContractError(c.ErrorCode.DUPLICATE,'event_id','duplicate observation event')
        stream=(tool.envelope.instance_id,tool.envelope.control_epoch,tool.envelope.request_id,tool.envelope.action_id)
        c.require(tool.envelope.sequence>self.sequences.get(stream,-1),'sequence','old observation sequence',c.ErrorCode.STALE_REQUEST)
        c.require(bool(tool.evidence),'evidence','observation required',c.ErrorCode.EVIDENCE_MISSING)
        stamps=[]; keys=[]
        for ev in tool.evidence:
            r.require_physical_evidence(ev,tool.envelope,context.now,context.origin)
            age=capture_now-ev.captured_at
            c.require(0<=age<=self.config.evidence_age_s,'captured_at','old/future exposure',c.ErrorCode.EXPIRED)
            c.require(context.now-ev.received_at<=self.config.evidence_age_s,'received_at','old receipt',c.ErrorCode.EXPIRED)
            if floor is not None:
                c.require(ev.captured_at-floor>0,'captured_at','exposure predates stage',c.ErrorCode.EXPIRED)
            stamps.append(ev.captured_at)
            keys.extend((('id',ev.evidence_id),('uri',ev.uri),('capture',ev.captured_at)))
        c.require(all(t==stamps[0] for t in stamps),'captured_at','one exposure per sample')
        stamp=stamps[0]
        c.require(not any(k in self.seen for k in keys),'observation','duplicate exposure',c.ErrorCode.DUPLICATE)
        if self.last_capture is not None:
            c.require(stamp-self.last_capture>0,'captured_at','out-of-order exposure',c.ErrorCode.EXPIRED)
        for name in ('support_visible','tool_visible','empty','dropped'):
            c.require(type(getattr(sample,name)) is c.Fact,name,'explicit Fact required')
        for name in ('arm','cup','brick'):
            value=getattr(sample,name)
            c.require(value is None or type(value) is c.ArmPoint,name,'C0 mm point required')
        for name in ('support_z_mm','support_error_mm'):
            value=getattr(sample,name)
            c.require(value is None or (type(value) in (int,float) and math.isfinite(value)),name,'finite numeric measurement required')
        c.require(sample.support_error_mm is None or sample.support_error_mm>=0,'support_error_mm','negative error')
        c.require(not (sample.empty is c.Fact.CONFIRMED and sample.brick is not None),'empty','contradictory brick observation')
        self.seen.update(keys); self.last_capture=stamp
        self.events[event_key]=c.dumps(tool); self.sequences[stream]=tool.envelope.sequence
        return stamp

    def window_passes(self,sample,stamp,valid):
        if not valid:
            self.reset(); return False
        if self.window and stamp-self.window[-1][0]>self.config.max_frame_gap_s:
            self.reset()
        self.window.append((stamp,sample))
        self.window=[v for v in self.window if stamp-v[0]<=self.config.evidence_age_s]
        return (len(self.window)>=self.config.confirmation_frames and
                stamp-self.window[0][0]>=self.config.confirmation_span_s)

    def refs(self):
        return tuple(ev for _,sample in self.window for ev in sample.tool.evidence)

    def at_target(self,sample,target):
        return (sample.arm is not None and target is not None and
                distance(sample.arm,target)+sample.arm.error_mm+target.error_mm<=self.config.pose_tolerance_mm)

    def sees_target(self,sample,target_id):
        return (sample.tool_visible is c.Fact.CONFIRMED and sample.tool.same_object is c.Fact.CONFIRMED and
                sample.tool.target_id==target_id and sample.cup is not None and sample.brick is not None)

    def supported(self,sample,target_id):
        return (self.sees_target(sample,target_id) and sample.support_visible is c.Fact.CONFIRMED and
                sample.support_z_mm is not None and sample.support_error_mm is not None and
                abs(sample.brick.z-sample.support_z_mm)+sample.brick.error_mm+sample.support_error_mm<=self.config.pose_tolerance_mm)

    def lifted(self,sample,baseline,target_id):
        if baseline is None or not self.sees_target(sample,target_id):
            return False
        if sample.support_visible is not c.Fact.CONFIRMED or sample.support_z_mm is None or sample.support_error_mm is None:
            return False
        distance(sample.cup,baseline.cup); distance(sample.brick,baseline.brick)
        cfg=self.config
        dz=sample.brick.z-baseline.brick.z
        error=sample.brick.error_mm+baseline.brick.error_mm
        separation=sample.brick.z-sample.support_z_mm-sample.brick.error_mm-sample.support_error_mm
        return (dz-error>=cfg.min_lift_mm and dz+error<=cfg.max_lift_mm and separation>=cfg.min_lift_mm and
                abs(sample.support_z_mm-baseline.support_z_mm)+sample.support_error_mm+baseline.support_error_mm<=cfg.pose_tolerance_mm and
                self.held(sample,offset(baseline),target_id))

    def held(self,sample,expected,target_id):
        if expected is None or not self.sees_target(sample,target_id):
            return False
        measured=offset(sample)
        return distance(measured,expected)+measured.error_mm+expected.error_mm<=self.config.relative_tolerance_mm
