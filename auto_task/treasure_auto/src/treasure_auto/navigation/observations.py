"""P03 local sampling inputs, NOT additions to the C0 wire registry."""
from dataclasses import dataclass, replace
from treasure_auto.contracts import core as c, models as m, rules as r
from .config import number


def reason(error):
    return c.Reason(error.code,error.field,error.detail)


def validate_env(value, reference, now, identity=False, motion=True):
    context=m.ConsumerContext(reference.run_id,reference.instance_id,reference.control_epoch,now,reference.versions,
        reference.request_id if identity else None,None,reference.target_id if identity else None,
        reference.layout_recovery_round,reference.rebuild_generation,reference.base_motion_epoch if motion else None,reference.origin)
    r.validate_envelope(value,context)


def fresh(evidence,envelope,now,max_age_s,target=None):
    c.require(bool(evidence),'evidence','physical evidence required',c.ErrorCode.EVIDENCE_MISSING)
    for e in evidence:
        r.require_physical_evidence(e,replace(envelope,action_id=None,target_id=target),now,envelope.origin)
        # A raw sensor/ROS stamp needs a verified P01/P02 conversion first.
        c.monotonic(e.captured_at,'captured_at')
        c.require(0 <= now-e.captured_at <= max_age_s and e.received_at-e.captured_at>=0,
                  'captured_at','stale/future or unmapped capture',c.ErrorCode.EXPIRED)


def validate_localization(value,reference,now,max_age_s):
    validate_env(value.envelope,reference,now)
    c.require(value.envelope.producer is c.Owner.P02 and value.trusted is c.Fact.CONFIRMED and value.pose is not None,
              'localization','trusted P02 pose required',c.ErrorCode.UNAVAILABLE)
    c.require(value.base_motion_epoch==value.envelope.base_motion_epoch,'base_motion_epoch','pose epoch mismatch')
    fresh(value.evidence,value.envelope,now,max_age_s)


def validate_readiness(value,reference,now,max_age_s):
    validate_env(value.envelope,reference,now)
    c.require(value.envelope.producer is c.Owner.P07,'readiness.producer','P07 owns holding/posture')
    c.require(value.holding.state in (c.Holding.EMPTY,c.Holding.HELD) and value.posture_confirmed is c.Fact.CONFIRMED,
              'readiness','confirmed holding and travel posture required',c.ErrorCode.UNAVAILABLE)
    fresh(value.holding.evidence,value.envelope,now,max_age_s,value.holding.target_id)
    fresh(value.evidence,value.envelope,now,max_age_s)


@dataclass(frozen=True)
class MotionSample:
    envelope: c.Envelope
    linear_m_s: float
    angular_rad_s: float
    evidence: c.EvidenceRef


@dataclass(frozen=True)
class DockFeedback:
    envelope: c.Envelope
    goal_name: str
    purpose: str
    forward_m: object
    lateral_m: object
    yaw_rad: object
    distance_basis: m.PlaneBasis
    geometry_valid: c.Fact
    work_condition: c.Fact
    self_occluded: bool
    evidence: tuple


class StopWindow:
    def __init__(self,policy):
        self.policy=policy;self.reset()

    def reset(self):
        self.samples=[];self.seen={};self.last_capture=None

    def observe(self,sample,reference,now,since):
        try:
            validate_env(sample.envelope,reference,now)
            c.require(sample.envelope.producer is c.Owner.P01,'motion','P01 motion observation required')
            fresh((sample.evidence,),sample.envelope,now,self.policy.max_age_s)
            c.require(type(sample.linear_m_s) in (int,float) and type(sample.angular_rad_s) in (int,float),
                      'motion','numeric velocities must not be bool')
            number(abs(sample.linear_m_s),'linear',False);number(abs(sample.angular_rad_s),'angular',False)
            at=sample.evidence.captured_at
            c.require(at-since>=0,'motion','capture before stop request')
            if sample.evidence.evidence_id in self.seen:
                c.require(self.seen[sample.evidence.evidence_id]==sample,'motion','same ID changed payload',c.ErrorCode.ID_CONFLICT)
                return False
            if self.last_capture is not None:
                c.require(at-self.last_capture>0,'motion','older capture cannot follow a newer observation')
            self.last_capture=at
            if self.samples:
                gap=at-self.samples[-1].evidence.captured_at
                c.require(gap>0,'motion','capture did not advance')
                if gap>self.policy.max_gap_s:self.samples=[]
            self.seen[sample.evidence.evidence_id]=sample
            if abs(sample.linear_m_s)>self.policy.stop_linear_m_s or abs(sample.angular_rad_s)>self.policy.stop_angular_rad_s:
                self.samples=[];return False
            self.samples.append(sample)
            return self.confirmed(now)
        except (c.ContractError,TypeError,OverflowError):
            self.samples=[];return False

    def confirmed(self,now):
        return (len(self.samples)>=2 and 0<=now-self.samples[-1].evidence.captured_at<=self.policy.max_gap_s and
                self.samples[-1].evidence.captured_at-self.samples[0].evidence.captured_at>=self.policy.stop_window_s)

    @property
    def evidence(self):return tuple(s.evidence for s in self.samples)
