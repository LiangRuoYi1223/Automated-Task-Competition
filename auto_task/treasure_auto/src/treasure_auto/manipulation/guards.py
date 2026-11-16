"""Context checks in addition to C0's structural validation."""
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m, rules as r


def context_for(context,envelope,action=True):
    return replace(context,request_id=envelope.request_id,action_id=envelope.action_id if action else None,
                   target_id=envelope.target_id)


def physical(refs,envelope,context):
    c.require(bool(refs),'evidence','fresh physical evidence required',c.ErrorCode.EVIDENCE_MISSING)
    for ev in refs:
        r.require_physical_evidence(ev,envelope,context.now,context.origin)


def stopped(report,context,firmware=False):
    c.require(type(report) is m.StopReport,'stop','C0 StopReport required')
    c.require(report.envelope.producer in (c.Owner.P01,c.Owner.P03),'stop.producer','only stop evidence owners')
    if firmware:
        c.require(report.envelope.producer is c.Owner.P01,'stop.producer',
                  'P01 firmware queue proof required in addition to base stop',c.ErrorCode.EVIDENCE_MISSING)
    r.validate_envelope(report.envelope,replace(context,request_id=None,action_id=None,target_id=None))
    c.require(report.physically_stopped is c.Fact.CONFIRMED and report.old_queue_empty is c.Fact.CONFIRMED,
              'stop','actual stop and old firmware queue must both be confirmed',c.ErrorCode.EVIDENCE_MISSING)
    physical(report.evidence,report.envelope,context)


def plan(plan,context,phase=None):
    c.require(type(plan) is m.GeometryPlan,'plan','fresh P06 stage plan unavailable',c.ErrorCode.UNAVAILABLE)
    c.require(plan.envelope.producer is c.Owner.P06,'plan.producer','only P06 geometry accepted')
    r.validate_envelope(plan.envelope,context)
    c.require(plan.reachable is c.Fact.CONFIRMED and plan.collision_free is c.Fact.CONFIRMED,
              'plan','reach/collision unknown',c.ErrorCode.STALE_GEOMETRY)
    c.require(bool(plan.segments),'segments','empty plan')
    if phase is not None:
        c.require(len(plan.segments)==1 and plan.segments[0].phase is phase,'segments',
                  'independently certified single stage required; do not retag P06 withdraw as transport',c.ErrorCode.STALE_GEOMETRY)
    c.require(all(s.target.frame_id==plan.target.frame_id for s in plan.segments),'frame_id','mixed frames')
    physical(plan.evidence,plan.envelope,context)


def permit(value,context,source):
    c.require(type(value) is m.ControlPermit,'permit','current P10A permit required',c.ErrorCode.NOT_AUTHORIZED)
    r.validate_envelope(value.envelope,context)
    c.require(value.source is source and value.purpose=='grasp','permit','wrong control source or purpose',c.ErrorCode.NOT_AUTHORIZED)
    physical(value.prerequisite_evidence,value.envelope,context)


def request(req,context,holding):
    c.require(type(req) is m.GraspRequest,'request','C0 GraspRequest required')
    c.require(req.envelope.producer is c.Owner.P09 and req.order.envelope.producer is c.Owner.P09,
              'producer','only P09 unique work orders',c.ErrorCode.NOT_AUTHORIZED)
    r.validate_envelope(req.envelope,context)
    expected=context_for(context,req.envelope)
    r.validate_envelope(req.order.envelope,replace(expected,action_id=None))
    c.require(req.envelope.action_id is not None and req.envelope.target_id is not None,'action_id','root action/target required')
    c.require(req.order.mapping_version==context.versions.mapping,'mapping_version','mapping not frozen/current',c.ErrorCode.VERSION_MISMATCH)
    c.require(all(getattr(context.versions,n) is not None for n in ('map','calibration','model','mapping','space')),
              'versions','complete grasp versions required',c.ErrorCode.VERSION_MISMATCH)
    c.require(holding.state is c.Holding.EMPTY and req.holding==holding,'holding','only P07 confirmed empty can admit another grasp',c.ErrorCode.OCCUPANCY_UNRESOLVED)
    physical(holding.evidence,replace(req.envelope,target_id=None,action_id=None),context)
    stopped(req.stop,context)
    plan(req.plan,expected)
    permit(req.permit,expected,m.MotionSource.ARM)
    b=req.order.budget
    c.require(b.attempts_limit is not None and b.attempts_limit>0 and b.seconds_limit is not None and b.seconds_limit>0,
              'budget','finite cumulative attempt/time limits required',c.ErrorCode.UNCONFIGURED)
