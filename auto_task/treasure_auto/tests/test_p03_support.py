"""P03 SIMULATED fixtures only; no robot, ROS, or measured parameters."""
from publication_support import simulated_space_readback
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c, models as m

def stamp(t=110.):
    return c.ClockStamp(t, c.ClockBasis.MONOTONIC, 'sim-monotonic', 'sim-boot')

def env(t=110., **kw):
    d = dict(run_id='sim-run-1', instance_id='sim-instance-1', control_epoch=1,
             producer=c.Owner.P03, event_id='e-' + str(t), sequence=int(t*1000),
             request_id='nav1', action_id=None, target_id=None,
             emitted_at=stamp(t), expires_at=stamp(t+10),
             versions=c.Versions('sim-config-v1','map-v1',None,None,None,None,0),
             origin=c.Origin.SIMULATED, layout_recovery_round=0, rebuild_generation=0, base_motion_epoch=0)
    d.update(kw)
    return c.Envelope(**d)

def proof(t=110., **kw):
    d=dict(evidence_id='f-'+str(t), run_id='sim-run-1', target_id=None, action_id=None,
           kind=c.EvidenceKind.PHYSICAL, origin=c.Origin.SIMULATED,
           captured_at=stamp(t), received_at=stamp(t), valid_until=stamp(t+10),
           uri='sim://frame/'+str(t), sha256=None)
    d.update(kw)
    return c.EvidenceRef(**d)

def poly(x0,y0,x1,y1):
    return tuple(m.Point2(x,y) for x,y in ((x0,y0),(x1,y0),(x1,y1),(x0,y1)))

def pose(x=2,y=2,yaw=0,**kw):
    d=dict(frame_id='map',x=x,y=y,yaw=yaw,position_unit='m',angle_unit='rad',error_m=0.,error_rad=0.)
    d.update(kw)
    return c.MapPose(**d)

def bundle():
    from treasure_auto.mapping.storage import MapBundle,Raster
    regions=tuple(m.RegionBoundary(a,'map',poly(i*2,0,i*2+1,1),pose(i*2+.5,.5),0.) for i,a in enumerate(m.Area))
    meta=m.MapMetadata(env(producer=c.Owner.P02), 'map-v1','sim-grid',.5,pose(0,0),regions,m.Measurement.SIMULATED)
    return MapBundle(meta,Raster(16,16,(255,)*256),poly(0,0,8,8),poly(3,3,4,4),(),.196,.65)

def profile():
    from treasure_auto.navigation.config import EnvelopeProfile
    return EnvelopeProfile(poly(-.3,-.2,.3,.2),poly(-.1,-.1,.4,.1),poly(-.1,-.2,.5,.2),.01,m.Measurement.SIMULATED)

def policy():
    from treasure_auto.navigation.config import NavigationPolicy,DockPolicy
    return NavigationPolicy('sim-config-v1',m.Measurement.SIMULATED,1.,.25,.2,.01,.02,
                            8.,3.,3.,2,2.,.1,.1,.3,.8,.5,1.,
                            (DockPolicy('observe',.03,.03,.04,.6),))

def readiness(t=110.,state=c.Holding.EMPTY,**kw):
    held='brick1' if state is c.Holding.HELD else None
    h=c.HoldingSnapshot(state,held,None,(proof(t,target_id=held),))
    return m.DriveReadiness(env(t,producer=c.Owner.P07,**kw),h,c.Fact.CONFIRMED,(proof(t),))

def space(t=110.,version=0,items=()):
    return m.SpaceSnapshot(env(t,producer=c.Owner.P08,versions=replace(env().versions,space=version)),version,items)

def loc(t=110.,p=None,**kw):
    e=env(t,producer=c.Owner.P02,**kw)
    return m.LocalizationSnapshot(e,p or pose(),c.Fact.CONFIRMED,e.base_motion_epoch,(proof(t),),None)

def request(t=110.,**kw):
    e=env(t,producer=c.Owner.P09,**kw)
    goal=m.WaypointResult(replace(e,producer=c.Owner.P02),'board','observe',pose(3,2),c.Fact.CONFIRMED,None)
    permit=m.ControlPermit(replace(e,producer=c.Owner.P10A),'coarse-lease',m.MotionSource.NAVIGATION,'observe',(proof(t),))
    return m.NavigationRequest(e,goal,loc(t),space(t),readiness(t),permit)

class Clock:
    def __init__(self,t=110.):self.t=t
    def now(self):return stamp(self.t)
    def advance(self,s):self.t+=s

def run():
    return m.RunContext(env(producer=c.Owner.P10A),m.RunState.OFFLINE,'start',stamp(100),stamp(700),False)

def obstacle(box=(2.4,1.95,2.5,2.05),state=m.OccupancyState.UNKNOWN):
    return m.Occupancy('low-brick',state,'map',poly(*box),0,.05,0.,(),stamp(),m.Visibility.OCCLUDED,None,())

def receipt(intent):
    from treasure_auto.contracts.rules import intent_digest
    return m.IntentReceipt(intent.envelope.run_id,intent.envelope.action_id,intent_digest(intent),
                           m.CommitReceipt('commit-'+intent.envelope.action_id,m.CommitStatus.COMMITTED,1,0,None))

class GatewayBoundary:
    """Only the external output/revocation boundary is replaced here."""
    def __init__(self):self.revocations=[];self.commands=[]
    def revoke(self,source):self.revocations.append(source);return True
    def health(self):return m.HealthSnapshot(env(producer=c.Owner.P01),(m.Capability('base',c.Fact.CONFIRMED,None,()),))
    def register_intent(self,command,intent,committed):
        from treasure_auto.contracts.rules import validate_intent_receipt
        validate_intent_receipt(intent,committed)
    def submit_base(self,command):
        self.commands.append(command)
        return m.ActionEvent(replace(command.envelope,producer=c.Owner.P01),c.ActionResult(c.Outcome.UNKNOWN,'submitted',
             c.Reason(c.ErrorCode.EVIDENCE_MISSING,'physical','not observed'),c.Fact.CONFIRMED,c.Fact.CONFIRMED,
             c.Fact.UNKNOWN,c.Fact.UNKNOWN,(),0.))

def service():
    from treasure_auto.navigation.service import NavigationService
    clock=Clock();gateway=GatewayBoundary()
    nav=NavigationService(run(),clock,policy(),profile(),bundle(),gateway,
                          lambda req:(req.localization.pose,req.goal.pose),space_readback=simulated_space_readback)
    return nav,clock,gateway

def start(nav,req=None):
    nav.navigate(req or request())
    nav.submit_goal(receipt(nav.goal_intent))

def arrived(nav,clock,status=c.Outcome.SUCCESS):
    e=replace(nav.goal_intent.envelope,producer=c.Owner.P01,event_id='arrive',emitted_at=clock.now())
    return m.ActionEvent(e,c.ActionResult(status,'coarse',None if status is c.Outcome.SUCCESS else
        c.Reason(c.ErrorCode.UNAVAILABLE,'planner','blocked'),c.Fact.CONFIRMED,c.Fact.CONFIRMED,
        c.Fact.CONFIRMED,c.Fact.UNKNOWN,(),0.))

def motion(nav,clock,v=0.,w=0.,**kw):
    from treasure_auto.navigation.observations import MotionSample
    e=replace(nav.context,producer=c.Owner.P01,event_id='motion-'+str(clock.t),emitted_at=clock.now())
    return MotionSample(e,v,w,proof(clock.t,**kw))

def stop_report(nav,clock,**kw):
    d=dict(envelope=replace(nav.context,producer=c.Owner.P01,event_id='stop-'+str(clock.t),emitted_at=clock.now()),
           requested=c.Fact.CONFIRMED,submitted=c.Fact.CONFIRMED,echoed=c.Fact.CONFIRMED,
           physically_stopped=c.Fact.CONFIRMED,old_queue_empty=c.Fact.CONFIRMED,evidence=(proof(clock.t),))
    d.update(kw);return m.StopReport(**d)

def stopped(nav,clock):
    nav.observe_stop(stop_report(nav,clock))
    nav.observe_motion(motion(nav,clock))
    clock.advance(.21)
    nav.observe_motion(motion(nav,clock))

def precision_permit(nav,clock):
    return m.ControlPermit(replace(nav.context,producer=c.Owner.P10A,event_id='precision-permit',emitted_at=clock.now()),
                           'precision-lease',m.MotionSource.PRECISION,'observe',(proof(clock.t),))

def to_precision(nav,clock):
    start(nav);nav.coarse_event(arrived(nav,clock));stopped(nav,clock)
    nav.install_precision(precision_permit(nav,clock))

def feedback(nav,clock,**kw):
    from treasure_auto.navigation.observations import DockFeedback
    d=dict(envelope=replace(nav.context,producer=c.Owner.P06,event_id='geo-'+str(clock.t),emitted_at=clock.now()),
           goal_name='board',purpose='observe',forward_m=0.,lateral_m=0.,yaw_rad=0.,
           distance_basis=m.PlaneBasis.CALIBRATION,geometry_valid=c.Fact.CONFIRMED,
           work_condition=c.Fact.CONFIRMED,self_occluded=False,evidence=(proof(clock.t),))
    d.update(kw);return DockFeedback(**d)
