"""P08-only SIMULATED fixtures. Values are not field configuration."""
import sys
from pathlib import Path
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c, models as m
from contract_fixtures import samples as s


def env(owner=c.Owner.P08, **kw):
    return s.envelope(owner, **kw)


def proof(t=110., target='brick-1', action='sim-action-1', ident=None):
    key=ident or 'frame-'+str(t)
    return s.evidence(evidence_id=key, target_id=target, action_id=action,
                      captured_at=s.stamp(t), received_at=s.stamp(t),
                      valid_until=s.stamp(t+5), uri='sim://p08/'+key)


def square(x=0.,y=0.,side=.1):
    h=side/2
    return tuple(m.Point2(a,b) for a,b in ((x-h,y-h),(x+h,y-h),(x+h,y+h),(x-h,y+h)))


def occupancy(target='brick-1',x=0.,y=0.,z=0.,t=110.,state=m.OccupancyState.ACTUAL,supports=(),area=m.Area.B,error=0.):
    return m.Occupancy(target,state,'map',square(x,y),z,z+.1,error,tuple(supports),s.stamp(t),
                       m.Visibility.VISIBLE,area,(proof(t,target),))


def space(items=(),version=0):
    return m.SpaceSnapshot(env(versions=s.versions(version),target_id=None,action_id=None),version,tuple(items))


def region(area=m.Area.B):
    return m.RegionBoundary(area,'map',square(side=.45),c.MapPose('map',0,0,0,'m','rad',0,0),0)


def layout(kind=m.LayoutKind.V3,count=0,round_=0,bricks=()):
    return m.LayoutSnapshot(env(layout_recovery_round=round_),m.Area.B,kind,round_,0,count,tuple(bricks),c.Fact.UNKNOWN,c.Fact.UNKNOWN)


def brick(n=1,round_=0,kind=m.LayoutKind.V3):
    layer=1 if kind is m.LayoutKind.V1 or (kind is m.LayoutKind.V2 and n<3) else (2 if kind is m.LayoutKind.V2 else n)
    supports=() if layer==1 else (('brick-1','brick-2') if kind is m.LayoutKind.V2 else ('brick-'+str(n-1),))
    return m.LayoutBrick('brick-'+str(n),n,layer,supports,.99 if layer>1 else None,c.Fact.CONFIRMED,c.Fact.CONFIRMED,
                          c.Fact.CONFIRMED,c.Fact.CONFIRMED if n==1 else c.Fact.UNKNOWN,round_,(proof(target='brick-'+str(n)),))


def config_data():
    return dict(version='sim-config-v1',origin='simulated',measurement='simulated',
        center_tolerance_m=.02,adjacent_gap_m=.005,height_tolerance_m=.002,ground_z_m=0.,
        evidence_max_age_s=1.,frame_gap_s=.3,release_window_s=.2,brick_motion_m=.002,
        cup_retreat_m=.02,still_translation_m=.003,still_angle_rad=.02,certificate_ttl_s=.3)


def config():
    from treasure_auto.placement.config import PlacementConfig
    return PlacementConfig.from_dict(config_data())


def context(now=110.,**kw):
    return replace(s.consumer_context(),now=s.stamp(now),**kw)


def request():
    e=env(c.Owner.P09)
    order=m.WorkOrder(e,m.Category.DOLL,m.Area.B,'sim-mapping-v1','inventory-1',m.Budget('recovery-B',5,30.,0,0.))
    p=c.ArmPoint('arm',100,0,200,'mm',.1)
    plan=m.GeometryPlan(env(c.Owner.P06),'plan-place',p,tuple(m.PathSegment(ph,p,1.,2.) for ph in
                           (m.Phase.APPROACH,m.Phase.DESCEND,m.Phase.PLACE,m.Phase.WITHDRAW)),c.Fact.CONFIRMED,c.Fact.CONFIRMED,(proof(),))
    stop=m.StopReport(env(c.Owner.P01),c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,(proof(),))
    held=c.HoldingSnapshot(c.Holding.HELD,'brick-1',c.ArmPoint('tool',2,3,4,'mm',.1),(proof(),))
    permit=m.ControlPermit(env(c.Owner.P10A),'p08-permit',m.MotionSource.ARM,'place',(proof(),))
    return m.PlacementRequest(e,order,s.mapping_samples()[0],plan,stop,held,permit,region())


def metric(t=110.,x=0.,cup=(0.,0.,.1),ids=('brick-1',),scene=0,visible=True):
    from treasure_auto.placement.inputs import MetricFrame
    items=tuple(occupancy(i,x=x,t=t) for i in ids)
    if not visible:items=tuple(replace(o,visibility=m.Visibility.OCCLUDED) for o in items)
    ev=proof(t,target=None,action=None)
    return MetricFrame(env(c.Owner.P04,emitted_at=s.stamp(t),expires_at=s.stamp(t+5),event_id=ev.evidence_id,
                            sequence=int(t*1000),target_id=None,action_id=None),ev.evidence_id,ev,items,
                       square(side=.8),'map',scene,c.Fact.CONFIRMED,cup,0.,tuple((i,0.,0.) for i in ids))


def release(target='brick-1',event='release-1',round_=0,t=110.):
    return m.ReleaseEvent(env(event_id=event,target_id=target,layout_recovery_round=round_,emitted_at=s.stamp(t)),
        m.DeliveryKey('sim-run-1','inventory-'+target[-1]),m.Category.DOLL,m.Area.B,'sim-mapping-v1',s.stamp(t),
        c.Fact.CONFIRMED,c.Fact.CONFIRMED,(proof(t,target),))
