"""Standalone fake-clock match. Every observation/adapter is SIMULATED.

Run: python -B -m treasure_auto.mission.simulator --layout V1
This module has no hardware or network adapters. It is not the formal entrypoint.
"""
import argparse
import json
from dataclasses import replace
from treasure_auto.contracts import core as c,models as m
from .config import MissionConfig
from .checkpoint import SimulationJournal
from .engine import MissionEngine
from .inputs import Stage,MissionFrame
from .scoring import ScoreBrick,ScoreScene,ScoreZone,estimate


class SimulationClock:
    def __init__(self):self.seconds=100.
    def now(self):return c.ClockStamp(self.seconds,c.ClockBasis.MONOTONIC,'p09-sim-clock','p09-sim-boot')
    def advance(self,seconds):self.seconds+=seconds


def simulation_config():
    return MissionConfig('p09-sim-config','simulated','simulated',2.,1.,2.,2.,2.,1.,3,30.,3,2.)


class SimulatedWorld:
    """In-memory fake P03–P08 facts; not a durable or production state store."""
    def __init__(self,kind):
        self.clock=SimulationClock();self.kind=m.LayoutKind(kind);self.seq=0
        self.versions=c.Versions('p09-sim-config','sim-map','sim-calibration','sim-model',None,kind,0)
        e=self.env(c.Owner.P10A)
        self.run=m.RunContext(e,m.RunState.OFFLINE,'sim-start',self.clock.now(),replace(self.clock.now(),seconds=700.),False)
        cats=[m.Category.CAKE]+[m.Category.BANANA,m.Category.DOLL,m.Category.CLOCK]*3
        self.inventory=tuple(m.InventoryItem('inv-'+str(i),('target-'+str(i),),c.Fact.CONFIRMED,k,m.InventoryLocation.CENTRAL,None) for i,k in enumerate(cats))
        self.snapshot=m.StateSnapshot(self.run,0,m.SpaceSnapshot(self.env(c.Owner.P08),0,()),
            c.HoldingSnapshot(c.Holding.EMPTY,None,None,(self.proof(),)),self.inventory,(),(),None,())
        self.engine=MissionEngine(simulation_config(),self.clock,SimulationJournal())
        self.release_times={};self.orders=[];self.exit=None

    def env(self,owner,target=None,action=None,request=None):
        self.seq+=1;now=self.clock.now()
        return c.Envelope('p09-sim-run','p09-sim-instance',1,owner,'sim-event-'+str(self.seq),self.seq,request,action,target,
                          now,replace(now,seconds=now.seconds+5),self.versions,c.Origin.SIMULATED,0,0,0)

    def proof(self,target=None,action=None):
        self.seq+=1;now=self.clock.now()
        return c.EvidenceRef('sim-proof-'+str(self.seq),'p09-sim-run',target,action,c.EvidenceKind.PHYSICAL,c.Origin.SIMULATED,
                             now,now,replace(now,seconds=now.seconds+5),'sim://p09/'+str(self.seq),None)

    def update(self,**kw):self.snapshot=replace(self.snapshot,revision=self.snapshot.revision+1,**kw)

    def refresh(self):
        holding=replace(self.snapshot.holding,evidence=(self.proof(self.snapshot.holding.target_id),))
        layouts=tuple(replace(l,envelope=self.env(c.Owner.P08),bricks=tuple(replace(b,evidence=(self.proof(b.target_id),)) for b in l.bricks)) for l in self.snapshot.layouts)
        self.update(holding=holding,layouts=layouts)

    def frame(self,events=(),exits=()):
        self.refresh();snap=self.snapshot
        ev=self.proof();readiness=m.DriveReadiness(self.env(c.Owner.P07),snap.holding,c.Fact.CONFIRMED,(ev,))
        stop=m.StopReport(self.env(c.Owner.P01),c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,(ev,))
        targets=[];plans=[]
        for item in snap.inventory:
            if item.location is not m.InventoryLocation.CENTRAL:continue
            target=item.target_ids[0];proof=(self.proof(target),)
            corners=tuple(m.PixelPoint(u,v) for u,v in ((100,100),(200,100),(200,200),(100,200)))
            top=m.TopSurface(corners,corners,640,480,'sim-camera',m.Distortion.CORRECTED,1.,m.PlaneBasis.CALIBRATION,100.,m.Visibility.VISIBLE,proof)
            targets.append(m.TargetObservation('observation-'+target,target,c.Fact.CONFIRMED,item.category,top,m.Visibility.VISIBLE,proof))
            point=c.ArmPoint('arm',100,0,100,'mm',.1)
            plans.append(m.GeometryPlan(self.env(c.Owner.P06,target), 'path-'+target,point,(m.PathSegment(m.Phase.APPROACH,point,1.,1.),),c.Fact.CONFIRMED,c.Fact.CONFIRMED,proof))
        ins=m.MissionInput(self.env(c.Owner.P09),snap,tuple(targets),tuple(events),tuple(exits))
        return MissionFrame(ins,tuple(plans),readiness,stop)

    def complete(self,directive,unknown=False):
        d=directive;target=d.envelope.target_id;snap=self.snapshot;exits=()
        if not unknown:
            if d.stage is Stage.NAV_BOARD:
                self.exit=m.ExitEvidence(self.env(c.Owner.P03),m.Area.A,c.Fact.CONFIRMED,(self.proof(),));exits=(self.exit,)
            if d.stage is Stage.READ_BOARD:
                self.versions=replace(self.versions,mapping='sim-mapping')
                mapping=m.MappingSnapshot(self.env(c.Owner.P05),'sim-mapping',tuple(m.SlotAssignment(a,k) for a,k in
                    ((m.Area.B,m.Category.DOLL),(m.Area.C,m.Category.CLOCK),(m.Area.D,m.Category.BANANA))),c.Fact.CONFIRMED,self.clock.now(),(self.proof(),))
                self.update(mapping=mapping,space=replace(snap.space,envelope=self.env(c.Owner.P08)))
            if d.stage is Stage.GRASP:
                self.update(holding=c.HoldingSnapshot(c.Holding.HELD,target,c.ArmPoint('tool',0,0,0,'mm',.1),(self.proof(target),)),
                    inventory=tuple(replace(i,location=m.InventoryLocation.HELD) if target in i.target_ids else i for i in snap.inventory))
            if d.stage is Stage.PLACE:
                order=d.order;area=order.destination;old=next((l for l in snap.layouts if l.area is area),None)
                bs=old.bricks if old else ();n=len(bs)+1
                layer=1 if self.kind is m.LayoutKind.V1 or area is m.Area.A or self.kind is m.LayoutKind.V2 and n<3 else (2 if self.kind is m.LayoutKind.V2 else n)
                supports=() if layer==1 else (tuple(b.target_id for b in bs) if self.kind is m.LayoutKind.V2 else (bs[-1].target_id,))
                brick=m.LayoutBrick(target,n,layer,supports,.95 if supports else None,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,
                    c.Fact.CONFIRMED if n==1 else c.Fact.UNKNOWN,0,(self.proof(target),))
                lay=m.LayoutSnapshot(self.env(c.Owner.P08),area,self.kind,0,0,n,bs+(brick,),c.Fact.CONFIRMED,
                    c.Fact.CONFIRMED if area is m.Area.A or n==3 else c.Fact.UNKNOWN)
                key=m.DeliveryKey(self.run.envelope.run_id,order.inventory_id)
                receipt=m.DeliveryReceipt(key,'sim-release-'+target,order.category,area,self.clock.now(),(self.proof(target).evidence_id,))
                self.release_times[target]=self.clock.now()
                self.update(holding=c.HoldingSnapshot(c.Holding.EMPTY,None,None,(self.proof(),)),
                    inventory=tuple(replace(i,location=m.InventoryLocation.REGION,area=area) if target in i.target_ids else i for i in snap.inventory),
                    deliveries=snap.deliveries+(receipt,),layouts=tuple(l for l in snap.layouts if l.area is not area)+(lay,))
        ev=self.proof(target,d.envelope.action_id)
        status=c.Outcome.UNKNOWN if unknown else c.Outcome.SUCCESS
        result=c.ActionResult(status,d.stage.value,c.Reason(c.ErrorCode.UNAVAILABLE,'physical','simulated unknown') if unknown else None,
            c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.UNKNOWN if unknown else c.Fact.CONFIRMED,(ev,),.1)
        envelope=replace(d.envelope,producer=d.owner,event_id='sim-done-'+d.envelope.action_id)
        return self.frame((m.ActionEvent(envelope,result),),exits)

    def score(self):
        bricks=[]
        for layout in self.snapshot.layouts:
            for b in layout.bricks:
                item=next(i for i in self.snapshot.inventory if b.target_id in i.target_ids)
                bricks.append(ScoreBrick(self.env(c.Owner.P08,b.target_id),b.target_id,item.inventory_id,item.category,layout.area,ScoreZone.COLLECTION,
                    b.layer,b.support_ids,c.Fact.CONFIRMED,c.Fact.CONFIRMED,self.release_times[b.target_id],b.layer*.1,c.Fact.CONFIRMED,(self.proof(b.target_id),)))
        # The engine retains validated historical exit; re-observation here is a
        # simulated current scene for the standalone conditional score function.
        ex=m.ExitEvidence(self.env(c.Owner.P03),m.Area.A,c.Fact.CONFIRMED,(self.proof(),)) if self.engine.exit_points else None
        if self.snapshot.mapping is None:return None
        return estimate(ScoreScene(self.snapshot.mapping,tuple(bricks),ex,c.Fact.CONFIRMED,0.),self.run,self.clock.now()).estimate.confirmed_points


def run_simulation(layout='V1',fault=None):
    c.require(fault in (None,'unknown_grasp','deadline'),'fault','unsupported simulation scenario')
    world=SimulatedWorld(layout);out=world.engine.advance(world.frame());issued=[]
    for _ in range(200):
        if out.directive is None:break
        d=out.directive;issued.append(d.stage.value)
        if d.stage is Stage.GRASP:world.orders.append(d.order.category.value)
        if fault=='deadline':
            world.clock.advance(600);out=world.engine.advance(world.frame());break
        world.clock.advance(.1)
        out=world.engine.advance(world.complete(d,unknown=fault=='unknown_grasp' and d.stage is Stage.GRASP))
    return dict(origin='simulated',layout=layout,fault=fault,phase=out.decision.phase.value,
        arrivals=len(world.snapshot.deliveries),layout_complete=out.decision.layout_complete.value,
        conditional_points=world.score(),orders=world.orders,elapsed_s=world.clock.seconds-100,
        original_deadline_s=world.run.deadline.seconds,issued=issued,trace=world.engine.trace,
        end_reason=out.decision.end_reason.detail if out.decision.end_reason else None)


def main():
    parser=argparse.ArgumentParser(description='SIMULATED only; no robot output')
    parser.add_argument('--layout',choices=('V1','V2','V3'),default='V1')
    parser.add_argument('--fault',choices=('unknown_grasp','deadline'))
    args=parser.parse_args();print(json.dumps(run_simulation(args.layout,args.fault),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
