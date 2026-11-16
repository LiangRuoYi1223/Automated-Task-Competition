import unittest
from dataclasses import replace
from test_p08_support import *
from treasure_auto.placement.inputs import LayoutFacts,TemporaryPlacement,PlacementBinding
from treasure_auto.placement.layout import LayoutTracker, assess_brick
from treasure_auto.placement.recovery import recovery_step


def facts(target='brick-1',t=110,round_=0,support=None):
    return LayoutFacts(env(c.Owner.P06,target_id=target,layout_recovery_round=round_,emitted_at=s.stamp(t)),target,
        c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,(proof(t,target),),support)


def support_for(upper,lowers,t=110):
    return m.SupportGeometryResult(env(c.Owner.P06,target_id=upper.target_id,emitted_at=s.stamp(t)),upper.target_id,
        tuple(o.target_id for o in lowers),.01,.01,.99,c.Fact.CONFIRMED,c.Fact.CONFIRMED,(proof(t,upper.target_id),),None)


def inventory(locations=(m.InventoryLocation.REGION,)*3):
    return tuple(m.InventoryItem('inventory-'+str(i),('brick-'+str(i),),c.Fact.CONFIRMED,m.Category.DOLL,loc,
        m.Area.B if loc is m.InventoryLocation.REGION else None) for i,loc in enumerate(locations,1))


def failure(event='failure-1',round_=0,t=110):
    return m.ActionEvent(env(c.Owner.P04,event_id=event,layout_recovery_round=round_,emitted_at=s.stamp(t)),
        c.ActionResult(c.Outcome.FAILED,'layout',c.Reason(c.ErrorCode.INVALID_LAYOUT,'layout','observed failed geometry'),
        c.Fact.UNKNOWN,c.Fact.UNKNOWN,c.Fact.UNKNOWN,c.Fact.CONFIRMED,(proof(t),),0.))


class LayoutTests(unittest.TestCase):
    def test_T04_suspected_keeps_count_confirmed_clears_once_full_only_generation(self):
        tracker=LayoutTracker(layout(count=3,bricks=(brick(),brick(2),brick(3))),config())
        tracker.failure(failure(),c.Fact.UNKNOWN,c.Fact.UNKNOWN,context())
        self.assertEqual(tracker.layout.confirmed_count,3)
        out=tracker.failure(failure('f2'),c.Fact.CONFIRMED,c.Fact.REJECTED,context())
        self.assertEqual((out.confirmed_count,out.layout_recovery_round,out.rebuild_generation),(0,1,0))
        self.assertEqual(tracker.failure(failure('f2'),c.Fact.CONFIRMED,c.Fact.REJECTED,context()),out)
        out=tracker.failure(failure('f3',1),c.Fact.CONFIRMED,c.Fact.CONFIRMED,context(layout_recovery_round=1))
        self.assertEqual((out.confirmed_count,out.layout_recovery_round,out.rebuild_generation),(0,2,1))

    def test_T06_center_first_kept_but_old_second_cannot_be_inherited(self):
        tr=LayoutTracker(layout(round_=1),config());sp=space((occupancy(),occupancy('brick-2',x=.105)))
        ctx=context(layout_recovery_round=1)
        tr.keep_first('brick-1',sp,facts(round_=1),region(),c.HoldingSnapshot(c.Holding.EMPTY,None,None,()),inventory(),ctx)
        self.assertEqual(tr.layout.confirmed_count,1)
        with self.assertRaises(c.ContractError):tr.keep_first('brick-2',sp,facts('brick-2',round_=1),region(),
            c.HoldingSnapshot(c.Holding.EMPTY,None,None,()),inventory(),ctx)
        with self.assertRaises(c.ContractError):tr.record_placement(release('brick-2',round_=0),sp,facts('brick-2',round_=1),region(),ctx)
        self.assertEqual(tr.layout.confirmed_count,1)

    def test_T08_repeated_failure_replays_zero_one_two_three(self):
        tr=LayoutTracker(layout(m.LayoutKind.V1),config());counts=[]
        for round_ in (0,1):
            for n,x in ((1,0),(2,.105),(3,-.105)):
                t=110+round_*.4+n*.1;ctx=context(t,layout_recovery_round=round_)
                items=tuple(occupancy('brick-'+str(i),x=xx,t=t) for i,xx in ((1,0),(2,.105),(3,-.105)) if i<=n)
                quality=replace(facts('brick-'+str(n),t=t,round_=round_),prior_facts=tuple(facts('brick-'+str(i),t=t,round_=round_) for i in range(1,n)))
                tr.record_placement(release('brick-'+str(n),'r{}-{}'.format(round_,n),round_,t),space(items),quality,region(),ctx)
                counts.append(tr.layout.confirmed_count)
            if round_==0:
                tr.failure(failure(t=110.3),c.Fact.CONFIRMED,c.Fact.REJECTED,ctx);counts.append(tr.layout.confirmed_count)
        self.assertEqual(counts,[1,2,3,0,1,2,3])

    def test_R04_release_does_not_imply_layout_success(self):
        tr=LayoutTracker(layout(),config())
        bad=replace(facts(),stable=c.Fact.REJECTED)
        with self.assertRaises(c.ContractError):tr.record_placement(release(),space((occupancy(),)),bad,region(),context())
        self.assertEqual(tr.layout.confirmed_count,0)
        self.assertEqual(release().separated,c.Fact.CONFIRMED)

    def test_R08_V3_exact80_rejected_actual_overlap_not_claimed_ratio(self):
        low=occupancy();upper=occupancy('brick-2',x=.02,z=.1,supports=('brick-1',))
        sp=space((low,upper));f=replace(facts('brick-2',support=support_for(upper,(low,))),prior_facts=(facts(),))
        with self.assertRaises(c.ContractError):assess_brick(layout(count=1,bricks=(brick(),)),upper,f,sp,region(),config(),s.stamp(110))
        upper=replace(upper,footprint_m=square(.019,0));sp=space((low,upper))
        out=assess_brick(layout(count=1,bricks=(brick(),)),upper,replace(f,support=support_for(upper,(low,))),sp,region(),config(),s.stamp(110))
        self.assertGreater(out.support_ratio_lower,.8)

    def test_R08_V2_dual_support_gap_and_height_checked(self):
        a=occupancy();b=occupancy('brick-2',x=.105);top=occupancy('brick-3',x=.0525,z=.1,supports=('brick-1','brick-2'))
        prev=layout(m.LayoutKind.V2,2,bricks=(brick(kind=m.LayoutKind.V2),brick(2,kind=m.LayoutKind.V2)))
        def run(bb,tt):
            quality=replace(facts('brick-3',support=support_for(tt,(a,bb))),prior_facts=(facts(),facts('brick-2')))
            return assess_brick(prev,tt,quality,space((a,bb,tt)),region(),config(),s.stamp(110))
        self.assertEqual(run(b,top).support_ids,('brick-1','brick-2'))
        for bb,tt in ((replace(b,footprint_m=square(.12,0)),top),(replace(b,z_max_m=.11),top),
                      (b,replace(top,footprint_m=square(-.03,0)))):
            with self.subTest(bb=bb,tt=tt),self.assertRaises(c.ContractError):run(bb,tt)


class RecoveryTests(unittest.TestCase):
    def base(self,**kw):
        args=dict(target_id='brick-1',inventory=inventory(),holding=c.HoldingSnapshot(c.Holding.EMPTY,None,None,()),
                  space=space((occupancy(),)),temporary=None,ctx=context(),config=config(),
                  budget=m.Budget('recovery-B',3,10,0,0),deadline=s.stamp(700))
        args.update(kw);return recovery_step(**args)

    def test_T07_held_unknown_or_missing_inventory_blocks_next_grasp(self):
        for held in (c.Holding.HELD,c.Holding.UNKNOWN,c.Holding.DROPPED):
            self.assertEqual(self.base(holding=c.HoldingSnapshot(held,'brick-3',None,())).operation,'resolve_holding')
        self.assertEqual(self.base(inventory=inventory((m.InventoryLocation.REGION,m.InventoryLocation.UNKNOWN,m.InventoryLocation.CENTRAL))).operation,'reconcile_inventory')

    def test_T07_bottom_loaded_requires_top_down_and_reserved_temporary_plan(self):
        items=(occupancy(),occupancy('brick-2',z=.1,supports=('brick-1',)),occupancy('brick-3',z=.2,supports=('brick-2',)))
        self.assertEqual(self.base(space=space(items)).operation,'await_temporary')
        tmp=occupancy('temp',x=.15,state=m.OccupancyState.RESERVED)
        plan=replace(request().plan,envelope=replace(request().plan.envelope,target_id='brick-3'),
            evidence=(proof(target='brick-3'),),segments=(replace(request().plan.segments[0],phase=m.Phase.TEMPORARY),))
        binding=PlacementBinding(tmp,c.ArmPoint('tool',0,0,0,'mm',0),plan.path_id,region().placement_center,(proof(target='brick-3'),))
        out=self.base(space=space(items+(tmp,)),temporary=TemporaryPlacement(tmp,plan,binding,region()))
        self.assertEqual((out.operation,out.target_id),('dismantle','brick-3'))

    def test_R07_third_central_is_fetch_not_fabricated_ground_and_budget_persists(self):
        inv=inventory((m.InventoryLocation.REGION,m.InventoryLocation.REGION,m.InventoryLocation.CENTRAL))
        self.assertEqual(self.base(target_id='brick-3',inventory=inv).operation,'fetch_central')
        exhausted=m.Budget('recovery-B',3,10,3,5)
        self.assertEqual(self.base(budget=exhausted).operation,'budget_exhausted')
        self.assertEqual(exhausted.attempts_used,3)
        self.assertEqual(self.base(deadline=s.stamp(110)).operation,'deadline')


if __name__=='__main__':unittest.main()
