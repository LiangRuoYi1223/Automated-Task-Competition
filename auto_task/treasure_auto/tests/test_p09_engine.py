import unittest
from dataclasses import replace
from test_p09_support import *
from treasure_auto.mission.config import MissionConfig
from treasure_auto.mission.inputs import MissionFrame,Stage
from treasure_auto.mission.engine import MissionEngine
from treasure_auto.mission.checkpoint import SimulationJournal


class Flow:
    def __init__(self,mapping=False):
        self.clock=FakeClock(s.stamp(110));self.snapshot=empty_snapshot(mapping)
        self.journal=SimulationJournal();self.engine=MissionEngine(MissionConfig.from_dict(config_data()),self.clock,self.journal)
        self.events=();self.candidates=();self.plans=();self.exits=()

    def env(self,owner,**kw):
        base=self.snapshot.space.envelope
        return replace(base,producer=owner,emitted_at=self.clock.now(),expires_at=s.stamp(self.clock.now().seconds+5),**kw)

    def frame(self):
        now=self.clock.now().seconds
        ev=proof(now)
        ready=m.DriveReadiness(self.env(c.Owner.P07,target_id=None,action_id=None),self.snapshot.holding,c.Fact.CONFIRMED,(ev,))
        stop=m.StopReport(self.env(c.Owner.P01,target_id=None,action_id=None),c.Fact.CONFIRMED,c.Fact.CONFIRMED,
            c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,(ev,))
        ins=m.MissionInput(self.env(c.Owner.P09,target_id=None,action_id=None),self.snapshot,self.candidates,self.events,self.exits)
        return MissionFrame(ins,self.plans,ready,stop)

    def step(self,**changes):return self.engine.advance(replace(self.frame(),**changes))

    def bump(self,**changes):self.snapshot=replace(self.snapshot,revision=self.snapshot.revision+1,**changes)

    def finish(self,status=c.Outcome.SUCCESS,physical=c.Fact.CONFIRMED,**envchanges):
        d=self.engine.pending
        ev=proof(self.clock.now().seconds,d.envelope.target_id,d.envelope.action_id,d.envelope.action_id)
        e=replace(d.envelope,producer=d.owner,event_id='done-'+d.envelope.action_id,**envchanges)
        why=None if status is c.Outcome.SUCCESS else c.Reason(c.ErrorCode.UNAVAILABLE,'physical','not complete')
        result=c.ActionResult(status,d.stage.value,why,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,physical,(ev,),1.)
        self.events=(m.ActionEvent(e,result),)
        out=self.step();self.events=();return out

    def board(self):
        self.step();self.finish()
        v=s.versions();self.bump(mapping=s.mapping_samples()[0],space=replace(self.snapshot.space,envelope=self.env(c.Owner.P08,versions=v)))
        return self.finish()

    def add_candidate(self,category=m.Category.CAKE,target='cake'):
        ev=proof(self.clock.now().seconds,target,None,target)
        obs=m.TargetObservation('obs-'+target,target,c.Fact.CONFIRMED,category,replace(s.top_surface(),evidence=(ev,)),m.Visibility.VISIBLE,(ev,))
        e=self.env(c.Owner.P06,target_id=target,action_id=None)
        point=c.ArmPoint('arm',100,0,100,'mm',.1)
        plan=m.GeometryPlan(e,'plan-'+target,point,(m.PathSegment(m.Phase.APPROACH,point,1.,1.),),c.Fact.CONFIRMED,c.Fact.CONFIRMED,(ev,))
        item=m.InventoryItem('inventory-'+target,(target,),c.Fact.CONFIRMED,category,m.InventoryLocation.CENTRAL,None)
        self.bump(inventory=self.snapshot.inventory+(item,));self.candidates+=(obs,);self.plans+=(plan,)

    def grasp(self):
        target=self.engine.order.envelope.target_id
        held=c.HoldingSnapshot(c.Holding.HELD,target,c.ArmPoint('tool',0,0,0,'mm',.1),(proof(target=target),))
        self.bump(holding=held,inventory=tuple(replace(i,location=m.InventoryLocation.HELD) if target in i.target_ids else i for i in self.snapshot.inventory))
        return self.finish()

    def release(self,quality=c.Fact.CONFIRMED):
        order=self.engine.order;target=order.envelope.target_id;now=self.clock.now().seconds
        e=self.env(c.Owner.P08,target_id=target,action_id=None)
        b=m.LayoutBrick(target,1,1,(),None,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,0,(proof(now,target),))
        lay=m.LayoutSnapshot(e,order.destination,m.LayoutKind.V1,0,0,1,(b,),c.Fact.UNKNOWN,quality)
        d=m.DeliveryReceipt(m.DeliveryKey('sim-run-1',order.inventory_id),'delivery-'+target,order.category,order.destination,s.stamp(now),('release-'+target,))
        self.bump(holding=c.HoldingSnapshot(c.Holding.EMPTY,None,None,(proof(),)),deliveries=self.snapshot.deliveries+(d,),
            layouts=tuple(l for l in self.snapshot.layouts if l.area is not lay.area)+(lay,),
            inventory=tuple(replace(i,location=m.InventoryLocation.REGION,area=order.destination) if i.inventory_id==order.inventory_id else i for i in self.snapshot.inventory))
        return self.finish()

    def cake_done(self):
        self.board();self.add_candidate();self.finish();self.grasp();self.finish();self.release();self.finish();self.finish()
        return self.finish()


class EngineTests(unittest.TestCase):
    def test_T01_missing_mapping_only_board_never_grasp(self):
        h=Flow();h.add_candidate();out=h.step()
        self.assertEqual(out.directive.stage,Stage.NAV_BOARD)
        self.assertIsNone(out.decision.order)
        self.assertIsNone(h.step().directive)
        self.assertEqual(h.finish().directive.stage,Stage.READ_BOARD)
        h.finish()
        self.assertIsNone(h.engine.order)
        self.assertNotEqual(h.engine.stage,Stage.GRASP)
        self.assertEqual(h.snapshot.run.deadline,s.stamp(700))

    def test_T01_cake_closure_required_before_ordinary(self):
        h=Flow();h.board();h.add_candidate(m.Category.DOLL,'doll');h.add_candidate()
        self.assertEqual(h.finish().decision.order.category,m.Category.CAKE)
        h.grasp();h.finish();h.release(quality=c.Fact.UNKNOWN);out=h.finish()
        self.assertEqual(out.decision.phase,m.MissionPhase.RECOVERY)
        self.assertEqual(h.engine.order.category,m.Category.CAKE)

    def test_T02_one_target_arbitrary_category_no_parallel_order(self):
        for category in (m.Category.BANANA,m.Category.CLOCK,m.Category.DOLL):
            h=Flow();h.cake_done();h.add_candidate(category,'first');h.add_candidate(m.Category.CLOCK,'second')
            out=h.finish();self.assertEqual(out.decision.order.category,category)
            self.assertEqual(out.decision.order.envelope.target_id,'first')
            self.assertIsNone(h.step().directive)
            self.assertEqual(h.engine.order,out.decision.order)

    def test_T02_bad_geometry_skipped_single_valid_candidate_dispatched(self):
        h=Flow();h.board();h.add_candidate();h.plans=(replace(h.plans[0],collision_free=c.Fact.UNKNOWN),)
        out=h.finish();self.assertIsNone(out.decision.order)
        self.assertEqual(out.directive.stage,Stage.OBSERVE)

    def test_unknown_failed_timedout_cancelled_do_not_advance_or_replace_order(self):
        for status in (c.Outcome.UNKNOWN,c.Outcome.FAILED,c.Outcome.TIMED_OUT,c.Outcome.CANCELLED):
            h=Flow();h.board();h.add_candidate();h.finish();order=h.engine.order
            out=h.finish(status,c.Fact.UNKNOWN)
            self.assertIsNone(out.directive);self.assertEqual(out.decision.phase,m.MissionPhase.RECOVERY)
            self.assertEqual(h.engine.order,order);self.assertFalse(h.engine.cake_closed)

    def test_echo_success_does_not_prove_grasp(self):
        h=Flow();h.board();h.add_candidate();h.finish()
        out=h.finish(physical=c.Fact.UNKNOWN)
        self.assertIsNone(out.directive);self.assertFalse(h.engine.cake_closed)

    def test_stale_action_and_duplicate_start_do_not_reissue(self):
        h=Flow();first=h.step();h.events=(m.ActionEvent(replace(first.directive.envelope,producer=c.Owner.P03,action_id='old'),s.timeout_result()),)
        out=h.step();self.assertIsNone(out.directive)
        self.assertEqual(h.engine.pending,first.directive)
        h.events=();self.assertIsNone(h.step().directive)

    def test_T08_budget_insufficient_blocks_new_grasp(self):
        h=Flow();h.board();h.add_candidate();h.clock.advance(579.)
        out=h.step();self.assertIsNone(out.directive)
        self.assertIsNone(h.engine.order)

    def test_deadline_stops_without_cleanup_commands(self):
        h=Flow();h.step();h.clock.advance(590.)
        out=h.step();self.assertEqual(out.decision.phase,m.MissionPhase.STOPPED)
        self.assertIsNone(out.directive)

    def test_missing_candidates_not_complete_and_bounded_observation(self):
        h=Flow();h.board();h.finish()
        for _ in range(3):out=h.finish()
        self.assertEqual(out.decision.delivery_complete,c.Fact.REJECTED)
        self.assertEqual(out.decision.phase,m.MissionPhase.STOPPED)
        self.assertIn('candidate',out.decision.end_reason.detail)

    def test_unknown_holding_or_space_blocks_grasp(self):
        for which in ('holding','space'):
            h=Flow();h.board();h.add_candidate()
            if which=='holding':h.bump(holding=c.HoldingSnapshot(c.Holding.UNKNOWN,None,None,()))
            else:h.bump(space=replace(h.snapshot.space,occupancies=s.space_snapshot().occupancies))
            out=h.finish();self.assertIsNone(out.directive)
            self.assertIsNone(h.engine.order)

    def test_T08_exit_once_survives_return_and_repeat(self):
        h=Flow();h.step()
        ex=m.ExitEvidence(h.env(c.Owner.P03,target_id=None,action_id=None,event_id='exit'),m.Area.A,c.Fact.CONFIRMED,(proof(),))
        h.exits=(ex,);h.step();h.exits=(replace(ex,envelope=replace(ex.envelope,event_id='exit-again')),)
        h.step();self.assertEqual(h.engine.exit_points,1)
        h.exits=();h.step();self.assertEqual(h.engine.exit_points,1)

    def test_checkpoint_failure_no_directive_published(self):
        h=Flow();h.journal.fail=True
        out=h.step();self.assertIsNone(out.directive)
        self.assertEqual(out.decision.phase,m.MissionPhase.RECOVERY)

    def test_C0_facade_missing_stage_bindings_does_not_make_them_up(self):
        h=Flow();decision=h.engine.step(h.frame().inputs)
        self.assertIsNone(decision.order)
        self.assertIsNotNone(decision.end_reason)


if __name__=='__main__':unittest.main()
