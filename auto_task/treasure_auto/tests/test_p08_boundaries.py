import unittest
from dataclasses import replace
from test_p08_support import *
import test_p08_execution as execution_fixtures
import test_p08_service as service_fixtures
import test_p08_recovery as recovery_fixtures
import test_p08_observation as observation_fixtures
from treasure_auto.placement.layout import LayoutTracker
from treasure_auto.placement.inputs import PlacementBinding


class BoundaryTests(unittest.TestCase):
    def test_expired_stage_plan_after_prepare_never_outputs(self):
        h=execution_fixtures.Harness();planner=h.executor.stage_planner
        def short_plan(*args):
            plan=planner(*args)
            return replace(plan,envelope=replace(plan.envelope,expires_at=s.stamp(110.1)))
        h.executor.stage_planner=short_plan;h.on_prepare=lambda:h.clock.advance(.2)
        h.executor.start(h.req);self.assertEqual(h.commands,[])

    def test_remaining_budget_must_cover_whole_bounded_stage(self):
        h=execution_fixtures.Harness();h.req=replace(h.req,order=replace(h.req.order,budget=m.Budget('recovery-B',3,1.,0,0)))
        h.executor.start(h.req);self.assertEqual(h.commands,[])

    def test_duplicate_old_physical_feedback_does_not_cancel_new_stage(self):
        h=execution_fixtures.Harness();h.executor.start(h.req);command=h.commands[-1];h.clock.advance(.05)
        event=m.ActionEvent(replace(command.envelope,producer=c.Owner.P01,emitted_at=h.clock.now(),event_id='completion-1'),
            c.ActionResult(c.Outcome.SUCCESS,'observed',None,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,
                           (proof(110.05,action=command.envelope.action_id),),.05))
        h.executor.confirm(event);h.executor.tick();self.assertEqual(len(h.commands),2)
        h.executor.confirm(event);self.assertEqual(h.executor.state,'waiting');self.assertEqual(len(h.commands),2)

    def test_committed_cycle_allows_next_order_but_keeps_area_budget(self):
        h=execution_fixtures.Harness();h.req=replace(h.req,order=replace(h.req.order,budget=m.Budget('recovery-B',1,30,0,0)))
        h.executor.start(h.req)
        for _ in range(5):h.complete();h.executor.tick()
        arrival=m.DeliveryReceipt(m.DeliveryKey('sim-run-1','inventory-1'),'release-1',m.Category.DOLL,m.Area.B,s.stamp(110.25),('proof',))
        snap=replace(s.state_snapshot(),holding=c.HoldingSnapshot(c.Holding.EMPTY,None,None,()),revision=1,deliveries=(arrival,))
        receipt=m.CommitReceipt('linked',m.CommitStatus.COMMITTED,1,0,None)
        h.executor.finish_committed(receipt,snap)
        self.assertEqual(h.executor.state,'idle')
        new=replace(h.req,envelope=replace(h.req.envelope,action_id='second',event_id='second'),
            order=replace(h.req.order,budget=m.Budget('recovery-B',1,30,0,0)))
        h.ctx=replace(h.ctx,action_id='second')
        h.executor.start(new)
        self.assertEqual(len(h.commands),5,'old cumulative attempt cannot be erased by new round/order')

    def test_same_geometry_new_event_cannot_count_as_new_replacement(self):
        tr=LayoutTracker(layout(m.LayoutKind.V1,count=1,bricks=(brick(kind=m.LayoutKind.V1),)),config())
        sp=space((occupancy(),occupancy('brick-2',x=.105)))
        with self.assertRaises(c.ContractError):
            tr.record_placement(release('brick-2'),sp,recovery_fixtures.facts('brick-2'),region(),context())

    def test_unlisted_moving_brick_cannot_be_ignored_in_stillness(self):
        w=observation_fixtures.still_window()
        for t in (110,110.25,110.5):
            result=w.push(metric(t,ids=('brick-1','brick-2')),context(t))
        self.assertIsNone(result)

    def test_late_pre_reset_exposure_cannot_restart_static_window(self):
        w=observation_fixtures.still_window();w.push(metric(110),context())
        w.push(metric(110.4,scene=1),context(110.4))
        for t in (110.2,110.45,110.7):result=w.push(metric(t),context(max(t,110.4)))
        self.assertIsNone(result)

    def test_target_binding_must_match_actual_current_support_height(self):
        h=service_fixtures.ServiceTests();svc=h.make()
        tr=occupancy('brick-1',z=0)
        second=occupancy('brick-2',z=.2,state=m.OccupancyState.RESERVED,supports=('brick-1',))
        req=replace(h.req,envelope=replace(h.req.envelope,target_id='brick-2'),order=replace(h.req.order,envelope=replace(h.req.order.envelope,target_id='brick-2')),
            holding=replace(h.req.holding,target_id='brick-2',evidence=(proof(target='brick-2'),)),
            plan=replace(h.req.plan,envelope=replace(h.req.plan.envelope,target_id='brick-2'),evidence=(proof(target='brick-2'),)),
            permit=replace(h.req.permit,envelope=replace(h.req.permit.envelope,target_id='brick-2')))
        h.state=replace(h.state,holding=req.holding,space=space((tr,second)),layouts=(layout(count=1,bricks=(brick(),)),))
        svc.context_provider=lambda:context(target_id='brick-2')
        svc.binding_provider=lambda request,state:PlacementBinding(second,request.holding.tool_offset,request.plan.path_id,
            c.MapPose('map',-.5,0,0,'m','rad',0,0),(proof(target=None,action=None),))
        out=svc.place(req)
        self.assertEqual(out.result.status,c.Outcome.REJECTED)
        self.assertEqual(out.result.reason.field,'target.geometry')


if __name__=='__main__':unittest.main()
