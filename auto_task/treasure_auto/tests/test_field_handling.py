"""Field policy tests use simulated physical facts, never device commands."""
import json
import unittest
from dataclasses import replace
from test_p08_support import c, m, s, occupancy, space, layout, brick, config, context, region
from test_p08_recovery import facts, failure, support_for
from treasure_auto.placement.layout import LayoutTracker

try:
    from treasure_auto.manipulation.strategy import HandlingPolicy, HandlingObservation
except ImportError:
    HandlingPolicy = HandlingObservation = None


class HandlingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(HandlingPolicy, 'field handling policy must exist')
        self.policy = HandlingPolicy()
        self.time = 0

    def event(self, phase, holding=c.Holding.EMPTY, **kw):
        self.time += 1
        args = dict(target_id='brick-1', event_id='e-'+str(self.time), observed_at_s=float(self.time),
                    phase=phase, holding=holding, visible=c.Fact.CONFIRMED,
                    neighbors_stable=c.Fact.CONFIRMED, action_id='a-'+str((self.time+1)//2))
        args.update(kw)
        return HandlingObservation(**args)

    def send(self, phase, holding=c.Holding.EMPTY, **kw):
        return self.policy.observe(self.event(phase, holding, **kw))

    def test_third_failed_probe_skips_without_claiming_negative_identity(self):
        for n in range(3):
            start=self.send('probe_start', action_id='probe-'+str(n))
            self.assertEqual(start.action, 'probe')
            out=self.send('probe_result', independent_lift=c.Fact.REJECTED, action_id='probe-'+str(n))
        self.assertEqual((out.action,out.probe_attempts,out.skip_this_run),('skip_target',3,True))
        self.assertEqual(self.send('probe_start').probe_attempts,3)
        self.assertFalse(out.independent_confirmed)

    def test_neighbor_motion_stops_immediately_even_if_holding_unknown(self):
        self.send('probe_start',action_id='probe')
        out=self.send('probe_result', c.Holding.UNKNOWN, action_id='probe', neighbors_stable=c.Fact.REJECTED)
        self.assertEqual((out.action,out.probe_attempts,out.skip_this_run),('stop_reobserve',1,True))
        self.assertEqual(self.send('probe_start').action,'stop_reobserve')
        out=self.send('resolve_stop',stopped=c.Fact.CONFIRMED)
        self.assertEqual(out.action,'skip_target')
        self.assertEqual(self.policy.pending_targets(),())

    def test_success_requires_independent_lift_following_neighbors_and_measured_extent(self):
        self.send('probe_start',action_id='probe')
        out=self.send('probe_result',c.Holding.HELD,action_id='probe',independent_lift=c.Fact.CONFIRMED,
                      following=c.Fact.UNKNOWN,measured_offset=c.ArmPoint('tool',12,2,0,'mm',3))
        self.assertEqual(out.action,'stop_reobserve')
        out=self.send('probe_result',c.Holding.HELD,action_id='probe',independent_lift=c.Fact.CONFIRMED,
                      following=c.Fact.CONFIRMED,measured_offset=c.ArmPoint('tool',12,2,0,'mm',3))
        self.assertEqual(out.action,'temporary_place')
        self.assertTrue(out.independent_confirmed)
        self.assertTrue(out.bias)

    def test_pending_start_duplicate_stale_conflict_and_restart_cannot_refund_or_resend(self):
        start=self.event('probe_start',action_id='probe')
        self.assertEqual(self.policy.observe(start).action,'probe')
        self.assertEqual(self.policy.observe(start).action,'await_observation')
        with self.assertRaises(c.ContractError):self.policy.observe(replace(start,holding=c.Holding.UNKNOWN))
        with self.assertRaises(c.ContractError):self.policy.observe(replace(start,event_id='other',observed_at_s=0))
        self.policy=HandlingPolicy.restore(json.loads(json.dumps(self.policy.checkpoint())))
        self.assertEqual(self.policy.observe(start).action,'stop_reobserve')
        out=self.send('probe_start',action_id='new')
        self.assertEqual((out.action,out.probe_attempts),('stop_reobserve',1))

    def test_conflicting_target_event_cannot_create_identity_and_checkpoint_cannot_refund(self):
        start=self.event('probe_start',action_id='probe')
        self.policy.observe(start)
        with self.assertRaises(c.ContractError):self.policy.observe(replace(start,target_id='renamed'))
        self.assertIsNone(self.policy.decision_for('renamed'))
        state=self.policy.checkpoint();state['targets']['brick-1']['probe_attempts']=0
        with self.assertRaises(c.ContractError):HandlingPolicy.restore(state)

    def test_three_recenters_keep_bias_actual_offset_and_separate_probe_counter(self):
        self.send('probe_start',action_id='probe')
        offset=c.ArmPoint('tool',12,2,0,'mm',3)
        self.send('probe_result',c.Holding.HELD,action_id='probe',independent_lift=c.Fact.CONFIRMED,
                  following=c.Fact.CONFIRMED,measured_offset=offset)
        for n in range(3):
            self.send('temporary_release')
            out=self.send('recenter_start',action_id='recenter-'+str(n))
            self.assertEqual(out.action,'recenter')
            out=self.send('recenter_result',c.Holding.HELD,action_id='recenter-'+str(n),
                          independent_lift=c.Fact.CONFIRMED,following=c.Fact.CONFIRMED,
                          offset_verified=c.Fact.REJECTED,measured_offset=offset)
        self.assertEqual((out.action,out.probe_attempts,out.recenter_attempts),('approximate_place',1,3))
        self.assertTrue(out.bias)
        self.assertEqual(out.measured_offset,offset)
        self.policy=HandlingPolicy.restore(json.loads(json.dumps(self.policy.checkpoint())))
        self.send('temporary_release')
        out=self.send('recenter_start',action_id='local-repair')
        self.assertEqual(out.recenter_attempts,3)
        self.assertNotEqual(out.action,'recenter')

    def test_regrasp_needs_release_empty_and_actual_verification_clears_bias(self):
        self.assertEqual(self.send('recenter_start',c.Holding.HELD).action,'stop_reobserve')
        self.assertEqual(self.send('recenter_start').action,'stop_reobserve')
        self.send('temporary_release')
        self.send('recenter_start',action_id='recenter')
        out=self.send('recenter_result',c.Holding.HELD,action_id='recenter',
                      independent_lift=c.Fact.CONFIRMED,following=c.Fact.CONFIRMED,
                      offset_verified=c.Fact.CONFIRMED,measured_offset=c.ArmPoint('tool',.1,0,0,'mm',.1))
        self.assertEqual(out.action,'transport')
        self.assertFalse(out.bias)

    def test_unsuccessful_empty_recenter_can_retry_without_fabricating_another_release(self):
        self.send('temporary_release')
        self.assertEqual(self.send('recenter_start',action_id='r1').action,'recenter')
        out=self.send('recenter_result',action_id='r1',independent_lift=c.Fact.REJECTED)
        self.assertEqual((out.action,out.recenter_attempts),('await_observation',1))
        out=self.send('recenter_start',action_id='r2')
        self.assertEqual((out.action,out.recenter_attempts),('recenter',2))
        self.assertEqual(self.policy.pending_targets(),('brick-1',))

    def test_three_empty_recenter_failures_never_invent_holding_or_grant_fourth_attempt(self):
        self.send('temporary_release')
        for n in range(1,4):
            out=self.send('recenter_start',action_id='r'+str(n))
            self.assertEqual((out.action,out.recenter_attempts),('recenter',n))
            out=self.send('recenter_result',action_id='r'+str(n),independent_lift=c.Fact.REJECTED)
            self.assertEqual(out.action,'await_observation')
            self.assertIsNone(out.measured_offset)
        self.policy=HandlingPolicy.restore(json.loads(json.dumps(self.policy.checkpoint())))
        out=self.send('recenter_start',action_id='r4')
        self.assertEqual((out.action,out.recenter_attempts),('await_observation',3))
        self.assertTrue(out.bias)
        self.assertFalse(out.independent_confirmed)
        self.assertEqual(self.policy.pending_targets(),())

    def test_read_only_lookup_does_not_reset_target_or_create_missing_identity(self):
        self.assertIsNone(self.policy.decision_for('missing'))
        out=self.send('probe_start',action_id='probe')
        self.assertEqual(self.policy.decision_for('brick-1'),out)
        self.assertNotIn('missing',self.policy.checkpoint()['targets'])
        self.assertEqual(self.policy.pending_targets(),('brick-1',))
        self.send('probe_result',action_id='probe',independent_lift=c.Fact.REJECTED)
        self.assertEqual(self.policy.pending_targets(),())


class TopCorrectionTests(unittest.TestCase):
    def setup_scene(self):
        low=occupancy();mid=occupancy('brick-2',z=.1,supports=('brick-1',))
        top=occupancy('brick-3',z=.2,supports=('brick-2',))
        quality=replace(facts('brick-3'),horizontal=c.Fact.REJECTED,prior_facts=(
            facts(),facts('brick-2',support=support_for(mid,(low,)))))
        return LayoutTracker(layout(count=3,bricks=(brick(),brick(2),brick(3))),config()),space((low,mid,top)),quality

    def correct(self,tr,sp,f,target='brick-3'):
        from test_p08_support import proof
        event=failure()
        event=replace(event,envelope=replace(event.envelope,target_id=target),
                      result=replace(event.result,evidence=(proof(target=target),)))
        return tr.correct_top(event,target,sp,f,region(),c.HoldingSnapshot(c.Holding.EMPTY,None,None,()),context(target_id=target))

    def test_top_only_retains_fresh_valid_prefix_and_does_not_clear_occupancy(self):
        tr,sp,f=self.setup_scene();before=c.dumps(sp)
        out=self.correct(tr,sp,f)
        self.assertEqual((out.confirmed_count,out.layout_recovery_round),(2,0))
        self.assertEqual(tuple(b.target_id for b in out.bricks),('brick-1','brick-2'))
        self.assertEqual(out.quality,c.Fact.UNKNOWN)
        self.assertEqual(c.dumps(sp),before)
        self.assertEqual(self.correct(tr,sp,f),out)

    def test_loaded_lower_never_can_be_directly_removed(self):
        tr,sp,f=self.setup_scene()
        with self.assertRaises(c.ContractError):self.correct(tr,sp,f,'brick-1')
        self.assertEqual(tr.layout.confirmed_count,3)

    def test_lower_failure_routes_full_rebuild_but_missing_evidence_preserves_count(self):
        tr,sp,f=self.setup_scene()
        bad=replace(f,prior_facts=(replace(f.prior_facts[0],stable=c.Fact.REJECTED),f.prior_facts[1]))
        out=self.correct(tr,sp,bad)
        self.assertEqual((out.confirmed_count,out.layout_recovery_round),(0,1))
        tr,sp,f=self.setup_scene()
        out=self.correct(tr,sp,replace(f,prior_facts=()))
        self.assertEqual(out.confirmed_count,3)
        self.assertEqual(out.quality,c.Fact.UNKNOWN)

    def test_A_cake_has_no_inherited_lower_or_three_brick_progress(self):
        tr=LayoutTracker(replace(layout(count=1,bricks=(brick(),)),area=m.Area.A),config())
        item=occupancy(area=m.Area.A)
        out=tr.correct_top(failure(),'brick-1',space((item,)),replace(facts(),horizontal=c.Fact.REJECTED),
                           region(m.Area.A),c.HoldingSnapshot(c.Holding.EMPTY,None,None,()),context())
        self.assertEqual((out.confirmed_count,out.layout_recovery_round),(0,0))

    def test_unknown_lower_pauses_while_actual_off_center_lower_resets(self):
        tr,sp,f=self.setup_scene()
        unknown=replace(f,prior_facts=(replace(f.prior_facts[0],horizontal=c.Fact.UNKNOWN),f.prior_facts[1]))
        self.assertEqual(self.correct(tr,sp,unknown).confirmed_count,3)
        tr,sp,f=self.setup_scene()
        bad=replace(sp,occupancies=(occupancy(x=.03),)+sp.occupancies[1:])
        out=self.correct(tr,bad,f)
        self.assertEqual((out.confirmed_count,out.layout_recovery_round),(0,1))

    def test_stale_bad_lower_or_unconfirmed_top_must_not_revoke_progress(self):
        tr,sp,f=self.setup_scene()
        stale=replace(f.prior_facts[0],stable=c.Fact.REJECTED,evidence=facts(t=105).evidence)
        out=self.correct(tr,sp,replace(f,prior_facts=(stale,f.prior_facts[1])))
        self.assertEqual(out.confirmed_count,3)
        tr,sp,f=self.setup_scene()
        out=self.correct(tr,sp,replace(f,horizontal=c.Fact.UNKNOWN))
        self.assertEqual(out.confirmed_count,3)

    def test_unknown_nested_support_facts_preserve_progress_pending_reobservation(self):
        for field in ('contact','horizontal'):
            with self.subTest(field=field):
                tr,sp,f=self.setup_scene()
                lower=f.prior_facts[1]
                lower=replace(lower,support=replace(lower.support,**{field:c.Fact.UNKNOWN}))
                original=tr.layout.bricks
                out=self.correct(tr,sp,replace(f,prior_facts=(f.prior_facts[0],lower)))
                self.assertEqual((out.confirmed_count,out.layout_recovery_round),(3,0))
                self.assertEqual(out.bricks,original)
                self.assertEqual(out.quality,c.Fact.UNKNOWN)
                self.assertTrue(tr.requires_stop)

    def test_explicit_rejected_nested_support_still_routes_whole_layout_failure(self):
        for field in ('contact','horizontal'):
            with self.subTest(field=field):
                tr,sp,f=self.setup_scene()
                lower=f.prior_facts[1]
                lower=replace(lower,support=replace(lower.support,**{field:c.Fact.REJECTED}))
                out=self.correct(tr,sp,replace(f,prior_facts=(f.prior_facts[0],lower)))
                self.assertEqual((out.confirmed_count,out.layout_recovery_round),(0,1))
                self.assertEqual(out.quality,c.Fact.REJECTED)


class ApproximateGeometryTests(unittest.TestCase):
    def test_explicit_nominal_center_keeps_actual_offset_in_collision_queries(self):
        from test_p06_paths import PathFixture
        fixture=PathFixture();fixture.setUp()
        req=fixture.placement(offset=(2.,1.,0.))
        fixture.ctx=replace(fixture.ctx,approximate_center=True)
        result=fixture.svc.plan_placement(req)
        self.assertIsNotNone(result.plan,result.result.reason)
        self.assertAlmostEqual(result.plan.target.x,196.)
        self.assertAlmostEqual(result.plan.target.y,203.)
        self.assertAlmostEqual(result.plan.target.z,150.)
        self.assertGreaterEqual(result.plan.target.error_mm,req.held_offset.error_mm+(5.**.5))
        self.assertTrue(all(q.held_offset==req.held_offset for q in fixture.calls if q.carrying))
        self.assertEqual(fixture.ctx.holding.tool_offset,req.held_offset)
        self.assertTrue(fixture.audit[-1]['diagnostics']['approximate_center'])
        fixture.ctx=replace(fixture.ctx,approximate_center=False)
        self.assertFalse(fixture.svc.validate_plan(result.plan).accepted)

    def test_approximate_center_does_not_bypass_contact_uncertainty_or_collision(self):
        from test_p06_paths import PathFixture
        fixture=PathFixture();fixture.setUp()
        req=fixture.placement(offset=(100.,0.,0.))
        fixture.ctx=replace(fixture.ctx,approximate_center=True)
        result=fixture.svc.plan_placement(req)
        self.assertIsNone(result.plan)
        self.assertEqual(fixture.ctx.holding.tool_offset,req.held_offset)
        fixture.setUp();req=fixture.placement(offset=(2.,1.,0.))
        fixture.ctx=replace(fixture.ctx,approximate_center=True)
        fixture.cert_override=lambda cert:replace(cert,static_clearance=c.Fact.REJECTED)
        self.assertIsNone(fixture.svc.plan_placement(req).plan)


class TemporaryPlacementTests(unittest.TestCase):
    def make(self,area=m.Area.B):
        from test_p08_service import ServiceTests
        from test_p08_support import square
        from treasure_auto.navigation.field_regions import FieldRegions
        fixture=ServiceTests();svc=fixture.make()
        centers={'A':2.,'B':0.,'C':4.,'D':6.}
        full={a:square(x,0,1) for a,x in centers.items()}
        collection={a:square(x,0,.45) for a,x in centers.items()}
        self.regions=FieldRegions('map',square(3,0,10),square(3,2,1),full,collection)
        self.x=centers[area.value]+.35;self.y=.35
        self.fixture=fixture;self.svc=svc
        target=replace(occupancy(x=self.x,y=self.y,area=area),state=m.OccupancyState.RESERVED)
        fixture.target=target
        req=replace(fixture.req,region=replace(region(area),polygon_m=full[area.value]))
        if area is m.Area.A:req=replace(req,order=replace(req.order,category=m.Category.CAKE,destination=area))
        fixture.req=req
        fixture.state=replace(fixture.state,space=space((target,)),layouts=(replace(layout(),area=area),),
            inventory=(replace(fixture.state.inventory[0],category=req.order.category),))
        return svc,req

    def start(self,**kwargs):
        return self.svc.place_temporary(self.fixture.req,self.regions,free_observed=True,
            approach_clear=True,regrasp_clear=True,**kwargs)

    def frames(self,x=None,y=None):
        from test_p08_observation import release_frame
        xx=self.x if x is None else x;yy=self.y if y is None else y
        result=None
        for t,z in ((110,.1),(110.1,.15),(110.35,.15)):
            self.fixture.clock.advance(t-self.fixture.clock.now().seconds)
            frame=release_frame(t,(xx,yy,z))
            frame=replace(frame,objects=(occupancy(x=xx,y=yy,t=t,area=self.fixture.req.region.area),),
                          coverage_polygon_m=self.fixture.req.region.polygon_m)
            result=self.svc.observe_metrics(frame)
        return result

    def test_transport_release_still_requires_real_separation_and_atomic_commit(self):
        from contract_fixtures.fakes import FakeStateStore
        svc,req=self.make();out=self.start()
        self.assertEqual(out.result.phase,'await_release',out.result.reason)
        self.assertIsNone(out.release)
        self.assertEqual(self.fixture.state.holding.state,c.Holding.HELD)
        result=self.frames()
        self.assertEqual(result.result.status,c.Outcome.SUCCESS,result.result.reason)
        self.assertEqual(result.layout.confirmed_count,0)
        self.assertEqual(result.proposed_holding.state,c.Holding.EMPTY)
        self.assertEqual(self.fixture.state.holding.state,c.Holding.HELD)
        store=FakeStateStore(self.fixture.state,self.fixture.clock);svc.persistence=store
        tx=m.StateTransaction(replace(result.release.envelope,producer=c.Owner.P09),'temp-tx',0,0,
            result.proposed_space,result.proposed_holding,
            (replace(self.fixture.state.inventory[0],location=m.InventoryLocation.REGION,area=m.Area.B),),
            self.fixture.state.layouts,self.fixture.state.mapping,(result.release,),(),svc.pending_holding,
            self.fixture.state.budgets,None)
        receipt=store.commit(tx)
        self.assertEqual(receipt.status,m.CommitStatus.COMMITTED,receipt.reason)
        self.assertEqual(svc.accept_commit('temp-tx',receipt).result.phase,'release_committed')
        self.assertEqual(store.commit(tx),receipt)
        self.assertEqual(len(store.read(req.envelope.run_id).snapshot.deliveries),1)

    def test_A_transport_release_does_not_require_a_three_brick_layout(self):
        self.make(m.Area.A);self.assertEqual(self.start().result.phase,'await_release')
        out=self.frames();self.assertEqual(out.result.status,c.Outcome.SUCCESS,out.result.reason)
        self.assertEqual((out.layout.area,out.layout.confirmed_count),(m.Area.A,0))

    def test_unknown_ground_wrong_full_region_collection_or_obstacle_rejects(self):
        from test_p08_support import square
        svc,req=self.make()
        self.assertEqual(svc.place_temporary(req,self.regions).result.status,c.Outcome.REJECTED)
        self.make();self.fixture.req=replace(self.fixture.req,region=region())
        self.assertEqual(self.start().result.status,c.Outcome.REJECTED)
        self.make();self.fixture.target=replace(self.fixture.target,footprint_m=square())
        self.assertEqual(self.start().result.status,c.Outcome.REJECTED)
        self.make();self.fixture.state=replace(self.fixture.state,space=space((self.fixture.target,
            occupancy('other',x=self.x,y=self.y,state=m.OccupancyState.UNKNOWN))))
        self.assertEqual(self.start().result.status,c.Outcome.REJECTED)

    def test_actual_release_in_collection_cannot_masquerade_as_temporary_transport(self):
        self.make();self.start()
        out=self.frames(x=0.,y=0.)
        self.assertIsNone(out.release)
        self.assertEqual(self.fixture.manip.calls,0)


if __name__=='__main__':unittest.main()
