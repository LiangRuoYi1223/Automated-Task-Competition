import unittest
from dataclasses import replace
from test_p08_support import *
from test_p08_observation import release_frame
from contract_fixtures.fakes import FakeClock
from treasure_auto.placement.service import PlacementService
from treasure_auto.placement.inputs import PlacementBinding


class ManipulationBoundary:
    def __init__(self):self.calls=0
    def propose_release(self,event,holding):
        self.calls+=1
        return m.HoldingUpdate(replace(event.envelope,producer=c.Owner.P07),holding.state,
            c.HoldingSnapshot(c.Holding.EMPTY,None,None,event.evidence),m.HoldingCause.RELEASE)
    def accept_holding_commit(self,receipt):return None


class ServiceTests(unittest.TestCase):
    def make(self,reserved=True):
        self.clock=FakeClock(s.stamp(110));self.req=request();self.target=occupancy(state=m.OccupancyState.RESERVED)
        self.state=replace(s.state_snapshot(),space=space((self.target,) if reserved else ()),holding=self.req.holding,
            inventory=(m.InventoryItem('inventory-1',('brick-1',),c.Fact.CONFIRMED,m.Category.DOLL,m.InventoryLocation.HELD,None),),layouts=(layout(),))
        self.manip=ManipulationBoundary()
        class ExecutedBoundary:
            state='await_release';completed_phases=[m.Phase.PLACE,m.Phase.WITHDRAW];reason=None
            def start(self,req):return self.state
            def cancel(self,detail):self.state='unresolved'
            def finish_committed(self,receipt,snapshot):self.state='idle'
        self.execution=ExecutedBoundary()
        return PlacementService(config(),s.run_context(),self.clock,context_provider=lambda:context(self.clock.now().seconds),
            state_provider=lambda:self.state,binding_provider=lambda req,snap:PlacementBinding(self.target,req.holding.tool_offset,req.plan.path_id,
                c.MapPose('map',-.5,0,0,'m','rad',0,0),(proof(self.clock.now().seconds,target=None,action=None),)),
            manipulation=self.manip,executor=self.execution)

    def test_place_without_committed_reservation_only_proposes(self):
        svc=self.make(False);out=svc.place(self.req)
        self.assertEqual(out.result.status,c.Outcome.UNKNOWN)
        self.assertEqual(out.result.phase,'await_reservation_commit')
        self.assertEqual(out.proposed_space.version,1);self.assertIsNone(svc.active_request)

    def test_frozen_mapping_content_offset_and_old_context_rejected(self):
        for bad in ('mapping','offset','run'):
            svc=self.make()
            if bad=='mapping':self.req=replace(self.req,mapping=s.mapping_samples()[1])
            elif bad=='offset':svc.binding_provider=lambda req,snap:PlacementBinding(self.target,c.ArmPoint('tool',99,0,0,'mm',0),req.plan.path_id,
                c.MapPose('map',-.5,0,0,'m','rad',0,0),(proof(),))
            else:self.req=replace(self.req,envelope=replace(self.req.envelope,run_id='old'))
            out=svc.place(self.req);self.assertEqual(out.result.status,c.Outcome.REJECTED)

    def test_release_precedes_layout_and_empty_is_only_P07_proposal(self):
        svc=self.make();svc.place(self.req)
        for t,cup in ((110,(0,0,.1)),(110.1,(0,0,.15)),(110.35,(0,0,.15))):
            self.clock.advance(t-self.clock.now().seconds);out=svc.observe_metrics(release_frame(t,cup))
        self.assertEqual(out.result.status,c.Outcome.SUCCESS)
        self.assertIsNotNone(out.release);self.assertEqual(out.layout.confirmed_count,0)
        self.assertEqual(out.proposed_holding.state,c.Holding.EMPTY)
        self.assertEqual(self.state.holding.state,c.Holding.HELD)
        self.assertEqual(self.manip.calls,1)
        svc.observe_metrics(release_frame(110.4));self.assertEqual(self.manip.calls,1)
        self.assertFalse(svc.return_permitted)

    def test_pixel_only_C0_observation_never_creates_map_or_stillness(self):
        svc=self.make();observation=s.all_samples()['perception']
        with self.assertRaises(c.ContractError):svc.propose_space(observation,self.state.space)
        self.assertIsNone(svc.assess_stillness((observation,),self.req.stop,self.state.space))
        out=svc.assess_layout(self.state.space,replace(layout(),confirmed_count=1,bricks=(brick(),),quality=c.Fact.CONFIRMED))
        self.assertEqual(out.confirmed_count,1);self.assertEqual(out.quality,c.Fact.UNKNOWN)

    def test_missing_stage_adapter_and_deadline_do_not_claim_success(self):
        svc=self.make();svc.executor=None
        self.assertEqual(svc.place(self.req).result.reason.code,c.ErrorCode.UNAVAILABLE)
        svc=self.make();self.clock.advance(600)
        self.assertEqual(svc.place(self.req).result.status,c.Outcome.REJECTED)

    def test_commit_unknown_then_atomic_readback_ends_cycle_without_return_permission(self):
        from contract_fixtures.fakes import FakeStateStore
        svc=self.make();svc.place(self.req)
        for t,cup in ((110,(0,0,.1)),(110.1,(0,0,.15)),(110.35,(0,0,.15))):
            self.clock.advance(t-self.clock.now().seconds);out=svc.observe_metrics(release_frame(t,cup))
        store=FakeStateStore(self.state,self.clock);svc.persistence=store
        unknown=m.CommitReceipt('tx',m.CommitStatus.UNKNOWN,None,None,c.Reason(c.ErrorCode.STORAGE_FAILURE,'commit','unknown'))
        with self.assertRaises(c.ContractError):svc.accept_commit('tx',unknown)
        self.assertIsNotNone(svc.active_request)
        tx=m.StateTransaction(replace(out.release.envelope,producer=c.Owner.P09),'tx',0,0,out.proposed_space,out.proposed_holding,
            (replace(self.state.inventory[0],location=m.InventoryLocation.REGION,area=m.Area.B),),self.state.layouts,self.state.mapping,
            (out.release,),(),svc.pending_holding,self.state.budgets,None)
        receipt=store.commit(tx);self.assertEqual(receipt.status,m.CommitStatus.COMMITTED,receipt.reason)
        result=svc.accept_commit('tx',receipt)
        self.assertEqual(result.result.phase,'release_committed');self.assertIsNone(svc.active_request)
        self.assertFalse(svc.return_permitted)


if __name__=='__main__':unittest.main()
