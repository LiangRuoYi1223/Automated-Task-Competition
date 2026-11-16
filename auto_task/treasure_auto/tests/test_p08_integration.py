"""Actual P06/P07/P10B code with simulated observations and local temporary DB."""
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from test_p08_support import *
from treasure_auto.contracts import rules as r
from treasure_auto.placement.evidence import ReleaseWindow
from treasure_auto.placement.space import SpaceProposals


class IntegrationTests(unittest.TestCase):
    def store(self,h,snapshot):
        from treasure_auto.persistence import PersistenceConfig,SQLitePersistence
        tmp=tempfile.TemporaryDirectory(prefix='p08-p10b-');self.addCleanup(tmp.cleanup)
        db=SQLitePersistence(PersistenceConfig(str(Path(tmp.name)/'runtime'),16,3.,1.,4096),h.clock,remaining=r.remaining_seconds)
        self.addCleanup(db.close)
        signal=m.RuntimeSignal(replace(snapshot.run.envelope,event_id=snapshot.run.start_signal_id),m.SignalKind.START,c.Fact.CONFIRMED)
        receipt=db.initialize_run(m.InitializeRunRequest(snapshot.run,snapshot,signal,None))
        self.assertEqual(receipt.status,m.CommitStatus.COMMITTED,receipt.reason)
        return db

    def test_P08_metric_release_real_P07_and_P10B_atomic_arrival_once(self):
        from test_p07_support import Harness
        h=Harness();h.reach('transport');h.confirm_stage();h.service.tick()
        req=request();order=h.request('place-root').order
        e=h.env(c.Owner.P09,action_id='place-root')
        req=replace(req,envelope=e,order=order,region=region(m.Area.A),holding=h.service.holding)
        previous=replace(s.state_snapshot(),holding=h.service.holding)
        db=self.store(h,previous);h.service.persistence=db
        start=h.clock.now().seconds;window=ReleaseWindow(req,config(),h.clock.now())
        event=None
        for dt,cup in ((0,(0,0,.1)),(.1,(0,0,.15)),(.35,(0,0,.15))):
            t=start+dt;h.clock.advance(t-h.clock.now().seconds)
            f=metric(t,cup=cup)
            ev=proof(t,action='place-root')
            f=replace(f,evidence=ev,objects=(replace(f.objects[0],area=m.Area.A,evidence=(ev,)),))
            event=window.push(f,replace(h.context,now=h.clock.now()),s.run_context())
        self.assertIsNotNone(event,window.last_reason)
        update=h.service.propose_release(event,h.service.holding)
        self.assertEqual(h.service.holding.state,c.Holding.HELD)
        proposed=SpaceProposals(config()).observe(previous.space,f,replace(h.context,now=h.clock.now()))
        tx=m.StateTransaction(replace(event.envelope,producer=c.Owner.P09,event_id='p09-coordinate'),'p08-atomic',previous.revision,
            previous.space.version,proposed,update.current,(replace(previous.inventory[0],location=m.InventoryLocation.REGION,area=m.Area.A),),
            previous.layouts,previous.mapping,(event,),(),update,previous.budgets,None)
        receipt=db.commit(tx);self.assertEqual(receipt.status,m.CommitStatus.COMMITTED,receipt.reason)
        self.assertEqual(db.commit(tx),receipt)
        h.service.accept_holding_commit(receipt)
        snap=db.read('sim-run-1').snapshot
        self.assertEqual(len(snap.deliveries),1);self.assertEqual(snap.holding.state,c.Holding.EMPTY)
        self.assertEqual(snap.space.occupancies[0].state,m.OccupancyState.ACTUAL)
        self.assertIsNone(h.service.readiness)

    def test_C0_conservative_unknown_downgrade_is_explicit_integration_gate(self):
        from contract_fixtures.fakes import FakeClock,FakeStateStore
        snap=s.state_snapshot();reserved=replace(occupancy(state=m.OccupancyState.RESERVED),evidence=())
        snap=replace(snap,space=space((reserved,)))
        proposal=SpaceProposals(config()).mark_unknown(snap.space,env(),'brick-1',s.stamp(110.1))
        tx=m.StateTransaction(env(c.Owner.P09,emitted_at=s.stamp(110.1)),'unknown-space',0,0,proposal,snap.holding,
            snap.inventory,snap.layouts,snap.mapping,(),(),None,snap.budgets,None)
        store=FakeStateStore(snap,FakeClock(s.stamp(110.1)));receipt=store.commit(tx)
        self.assertEqual(receipt.status,m.CommitStatus.REJECTED)
        self.assertEqual(store.read('sim-run-1').snapshot.space.occupancies[0].state,m.OccupancyState.RESERVED)
        self.assertEqual(proposal.occupancies[0].state,m.OccupancyState.UNKNOWN)


if __name__=='__main__':unittest.main()
