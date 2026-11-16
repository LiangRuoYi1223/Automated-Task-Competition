from publication_support import simulated_run_proposal
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from test_p09_support import *
from treasure_auto.mission.transactions import ReleaseCoordinator
from treasure_auto.mission.simulator import run_simulation


class SimulationTests(unittest.TestCase):
    def test_whole_match_fake_clock_all_layouts_and_arbitrary_categories(self):
        for kind,want in [('V1',21),('V2',27),('V3',42)]:
            result=run_simulation(kind)
            self.assertEqual(result['arrivals'],10,result)
            self.assertEqual(result['layout_complete'],'confirmed',result)
            self.assertEqual(result['conditional_points'],want,result)
            self.assertEqual(result['orders'][0],'cake')
            self.assertEqual(result['orders'][1:4],['banana','doll','clock'])
            self.assertEqual(len(result['orders']),10)
            self.assertTrue(result['trace'])

    def test_unknown_grasp_keeps_current_order_no_success_or_second_grasp(self):
        result=run_simulation('V1',fault='unknown_grasp')
        self.assertEqual(result['arrivals'],0)
        self.assertEqual(result['orders'],['cake'])
        self.assertEqual(result['phase'],'recovery')

    def test_deadline_stops_without_followup_action(self):
        result=run_simulation('V1',fault='deadline')
        self.assertEqual(result['arrivals'],0)
        self.assertEqual(result['phase'],'stopped')
        self.assertEqual(result['orders'],[])


class ActualBoundaryTests(unittest.TestCase):
    def test_P08_release_P07_proposal_P09_coordinator_real_P10B_atomic_and_replay(self):
        from test_p07_support import Harness
        from test_p08_support import request,region,metric,config,proof as p8proof
        from treasure_auto.placement.evidence import ReleaseWindow
        from treasure_auto.placement.space import SpaceProposals
        from treasure_auto.persistence import PersistenceConfig,SQLitePersistence
        h=Harness();h.reach('transport');h.confirm_stage();h.service.tick()
        order=h.request('place-root').order
        req=replace(request(),envelope=h.env(c.Owner.P09,action_id='place-root'),order=order,
                    region=region(m.Area.A),holding=h.service.holding)
        before=replace(s.state_snapshot(),holding=h.service.holding)
        tmp=tempfile.TemporaryDirectory(prefix='p09-p10b-');self.addCleanup(tmp.cleanup)
        db=SQLitePersistence(PersistenceConfig(str(Path(tmp.name)/'runtime'),16,3.,1.,4096),h.clock,remaining=r.remaining_seconds)
        self.addCleanup(db.close)
        signal=m.RuntimeSignal(replace(before.run.envelope,event_id=before.run.start_signal_id),m.SignalKind.START,c.Fact.CONFIRMED)
        self.assertEqual(db.initialize_run(m.InitializeRunRequest(before.run,before,signal,None)).status,m.CommitStatus.COMMITTED)
        h.service.persistence=db;window=ReleaseWindow(req,config(),h.clock.now());start=h.clock.now().seconds
        for dt,cup in ((0,(0,0,.1)),(.1,(0,0,.15)),(.35,(0,0,.15))):
            t=start+dt;h.clock.advance(t-h.clock.now().seconds);f=metric(t,cup=cup);ev=p8proof(t,action='place-root')
            f=replace(f,evidence=ev,objects=(replace(f.objects[0],area=m.Area.A,evidence=(ev,)),))
            release=window.push(f,replace(h.context,now=h.clock.now()),before.run)
        self.assertIsNotNone(release,window.last_reason)
        update=h.service.propose_release(release,h.service.holding)
        space=SpaceProposals(config()).observe(before.space,f,replace(h.context,now=h.clock.now()))
        lay=m.LayoutSnapshot(replace(release.envelope,target_id=None,action_id=None),m.Area.A,m.LayoutKind.V1,0,0,0,(),c.Fact.UNKNOWN,c.Fact.UNKNOWN)
        p=m.PlacementResult(release.envelope,s.timeout_result(),release,lay,space,update.current,None)
        coord=ReleaseCoordinator(db,h.clock,run_update_provider=simulated_run_proposal);out=coord.commit_release(order,p,update)
        self.assertEqual(out.receipt.status,m.CommitStatus.COMMITTED,out.receipt.reason)
        h.service.accept_holding_commit(out.receipt)
        self.assertEqual(h.service.holding.state,c.Holding.EMPTY)
        self.assertEqual(len(db.read('sim-run-1').snapshot.deliveries),1)
        self.assertEqual(coord.commit_release(order,p,update),out)


if __name__=='__main__':unittest.main()
