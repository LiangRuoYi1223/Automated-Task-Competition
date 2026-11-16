from publication_support import simulated_run_proposal
import unittest
from dataclasses import replace
from test_p09_support import *
from test_p09_engine import Flow
from test_p09_review import retry_review
from treasure_auto.mission.engine import MissionEngine
from treasure_auto.mission.checkpoint import SimulationJournal
from treasure_auto.mission.transactions import ReleaseCoordinator


class PersistenceBoundaryTests(unittest.TestCase):
    def test_renamed_same_inventory_replacement_does_not_add_arrival(self):
        before,order,p,u=release_case();name='renamed-physical'
        def ev(proof):return replace(proof,target_id=name)
        def holding(h):return replace(h,target_id=name if h.state is c.Holding.HELD else None,evidence=tuple(ev(x) for x in h.evidence))
        def space(sp):return replace(sp,occupancies=tuple(replace(o,target_id=name,evidence=tuple(ev(x) for x in o.evidence)) for o in sp.occupancies))
        old=m.DeliveryReceipt(m.DeliveryKey('sim-run-1','inventory-1'),'original-release',m.Category.CAKE,m.Area.A,s.stamp(108),('original-proof',))
        before=replace(before,deliveries=(old,),holding=holding(before.holding),space=space(before.space),
            inventory=(replace(before.inventory[0],target_ids=('brick-1',name)),))
        order=replace(order,envelope=replace(order.envelope,target_id=name))
        release=replace(p.release,envelope=replace(p.release.envelope,target_id=name,event_id='replacement'),evidence=tuple(ev(x) for x in p.release.evidence))
        p=replace(p,envelope=replace(p.envelope,target_id=name),release=release,proposed_space=space(p.proposed_space),proposed_holding=holding(p.proposed_holding))
        u=replace(u,envelope=replace(u.envelope,target_id=name),current=p.proposed_holding)
        clock=FakeClock(s.stamp(111));store=FakeStateStore(before,clock)
        out=ReleaseCoordinator(store,clock,run_update_provider=simulated_run_proposal).commit_release(order,p,u)
        self.assertEqual(out.receipt.status,m.CommitStatus.COMMITTED,out.receipt.reason)
        self.assertEqual(out.snapshot.deliveries,(old,))

    def test_foreign_exit_and_wrong_C0_type_in_checkpoint_rejected(self):
        h=Flow();h.step()
        ex=m.ExitEvidence(replace(h.env(c.Owner.P03,target_id=None,action_id=None),run_id='other'),m.Area.A,c.Fact.CONFIRMED,(proof(),))
        for field,value in [('exit',c.dumps(ex)),('run',c.dumps(s.arm_command())),('last_now',c.dumps(s.arm_command()))]:
            data=h.journal.load();data[field]=value
            with self.subTest(field=field),self.assertRaises(c.ContractError):
                MissionEngine(h.engine.config,h.clock,SimulationJournal(data))

    def test_restore_never_drops_saved_state_revision_floor(self):
        h=Flow();h.bump();h.bump();h.step()
        engine=MissionEngine(h.engine.config,h.clock,SimulationJournal(h.journal.load()))
        self.assertEqual(engine.last_revision,2)

    def test_retry_elapsed_cost_exhausts_budget_before_next_dispatch(self):
        h=Flow();h.board();h.add_candidate();h.finish();h.finish(c.Outcome.UNKNOWN,c.Fact.UNKNOWN);h.clock.advance(.1)
        review=retry_review(h);review=replace(review,result=replace(review.result,elapsed_s=100.))
        with self.assertRaises(c.ContractError):h.engine.retry_current(h.frame(),review)

    def test_unknown_checkpoint_cannot_be_bypassed_via_action_retry(self):
        h=Flow();h.board();h.add_candidate();h.finish();h.journal.fail=True;h.finish(c.Outcome.UNKNOWN,c.Fact.UNKNOWN);h.clock.advance(.1)
        with self.assertRaises(c.ContractError):h.engine.retry_current(h.frame(),retry_review(h))


if __name__=='__main__':unittest.main()
