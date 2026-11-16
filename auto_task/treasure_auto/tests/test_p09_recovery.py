import json
import unittest
from dataclasses import replace
from test_p09_support import *
from test_p09_engine import Flow
from treasure_auto.mission.engine import MissionEngine
from treasure_auto.mission.inputs import Stage
from treasure_auto.mission.checkpoint import SimulationJournal
from treasure_auto.mission.config import MissionConfig
from treasure_auto.mission.recovery import LayoutRecovery


def layout(count,round_=1,t=110.,kind=m.LayoutKind.V1):
    bs=tuple(m.LayoutBrick('brick-'+str(i),i,1,(),None,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,
        c.Fact.CONFIRMED if i==1 else c.Fact.UNKNOWN,round_,(proof(t,'brick-'+str(i),None,'layout-'+str(i)),)) for i in range(1,count+1))
    return m.LayoutSnapshot(envelope(c.Owner.P08,t=t,target_id=None,action_id=None,layout_recovery_round=round_),m.Area.B,kind,round_,0,count,bs,
        c.Fact.UNKNOWN,c.Fact.CONFIRMED if count==3 else c.Fact.UNKNOWN)


def replay(n,round_=1,t=111.):
    e=envelope(c.Owner.P08,t=t,event_id='replay-'+str(n)+'-'+str(round_),target_id='brick-'+str(n),action_id='replay-'+str(n),layout_recovery_round=round_)
    return m.ReleaseEvent(e,m.DeliveryKey(e.run_id,'inventory-'+str(n)),m.Category.DOLL,m.Area.B,'sim-mapping-v1',s.stamp(t),
        c.Fact.CONFIRMED,c.Fact.CONFIRMED,(proof(t,e.target_id,e.action_id,e.event_id),))


class RecoveryTests(unittest.TestCase):
    def tracker(self):return LayoutRecovery(layout(0),m.Budget('recovery-B',3,30,0,0),2.)

    def test_T06_center_first_then_actual_second_third_and_no_arrival_changes(self):
        tracker=self.tracker();tracker.accept(layout(1),s.stamp(110.))
        tracker.accept(layout(2,t=111),s.stamp(111),replay(2,t=111))
        tracker.accept(layout(3,t=112),s.stamp(112),replay(3,t=112))
        self.assertEqual(tracker.layout.confirmed_count,3)
        self.assertEqual(tracker.budget.attempts_used,0)

    def test_old_second_jump_and_same_frame_replay_rejected(self):
        for candidate,event in [(layout(2),None),(layout(2,t=111),replay(2,round_=0)),
                                (layout(2,t=111),replay(2,t=110))]:
            tracker=self.tracker();tracker.accept(layout(1),s.stamp(110))
            with self.assertRaises(c.ContractError):tracker.accept(candidate,s.stamp(111),event)
            self.assertEqual(tracker.layout.confirmed_count,1)

    def test_failure_resets_only_on_new_physical_event_and_budget_survives(self):
        tr=self.tracker();tr.accept(layout(1),s.stamp(110))
        e=envelope(c.Owner.P08,t=111,action_id=None,target_id=None,event_id='failure')
        ev=m.ActionEvent(e,c.ActionResult(c.Outcome.FAILED,'layout_failure',c.Reason(c.ErrorCode.INVALID_LAYOUT,'layout','confirmed'),
            c.Fact.CONFIRMED,c.Fact.UNKNOWN,c.Fact.UNKNOWN,c.Fact.CONFIRMED,(proof(111),),3.))
        reset=replace(layout(0,round_=2,t=111),rebuild_generation=1,envelope=replace(layout(0,round_=2,t=111).envelope,rebuild_generation=1))
        tr.failed(reset,ev,True,s.stamp(111));tr.failed(reset,ev,True,s.stamp(111))
        self.assertEqual(tr.layout.confirmed_count,0);self.assertEqual(tr.layout.rebuild_generation,1)
        self.assertEqual(tr.budget.attempts_used,1);self.assertEqual(tr.budget.seconds_used,3.)

    def test_suspected_failure_does_not_clear(self):
        tr=self.tracker();tr.accept(layout(1),s.stamp(110))
        e=m.ActionEvent(envelope(c.Owner.P08),s.timeout_result())
        with self.assertRaises(c.ContractError):tr.failed(layout(0,round_=2),e,False,s.stamp(111))
        self.assertEqual(tr.layout.confirmed_count,1)

    def test_retry_budget_never_reset_and_unknown_limits_block(self):
        tr=self.tracker()
        for i in range(3):tr.charge('attempt-'+str(i),10.)
        self.assertFalse(tr.can_retry(100))
        tr.charge('attempt-2',10.);self.assertEqual(tr.budget.attempts_used,3)
        with self.assertRaises(c.ContractError):tr.charge('attempt-2',1.)
        tr=LayoutRecovery(layout(0),m.Budget('recovery-B',None,None,0,0),2.)
        self.assertFalse(tr.can_retry(100))

    def test_T05_ten_arrivals_not_complete_with_held_or_repair(self):
        h=Flow(mapping=True)
        ds=tuple(receipt(k,i) for k in m.Category for i in range(1,2 if k is m.Category.CAKE else 4))
        h.bump(deliveries=ds,holding=c.HoldingSnapshot(c.Holding.UNKNOWN,None,None,()))
        out=h.step()
        self.assertEqual(out.decision.delivery_complete,c.Fact.CONFIRMED)
        self.assertNotEqual(out.decision.layout_complete,c.Fact.CONFIRMED)

    def test_restart_roundtrip_retains_exit_pending_identity_and_deadline(self):
        h=Flow();h.step();ex=m.ExitEvidence(h.env(c.Owner.P03,target_id=None,action_id=None),m.Area.A,c.Fact.CONFIRMED,(proof(),))
        h.exits=(ex,);h.step();payload=json.loads(json.dumps(h.journal.payload));h.clock.advance(10)
        engine=MissionEngine(h.engine.config,h.clock,SimulationJournal(payload))
        self.assertEqual(engine.stage,Stage.REVALIDATE)
        self.assertEqual(engine.exit_points,1)
        self.assertEqual(engine.run.deadline,s.stamp(700))
        self.assertEqual(engine.pending.envelope.action_id,h.engine.pending.envelope.action_id)
        out=engine.advance(h.frame());self.assertIsNone(out.directive)
        self.assertIsNotNone(engine.checkpoint()['pending'])

    def test_checkpoint_corrupt_types_and_old_start_rejected(self):
        h=Flow();h.step();data=h.journal.load();data['sequence']=-1
        with self.assertRaises(c.ContractError):MissionEngine(h.engine.config,h.clock,SimulationJournal(data))
        data=h.journal.load();data['cake_closed']='yes'
        with self.assertRaises(c.ContractError):MissionEngine(h.engine.config,h.clock,SimulationJournal(data))

    def test_resume_requires_new_controller_and_fresh_physical_review(self):
        h=Flow();h.step();engine=MissionEngine(h.engine.config,h.clock,SimulationJournal(h.journal.load()))
        event=m.ActionEvent(h.env(c.Owner.P10A,target_id=None,action_id=None),s.timeout_result())
        with self.assertRaises(c.ContractError):engine.resume(h.frame(),event)
        self.assertEqual(engine.stage,Stage.REVALIDATE)


if __name__=='__main__':unittest.main()
