import unittest
from dataclasses import replace
from test_p09_support import *
from test_p09_engine import Flow
from test_p09_scoring import scored,scene
from treasure_auto.mission.scoring import estimate
from treasure_auto.mission.inputs import Stage
from treasure_auto.mission.engine import MissionEngine
from treasure_auto.mission.checkpoint import SimulationJournal


def retry_review(h):
    d=h.engine.pending;t=h.clock.now().seconds
    e=replace(d.envelope,producer=c.Owner.P10A,event_id='review-'+d.envelope.action_id,emitted_at=s.stamp(t),expires_at=s.stamp(t+5))
    result=c.ActionResult(c.Outcome.SUCCESS,'action_reconciled',None,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.UNKNOWN,c.Fact.CONFIRMED,
        (proof(t,e.target_id,e.action_id,e.event_id),),.1)
    return m.ActionEvent(e,result)


class ReviewTests(unittest.TestCase):
    def carry(self):
        h=Flow();h.board();h.add_candidate();h.finish();h.grasp();return h

    def test_review_unknown_transport_posture_revokes_pending_navigation(self):
        h=self.carry();frame=h.frame()
        out=h.engine.advance(replace(frame,readiness=replace(frame.readiness,posture_confirmed=c.Fact.UNKNOWN)))
        self.assertIsNotNone(out.stop_request)
        self.assertIsNone(out.directive)

    def test_review_changed_held_target_cannot_dispatch_place(self):
        h=self.carry();h.bump(holding=replace(h.snapshot.holding,target_id='other'))
        out=h.finish()
        self.assertIsNotNone(out.stop_request);self.assertIsNone(out.directive)

    def test_review_retry_grasp_requires_fresh_current_geometry(self):
        for mode in ('missing','unknown','old_epoch','wrong_target'):
            h=Flow();h.board();h.add_candidate();h.finish();h.finish(c.Outcome.UNKNOWN,c.Fact.UNKNOWN)
            h.clock.advance(.1);frame=h.frame();plan=frame.plans[0]
            if mode=='missing':plans=()
            if mode=='unknown':plans=(replace(plan,collision_free=c.Fact.UNKNOWN),)
            if mode=='old_epoch':plans=(replace(plan,envelope=replace(plan.envelope,base_motion_epoch=99)),)
            if mode=='wrong_target':plans=(replace(plan,envelope=replace(plan.envelope,target_id='other')),)
            with self.subTest(mode=mode),self.assertRaises(c.ContractError):
                h.engine.retry_current(replace(frame,plans=plans),retry_review(h))
            self.assertEqual(h.engine.recovery_attempts,0)

    def test_review_unknown_ancestor_release_propagates_to_upper(self):
        for release in (c.Fact.UNKNOWN,c.Fact.REJECTED):
            base=replace(scored('base'),released=release)
            top=scored('top',layer=2,supports=('base',))
            out=estimate(scene((base,top)),s.run_context(),s.stamp(111))
            self.assertIsNone(out.terms[1].points)
            self.assertIn('top',out.estimate.unresolved_target_ids)

    def test_valid_retry_preserves_work_identity_and_charges_budget(self):
        h=Flow();h.board();h.add_candidate();h.finish();old=h.engine.pending
        h.finish(c.Outcome.UNKNOWN,c.Fact.UNKNOWN);h.clock.advance(.1)
        out=h.engine.retry_current(h.frame(),retry_review(h))
        self.assertEqual(out.directive.stage,Stage.GRASP)
        self.assertEqual(out.directive.envelope.request_id,old.envelope.request_id)
        self.assertNotEqual(out.directive.envelope.action_id,old.envelope.action_id)
        self.assertEqual(h.engine.recovery_attempts,1)
        self.assertEqual(h.engine.recovery_seconds,.1)

    def test_positive_restart_review_preserves_original_cutoff_and_new_action(self):
        h=Flow();h.step();old=h.engine.pending
        engine=MissionEngine(h.engine.config,h.clock,SimulationJournal(h.journal.load()))
        h.clock.advance(10.)
        newrun=replace(h.snapshot.run,envelope=replace(h.snapshot.run.envelope,instance_id='new',control_epoch=2),ever_restarted=True)
        newspace=replace(h.snapshot.space,envelope=replace(h.snapshot.space.envelope,instance_id='new',control_epoch=2))
        h.bump(run=newrun,space=newspace,holding=c.HoldingSnapshot(c.Holding.EMPTY,None,None,(proof(120),)))
        e=h.env(c.Owner.P10A,target_id=None,action_id=None,event_id='resume-review')
        result=c.ActionResult(c.Outcome.SUCCESS,'mission_reconciled',None,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.UNKNOWN,c.Fact.CONFIRMED,(proof(120),),1.)
        out=engine.resume(h.frame(),m.ActionEvent(e,result))
        self.assertEqual(engine.run.deadline,s.stamp(700))
        self.assertEqual(out.directive.stage,Stage.NAV_BOARD)
        self.assertEqual(out.directive.envelope.instance_id,'new')
        self.assertNotEqual(out.directive.envelope.action_id,old.envelope.action_id)


if __name__=='__main__':unittest.main()
