from publication_support import simulated_run_proposal
import unittest
from dataclasses import replace
from test_p09_support import *
from test_p09_engine import Flow
from test_p09_scoring import scored,scene
from treasure_auto.mission.inputs import Stage
from treasure_auto.mission.transactions import ReleaseCoordinator
from treasure_auto.mission.scoring import estimate


class BoundaryTests(unittest.TestCase):
    def test_pending_navigation_collapse_requests_stop_before_checkpoint(self):
        h=Flow();h.step();order=[]
        h.engine.stop_sink=lambda request:order.append('stop')
        save=h.journal.save
        h.journal.save=lambda payload:(order.append('save'),save(payload))[1]
        out=h.step(affected_targets=('fallen',))
        self.assertIsNone(out.directive)
        self.assertIsNotNone(out.stop_request)
        self.assertEqual(order[:2],['stop','save'])
        self.assertEqual(out.decision.phase,m.MissionPhase.RECOVERY)

    def test_clock_advancing_during_checkpoint_withholds_expired_dispatch(self):
        h=Flow();save=h.journal.save
        def slow(payload):
            h.clock.advance(600);return save(payload)
        h.journal.save=slow
        out=h.step()
        self.assertIsNone(out.directive)
        self.assertEqual(out.decision.phase,m.MissionPhase.STOPPED)
        self.assertIsNotNone(out.stop_request)

    def test_payload_of_rejected_old_callback_is_traceable(self):
        h=Flow();d=h.step().directive
        event=m.ActionEvent(replace(d.envelope,producer=c.Owner.P03,action_id='obsolete-action'),s.timeout_result())
        h.events=(event,);h.step()
        self.assertIn(c.dumps(event),h.engine.trace[-1]['action_events'])

    def test_current_ready_does_not_override_unknown_inventory(self):
        h=Flow();h.board();h.add_candidate()
        unknown=m.InventoryItem('uncertain',('uncertain',),c.Fact.UNKNOWN,None,m.InventoryLocation.UNKNOWN,None)
        h.bump(inventory=h.snapshot.inventory+(unknown,))
        out=h.finish();self.assertIsNone(out.directive)

    def test_unknown_read_status_cannot_confirm_transaction(self):
        before,order,p,u=release_case();clock=FakeClock(s.stamp(111));store=FakeStateStore(before,clock)
        read=store.read
        store.read=lambda run:replace(read(run),status=c.Outcome.UNKNOWN)
        out=ReleaseCoordinator(store,clock,run_update_provider=simulated_run_proposal).commit_release(order,p,u)
        self.assertIsNone(out.snapshot)

    def test_known_storage_rejection_latches_review_for_new_events(self):
        before,order,p,u=release_case();clock=FakeClock(s.stamp(111));store=FakeStateStore(before,clock)
        coord=ReleaseCoordinator(store,clock,run_update_provider=simulated_run_proposal);store.fail_next=True
        coord.commit_release(order,p,u)
        p=replace(p,release=replace(p.release,envelope=replace(p.release.envelope,event_id='new-event')))
        out=coord.commit_release(order,p,u)
        self.assertEqual(out.receipt.status,m.CommitStatus.REJECTED)
        self.assertEqual(store.read('sim-run-1').snapshot,before)

    def test_score_old_capture_with_new_receipt_or_changed_map_stays_unknown(self):
        b=scored()
        stale=replace(b,evidence=(replace(b.evidence[0],captured_at=s.stamp(101)),))
        wrongmap=replace(b,envelope=replace(b.envelope,versions=replace(b.envelope.versions,map='other-map')))
        for obj in (stale,wrongmap):
            self.assertIsNone(estimate(scene((obj,)),s.run_context(),s.stamp(111)).estimate.confirmed_points)


if __name__=='__main__':unittest.main()
