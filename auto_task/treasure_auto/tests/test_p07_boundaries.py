import tempfile
from pathlib import Path
from dataclasses import replace
from test_p07_support import Case, Harness, Checkpoints, c, m, r, s, point


class BoundaryTests(Case):
    def test_stale_run_version_owner_and_origin_denied(self):
        h=self.h; req=h.request()
        for changes in (dict(run_id='old'),dict(producer=c.Owner.P08),dict(origin=c.Origin.LIVE),
                        dict(control_epoch=0),dict(base_motion_epoch=8),
                        dict(versions=replace(req.envelope.versions,space=1))):
            with self.subTest(changes=changes):
                result=h.service.grasp(replace(req,envelope=replace(req.envelope,**changes)))
                self.assertEqual(result.result.status,c.Outcome.REJECTED)
        self.assertEqual(h.commands,[])

    def test_T02_repeated_frame_never_completes_trial(self):
        h=self.h; h.reach('trial_lift'); sample=h.observation()
        h.service.observe(sample); h.service.observe(sample); h.service.tick()
        self.assertEqual(h.service.stage,'trial_lift')
        self.assertEqual(h.service.holding.state,c.Holding.UNKNOWN)

    def test_T02_old_capture_with_new_receipt_is_not_new_evidence(self):
        h=self.h; h.reach('trial_lift')
        for _ in range(2):
            sample=h.observation()
            ev=replace(sample.tool.evidence[0],captured_at=s.stamp(100))
            h.service.observe(replace(sample,tool=replace(sample.tool,evidence=(ev,))))
        self.assertEqual(h.service.holding.state,c.Holding.UNKNOWN)

    def test_T02_wrong_object_or_brick_not_following_cup_rejected(self):
        h=self.h; h.reach('trial_lift')
        for _ in range(2):
            h.service.observe(h.observation(brick=point(20,8)))
        self.assertEqual(h.service.holding.state,c.Holding.UNKNOWN)

    def test_R03_lift_lower_bound_and_upper_distance_are_enforced(self):
        for z in (14.9,41.):
            h=Harness(); h.reach('trial_lift')
            for _ in range(2): h.service.observe(h.observation(z=z))
            self.assertEqual(h.service.holding.state,c.Holding.UNKNOWN)

    def test_R03_sensor_clock_change_and_visibility_gap_reset_window(self):
        h=self.h; h.reach('trial_lift'); h.service.observe(h.observation())
        sample=h.observation()
        ev=replace(sample.tool.evidence[0],captured_at=s.sensor_stamp())
        h.service.observe(replace(sample,tool=replace(sample.tool,evidence=(ev,))))
        h.service.observe(h.observation())
        self.assertEqual(h.service.holding.state,c.Holding.UNKNOWN)

    def test_R03_broken_window_revokes_previously_completed_stage(self):
        h=self.h; h.reach('trial_lift'); h.confirm_stage()
        h.service.observe(h.observation(brick=None))
        before=len(h.commands); h.service.tick()
        self.assertEqual(len(h.commands),before)

    def test_R04_ready_result_passes_C0_action_evidence_validation(self):
        h=self.h; h.reach('transport'); h.confirm_stage(); result=h.service.tick()
        r.validate_action_result(result.envelope,result.result,h.clock.now())
        for ev in result.readiness.evidence:
            r.require_physical_evidence(ev,result.readiness.envelope,h.clock.now(),c.Origin.SIMULATED)

    def test_R06_observation_after_deadline_cannot_issue_readiness(self):
        h=self.h; h.reach('transport'); h.clock.advance(600)
        h.confirm_stage()
        self.assertIsNone(h.service.result.readiness)

    def test_R06_observation_after_stage_timeout_cannot_complete(self):
        h=self.h; h.reach('trial_lift'); h.clock.advance(2.1)
        h.confirm_stage()
        self.assertEqual(h.service.result.result.status,c.Outcome.TIMED_OUT)
        self.assertNotEqual(h.service.holding.state,c.Holding.HELD)

    def test_R06_send_exception_is_unknown_not_safe_to_retry(self):
        h=self.h; h.send_raises=True; req=h.request()
        result=h.service.grasp(req); h.service.grasp(req)
        self.assertEqual(len(h.commands),1)
        self.assertEqual(result.result.status,c.Outcome.UNKNOWN)

    def test_R06_checkpoint_unknown_stops_before_device(self):
        h=self.h; h.checkpoints.fail=True
        h.service.grasp(h.request())
        self.assertEqual(h.commands,[])
        self.assertTrue(h.service.reconciliation_required)
        self.assertTrue(h.stops)

    def test_R06_late_frames_cannot_override_cancelled(self):
        h=self.h; h.reach('trial_lift')
        h.service.cancel(m.StopRequest(h.env(c.Owner.P09),c.Reason(c.ErrorCode.CANCELLED,'run','cancel'),h.service.holding))
        h.confirm_stage(); h.service.tick()
        self.assertEqual(h.service.result.result.status,c.Outcome.CANCELLED)
        self.assertIsNone(h.service.result.readiness)

    def test_R06_expired_stage_permission_after_checkpoint_not_sent(self):
        h=self.h; original=h.checkpoints.save; count=[0]
        def save(*args):
            receipt=original(*args); count[0]+=1
            if count[0]==2: h.clock.advance(11)
            return receipt
        h.checkpoints.save=save
        h.service.grasp(h.request())
        self.assertEqual(h.commands,[])


class RecoveryTests(Case):
    def test_T06_file_checkpoint_restart_does_not_replay(self):
        with tempfile.TemporaryDirectory(prefix='p07-checkpoint-') as directory:
            path=Path(directory)/'checkpoint.json'
            h=Harness(Checkpoints(path)); req=h.request(); h.service.grasp(req)
            h.checkpoints=Checkpoints(path); h.clock.advance(1)
            restored=h.new_service(); h.service=restored
            self.assertTrue(restored.reconciliation_required)
            result=restored.grasp(req)
            self.assertEqual(len(h.commands),1)
            self.assertEqual(result.holding.state,c.Holding.UNKNOWN)
            self.assertIsNone(result.readiness)
            self.assertEqual(restored.grasp(h.request('next',1)).result.status,c.Outcome.REJECTED)

    def test_T06_checkpoint_read_failure_cannot_be_reconciled_as_empty_history(self):
        h=self.h; h.checkpoints.fail=True; restored=h.new_service(); h.service=restored
        h.checkpoints.fail=False
        with self.assertRaises(c.ContractError):
            restored.reconcile(h.observation(empty=True,action_id=None,target_id=None),h.stop_report())
        self.assertTrue(restored.reconciliation_required)

    def test_T07_retry_limit_survives_restart_and_same_target(self):
        h=self.h; first=h.request(); h.service.grasp(first)
        h.clock.advance(.3)
        h.service.cancel(m.StopRequest(h.env(c.Owner.P09),c.Reason(c.ErrorCode.CANCELLED,'run','cancel'),h.service.holding))
        h.service=h.new_service()
        h.service.reconcile(h.observation(empty=True,action_id=None,target_id=None),h.stop_report())
        budget=h.service.budget('inventory-1')
        req=h.request('retry',budget.attempts_used)
        req=replace(req,order=replace(req.order,budget=replace(req.order.budget,seconds_used=budget.seconds_used)))
        h.service.grasp(req)
        h.service.cancel(m.StopRequest(h.env(c.Owner.P09),c.Reason(c.ErrorCode.CANCELLED,'run','cancel'),h.service.holding))
        h.service.reconcile(h.observation(empty=True,action_id=None,target_id=None),h.stop_report())
        budget=h.service.budget('inventory-1')
        req=h.request('exhausted',budget.attempts_used)
        req=replace(req,order=replace(req.order,budget=replace(req.order.budget,seconds_used=budget.seconds_used)))
        result=h.service.grasp(req)
        self.assertEqual(result.result.status,c.Outcome.FAILED)
        self.assertEqual(result.result.reason.code,c.ErrorCode.LIMIT_EXCEEDED)
        self.assertEqual(len(h.commands),2)
        self.assertGreaterEqual(budget.seconds_used,.3-1e-9)

    def test_T06_restart_keeps_active_downtime_in_budget(self):
        h=self.h; h.service.grasp(h.request()); h.clock.advance(5)
        h.service=h.new_service()
        self.assertGreaterEqual(h.service.budget('inventory-1').seconds_used,5)


class ReleaseTests(Case):
    def release(self):
        h=self.h
        return m.ReleaseEvent(h.env(c.Owner.P08,action_id='place-1'),m.DeliveryKey('sim-run-1','inventory-1'),
               m.Category.CAKE,m.Area.A,'sim-mapping-v1',h.clock.now(),c.Fact.CONFIRMED,c.Fact.CONFIRMED,
               (h.ev('place-1','brick-1'),))

    def test_release_only_proposes_empty_before_atomic_commit(self):
        h=self.h; h.reach('transport'); h.confirm_stage(); h.service.tick()
        update=h.service.propose_release(self.release(),h.service.holding)
        self.assertEqual(update.current.state,c.Holding.EMPTY)
        self.assertEqual(h.service.holding.state,c.Holding.HELD)
        self.assertIsNone(h.service.result.readiness)

    def test_release_echo_cannot_clear_holding(self):
        h=self.h; h.reach('transport'); h.confirm_stage(); h.service.tick()
        release=self.release()
        with self.assertRaises(c.ContractError):
            h.service.propose_release(replace(release,evidence=(replace(release.evidence[0],kind=c.EvidenceKind.ECHO),)),h.service.holding)
        self.assertEqual(h.service.holding.state,c.Holding.HELD)

    def test_T04_drop_records_range_and_requires_empty_commit(self):
        h=self.h; h.reach('transport'); h.confirm_stage(); h.service.tick()
        region=m.RegionBoundary(m.Area.A,'map',s.footprint(),c.MapPose('map',0,0,0,'m','rad',.01,.01),.01)
        h.service.observe(h.observation(dropped=c.Fact.CONFIRMED,drop_region=region))
        self.assertEqual(h.service.holding.state,c.Holding.DROPPED)
        self.assertEqual(h.service.drop_region,region)
        h.service.observe(h.observation(empty=True))
        self.assertEqual(h.service.holding.state,c.Holding.DROPPED)
        self.assertEqual(h.service.pending_update.cause,m.HoldingCause.DROP_RECONCILED)

    def test_empty_arm_still_needs_fresh_independent_posture(self):
        h=self.h
        plan=h.stage_plan(m.Phase.WITHDRAW,h.env(action_id='empty-posture',target_id=None),None,h.service.holding)
        for _ in range(2):
            h.service.confirm_empty_readiness(h.observation(empty=True,action_id='empty-posture',target_id=None,z=30,posture_id='sim-empty'),plan,'sim-empty')
        self.assertEqual(h.service.readiness.holding.state,c.Holding.EMPTY)
        self.assertEqual(h.service.readiness.posture_confirmed,c.Fact.CONFIRMED)
