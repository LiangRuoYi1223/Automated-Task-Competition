from dataclasses import replace
from test_p07_support import Case, Harness, c, m, s


class ReviewTests(Case):
    def test_cancel_after_success_still_stops_transport_and_cannot_rearm_by_frames(self):
        h=self.h; h.reach('transport'); h.confirm_stage(); h.service.tick(); before=len(h.stops)
        h.service.cancel(m.StopRequest(h.env(c.Owner.P09),c.Reason(c.ErrorCode.CANCELLED,'run','cancel transport'),h.service.holding))
        h.confirm_stage()
        self.assertGreater(len(h.stops),before)
        self.assertIsNone(h.service.readiness)

    def test_completed_stage_cannot_advance_on_expired_observations(self):
        h=self.h
        h.service.config=replace(h.config,evidence_age_s=.4,max_frame_gap_s=.3)
        h.service.observer.config=h.service.config
        h.service.grasp(h.request()); h.confirm_stage(); h.clock.advance(.5)
        h.service.tick()
        self.assertEqual(len(h.commands),1)

    def test_same_event_id_cannot_count_as_new_observation(self):
        h=self.h; h.reach('trial_lift'); sample=h.observation(); h.service.observe(sample)
        newer=h.observation()
        newer=replace(newer,tool=replace(newer.tool,envelope=replace(newer.tool.envelope,event_id=sample.tool.envelope.event_id)))
        h.service.observe(newer)
        self.assertEqual(h.service.holding.state,c.Holding.UNKNOWN)

    def test_observation_checkpoint_failure_latches_motion_closed(self):
        h=self.h; h.reach('trial_lift'); h.checkpoints.fail=True; h.confirm_stage()
        h.checkpoints.fail=False; before=len(h.commands); h.confirm_stage(); h.service.tick()
        self.assertTrue(h.service.reconciliation_required)
        self.assertEqual(len(h.commands),before)

    def test_terminal_unknown_transport_cannot_restore_readiness_without_stopped_base(self):
        h=self.h; h.reach('transport'); h.confirm_stage(); h.service.tick()
        h.service.observe(h.observation(brick=None)); h.stop_confirmed=False
        h.confirm_stage()
        self.assertIsNone(h.service.readiness)

    def test_release_before_this_attempt_cannot_clear_current_object(self):
        h=self.h; h.reach('transport'); h.confirm_stage(); h.service.tick()
        release=m.ReleaseEvent(h.env(c.Owner.P08,action_id='release'),m.DeliveryKey('sim-run-1','inventory-1'),
                  m.Category.CAKE,m.Area.A,'sim-mapping-v1',s.stamp(105),c.Fact.CONFIRMED,c.Fact.CONFIRMED,
                  (h.ev('release','brick-1'),))
        with self.assertRaises(c.ContractError): h.service.propose_release(release,h.service.holding)

    def test_checkpoint_invalid_budget_cannot_be_emptied_and_retried(self):
        h=self.h; h.service.grasp(h.request())
        h.checkpoints.value['budgets']['inventory-1']['seconds']=float('nan')
        h.service=h.new_service()
        with self.assertRaises(c.ContractError):
            h.service.reconcile(h.observation(empty=True,action_id=None,target_id=None),h.stop_report())

    def test_new_inventory_id_cannot_reset_same_physical_target_budget(self):
        h=self.h; h.service.grasp(h.request())
        h.service.cancel(m.StopRequest(h.env(c.Owner.P09),c.Reason(c.ErrorCode.CANCELLED,'run','cancel'),h.service.holding))
        h.service.reconcile(h.observation(empty=True,action_id=None,target_id=None),h.stop_report())
        req=h.request('alias-reset')
        req=replace(req,order=replace(req.order,inventory_id='new-inventory'))
        self.assertEqual(h.service.grasp(req).result.status,c.Outcome.REJECTED)
        self.assertEqual(len(h.commands),1)

    def test_stage_requires_P01_queue_evidence_beyond_P03_navigation_stop(self):
        # A P03 navigation stop alone does not attest to the serial firmware queue.
        h=self.h
        h.service.grasp(h.request())
        h.service.stop_provider=lambda: replace(h.stop_report(),envelope=replace(h.stop_report().envelope,producer=c.Owner.P03))
        h.confirm_stage(); h.service.tick()
        self.assertEqual(len(h.commands),1)

    def test_new_empty_view_after_lift_prevents_withdraw(self):
        h=self.h; h.reach('trial_lift'); h.confirm_stage(); before=len(h.commands)
        h.service.observe(h.observation(empty=True)); h.service.tick()
        self.assertEqual(len(h.commands),before)
        self.assertEqual(h.service.holding.state,c.Holding.UNKNOWN)

    def test_new_empty_view_during_transport_revokes_readiness(self):
        h=self.h; h.reach('transport'); h.confirm_stage(); h.service.tick()
        h.service.observe(h.observation(empty=True))
        self.assertIsNone(h.service.readiness)
        self.assertEqual(h.service.holding.state,c.Holding.UNKNOWN)

    def test_new_wrong_arm_position_revokes_approach_completion(self):
        h=self.h; h.service.grasp(h.request()); h.confirm_stage()
        h.service.observe(h.observation(z=50)); h.service.tick()
        self.assertEqual(len(h.commands),1)

    def test_readiness_expiry_is_bounded_by_config_not_camera_claim(self):
        h=self.h; h.reach('transport')
        for _ in range(2):
            sample=h.observation()
            ev=replace(sample.tool.evidence[0],valid_until=s.stamp(500))
            h.service.observe(replace(sample,tool=replace(sample.tool,evidence=(ev,))))
        result=h.service.tick()
        self.assertLessEqual(result.readiness.envelope.expires_at-h.clock.now(),h.config.evidence_age_s)

    def test_reconcile_cannot_bypass_release_transaction_for_held_or_lost_object(self):
        for unknown in (False,True):
            h=Harness(); h.reach('transport'); h.confirm_stage(); h.service.tick()
            if unknown: h.service.observe(h.observation(brick=None))
            with self.assertRaises(c.ContractError):
                h.service.reconcile(h.observation(empty=True,action_id=None,target_id=None),h.stop_report())
            self.assertNotEqual(h.service.holding.state,c.Holding.EMPTY)

    def test_checkpoint_missing_budget_cannot_reset_used_attempts(self):
        h=self.h; h.service.grasp(h.request())
        h.service.cancel(m.StopRequest(h.env(c.Owner.P09),c.Reason(c.ErrorCode.CANCELLED,'run','cancel'),h.service.holding))
        h.checkpoints.value['budgets']={}
        h.service=h.new_service()
        with self.assertRaises(c.ContractError):
            h.service.reconcile(h.observation(empty=True,action_id=None,target_id=None),h.stop_report())

    def test_new_attempt_cannot_inherit_old_submission_and_echo(self):
        h=self.h; h.service.grasp(h.request())
        h.service.cancel(m.StopRequest(h.env(c.Owner.P09),c.Reason(c.ErrorCode.CANCELLED,'run','cancel'),h.service.holding))
        h.service.reconcile(h.observation(empty=True,action_id=None,target_id=None),h.stop_report())
        b=h.service.budget('inventory-1'); req=h.request('retry',b.attempts_used)
        req=replace(req,order=replace(req.order,budget=replace(req.order.budget,seconds_used=b.seconds_used)))
        h.intent_status=m.CommitStatus.UNKNOWN
        result=h.service.grasp(req)
        self.assertNotEqual(result.result.submitted,c.Fact.CONFIRMED)
        self.assertNotEqual(result.result.echoed,c.Fact.CONFIRMED)
        self.assertEqual(len(h.commands),1)

    def test_transport_drop_range_and_lock_survive_restart(self):
        h=self.h; h.reach('transport'); h.confirm_stage(); h.service.tick()
        region=m.RegionBoundary(m.Area.A,'map',s.footprint(),c.MapPose('map',0,0,0,'m','rad',.01,.01),.01)
        h.service.observe(h.observation(dropped=c.Fact.CONFIRMED,drop_region=region))
        restored=h.new_service()
        self.assertEqual(restored.holding.state,c.Holding.DROPPED)
        self.assertEqual(restored.drop_region,region)

    def test_final_stop_query_crossing_deadline_never_submits(self):
        h=self.h
        def delayed_stop():
            report=h.stop_report()
            if h.service._sent: h.clock.advance(600)
            return report
        h.service.stop_provider=delayed_stop
        h.service.grasp(h.request())
        self.assertEqual(h.commands,[])

    def test_final_stop_query_cancellation_never_submits(self):
        h=self.h
        def cancelling_stop():
            report=h.stop_report()
            if h.service._sent:
                h.service.cancel(m.StopRequest(h.env(c.Owner.P09),c.Reason(c.ErrorCode.CANCELLED,'run','cancel in callback'),h.service.holding))
            return report
        h.service.stop_provider=cancelling_stop
        h.service.grasp(h.request())
        self.assertEqual(h.commands,[])

    def test_trial_lift_physical_completion_has_durable_action_event(self):
        h=self.h; h.reach('trial_lift'); h.confirm_stage()
        events=[ev for ev in h.events if ev.result.phase=='trial_lift' and ev.result.physical is c.Fact.CONFIRMED]
        self.assertEqual(len(events),1)
        self.assertEqual(len(events[0].result.evidence),2)

    def test_observation_journal_unknown_latches_stop_before_next_stage(self):
        h=self.h; h.reach('trial_lift')
        h.append_event=lambda event: m.CommitReceipt('unknown',m.CommitStatus.UNKNOWN,None,None,
                                                     c.Reason(c.ErrorCode.STORAGE_FAILURE,'event','unknown commit'))
        h.confirm_stage(); before=len(h.commands); h.service.tick()
        self.assertEqual(len(h.commands),before)
        self.assertTrue(h.service.reconciliation_required)

    def test_cancel_terminal_event_is_recorded_after_stop(self):
        h=self.h; h.service.grasp(h.request())
        h.service.cancel(m.StopRequest(h.env(c.Owner.P09),c.Reason(c.ErrorCode.CANCELLED,'run','cancel'),h.service.holding))
        self.assertTrue(any(ev.result.status is c.Outcome.CANCELLED for ev in h.events))
