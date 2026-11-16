from dataclasses import replace
from test_p07_support import Case, c, m


class BehaviorTests(Case):
    def test_T01_unknown_or_held_rejects_second_grasp(self):
        h=self.h
        h.reach('pump')
        result=h.service.grasp(h.request('second'))
        self.assertEqual(result.result.status,c.Outcome.REJECTED)
        self.assertNotEqual(result.holding.state,c.Holding.EMPTY)
        self.assertEqual(len(h.commands),4)
        h.reach('transport'); h.confirm_stage(); h.service.tick()
        self.assertEqual(h.service.holding.state,c.Holding.HELD)
        result=h.service.grasp(h.request('held-second'))
        self.assertEqual(result.result.status,c.Outcome.REJECTED)
        self.assertEqual(len(h.commands),7)

    def test_T01_unstopped_or_queue_unknown_never_dispatches(self):
        h=self.h
        req=h.request()
        for field in ('physically_stopped','old_queue_empty'):
            result=h.service.grasp(replace(req,stop=replace(req.stop,**{field:c.Fact.UNKNOWN})))
            self.assertEqual(result.result.status,c.Outcome.REJECTED)
        self.assertEqual(h.commands,[])

    def test_T01_base_moves_between_stages_stops_before_descent(self):
        h=self.h
        h.service.grasp(h.request()); h.confirm_stage(); h.stop_confirmed=False
        h.service.tick()
        self.assertEqual(len(h.commands),1)
        self.assertTrue(h.stops)

    def test_R02_only_one_bound_stage_and_echo_cannot_advance(self):
        h=self.h
        h.service.grasp(h.request()); h.service.tick(); h.service.tick()
        self.assertEqual(len(h.commands),1)
        cmd=h.commands[0]
        self.assertEqual((cmd.phase,cmd.target.z,cmd.speed_mm_s,cmd.timeout_s),(m.Phase.APPROACH,20,1.,2.))
        self.assertEqual(cmd.speed_unit,'mm/s')
        self.assertEqual(len(h.intents[0].arm_plan.segments),1)

    def test_T02_pump_echo_is_not_held(self):
        h=self.h; h.reach('pump')
        self.assertEqual(h.service.holding.state,c.Holding.UNKNOWN)
        self.assertIsNone(h.service.result.readiness)

    def test_T02_source_empty_or_occluded_cannot_prove_trial(self):
        h=self.h; h.reach('trial_lift')
        for _ in range(2):
            h.service.observe(h.observation(brick=None,support_visible=c.Fact.UNKNOWN))
        self.assertNotEqual(h.service.holding.state,c.Holding.HELD)
        self.assertIsNone(h.service.result.readiness)

    def test_T02_fresh_same_object_motion_proves_lift_only(self):
        h=self.h; h.reach('trial_lift'); h.confirm_stage()
        self.assertEqual(h.service.holding.state,c.Holding.HELD)
        self.assertIsNone(h.service.result.readiness)

    def test_T03_wrong_transport_pose_never_readies_drive(self):
        h=self.h; h.reach('transport')
        for _ in range(2):
            h.service.observe(h.observation(posture_id='wrong-pose'))
        self.assertIsNone(h.service.result.readiness)

    def test_T03_transport_requires_independent_new_held_frames(self):
        h=self.h; h.reach('transport'); h.confirm_stage()
        result=h.service.tick()
        self.assertEqual(result.result.status,c.Outcome.SUCCESS)
        self.assertEqual(result.readiness.posture_confirmed,c.Fact.CONFIRMED)
        self.assertEqual(result.holding.target_id,'brick-1')

    def test_T04_occlusion_requests_stop_and_arm_only_recovery(self):
        h=self.h; h.reach('transport'); h.confirm_stage(); h.service.tick()
        h.service.observe(h.observation(brick=None,tool_visible=c.Fact.UNKNOWN))
        self.assertEqual(h.service.holding.state,c.Holding.UNKNOWN)
        self.assertIsNone(h.service.result.readiness)
        self.assertTrue(h.stops)
        self.assertEqual(h.service.recovery_request,'bounded_arm_observation')
        self.assertTrue(all(not isinstance(x,m.BaseCommand) for x in h.commands))

    def test_T04_evidence_expiry_requests_stop(self):
        h=self.h; h.reach('transport'); h.confirm_stage(); h.service.tick()
        h.clock.advance(2.1); h.service.tick()
        self.assertIsNone(h.service.result.readiness)
        self.assertEqual(h.service.holding.state,c.Holding.UNKNOWN)
        self.assertTrue(h.stops)

    def test_T05_cancel_before_descent_retains_empty(self):
        h=self.h; h.service.grasp(h.request())
        result=h.service.cancel(m.StopRequest(h.env(c.Owner.P09),c.Reason(c.ErrorCode.CANCELLED,'run','cancel'),h.service.holding))
        self.assertEqual(result.result.status,c.Outcome.CANCELLED)
        self.assertEqual(result.holding.state,c.Holding.EMPTY)
        self.assertFalse(h.service.pump_required)

    def test_T05_contact_cancel_holds_suction_and_queue_unknown(self):
        h=self.h; h.reach('pump'); h.stop_confirmed=False
        result=h.service.cancel(m.StopRequest(h.env(c.Owner.P09),c.Reason(c.ErrorCode.CANCELLED,'run','cancel'),h.service.holding))
        self.assertEqual(result.holding.state,c.Holding.UNKNOWN)
        self.assertTrue(h.service.pump_required)
        self.assertEqual(h.service.stop_report.old_queue_empty,c.Fact.UNKNOWN)
        self.assertFalse(any(isinstance(x,m.PumpCommand) and not x.enabled for x in h.commands))

    def test_T05_deadline_sends_no_extra_motion(self):
        h=self.h; h.reach('pump'); before=len(h.commands)
        h.clock.advance(600); result=h.service.tick()
        self.assertEqual(result.result.reason.code,c.ErrorCode.DEADLINE)
        self.assertEqual(len(h.commands),before)
        self.assertTrue(h.service.pump_required)

    def test_T07_duplicate_action_queries_without_resending(self):
        h=self.h; req=h.request()
        h.service.grasp(req); h.service.grasp(req)
        self.assertEqual(len(h.commands),1)
        conflict=replace(req,order=replace(req.order,inventory_id='other'))
        self.assertEqual(h.service.grasp(conflict).result.reason.code,c.ErrorCode.ID_CONFLICT)
        self.assertEqual(len(h.commands),1)

    def test_R06_unconfirmed_send_timeout_retains_uncertainty(self):
        h=self.h; h.reach('contact'); h.clock.advance(2.1)
        result=h.service.tick()
        self.assertEqual(result.result.status,c.Outcome.TIMED_OUT)
        self.assertEqual(result.holding.state,c.Holding.UNKNOWN)
        self.assertTrue(h.service.pump_required)

    def test_R06_storage_unknown_never_dispatches(self):
        h=self.h; h.intent_status=m.CommitStatus.UNKNOWN
        result=h.service.grasp(h.request())
        self.assertEqual(h.commands,[])
        self.assertNotEqual(result.result.status,c.Outcome.SUCCESS)

    def test_R06_deadline_during_commit_never_dispatches(self):
        h=self.h; h.expire_during_commit=True
        h.service.grasp(h.request())
        self.assertEqual(h.commands,[])

    def test_R06_cancel_during_prepare_never_dispatches(self):
        h=self.h
        h.prepare_callback=lambda: h.service.cancel(m.StopRequest(h.env(c.Owner.P09),c.Reason(c.ErrorCode.CANCELLED,'run','stop during prepare'),h.service.holding))
        h.service.grasp(h.request())
        self.assertEqual(h.commands,[])

    def test_R02_missing_stage_plan_is_explicit_gate(self):
        h=self.h; h.plan_available=False
        result=h.service.grasp(h.request())
        self.assertEqual(h.commands,[])
        self.assertEqual(result.result.reason.code,c.ErrorCode.UNAVAILABLE)
