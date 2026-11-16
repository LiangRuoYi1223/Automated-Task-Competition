import threading
from dataclasses import replace
from test_p05_support import P05Case, Harness, frame, config, c, m, s


class BoundaryTests(P05Case):
    def test_R01_stale_sensor_fresh_receipt_cannot_freeze(self):
        h=Harness()
        f=frame(1)
        f=replace(f,sensor_now=s.sensor_stamp(30))
        self.assertIsNone(h.send((frame(0),f)).mapping)

    def test_R01_capture_before_barrier_and_missing_stability_reject(self):
        for field,value in [('capture_not_before',s.sensor_stamp(25)),('stationary',c.Fact.UNKNOWN),
                            ('stationary_evidence',()),('waypoint_version','old'),
                            ('sensor_now',replace(s.sensor_stamp(20.3),clock_id='wrong'))]:
            with self.subTest(field=field):
                h=Harness(); self.assertIsNone(h.send((frame(0),replace(frame(1),**{field:value}))).mapping)

    def test_R01_stale_foreign_and_echo_evidence_reject(self):
        for fields in ({'run_id':'old'},{'origin':c.Origin.REPLAY},{'kind':c.EvidenceKind.ECHO},
                       {'valid_until':s.stamp(110.21)},{'received_at':s.stamp(115),'valid_until':s.stamp(120)}):
            with self.subTest(fields=fields):
                h=Harness(); f=frame(1)
                f=replace(f,stationary_evidence=(replace(f.image.image,**fields),))
                self.assertIsNone(h.send((frame(0),f)).mapping)

    def test_R04_confirmation_expires_while_adapter_is_running(self):
        h=Harness(); original=h.service.resolve_frame
        def slow(obs):
            value=original(obs)
            if value.image.observation_id=='obs-1': h.clock.advance(3)
            return value
        h.service.resolve_frame=slow
        self.assertIsNone(h.send((frame(0),frame(1))).mapping)
        self.assertFalse(h.service.dispatch_allowed)

    def test_R04_explicit_C0_slot_disagreement_rejects(self):
        h=Harness(); request=h.request((frame(0),frame(1)))
        observation=replace(request.observations[-1],slots=(m.SlotAssignment(m.Area.B,m.Category.CLOCK),))
        result=h.service.read_mapping(replace(request,observations=request.observations[:-1]+(observation,)))
        self.assertIsNone(result.mapping)

    def test_R04_clock_reboot_and_regression_return_failure_without_crash(self):
        for stamp in (s.stamp(111,boot_id='new-boot'),s.stamp(109)):
            with self.subTest(stamp=stamp):
                h=Harness(); h.send((frame(0),))
                h.clock.current=stamp
                # FakeClock stores its stamp in .value; direct replacement avoids pretending continuity.
                h.service.clock=type('BrokenClock',(),{'now':lambda self:stamp})()
                result=h.service.read_mapping(h.request())
                self.assertNotEqual(result.result.status,c.Outcome.SUCCESS)
                self.assertFalse(h.service.dispatch_allowed)

    def test_R04_unknown_commit_then_conflicting_frame_pauses_before_dispatch(self):
        h=Harness(); h.mode='unknown_after'; h.send((frame(0),frame(1)))
        result=h.send((frame(2,categories=(m.Category.CLOCK,m.Category.DOLL,m.Category.BANANA)),))
        self.assertIsNotNone(result.mapping)
        self.assertFalse(h.service.dispatch_allowed)
        self.assertEqual(result.result.reason.code,c.ErrorCode.CONFLICT)

    def test_R05_frozen_cake_or_multiple_patterns_pause(self):
        for bad in ('cake','multiple'):
            with self.subTest(bad=bad):
                h=Harness(); h.send((frame(0),frame(1)))
                f=frame(2)
                if bad=='cake': f=replace(f,patterns=(replace(f.patterns[0],category=m.Category.CAKE),)+f.patterns[1:])
                else: f=replace(f,patterns=f.patterns+(replace(f.patterns[0],category=m.Category.CLOCK),))
                h.send((f,))
                self.assertFalse(h.service.dispatch_allowed)

    def test_R06_id_conflict_does_not_mutate_retry_budget(self):
        h=Harness(); request=h.request(); h.service.read_mapping(request)
        h.service.read_mapping(replace(request,max_additional_views=0))
        self.assertIsNotNone(h.send().additional_view)

    def test_R04_cancellation_during_commit_prevents_success(self):
        h=Harness(); original=h.service.submit_mapping
        request=h.request((frame(0),frame(1)))
        def cancel_then_commit(mapping,audit):
            h.service.cancel(request)
            return original(mapping,audit)
        h.service.submit_mapping=cancel_then_commit
        result=h.service.read_mapping(request)
        self.assertEqual(result.result.status,c.Outcome.CANCELLED)
        self.assertFalse(h.service.dispatch_allowed)

    def test_R04_concurrent_calls_do_not_duplicate_commit(self):
        h=Harness(); request=h.request((frame(0),frame(1)))
        entered=threading.Event(); release=threading.Event(); original=h.service.submit_mapping
        def blocked(mapping,audit):
            entered.set(); self.assertTrue(release.wait(2)); return original(mapping,audit)
        h.service.submit_mapping=blocked
        result=[]
        thread=threading.Thread(target=lambda:result.append(h.service.read_mapping(request)))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            concurrent=h.service.read_mapping(request)
            self.assertNotEqual(concurrent.result.status,c.Outcome.SUCCESS)
        finally:
            release.set(); thread.join(3)
        self.assertFalse(thread.is_alive()); self.assertEqual(len(h.calls),1)

