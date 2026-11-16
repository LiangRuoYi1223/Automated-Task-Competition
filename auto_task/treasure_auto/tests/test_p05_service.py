from dataclasses import replace
from test_p05_support import P05Case, Harness, frame, config, c, m, s


class ServiceTests(P05Case):
    def test_T01_complete_new_frames_commit_full_mapping_and_evidence_once(self):
        h=Harness()
        self.assertIsNone(h.send((frame(0),)).mapping)
        result=h.send((frame(1),))
        self.assertEqual(result.result.status,c.Outcome.SUCCESS,result.result.reason)
        self.assertTrue(h.service.dispatch_allowed)
        self.assertEqual(result.mapping.destination(m.Category.CAKE),m.Area.A)
        self.assertEqual(len(result.mapping.assignments),3)
        self.assertEqual({e.evidence_id for e in result.mapping.evidence},{'image-0','image-1'})
        self.assertEqual(len(h.calls),1)
        self.assertEqual(h.store.read(h.run.envelope.run_id).snapshot.mapping,result.mapping)
        self.assertEqual(c.loads(c.dumps(result)),result)

    def test_T04_replayed_frames_and_same_capture_with_new_ids_do_not_confirm(self):
        h=Harness(); f=frame(0)
        h.send((f,))
        duplicate=replace(f,image=replace(f.image,observation_id='renamed'))
        self.assertIsNone(h.send((duplicate,)).mapping)
        self.assertEqual(len(h.calls),0)
        # A fresh observation is still required; duplicate delivery itself is not a new sample.
        self.assertEqual(h.send((frame(1),)).result.status,c.Outcome.SUCCESS)

    def test_T04_different_capture_id_same_timestamp_does_not_confirm(self):
        h=Harness(); h.send((frame(0),))
        f=frame(1)
        f=replace(f,image=replace(f.image,image=replace(f.image.image,captured_at=s.sensor_stamp(20))))
        self.assertIsNone(h.send((f,)).mapping)
        self.assertEqual(len(h.calls),0)

    def test_T04_direction_conflict_resets_window_without_voting(self):
        h=Harness(replace(config(),max_additional_views=10))
        for i in range(6):
            f=frame(i)
            if i%2:
                b=f.boards[0]
                f=replace(f,boards=(replace(b,directions=b.directions+(replace(b.directions[0],corner_order=(2,3,0,1)),)),))
            result=h.send((f,),views=10)
            self.assertIsNone(result.mapping)
        self.assertEqual(h.calls,[])

    def test_R04_class_change_segment_change_and_gap_restart_window(self):
        for change in ('class','segment','gap'):
            with self.subTest(change=change):
                h=Harness(); h.send((frame(0),))
                f=frame(1)
                if change=='class': f=frame(1,categories=(m.Category.CLOCK,m.Category.DOLL,m.Category.BANANA))
                if change=='segment': f=replace(f,segment_id='stable-2')
                if change=='gap': f=frame(6)
                self.assertIsNone(h.send((f,)).mapping)

    def test_R04_batch_is_validated_fully_before_commit(self):
        h=Harness()
        conflicting=frame(2,categories=(m.Category.CLOCK,m.Category.DOLL,m.Category.BANANA))
        self.assertIsNone(h.send((frame(0),frame(1),conflicting)).mapping)
        self.assertEqual(h.calls,[])

    def test_T05_frozen_conflict_latches_pause_and_retains_historical_mapping(self):
        h=Harness(); original=h.send((frame(0),frame(1))).mapping
        conflict=h.send((frame(2,categories=(m.Category.CLOCK,m.Category.DOLL,m.Category.BANANA)),))
        self.assertEqual(conflict.mapping,original)
        self.assertEqual(conflict.result.reason.code,c.ErrorCode.CONFLICT)
        self.assertFalse(h.service.dispatch_allowed)
        h.send((frame(3),))
        self.assertFalse(h.service.dispatch_allowed)
        self.assertEqual(len(h.calls),1)

    def test_T05_late_single_slot_cannot_overwrite_or_hide_conflict(self):
        h=Harness(); original=h.send((frame(0),frame(1))).mapping
        f=frame(2,categories=(m.Category.CLOCK,m.Category.DOLL,m.Category.BANANA))
        result=h.send((replace(f,patterns=f.patterns[:1]),))
        self.assertEqual(result.mapping,original)
        self.assertFalse(h.service.dispatch_allowed)
        self.assertEqual(len(h.calls),1)

    def test_T05_old_run_and_old_epoch_never_overwrite(self):
        h=Harness(); original=h.send((frame(0),frame(1))).mapping
        for fields in ({'run_id':'previous-run'},{'control_epoch':0},{'base_motion_epoch':1},{'instance_id':'old'}):
            request=h.request((frame(2),))
            result=h.service.read_mapping(replace(request,envelope=replace(request.envelope,**fields)))
            self.assertNotEqual(result.result.status,c.Outcome.SUCCESS)
            self.assertEqual(h.service.frozen,original)

    def test_R06_no_board_exits_finite_budget(self):
        h=Harness()
        for i in range(2):
            result=h.send((replace(frame(i),boards=()),))
            self.assertIsNotNone(result.additional_view)
            self.assertIsNone(result.mapping)
        final=h.send((replace(frame(2),boards=()),))
        self.assertEqual(final.result.status,c.Outcome.FAILED)
        self.assertIsNone(final.additional_view)
        self.assertEqual(h.service.views_used,2)

    def test_R06_duplicate_request_does_not_spend_budget_again(self):
        h=Harness(); request=h.request()
        a=h.service.read_mapping(request); b=h.service.read_mapping(request)
        self.assertEqual(a,b); self.assertEqual(h.service.views_used,1)
        changed=replace(request,max_additional_views=1)
        self.assertEqual(h.service.read_mapping(changed).result.reason.code,c.ErrorCode.ID_CONFLICT)

    def test_R06_deadline_cannot_be_extended_by_later_request(self):
        h=Harness(); h.send(deadline=111)
        h.clock.advance(1)
        result=h.send((frame(1),),deadline=200)
        self.assertEqual(result.result.status,c.Outcome.TIMED_OUT)
        self.assertIsNone(result.additional_view)

    def test_R06_cancel_stops_views_and_cannot_later_commit(self):
        h=Harness(); request=h.request()
        h.service.read_mapping(request)
        result=h.service.cancel(request)
        self.assertEqual(result.result.status,c.Outcome.CANCELLED)
        self.assertIsNone(h.send((frame(0),frame(1))).mapping)
        self.assertFalse(h.service.dispatch_allowed)

    def test_R01_missing_adapter_or_live_request_never_trusts_C0_slots(self):
        h=Harness(); h.service.resolve_frame=None
        request=h.request((frame(0),frame(1)))
        result=h.service.read_mapping(request)
        self.assertNotEqual(result.result.status,c.Outcome.SUCCESS)
        self.assertIsNone(result.mapping)
        self.assertEqual(len(h.calls),0)

