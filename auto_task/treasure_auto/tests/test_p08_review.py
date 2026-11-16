import unittest
from dataclasses import replace
from test_p08_support import *
import test_p08_service as sf
import test_p08_recovery as rf
from test_p08_observation import release_frame
from treasure_auto.placement.layout import LayoutTracker
from treasure_auto.placement.recovery import recovery_step


class ReviewTests(unittest.TestCase):
    def test_prior_unknown_out_of_region_brick_cannot_complete_V1(self):
        previous=layout(m.LayoutKind.V1,2,bricks=(brick(kind=m.LayoutKind.V1),brick(2,kind=m.LayoutKind.V1)))
        tr=LayoutTracker(previous,config())
        sp=space((occupancy(x=1,state=m.OccupancyState.UNKNOWN,t=110.1),occupancy('brick-2',x=.105,t=110.1),occupancy('brick-3',x=-.105,t=110.1)))
        quality=replace(rf.facts('brick-3',t=110.1),prior_facts=(rf.facts(t=110.1),rf.facts('brick-2',t=110.1)))
        with self.assertRaises(c.ContractError):tr.record_placement(release('brick-3',t=110.1),sp,quality,region(),context(110.1))
        self.assertEqual(tr.layout.confirmed_count,2)

    def test_temporary_route_for_other_brick_is_unresolved(self):
        tmp=occupancy('temp',x=.15,state=m.OccupancyState.RESERVED)
        items=(occupancy(),occupancy('brick-2',z=.1,supports=('brick-1',)),occupancy('brick-3',z=.2,supports=('brick-2',)),tmp)
        plan=replace(request().plan,segments=(replace(request().plan.segments[0],phase=m.Phase.TEMPORARY),))
        result=rf.RecoveryTests().base(space=space(items),temporary=(tmp,plan))
        self.assertEqual(result.operation,'unresolved')

    def test_cancel_in_P07_callback_preserves_history_without_current_success(self):
        h=sf.ServiceTests();svc=h.make();svc.place(h.req)
        original=h.manip.propose_release
        def interrupt(event,holding):
            update=original(event,holding);h.execution.cancel('simulated cancel');return update
        h.manip.propose_release=interrupt
        for t,cup in ((110,(0,0,.1)),(110.1,(0,0,.15)),(110.35,(0,0,.15))):
            h.clock.advance(t-h.clock.now().seconds);out=svc.observe_metrics(release_frame(t,cup))
        self.assertEqual(out.result.status,c.Outcome.UNKNOWN)
        self.assertEqual(out.proposed_holding.state,c.Holding.HELD)
        self.assertIsNotNone(svc.window.event,'historical observed release must remain traceable')

    def test_cancel_after_pending_result_does_not_republish_success(self):
        h=sf.ServiceTests();svc=h.make();svc.place(h.req)
        for t,cup in ((110,(0,0,.1)),(110.1,(0,0,.15)),(110.35,(0,0,.15))):
            h.clock.advance(t-h.clock.now().seconds);out=svc.observe_metrics(release_frame(t,cup))
        self.assertEqual(out.result.status,c.Outcome.SUCCESS)
        h.execution.cancel('after proposal')
        self.assertEqual(svc.observe_metrics(release_frame(110.4)).result.status,c.Outcome.UNKNOWN)

    def test_configuration_version_mismatch_is_rejected(self):
        from treasure_auto.placement.targets import validate_request
        with self.assertRaises(c.ContractError):
            validate_request(request(),context(),space(),occupancy(),replace(config(),version='different'))

    def test_old_second_round_cannot_mutate_count_before_rejection(self):
        previous=layout(m.LayoutKind.V1,2,round_=1,bricks=(brick(round_=1,kind=m.LayoutKind.V1),brick(2,round_=0,kind=m.LayoutKind.V1)))
        tr=LayoutTracker(previous,config())
        sp=space((occupancy(t=110.1),occupancy('brick-2',x=.105,t=110.1),occupancy('brick-3',x=-.105,t=110.1)))
        quality=replace(rf.facts('brick-3',t=110.1,round_=1),prior_facts=(rf.facts(t=110.1,round_=1),rf.facts('brick-2',t=110.1,round_=1)))
        with self.assertRaises(c.ContractError):tr.record_placement(release('brick-3',round_=1,t=110.1),sp,quality,region(),context(110.1,layout_recovery_round=1))
        self.assertEqual(tr.layout.confirmed_count,2)

    def test_anomaly_requests_stop_before_any_state_read(self):
        h=sf.ServiceTests();svc=h.make();svc.place(h.req)
        def state():
            self.assertEqual(h.execution.state,'unresolved','storage reads must not precede cancellation')
            return h.state
        svc.state_provider=state
        svc.notify_anomaly(rf.failure(),layout(),c.Fact.UNKNOWN,c.Fact.UNKNOWN)

    def test_timeout_keeps_separate_actual_stop_report(self):
        import test_p08_execution as ef
        h=ef.Harness();h.executor.start(h.req);h.clock.advance(3);h.executor.tick()
        self.assertEqual(h.executor.stop_report.physically_stopped,c.Fact.CONFIRMED)
        self.assertEqual(h.executor.state,'unresolved')


if __name__=='__main__':unittest.main()
