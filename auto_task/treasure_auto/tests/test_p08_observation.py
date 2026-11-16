import unittest
from dataclasses import replace
from test_p08_support import *
from treasure_auto.contracts import rules
from treasure_auto.placement.space import SpaceProposals, accept_committed
from treasure_auto.placement.evidence import ReleaseWindow, StillnessWindow


def release_frame(t,cup=(0,0,.15),x=0):
    f=metric(t,x=x,cup=cup)
    return replace(f,evidence=replace(f.evidence,target_id='brick-1',action_id='sim-action-1'))


def still_window():
    stop=replace(request().stop,envelope=env(c.Owner.P01,emitted_at=s.stamp(109.9)),evidence=(proof(109.9,target=None,action=None),))
    return StillnessWindow(config(),('brick-1',),square(side=.5),'map',0,stop,context())


class SpaceTests(unittest.TestCase):
    def test_R03_reservation_idempotent_conflict_and_unknown_retained(self):
        proposals=SpaceProposals(config());base=space();target=occupancy(state=m.OccupancyState.RESERVED)
        a=proposals.reserve(base,env(),target,s.stamp(110))
        b=proposals.reserve(base,env(),target,s.stamp(110))
        self.assertEqual(a,b);self.assertEqual(base.occupancies,())
        uncertain=proposals.mark_unknown(a,replace(env(),versions=s.versions(1)),'brick-1',s.stamp(110.2))
        self.assertEqual(uncertain.occupancies[0].state,m.OccupancyState.UNKNOWN)
        self.assertEqual(uncertain.occupancies[0].footprint_m,target.footprint_m)
        with self.assertRaises(c.ContractError):
            proposals.reserve(uncertain,replace(env(),versions=s.versions(2)),target,s.stamp(110.3))

    def test_R03_absence_and_timeout_cannot_clear_space(self):
        p=SpaceProposals(config());base=space((occupancy(t=109.9),))
        f=metric();f=replace(f,objects=(),orientations=())
        out=p.observe(base,f,context())
        self.assertEqual(len(out.occupancies),1)
        with self.assertRaises(c.ContractError):p.clear(base,env(),'brick-1',None,s.stamp(110))
        old=m.OccupancyClearance('brick-1',m.ClearanceCause.EMPTY,c.Fact.CONFIRMED,(proof(109.8),))
        with self.assertRaises(c.ContractError):p.clear(base,env(),'brick-1',old,s.stamp(110))
        fresh=replace(old,evidence=(proof(110),))
        self.assertEqual(p.clear(base,env(),'brick-1',fresh,s.stamp(110)).occupancies,())

    def test_R03_commit_unknown_and_wrong_readback_do_not_publish(self):
        expected=space((occupancy(),),1)
        for receipt in (m.CommitReceipt('tx',m.CommitStatus.UNKNOWN,None,None,c.Reason(c.ErrorCode.STORAGE_FAILURE,'commit','unknown')),
                        m.CommitReceipt('wrong',m.CommitStatus.COMMITTED,1,1,None)):
            with self.assertRaises(c.ContractError):accept_committed('tx',receipt,expected,s.state_snapshot())
        state=replace(s.state_snapshot(),revision=1,space=expected)
        good=m.CommitReceipt('tx',m.CommitStatus.COMMITTED,1,1,None)
        self.assertIs(accept_committed('tx',good,expected,state),expected)


class ReleaseTests(unittest.TestCase):
    def test_T02_cup_retreat_and_stationary_brick_required(self):
        w=ReleaseWindow(request(),config(),s.stamp(110))
        self.assertIsNone(w.push(release_frame(110,cup=(0,0,.1)),context(110),s.run_context()))
        self.assertIsNone(w.push(release_frame(110.1),context(110.1),s.run_context()))
        event=w.push(release_frame(110.35),context(110.35),s.run_context())
        self.assertEqual(event.separated,c.Fact.CONFIRMED)
        self.assertEqual(event.released_at,s.stamp(110.1))
        self.assertEqual(event.delivery_key,m.DeliveryKey('sim-run-1','inventory-1'))

    def test_T02_brick_following_cup_never_release(self):
        w=ReleaseWindow(request(),config(),s.stamp(110))
        w.push(release_frame(110,cup=(0,0,.1)),context(110),s.run_context())
        for t,x in ((110.1,.03),(110.35,.06)):
            self.assertIsNone(w.push(release_frame(t,cup=(x,0,.1),x=x),context(t),s.run_context()))

    def test_T03_missing_frame_and_wrong_region_no_release(self):
        for f in (replace(release_frame(110.1),objects=()),release_frame(110.1,x=.2)):
            w=ReleaseWindow(request(),config(),s.stamp(110))
            w.push(release_frame(110,cup=(0,0,.1)),context(110),s.run_context())
            self.assertIsNone(w.push(f,context(110.1),s.run_context()))
            self.assertIsNone(w.push(release_frame(110.4),context(110.4),s.run_context()))

    def test_R04_duplicate_sensor_clock_and_deadline_cannot_confirm(self):
        w=ReleaseWindow(request(),config(),s.stamp(110));f=release_frame(110,cup=(0,0,.1))
        w.push(f,context(110),s.run_context())
        self.assertIsNone(w.push(f,context(110.2),s.run_context()))
        sensor=replace(f,evidence=replace(f.evidence,captured_at=s.sensor_stamp(20)))
        self.assertIsNone(w.push(sensor,context(110.2),s.run_context()))
        late=release_frame(700)
        self.assertIsNone(w.push(late,context(700),s.run_context()))


class StillnessTests(unittest.TestCase):
    def feed(self,w,t,**kw):return w.push(metric(t,**kw),context(t))

    def test_T05_499ms_refused_500ms_certificate_is_C0_valid(self):
        w=still_window()
        for t in (110,110.25,110.499):self.assertIsNone(self.feed(w,t))
        cert=self.feed(w,110.5)
        self.assertIsNotNone(cert)
        rules.validate_stillness(cert,replace(env(),target_id=None,action_id=None),s.stamp(110.5),('brick-1',),0)

    def test_T05_duplicate_gap_occlusion_reset_window(self):
        for fault in ('duplicate','gap','occlusion'):
            w=still_window();self.feed(w,110);self.feed(w,110.2)
            if fault=='duplicate':w.push(metric(110.2),context(110.3))
            elif fault=='gap':self.feed(w,110.6)
            else:self.feed(w,110.3,visible=False)
            self.assertIsNone(self.feed(w,110.7))

    def test_T05_slow_drift_rotation_and_changed_support_reset(self):
        w=still_window()
        for t,x in ((110,0),(110.2,.0015),(110.4,.003),(110.6,.0045)):
            self.assertIsNone(self.feed(w,t,x=x))
        for change in ('rotation','support'):
            w=still_window();self.feed(w,110);f=metric(110.25)
            if change=='rotation':f=replace(f,orientations=(('brick-1',.1,0.),))
            else:f=replace(f,objects=(replace(f.objects[0],support_ids=('unknown-base',)),))
            self.assertIsNone(w.push(f,context(110.25)))
            self.assertIsNone(w.push(replace(f,envelope=replace(f.envelope,emitted_at=s.stamp(110.5))),context(110.5)))

    def test_R06_scene_scatter_coverage_missing_identity_and_pre_stop_frame(self):
        for f in (metric(110,scene=1),replace(metric(110),coverage_polygon_m=square(side=.1)),
                  replace(metric(110),identity_complete=c.Fact.UNKNOWN),metric(109.8)):
            w=still_window();self.assertIsNone(w.push(f,context(110)))
            self.assertIsNone(self.feed(w,110.5))

    def test_R06_new_motion_revokes_old_certificate(self):
        w=still_window()
        for t in (110,110.25,110.5):self.feed(w,t)
        self.assertIsNotNone(w.certificate)
        self.assertIsNone(self.feed(w,110.6,x=.02));self.assertIsNone(w.certificate)


if __name__=='__main__':unittest.main()
