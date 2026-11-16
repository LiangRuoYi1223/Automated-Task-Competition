from publication_support import simulated_space_readback
import unittest
from test_p03_support import *
try:
    from treasure_auto.navigation.service import NavigationService
except ImportError:NavigationService=None

def certificate(nav,clock,span=.6):
    times=tuple(clock.t-span+span*i/3 for i in range(4))
    refs=tuple(proof(t) for t in times)
    return m.StillnessCertificate(replace(nav.context,producer=c.Owner.P08,emitted_at=clock.now()),
        ('low-brick',),poly(0,0,7,7),'map',stamp(times[0]),stamp(times[-1]),
        tuple('obs-'+str(t) for t in times),c.Fact.CONFIRMED,c.Fact.CONFIRMED,0.,0.,.001,1,refs)

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(NavigationService,'P03 bounded recovery required')
        self.nav,self.clock,self.gateway=service();start(self.nav)

    def collapsed(self):
        self.nav.collapse(('low-brick',),poly(0,0,6,6),1)
        stopped(self.nav,self.clock)

    def test_T04_499ms_blocks_navigation_precision_and_observation_micromotion(self):
        self.collapsed();self.clock.advance(.499)
        self.assertFalse(self.nav.confirm_stillness(certificate(self.nav,self.clock,.499)).accepted)
        for source in (m.MotionSource.NAVIGATION,m.MotionSource.PRECISION,m.MotionSource.RECOVERY):
            self.assertIsNone(self.nav.prepare_velocity(.01,0,source))
        self.assertEqual(self.gateway.commands,[])

    def test_T05_500ms_does_not_override_unknown_holding_or_posture(self):
        self.collapsed();self.clock.advance(.6)
        self.assertTrue(self.nav.confirm_stillness(certificate(self.nav,self.clock)).accepted)
        unknown=readiness(self.clock.t,state=c.Holding.UNKNOWN)
        self.nav.refresh(loc(self.clock.t),unknown)
        self.assertFalse(self.nav.recover(request(self.clock.t).permit).accepted)
        self.assertEqual(self.gateway.commands,[])

    def test_certificate_before_actual_stop_wrong_scene_duplicates_or_gaps_fail(self):
        self.collapsed();self.clock.advance(.6);good=certificate(self.nav,self.clock)
        for cert in (replace(good,scene_epoch=2),replace(good,observation_ids=('one',)*4),
                     replace(good,coverage_polygon_m=poly(0,0,1,1)),
                     replace(good,evidence=good.evidence[:1]*4)):
            self.assertFalse(self.nav.confirm_stillness(cert).accepted)

    def test_T07_retry_count_and_total_time_cannot_reset(self):
        for attempt in range(2):
            self.nav.blocked('planner obstructed');stopped(self.nav,self.clock)
            self.assertTrue(self.nav.recover(request(self.clock.t).permit).accepted)
            self.nav.submit_goal(receipt(self.nav.goal_intent))
        self.nav.blocked('blocked again');stopped(self.nav,self.clock)
        self.assertFalse(self.nav.recover(request(self.clock.t).permit).accepted)
        self.assertTrue(self.nav.terminal)
        self.assertEqual(self.nav.budget.attempts_used,2)

    def test_T07_distance_increasing_detour_counts_as_real_progress(self):
        for i in range(3):
            self.clock.advance(.3)
            self.nav.refresh(loc(self.clock.t,p=pose(2-.15*(i+1),2+.15*(i+1))),readiness(self.clock.t))
            self.nav.tick()
        self.assertEqual(self.nav.phase,'coarse')
        self.assertEqual(self.nav.budget.attempts_used,0)

    def test_restart_requires_restored_budget_and_never_restores_path(self):
        resumed=replace(run(),ever_restarted=True)
        nav=NavigationService(resumed,self.clock,policy(),profile(),bundle(),self.gateway,lambda r:(r.localization.pose,r.goal.pose),space_readback=simulated_space_readback)
        self.assertEqual(nav.navigate(request()).result.status,c.Outcome.REJECTED)
        budget=m.Budget('p03_recovery',2,2.,2,.5)
        nav=NavigationService(resumed,self.clock,policy(),profile(),bundle(),self.gateway,
                              lambda r:(r.localization.pose,r.goal.pose),restored_budget=budget,space_readback=simulated_space_readback)
        start(nav);nav.blocked('blocked');stopped(nav,self.clock)
        self.assertFalse(nav.recover(request(self.clock.t).permit).accepted)
        self.assertEqual(nav.budget.attempts_used,2)
