import unittest
from test_p03_support import *
try:
    from treasure_auto.navigation.geometry import check_route
    from treasure_auto.navigation.exit import ExitTracker
    from treasure_auto.navigation.config import load_policy
except ImportError:
    check_route=ExitTracker=load_policy=None

class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(check_route,'P03 full swept envelope checks missing')

    def test_route_checks_between_endpoints_and_unknown_low_bricks(self):
        for state in m.OccupancyState:
            with self.subTest(state=state):
                result=check_route(bundle(),(pose(),pose(3,2)),profile(),c.Holding.EMPTY,space(items=(obstacle(state=state),)))
                self.assertFalse(result.accepted)
        self.assertTrue(check_route(bundle(),(pose(),pose(3,2)),profile(),c.Holding.EMPTY,space()).accepted)

    def test_arm_and_carried_shape_block_center_clear_path(self):
        p=replace(profile(),arm=poly(-.1,-.1,.8,.1))
        self.assertFalse(check_route(bundle(),(pose(),pose(3,2)),p,c.Holding.EMPTY,
                                     space(items=(obstacle((2.5,2.65,2.6,2.75)),))).accepted)
        self.assertFalse(check_route(bundle(),(pose(),pose(3,2)),profile(),c.Holding.UNKNOWN,space()).accepted)
        self.assertFalse(check_route(bundle(),(pose(),pose(3,2)),replace(profile(),carried=None),c.Holding.HELD,space()).accepted)

    def test_static_unknown_cells_boundary_and_frame_block(self):
        b=bundle();pixels=list(b.raster.pixels);pixels[11*16+5]=128
        self.assertFalse(check_route(replace(b,raster=replace(b.raster,pixels=tuple(pixels))),
                                     (pose(),pose(3,2)),profile(),c.Holding.EMPTY,space()).accepted)
        for path in ((pose(),pose(9,2)),(pose(),pose(3,2,frame_id='odom'))):
            self.assertFalse(check_route(b,path,profile(),c.Holding.EMPTY,space()).accepted)

    def test_unconfigured_and_invalid_numeric_parameters_fail_closed(self):
        path=Path(__file__).resolve().parents[1]/'config/navigation/unmeasured.json'
        self.assertIsNotNone(load_policy)
        with self.assertRaises(c.ContractError):load_policy(path)
        for key,value in [('max_age_s',float('nan')),('stop_window_s',0),('retry_limit',True),('max_linear_m_s',-1)]:
            with self.subTest(key=key),self.assertRaises(c.ContractError):replace(policy(),**{key:value}).validate()

class ExitTests(unittest.TestCase):
    def setUp(self):self.assertIsNotNone(ExitTracker,'P03 exit tracker missing')

    def test_T06_center_outside_tail_touching_is_not_exit(self):
        tracker=ExitTracker(run(),bundle().metadata,profile(),1.)
        for x in (1.01,1.3,1.31):
            self.assertIsNone(tracker.observe(loc(p=pose(x,.5)),readiness(),stamp()))
        result=tracker.observe(loc(p=pose(1.32,.5)),readiness(),stamp())
        self.assertEqual(result.entire_envelope_outside,c.Fact.CONFIRMED)

    def test_T06_arm_orientation_errors_and_holding_unknown(self):
        p=replace(profile(),arm=poly(-.6,-.1,.1,.1))
        tracker=ExitTracker(run(),bundle().metadata,p,1.)
        self.assertIsNone(tracker.observe(loc(p=pose(1.5,.5)),readiness(),stamp()))
        self.assertIsNone(tracker.observe(loc(p=pose(2,.5)),readiness(state=c.Holding.UNKNOWN),stamp()))
        self.assertIsNone(tracker.observe(loc(p=pose(1.7,.5,error_m=.2)),readiness(),stamp()))

    def test_T06_return_reexit_and_restart_share_dedup_key(self):
        tracker=ExitTracker(run(),bundle().metadata,profile(),1.)
        first=tracker.observe(loc(p=pose(2,.5)),readiness(),stamp())
        self.assertIsNone(tracker.observe(loc(p=pose(.5,.5)),readiness(),stamp()))
        self.assertIsNone(tracker.observe(loc(p=pose(2,.5)),readiness(),stamp()))
        fresh=ExitTracker(replace(run(),envelope=replace(run().envelope,instance_id='new',control_epoch=2)),bundle().metadata,profile(),1.)
        newer=replace(loc(p=pose(2,.5)),envelope=replace(loc().envelope,instance_id='new',control_epoch=2))
        ready=replace(readiness(),envelope=replace(readiness().envelope,instance_id='new',control_epoch=2))
        self.assertEqual(fresh.observe(newer,ready,stamp()).envelope.event_id,first.envelope.event_id)
        restored=ExitTracker(run(),bundle().metadata,profile(),1.,committed=first)
        self.assertIsNone(restored.observe(loc(p=pose(2,.5)),readiness(),stamp()))

    def test_exit_rejects_expired_future_or_untrusted_evidence(self):
        tracker=ExitTracker(run(),bundle().metadata,profile(),1.)
        for value in (replace(loc(p=pose(2,.5)),trusted=c.Fact.UNKNOWN),
                      replace(loc(p=pose(2,.5)),evidence=(proof(captured_at=stamp(90)),)),
                      replace(loc(p=pose(2,.5)),evidence=(proof(captured_at=stamp(111)),))):
            self.assertIsNone(tracker.observe(value,readiness(),stamp()))
