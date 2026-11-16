"""P02 MappingPort integration with real C0 records and offline inputs."""
from publication_support import simulated_space_readback
import math
import unittest
from dataclasses import replace
from test_p02_support import c, m, env, pose, poly, proof, bundle, policy, transforms, Clock, occupancy, space

try:
    from treasure_auto.mapping import service as svc
    from treasure_auto.mapping.localization import LocalizationTracker
except ImportError:
    svc = None


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(svc, 'P02 MappingPort implementation missing')
        self.clock = Clock()
        self.tracker = LocalizationTracker(env(), policy(), self.clock, 'base_link')
        self.profile = svc.FootprintProfile(poly(-.2,-.15,.2,.15), None, 0, m.Measurement.SIMULATED)
        self.service = svc.MappingService(bundle(), self.tracker, self.profile, 'mapper', 'localizer', space_readback=simulated_space_readback)
        # The positive cases supply a known empty tool; production defaults to UNKNOWN.
        self.service.observe_holding(c.HoldingSnapshot(c.Holding.EMPTY,None,None,(proof(10,'empty'),)),env(producer=c.Owner.P07))

    def ready(self, yaw=0, stopped=False):
        self.tracker.initialize(pose(yaw=yaw))
        for t, heading in zip((10,10.2,10.4), (yaw,yaw,yaw)):
            self.clock.t = t
            self.tracker.observe(*transforms(t,yaw=heading),
                stopped=c.Fact.CONFIRMED if stopped else c.Fact.UNKNOWN,
                stop_evidence=proof(t,'stopped-'+str(t)) if stopped else None)

    def competition(self):
        return self.service.set_mode(svc.Mode.COMPETITION, ('localizer',), c.ClockStamp(20,c.ClockBasis.MONOTONIC,'sim-clock','sim-boot'))

    def query(self, name='board', purpose='observe', **kw):
        e = env(self.clock.t, 'query', base_motion_epoch=self.tracker.base_motion_epoch, producer=c.Owner.P03, **kw)
        return m.WaypointQuery(e,name,purpose)

    def purpose_ready(self):
        self.service.set_purpose_evidence(svc.PurposeEvidence('board','observe','map-v1',0,
            self.tracker.base_motion_epoch,c.Fact.CONFIRMED,(proof(self.clock.t,'view'),)))

    def test_R01_conflicting_sources_and_timeout_rejected(self):
        for sources in (('mapper','localizer'),(),('other',),('localizer','localizer')):
            self.assertFalse(self.service.set_mode(svc.Mode.COMPETITION,sources,self.clock()).accepted)
        result = self.service.set_mode(svc.Mode.COMPETITION,('localizer',),self.clock())
        self.assertEqual(result.reason.code,c.ErrorCode.TIMEOUT)
        self.assertTrue(self.competition().accepted)
        self.assertFalse(self.service.set_mode(svc.Mode.COMPETITION,('mapper','localizer'),env().expires_at).accepted)
        self.assertEqual(self.service.mode,svc.Mode.STOPPED)

    def test_R01_unmeasured_bundle_never_enters_competition(self):
        from treasure_auto.mapping.storage import with_version
        b = bundle()
        self.service.install(with_version(replace(b, metadata=replace(b.metadata, measurement=m.Measurement.UNMEASURED)),'map-v2'))
        self.assertFalse(self.competition().accepted)

    def test_R01_competition_freezes_map_and_point_records(self):
        self.ready(stopped=True)
        self.assertTrue(self.competition().accepted)
        self.assertFalse(self.service.install(bundle()).accepted)
        self.assertFalse(self.service.record(m.WaypointResult(self.query().envelope,'new','transit',pose(),c.Fact.UNKNOWN,None)).accepted)

    def test_R06_named_and_purpose_queries_require_fresh_localization_and_purpose(self):
        self.competition()
        self.ready()
        shared = space(t=self.clock.t)
        self.assertEqual(self.service.names_for('observe'),('board',))
        self.assertFalse(self.service.query(self.query(), shared).valid is c.Fact.CONFIRMED)
        self.purpose_ready()
        self.assertEqual(self.service.query(self.query(),shared).valid,c.Fact.CONFIRMED)
        self.assertEqual(self.service.query(self.query('absent'),shared).valid,c.Fact.UNKNOWN)
        self.assertEqual(self.service.query(self.query(purpose='place'),shared).valid,c.Fact.UNKNOWN)
        self.clock.t = 12
        self.assertEqual(self.service.query(self.query(),shared).valid,c.Fact.UNKNOWN)

    def test_T05_motion_invalidates_purpose_and_keeps_shared_occupancy(self):
        self.competition()
        self.ready()
        self.purpose_ready()
        shared = space((occupancy(box=(4,4,4.1,4.1)),),t=self.clock.t)
        serialized = c.dumps(shared)
        self.assertEqual(self.service.query(self.query(),shared).valid,c.Fact.CONFIRMED)
        old_request = self.query()
        self.clock.t = 10.6
        self.tracker.observe(*transforms(10.6,x=2.1))
        self.assertEqual(self.service.query(old_request,shared).valid,c.Fact.UNKNOWN)
        self.assertEqual(self.service.query(self.query(),shared).valid,c.Fact.UNKNOWN)
        self.assertEqual(c.dumps(shared),serialized)

    def test_T03_shared_unknown_and_uncommitted_or_wrong_version_rejected(self):
        self.competition()
        self.ready()
        self.purpose_ready()
        good = space(t=self.clock.t)
        for shared in (space((occupancy(),),t=self.clock.t),
                       replace(good,envelope=replace(good.envelope,producer=c.Owner.P10B)),
                       replace(good,version=1,envelope=replace(good.envelope,versions=replace(good.envelope.versions,space=1)))):
            self.assertEqual(self.service.query(self.query(),shared).valid,c.Fact.UNKNOWN)
        self.service.space_readback = None
        self.assertEqual(self.service.query(self.query(),good).valid,c.Fact.UNKNOWN)

    def test_T03_unconfigured_footprint_and_unknown_holding_rejected(self):
        self.competition()
        self.ready()
        self.purpose_ready()
        self.service.observe_holding(c.HoldingSnapshot(c.Holding.UNKNOWN,None,None,()),env(producer=c.Owner.P07))
        self.assertEqual(self.service.query(self.query(),space(t=self.clock.t)).valid,c.Fact.UNKNOWN)

    def test_T04_cross_pi_stable_record_uses_circular_mean(self):
        self.tracker.initialize(pose(yaw=math.pi))
        for t,yaw in ((10,math.pi-.01),(10.2,-math.pi+.01),(10.4,math.pi-.005)):
            self.clock.t=t
            self.tracker.observe(*transforms(t,yaw=yaw),stopped=c.Fact.CONFIRMED,stop_evidence=proof(t,'stop'+str(t)))
        request=m.WaypointResult(self.query().envelope,'park_A','place',pose(yaw=math.pi),c.Fact.UNKNOWN,None)
        self.assertTrue(self.service.record(request).accepted)
        draft = self.service.export_draft('map-v2')
        point=next(p for p in draft.waypoints if p.name=='park_A')
        self.assertLess(abs(abs(point.pose.yaw)-math.pi),.01)
        self.assertEqual(point.envelope.versions.map,'map-v2')
        self.assertNotEqual(point.pose,draft.metadata.regions[0].placement_center)
        self.assertTrue(dict(draft.waypoint_evidence)['park_A'])

    def test_R04_record_without_actual_stop_or_requested_pose_mismatch_rejected(self):
        self.ready()
        request=m.WaypointResult(self.query().envelope,'park','place',pose(),c.Fact.UNKNOWN,None)
        self.assertFalse(self.service.record(request).accepted)
        self.clock.t=11
        self.tracker.initialize(pose())
        for t in (11,11.2,11.4):
            self.clock.t=t
            self.tracker.observe(*transforms(t),stopped=c.Fact.CONFIRMED,stop_evidence=proof(t,'stop'+str(t)))
        request=replace(request,envelope=self.query().envelope,pose=pose(4,4))
        self.assertFalse(self.service.record(request).accepted)

    def test_R06_record_deduplicates_and_requires_explicit_overwrite(self):
        self.ready(stopped=True)
        request=m.WaypointResult(self.query().envelope,'park','place',pose(),c.Fact.UNKNOWN,None)
        self.assertTrue(self.service.record(request).accepted)
        self.assertTrue(self.service.record(request).accepted)
        self.assertEqual(len(self.service.record_history),1)
        changed=replace(request,pose=pose(2.01,2))
        self.assertEqual(self.service.record(changed).reason.code,c.ErrorCode.ID_CONFLICT)
        changed=replace(request,envelope=replace(request.envelope,event_id='new'))
        self.assertFalse(self.service.record(changed).accepted)
        self.assertTrue(self.service.replace_waypoint(changed).accepted)
        self.assertEqual(len(self.service.record_history),2)

    def test_R06_region_query_returns_boundary_and_version_change_event(self):
        area=self.service.region(m.Area.A)
        self.assertEqual(area.polygon_m,poly(0,0,.45,.45))
        self.assertEqual(area.placement_center,pose(.225,.225))
        from treasure_auto.mapping.storage import with_version
        self.assertTrue(self.service.install(with_version(bundle(),'map-v2')).accepted)
        events=self.service.take_events()
        self.assertTrue(any(type(e) is m.MapMetadata and e.map_version=='map-v2' for e in events))
        self.assertEqual(self.service.localization().trusted,c.Fact.UNKNOWN)

    def test_R06_public_results_roundtrip_without_extending_C0_registry(self):
        from treasure_auto.contracts import core
        before=set(core._RECORDS)
        for output in (self.service.map_metadata(),self.service.localization(),
                       self.service.query(self.query(),space())):
            self.assertEqual(c.loads(c.dumps(output)),output)
        self.assertEqual(set(core._RECORDS),before)
