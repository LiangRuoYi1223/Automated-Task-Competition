"""Boundary regressions discovered while reviewing cross-component behavior."""
from publication_support import simulated_space_readback
import tempfile
import unittest
from dataclasses import replace
from test_p02_support import c, m, env, proof, pose, poly, bundle, policy, transforms, Clock, space
from treasure_auto.mapping import localization as loc, service as svc, storage


def system():
    clock = Clock()
    tracker = loc.LocalizationTracker(env(),policy(),clock,'base_link')
    service = svc.MappingService(bundle(),tracker,
            svc.FootprintProfile(poly(-.2,-.15,.2,.15),None,0,m.Measurement.SIMULATED),'mapper','localizer',space_readback=simulated_space_readback)
    return clock,tracker,service


def confirmed(clock,tracker,start=10):
    tracker.initialize(pose())
    for t in (start,start+.2,start+.4):
        clock.t=t
        tracker.observe(*transforms(t))


class ReviewTests(unittest.TestCase):
    def test_R01_first_publisher_switch_revokes_preparation_localization(self):
        clock,tracker,service=system()
        confirmed(clock,tracker)
        self.assertTrue(service.set_mode(svc.Mode.COMPETITION,('localizer',),env().expires_at).accepted)
        self.assertEqual(tracker.snapshot().trusted,c.Fact.UNKNOWN)

    def test_R04_confirmed_snapshot_expires_with_underlying_observation(self):
        clock,tracker,service=system()
        confirmed(clock,tracker)
        clock.t=11.3
        result=tracker.snapshot()
        self.assertEqual(result.trusted,c.Fact.CONFIRMED)
        self.assertLessEqual(result.envelope.expires_at.seconds,11.4)
        clock.t=11.4
        self.assertEqual(tracker.snapshot().trusted,c.Fact.UNKNOWN)

    def test_R05_small_map_correction_invalidates_geometry_too(self):
        clock,tracker,service=system()
        confirmed(clock,tracker)
        old=tracker.base_motion_epoch
        clock.t=10.6
        result=tracker.observe(*transforms(10.6,correction=pose(.01,0)))
        self.assertGreater(result.base_motion_epoch,old)

    def test_R03_capture_future_in_own_domain_is_rejected(self):
        clock,tracker,service=system()
        a,b=transforms()
        a=replace(a,evidence=replace(a.evidence,captured_at=proof(11).captured_at))
        b=replace(b,evidence=replace(b.evidence,captured_at=proof(11).captured_at))
        result=loc.resolve_chain(a,b,env(),clock(),policy(),'base_link')
        self.assertIsNone(result.pose)

    def test_R02_region_must_fit_field_and_placement_center_not_park(self):
        b=bundle()
        r=replace(b.metadata.regions[0],polygon_m=poly(10,10,11,11),placement_center=pose(10.5,10.5))
        with self.assertRaises(c.ContractError):
            replace(b,metadata=replace(b.metadata,regions=(r,)+b.metadata.regions[1:]))

    def test_R02_central_polygon_must_fit_field(self):
        with self.assertRaises(c.ContractError):
            replace(bundle(),central=poly(7,7,8,8))

    def test_R01_same_version_cannot_replace_different_static_content(self):
        clock,tracker,service=system()
        b=bundle()
        changed=replace(b,waypoints=(replace(b.waypoints[0],pose=pose(3,3)),))
        self.assertFalse(service.install(changed).accepted)

    def test_R06_expired_P07_holding_cannot_keep_empty_footprint(self):
        clock,tracker,service=system()
        service.set_mode(svc.Mode.COMPETITION,('localizer',),env().expires_at)
        confirmed(clock,tracker)
        holding=c.HoldingSnapshot(c.Holding.EMPTY,None,None,(proof(10,'empty'),))
        # Existing enum-only setter is the bug: add a bound C0 input hook.
        self.assertTrue(hasattr(service,'observe_holding'),'P07 envelope/evidence hook required')
        service.observe_holding(holding,env(10,producer=c.Owner.P07))
        service.set_purpose_evidence(svc.PurposeEvidence('board','observe','map-v1',0,
            tracker.base_motion_epoch,c.Fact.CONFIRMED,(proof(10.4,'view'),)))
        def query():
            return service.query(m.WaypointQuery(env(clock.t,producer=c.Owner.P03,
                      base_motion_epoch=tracker.base_motion_epoch),'board','observe'),space(t=clock.t))
        self.assertEqual(query().valid,c.Fact.CONFIRMED)
        self.assertLessEqual(query().envelope.expires_at.seconds,11)
        clock.t=11.1
        # Refresh source inventory only; the old holding observation stays expired.
        service.set_mode(svc.Mode.COMPETITION,('localizer',),env().expires_at)
        self.assertEqual(query().valid,c.Fact.UNKNOWN)

    def test_R01_write_failure_does_not_publish_or_modify_old_version(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            storage.save_bundle(bundle(),d,'map-v1')
            with patch.object(storage.os,'fsync',side_effect=OSError('disk full')):
                with self.assertRaises(c.ContractError) as error:
                    storage.save_bundle(bundle(),d,'map-v2')
            self.assertEqual(error.exception.code,c.ErrorCode.STORAGE_FAILURE)
            self.assertEqual(storage.load_bundle(__import__('pathlib').Path(d)/'map-v1/map.json','map-v1').metadata.map_version,'map-v1')
            self.assertFalse((__import__('pathlib').Path(d)/'map-v2').exists())

    def test_R03_malformed_transform_returns_reason_not_attribute_error(self):
        a,b=transforms()
        result=loc.resolve_chain(replace(a,pose=None),b,env(),Clock()(),policy(),'base_link')
        self.assertIsNone(result.pose)
        self.assertIsNotNone(result.reason)

    def test_R04_each_TF_event_and_evidence_identity_are_immutable(self):
        for reuse in ('event','evidence'):
            clock,tracker,service=system()
            tracker.initialize(pose())
            for t in (10,10.2,10.4):
                clock.t=t
                a,b=transforms(t)
                if reuse=='event':
                    a=replace(a,envelope=replace(a.envelope,event_id='reused-map-event'))
                else:
                    a=replace(a,evidence=replace(a.evidence,evidence_id='same-map-proof',uri='sim://same-map-proof'))
                result=tracker.observe(a,b)
            self.assertEqual(result.trusted,c.Fact.UNKNOWN)
            self.assertEqual(result.reason.code,c.ErrorCode.ID_CONFLICT)

    def test_R04_both_TF_envelope_expirations_bound_trust(self):
        clock,tracker,service=system()
        tracker.initialize(pose())
        for t in (10,10.2,10.4):
            clock.t=t
            a,b=transforms(t)
            a=replace(a,envelope=replace(a.envelope,expires_at=replace(a.envelope.emitted_at,seconds=t+.05)))
            tracker.observe(a,b)
        clock.t=10.6
        self.assertEqual(tracker.snapshot().trusted,c.Fact.UNKNOWN)

    def test_R04_source_capture_age_consumes_output_lifetime(self):
        clock,tracker,service=system()
        clock.t=9
        tracker.initialize(pose())
        for t in (10,10.2,10.4):
            clock.t=t
            a,b=transforms(t)
            a=replace(a,evidence=replace(a.evidence,captured_at=proof(t-.9).captured_at))
            b=replace(b,evidence=replace(b.evidence,captured_at=proof(t-.9).captured_at))
            tracker.observe(a,b)
        self.assertEqual(tracker.snapshot().trusted,c.Fact.CONFIRMED)
        clock.t=10.6
        self.assertEqual(tracker.snapshot().trusted,c.Fact.UNKNOWN)

    def test_R05_old_empty_fact_cannot_replace_newer_held_fact(self):
        clock,tracker,service=system()
        empty=c.HoldingSnapshot(c.Holding.EMPTY,None,None,(proof(10,'empty'),))
        old=env(10,'old-empty',producer=c.Owner.P07)
        self.assertTrue(service.observe_holding(empty,old).accepted)
        clock.t=10.4
        held=c.HoldingSnapshot(c.Holding.HELD,'brick',None,(proof(10.4,'held'),))
        self.assertTrue(service.observe_holding(held,env(10.4,'new-held',producer=c.Owner.P07)).accepted)
        clock.t=10.5
        service.observe_holding(empty,old)
        self.assertNotEqual(service.holding,c.Holding.EMPTY)

    def test_R04_pre_initialization_captures_cannot_confirm_new_seed(self):
        clock,tracker,service=system()
        tracker.initialize(pose())
        for t,capture in ((10,9.4),(10.1,9.6),(10.2,9.8)):
            clock.t=t
            a,b=transforms(t)
            a=replace(a,evidence=replace(a.evidence,captured_at=proof(capture).captured_at))
            b=replace(b,evidence=replace(b.evidence,captured_at=proof(capture).captured_at))
            result=tracker.observe(a,b)
        self.assertEqual(result.trusted,c.Fact.UNKNOWN)
        self.assertEqual(result.reason.code,c.ErrorCode.EXPIRED)

    def test_R06_new_holding_action_can_restart_its_sequence(self):
        clock,tracker,service=system()
        held=c.HoldingSnapshot(c.Holding.HELD,'brick',None,(proof(10,'held',action_id='grasp',target_id='brick'),))
        self.assertTrue(service.observe_holding(held,env(10,'held',producer=c.Owner.P07,
             request_id='work',action_id='grasp',sequence=20,target_id='brick')).accepted)
        clock.t=10.4
        empty=c.HoldingSnapshot(c.Holding.EMPTY,None,None,(proof(10.4,'empty',action_id='release'),))
        result=service.observe_holding(empty,env(10.4,'empty',producer=c.Owner.P07,
             request_id='work',action_id='release',sequence=0))
        self.assertTrue(result.accepted,result.reason)
        self.assertEqual(service.holding,c.Holding.EMPTY)
