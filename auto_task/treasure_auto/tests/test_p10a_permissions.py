import unittest
from dataclasses import replace
from test_p10a_support import *


def stillness(req, scene=1):
    cert = s.stillness_certificate()
    env = replace(req.space.envelope, emitted_at=req.envelope.emitted_at)
    ev = req.readiness.evidence[0]
    return replace(cert, envelope=env, scene_epoch=scene,
                   evidence=(replace(ev,captured_at=s.sensor_stamp(20)),
                             replace(ev,evidence_id='sim-ev-2',captured_at=s.sensor_stamp(20.5))))


class PermissionTests(unittest.TestCase):
    def test_T05_unknown_posture_holding_stop_or_device_never_grants(self):
        for kind in ('posture', 'holding', 'device', 'queue', 'stopped', 'location', 'echo'):
            rt, clock = started(); req = request(rt, clock)
            if kind == 'posture': req = replace(req, readiness=replace(req.readiness,posture_confirmed=c.Fact.UNKNOWN))
            if kind == 'holding': req = replace(req, readiness=replace(req.readiness,holding=c.HoldingSnapshot(c.Holding.UNKNOWN,None,None,())))
            if kind == 'device': req = replace(req, health=replace(req.health,capabilities=()))
            if kind == 'queue': req = replace(req, old_stop=replace(req.old_stop,old_queue_empty=c.Fact.UNKNOWN))
            if kind == 'stopped': req = replace(req, old_stop=replace(req.old_stop,physically_stopped=c.Fact.UNKNOWN))
            if kind == 'location': req = replace(req,localization=replace(req.localization,trusted=c.Fact.UNKNOWN))
            if kind == 'echo': req = replace(req,readiness=replace(req.readiness,evidence=(replace(req.readiness.evidence[0],kind=c.EvidenceKind.ECHO),)))
            with self.subTest(kind=kind): self.assertIsNone(rt.request_permission(req).permit)

    def test_missing_mapping_allows_read_board_but_not_central_work(self):
        rt, clock = started(mapping=False)
        self.assertIsNone(rt.request_permission(request(rt, clock,purpose='central_pick')).permit)
        self.assertIsNotNone(rt.request_permission(request(rt, clock,seq=20)).permit)

    def test_receipt_must_bind_exact_registered_intent(self):
        rt, clock = started(); req = request(rt, clock)
        wrong = replace(req.intent_receipt,intent_sha256='0'*64)
        self.assertIsNone(rt.request_permission(replace(req,intent_receipt=wrong)).permit)
        self.assertIsNotNone(rt.request_permission(req).permit)

    def test_future_old_version_owner_and_simulated_origin_mismatch_rejected(self):
        for field in ('future', 'version', 'owner', 'origin', 'boot'):
            rt, clock = started(); req = request(rt, clock)
            env = req.readiness.envelope
            if field == 'future': env = replace(env,emitted_at=s.stamp(101))
            if field == 'version': env = replace(env,versions=replace(env.versions,space=3))
            if field == 'owner': env = replace(env,producer=c.Owner.P09)
            if field == 'origin': env = replace(env,origin=c.Origin.LIVE)
            if field == 'boot': env = replace(env,emitted_at=s.stamp(100,'other'),expires_at=s.stamp(110,'other'))
            with self.subTest(field=field): self.assertIsNone(rt.request_permission(replace(req,readiness=replace(req.readiness,envelope=env))).permit)

    def test_permission_expires_with_shortest_physical_evidence(self):
        rt, clock = started(); req = request(rt, clock)
        ev = replace(req.readiness.evidence[0],valid_until=s.stamp(100.25))
        req = replace(req,readiness=replace(req.readiness,evidence=(ev,)))
        permit = rt.request_permission(req).permit
        self.assertEqual(permit.envelope.expires_at.seconds,100.25)
        clock.advance(.25)
        self.assertFalse(rt.authorize_output(permit,req.envelope,req.source,real_output=False))

    def test_T05_stillness_500ms_is_necessary_but_not_sufficient(self):
        for bad in ('short', 'duplicate', 'posture', 'missing_target', 'missing'):
            rt, clock = started(); req = request(rt, clock, m.MotionSource.RECOVERY)
            rt.scene_changed(req.space,1,('brick-1',))
            cert=stillness(req)
            if bad == 'short': cert=replace(cert,capture_end=s.sensor_stamp(20.499))
            if bad == 'duplicate': cert=replace(cert,observation_ids=('x','x'))
            if bad == 'missing_target': cert=replace(cert,covered_target_ids=('other',))
            if bad == 'missing': cert=None
            if bad == 'posture': req=replace(req,readiness=replace(req.readiness,posture_confirmed=c.Fact.UNKNOWN))
            with self.subTest(bad=bad): self.assertIsNone(rt.resume(replace(req,stillness=cert)).permit)
        rt, clock=started(); req=request(rt,clock,m.MotionSource.RECOVERY)
        rt.scene_changed(req.space,1,('brick-1',))
        permit=rt.resume(replace(req,stillness=stillness(req))).permit
        self.assertIsNotNone(permit)
        rt.scene_changed(req.space,2,('brick-1',))
        self.assertFalse(rt.authorize_output(permit,req.envelope,req.source,real_output=False))


if __name__=='__main__': unittest.main()
