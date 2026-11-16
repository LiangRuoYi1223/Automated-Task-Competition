from publication_support import simulated_space_readback
from dataclasses import replace
from test_p06_paths import PathFixture
from test_p06_fixtures import *


class ServiceTests(PathFixture):
    def test_T04_base_motion_invalidates_entire_plan_but_never_erases_occupancy(self):
        obs=replace(occupancy('far'),footprint_m=square(2,2,.1))
        self.ctx=replace(self.ctx,space=space((obs,)))
        before=c.dumps(self.ctx.space);r=self.svc.plan(self.request())
        self.assertIsNotNone(r.plan,r.result.reason)
        self.ctx=replace(self.ctx,consumer=replace(self.ctx.consumer,base_motion_epoch=1),
                         localization=replace(self.ctx.localization,base_motion_epoch=1))
        self.assertFalse(self.svc.validate_plan(r.plan).accepted)
        self.assert_rejected(self.svc.plan(self.request()))
        self.assertEqual(c.dumps(self.ctx.space),before)

    def test_R06_camera_tool_space_or_restore_invalidates_old_path(self):
        r=self.svc.plan(self.request());self.assertIsNotNone(r.plan,r.result.reason)
        original=self.ctx
        for changes in (dict(tool_transform=replace(self.tool,version='new-tool')),
                        dict(camera_to_base=replace(self.ctx.camera_to_base,version='new-camera')),
                        dict(space=replace(self.ctx.space,occupancies=(occupancy('new-obstacle'),))),
                        dict(arm_geometry_version='new-arm')):
            self.ctx=replace(original,**changes)
            with self.subTest(changes=changes):self.assertFalse(self.svc.validate_plan(r.plan).accepted)
        self.ctx=original
        restored=self.g.GeometryService(self.cfg,lambda:self.ctx,self.verify,self.audit.append,space_readback=simulated_space_readback)
        self.assertFalse(restored.validate_plan(r.plan).accepted)

    def test_R06_duplicate_request_returns_same_result_without_rechecking_hardware(self):
        req=self.request();a=self.svc.plan(req);count=len(self.calls);b=self.svc.plan(req)
        self.assertIsNotNone(a.plan,a.result.reason)
        self.assertEqual(a,b);self.assertEqual(len(self.calls),count)
        self.assert_rejected(self.svc.plan(replace(req,calibration=replace(req.calibration,fit_error_px=2.))),c.ErrorCode.ID_CONFLICT)

    def test_R06_expired_duplicate_does_not_gain_new_lifetime(self):
        req=self.request();r=self.svc.plan(req);self.assertIsNotNone(r.plan,r.result.reason)
        self.ctx=replace(self.ctx,consumer=replace(self.ctx.consumer,now=stamp(120)))
        self.assert_rejected(self.svc.plan(req),c.ErrorCode.EXPIRED)

    def test_R06_inputs_change_during_verifier_or_audit_cannot_publish_plan(self):
        def change(cert):
            self.ctx=replace(self.ctx,stopped=c.Fact.UNKNOWN)
            return cert
        self.cert_override=change
        self.assert_rejected(self.svc.plan(self.request()))

    def test_R06_audit_failure_does_not_publish_success_or_cache_it(self):
        def fail(_):raise OSError('simulated disk failure')
        self.svc.audit_sink=fail
        self.assert_rejected(self.svc.plan(self.request()),c.ErrorCode.STORAGE_FAILURE)
        self.svc.audit_sink=self.audit.append
        self.assertIsNotNone(self.svc.plan(self.request()).plan)

    def test_R01_stale_or_wrong_clock_image_and_old_identity_rejected(self):
        original=self.ctx
        for changes in (dict(capture_now=stamp(111)),
                        dict(capture_floor=replace(self.ctx.capture_floor,seconds=21)),
                        dict(image=replace(self.image,capture_pose=pose(1.,0.))),
                        dict(localization=replace(self.ctx.localization,trusted=c.Fact.UNKNOWN))):
            self.ctx=replace(original,**changes)
            with self.subTest(changes=changes):self.assert_rejected(self.svc.plan(self.request()))

    def test_R06_result_expiry_is_bounded_by_weakest_fresh_evidence(self):
        ev=replace(self.ev,valid_until=stamp(111.5));self.ctx=replace(self.ctx,stop_evidence=(ev,))
        r=self.svc.plan(self.request());self.assertIsNotNone(r.plan,r.result.reason)
        self.assertLessEqual(r.envelope.expires_at.seconds,111.5)
        self.ctx=replace(self.ctx,consumer=replace(self.ctx.consumer,now=stamp(111.5)))
        self.assertFalse(self.svc.validate_plan(r.plan).accepted)

    def test_R06_versioned_report_and_domains_required_no_fallback(self):
        original=self.ctx
        for changes in (dict(image_report=None),dict(image_report=replace(self.ctx.image_report,accepted=False)),
                        dict(image_report=replace(self.ctx.image_report,version='old')),
                        dict(image_report=replace(self.ctx.image_report,domain_mm=((0.,0.,0.),(1.,1.,1.))))):
            self.ctx=replace(original,**changes)
            with self.subTest(changes=changes):self.assert_rejected(self.svc.plan(self.request()))
