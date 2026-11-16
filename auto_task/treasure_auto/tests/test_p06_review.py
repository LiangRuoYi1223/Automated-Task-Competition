"""Minimal reproductions from independent read-only review."""
from publication_support import simulated_space_readback
from dataclasses import replace
from test_p06_paths import PathFixture
from test_p06_fixtures import *


class ReviewTests(PathFixture):
    def test_review_01_report_binds_actual_camera_and_all_transform_contents(self):
        req=self.request();k=list(req.calibration.intrinsics);k[2]+=10.
        self.assert_rejected(self.svc.plan(replace(req,calibration=replace(req.calibration,intrinsics=tuple(k)))))
        # A changed extrinsic with an unchanged version/report must also fail on first use.
        a=list(self.ctx.base_to_arm.matrix);a[3]=10.
        self.ctx=replace(self.ctx,base_to_arm=replace(self.ctx.base_to_arm,matrix=tuple(a)))
        self.svc=self.g.GeometryService(self.cfg,lambda:self.ctx,self.verify,self.audit.append,space_readback=simulated_space_readback)
        self.assert_rejected(self.svc.plan(self.request()))

    def test_review_02_image_envelope_expiry_bounds_plan_and_revalidation(self):
        self.ctx=replace(self.ctx,image=replace(self.image,envelope=replace(self.image.envelope,expires_at=stamp(111.1))))
        r=self.svc.plan(self.request());self.assertIsNotNone(r.plan,r.result.reason)
        self.assertLessEqual(r.envelope.expires_at.seconds,111.1)
        self.ctx=replace(self.ctx,consumer=replace(self.ctx.consumer,now=stamp(111.2)))
        self.assertFalse(self.svc.validate_plan(r.plan).accepted)

    def test_review_03_start_uncertainty_must_fit_inside_original_sweep_bound(self):
        wall=replace(occupancy('wall',z0=.19,z1=.21),footprint_m=square(-.018,-.001,.002))
        self.ctx=replace(self.ctx,space=space((wall,)))
        r=self.svc.plan(self.request());self.assertIsNotNone(r.plan,r.result.reason)
        self.ctx=replace(self.ctx,start=replace(self.ctx.start,x=-10.4,error_mm=10.))
        self.assertFalse(self.svc.validate_plan(r.plan).accepted)

    def test_review_04_public_ratio_never_rounds_exact_80_percent_to_passing_side(self):
        lower=replace(occupancy(),footprint_m=square(.3,.3))
        upper=replace(occupancy('upper',z0=.1,z1=.2),footprint_m=square(.32,.3))
        ev=evidence(target_id='upper')
        req=m.SupportGeometryRequest(envelope(c.Owner.P08,target_id='upper'),upper,(lower,),space((upper,lower)),(ev,))
        proof=self.i.ContactProof('upper',('lower',),0,c.Fact.CONFIRMED,c.Fact.CONFIRMED,(ev,))
        r=module(self,'support').compute_support(req,proof,.001)
        self.assertLessEqual(r.ratio_lower_bound,.8)
        self.assertAlmostEqual(r.ratio_lower_bound,.8)

    def test_review_config_constructor_cannot_bypass_numeric_validation(self):
        cfg=replace(self.cfg,tool_error_mm=-1.)
        with self.assertRaises(c.ContractError):cfg.require_ready(c.Origin.SIMULATED)
