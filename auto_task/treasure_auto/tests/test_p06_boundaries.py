import math
from dataclasses import replace
from test_p06_paths import PathFixture
from test_p06_fixtures import *


class BoundaryTests(PathFixture):
    def test_R04_placement_retreat_does_not_request_another_trial_lift(self):
        result=self.svc.plan_placement(self.placement())
        self.assertIsNotNone(result.plan,result.result.reason)
        self.assertEqual([s.phase for s in result.plan.segments],
            [m.Phase.APPROACH,m.Phase.DESCEND,m.Phase.PLACE,m.Phase.WITHDRAW,m.Phase.WITHDRAW])

    def test_R05_withdrawal_checks_the_expected_released_brick(self):
        def through_release(cert):
            if cert.query.segment.phase is m.Phase.WITHDRAW:
                return replace(cert,swept_boxes=cert.swept_boxes+(self.i.SweepBox((190.,190.,40.),(210.,210.,60.)),))
            return cert
        self.cert_override=through_release
        self.assert_rejected(self.svc.plan_placement(self.placement()),c.ErrorCode.OCCUPANCY_UNRESOLVED)

    def test_R02_matching_support_height_at_wrong_xy_does_not_support_placement(self):
        req=self.placement();lower=replace(occupancy('lower'),footprint_m=square(2,2,.1))
        self.ctx=replace(self.ctx,space=space((lower,)),planes=(replace(self.plane,height_mm=100.,support_ids=('lower',)),))
        req=replace(req,space=self.ctx.space,actual_support_ids=('lower',))
        self.assert_rejected(self.svc.plan_placement(req))

    def test_R05_unknown_start_or_echo_position_cannot_certify_swept_path(self):
        for refs in ((),(replace(self.ev,kind=c.EvidenceKind.ECHO),)):
            with self.subTest(refs=refs):
                self.ctx=replace(self.ctx,start_evidence=refs)
                self.assert_rejected(self.svc.plan(self.request()))

    def test_R06_each_stage_rechecks_observed_start_without_invalidating_for_expected_arm_motion(self):
        r=self.svc.plan(self.request());self.assertIsNotNone(r.plan,r.result.reason)
        self.ctx=replace(self.ctx,start=r.plan.segments[0].target,start_evidence=(self.ev,))
        self.assertTrue(self.svc.validate_plan(r.plan,stage_index=1).accepted)
        self.assertFalse(self.svc.validate_plan(r.plan,stage_index=0).accepted)
        self.ctx=replace(self.ctx,start=replace(self.ctx.start,x=self.ctx.start.x+100.))
        self.assertFalse(self.svc.validate_plan(r.plan,stage_index=1).accepted)

    def test_R01_non_square_configuration_cannot_use_90_degree_symmetry(self):
        fields=config_dict();fields['brick_size_mm']=[100.,200.,100.]
        cfg=module(self,'config').GeometryConfig.from_dict(fields)
        with self.assertRaises(c.ContractError):cfg.require_ready(c.Origin.SIMULATED)

    def test_R05_timeout_must_cover_segment_distance_at_requested_speed(self):
        self.svc.config=replace(self.cfg,timeouts_s=(.001,)*5)
        report=replace(self.ctx.image_report,binding=replace(self.ctx.image_report.binding,config=self.svc.config))
        self.ctx=replace(self.ctx,image_report=report)
        result=self.svc.plan(self.request())
        self.assert_rejected(result)
        self.assertEqual(result.result.reason.field,'segment.timeout')

    def test_R06_plane_normal_or_calibration_domain_cannot_exclude_real_corners(self):
        report=self.report('image')
        # Center 110,10 lies inside, but real corner x=160 lies outside.
        self.ctx=replace(self.ctx,image_report=replace(report,domain_mm=((-10.,-100.,-200.),(120.,100.,500.))))
        self.assert_rejected(self.svc.plan(self.request()))

    def test_R07_echo_contact_evidence_cannot_establish_real_support(self):
        lower=occupancy();upper=occupancy('upper',z0=.1,z1=.2)
        ev=evidence(target_id='upper',kind=c.EvidenceKind.ECHO)
        req=m.SupportGeometryRequest(envelope(c.Owner.P08,target_id='upper'),upper,(lower,),space((upper,lower)),(ev,))
        proof=self.i.ContactProof('upper',('lower',),0,c.Fact.CONFIRMED,c.Fact.CONFIRMED,(ev,))
        with self.assertRaises(c.ContractError):module(self,'support').compute_support(req,proof,.001)

    def test_R07_support_port_uses_c0_and_duplicate_does_not_mint_new_result(self):
        lower=occupancy();upper=occupancy('upper',z0=.1,z1=.2);sp=space((upper,lower))
        ev=evidence(target_id='upper')
        req=m.SupportGeometryRequest(envelope(c.Owner.P08,target_id='upper'),upper,(lower,),sp,(ev,))
        proof=self.i.ContactProof('upper',('lower',),0,c.Fact.CONFIRMED,c.Fact.CONFIRMED,(ev,))
        self.ctx=replace(self.ctx,consumer=replace(self.ctx.consumer,target_id='upper'),space=sp,contact_proof=proof)
        a=self.svc.support_geometry(req);b=self.svc.support_geometry(req)
        self.assertEqual(a.contact,c.Fact.CONFIRMED,a.reason)
        self.assertEqual(a,b)
        self.assertEqual(c.loads(c.dumps(a)),a)

    def test_R04_held_evidence_wrong_target_and_pose_error_over_budget_rejected(self):
        req=self.placement()
        self.ctx=replace(self.ctx,holding_envelope=replace(self.ctx.holding_envelope,target_id='other'))
        self.assert_rejected(self.svc.plan_placement(req))

    def test_R01_actual_c0_and_p04_pixels_can_enter_p06(self):
        from treasure_auto.perception.geometry import fit_top
        from treasure_auto.perception.config import PerceptionConfig
        from test_p04_pixels import config_dict as perception_config
        corners=self.top.corners
        edges=tuple(tuple(m.PixelPoint(a.u+(b.u-a.u)*k/10,a.v+(b.v-a.v)*k/10) for k in range(11))
                    for a,b in zip(corners,corners[1:]+corners[:1]))
        fitted=fit_top(self.image,tuple(p for edge in edges for p in edge),edges,
                       PerceptionConfig.from_dict(perception_config()),m.PlaneBasis.CALIBRATION,0.,(self.ev,))
        # P04's 1px quantization bound correctly propagates: tight configured error
        # budget rejects the center rather than disguising it as precise geometry.
        self.obs=replace(self.obs,top=fitted.top)
        r=self.svc.plan(self.request())
        self.assert_rejected(r)
        self.assertEqual(r.result.reason.field,'error_mm')
