from publication_support import simulated_space_readback
import math
import unittest
from dataclasses import replace
from test_p06_fixtures import *
from contract_fixtures.samples import sensor_stamp


class PathFixture(unittest.TestCase):
    def setUp(self):
        self.g=module(self,'service');self.i=module(self,'inputs');self.cal=module(self,'calibration')
        self.cfg=module(self,'config').GeometryConfig.from_dict(config_dict())
        self.ev=evidence();self.env=envelope(c.Owner.P07)
        points=tuple(m.PixelPoint(u,v) for u,v in ((350,210),(400,210),(400,260),(350,260)))
        self.top=m.TopSurface(points,points,640,480,'camera',m.Distortion.CORRECTED,.1,m.PlaneBasis.CALIBRATION,0.,m.Visibility.VISIBLE,(self.ev,))
        self.obs=m.TargetObservation('obs','brick-1',c.Fact.CONFIRMED,m.Category.DOLL,self.top,m.Visibility.VISIBLE,(self.ev,))
        self.tool=transform('cup','arm_reference',(4.,-3.,-50.),'tool-v1')
        self.image=m.ImageObservation(self.env,'obs',self.ev,640,480,'mono8','camera',m.Distortion.CORRECTED,pose(),m.PixelTransform(0,0,1,1,0,0))
        self.hist=self.i.PixelHistory('obs','uv','full',self.image.preprocessing,0,m.Distortion.CORRECTED,0,None)
        self.plane=self.i.PlaneSolution('surface-1',0.,.1,(0.,0.,1.),m.PlaneBasis.CALIBRATION,(),(self.ev,))
        self.ctx=self.i.GeometryInputs(consumer_context(),m.LocalizationSnapshot(envelope(c.Owner.P02),pose(),c.Fact.CONFIRMED,0,(self.ev,),None),
            c.Fact.CONFIRMED,(self.ev,),self.image,self.hist,sensor_stamp(20.1),sensor_stamp(19.),(self.plane,),camera_transform(),transform(),self.tool,
            c.HoldingSnapshot(c.Holding.EMPTY,None,None,(evidence(target_id=None),)),envelope(c.Owner.P07,target_id=None),None,
            c.ArmPoint('arm',0.,0.,200.,'mm',.5),(self.ev,),'sim-arm-geometry-v1',space(),c.Fact.CONFIRMED,(self.ev,),
            self.report('image'),self.report('placement'),None)
        self.audit=[];self.calls=[];self.cert_override=None
        self.svc=self.g.GeometryService(self.cfg,lambda:self.ctx,self.verify,self.audit.append,space_readback=simulated_space_readback)

    def report(self,chain):
        def sample(name):
            return self.cal.CalibrationSample(chain+name,c.ArmPoint('arm',0,0,0,'mm',0),c.ArmPoint('arm',0,0,0,'mm',.5),evidence(evidence_id=chain+name))
        cal=replace(camera(),independent_validation=(evidence(evidence_id='imagecheck'),))
        binding=self.cal.CalibrationBinding(cal,camera_transform(),transform(),self.tool,self.cfg)
        return self.cal.make_report('sim-calibration-v1',chain,'sim-extrinsic-v1','tool-v1',
             ((-500.,-500.,-200.),(500.,500.,500.)),(sample('fit'),),(sample('check'),),2.,binding)

    def verify(self,query):
        self.calls.append(query)
        pts=(query.start,query.segment.target)
        box=self.i.SweepBox(tuple(min(getattr(p,a) for p in pts)-.5 for a in ('x','y','z')),
                           tuple(max(getattr(p,a) for p in pts)+.5 for a in ('x','y','z')))
        cert=self.i.MotionCertificate(query,c.Fact.CONFIRMED,c.Fact.CONFIRMED,c.Fact.CONFIRMED,(box,),(self.ev,),None)
        return self.cert_override(cert) if self.cert_override else cert

    def request(self):
        cal=replace(camera(),independent_validation=tuple(s.evidence for s in self.ctx.image_report.validation_samples)) if self.ctx.image_report is not None else camera()
        return m.GeometryRequest(self.env,self.obs,pose(),self.ctx.localization.pose,cal,self.tool,None,self.ctx.space)

    def placement(self,base=None,offset=(10.,2.,0.)):
        base=base or pose()
        off=c.ArmPoint('cup',*offset,'mm',.5)
        self.ctx=replace(self.ctx,localization=replace(self.ctx.localization,pose=base),
            holding=c.HoldingSnapshot(c.Holding.HELD,'brick-1',off,(self.ev,)),holding_envelope=self.env,held_yaw_rad=0.)
        region=m.RegionBoundary(m.Area.B,'map',square(-1,-1,2),pose(.2,.2),0.)
        return m.PlacementGeometryRequest(self.env,pose(.2,.2),region,base,(),self.tool,off,self.ctx.space)

    def assert_rejected(self,result,code=None):
        self.assertIsNone(result.plan)
        self.assertNotEqual(result.result.status,c.Outcome.SUCCESS)
        self.assertIsNotNone(result.result.reason)
        if code:self.assertEqual(result.result.reason.code,code)


class PathTests(PathFixture):
    def test_R04_five_segments_have_distinct_heights_speeds_timeouts_and_no_physical_success(self):
        r=self.svc.plan(self.request())
        self.assertEqual(r.result.status,c.Outcome.SUCCESS,r.result.reason)
        self.assertEqual([s.phase for s in r.plan.segments],[m.Phase.APPROACH,m.Phase.DESCEND,m.Phase.CONTACT,m.Phase.TRIAL_LIFT,m.Phase.WITHDRAW])
        self.assertEqual([s.target.z for s in r.plan.segments],[130.,70.,50.,150.,150.])
        self.assertEqual([s.speed_mm_s for s in r.plan.segments],[10.,5.,1.,5.,10.])
        self.assertEqual(r.result.physical,c.Fact.UNKNOWN)
        self.assertEqual([q.carrying for q in self.calls],[False,False,False,True,True])
        self.assertEqual(c.loads(c.dumps(r)),r)

    def test_T05_endpoint_safe_but_approach_crosses_obstacle(self):
        obstacle=occupancy('fence',x=.04,z0=.11,z1=.18)
        obstacle=replace(obstacle,footprint_m=square(.04,-.01,.02))
        self.ctx=replace(self.ctx,space=space((obstacle,)))
        self.assert_rejected(self.svc.plan(self.request()),c.ErrorCode.OCCUPANCY_UNRESOLVED)

    def test_T05_payload_envelope_blocks_withdrawal_even_when_tool_clears(self):
        obstacle=replace(occupancy('brick-other',z0=.03,z1=.09),footprint_m=square(.12,.04,.02))
        self.ctx=replace(self.ctx,space=space((obstacle,)))
        self.assert_rejected(self.svc.plan(self.request()),c.ErrorCode.OCCUPANCY_UNRESOLVED)

    def test_T05_unknown_occupancy_and_missing_full_arm_model_fail_closed(self):
        obs=replace(occupancy('unknown',z0=.11,z1=.18,state=m.OccupancyState.UNKNOWN),footprint_m=square(.04,-.01,.02))
        self.ctx=replace(self.ctx,space=space((obs,)))
        self.assert_rejected(self.svc.plan(self.request()))
        self.ctx=replace(self.ctx,space=space())
        self.svc=self.g.GeometryService(self.cfg,lambda:self.ctx,None,self.audit.append,space_readback=simulated_space_readback)
        self.assert_rejected(self.svc.plan(self.request()),c.ErrorCode.UNAVAILABLE)

    def test_R05_unknown_reachability_static_clearance_or_incomplete_certificate_rejected(self):
        for changes in (dict(reachable=c.Fact.UNKNOWN),dict(full_arm_covered=c.Fact.UNKNOWN),
                        dict(static_clearance=c.Fact.UNKNOWN),dict(swept_boxes=()),dict(evidence=())):
            self.cert_override=lambda cert:replace(cert,**changes)
            self.svc=self.g.GeometryService(self.cfg,lambda:self.ctx,self.verify,self.audit.append,space_readback=simulated_space_readback)
            with self.subTest(changes=changes):self.assert_rejected(self.svc.plan(self.request()))

    def test_R05_certificate_must_bind_actual_segment_payload_and_versions(self):
        self.cert_override=lambda cert:replace(cert,query=replace(cert.query,geometry_version='old-robot'))
        self.assert_rejected(self.svc.plan(self.request()),c.ErrorCode.ID_CONFLICT)

    def test_T06_same_map_target_different_parking_and_held_offset(self):
        a=self.svc.plan_placement(self.placement())
        self.assertEqual(a.result.status,c.Outcome.SUCCESS,a.result.reason)
        self.assertAlmostEqual(a.plan.target.x,186.)
        self.assertAlmostEqual(a.plan.target.y,201.)
        self.assertAlmostEqual(a.plan.target.z,150.)
        # Use a fresh synthetic context for the second independent parking scenario.
        self.setUp()
        b=self.svc.plan_placement(self.placement(base=pose(.1,0.),offset=(20.,2.,0.)))
        self.assertEqual(b.result.status,c.Outcome.SUCCESS,b.result.reason)
        self.assertAlmostEqual(b.plan.target.x,76.)
        self.assertAlmostEqual(b.plan.target.y,201.)

    def test_R04_unknown_holding_offset_or_unsupported_yaw_rejected(self):
        req=self.placement()
        for changes in (dict(holding=c.HoldingSnapshot(c.Holding.UNKNOWN,None,None,())),dict(held_yaw_rad=None),dict(held_yaw_rad=.2)):
            old=self.ctx;self.ctx=replace(self.ctx,**changes)
            with self.subTest(changes=changes):self.assert_rejected(self.svc.plan_placement(req))
            self.ctx=old

    def test_R02_placement_requires_current_real_support_and_layer(self):
        req=replace(self.placement(),actual_support_ids=('absent',))
        self.assert_rejected(self.svc.plan_placement(req))

    def test_R05_missing_coverage_or_not_stopped_cannot_generate_plan(self):
        for changes in (dict(coverage=c.Fact.UNKNOWN),dict(stopped=c.Fact.UNKNOWN),dict(coverage_evidence=())):
            old=self.ctx;self.ctx=replace(self.ctx,**changes)
            with self.subTest(changes=changes):self.assert_rejected(self.svc.plan(self.request()))
            self.ctx=old
