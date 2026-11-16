import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from test_p06_fixtures import module, transform, camera_transform, camera, pose, config_dict, evidence, c, m


class CoordinateTests(unittest.TestCase):
    def setUp(self):
        self.g = module(self, 'math3d')

    def test_T01_map_m_to_arm_mm_once_and_rotated_parking(self):
        t = transform(xyz=(10., 20., 30.))
        for actual, want in zip(self.g.map_to_arm(pose(1.2, 2.3), 100., pose(1., 2.), t), (210., 320., 130.)):
            self.assertAlmostEqual(actual, want)
        p = self.g.map_to_arm(pose(1.2, 2.3), 100., pose(1., 2., math.pi / 2), t)
        for actual, want in zip(p, (310., -180., 130.)):
            self.assertAlmostEqual(actual, want)
        back = self.g.arm_to_map(p, pose(1., 2., math.pi / 2), t)
        for actual, want in zip(back, (1.2, 2.3, .1)):
            self.assertAlmostEqual(actual, want)

    def test_T01_singular_scaled_reflected_and_wrong_direction_transforms_rejected(self):
        for matrix in ((0.,)*16, (2.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.),
                       (-1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.)):
            with self.subTest(matrix=matrix), self.assertRaises(c.ContractError):
                self.g.validate_transform(replace(transform(), matrix=matrix), 'base', 'arm')
        with self.assertRaises(c.ContractError):
            self.g.validate_transform(transform('arm', 'base'), 'base', 'arm')

    def test_T02_same_pixel_different_planes_and_parallel_ray(self):
        a, ae = self.g.ray_on_plane(m.PixelPoint(420., 240.), camera(), camera_transform(), 100., 1., 1.)
        b, be = self.g.ray_on_plane(m.PixelPoint(420., 240.), camera(), camera_transform(), 200., 1., 1.)
        self.assertEqual(a, (180., 0., 100.))
        self.assertEqual(b, (160., 0., 200.))
        self.assertGreater(ae, 1.8)
        self.assertGreater(be, 1.6)
        with self.assertRaises(c.ContractError):
            horizontal=replace(camera_transform(),matrix=(0.,0.,1.,0.,0.,1.,0.,0.,-1.,0.,0.,0.,0.,0.,0.,1.))
            self.g.ray_on_plane(m.PixelPoint(320., 240.), camera(), horizontal, 100., 0., 0.)

    def test_T01_raw_pixels_require_rectification_and_corrected_does_not_repeat_it(self):
        p=m.PixelPoint(420., 240.)
        self.assertEqual(self.g.correct_pixel(p, camera()), p)
        raw=replace(camera(), distortion=m.Distortion.RAW, distortion_coefficients=(.1,0.,0.,0.,0.))
        corrected=self.g.correct_pixel(m.PixelPoint(420.4,240.), raw)
        self.assertAlmostEqual(corrected.u,420.,places=6)
        with self.assertRaises(c.ContractError):
            self.g.correct_pixel(p,replace(camera(),distortion=m.Distortion.UNKNOWN))


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.g=module(self,'calibration')

    def sample(self, name, dx=0., kind=c.EvidenceKind.PHYSICAL):
        return self.g.CalibrationSample(name, c.ArmPoint('arm',0.,0.,0.,'mm',0.),
            c.ArmPoint('arm',dx,0.,0.,'mm',0.), evidence(evidence_id=name,kind=kind))

    def report(self, fits=None, checks=None, limit=2.):
        return self.g.make_report('sim-calibration-v1','image','camera-v1','tool-v1',
            ((-300.,-300.,0.),(300.,300.,1000.)),
            fits or (self.sample('fit'),), checks or (self.sample('v1',1.),self.sample('v2',-2.)),limit)

    def test_R06_pointwise_max_rms_bias_and_no_mean_hiding_boundary_error(self):
        r=self.report()
        self.assertEqual(r.validation_errors_mm,(1.,2.))
        self.assertEqual(r.max_error_mm,2.)
        self.assertAlmostEqual(r.rms_error_mm,math.sqrt(2.5))
        self.assertEqual(r.bias_mm,(-.5,0.,0.))
        self.assertTrue(r.accepted)
        self.assertFalse(self.report(limit=1.9).accepted)

    def test_R06_fit_and_validation_identity_or_evidence_overlap_rejected(self):
        for sample in (self.sample('fit'),replace(self.sample('v'),evidence=evidence(evidence_id='fit'))):
            with self.subTest(sample=sample), self.assertRaises(c.ContractError):
                self.report(checks=(sample,))

    def test_R06_echo_truth_unknown_domain_and_nonfinite_limit_rejected(self):
        with self.assertRaises(c.ContractError):
            self.report(checks=(self.sample('v',kind=c.EvidenceKind.ECHO),))
        with self.assertRaises(c.ContractError):
            self.report(limit=float('nan'))

    def test_R06_report_saved_with_points_domain_and_version_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'report.json'
            self.g.save_report(self.report(),p)
            data=json.loads(p.read_text(encoding='utf-8'))
            self.assertEqual(data['validation_errors_mm'],[1.,2.])
            self.assertEqual(data['version'],'sim-calibration-v1')
            self.assertEqual(len(data['validation_samples']),2)
            with self.assertRaises(FileExistsError):
                self.g.save_report(self.report(),p)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.g=module(self,'config')

    def test_R01_missing_unknown_nonfinite_or_unmeasured_configuration_cannot_plan(self):
        for key,value in (('approach_mm',None),('speeds_mm_s',[0.,1.,1.,1.,1.]),
                          ('brick_size_mm',[100.,100.,float('nan')]),('image_size',[640,False])):
            fields=config_dict();fields[key]=value
            with self.subTest(key=key),self.assertRaises(c.ContractError):
                self.g.GeometryConfig.from_dict(fields).require_ready(c.Origin.SIMULATED)
        fields=config_dict();fields['unexpected']=1
        with self.assertRaises(c.ContractError):self.g.GeometryConfig.from_dict(fields)

    def test_R01_simulation_cannot_enable_live_geometry(self):
        cfg=self.g.GeometryConfig.from_dict(config_dict())
        cfg.require_ready(c.Origin.SIMULATED)
        with self.assertRaises(c.ContractError):cfg.require_ready(c.Origin.LIVE)

    def test_R01_shipped_config_remains_unconfigured(self):
        path=Path(__file__).resolve().parents[1]/'config/geometry/unmeasured.json'
        self.assertTrue(path.exists(),'explicit unmeasured configuration missing')
        cfg=self.g.GeometryConfig.from_dict(json.loads(path.read_text(encoding='utf-8')))
        with self.assertRaises(c.ContractError):cfg.require_ready(c.Origin.LIVE)
