import math
import unittest
from dataclasses import replace
from test_p06_fixtures import *


class SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.g=module(self,'surface');self.i=module(self,'inputs')
        self.cfg=module(self,'config').GeometryConfig.from_dict(config_dict())
        points=tuple(m.PixelPoint(u,v) for u,v in ((350,210),(400,210),(400,260),(350,260)))
        self.top=m.TopSurface(points,points,640,480,'camera',m.Distortion.CORRECTED,.1,
                             m.PlaneBasis.CALIBRATION,0.,m.Visibility.VISIBLE,(evidence(),))
        self.plane=self.i.PlaneSolution('surface-1',0.,.1,(0.,0.,1.),m.PlaneBasis.CALIBRATION,(),(evidence(),))
        self.history=self.i.PixelHistory('obs','uv','full',m.PixelTransform(0,0,1,1,0,0),0,
                                        m.Distortion.CORRECTED,0,None)

    def solve(self,top=None,planes=None,history=None):
        plane=self.g.resolve_plane((self.plane,) if planes is None else planes,self.cfg)
        return self.g.solve_top(top or self.top,camera(),camera_transform(),transform(),plane,
                               history or self.history,self.cfg,.5)

    def test_T02_current_plane_unknown_ambiguous_or_conflicting_rejected(self):
        for planes in ((),(replace(self.plane,basis=m.PlaneBasis.UNKNOWN),),
             (self.plane,replace(self.plane,solution_id='second')),
             (self.plane,replace(self.plane,height_mm=100.)),
             (replace(self.plane,normal=(.1,0.,.99)),)):
            with self.subTest(planes=planes),self.assertRaises(c.ContractError):self.solve(planes=planes)
        with self.assertRaises(c.ContractError):self.solve(replace(self.top,height_mm=None))

    def test_T03_true_diagonal_center_and_square_rotation_equivalence(self):
        s=self.solve()
        self.assertAlmostEqual(s.center.x,110.)
        self.assertAlmostEqual(s.center.y,10.)
        self.assertAlmostEqual(s.center.z,0.)
        self.assertAlmostEqual(s.yaw_rad,0.)
        rotated=replace(self.top,corners=self.top.corners[1:]+self.top.corners[:1])
        r=self.solve(rotated)
        for field in ('x','y','z','error_mm'):
            self.assertAlmostEqual(getattr(r.center,field),getattr(s.center,field))
        self.assertAlmostEqual(r.yaw_rad,s.yaw_rad)
        # Perspective quadrilateral diagonal center is not vertex arithmetic mean (1.75,1.5).
        center,error=self.g.diagonal_center(((0.,0.),(4.,0.),(3.,3.),(0.,3.)),0.)
        self.assertAlmostEqual(center[0],12./7)
        self.assertAlmostEqual(center[1],12./7)

    def test_T01_row_column_roi_and_double_correction_metadata_rejected(self):
        for h in (replace(self.history,coordinate_order='vu'),replace(self.history,pixel_space='roi'),
                  replace(self.history,roi_restorations=1),replace(self.history,distortion_passes=2),
                  replace(self.history,preprocessing=m.PixelTransform(10,0,1,1,0,0))):
            with self.subTest(h=h),self.assertRaises(c.ContractError):self.solve(history=h)

    def test_R03_crossed_or_tiny_top_and_excess_uncertainty_rejected(self):
        a,b,d,e=self.top.corners
        for top in (replace(self.top,corners=(a,d,b,e)),replace(self.top,error_px=4.),
                    replace(self.top,corners=tuple(m.PixelPoint(350+u,210+v) for u,v in ((0,0),(1,0),(1,1),(0,1))))):
            with self.subTest(top=top),self.assertRaises(c.ContractError):self.solve(top)

    def test_R04_tool_length_and_mount_offset_in_same_chain(self):
        target=self.g.tool_target(self.solve().center,transform('cup','arm_reference',(4.,-3.,-50.)),None,0.)
        self.assertAlmostEqual(target.x,106.)
        self.assertAlmostEqual(target.y,13.)
        self.assertAlmostEqual(target.z,50.)
        target2=self.g.tool_target(self.solve().center,transform('cup','arm_reference',(4.,-3.,-50.)),
                                  c.ArmPoint('cup',10.,2.,0.,'mm',1.),0.)
        self.assertAlmostEqual(target2.x,96.)
        self.assertAlmostEqual(target2.y,11.)

    def test_R01_raw_distortion_requires_measured_uncertainty_gain(self):
        with self.assertRaises(c.ContractError):
            self.g.solve_top(replace(self.top,distortion=m.Distortion.RAW),
                replace(camera(),distortion=m.Distortion.RAW,distortion_coefficients=(0.,)*5),
                camera_transform(),transform(),self.plane,
                replace(self.history,input_distortion=m.Distortion.RAW),self.cfg,.5)
