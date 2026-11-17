"""Synthetic observed-pixel regressions; not field accuracy measurements."""
import importlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class LineWorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('treasure_auto.vision_workbench.lines')
        self.config = dict(min_area_px2=500., max_area_fraction=.95,
                           max_line_error_px=2., min_edge_points=8,
                           min_edge_span_px=12., min_corner_sine=.15,
                           approx_epsilon_px=2., roi=None, board_regions=[],
                           canny_low=40., canny_high=120.)

    def detect(self, image, **settings):
        return self.module.detect_candidates(image, self.config,
                    dict(frame_budget_ms=2000., **settings))

    def image(self, quad=None):
        bgr = np.zeros((240, 360, 3), np.uint8)
        quad = quad or [[40,35],[190,55],[175,180],[25,155]]
        cv2.fillConvexPoly(bgr, np.array(quad, np.int32), (180,210,230))
        return bgr

    def test_perspective_real_pixels_and_order(self):
        image = self.image()
        before = image.copy()
        result = self.detect(image)
        self.assertEqual(len(result['candidates']), 1, result)
        candidate = result['candidates'][0]
        for actual, expected in zip(candidate['corners'], [[40,35],[190,55],[175,180],[25,155]]):
            self.assertLess(np.linalg.norm(np.array(actual)-expected), 3.)
        self.assertEqual(len(candidate['edges']), 4)
        for edge, metric in zip(candidate['edges'], candidate['edge_metrics']):
            self.assertGreaterEqual(len(edge), 8)
            self.assertGreater(metric['support_rate'], .8)
            self.assertTrue(metric['support_intervals'])
            self.assertTrue(metric['segment_ids'])
            for x, y in edge:
                self.assertEqual(int(x), x)
                self.assertEqual(int(y), y)
        self.assertFalse(candidate['actionable'])
        self.assertFalse(candidate['body_confirmed'])
        self.assertEqual(candidate['plane_basis'], 'unknown')
        self.assertTrue(np.array_equal(before, image))
        json.dumps(result, allow_nan=False)

    def test_disjoint_objects(self):
        image = np.zeros((240,360,3), np.uint8)
        cv2.rectangle(image,(25,30),(140,160),(220,220,220),-1)
        cv2.rectangle(image,(200,55),(325,185),(150,190,220),-1)
        self.assertEqual(len(self.detect(image)['candidates']), 2)

    def test_three_edges_never_complete(self):
        image = np.zeros((240,360,3),np.uint8)
        cv2.polylines(image,[np.array([[30,160],[30,40],[190,40],[190,160]],np.int32)],
                      False,(220,220,220),2)
        self.assertFalse(self.detect(image)['candidates'])

    def test_long_missing_interval_not_filled(self):
        image = self.image([[40,40],[200,40],[200,180],[40,180]])
        cv2.rectangle(image,(35,85),(48,135),(0,0,0),-1)
        result = self.detect(image)
        self.assertFalse(result['candidates'], result)

    def test_bounded_support_growth_requires_actual_continuing_pixels(self):
        image=self.image([[40,40],[210,40],[210,180],[40,180]])
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3)
        gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
        cfg=self.module.validate_config({'max_support_extension_px':64.})
        points=self.module._observed_support([70,39,150,39],gx,gy,cfg,np)
        self.assertLess(points[:,0].min(),45)
        self.assertGreater(points[:,0].max(),200)
        for x,y in points.astype(int):
            self.assertGreaterEqual(np.hypot(gx[y,x],gy[y,x]),cfg['gradient_threshold'])
        cv2.rectangle(image,(100,35),(145,50),(0,0,0),-1)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        points=self.module._observed_support([55,39,85,39],
            cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3),
            cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3),cfg,np)
        self.assertLess(points[:,0].max(),110,'growth must stop before a long invisible run')

    def test_growth_does_not_create_an_absent_fourth_side(self):
        image=np.zeros((240,360,3),np.uint8)
        cv2.polylines(image,[np.array([[30,160],[30,40],[190,40],[190,160]],np.int32)],
                      False,(220,220,220),2)
        result=self.detect(image,max_support_extension_px=64.)
        self.assertFalse(result['candidates'])
        self.assertFalse(result['unresolved'])

    def test_short_fragment_gap_preserves_gap_measurement(self):
        image = self.image([[40,40],[200,40],[200,180],[40,180]])
        cv2.rectangle(image,(35,100),(45,104),(0,0,0),-1)
        result = self.detect(image)
        geometric=result['candidates']+result['unresolved']
        self.assertTrue(geometric, result)
        self.assertGreater(max(m['max_gap_px'] for m in geometric[0]['edge_metrics']), 1.)

    def test_roi_cannot_supply_a_fourth_edge(self):
        self.config['roi'] = [70,60,160,140]
        self.assertFalse(self.detect(self.image())['candidates'])

    def test_clipped_frame_never_completes(self):
        self.assertFalse(self.detect(self.image([[-20,30],[190,30],[190,180],[-20,180]]))['candidates'])

    def test_internal_rectangle_is_not_qualified_as_top(self):
        image = self.image([[30,30],[220,30],[220,200],[30,200]])
        cv2.rectangle(image,(75,75),(160,145),(0,0,0),-1)
        result = self.detect(image)
        self.assertTrue(result['unresolved'], result)
        self.assertFalse(any(c['bbox'][0]>60 for c in result['candidates']))

    def test_adjacent_bricks_do_not_qualify_combined_frame(self):
        image=np.zeros((240,360,3),np.uint8)
        cv2.rectangle(image,(35,35),(285,185),(200,200,200),-1)
        cv2.line(image,(155,35),(155,185),(60,60,60),2)
        result=self.detect(image)
        self.assertFalse(any(c['bbox'][2]>230 for c in result['candidates']),result)
        self.assertTrue(result['unresolved'],result)

    def test_nonquadrilateral_internal_boundary_stays_unresolved(self):
        for shape in ('circle','triangle'):
            image=self.image([[30,30],[220,30],[220,200],[30,200]])
            if shape=='circle':
                cv2.circle(image,(110,110),25,(0,0,0),-1)
            else:
                cv2.fillConvexPoly(image,np.array([[80,80],[160,100],[100,150]],np.int32),(0,0,0))
            result=self.detect(image)
            with self.subTest(shape=shape):
                self.assertFalse(result['candidates'])
                self.assertTrue(any('internal_boundary_print_or_occlusion_unresolved' in c['ambiguity_reasons']
                                    for c in result['unresolved']))

    def test_outline_inner_rim_dedup_does_not_remove_ambiguity(self):
        image=np.zeros((240,360,3),np.uint8)
        cv2.rectangle(image,(40,40),(210,180),(220,220,220),5)
        result=self.detect(image)
        self.assertFalse(result['candidates'])
        self.assertTrue(result['unresolved'])

    def test_thick_frame_inner_candidate_keeps_rejected_outer_boundary_evidence(self):
        image=np.zeros((240,360,3),np.uint8)
        cv2.rectangle(image,(30,30),(220,200),(220,220,220),10)
        result=self.detect(image)
        self.assertFalse(result['candidates'])
        self.assertTrue(any('enclosing_observed_boundary_may_be_internal_frame' in c['ambiguity_reasons']
                            for c in result['unresolved']))

    def test_color_channel_can_detect_equal_gray_boundary(self):
        image=np.full((240,360,3),(0,0,100),np.uint8)
        cv2.rectangle(image,(40,40),(210,180),(255,0,3),-1)
        self.assertFalse(self.detect(image)['candidates'])
        result=self.detect(image,channels=['b'])
        self.assertTrue(result['candidates'],result)
        self.assertEqual(result['candidates'][0]['edge_metrics'][0]['channels'],['b'])

    def test_manual_board_exclusion_and_diagnostics(self):
        self.config['board_regions']=[[20,20,200,190]]
        result=self.detect(self.image())
        self.assertFalse(result['candidates'])
        self.assertTrue(any('board' in r['reason'] for r in result['rejected']))
        self.assertEqual(result['no_candidate_reason'],'all_four_edge_hypotheses_failed_quality_gates')
        self.assertTrue(result['groups'][0]['observed_points'])
        self.assertTrue(result['groups'][0]['support_intervals'])

    def test_budget_state_and_missing_frontend_are_explicit(self):
        result = self.detect(self.image(), max_segments=1)
        self.assertTrue(result['budget_truncated'])
        self.assertTrue(result['truncation_reasons'])
        with patch.object(self.module, 'capabilities', return_value={'lsd':False,'edgedrawing':False}):
            with self.assertRaisesRegex(ValueError, 'unavailable'):
                self.detect(self.image())

    def test_blank_and_bad_inputs(self):
        result = self.detect(np.zeros((240,360,3),np.uint8))
        self.assertEqual(result['candidates'], [])
        self.assertFalse(result['budget_truncated'])
        self.assertEqual(result['no_candidate_reason'],'no_observed_four_edge_cycle')
        for bad in [None,np.zeros((2,2,3),np.uint8),np.zeros((20,20),np.uint8),
                    np.zeros((20,20,3),np.float32)]:
            with self.assertRaises(ValueError):
                self.detect(bad)
        for bad in [{'channels':[]},{'max_segments':True},{'frame_budget_ms':float('nan')},
                    {'min_support_rate':1.1},{'unknown':1},{'lsd_scale':0},
                    {'lsd_refine':'hough'}]:
            with self.assertRaises(ValueError):
                self.module.validate_config(bad)


if __name__ == '__main__':
    unittest.main()
