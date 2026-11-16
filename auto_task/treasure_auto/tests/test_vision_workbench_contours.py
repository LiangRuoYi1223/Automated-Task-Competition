"""Synthetic pixel tests; these do not measure competition-photo accuracy."""
import importlib
import json
import math
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))


class ContourWorkbenchTests(unittest.TestCase):
    def setUp(self):
        try:
            self.detect = importlib.import_module(
                'treasure_auto.vision_workbench.contours').detect_candidates
        except ModuleNotFoundError:
            self.fail('candidate contour pixel implementation is missing')
        self.config = dict(min_area_px2=500., max_area_fraction=.9,
                           max_line_error_px=1.8, min_edge_points=8,
                           min_edge_span_px=15., min_corner_sine=.15,
                           approx_epsilon_px=2., roi=None, board_regions=[],
                           canny_low=40., canny_high=100.)

    def image(self, points=None):
        bgr = np.zeros((240, 360, 3), np.uint8)
        points = points if points is not None else [(40, 35), (190, 55), (175, 180), (25, 155)]
        cv2.fillPoly(bgr, [np.array(points, np.int32)], (180, 210, 230))
        return bgr

    def test_complete_perspective_has_one_four_edge_candidate(self):
        image = self.image()
        original = image.copy()
        result = self.detect(image, self.config)
        self.assertEqual(len(result['candidates']), 1)
        candidate = result['candidates'][0]
        self.assertEqual(len(candidate['edges']), 4)
        self.assertTrue(all(len(e) >= 8 for e in candidate['edges']))
        for actual, expected in zip(candidate['corners'], [(40, 35), (190, 55), (175, 180), (25, 155)]):
            self.assertLess(math.hypot(actual[0] - expected[0], actual[1] - expected[1]), 2.)
        self.assertFalse(candidate['body_confirmed'])
        self.assertFalse(candidate['actionable'])
        self.assertEqual(candidate['plane_basis'], 'unknown')
        self.assertEqual(result['region_basis'], 'unconfigured')
        self.assertTrue(np.array_equal(image, original))
        json.dumps(result, allow_nan=False)

    def test_two_separate_objects_are_two_candidates(self):
        image = np.zeros((240, 360, 3), np.uint8)
        cv2.rectangle(image, (25, 30), (140, 160), (220, 220, 220), -1)
        cv2.rectangle(image, (200, 55), (325, 185), (160, 190, 220), -1)
        result = self.detect(image, self.config)
        self.assertEqual(len(result['candidates']), 2)

    def test_manual_board_region_excludes_overlapping_candidate(self):
        image = np.zeros((240, 360, 3), np.uint8)
        cv2.rectangle(image, (25, 30), (140, 160), (220, 220, 220), -1)
        cv2.rectangle(image, (200, 55), (325, 185), (160, 190, 220), -1)
        result = self.detect(image, dict(self.config, board_regions=[[190, 45, 145, 150]]))
        self.assertEqual(len(result['candidates']), 1)
        self.assertLess(result['candidates'][0]['bbox'][0], 100)
        self.assertEqual(result['region_basis'], 'manual')
        self.assertTrue(any('board' in r['reason'] for r in result['rejected']))

    def test_roi_is_not_an_artificial_source_of_four_edges(self):
        result = self.detect(self.image(), dict(self.config, roi=[75, 60, 160, 140]))
        self.assertFalse(result['candidates'])
        self.assertEqual(result['region_basis'], 'manual')
        self.assertTrue(any('roi' in r['reason'] for r in result['rejected']))

    def test_fitted_intersections_also_stay_inside_manual_roi(self):
        # The rightmost observed pixel is x=190; the line intersection is
        # slightly farther right. The ROI must constrain the published fit too.
        result = self.detect(self.image(), dict(self.config, roi=[0, 0, 190.1, 240]))
        self.assertEqual(len(result['candidates']), 0)
        self.assertTrue(any('roi' in r['reason'] for r in result['rejected']))

    def test_concave_notch_is_not_completed_to_a_rectangle(self):
        image = self.image([(30, 30), (190, 30), (190, 180), (30, 180),
                            (30, 130), (100, 130), (100, 75), (30, 75)])
        result = self.detect(image, self.config)
        self.assertFalse(result['candidates'])
        self.assertTrue(result['rejected'])

    def test_frame_truncation_is_rejected(self):
        result = self.detect(self.image([(-20, 30), (190, 30), (190, 180), (-20, 180)]), self.config)
        self.assertFalse(result['candidates'])
        self.assertTrue(any('truncated' in r['reason'] for r in result['rejected']))

    def test_internal_print_or_hole_stays_unresolved_without_losing_outer_edges(self):
        image = self.image()
        cv2.rectangle(image, (75, 80), (130, 120), (0, 0, 0), -1)
        result = self.detect(image, self.config)
        self.assertEqual(len(result['candidates']), 1)
        candidate = result['candidates'][0]
        self.assertGreater(candidate['internal_boundaries'], 0)
        self.assertTrue(candidate['occlusion_unresolved'])
        self.assertFalse(candidate['actionable'])

    def test_open_three_edge_outline_does_not_invent_missing_edge(self):
        image = np.zeros((240, 360, 3), np.uint8)
        cv2.polylines(image, [np.array([(30, 160), (30, 40), (190, 40), (190, 160)], np.int32)],
                      False, (220, 220, 220), 2)
        result = self.detect(image, self.config)
        self.assertFalse(result['candidates'])

    def test_configured_real_edge_support_is_required(self):
        result = self.detect(self.image(), dict(self.config, min_edge_points=500))
        self.assertFalse(result['candidates'])
        self.assertTrue(any('edge' in r['reason'] for r in result['rejected']))

    def test_blank_frame_reports_no_candidates(self):
        result = self.detect(np.zeros((240, 360, 3), np.uint8), self.config)
        self.assertEqual(result['candidates'], [])

    def test_bad_pixels_and_invalid_thresholds_are_explained(self):
        for pixels in [None, np.zeros((20, 20), np.uint8), np.zeros((20, 20, 3), np.float32),
                       np.zeros((0, 20, 3), np.uint8)]:
            with self.subTest(pixels=type(pixels).__name__), self.assertRaises(ValueError):
                self.detect(pixels, self.config)
        for update in [dict(min_area_px2=-1), dict(canny_low=float('nan')),
                       dict(canny_high=10), dict(min_corner_sine=2),
                       dict(roi=[-1, 0, 20, 20]), dict(board_regions=[[10, 10, 0, 20]]),
                       dict(min_edge_points=True)]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.detect(self.image(), dict(self.config, **update))


if __name__ == '__main__':
    unittest.main()
