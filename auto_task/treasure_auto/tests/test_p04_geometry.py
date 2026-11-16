"""P04-T03 real numerical fitting; synthetic contours/masks, not recognition accuracy."""
import math
from dataclasses import replace
import unittest
from test_p04_pixels import module, image_frame, config_dict
from treasure_auto.contracts import core as c, models as m


def quad():
    return (m.PixelPoint(40, 30), m.PixelPoint(160, 50), m.PixelPoint(145, 140), m.PixelPoint(25, 120))


def edge_points(corners, trim=0.):
    groups = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        groups.append(tuple(m.PixelPoint(a.u + (b.u - a.u) * t, a.v + (b.v - a.v) * t)
                            for t in [trim + (1 - 2 * trim) * i / 20 for i in range(21)]))
    return tuple(groups)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = module(self, 'config').PerceptionConfig.from_dict(config_dict())
        self.g = module(self, 'geometry')
        self.im, _ = image_frame()

    def fit(self, edges=None, contour=None, **changes):
        args = dict(image=self.im, contour=contour or tuple(p for e in edge_points(quad()) for p in e),
                    edges=edge_points(quad()) if edges is None else edges, config=self.cfg,
                    plane_basis=m.PlaneBasis.CALIBRATION, height_mm=50., evidence=(self.im.image,))
        args.update(changes)
        return self.g.fit_top(**args)

    def test_T03_four_observed_lines_recover_perspective_corners(self):
        result = self.fit()
        for actual, expected in zip(result.top.corners, quad()):
            self.assertAlmostEqual(actual.u, expected.u, places=6)
            self.assertAlmostEqual(actual.v, expected.v, places=6)
        self.assertEqual(result.top.distortion, m.Distortion.RAW)
        self.assertEqual(len(result.corners), 4)
        self.assertTrue(all(x.visibility is m.Visibility.VISIBLE for x in result.corners))
        self.assertGreater(result.top.error_px, 0)
        self.assertEqual(len(result.edges), 4)

    def test_T03_occluded_corner_records_bounded_extrapolation(self):
        edges = edge_points(quad(), trim=.02)
        result = self.fit(edges, tuple(p for e in edges for p in e))
        self.assertEqual(result.top.visibility, m.Visibility.PARTIAL)
        self.assertTrue(all(x.visibility is m.Visibility.PARTIAL and x.extrapolation_px > 0
                            and x.basis == 'intersection_of_observed_edges' for x in result.corners))
        self.assertGreater(result.top.error_px, self.fit().top.error_px)

    def test_T03_occlusion_never_blindly_completes_missing_edges(self):
        for edges in [edge_points(quad())[:3], edge_points(quad(), trim=.25),
                      tuple((e[0], e[-1]) for e in edge_points(quad()))]:
            with self.subTest(edges=len(edges)), self.assertRaises(c.ContractError):
                self.fit(edges)

    def test_T03_nonconvex_parallel_self_crossing_and_noisy_edges_rejected(self):
        bow = (m.PixelPoint(30, 30), m.PixelPoint(140, 140), m.PixelPoint(140, 30), m.PixelPoint(30, 140))
        concave = (m.PixelPoint(30, 30), m.PixelPoint(140, 30), m.PixelPoint(60, 60), m.PixelPoint(30, 140))
        noisy = list(edge_points(quad()))
        noisy[0] = noisy[0][:10] + (m.PixelPoint(80, 80),) + noisy[0][11:]
        for edges in [edge_points(bow), edge_points(concave), (edge_points(quad())[0],) * 4, tuple(noisy)]:
            with self.subTest(edges=edges), self.assertRaises(c.ContractError):
                self.fit(edges)

    def test_T03_interior_shadow_edge_or_missing_plane_cannot_be_top(self):
        with self.assertRaises(c.ContractError):
            self.fit(contour=quad() + (m.PixelPoint(80, 80),))
        for changes in [dict(plane_basis=m.PlaneBasis.UNKNOWN), dict(height_mm=None), dict(evidence=()),
                        dict(height_mm=float('nan'))]:
            with self.subTest(changes=changes), self.assertRaises(c.ContractError):
                self.fit(**changes)

    def test_T03_real_binary_mask_extracts_edges_and_not_pattern_rectangle(self):
        import cv2
        import numpy as np
        mask = np.zeros((200, 320), np.uint8)
        cv2.fillConvexPoly(mask, np.array([(40, 30), (160, 50), (145, 140), (25, 120)]), 255)
        result = self.g.top_from_mask(mask, self.im, self.cfg, m.PlaneBasis.CALIBRATION, 50., (self.im.image,))
        for actual, expected in zip(result.top.corners, quad()):
            self.assertLess(math.hypot(actual.u - expected.u, actual.v - expected.v), 1.5)
        # Offset printed pattern occupies only x=55..75,y=55..70: no algorithm receives it as a top edge.
        self.assertGreater(result.top.corners[1].u, 150)

    def test_T03_masks_with_neighbors_holes_shadow_notches_and_clipped_edges_rejected(self):
        import numpy as np
        good = np.zeros((200, 320), np.uint8)
        good[30:100, 40:140] = 255
        hole = good.copy(); hole[50:60, 60:70] = 0
        neighbor = good.copy(); neighbor[110:140, 40:140] = 255
        notch = good.copy(); notch[30:60, 80:100] = 0
        clip = good.copy(); clip[0:60, 40:140] = 255
        for mask in [hole, neighbor, notch, clip]:
            with self.subTest(mask=mask.sum()), self.assertRaises(c.ContractError):
                self.g.top_from_mask(mask, self.im, self.cfg, m.PlaneBasis.CALIBRATION, 50., (self.im.image,))


if __name__ == '__main__':
    unittest.main()
