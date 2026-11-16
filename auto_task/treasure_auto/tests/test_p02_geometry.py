"""P02-T03: footprint and shared unknown occupancy, no route claims."""
import math
import unittest
from dataclasses import replace
from test_p02_support import c, m, bundle, pose, poly, occupancy, space

try:
    from treasure_auto.mapping import geometry as g
except ImportError:
    g = None


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(g, 'P02 geometry implementation missing')
        self.body = poly(-.2, -.15, .2, .15)

    def check(self, p=None, b=None, shared=None, **kw):
        return g.check_parking(b or bundle(), p or pose(), self.body,
                               space() if shared is None else shared, c.Holding.EMPTY, None, **kw)

    def test_T03_center_inside_but_body_outside(self):
        self.assertFalse(self.check(pose(.1, 2)).accepted)
        self.assertTrue(self.check(pose(2, 2)).accepted)

    def test_T03_actual_reserved_unknown_all_block(self):
        for state in m.OccupancyState:
            with self.subTest(state=state):
                result = self.check(shared=space((occupancy(state),)))
                self.assertFalse(result.accepted)
                self.assertEqual(result.reason.code, c.ErrorCode.OCCUPANCY_UNRESOLVED)

    def test_T03_carried_envelope_and_unknown_holding(self):
        carry = poly(-.2, -.2, .7, .2)
        b = bundle()
        for holding, carried in ((c.Holding.UNKNOWN, None), (c.Holding.DROPPED, None), (c.Holding.HELD, None)):
            self.assertFalse(g.check_parking(b, pose(), self.body, space(), holding, carried).accepted)
        self.assertFalse(g.check_parking(b, pose(5.5, 2), self.body, space(), c.Holding.HELD, carry).accepted)

    def test_T03_static_unknown_and_occupied_cells(self):
        for pixel in (0, 128):
            b = bundle()
            cells = list(b.raster.pixels)
            cells[7 * 12 + 4] = pixel  # x=2..2.5, y=2..2.5
            b = replace(b, raster=replace(b.raster, pixels=tuple(cells)))
            self.assertFalse(self.check(p=pose(2, 2), b=b).accepted)

    def test_T03_errors_and_rotation_expand_clearance(self):
        shared = space((occupancy(box=(2.3, 1.9, 2.4, 2.1)),))
        self.assertTrue(self.check(shared=shared).accepted)
        self.assertFalse(self.check(p=pose(error=.11), shared=shared).accepted)
        p = replace(pose(), error_rad=math.pi / 2)
        self.assertFalse(self.check(p=p, shared=shared).accepted)

    def test_T03_concave_field_edges_may_cross_outside(self):
        b = replace(bundle(), central=poly(4,4,5,5), boundary=tuple(m.Point2(*p) for p in
                    ((0,0),(6,0),(6,6),(3,6),(3,2),(2,2),(2,6),(0,6))))
        # All four corners inside, but horizontal edges cross the notch.
        body = poly(-1, -.1, 1, .1)
        self.assertFalse(g.check_parking(b, pose(2.5, 4), body, space(), c.Holding.EMPTY, None).accepted)

    def test_T03_wrong_space_map_or_frame_rejected(self):
        old = space()
        old = replace(old, envelope=replace(old.envelope, versions=replace(old.envelope.versions, map='old')))
        self.assertFalse(self.check(shared=old).accepted)
        self.assertFalse(self.check(p=pose(frame='odom')).accepted)
        wrong = replace(occupancy(), frame_id='odom')
        self.assertFalse(self.check(shared=space((wrong,))).accepted)

    def test_T03_map_origin_angular_error_expands_static_obstacles(self):
        b=bundle()
        cells=list(b.raster.pixels)
        cells[7*12+8]=0
        b=replace(b,raster=replace(b.raster,pixels=tuple(cells)),
                  metadata=replace(b.metadata,origin=replace(b.metadata.origin,error_rad=.1)))
        self.assertFalse(self.check(p=pose(3.6,2.6),b=b).accepted)
