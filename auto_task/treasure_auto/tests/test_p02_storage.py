"""P02-T01: map conversion, immutable versions and real filesystem failures."""
import json
import hashlib
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from test_p02_support import c, m, bundle, pose, poly

try:
    from treasure_auto.mapping import storage as s
except ImportError:
    s = None


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(s, 'P02 storage implementation missing')

    def test_T01_rotated_origin_and_top_down_rows(self):
        b = bundle()
        b = replace(b, metadata=replace(b.metadata, origin=pose(10, -2, math.pi / 2)))
        p = b.cell_center(0, 0)
        self.assertAlmostEqual(p.x, 4.25)
        self.assertAlmostEqual(p.y, -1.75)
        self.assertEqual(b.world_cell(p.x, p.y), (0, 0))
        p = b.cell_center(11, 11)
        self.assertAlmostEqual(p.x, 9.75)
        self.assertAlmostEqual(p.y, 3.75)
        with self.assertRaises(c.ContractError):
            b.world_cell(100, 100)

    def test_T01_save_reload_retains_old_version(self):
        with tempfile.TemporaryDirectory() as d:
            first = s.save_bundle(bundle(), d, 'map-v1')
            original = (Path(d) / 'map-v1/map.json').read_bytes()
            second = s.save_bundle(first, d, 'map-v2')
            self.assertEqual(second.metadata.map_version, 'map-v2')
            self.assertEqual(second.waypoints[0].envelope.versions.map, 'map-v2')
            self.assertEqual(second.raster.pixels, (255,) * 144)
            self.assertEqual((Path(d) / 'map-v1/map.json').read_bytes(), original)
            with self.assertRaises(c.ContractError):
                s.save_bundle(first, d, 'map-v1')

    def test_T01_version_image_and_dimension_tampering_rejected(self):
        for change in ('version', 'image', 'dimension', 'duplicate'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                s.save_bundle(bundle(), d, 'map-v1')
                p = Path(d) / 'map-v1/map.json'
                data = json.loads(p.read_text(encoding='utf-8'))
                if change == 'version':
                    points=p.parent/'waypoints.json'
                    payload=json.loads(points.read_text(encoding='utf-8'))
                    payload['waypoints'][0]['envelope']['versions']['map']='other'
                    points.write_text(json.dumps(payload),encoding='utf-8')
                    data['waypoints_sha256']=hashlib.sha256(points.read_bytes()).hexdigest()
                elif change == 'dimension':
                    data['width'] = 99
                elif change == 'image':
                    (p.parent / 'grid.pgm').write_bytes(b'P5\n1 1\n255\n\x00')
                else:
                    p.write_text('{"format":1,"format":1}', encoding='utf-8')
                if change != 'duplicate':
                    p.write_text(json.dumps(data), encoding='utf-8')
                with self.assertRaises(c.ContractError):
                    s.load_bundle(p, 'map-v1')

    def test_T01_missing_map_and_wrong_expected_version(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(c.ContractError) as err:
                s.load_bundle(Path(d) / 'absent.json', 'map-v1')
            self.assertEqual(err.exception.code, c.ErrorCode.UNAVAILABLE)
            s.save_bundle(bundle(), d, 'map-v1')
            with self.assertRaises(c.ContractError) as err:
                s.load_bundle(Path(d) / 'map-v1/map.json', 'other')
            self.assertEqual(err.exception.code, c.ErrorCode.VERSION_MISMATCH)

    def test_T01_pgm_ascii_binary_and_malformed(self):
        for data in (b'P2\n# unmeasured test\n2 1\n255\n0 255\n', b'P5\n2 1\n255\n\x00\xff'):
            self.assertEqual(s.read_pgm(data).pixels, (0, 255))
        for data in (b'P5\n2 1\n255\n\xff', b'P2\n1 1\n255\n256', b'P2\n0 1\n255\n', b'garbage'):
            with self.subTest(data=data), self.assertRaises(c.ContractError):
                s.read_pgm(data)

    def test_T01_reject_unversioned_and_path_escape(self):
        b = bundle()
        with self.assertRaises(c.ContractError):
            replace(b, waypoints=(replace(b.waypoints[0], envelope=replace(b.waypoints[0].envelope,
                     versions=replace(b.waypoints[0].envelope.versions, map='other'))),))
        with tempfile.TemporaryDirectory() as d, self.assertRaises(c.ContractError):
            s.save_bundle(b, d, '../outside')

    def test_T01_reject_invalid_polygon_and_region_center(self):
        b = bundle()
        with self.assertRaises(c.ContractError):
            replace(b, boundary=(m.Point2(0, 0), m.Point2(1, 1), m.Point2(0, 1), m.Point2(1, 0)))
        bad = replace(b.metadata.regions[0], placement_center=pose(5, 5))
        with self.assertRaises(c.ContractError):
            replace(b, metadata=replace(b.metadata, regions=(bad,) + b.metadata.regions[1:]))
