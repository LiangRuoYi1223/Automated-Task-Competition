"""Additional P04 failure/unknown boundaries; imports do not duplicate test collection."""
import ast
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import unittest
import test_p04_service as support
from test_p04_pixels import PACKAGE, image_frame, context_for, config_dict
from treasure_auto.contracts import core as c, models as m
from contract_fixtures.samples import sensor_stamp, stamp


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.h = support.ServiceTests()
        self.h.setUp()

    def test_R01_context_must_bind_camera_and_capture_aligned_pose(self):
        h = self.h
        for change in [dict(camera_frame='other-camera'), dict(capture_pose=c.MapPose('map', 5, 0, 0, 'm', 'rad', .01, .01))]:
            with self.subTest(change=change):
                original = h.im
                h.im = replace(h.im, **change)
                result = h.service().observe(h.request())
                self.assertEqual(result.state, m.ObservationState.BAD_IMAGE)
                self.assertFalse(any(t.top for t in result.targets))
                h.im = original

    def test_R01_context_without_active_request_cannot_accept_observation(self):
        h = self.h
        svc = h.service()
        ctx = h.context()
        svc.context_provider = lambda: replace(ctx, consumer=replace(ctx.consumer, request_id=None))
        result = svc.observe(h.request())
        self.assertEqual(result.state, m.ObservationState.BAD_IMAGE)
        self.assertEqual(result.reason.code, c.ErrorCode.STALE_REQUEST)

    def test_T02_board_inside_top_and_edge_crossing_region_are_excluded(self):
        h = self.h
        for polygon in [(m.PixelPoint(85, 80), m.PixelPoint(95, 80), m.PixelPoint(95, 90), m.PixelPoint(85, 90)),
                        (m.PixelPoint(0, 85), m.PixelPoint(190, 85), m.PixelPoint(190, 90), m.PixelPoint(0, 90))]:
            with self.subTest(polygon=polygon):
                svc = h.service(); ctx = replace(h.context(), board_polygons=(polygon,))
                svc.context_provider = lambda: ctx
                result = svc.observe(h.request())
                self.assertFalse(any(t.top is not None for t in result.targets))

    def test_R01_result_expiration_is_bounded_by_receipt_and_capture_age(self):
        h = self.h
        h.now = 111.8; h.sensor_now = 21.9
        result = h.service().observe(h.request())
        self.assertEqual(result.state, m.ObservationState.DETECTED)
        self.assertAlmostEqual(result.envelope.expires_at.seconds, 111.9, places=6)
        self.assertLess(result.envelope.expires_at.seconds, h.im.image.valid_until.seconds)

    def test_T04_duplicate_identifier_with_changed_content_is_conflict(self):
        h = self.h
        svc = h.service(); svc.observe(h.request())
        original = h.im
        h.advance()
        import hashlib
        new = bytes([81]) * len(h.data)
        h.im = replace(h.im, image=replace(h.im.image, evidence_id=original.image.evidence_id,
                                           sha256=hashlib.sha256(new).hexdigest()))
        h.expected = h.im
        h.raws[h.im.image.uri] = h.p.RawImage(320, 200, 'mono8', new)
        result = svc.observe(h.request())
        self.assertEqual(result.reason.code, c.ErrorCode.ID_CONFLICT)
        self.assertEqual(svc.accepted_frames, 1)

    def test_T05_long_capture_gap_cannot_confirm_identity(self):
        h = self.h
        svc = h.service(); svc.observe(h.request())
        h.advance(2, 30.)
        result = svc.observe(h.request())
        self.assertFalse(any(t.target_id for t in result.targets))

    def test_T05_ambiguous_nearest_matches_never_select_one_greedily(self):
        h = self.h
        svc = h.service(); svc.observe(h.request()); h.advance()
        old = svc.observe(h.request()).targets[0].target_id
        h.advance(3, 20.8)
        svc.backend = h.b.ExistingInterfaceBackend(lambda im, raw: h.batch(
            patterns=(h.pattern(), h.pattern('other')),
            entities=(h.entity(), h.entity('other'))), 'sim-model-v1', 'sim-classes-v1')
        result = svc.observe(h.request())
        self.assertEqual({t.target_id for t in result.targets if t.target_id}, {old})
        self.assertFalse(any(t.target_id and t.top is not None for t in result.targets))

    def test_R01_unconfigured_or_nonconvex_roi_does_not_accept_top(self):
        h = self.h
        for polygon in [(), (m.PixelPoint(0, 0), m.PixelPoint(300, 180), m.PixelPoint(300, 0), m.PixelPoint(0, 180))]:
            with self.subTest(polygon=polygon):
                svc = h.service(); ctx = replace(h.context(), allowed_polygon=polygon)
                svc.context_provider = lambda: ctx
                result = svc.observe(h.request())
                self.assertFalse(any(t.top for t in result.targets))
                self.assertIn(result.state, (m.ObservationState.BAD_IMAGE, m.ObservationState.BACKEND_FAILURE))

    def test_R01_image_quality_cannot_be_overridden_by_model_confidence(self):
        h = self.h
        h.cfg = replace(h.cfg, min_contrast=5.)
        result = h.service().observe(h.request())
        self.assertEqual(result.state, m.ObservationState.BAD_IMAGE)
        self.assertEqual(result.reason.field, 'image.quality')

    def test_R01_storage_failure_diagnostic_matches_returned_failure(self):
        h = self.h
        def disk_full(record):
            raise OSError('disk full')
        svc = h.service(audit=disk_full)
        result = svc.observe(h.request())
        self.assertEqual(svc.last_audit['result'], c.to_data(result))
        self.assertFalse(svc.last_audit['fresh_for_temporal_evidence'])

    def test_T06_tool_outside_same_body_does_not_confirm_relation(self):
        h = self.h
        def infer(im, raw):
            rim = tuple(m.PixelPoint(p.u + 160, p.v) for p in h.tool().rim.points)
            return h.batch(tools=(replace(h.tool(), rim=h.points(rim)),))
        svc = h.service(infer); svc.observe(h.request(m.Scene.TOOL)); h.advance()
        result = svc.observe(h.request(m.Scene.TOOL))
        self.assertIsNotNone(result.tool.center)
        self.assertEqual(result.tool.same_object, c.Fact.UNKNOWN)
        self.assertIsNone(result.tool.target_id)

    def test_R01_pose_source_reset_rejected_instead_of_reusing_old_stream_clock(self):
        h = self.h
        svc = h.service(); svc.observe(h.request()); h.advance()
        h.im = replace(h.im, image=replace(h.im.image, captured_at=replace(h.im.image.captured_at, clock_id='new-camera-clock')))
        h.expected = h.im
        result = svc.observe(h.request())
        self.assertEqual(result.reason.code, c.ErrorCode.CLOCK_MISMATCH)
        self.assertFalse(any(t.top for t in result.targets))

    def test_T05_corner_start_vertex_change_preserves_same_physical_target(self):
        h = self.h
        q = [m.PixelPoint(40, 30), m.PixelPoint(160, 30.1), m.PixelPoint(160, 130), m.PixelPoint(40, 130)]
        svc = h.service(lambda im, raw: h.batch(entities=(h.entity(corners=tuple(q)),)))
        svc.observe(h.request()); h.advance()
        target_id = svc.observe(h.request()).targets[0].target_id
        h.advance(3, 20.8)
        q[0], q[1] = m.PixelPoint(40, 30.1), m.PixelPoint(160, 30)
        result = svc.observe(h.request())
        self.assertEqual(len(result.targets), 1)
        self.assertEqual(result.targets[0].target_id, target_id)
        self.assertIsNotNone(result.targets[0].top)

    def test_T06_central_to_tool_same_view_preserves_identity_and_relation(self):
        h = self.h
        svc = h.service(lambda im, raw: h.batch(tools=(h.tool(),)))
        svc.observe(h.request()); h.advance()
        target_id = svc.observe(h.request()).targets[0].target_id
        h.advance(3, 20.8)
        result = svc.observe(h.request(m.Scene.TOOL))
        self.assertEqual(result.tool.same_object, c.Fact.CONFIRMED)
        self.assertEqual(result.tool.target_id, target_id)

    def test_T06_one_rim_over_two_visible_bodies_has_unknown_relationship(self):
        h = self.h
        def infer(im, raw):
            a = (m.PixelPoint(40, 30), m.PixelPoint(160, 30), m.PixelPoint(160, 140), m.PixelPoint(40, 140))
            b = (m.PixelPoint(100, 30), m.PixelPoint(220, 30), m.PixelPoint(220, 140), m.PixelPoint(100, 140))
            rim = tuple(m.PixelPoint(p.u + 50, p.v) for p in h.tool().rim.points)
            return h.batch(patterns=(h.pattern('a'), h.pattern('b', bounds=(190, 55, 215, 75))),
                           entities=(h.entity('a', a), h.entity('b', b)),
                           tools=(replace(h.tool('a'), rim=h.points(rim)),))
        svc = h.service(infer); svc.observe(h.request(m.Scene.TOOL)); h.advance()
        result = svc.observe(h.request(m.Scene.TOOL))
        self.assertEqual(len([t for t in result.targets if t.target_id]), 2)
        self.assertIsNotNone(result.tool.center)
        self.assertEqual(result.tool.same_object, c.Fact.UNKNOWN)
        self.assertIsNone(result.tool.target_id)

    def test_R01_missing_image_library_is_backend_unavailable_not_bad_pixels(self):
        h = self.h
        from unittest.mock import patch
        import builtins
        original = builtins.__import__
        def unavailable(name, *args, **kwargs):
            if name == 'cv2':
                raise ImportError('OpenCV unavailable')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=unavailable):
            result = h.service().observe(h.request())
        self.assertEqual(result.state, m.ObservationState.BACKEND_FAILURE)
        self.assertEqual(result.reason.code, c.ErrorCode.UNAVAILABLE)

    def test_T03_pattern_near_uncertain_top_boundary_requires_review(self):
        h = self.h
        q = (m.PixelPoint(40, 30), m.PixelPoint(160, 30), m.PixelPoint(160, 140), m.PixelPoint(40, 140))
        result = h.service(lambda im, raw: h.batch(patterns=(h.pattern(bounds=(40.2, 50, 60, 70)),),
                                                   entities=(h.entity(corners=q),))).observe(h.request())
        self.assertFalse(any(t.top is not None for t in result.targets))

    def test_T04_unknown_or_occluded_frame_never_offers_temporal_confirmation(self):
        h = self.h
        svc = h.service()
        svc.observe(h.request())
        self.assertFalse(svc.last_audit['fresh_for_temporal_evidence'])
        h.advance(); svc.observe(h.request())
        self.assertTrue(svc.last_audit['fresh_for_temporal_evidence'])
        self.assertFalse(svc.last_audit['stillness_certificate'])
        h.advance(3, 20.8)
        svc.backend = h.b.ExistingInterfaceBackend(lambda im, raw: h.batch((), ()), 'sim-model-v1', 'sim-classes-v1')
        svc.observe(h.request())
        self.assertFalse(svc.last_audit['fresh_for_temporal_evidence'])


class CompatibilityTests(unittest.TestCase):
    def test_R07_38_grammar_and_import_without_device_or_optional_image_packages(self):
        paths = list((PACKAGE / 'src/treasure_auto/perception').glob('*.py')) + list((PACKAGE / 'tests').glob('test_p04_*.py'))
        for path in paths:
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 8))
        script = '''import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'rospy','roslib','serial','socket','cv2','numpy'}:
   raise AssertionError('unexpected dependency at import: ' + fullname)
sys.meta_path.insert(0, Block()); sys.path.insert(0, sys.argv[1])
from treasure_auto.perception import config, pixels, geometry, backend, association, tool, service
print('offline-import: PASS')
'''
        done = subprocess.run([sys.executable, '-B', '-c', script, str(PACKAGE / 'src')], capture_output=True, text=True, timeout=15)
        self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == '__main__':
    unittest.main()
