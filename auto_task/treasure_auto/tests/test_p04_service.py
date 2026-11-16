"""P04 integration using annotated SIMULATED frames; no recognition model is faked as live."""
from dataclasses import replace
import math
import unittest
from test_p04_pixels import module, config_dict, image_frame, context_for
from test_p04_geometry import quad, edge_points
from treasure_auto.contracts import core as c, models as m
from contract_fixtures.samples import sensor_stamp, stamp


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.p = module(self, 'pixels')
        self.b = module(self, 'backend')
        self.s = module(self, 'service')
        self.cfg = module(self, 'config').PerceptionConfig.from_dict(config_dict())
        self.im, self.data = image_frame()
        self.now, self.sensor_now = 111., 21.
        self.expected = self.im
        self.raws = {self.im.image.uri: self.p.RawImage(320, 200, 'mono8', self.data)}
        self.audit = []

    def context(self):
        return self.s.ObservationContext(context_for(self.expected, self.now), sensor_stamp(19),
                    sensor_stamp(self.sensor_now), (m.PixelPoint(0, 0), m.PixelPoint(319, 0),
                    m.PixelPoint(319, 199), m.PixelPoint(0, 199)), (), self.expected.camera_frame, self.expected.capture_pose)

    def points(self, points):
        return self.p.PixelSet(tuple(points), 'backend', 'identity-v1')

    def entity(self, name='body-1', corners=None, **changes):
        corners = corners or quad()
        edges = edge_points(corners)
        fields = dict(entity_id=name, kind='brick', body_confirmed=c.Fact.CONFIRMED,
                      member_ids=(name,), contour=self.points(tuple(p for e in edges for p in e)),
                      edges=tuple(self.points(e) for e in edges), plane_basis=m.PlaneBasis.CALIBRATION,
                      height_mm=50., evidence=(self.im.image,))
        fields.update(changes)
        return self.b.EntityEvidence(**fields)

    def pattern(self, name='body-1', raw_class='7', bounds=(60, 55, 85, 75), **changes):
        x1, y1, x2, y2 = bounds
        fields = dict(raw_class=raw_class, confidence=.95, entity_id=name,
                      box=self.points((m.PixelPoint(x1, y1), m.PixelPoint(x2, y2))))
        fields.update(changes)
        return self.b.Pattern(**fields)

    def batch(self, patterns=None, entities=None, tools=()):
        return self.b.BackendBatch(self.im.observation_id, self.im.image.sha256, c.Origin.SIMULATED,
                    'sim-model-v1', 'sim-classes-v1', self.p.TransformPlan('identity-v1', 320, 200, 320, 200,
                    320, 200, self.im.preprocessing, self.im.distortion),
                    (self.pattern(),) if patterns is None else patterns,
                    (self.entity(),) if entities is None else entities, tools)

    def service(self, callback=None, backend=True, audit=None):
        wrapped = self.b.ExistingInterfaceBackend(callback or (lambda image, raw: self.batch()),
                    'sim-model-v1', 'sim-classes-v1') if backend else None
        return self.s.PerceptionService(self.cfg, self.raws.__getitem__, self.context,
                                       wrapped, self.audit.append if audit is None else audit)

    def advance(self, n=2, t=20.5, pose=None, epoch=None):
        self.im, self.data = image_frame(n, t)
        if pose is not None:
            self.im = replace(self.im, capture_pose=pose)
        if epoch is not None:
            self.im = replace(self.im, envelope=replace(self.im.envelope, base_motion_epoch=epoch))
        self.expected = self.im
        self.raws[self.im.image.uri] = self.p.RawImage(320, 200, 'mono8', self.data)
        self.sensor_now = t + .1

    def request(self, scene=m.Scene.CENTRAL, previous=()):
        return m.PerceptionRequest(self.im, scene, previous)

    def test_R01_valid_empty_bad_image_backend_failure_and_unknown_class_are_distinct(self):
        cases = [('empty', m.ObservationState.EMPTY), ('bad', m.ObservationState.BAD_IMAGE),
                 ('failed', m.ObservationState.BACKEND_FAILURE), ('unknown', m.ObservationState.UNKNOWN_CLASS)]
        for mode, expected in cases:
            with self.subTest(mode=mode):
                if mode == 'failed':
                    def infer(image, raw):
                        raise RuntimeError('backend disconnected')
                else:
                    def infer(image, raw):
                        return self.batch(patterns=() if mode == 'empty' else (self.pattern(raw_class='99'),), entities=())
                svc = self.service(infer)
                original = self.raws[self.im.image.uri]
                if mode == 'bad':
                    self.raws[self.im.image.uri] = replace(original, data=b'')
                result = svc.observe(self.request())
                self.assertEqual(result.state, expected)
                self.assertEqual(result.envelope.producer, c.Owner.P04)
                self.assertEqual(c.loads(c.dumps(result)), result)
                self.raws[self.im.image.uri] = original

    def test_T07_no_backend_is_explainable_and_never_returns_previous_success(self):
        svc = self.service(backend=False)
        result = svc.observe(self.request())
        self.assertEqual(result.state, m.ObservationState.BACKEND_FAILURE)
        self.assertEqual(result.reason.code, c.ErrorCode.UNAVAILABLE)
        self.assertFalse(result.targets)
        self.assertEqual(len(self.audit), 1)

    def test_T02_board_and_physical_brick_in_one_frame_only_brick_can_be_confirmed(self):
        svc = self.service(lambda image, raw: self.batch(patterns=(self.pattern(), self.pattern('board', bounds=(200, 40, 230, 70))),
                    entities=(self.entity(), self.entity('board', kind='board'))))
        first = svc.observe(self.request())
        self.assertEqual(len(first.targets), 1)
        self.assertIsNone(first.targets[0].target_id)
        self.advance()
        second = svc.observe(self.request())
        self.assertEqual(len(second.targets), 1)
        self.assertIsNotNone(second.targets[0].target_id)
        self.assertEqual(second.targets[0].identity_confirmed, c.Fact.CONFIRMED)

    def test_T02_board_scene_has_semantics_without_grabbable_targets(self):
        result = self.service().observe(self.request(m.Scene.BOARD))
        self.assertEqual(result.state, m.ObservationState.DETECTED)
        self.assertEqual(result.targets, ())
        self.assertTrue(self.audit[-1]['patterns'])

    def test_T03_pattern_box_is_separate_from_true_surface(self):
        result = self.service().observe(self.request())
        self.assertEqual(result.targets[0].category, m.Category.DOLL)
        for actual, expected in zip(result.targets[0].top.corners, quad()):
            self.assertAlmostEqual(actual.u, expected.u)
            self.assertAlmostEqual(actual.v, expected.v)
        self.assertEqual(self.audit[-1]['patterns'][0]['full_box'], [[60., 55.], [85., 75.]])
        self.assertEqual(len(self.audit[-1]['surfaces'][0]['corners']), 4)

    def test_T03_shadow_joined_bricks_missing_body_and_plane_do_not_generate_top(self):
        for change in [dict(kind='shadow'), dict(member_ids=('a', 'b')), dict(body_confirmed=c.Fact.UNKNOWN),
                       dict(plane_basis=m.PlaneBasis.UNKNOWN), dict(edges=())]:
            with self.subTest(change=change):
                svc = self.service(lambda image, raw: self.batch(entities=(self.entity(**change),)))
                result = svc.observe(self.request())
                self.assertFalse(any(t.top is not None for t in result.targets))
                self.assertTrue(self.audit[-1]['rejections'])

    def test_T03_mismatched_or_ambiguous_entity_containment_rejected(self):
        for patterns, entities in [((self.pattern(bounds=(240, 150, 260, 170)),), (self.entity(),)),
                                    ((self.pattern(),), (self.entity(), self.entity('other'))),
                                    ((self.pattern('missing'),), (self.entity(),))]:
            with self.subTest(patterns=patterns):
                result = self.service(lambda image, raw: self.batch(patterns, entities)).observe(self.request())
                self.assertFalse(any(t.top is not None for t in result.targets))

    def test_T04_duplicate_frames_do_not_confirm_and_offer_no_new_stillness_evidence(self):
        svc = self.service()
        first = svc.observe(self.request())
        for _ in range(5):
            duplicate = svc.observe(self.request())
            self.assertEqual(duplicate.state, m.ObservationState.BAD_IMAGE)
            self.assertEqual(duplicate.reason.code, c.ErrorCode.DUPLICATE)
            self.assertFalse(any(t.top is not None for t in duplicate.targets))
        self.assertEqual(svc.accepted_frames, 1)
        self.assertEqual(self.audit[-1]['fresh_for_temporal_evidence'], False)
        self.assertEqual(first.targets[0].target_id, None)

    def test_T05_three_same_class_bricks_keep_ids_across_new_frames_and_occlusion(self):
        def infer(image, raw):
            patterns, entities = [], []
            for i in range(3):
                x = 15 + 95 * i
                q = (m.PixelPoint(x, 30), m.PixelPoint(x + 65, 30), m.PixelPoint(x + 65, 100), m.PixelPoint(x, 100))
                patterns.append(self.pattern(str(i), bounds=(x + 10, 50, x + 30, 70)))
                entities.append(self.entity(str(i), q))
            return self.batch(tuple(patterns), tuple(entities))
        svc = self.service(infer)
        svc.observe(self.request()); self.advance()
        second = svc.observe(self.request())
        ids = {t.target_id for t in second.targets}
        self.assertEqual(len(ids), 3); self.assertNotIn(None, ids)
        self.advance(3, 20.8)
        svc.backend = self.b.ExistingInterfaceBackend(lambda image, raw: self.batch((), ()), 'sim-model-v1', 'sim-classes-v1')
        missing = svc.observe(self.request())
        self.assertEqual({t.target_id for t in missing.targets}, ids)
        self.assertTrue(all(t.visibility is m.Visibility.OCCLUDED and t.top is None for t in missing.targets))
        self.advance(4, 21.1)
        svc.backend = self.b.ExistingInterfaceBackend(infer, 'sim-model-v1', 'sim-classes-v1')
        returned = svc.observe(self.request())
        self.assertEqual({t.target_id for t in returned.targets}, ids)

    def test_T05_camera_move_or_rebuild_leaves_unknown_without_new_physical_ids(self):
        for move in ('camera', 'base', 'rebuild'):
            with self.subTest(move=move):
                self.setUp()
                svc = self.service(); svc.observe(self.request()); self.advance()
                confirmed = svc.observe(self.request()).targets[0].target_id
                self.advance(3, 20.8, pose=c.MapPose('map', 1, 0, 0, 'm', 'rad', .01, .01) if move == 'camera' else None,
                             epoch=1 if move == 'base' else None)
                if move == 'rebuild':
                    self.im = replace(self.im, envelope=replace(self.im.envelope, rebuild_generation=1))
                    self.expected = self.im
                result = svc.observe(self.request())
                self.assertEqual({t.target_id for t in result.targets if t.target_id}, {confirmed})
                self.assertTrue(any(t.target_id is None and t.identity_confirmed is c.Fact.UNKNOWN for t in result.targets))
                self.assertFalse(any(t.target_id and t.top is not None for t in result.targets))

    def test_T05_restart_previous_targets_not_trusted_as_current_geometry(self):
        svc = self.service(); svc.observe(self.request()); self.advance()
        old = svc.observe(self.request())
        self.advance(3, 20.8)
        fresh_service = self.service()
        result = fresh_service.observe(self.request(previous=old.targets))
        self.assertTrue(any(t.target_id == old.targets[0].target_id and t.top is None for t in result.targets))
        self.assertFalse(any(t.target_id and t.top is not None for t in result.targets))

    def test_R02_bad_batch_version_old_frame_and_bad_confidence_are_failures(self):
        for change in [dict(model_version='wrong'), dict(class_map_version='wrong'), dict(observation_id='old'),
                       dict(origin=c.Origin.LIVE), dict(image_sha256='0' * 64),
                       dict(patterns=(self.pattern(confidence=float('nan')),))]:
            with self.subTest(change=change):
                result = self.service(lambda image, raw: replace(self.batch(), **change)).observe(self.request())
                self.assertEqual(result.state, m.ObservationState.BACKEND_FAILURE)
                self.assertFalse(any(t.top for t in result.targets))

    def test_R02_duplicate_detection_is_one_candidate_conflicting_classes_unknown(self):
        result = self.service(lambda image, raw: self.batch(patterns=(self.pattern(), self.pattern()))).observe(self.request())
        self.assertEqual(len(result.targets), 1)
        conflict = self.service(lambda image, raw: self.batch(patterns=(self.pattern(), self.pattern(raw_class='9')))).observe(self.request())
        self.assertEqual(conflict.state, m.ObservationState.UNKNOWN_CLASS)
        self.assertTrue(all(t.category is None for t in conflict.targets))

    def test_R01_late_result_rechecked_after_backend_and_does_not_replace_latest(self):
        svc = self.service()
        original = svc.observe(self.request())
        self.advance()
        def delayed(image, raw):
            result = self.batch()
            self.expected = replace(self.expected, envelope=replace(self.expected.envelope, request_id='next-request'))
            return result
        svc.backend = self.b.ExistingInterfaceBackend(delayed, 'sim-model-v1', 'sim-classes-v1')
        late = svc.observe(self.request())
        self.assertEqual(late.reason.code, c.ErrorCode.STALE_REQUEST)
        self.assertIs(svc.latest, original)
        self.assertFalse(self.audit[-1]['published_current'])

    def test_R01_backend_latency_can_expire_evidence(self):
        def slow(image, raw):
            result = self.batch(); self.now = 114
            return result
        result = self.service(slow).observe(self.request())
        self.assertEqual(result.reason.code, c.ErrorCode.EXPIRED)
        self.assertFalse(any(t.top for t in result.targets))

    def test_R01_audit_failure_prevents_publishing_fresh_candidate(self):
        def broken_audit(record):
            raise OSError('disk full')
        svc = self.service(audit=broken_audit)
        result = svc.observe(self.request())
        self.assertEqual(result.reason.code, c.ErrorCode.STORAGE_FAILURE)
        self.assertFalse(any(t.top for t in result.targets))
        self.assertIsNone(svc.latest)

    def tool(self, name='body-1', visibility=m.Visibility.VISIBLE, evidence=None, source='visual_edge'):
        rim = tuple(m.PixelPoint(80 + 8 * math.cos(t), 70 + 5 * math.sin(t)) for t in [i * math.pi / 16 for i in range(32)])
        return self.b.ToolEdges(self.points(rim), name, visibility,
                               (self.im.image,) if evidence is None else evidence, source)

    def test_T06_new_visual_rim_center_and_same_brick_observation(self):
        svc = self.service(lambda image, raw: self.batch(tools=(self.tool(),)))
        svc.observe(self.request(m.Scene.TOOL)); self.advance()
        result = svc.observe(self.request(m.Scene.TOOL))
        self.assertAlmostEqual(result.tool.center.u, 80, places=4)
        self.assertAlmostEqual(result.tool.center.v, 70, places=4)
        self.assertEqual(result.tool.same_object, c.Fact.CONFIRMED)
        self.assertEqual(result.tool.target_id, result.targets[0].target_id)
        self.assertEqual(result.tool.evidence[0].captured_at, self.im.image.captured_at)
        self.assertIn('tool', self.audit[-1])

    def test_T06_occluded_multiple_predicted_echo_or_partial_rim_returns_unknown(self):
        for mode in ('occluded', 'multiple', 'prediction', 'echo', 'partial'):
            with self.subTest(mode=mode):
                t = self.tool()
                if mode == 'occluded': t = replace(t, visibility=m.Visibility.OCCLUDED)
                if mode == 'prediction': t = replace(t, source='control_prediction')
                if mode == 'echo': t = replace(t, evidence=(replace(self.im.image, kind=c.EvidenceKind.ECHO),))
                if mode == 'partial': t = replace(t, rim=self.points(t.rim.points[:12]))
                result = self.service(lambda image, raw: self.batch(tools=(t, t) if mode == 'multiple' else (t,))).observe(self.request(m.Scene.TOOL))
                self.assertEqual(result.tool.same_object, c.Fact.UNKNOWN)
                self.assertIsNone(result.tool.center)


if __name__ == '__main__':
    unittest.main()
