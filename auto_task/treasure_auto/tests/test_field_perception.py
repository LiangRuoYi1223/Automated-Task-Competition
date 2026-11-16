"""Field facts exercise the actual P04 service, with explicit simulated evidence."""
from dataclasses import replace
import unittest

import test_p04_service as p04_fixtures
from test_p04_geometry import quad
from treasure_auto.contracts import core as c, models as m
from treasure_auto.geometry import surface


class FieldPerceptionTests(unittest.TestCase):
    setUp = p04_fixtures.ServiceTests.setUp
    context = p04_fixtures.ServiceTests.context
    points = p04_fixtures.ServiceTests.points
    entity = p04_fixtures.ServiceTests.entity
    pattern = p04_fixtures.ServiceTests.pattern
    batch = p04_fixtures.ServiceTests.batch
    service = p04_fixtures.ServiceTests.service
    advance = p04_fixtures.ServiceTests.advance
    request = p04_fixtures.ServiceTests.request

    def report(self, svc):
        self.assertTrue(hasattr(svc, 'policy_report'), 'P04 must publish typed current-frame policy facts')
        self.assertIsNotNone(svc.policy_report)
        return svc.policy_report

    def complete_pattern(self, *args, **kwargs):
        self.assertIn('complete', self.b.Pattern.__dataclass_fields__, 'pattern completeness must be explicit')
        kwargs['complete'] = c.Fact.CONFIRMED
        return self.pattern(*args, **kwargs)

    def known_layer(self, **changes):
        self.assertTrue(hasattr(surface, 'KnownLayerEvidence'), 'known layer needs explicit base-ground height adapter')
        values = dict(solution_id='layer-2', base_frame='base_link', ground_height_mm=-35.,
                      cube_edge_mm=50., confirmed_layer=2, ground_error_mm=1., cube_error_mm=.5,
                      horizontal=c.Fact.CONFIRMED, support_unchanged=c.Fact.CONFIRMED,
                      identity_consistent=c.Fact.CONFIRMED, support_ids=('lower-brick',),
                      evidence=(self.im.image,))
        values.update(changes)
        return surface.KnownLayerEvidence(**values)

    def test_default_legacy_pattern_does_not_claim_complete_pattern(self):
        svc = self.service()
        result = svc.observe(self.request())
        report = self.report(svc)
        self.assertEqual((report.pattern_count, report.independent_count, report.geometry_count), (0, 1, 1))
        self.assertEqual(report.envelope, result.envelope)
        self.assertEqual((report.observation_id, report.image_sha256, report.camera_frame, report.capture_pose),
                         (self.im.observation_id, self.im.image.sha256, self.im.camera_frame, self.im.capture_pose))
        self.assertTrue(report.current)
        self.assertEqual(report.review_targets, ())

    def test_three_counts_separate_print_patterns_independent_bodies_and_true_geometry(self):
        patterns = (self.complete_pattern(), self.complete_pattern('board', bounds=(200, 40, 220, 60)),
                    self.complete_pattern('edge-missing', bounds=(200, 110, 220, 130)))
        entities = (self.entity(), self.entity('board', kind='board'),
                    self.entity('edge-missing', corners=(m.PixelPoint(190, 100), m.PixelPoint(230, 100),
                        m.PixelPoint(230, 140), m.PixelPoint(190, 140)), edges=()))
        svc = self.service(lambda image, raw: self.batch(patterns, entities))
        result = svc.observe(self.request())
        report = self.report(svc)
        self.assertEqual((report.pattern_count, report.independent_count, report.geometry_count), (3, 2, 1))
        self.assertEqual(len(result.targets), 2)
        self.assertIn('edge-missing', [r.entity_id for r in report.review_targets])

    def test_duplicate_boxes_merge_but_incomplete_and_unrecognized_patterns_do_not_count(self):
        complete = self.complete_pattern()
        patterns = (complete, replace(complete, box=self.points((m.PixelPoint(61, 55), m.PixelPoint(86, 75)))),
                    self.pattern('partial', bounds=(160, 55, 185, 75), complete=c.Fact.REJECTED),
                    self.pattern('unknown', bounds=(260, 55, 285, 75), raw_class='not-a-class', complete=c.Fact.CONFIRMED))
        svc = self.service(lambda image, raw: self.batch(patterns))
        svc.observe(self.request())
        report = self.report(svc)
        self.assertEqual(report.pattern_count, 1)
        self.assertTrue({'partial', 'unknown'}.issubset({r.entity_id for r in report.review_targets}))

    def test_two_distinct_patterns_on_one_body_do_not_make_two_bricks(self):
        patterns = (self.complete_pattern(bounds=(60, 55, 68, 63)),
                    self.complete_pattern(bounds=(75, 65, 83, 73)))
        svc = self.service(lambda image, raw: self.batch(patterns))
        svc.observe(self.request())
        report = self.report(svc)
        self.assertEqual((report.pattern_count, report.independent_count, report.geometry_count), (2, 1, 1))

    def test_board_overlap_preserves_small_object_as_exploratory_not_normal_grab(self):
        self.assertIn('suspicious_small_object', self.b.EntityEvidence.__dataclass_fields__)
        parent_context = self.context
        board = (m.PixelPoint(30, 25), m.PixelPoint(150, 25), m.PixelPoint(150, 140), m.PixelPoint(30, 140))
        self.context = lambda: replace(parent_context(), board_polygons=(board,))
        patterns = (self.complete_pattern(), self.complete_pattern('board', bounds=(100, 100, 120, 120)))
        entity = self.entity(kind='unknown', body_confirmed=c.Fact.UNKNOWN, edges=(),
                             suspicious_small_object=c.Fact.CONFIRMED,
                             local_contact_confirmed=c.Fact.CONFIRMED, approach_clear=c.Fact.CONFIRMED)
        svc = self.service(lambda image, raw: self.batch(patterns, (entity, self.entity('board', kind='board'))))
        result = svc.observe(self.request())
        report = self.report(svc)
        self.assertFalse(any(t.top is not None for t in result.targets))
        self.assertEqual([x.entity_id for x in report.exploratory_candidates], ['body-1'])
        self.assertTrue(report.exploratory_candidates[0].eligible)
        self.assertEqual(report.exploratory_candidates[0].center, m.PixelPoint(72.5, 65.))
        self.assertEqual(report.geometry_count, 0)

    def test_board_print_and_unsupported_local_patch_never_become_exploratory_permission(self):
        self.assertIn('suspicious_small_object', self.b.EntityEvidence.__dataclass_fields__)
        svc = self.service(lambda image, raw: self.batch(
            (self.complete_pattern(), self.complete_pattern('board')),
            (self.entity(edges=(), suspicious_small_object=c.Fact.CONFIRMED), self.entity('board', kind='board'))))
        svc.observe(self.request())
        report = self.report(svc)
        self.assertEqual([x.entity_id for x in report.exploratory_candidates], ['body-1'])
        self.assertFalse(report.exploratory_candidates[0].eligible)
        self.assertTrue(report.exploratory_candidates[0].reasons)

    def test_absent_lidar_keeps_visual_geometry_and_conflict_withholds_it(self):
        self.assertIn('correspondence_conflicts', self.s.ObservationContext.__dataclass_fields__)
        parent_context = self.context
        self.context = lambda: replace(parent_context(), lidar_entity_ids=())
        svc = self.service()
        self.assertIsNotNone(svc.observe(self.request()).targets[0].top)
        self.assertEqual(self.report(svc).geometry_count, 1)
        self.advance()
        self.context = lambda: replace(parent_context(), correspondence_conflicts=('body-1',))
        svc.context_provider = self.context
        result = svc.observe(self.request())
        report = self.report(svc)
        self.assertFalse(any(t.top for t in result.targets))
        self.assertEqual(report.geometry_count, 0)
        self.assertIn('body-1', [r.entity_id for r in report.review_targets])

    def test_occluded_history_and_duplicate_frames_cannot_inflate_current_counts(self):
        svc = self.service()
        svc.observe(self.request())
        self.advance()
        svc.observe(self.request())
        duplicate = svc.observe(self.request())
        report = self.report(svc)
        self.assertEqual(duplicate.reason.code, c.ErrorCode.DUPLICATE)
        self.assertFalse(report.current)
        self.assertEqual((report.pattern_count, report.independent_count, report.geometry_count), (0, 0, 0))
        self.advance(3, 20.8)
        svc.backend = self.b.ExistingInterfaceBackend(lambda image, raw: self.batch((), ()),
                                                    'sim-model-v1', 'sim-classes-v1')
        result = svc.observe(self.request())
        self.assertTrue(result.targets)
        self.assertEqual((self.report(svc).independent_count, self.report(svc).geometry_count), (0, 0))

    def test_storage_failure_and_late_result_do_not_publish_policy_success(self):
        def bad_sink(audit):
            raise OSError('full')
        svc = self.service(audit=bad_sink)
        svc.observe(self.request())
        self.assertFalse(self.report(svc).current)
        svc = self.service()
        svc.observe(self.request())
        self.advance()
        def delayed(image, raw):
            batch = self.batch()
            self.expected = replace(self.expected, envelope=replace(self.expected.envelope, request_id='next'))
            return batch
        svc.backend = self.b.ExistingInterfaceBackend(delayed, 'sim-model-v1', 'sim-classes-v1')
        svc.observe(self.request())
        self.assertFalse(self.report(svc).current)
        self.assertEqual(self.report(svc).geometry_count, 0)

    def test_known_layer_uses_explicit_base_ground_and_propagates_measurement_error(self):
        layer = self.known_layer()
        plane = surface.known_layer_plane(layer, 'base_link')
        self.assertEqual(plane.height_mm, 65.)
        self.assertEqual(plane.error_mm, 2.)
        self.assertEqual(plane.support_ids, ('lower-brick',))
        self.assertEqual(plane.basis, m.PlaneBasis.KNOWN_DIMENSIONS)
        self.assertEqual(plane.evidence, (self.im.image,))

    def test_known_layer_rejects_tilt_changed_support_identity_conflict_and_invalid_measurements(self):
        for changes in (dict(horizontal=c.Fact.REJECTED), dict(support_unchanged=c.Fact.REJECTED),
                        dict(identity_consistent=c.Fact.UNKNOWN), dict(confirmed_layer=0),
                        dict(confirmed_layer=True), dict(cube_edge_mm=None), dict(cube_edge_mm=float('nan')),
                        dict(ground_height_mm=None), dict(ground_height_mm=float('nan')), dict(evidence=()),
                        dict(support_ids=()), dict(base_frame='map')):
            with self.subTest(changes=changes):
                layer = self.known_layer(**changes)
                with self.assertRaises(c.ContractError):
                    surface.known_layer_plane(layer, 'base_link')

    def test_real_observe_uses_known_layer_and_exposes_same_plane_to_geometry(self):
        self.assertIn('known_layer', self.b.EntityEvidence.__dataclass_fields__)
        parent_context = self.context
        self.context = lambda: replace(parent_context(), base_frame='base_link')
        svc = self.service(lambda image, raw: self.batch(entities=(
            self.entity(height_mm=None, plane_basis=m.PlaneBasis.UNKNOWN, known_layer=self.known_layer()),)))
        result = svc.observe(self.request())
        self.assertEqual(result.targets[0].top.height_mm, 65.)
        self.assertEqual(self.report(svc).known_planes[0][0], 'body-1')
        self.assertEqual(self.report(svc).known_planes[0][1].height_mm, 65.)
        self.advance()
        svc.backend = self.b.ExistingInterfaceBackend(lambda image, raw: self.batch(entities=(self.entity(
            known_layer=self.known_layer(support_unchanged=c.Fact.REJECTED)),)), 'sim-model-v1', 'sim-classes-v1')
        self.assertFalse(any(t.top for t in svc.observe(self.request()).targets))
        self.assertEqual(self.report(svc).known_planes, ())

    def test_context_conflict_arriving_during_inference_or_publication_invalidates_geometry(self):
        for phase in ('infer', 'audit'):
            with self.subTest(phase=phase):
                self.setUp()
                conflicts = []
                parent_context = lambda: p04_fixtures.ServiceTests.context(self)
                self.context = lambda: replace(parent_context(), correspondence_conflicts=tuple(conflicts))
                def infer(image, raw):
                    result = self.batch()
                    if phase == 'infer':
                        conflicts.append('body-1')
                    return result
                def audit(record):
                    if phase == 'audit':
                        conflicts.append('body-1')
                svc = self.service(infer, audit=audit)
                result = svc.observe(self.request())
                self.assertFalse(any(target.top for target in result.targets))
                self.assertFalse(self.report(svc).current)

    def test_duplicate_body_outline_is_not_two_independent_bricks(self):
        svc = self.service(lambda image, raw: self.batch(
            patterns=(self.complete_pattern(), self.complete_pattern('alias')),
            entities=(self.entity(), self.entity('alias'))))
        svc.observe(self.request())
        report = self.report(svc)
        self.assertEqual((report.pattern_count, report.independent_count, report.geometry_count), (1, 1, 0))

    def test_stale_known_layer_evidence_cannot_replace_current_height(self):
        old_layer = self.known_layer()
        self.advance()
        parent_context = self.context
        self.context = lambda: replace(parent_context(), base_frame='base_link')
        svc = self.service(lambda image, raw: self.batch(entities=(
            self.entity(height_mm=None, plane_basis=m.PlaneBasis.UNKNOWN, known_layer=old_layer),)))
        result = svc.observe(self.request())
        self.assertFalse(any(target.top for target in result.targets))
        self.assertEqual(self.report(svc).known_planes, ())

    def test_shifted_and_resampled_outline_aliases_do_not_inflate_independent_count(self):
        for resample in (False, True):
            with self.subTest(resample=resample):
                corners = tuple(m.PixelPoint(p.u + .2, p.v) for p in quad())
                alias = self.entity('alias', corners=corners)
                if resample:
                    alias = replace(alias, contour=self.points(corners))
                for entities in ((self.entity(), alias), (alias, self.entity())):
                    svc = self.service(lambda image, raw: self.batch(
                        (self.complete_pattern(), self.complete_pattern('alias')), entities))
                    result = svc.observe(self.request())
                    report = self.report(svc)
                    self.assertEqual(report.independent_count, 1)
                    self.assertEqual(report.geometry_count, 0)
                    self.assertFalse(any(t.top for t in result.targets))
                    self.assertTrue({'body-1', 'alias'}.issubset({r.entity_id for r in report.review_targets}))

    def test_outline_aliases_never_offer_multiple_or_eligible_exploratory_contacts(self):
        for delta in (0., .2):
            with self.subTest(delta=delta):
                contact = dict(edges=(), local_contact_confirmed=c.Fact.CONFIRMED, approach_clear=c.Fact.CONFIRMED)
                alias = self.entity('alias', corners=tuple(m.PixelPoint(p.u + delta, p.v) for p in quad()), **contact)
                svc = self.service(lambda image, raw: self.batch(
                    (self.complete_pattern(), self.complete_pattern('alias')), (self.entity(**contact), alias)))
                svc.observe(self.request())
                report = self.report(svc)
                self.assertEqual(report.independent_count, 1)
                self.assertEqual(len(report.exploratory_candidates), 1)
                self.assertFalse(report.exploratory_candidates[0].eligible)
                self.assertIsNone(report.exploratory_candidates[0].target_id)
                self.assertTrue(report.exploratory_candidates[0].reasons)

    def test_adjacent_bricks_with_touching_boundaries_keep_separate_exploratory_candidates(self):
        corners = (m.PixelPoint(20, 40), m.PixelPoint(70, 40), m.PixelPoint(70, 90), m.PixelPoint(20, 90))
        contact = dict(edges=(), local_contact_confirmed=c.Fact.CONFIRMED, approach_clear=c.Fact.CONFIRMED)
        svc = self.service(lambda image, raw: self.batch(
            (self.complete_pattern(bounds=(30, 50, 45, 65)), self.complete_pattern('neighbor', bounds=(80, 50, 95, 65))),
            (self.entity(corners=corners, **contact),
             self.entity('neighbor', corners=tuple(m.PixelPoint(p.u + 50, p.v) for p in corners), **contact))))
        svc.observe(self.request())
        report = self.report(svc)
        self.assertEqual(report.independent_count, 2)
        self.assertEqual(len(report.exploratory_candidates), 2)
        self.assertTrue(all(item.eligible for item in report.exploratory_candidates))

    def test_known_layer_top_preserves_signed_base_frame_height(self):
        parent_context = self.context
        self.context = lambda: replace(parent_context(), base_frame='base_link')
        layer = self.known_layer(ground_height_mm=-100., confirmed_layer=1, support_ids=('ground',))
        svc = self.service(lambda image, raw: self.batch(entities=(
            self.entity(height_mm=None, plane_basis=m.PlaneBasis.UNKNOWN, known_layer=layer),)))
        result = svc.observe(self.request())
        self.assertIsNotNone(result.targets[0].top)
        self.assertEqual(result.targets[0].top.height_mm, -50.)
        self.assertEqual(self.report(svc).known_planes[0][1].height_mm, -50.)
        self.assertEqual(self.report(svc).geometry_count, 1)
        self.assertEqual(c.loads(c.dumps(result)), result)

    def test_nonfinite_or_non_numeric_plane_height_still_withholds_geometry(self):
        for height in (None, float('nan'), float('inf'), -float('inf'), True):
            with self.subTest(height=height):
                svc = self.service(lambda image, raw: self.batch(entities=(self.entity(height_mm=height),)))
                result = svc.observe(self.request())
                self.assertFalse(any(t.top for t in result.targets))
                self.assertEqual(self.report(svc).geometry_count, 0)


if __name__ == '__main__':
    unittest.main()
