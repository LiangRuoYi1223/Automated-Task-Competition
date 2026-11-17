"""Offline metric arithmetic and provenance checks, not field accuracy claims."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
import unittest


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))


def quad(x=30, y=30, width=100, height=100):
    return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        try:
            self.module = importlib.import_module('treasure_auto.vision_workbench.evaluation')
        except ModuleNotFoundError:
            self.fail('offline candidate evaluator is missing')

    def pair(self, index=0, candidates=None, instances=None, split='acceptance', scene=None):
        digest = '{:064x}'.format(index + 1)
        frame = dict(index=index, source=dict(width=400, height=300, pixel_sha256=digest),
                     candidates=[dict(corners=p) for p in (candidates if candidates is not None else [quad()])],
                     unresolved=[], timing_ms=25., budget_truncated=False)
        reference = dict(index=index, width=400, height=300, pixel_sha256=digest,
                         scene_id=scene or 'scene_{}'.format(index), split=split,
                         instances=[dict(corners=p, visibility='complete') for p in
                                    (instances if instances is not None else [quad()])])
        return frame, reference

    def references(self, *frames):
        return dict(schema_version=1, measurement='manual_reference_unverified', frames=list(frames))

    def test_exact_match_reports_metrics_without_acceptance_claim(self):
        frame, reference = self.pair()
        result = self.module.evaluate_frames([frame], self.references(reference))
        self.assertEqual(result['matched_instances'], 1)
        self.assertEqual(result['precision'], 1.)
        self.assertEqual(result['recall'], 1.)
        self.assertEqual(result['corner_error_px']['median'], 0.)
        self.assertEqual(result['algorithm_time_ms']['p95'], 25.)
        self.assertFalse(result['sufficient_data'])
        self.assertIsNone(result['passed'])
        self.assertFalse(result['actionable'])
        json.dumps(result, allow_nan=False)

    def test_matching_maximizes_cardinality_instead_of_greedy_iou(self):
        a, b = quad(30, 30), quad(50, 30)
        flexible, unique = quad(40, 30), quad(20, 30)
        frame, reference = self.pair(candidates=[flexible, unique], instances=[a, b])
        result = self.module.evaluate_frames([frame], self.references(reference))
        self.assertEqual(result['matched_instances'], 2)
        self.assertEqual(result['precision'], 1.)
        self.assertEqual(result['recall'], 1.)

    def test_duplicate_candidate_is_false_positive_and_corners_align_cyclically(self):
        points = quad()
        reversed_cycle = list(reversed(points[2:] + points[:2]))
        frame, reference = self.pair(candidates=[reversed_cycle, points])
        result = self.module.evaluate_frames([frame], self.references(reference))
        self.assertEqual(result['matched_instances'], 1)
        self.assertEqual(result['false_positive_candidates'], 1)
        self.assertEqual(result['precision'], .5)
        self.assertEqual(result['corner_error_px']['p95'], 0.)

    def test_negative_image_rate_and_unresolved_do_not_hide_missed_target(self):
        a, ra = self.pair(candidates=[])
        a['unresolved'] = [dict(corners=quad(), reason='side unresolved')]
        b, rb = self.pair(index=1, instances=[])
        c, rc = self.pair(index=2, candidates=[], instances=[])
        result = self.module.evaluate_frames([a, b, c], self.references(ra, rb, rc))
        self.assertEqual(result['recall'], 0.)
        self.assertEqual(result['precision'], 0.)
        self.assertEqual(result['unresolved_candidates'], 1)
        self.assertEqual(result['negative_image_false_positive_rate'], .5)
        self.assertIsNone(result['corner_error_px']['p95'])

    def test_partial_references_are_difficult_cases_and_not_negative_images(self):
        frame, reference = self.pair(candidates=[])
        reference['instances'][0]['visibility'] = 'partial'
        result = self.module.evaluate_frames([frame], self.references(reference))
        self.assertEqual(result['complete_instances'], 0)
        self.assertEqual(result['partial_instances'], 1)
        self.assertEqual(result['negative_images'], 0)
        self.assertIsNone(result['recall'])
        self.assertIsNone(result['precision'])

    def test_truncated_search_keeps_miss_in_metrics_and_marks_incomplete(self):
        frame, reference = self.pair(candidates=[])
        frame['budget_truncated'] = True
        result = self.module.evaluate_frames([frame], self.references(reference))
        self.assertEqual(result['recall'], 0.)
        self.assertEqual(result['truncated_frames'], 1)
        self.assertFalse(result['acceptance_evaluated'])

    def test_tuning_frames_are_excluded_from_acceptance_metrics(self):
        a, ra = self.pair()
        b, rb = self.pair(index=1, candidates=[], split='tuning')
        result = self.module.evaluate_frames([a, b], self.references(ra, rb))
        self.assertEqual(result['evaluated_frames'], 1)
        self.assertEqual(result['tuning_frames_excluded'], 1)
        self.assertEqual(result['recall'], 1.)

    def test_scene_or_pixel_leakage_between_splits_is_rejected(self):
        a, ra = self.pair()
        b, rb = self.pair(index=1, split='tuning', scene=ra['scene_id'])
        with self.assertRaisesRegex(ValueError, 'split'):
            self.module.evaluate_frames([a, b], self.references(ra, rb))
        rb['scene_id'] = 'other'
        rb['pixel_sha256'] = ra['pixel_sha256']
        b['source']['pixel_sha256'] = ra['pixel_sha256']
        with self.assertRaisesRegex(ValueError, 'split'):
            self.module.evaluate_frames([a, b], self.references(ra, rb))

    def test_reference_hash_and_dimensions_must_match_report_source(self):
        frame, reference = self.pair()
        for key, bad in [('pixel_sha256', 'f' * 64), ('width', 401), ('height', 301)]:
            mutated = deepcopy(reference)
            mutated[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.module.evaluate_frames([frame], self.references(mutated))

    def test_duplicate_or_missing_mappings_are_rejected(self):
        frame, reference = self.pair()
        for frames, refs in [([frame, frame], [reference]), ([frame], [reference, reference]),
                             ([], [reference]), ([frame], [])]:
            with self.subTest(counts=(len(frames), len(refs))), self.assertRaises(ValueError):
                self.module.evaluate_frames(frames, self.references(*refs))

    def test_bad_quads_thresholds_schema_and_timing_are_rejected(self):
        frame, reference = self.pair()
        for corners in [[[30, 30], [130, 130], [130, 30], [30, 130]],
                        [[30, 30], [130, 30], [50, 50], [30, 130]],
                        [[30, 30], [130, 30], [float('nan'), 130], [30, 130]],
                        quad(350, 30)]:
            bad = deepcopy(frame)
            bad['candidates'][0]['corners'] = corners
            with self.subTest(corners=corners), self.assertRaises(ValueError):
                self.module.evaluate_frames([bad], self.references(reference))
        for threshold in [True, 0, 1.1, float('nan')]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.module.evaluate_frames([frame], self.references(reference), threshold)
        for key, bad_value in [('schema_version', True), ('measurement', 'validated')]:
            refs = self.references(reference)
            refs[key] = bad_value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.module.evaluate_frames([frame], refs)
        frame['timing_ms'] = float('inf')
        with self.assertRaises(ValueError):
            self.module.evaluate_frames([frame], self.references(reference))

    def test_sufficiency_counts_independent_scenes_not_neighboring_frames(self):
        frames, refs = [], []
        for index in range(90):
            f, r = self.pair(index=index, scene='same_scene')
            frames.append(f)
            refs.append(r)
        result = self.module.evaluate_frames(frames, self.references(*refs))
        self.assertEqual(result['independent_positive_scenes'], 1)
        self.assertFalse(result['sufficient_data'])

    def test_report_wrapper_refuses_incomplete_report(self):
        frame, reference = self.pair()
        report = dict(schema_version=1, processing_complete=False, frames=[frame])
        with self.assertRaisesRegex(ValueError, 'complete'):
            self.module.evaluate_report(report, self.references(reference))
        report['processing_complete'] = True
        report['origin'] = 'synthetic_demo'
        result = self.module.evaluate_report(report, self.references(reference))
        self.assertFalse(result['real_world_validated'])
        self.assertEqual(result['report_origin'], 'synthetic_demo')
        self.assertIsNone(result['passed'])

    def test_identical_pixels_cannot_be_renamed_into_independent_scenes(self):
        a, ra = self.pair(scene='first')
        b, rb = self.pair(index=1, scene='invented-second')
        b['source']['pixel_sha256'] = a['source']['pixel_sha256']
        rb['pixel_sha256'] = ra['pixel_sha256']
        with self.assertRaisesRegex(ValueError, 'scene'):
            self.module.evaluate_frames([a, b], self.references(ra, rb))

    def test_report_wrapper_checks_stored_frozen_configuration_hash(self):
        frame, reference = self.pair()
        report = dict(schema_version=1, processing_complete=True, frames=[frame],
                      configuration={'method': 'lsd'}, configuration_sha256='b' * 64)
        with self.assertRaisesRegex(ValueError, 'configuration_sha256'):
            self.module.evaluate_report(report, self.references(reference))


if __name__ == '__main__':
    unittest.main()
