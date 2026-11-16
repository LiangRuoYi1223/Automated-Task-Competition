"""Synthetic fixtures only: these checks do not establish competition accuracy."""
import importlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
CATEGORIES = ('doll', 'clock', 'banana', 'cake')


def synthetic_pattern(category):
    """Different centered marks, not substitutes for the actual competition art."""
    image = np.full((160, 160, 3), 220, dtype=np.uint8)
    color = (25, 25, 25)
    if category == 'doll':
        cv2.circle(image, (80, 50), 17, color, -1)
        cv2.rectangle(image, (70, 65), (90, 112), color, -1)
        cv2.line(image, (52, 78), (108, 78), color, 9)
        cv2.line(image, (80, 105), (58, 129), color, 9)
    elif category == 'clock':
        cv2.circle(image, (80, 80), 43, color, 7)
        cv2.line(image, (80, 80), (80, 49), color, 6)
        cv2.line(image, (80, 80), (103, 94), color, 6)
    elif category == 'banana':
        cv2.ellipse(image, (66, 67), (49, 48), -20, 10, 155, color, 13)
        cv2.line(image, (106, 68), (118, 51), color, 8)
    else:
        cv2.rectangle(image, (47, 78), (114, 120), color, -1)
        cv2.rectangle(image, (54, 59), (108, 71), color, -1)
        cv2.line(image, (65, 52), (65, 37), color, 5)
        cv2.line(image, (94, 52), (94, 37), color, 5)
    return image


def write_png(path, pixels):
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', pixels)
    if not ok:
        raise AssertionError('could not encode synthetic fixture')
    path.write_bytes(encoded.tobytes())


def write_templates(directory, categories=CATEGORIES):
    for category in categories:
        write_png(directory / category / 'synthetic.png', synthetic_pattern(category))


class TemplateClassifierTests(unittest.TestCase):
    def classifier(self, *args, **kwargs):
        try:
            module = importlib.import_module('treasure_auto.vision_workbench.templates')
        except ModuleNotFoundError as error:
            self.fail('template classifier implementation missing: ' + str(error))
        return module.TemplateClassifier(*args, **kwargs)

    def assert_unknown(self, result, reason, ready):
        self.assertEqual(result['category'], 'unknown')
        self.assertEqual(result['reason'], reason)
        self.assertIs(result['templates_ready'], ready)
        self.assertIsInstance(result['scores'], dict)
        self.assertIsInstance(result['method'], str)
        json.dumps(result, allow_nan=False)

    def test_no_templates_is_explicit_preparation_state(self):
        classifier = self.classifier()
        self.assertFalse(classifier.ready)
        result = classifier.classify(synthetic_pattern('clock'))
        self.assert_unknown(result, 'templates_missing', False)
        self.assertIsNone(result['score'])
        self.assertEqual(result['scores'], {})

    def test_partial_templates_cannot_claim_a_known_category(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_templates(directory, ('clock',))
            classifier = self.classifier(directory)
            self.assertFalse(classifier.ready)
            self.assert_unknown(classifier.classify(synthetic_pattern('clock')),
                                'templates_missing', False)

    def test_manifest_binds_exact_template_files_and_cannot_be_mutated(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_templates(directory)
            classifier = self.classifier(directory)
            self.assertTrue(hasattr(classifier, 'manifest'), 'missing reproducible template manifest')
            manifest = classifier.manifest
            self.assertEqual(len(manifest['files']), 4)
            self.assertEqual(manifest['rotation_step_degrees'], 5)
            for entry in manifest['files']:
                self.assertEqual(entry['sha256'], hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest())
                self.assertIn(entry['category'], CATEGORIES)
            manifest['files'].clear()
            self.assertEqual(len(classifier.manifest['files']), 4)

    def test_all_four_classes_recognize_their_distinct_synthetic_pattern(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / '中文模板'
            write_templates(directory)
            classifier = self.classifier(directory)
            self.assertTrue(classifier.ready)
            for category in CATEGORIES:
                with self.subTest(category=category):
                    query = synthetic_pattern(category)
                    unchanged = query.copy()
                    result = classifier.classify(query)
                    self.assertEqual(result['category'], category, result)
                    self.assertEqual(result['reason'], 'matched')
                    self.assertTrue(result['templates_ready'])
                    self.assertGreater(result['score'], 0.95)
                    self.assertEqual(set(result['scores']), set(CATEGORIES))
                    self.assertEqual(result['score'], result['scores'][category])
                    for score in result['scores'].values():
                        self.assertIsInstance(score, float)
                        self.assertGreaterEqual(score, -1.0)
                        self.assertLessEqual(score, 1.0)
                    self.assertTrue(np.array_equal(query, unchanged))
                    json.dumps(result, allow_nan=False)

    def test_planar_rotations_include_between_grid_angles(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_templates(directory)
            classifier = self.classifier(directory)
            image = synthetic_pattern('doll')
            for angle in (90, 180, 270, 37):
                with self.subTest(angle=angle):
                    matrix = cv2.getRotationMatrix2D((79.5, 79.5), angle, 1.0)
                    query = cv2.warpAffine(image, matrix, (160, 160),
                                           borderValue=(220, 220, 220))
                    result = classifier.classify(query)
                    self.assertEqual(result['category'], 'doll', result)
                    self.assertGreater(result['score'], 0.80)

    def test_global_brightness_and_contrast_change_keeps_pattern(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_templates(directory)
            classifier = self.classifier(directory)
            query = (synthetic_pattern('banana').astype(np.float32) * 0.6 + 20).astype(np.uint8)
            self.assertEqual(classifier.classify(query)['category'], 'banana')

    def test_blank_and_too_small_queries_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_templates(directory)
            classifier = self.classifier(directory)
            self.assert_unknown(classifier.classify(np.full((160, 160, 3), 127, np.uint8)),
                                'image_blank', True)
            self.assert_unknown(classifier.classify(np.zeros((8, 8, 3), np.uint8)),
                                'image_too_small', True)

    def test_unrelated_pixels_reject_for_low_similarity(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_templates(directory)
            classifier = self.classifier(directory)
            query = np.random.RandomState(31).randint(0, 256, (160, 160, 3)).astype(np.uint8)
            result = classifier.classify(query)
            self.assert_unknown(result, 'low_score', True)
            self.assertIsInstance(result['score'], float)
            self.assertLess(result['score'], 0.75)

    def test_identical_art_in_two_classes_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_templates(directory)
            write_png(directory / 'clock' / 'synthetic.png', synthetic_pattern('doll'))
            result = self.classifier(directory).classify(synthetic_pattern('doll'))
            self.assert_unknown(result, 'ambiguous', True)
            self.assertGreater(result['score'], 0.95)
            self.assertAlmostEqual(result['scores']['doll'], result['scores']['clock'])

    def test_close_runner_up_rejects_even_when_best_match_is_exact(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_templates(directory)
            competitor = synthetic_pattern('doll')
            cv2.circle(competitor, (110, 110), 8, (25, 25, 25), -1)
            write_png(directory / 'clock' / 'synthetic.png', competitor)
            query = synthetic_pattern('doll')
            result = self.classifier(directory).classify(query)
            self.assert_unknown(result, 'ambiguous', True)
            self.assertGreater(result['scores']['doll'], result['scores']['clock'])
            self.assertLess(result['scores']['doll'] - result['scores']['clock'], 0.08)
            # A deliberately relaxed margin accepts the same distinguishable pair.
            relaxed = self.classifier(directory, min_margin=0.0).classify(query)
            self.assertEqual(relaxed['category'], 'doll', relaxed)

    def test_zero_margin_does_not_accept_an_exact_cross_class_tie(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_templates(directory)
            write_png(directory / 'clock' / 'synthetic.png', synthetic_pattern('doll'))
            result = self.classifier(directory, min_margin=0.0).classify(synthetic_pattern('doll'))
            self.assert_unknown(result, 'ambiguous', True)

    def test_multiple_templates_per_class_and_jpeg_are_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_templates(directory)
            other = synthetic_pattern('doll')
            cv2.circle(other, (112, 118), 10, (25, 25, 25), -1)
            ok, encoded = cv2.imencode('.jpg', other)
            self.assertTrue(ok)
            (directory / 'doll' / 'second.JPEG').write_bytes(encoded.tobytes())
            classifier = self.classifier(directory)
            self.assertEqual(classifier.classify(other)['category'], 'doll')

    def test_nonimage_files_do_not_fill_missing_classes(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            write_templates(directory, ('doll', 'clock', 'banana'))
            (directory / 'cake').mkdir()
            (directory / 'cake' / 'README.txt').write_text('no real template yet')
            classifier = self.classifier(directory)
            self.assertFalse(classifier.ready)
            self.assert_unknown(classifier.classify(synthetic_pattern('cake')),
                                'templates_missing', False)

    def test_invalid_parameters_reject_instead_of_disabling_thresholds(self):
        for kwargs in ({'min_score': float('nan')}, {'min_score': float('inf')},
                       {'min_score': -0.1}, {'min_score': 1.1}, {'min_score': True},
                       {'min_margin': -0.1}, {'min_margin': 1.1},
                       {'min_margin': float('nan')}, {'min_margin': '0.1'}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    self.classifier(**kwargs)

    def test_bad_directory_and_bad_image_are_explicit_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with self.assertRaises(FileNotFoundError):
                self.classifier(directory / 'missing')
            (directory / 'file').write_text('not a directory')
            with self.assertRaises(NotADirectoryError):
                self.classifier(directory / 'file')
            with self.assertRaises(ValueError):
                self.classifier('')
            (directory / 'doll').write_text('not a class directory')
            with self.assertRaises(NotADirectoryError):
                self.classifier(directory)
            classifier = self.classifier()
            for image in (None, [], np.zeros((20, 20), np.uint8),
                          np.zeros((20, 20, 4), np.uint8), np.zeros((0, 20, 3), np.uint8),
                          np.zeros((40, 40, 3), np.float32)):
                with self.subTest(image_type=type(image), shape=getattr(image, 'shape', None)):
                    with self.assertRaises((TypeError, ValueError)):
                        classifier.classify(image)

    def test_blank_too_small_and_corrupt_template_files_are_explicit_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = directory / 'doll' / 'invalid.png'
            for pixels in (np.zeros((160, 160, 3), np.uint8), np.zeros((8, 8, 3), np.uint8)):
                write_png(path, pixels)
                with self.assertRaises(ValueError):
                    self.classifier(directory)
            path.write_bytes(b'not a PNG')
            with self.assertRaises(ValueError):
                self.classifier(directory)

    def test_module_import_does_not_import_opencv_or_numpy(self):
        self.classifier()  # Expected RED failure is the missing feature, not a subprocess error.
        script = ('import sys; sys.path.insert(0, ' + repr(str(PACKAGE / 'src')) + '); '
                  'from treasure_auto.vision_workbench.templates import TemplateClassifier; '
                  'assert not TemplateClassifier().ready; '
                  'assert "cv2" not in sys.modules; assert "numpy" not in sys.modules')
        result = subprocess.run([sys.executable, '-B', '-c', script],
                                capture_output=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
