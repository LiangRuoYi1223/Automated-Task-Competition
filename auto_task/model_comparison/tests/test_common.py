import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

from brick_benchmark.contracts import Instance
from brick_benchmark.data import decode_segmentation, encode_mask, load_dataset
from brick_benchmark.geometry import extract_quad
from brick_benchmark.metrics import evaluate_frames


class CommonTests(unittest.TestCase):
    def test_rle_preserves_hole_and_disconnected_parts(self):
        mask = np.zeros((16, 20), bool)
        mask[1:8, 1:8] = True
        mask[3:5, 3:5] = False
        mask[10:14, 15:19] = True
        np.testing.assert_array_equal(decode_segmentation(encode_mask(mask), 16, 20), mask)

    def test_geometry_uses_perspective_quad_and_rejects_cut_corner(self):
        mask = np.zeros((100, 120), np.uint8)
        cv2.fillConvexPoly(mask, np.array([[15, 10], [100, 25], [85, 85], [25, 75]]), 1)
        corners, reason = extract_quad(mask.astype(bool))
        self.assertIsNone(reason)
        self.assertEqual(len(corners), 4)
        self.assertLess(np.linalg.norm(np.array(corners[0]) - [15, 10]), 2)
        mask[0:45, 0:45] = 0
        self.assertIsNone(extract_quad(mask.astype(bool))[0])

    def test_hole_does_not_become_full_grasp_quad(self):
        mask = np.zeros((50, 50), bool)
        mask[5:45, 5:45] = True
        mask[20:30, 20:30] = False
        self.assertIsNone(extract_quad(mask)[0])

    def test_small_fraction_missing_corner_is_not_silently_simplified(self):
        mask = np.zeros((1020, 1020), np.uint8)
        cv2.fillConvexPoly(mask, np.array([[40,10],[1009,10],[1009,1009],[10,1009],[10,40]]), 1)
        self.assertIsNone(extract_quad(mask.astype(bool))[0])

    def test_polygon_uses_coco_pixel_centers_not_rounded_inclusive_edges(self):
        mask = decode_segmentation([[1,1,4,1,4,4,1,4]], 6, 6)
        self.assertEqual(int(mask.sum()), 9)
        self.assertTrue(mask[1:4, 1:4].all())
        self.assertFalse(mask[4, 4])

    def test_matching_tie_retains_high_confidence_best_overlap(self):
        a = np.zeros((12, 15), bool)
        b = np.zeros_like(a)
        a[1:11, 1:11] = True
        b[1:11, 4:14] = True
        truth = [Instance('cake', 1., a), Instance('banana', 1., b)]
        predicted = [Instance('cake', .9, a), Instance('banana', .1, a)]
        result = evaluate_frames([(truth, predicted, 1., 15, 12)])
        self.assertEqual(result['classification']['correct'], 2)

    def test_duplicate_wrong_class_and_missing_corners_reduce_scores(self):
        mask = np.zeros((20, 20), bool)
        mask[2:17, 2:17] = True
        truth = Instance('cake', 1., mask, [[2, 2], [16, 2], [16, 16], [2, 16]])
        pred = Instance('banana', .9, mask)
        result = evaluate_frames([([truth], [pred, pred], 20., 20, 20)])
        self.assertEqual(result['counts']['tp'], 1)
        self.assertEqual(result['counts']['fp'], 1)
        self.assertEqual(result['classification']['correct'], 0)
        self.assertEqual(result['corners']['success_recall'], 0.)
        self.assertAlmostEqual(result['geometry']['precision'], .5)
        self.assertIsNone(result['corners']['error_px_p95'])

    def test_negative_frame_false_positive_counts(self):
        mask = np.ones((5, 5), bool)
        result = evaluate_frames([([], [Instance('cake', .9, mask)], 1., 5, 5)])
        self.assertEqual(result['counts']['fp'], 1)
        self.assertEqual(result['negative_frames']['false_positive_frames'], 1)

    def test_empty_evaluation_is_not_a_perfect_score(self):
        with self.assertRaises(ValueError):
            evaluate_frames([])

    def test_dataset_rejects_same_image_content_across_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imencode('.png', np.zeros((10, 10, 3), np.uint8))[1].tofile(str(root / 'a.png'))
            splits = {}
            for split in ('train', 'val'):
                coco = {'images': [{'id': 1, 'file_name': 'a.png', 'width': 10, 'height': 10,
                                    'scene_id': split}], 'annotations': [],
                        'categories': [{'id': i + 1, 'name': name} for i, name in enumerate(
                            ('cake', 'banana', 'doll', 'clock'))]}
                (root / (split + '.json')).write_text(json.dumps(coco), encoding='utf-8')
                splits[split] = {'annotations': split + '.json', 'image_root': '.'}
            manifest = root / 'dataset.json'
            manifest.write_text(json.dumps({'schema_version': 1, 'splits': splits}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'content|duplicate|leak'):
                load_dataset(manifest, 'val')


if __name__ == '__main__':
    unittest.main()
