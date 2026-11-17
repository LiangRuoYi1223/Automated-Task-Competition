import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import cv2
from brick_benchmark.runner import compare_reports, run_prediction, score_prediction


def fixture(root):
    image = np.zeros((80, 100, 3), np.uint8)
    cv2.rectangle(image, (20, 15), (70, 65), (255, 255, 255), -1)
    (root / 'a.png').write_bytes(cv2.imencode('.png', image)[1].tobytes())
    coco = {'images': [{'id': 1, 'file_name': 'a.png', 'width': 100, 'height': 80,
                        'scene_id': 'val_scene'}],
            'categories': [{'id': i + 1, 'name': n} for i, n in enumerate(('cake','banana','doll','clock'))],
            'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                'segmentation': [[20,15,70,15,70,65,20,65]],
                'corners': [[20,15],[70,15],[70,65],[20,65]]}]}
    (root / 'val.json').write_text(json.dumps(coco), encoding='utf-8')
    manifest = root / 'dataset.json'
    manifest.write_text(json.dumps({'schema_version': 1,
        'splits': {'val': {'annotations': 'val.json', 'image_root': '.'}}}), encoding='utf-8')
    return manifest


class RunnerTests(unittest.TestCase):
    def test_nontrained_pipeline_scoring_and_exclusive_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            output = root / 'predictions.json'
            config = {'variant': 'nontrained', 'method': 'contour', 'detection': {'min_area_px2': 100.}}
            result = run_prediction(manifest, 'val', config, output, warmup=0)
            self.assertEqual(result['variant'], 'nontrained')
            self.assertTrue(result['source_hashes'])
            report = score_prediction(manifest, output)
            self.assertEqual(report['metrics']['counts']['images'], 1)
            self.assertFalse(report['metrics']['selection_ready'])
            with self.assertRaises(FileExistsError):
                run_prediction(manifest, 'val', config, output, warmup=0)

    def test_scoring_rejects_missing_frames_and_changed_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            path = root / 'prediction.json'
            prediction = run_prediction(manifest, 'val', {'variant':'nontrained','method':'contour'}, path, 0)
            prediction['frames'] = []
            path.write_text(json.dumps(prediction), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'frame'):
                score_prediction(manifest, path)

    def test_nontrained_template_file_hashes_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = fixture(root)
            templates = root / 'templates'
            for index, name in enumerate(('cake','banana','doll','clock')):
                folder = templates / name
                folder.mkdir(parents=True)
                image = np.zeros((80, 80, 3), np.uint8)
                cv2.putText(image, str(index), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2., (255,255,255), 3)
                (folder / 'ref.png').write_bytes(cv2.imencode('.png', image)[1].tobytes())
            prediction = run_prediction(manifest, 'val', {'variant':'nontrained', 'method':'contour',
                'templates':str(templates)}, root / 'prediction.json', 0)
            self.assertEqual(len(prediction['artifact_manifests']['classification_templates']['files']), 4)
            self.assertEqual(len(prediction['artifact_manifest_sha256']), 64)

    def test_compare_rejects_mixed_hardware_and_duplicate_variant(self):
        base = {'variant':'one', 'dataset_fingerprint':'abc', 'split':'val',
                'hardware': {'host':'one'}, 'policy': {},
                'metrics': {'score': 25., 'geometry_score': 30., 'selection_ready':True}}
        other = dict(base, variant='two', hardware={'host':'two'})
        with self.assertRaisesRegex(ValueError, 'hardware'):
            compare_reports([base, other])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            compare_reports([base, base])


if __name__ == '__main__':
    unittest.main()
