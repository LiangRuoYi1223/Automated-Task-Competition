"""Boundary and export tests; these do not claim neural-network validation."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np


CLASSES = ('cake', 'banana', 'doll', 'clock')


class UltralyticsTests(unittest.TestCase):
    def setUp(self):
        try:
            self.backend = importlib.import_module('brick_benchmark.backends.ultralytics_backend')
        except ModuleNotFoundError as exc:
            self.fail(f'YOLO backend must be importable without torch/ultralytics: {exc}')

    def test_polygon_roundtrip_keeps_concavity_and_original_pixels(self):
        mask = np.zeros((10, 12), bool)
        mask[1:8, 2:5] = True
        mask[5:8, 5:10] = True
        polygon = self.backend.mask_to_polygon(mask)
        self.assertTrue(np.all((polygon >= 0) & (polygon <= 1)))
        restored = np.zeros(mask.shape, np.uint8)
        points = np.rint(polygon * [12, 10]).astype(np.int32)
        cv2.fillPoly(restored, [points], 1)
        np.testing.assert_array_equal(restored.astype(bool), mask)

    def test_export_rejects_holes_disconnected_and_degenerate_instances(self):
        hole = np.zeros((12, 12), bool)
        hole[1:11, 1:11] = True
        hole[4:8, 4:8] = False
        disconnected = np.zeros((12, 12), bool)
        disconnected[1:4, 1:4] = True
        disconnected[8:11, 8:11] = True
        line = np.zeros((12, 12), bool)
        line[3, 2:8] = True
        for mask, message in ((hole, 'hole'), (disconnected, 'component'),
                              (line, 'degenerate'), (np.zeros((3, 3), bool), 'empty')):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.backend.mask_to_polygon(mask)

    def make_manifest(self, directory, *, hole=False):
        root = Path(directory)
        splits = {}
        for split, color in (('train', 20), ('val', 80), ('test', 160)):
            image = root / f'{split} 中文图像.png'
            image.write_bytes(cv2.imencode('.png', np.full((12, 16, 3), color, np.uint8))[1].tobytes())
            # COCO uncompressed RLE, with explicit Fortran ordering.
            mask = np.zeros((12, 16), bool)
            mask[2:10, 3:13] = True
            if hole:
                mask[4:7, 5:8] = False
            counts, previous, count = [], False, 0
            for pixel in mask.reshape(-1, order='F'):
                if pixel == previous:
                    count += 1
                else:
                    counts.append(count)
                    count, previous = 1, bool(pixel)
            counts.append(count)
            coco = {'images': [{'id': 1, 'file_name': image.name, 'width': 16,
                                'height': 12, 'scene_id': split}],
                    'categories': [{'id': i + 1, 'name': c} for i, c in enumerate(CLASSES)],
                    'annotations': [{'id': 1, 'image_id': 1, 'category_id': 4,
                                     'segmentation': {'size': [12, 16], 'counts': counts},
                                     'iscrowd': 0, 'bbox': [3, 2, 10, 8], 'area': int(mask.sum())}]}
            annotation = root / f'{split}.json'
            annotation.write_text(json.dumps(coco), encoding='utf-8')
            splits[split] = {'annotations': annotation.name, 'image_root': '.'}
        manifest = root / 'dataset.json'
        manifest.write_text(json.dumps({'schema_version': 1, 'splits': splits}), encoding='utf-8')
        return manifest

    def test_export_writes_all_train_val_images_with_four_class_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(directory)
            output = Path(directory) / '导出 独占目录'
            result = self.backend.export_yolo_dataset(manifest, output)
            config = json.loads(Path(result['data']).read_text(encoding='utf-8'))
            self.assertEqual(config['names'], list(CLASSES))
            self.assertEqual(config['train'], 'images/train')
            self.assertEqual(config['val'], 'images/val')
            self.assertNotIn('test', config)
            for split, pixel in (('train', 20), ('val', 80)):
                images = list((output / 'images' / split).glob('*.png'))
                labels = list((output / 'labels' / split).glob('*.txt'))
                self.assertEqual(len(images), 1)
                self.assertEqual(len(labels), 1)
                self.assertEqual(labels[0].stem, images[0].stem)
                decoded = cv2.imdecode(np.frombuffer(images[0].read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(int(decoded[0, 0, 0]), pixel)
                fields = labels[0].read_text().split()
                self.assertEqual(fields[0], '3')
                points = np.array(fields[1:], float).reshape(-1, 2)
                restored = np.zeros((12, 16), np.uint8)
                cv2.fillPoly(restored, [np.rint(points * [16, 12]).astype(np.int32)], 1)
                self.assertEqual(int(restored.sum()), 80)
                self.assertEqual(restored[2, 3], 1)
                self.assertEqual(restored[9, 12], 1)
            self.assertFalse((output / 'images' / 'test').exists())
            self.assertEqual(result['counts'], {'train': 1, 'val': 1})

    def test_export_refuses_existing_output_and_hole_instead_of_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(directory, hole=True)
            output = Path(directory) / 'export'
            with self.assertRaisesRegex(ValueError, 'hole'):
                self.backend.export_yolo_dataset(manifest, output)
            existing = Path(directory) / 'existing'
            existing.mkdir()
            marker = existing / 'keep.txt'
            marker.write_text('untouched')
            with self.assertRaises(FileExistsError):
                self.backend.export_yolo_dataset(manifest, existing)
            self.assertEqual(marker.read_text(), 'untouched')

    def test_export_includes_later_records_and_negative_empty_label_files(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(directory)
            root = Path(directory)
            for split, color in (('train', 25), ('val', 85)):
                path = root / f'{split}_negative.png'
                path.write_bytes(cv2.imencode('.png', np.full((12, 16, 3), color, np.uint8))[1].tobytes())
                coco_path = root / f'{split}.json'
                coco = json.loads(coco_path.read_text(encoding='utf-8'))
                coco['images'].append({'id': 2, 'file_name': path.name, 'width': 16,
                                       'height': 12, 'scene_id': split + '_negative'})
                coco_path.write_text(json.dumps(coco), encoding='utf-8')
            output = root / 'export'
            result = self.backend.export_yolo_dataset(manifest, output)
            self.assertEqual(result['counts'], {'train': 2, 'val': 2})
            for split in ('train', 'val'):
                self.assertEqual(len(list((output / 'images' / split).glob('*.png'))), 2)
                labels = list((output / 'labels' / split).glob('*.txt'))
                self.assertEqual(len(labels), 2)
                self.assertEqual(sum(label.read_text() == '' for label in labels), 1)

    def model_fixture(self, family='11', names=CLASSES, task='segment', scale='n', result=None):
        return SimpleNamespace(task=task, names=dict(enumerate(names)),
                               model=SimpleNamespace(yaml={'yaml_file': f'yolo{family}n-seg.yaml', 'scale': scale}),
                               predict=lambda **kw: [result])

    def config(self, checkpoint, **kwargs):
        return {'variant': 'yolo11_seg', 'checkpoint': str(checkpoint), 'device': 'cpu',
                'image_size': 640, 'confidence': 0.25, 'mask_threshold': 0.5, **kwargs}

    def fake_yolo(self, model):
        return patch.dict(sys.modules, {'ultralytics': SimpleNamespace(YOLO=lambda path: model)})

    def test_predict_preserves_original_holes_and_components(self):
        masks = np.zeros((2, 12, 16), np.float32)
        masks[0, 1:10, 1:12] = 1
        masks[0, 4:7, 4:7] = 0
        masks[1, 1:3, 1:3] = 0.8
        masks[1, 8:10, 12:14] = 0.9
        result = SimpleNamespace(orig_shape=(12, 16), masks=SimpleNamespace(data=masks),
                                 boxes=SimpleNamespace(cls=np.array([0., 3.]), conf=np.array([.9, .8])))
        model = self.model_fixture(result=result)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'best.pt'
            checkpoint.write_bytes(b'local boundary fixture, not real weights')
            with self.fake_yolo(model):
                backend = self.backend.create(self.config(checkpoint))
                predictions = backend.predict(np.zeros((12, 16, 3), np.uint8))
            self.assertEqual([p.category for p in predictions], ['cake', 'clock'])
            self.assertEqual(predictions[0].mask.dtype, bool)
            np.testing.assert_array_equal(predictions[0].mask, masks[0] > .5)
            np.testing.assert_array_equal(predictions[1].mask, masks[1] > .5)
            self.assertIsNone(predictions[0].corners)

    def test_predict_requests_retina_masks_and_rejects_wrong_size(self):
        result = SimpleNamespace(orig_shape=(12, 16), masks=SimpleNamespace(data=np.ones((1, 6, 8))),
                                 boxes=SimpleNamespace(cls=np.array([0.]), conf=np.array([.8])))
        model = self.model_fixture(result=result)
        def predict(**kwargs):
            self.assertIs(kwargs['retina_masks'], True)
            self.assertEqual(kwargs['imgsz'], 640)
            self.assertEqual(kwargs['device'], 'cpu')
            self.assertEqual(kwargs.get('max_det'), 11)
            return [result]
        model.predict = predict
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'best.pt'
            checkpoint.write_bytes(b'x')
            with self.fake_yolo(model), self.assertRaisesRegex(ValueError, 'original|shape|size'):
                self.backend.create(self.config(checkpoint, max_detections=11)).predict(np.zeros((12, 16, 3), np.uint8))

    def test_model_rejects_wrong_family_scale_task_and_inference_names(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'best.pt'
            checkpoint.write_bytes(b'x')
            for model in (self.model_fixture(family='26'), self.model_fixture(scale='s'),
                          self.model_fixture(task='detect'), self.model_fixture(names=('person',))):
                with self.subTest(model=model), self.fake_yolo(model), self.assertRaises(ValueError):
                    self.backend.create(self.config(checkpoint))
            with self.fake_yolo(self.model_fixture(family='26')):
                self.backend.create(self.config(checkpoint, variant='yolo26_seg'))

    def test_missing_checkpoint_and_bad_configuration_fail_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'missing.pt'
            with self.assertRaises(FileNotFoundError):
                self.backend.create(self.config(checkpoint))
            checkpoint.write_bytes(b'x')
            for override in ({'variant': 'yolo8_seg'}, {'confidence': -1}, {'mask_threshold': 2},
                             {'image_size': 0}, {'epochs': 0}, {'batch': 0}, {'max_detections': 0}):
                with self.subTest(override=override), self.assertRaises(ValueError):
                    self.backend.create(self.config(checkpoint, **override))

    def test_mask_threshold_is_inclusive(self):
        result = SimpleNamespace(orig_shape=(12, 16), masks=SimpleNamespace(data=np.full((1, 12, 16), .5)),
                                 boxes=SimpleNamespace(cls=np.array([0.]), conf=np.array([.8])))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'best.pt'
            checkpoint.write_bytes(b'x')
            with self.fake_yolo(self.model_fixture(result=result)):
                backend = self.backend.create(self.config(checkpoint))
            predictions = backend.predict(np.zeros((12, 16, 3), np.uint8))
            self.assertEqual(int(predictions[0].mask.sum()), 192)

    def test_empty_result_and_low_confidence_filter_are_unambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'best.pt'
            checkpoint.write_bytes(b'x')
            result = SimpleNamespace(orig_shape=(12, 16), masks=None,
                                     boxes=SimpleNamespace(cls=np.array([]), conf=np.array([])))
            model = self.model_fixture(result=result)
            with self.fake_yolo(model):
                backend = self.backend.create(self.config(checkpoint))
            self.assertEqual(backend.predict(np.zeros((12, 16, 3), np.uint8)), [])
            result.masks = SimpleNamespace(data=np.ones((2, 12, 16)))
            result.boxes = SimpleNamespace(cls=np.array([1., 2.]), conf=np.array([.1, .25]))
            self.assertEqual([x.category for x in backend.predict(np.zeros((12, 16, 3), np.uint8))], ['doll'])

    def test_invalid_results_raise_instead_of_silently_losing_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'best.pt'
            checkpoint.write_bytes(b'x')
            for categories, scores, masks in (([4.], [.1], np.ones((1, 12, 16))),
                                              ([.5], [.9], np.ones((1, 12, 16))),
                                              ([0.], [np.nan], np.ones((1, 12, 16))),
                                              ([0.], [.9, .8], np.ones((1, 12, 16))),
                                              ([0.], [.9], None),
                                              ([0.], [.9], np.full((1, 12, 16), np.nan)),
                                              ([0.], [.9], np.full((1, 12, 16), 2.)),
                                              ([0.], [.9], np.zeros((1, 12, 16)))):
                result = SimpleNamespace(orig_shape=(12, 16),
                                         masks=None if masks is None else SimpleNamespace(data=masks),
                                         boxes=SimpleNamespace(cls=np.array(categories), conf=np.array(scores)))
                model = self.model_fixture(result=result)
                with self.subTest(categories=categories, scores=scores), self.fake_yolo(model):
                    backend = self.backend.create(self.config(checkpoint))
                with self.assertRaises(ValueError):
                    backend.predict(np.zeros((12, 16, 3), np.uint8))

    def test_cpu_needs_no_torch_and_cuda_synchronizes_requested_device(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'best.pt'
            checkpoint.write_bytes(b'x')
            with self.fake_yolo(self.model_fixture()):
                backend = self.backend.create(self.config(checkpoint))
            with patch.dict(sys.modules, {'torch': None}):
                backend.synchronize()
            devices = []
            torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda *, device: devices.append(device)))
            backend.config['device'] = '0'
            with patch.dict(sys.modules, {'torch': torch}):
                backend.synchronize()
            self.assertEqual(devices, ['cuda:0'])

    def test_unavailable_dependency_has_an_explicit_error_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'best.pt'
            checkpoint.write_bytes(b'x')
            with patch.dict(sys.modules, {'ultralytics': None}), self.assertRaisesRegex(RuntimeError, 'ultralytics.*PyTorch'):
                self.backend.create(self.config(checkpoint))

    def test_train_exports_train_val_and_calls_trainer_with_reproducible_options(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.make_manifest(directory)
            checkpoint = Path(directory) / 'yolo11n-seg.pt'
            checkpoint.write_bytes(b'x')
            model = self.model_fixture(names=('person', 'car'))
            def train(**kwargs):
                self.assertEqual(kwargs['epochs'], 3)
                self.assertEqual(kwargs['batch'], 2)
                self.assertEqual(kwargs['seed'], 17)
                self.assertEqual(kwargs['pretrained'], str(checkpoint.resolve()))
                self.assertIs(kwargs['amp'], False)
                self.assertEqual(kwargs['workers'], 0)
                self.assertIs(kwargs.get('single_cls'), False)
                self.assertIsNone(kwargs['classes'])
                self.assertEqual(kwargs['fraction'], 1.0)
                self.assertEqual(kwargs.get('max_det'), 100)
                request_path = output / 'training_request.json'
                self.assertTrue(request_path.is_file())
                request = json.loads(request_path.read_text(encoding='utf-8'))
                self.assertEqual(request['settings']['max_det'], 100)
                self.assertEqual(request['init_checkpoint'], str(checkpoint.resolve()))
                config = json.loads(Path(kwargs['data']).read_text(encoding='utf-8'))
                self.assertNotIn('test', config)
                save_dir = Path(kwargs['project']) / kwargs['name']
                (save_dir / 'weights').mkdir(parents=True)
                (save_dir / 'weights' / 'best.pt').write_bytes(b'trained fixture')
                model.trainer = SimpleNamespace(save_dir=save_dir)
                return SimpleNamespace(results_dict={'metrics/seg/mAP50-95': 0.4})
            model.train = train
            output = Path(directory) / 'training output'
            with self.fake_yolo(model):
                config = self.config(checkpoint, epochs=3, batch=2, seed=17)
                config['init_checkpoint'] = config.pop('checkpoint')
                result = self.backend.train(manifest, output, config)
            self.assertTrue(Path(result['checkpoint']).is_file())
            self.assertEqual(result['metrics']['metrics/seg/mAP50-95'], .4)
            self.assertEqual(result['variant'], 'yolo11_seg')


if __name__ == '__main__':
    unittest.main()
