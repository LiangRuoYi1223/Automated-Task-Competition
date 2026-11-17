"""Offline boundary tests; no torch/MMCV/MMDetection installation is needed."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from brick_benchmark.contracts import CLASSES, Dataset, Instance, Record
from brick_benchmark.data import decode_segmentation


def official_tiny_fixture():
    """Resolved essentials from MMDetection v3.3.0 tiny -> s -> l configs."""
    stage1 = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True, with_mask=True, poly2mask=False),
        dict(type='CachedMosaic', img_scale=(640, 640)),
        dict(type='RandomResize', scale=(1280, 1280), ratio_range=(.5, 2.)),
        dict(type='RandomCrop', crop_size=(640, 640)),
        dict(type='RandomFlip', prob=.5),
        dict(type='Pad', size=(640, 640)),
        dict(type='CachedMixUp', img_scale=(640, 640)),
        dict(type='PackDetInputs'),
    ]
    stage2 = [copy.deepcopy(stage1[i]) for i in (0, 1, 3, 4, 5, 6, 8)]
    stage2[2]['scale'] = (640, 640)
    dataset = dict(type='CocoDataset', ann_file='data/coco/old.json',
                   data_root='data/coco/', data_prefix=dict(img='train2017/'),
                   metainfo=dict(classes=('person',)), pipeline=stage1)
    return dict(
        model=dict(type='RTMDet',
                   backbone=dict(deepen_factor=.167, widen_factor=.375,
                                 init_cfg=dict(type='Pretrained', checkpoint='https://example/imagenet.pth')),
                   neck=dict(out_channels=96),
                   bbox_head=dict(type='RTMDetInsSepBNHead', num_classes=80,
                                  norm_cfg=dict(type='SyncBN')),
                   test_cfg=dict(mask_thr_binary=.5, score_thr=.05)),
        train_pipeline=stage1, train_pipeline_stage2=stage2,
        test_pipeline=[dict(type='LoadImageFromFile'), dict(type='Resize', scale=(640, 640)),
                       dict(type='Pad', size=(640, 640)), dict(type='LoadAnnotations', with_bbox=True),
                       dict(type='PackDetInputs')],
        train_dataloader=dict(batch_size=32, num_workers=10, persistent_workers=True,
                              dataset=copy.deepcopy(dataset)),
        val_dataloader=dict(batch_size=5, num_workers=10, dataset=copy.deepcopy(dataset)),
        test_dataloader=dict(batch_size=5, dataset=copy.deepcopy(dataset)),
        val_evaluator=dict(type='CocoMetric', ann_file='data/coco/old.json'),
        test_evaluator=dict(type='CocoMetric', ann_file='data/coco/test.json'),
        train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=300,
                       dynamic_intervals=[(280, 1)]),
        val_cfg=dict(type='ValLoop'), test_cfg=dict(type='TestLoop'),
        optim_wrapper=dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=.004)),
        custom_hooks=[dict(type='EMAHook'), dict(type='PipelineSwitchHook', switch_epoch=280,
                                             switch_pipeline=copy.deepcopy(stage2))],
        param_scheduler=[dict(type='CosineAnnealingLR', begin=150, end=300, T_max=150)],
        default_hooks=dict(checkpoint=dict(type='CheckpointHook', interval=10)),
        load_from='https://example/old.pth', resume=True,
    )


class RTMDetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.backend = importlib.import_module('brick_benchmark.backends.rtmdet_backend')
        except ModuleNotFoundError:
            cls.backend = None

    def setUp(self):
        self.assertIsNotNone(self.backend, 'RTMDet backend must exist and import without optional frameworks')

    def build(self, base=None, **options):
        return self.backend.build_training_config(
            base or official_tiny_fixture(), 'local_train.json', 'local_val.json', 'run',
            dict(variant='rtmdet_ins', epochs=8, batch=2, image_size=320, seed=11,
                 device='cpu', **options))

    def test_configuration_retargets_both_splits_and_disables_test(self):
        base = official_tiny_fixture()
        snapshot = copy.deepcopy(base)
        cfg = self.build(base)
        self.assertEqual(base, snapshot)
        for split, annotation in [('train', 'local_train.json'), ('val', 'local_val.json')]:
            ds = cfg[split + '_dataloader']['dataset']
            self.assertEqual(Path(ds['ann_file']).name, annotation)
            self.assertEqual(ds['metainfo']['classes'], CLASSES)
            self.assertEqual(ds['data_prefix'], {'img': ''})
            self.assertEqual(ds['data_root'], '')
        self.assertIsNone(cfg['test_dataloader'])
        self.assertIsNone(cfg['test_evaluator'])
        self.assertIsNone(cfg['test_cfg'])
        self.assertEqual(cfg['model']['bbox_head']['num_classes'], 4)
        self.assertIsNone(cfg['model']['backbone']['init_cfg'])
        self.assertIsNone(cfg['load_from'])
        self.assertFalse(cfg['resume'])
        self.assertEqual(cfg['train_dataloader']['batch_size'], 2)
        self.assertEqual(cfg['train_dataloader']['num_workers'], 0)
        self.assertFalse(cfg['train_dataloader']['persistent_workers'])
        self.assertEqual(cfg['randomness']['seed'], 11)
        self.assertEqual(cfg['train_cfg']['max_epochs'], 8)
        self.assertEqual(cfg['param_scheduler'][-1]['end'], 8)
        self.assertEqual(cfg['param_scheduler'][-1]['T_max'], 4)

    def test_rle_settings_reach_stage2_and_pipeline_switch_without_mixup(self):
        cfg = self.build()
        pipelines = [cfg['train_pipeline'], cfg['train_pipeline_stage2'],
                     cfg['train_dataloader']['dataset']['pipeline'],
                     cfg['custom_hooks'][1]['switch_pipeline']]
        for pipeline in pipelines:
            self.assertNotIn('CachedMixUp', [t['type'] for t in pipeline])
            annotation = next(t for t in pipeline if t['type'] == 'LoadAnnotations')
            self.assertTrue(annotation['poly2mask'])
            self.assertTrue(annotation['with_mask'])
            crop = next(t for t in pipeline if t['type'] == 'RandomCrop')
            self.assertTrue(crop['recompute_bbox'])
            self.assertTrue(crop['allow_negative_crop'])
            self.assertEqual(crop['crop_size'], (320, 320))
        self.assertEqual(cfg['train_pipeline'][3]['scale'], (640, 640))
        self.assertEqual(cfg['train_pipeline_stage2'][2]['scale'], (320, 320))
        self.assertLess(cfg['custom_hooks'][1]['switch_epoch'], 8)

    def test_local_initial_checkpoint_and_small_epoch_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            weight = Path(directory) / 'pretrained.pth'
            weight.write_bytes(b'local')
            cfg = self.backend.build_training_config(
                official_tiny_fixture(), 'train.json', 'val.json', directory,
                dict(init_checkpoint=str(weight), epochs=1, batch=1, device='cpu'))
            self.assertEqual(cfg['load_from'], str(weight.resolve()))
            self.assertEqual(cfg['param_scheduler'][-1]['end'], 1)
            self.assertGreater(cfg['param_scheduler'][-1]['T_max'], 0)
        with self.assertRaisesRegex(ValueError, 'local|remote'):
            self.build(init_checkpoint='https://example/weights.pth')

    def test_rejects_wrong_architecture_and_non_square_size(self):
        base = official_tiny_fixture()
        base['model']['backbone']['widen_factor'] = .5
        with self.assertRaisesRegex(ValueError, 'tiny'):
            self.build(base)
        with self.assertRaisesRegex(ValueError, 'square|image_size'):
            self.backend.build_training_config(official_tiny_fixture(), 't', 'v', 'r',
                                                dict(image_size=[320, 640]))

    def test_metadata_rejects_missing_or_permuted_class_order(self):
        for meta in ({}, {'classes': CLASSES[::-1]}, {'classes': ('person',)}):
            with self.assertRaisesRegex(ValueError, 'class|classes'):
                self.backend.validate_class_metadata(meta)
        self.backend.validate_class_metadata({'classes': list(CLASSES)})

    def test_detection_limit_is_applied_to_the_model(self):
        self.assertEqual(self.build()['model']['test_cfg']['max_per_img'], 100)
        self.assertEqual(self.build(max_detections=7)['model']['test_cfg']['max_per_img'], 7)

    def test_upstream_strict_mask_cutoff_includes_exact_threshold_pixels(self):
        cutoff = self.build()['model']['test_cfg']['mask_thr_binary']
        # MMDetection 3.3 returns bool masks using a strict > comparison.
        values = np.array([np.nextafter(np.float32(.5), np.float32(0)), .5, .6], np.float32)
        np.testing.assert_array_equal(values > cutoff, [False, True, True])

    def test_prediction_maps_labels_and_preserves_original_mask_topology(self):
        masks = np.zeros((3, 10, 12), bool)
        masks[0, 1:8, 1:8] = True
        masks[0, 3:5, 3:5] = False
        masks[0, 8:10, 10:12] = True
        masks[1, 2:4, 2:4] = True
        masks[2, 4:6, 4:6] = True
        result = SimpleNamespace(pred_instances=SimpleNamespace(
            scores=np.array([.9, .8, .1]), labels=np.array([0, 3, 2]), masks=masks))
        values = self.backend.convert_predictions(result, (10, 12), .25, .5)
        self.assertEqual([v.category for v in values], ['cake', 'clock'])
        np.testing.assert_array_equal(values[0].mask, masks[0])
        self.assertEqual(values[0].mask.dtype, np.bool_)
        self.assertIsNone(values[0].corners)

    def test_prediction_rejects_unknown_labels_and_resized_masks(self):
        result = SimpleNamespace(pred_instances=SimpleNamespace(
            scores=np.array([.9]), labels=np.array([4]), masks=np.ones((1, 5, 5), bool)))
        with self.assertRaisesRegex(ValueError, 'label'):
            self.backend.convert_predictions(result, (5, 5), .25, .5)
        result.pred_instances.labels = np.array([0])
        with self.assertRaisesRegex(ValueError, 'original|shape'):
            self.backend.convert_predictions(result, (10, 12), .25, .5)

    def test_coco_export_keeps_negative_frames_holes_and_disconnected_pixels(self):
        mask = np.zeros((10, 12), bool)
        mask[1:8, 1:8] = True
        mask[3:5, 3:5] = False
        mask[9, 11] = True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive = root / 'positive.png'
            negative = root / 'negative.png'
            positive.touch()
            negative.touch()
            dataset = Dataset(CLASSES, [
                Record('source-1', positive, 12, 10, 'train-1', [Instance('banana', 1., mask)]),
                Record('source-2', negative, 12, 10, 'train-2', []),
            ], root / 'manifest.json', 'hash', 'train')
            output = root / 'train.json'
            self.backend.export_coco(dataset, output)
            value = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(len(value['images']), 2)
            self.assertEqual([c['name'] for c in value['categories']], list(CLASSES))
            self.assertEqual(value['annotations'][0]['category_id'], 2)
            self.assertEqual(value['annotations'][0]['area'], 46)
            self.assertEqual(value['annotations'][0]['bbox'], [1, 1, 11, 9])
            np.testing.assert_array_equal(
                decode_segmentation(value['annotations'][0]['segmentation'], 10, 12), mask)

    def test_missing_local_files_fail_before_optional_dependency_import(self):
        with self.assertRaisesRegex((ValueError, FileNotFoundError), 'config_file|local|exist'):
            self.backend.create(dict(config_file='missing.py', checkpoint='missing.pth'))
        with self.assertRaisesRegex(ValueError, 'local|remote'):
            self.backend.create(dict(config_file='https://example/config.py', checkpoint='missing.pth'))

    def test_create_predict_and_cuda_synchronization_use_local_configuration(self):
        captured = {}

        class LocalConfig(dict):
            @classmethod
            def fromfile(cls, path):
                return cls(official_tiny_fixture())

            def to_dict(self):
                return dict(self)

        def initialize(cfg, checkpoint, device, palette):
            captured.update(cfg=cfg, checkpoint=checkpoint, device=device)
            return SimpleNamespace(dataset_meta={'classes': CLASSES})

        def inference(model, image):
            return SimpleNamespace(pred_instances=SimpleNamespace(
                scores=np.array([.8]), labels=np.array([2]),
                masks=np.ones((1, *image.shape[:2]), bool)))

        stack = SimpleNamespace(Config=LocalConfig, init_detector=initialize,
                                inference_detector=inference,
                                torch=SimpleNamespace(cuda=SimpleNamespace(
                                    is_available=lambda: True,
                                    set_device=lambda index: captured.update(index=index),
                                    synchronize=lambda device: captured.update(synchronized=device))))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file, checkpoint = root / 'tiny.py', root / 'four.pth'
            config_file.touch()
            checkpoint.touch()
            with patch.object(self.backend, '_load_stack', return_value=stack):
                backend = self.backend.create(dict(config_file=str(config_file),
                                                    checkpoint=str(checkpoint), device='1',
                                                    image_size=320, mask_threshold=.7))
                values = backend.predict(np.zeros((10, 12, 3), np.uint8))
                backend.synchronize()
            self.assertEqual(values[0].category, 'doll')
            self.assertEqual(values[0].mask.shape, (10, 12))
            self.assertEqual(captured['checkpoint'], str(checkpoint.resolve()))
            self.assertEqual(captured['device'], 'cuda:1')
            self.assertEqual(captured['synchronized'], 'cuda:1')
            self.assertEqual(captured['cfg']['model']['bbox_head']['num_classes'], 4)
            self.assertLess(captured['cfg']['model']['test_cfg']['mask_thr_binary'], .7)
            self.assertIsNone(captured['cfg']['model']['backbone']['init_cfg'])

    def test_train_exports_only_train_val_runs_runner_and_returns_validation_best(self):
        splits = []

        class LocalConfig(dict):
            @classmethod
            def fromfile(cls, path):
                return cls(official_tiny_fixture())

            def to_dict(self):
                return dict(self)

            def dump(self, path):
                Path(path).write_text(repr(dict(self)), encoding='utf-8')

        class RunnerBoundary:
            def __init__(self, cfg):
                self.cfg = cfg
                self.model = SimpleNamespace(to=lambda device: None)
                self.message_hub = SimpleNamespace(runtime_info={})

            @classmethod
            def from_cfg(cls, cfg):
                return cls(cfg)

            def train(self):
                root = Path(self.cfg['work_dir'])
                (root / 'epoch_2.pth').write_bytes(b'last')
                (root / 'best_coco_segm_mAP_epoch_1.pth').write_bytes(b'best')
                (root / 'last_checkpoint').write_text(str(root / 'epoch_2.pth'), encoding='utf-8')
                self.message_hub.runtime_info['best_ckpt'] = str(root / 'best_coco_segm_mAP_epoch_1.pth')
                (root / 'upstream_train_was_run').touch()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / 'tiny.py'
            config_file.touch()
            def load_split(path, split):
                splits.append(split)
                return Dataset(CLASSES, [Record(split, root / (split + '.png'), 6, 5, split,
                    [Instance('cake', 1., np.ones((5, 6), bool))])], root / 'manifest.json',
                    'fingerprint', split)
            stack = SimpleNamespace(Config=LocalConfig, Runner=RunnerBoundary)
            with patch.object(self.backend, '_load_stack', return_value=stack), \
                    patch('brick_benchmark.data.load_dataset', side_effect=load_split):
                result = self.backend.train(root / 'manifest.json', root / 'run',
                    dict(config_file=str(config_file), epochs=2, batch=1, device='cpu'))
            self.assertEqual(splits, ['train', 'val'])
            self.assertTrue((root / 'run/upstream_train_was_run').is_file())
            self.assertEqual(Path(result['checkpoint']).read_bytes(), b'best')
            self.assertEqual(Path(result['last_checkpoint']).read_bytes(), b'last')
            self.assertTrue(Path(result['config_file']).is_file())
            for split in ('train', 'val'):
                value = json.loads((root / 'run' / (split + '.coco.json')).read_text(encoding='utf-8'))
                self.assertEqual(value['images'][0]['scene_id'], split)
            self.assertFalse((root / 'run/test.coco.json').exists())

    def test_probability_masks_are_thresholded_and_empty_predictions_stay_empty(self):
        result = SimpleNamespace(pred_instances=SimpleNamespace(
            scores=np.array([.9]), labels=np.array([1]),
            masks=np.array([[[.1, .8], [.6, .4]]])))
        instances = self.backend.convert_predictions(result, (2, 2), .25, .7)
        np.testing.assert_array_equal(instances[0].mask, [[False, True], [False, False]])
        result.pred_instances = SimpleNamespace(scores=np.empty((0,)), labels=np.empty((0,), int))
        self.assertEqual(self.backend.convert_predictions(result, (2, 2)), [])


if __name__ == '__main__':
    unittest.main()
