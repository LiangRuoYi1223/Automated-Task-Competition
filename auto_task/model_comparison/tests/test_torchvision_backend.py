"""Dependency-free boundary tests; these do not claim a network training run."""
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

try:
    backend = importlib.import_module('brick_benchmark.backends.torchvision_backend')
except ModuleNotFoundError:
    backend = None


class TorchvisionBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(backend, 'Mask R-CNN backend has not been implemented')

    def record(self, instances):
        return SimpleNamespace(image_id='img-a', image_path=Path('a.png'), width=5,
                               height=4, scene_id='scene-a', instances=instances)

    def instance(self, category='cake', mask=None):
        if mask is None:
            mask = np.zeros((4, 5), bool)
            mask[1:3, 2:5] = True
        return SimpleNamespace(category=category, mask=mask)

    def prediction(self, labels=(1, 2, 3, 4), scores=(.9, .8, .7, .6)):
        masks = np.zeros((len(labels), 1, 4, 5), np.float32)
        masks[:, :, 1:3, 2:5] = .8
        label_array = np.asarray(labels) if len(labels) else np.empty(0, np.int64)
        return dict(labels=label_array, scores=np.asarray(scores), masks=masks)

    def test_bgr_becomes_contiguous_rgb_chw_float_without_double_normalization(self):
        bgr = np.array([[[0, 127, 255], [255, 0, 0]]], np.uint8)
        result = backend.bgr_to_chw(bgr)
        self.assertEqual(result.shape, (3, 1, 2))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)
        np.testing.assert_allclose(result[:, 0, :], [[1., 0.], [127 / 255, 0.], [0., 1.]])
        np.testing.assert_array_equal(bgr, [[[0, 127, 255], [255, 0, 0]]])

    def test_rejects_invalid_image_inputs(self):
        for image in (np.zeros((4, 5), np.uint8), np.zeros((4, 5, 4), np.uint8),
                      np.zeros((0, 5, 3), np.uint8), np.zeros((4, 5, 3), np.float32), None):
            with self.subTest(image=None if image is None else image.shape):
                with self.assertRaises((TypeError, ValueError)):
                    backend.bgr_to_chw(image)

    def test_target_uses_one_based_classes_and_pixel_outer_boundary_boxes(self):
        single = np.zeros((4, 5), bool)
        single[0, 0] = True
        target = backend.target_arrays(self.record([self.instance(), self.instance('clock', single)]), 7)
        np.testing.assert_array_equal(target['boxes'], [[2., 1., 5., 3.], [0., 0., 1., 1.]])
        np.testing.assert_array_equal(target['labels'], [1, 4])
        np.testing.assert_array_equal(target['area'], [6., 1.])
        np.testing.assert_array_equal(target['image_id'], [7])
        np.testing.assert_array_equal(target['iscrowd'], [0, 0])
        self.assertEqual(target['boxes'].dtype, np.float32)
        self.assertEqual(target['labels'].dtype, np.int64)
        self.assertEqual(target['masks'].dtype, np.uint8)
        self.assertEqual(target['masks'].shape, (2, 4, 5))

    def test_target_preserves_holes_and_disconnected_regions(self):
        mask = np.ones((4, 5), bool)
        mask[1:3, 1:4] = False
        mask[1, 2] = True
        target = backend.target_arrays(self.record([self.instance(mask=mask)]), 0)
        np.testing.assert_array_equal(target['masks'][0], mask.astype(np.uint8))

    def test_negative_record_has_correct_empty_target_shapes_and_dtypes(self):
        target = backend.target_arrays(self.record([]), 0)
        for name, shape, dtype in [('boxes', (0, 4), np.float32), ('labels', (0,), np.int64),
                                   ('masks', (0, 4, 5), np.uint8), ('area', (0,), np.float32),
                                   ('iscrowd', (0,), np.int64)]:
            with self.subTest(name=name):
                self.assertEqual(target[name].shape, shape)
                self.assertEqual(target[name].dtype, dtype)

    def test_target_rejects_unknown_class_empty_mask_and_wrong_dimensions(self):
        bad = [self.instance('person'), self.instance(mask=np.zeros((4, 5), bool)),
               self.instance(mask=np.ones((5, 4), bool)),
               self.instance(mask=np.ones((4, 5), np.float32))]
        for instance in bad:
            with self.subTest(instance=instance):
                with self.assertRaises((TypeError, ValueError)):
                    backend.target_arrays(self.record([instance]), 0)

    def test_predictions_map_all_four_labels_to_original_bool_masks(self):
        instances = backend.prediction_to_instances(self.prediction(), (4, 5), .7, .5)
        self.assertEqual([instance.category for instance in instances], ['cake', 'banana', 'doll'])
        self.assertAlmostEqual(instances[0].score, .9)
        self.assertEqual(instances[0].mask.dtype, bool)
        self.assertEqual(instances[0].mask.shape, (4, 5))
        self.assertEqual(int(instances[0].mask.sum()), 6)
        all_classes = backend.prediction_to_instances(self.prediction(), (4, 5), .5, .5)
        self.assertEqual(all_classes[-1].category, 'clock')

    def test_predictions_reject_out_of_range_labels_even_below_confidence(self):
        for label in (0, 5, 90, 1.5, float('nan')):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    backend.prediction_to_instances(self.prediction((label,), (.01,)), (4, 5), .5, .5)

    def test_predictions_reject_bad_shapes_nonfinite_values_and_probability_ranges(self):
        bad = [dict(labels=np.array([1]), scores=np.array([.9]), masks=np.ones((1, 1, 5, 4))),
               dict(labels=np.array([1, 2]), scores=np.array([.9]), masks=np.ones((1, 1, 4, 5))),
               self.prediction((1,), (float('nan'),)), self.prediction((1,), (1.1,))]
        masks = self.prediction((1,), (.9,))
        masks['masks'][0, 0, 0, 0] = float('nan')
        bad.append(masks)
        for output in bad:
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    backend.prediction_to_instances(output, (4, 5), .5, .5)

    def test_empty_prediction_is_valid_and_threshold_is_inclusive(self):
        self.assertEqual(backend.prediction_to_instances(self.prediction((), ()), (4, 5), .5, .5), [])
        output = self.prediction((4,), (.5,))
        output['masks'][0, 0, 0, 0] = .5
        instances = backend.prediction_to_instances(output, (4, 5), .5, .5)
        self.assertTrue(instances[0].mask[0, 0])

    def test_create_requires_local_checkpoint_before_loading_heavy_dependencies(self):
        for config in ({'variant': 'mask_rcnn'}, {'checkpoint': 'https://example.com/weights.pth'},
                       {'checkpoint': 'missing-local-checkpoint.pth'}):
            with self.subTest(config=config):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    backend.create(config)

    def test_config_rejects_invalid_variant_thresholds_and_training_parameters(self):
        bad = [{'variant': 'maskrcnn_resnet101'}, {'image_size': 0}, {'image_size': True},
               {'confidence': -1}, {'mask_threshold': 1.1}, {'epochs': 0}, {'batch': 0},
               {'seed': -1}, {'learning_rate': 0}, {'learning_rate': float('nan')},
               {'device': 'auto'}, {'max_detections': 0}, {'max_detections': True}]
        for config in bad:
            with self.subTest(config=config):
                with self.assertRaises((TypeError, ValueError)):
                    backend.validate_config(config, training=True)

    def test_config_records_random_initialization_and_defaults(self):
        config = backend.validate_config({'variant': 'mask_rcnn'}, training=True)
        self.assertEqual(config['initialization'], 'random')
        self.assertGreater(config['epochs'], 0)
        self.assertGreater(config['batch'], 0)

    def test_train_rejects_remote_or_missing_initial_weights(self):
        for path in ('https://example.com/model.pth', 'missing-init-weights.pth'):
            with self.subTest(path=path):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    backend.validate_config({'init_checkpoint': path}, training=True)

    def test_checkpoint_metadata_rejects_coco_or_reordered_classes_and_wrong_variant(self):
        valid = {'state_dict': {'some.weight': np.ones(1)},
                 'classes': ['cake', 'banana', 'doll', 'clock'],
                 'model_variant': 'maskrcnn_resnet50_fpn_v2'}
        self.assertIs(backend.validate_checkpoint(valid), valid['state_dict'])
        for key, value in [('classes', ['clock', 'doll', 'banana', 'cake']),
                           ('classes', ['person']), ('model_variant', 'maskrcnn_resnet50_fpn'),
                           ('state_dict', {})]:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ValueError):
                    backend.validate_checkpoint({**valid, key: value})
        with self.assertRaises(ValueError):
            backend.validate_checkpoint({'some.weight': np.ones(1)})

    def test_training_refuses_conflicting_initialization_sources(self):
        with tempfile.NamedTemporaryFile(suffix='.pth') as checkpoint:
            with self.assertRaises(ValueError):
                backend.validate_config({'checkpoint': checkpoint.name,
                                         'init_checkpoint': checkpoint.name}, training=True)

    def test_checkpoint_rejects_malformed_class_metadata_with_clear_value_error(self):
        for classes in (None, 4, 'cake,banana,doll,clock'):
            with self.subTest(classes=classes):
                with self.assertRaises(ValueError):
                    backend.validate_checkpoint({'state_dict': {'weight': np.ones(1)},
                                                 'classes': classes,
                                                 'model_variant': 'maskrcnn_resnet50_fpn_v2'})

    def test_model_builder_sets_explicit_detection_budget_and_five_class_heads_without_download(self):
        settings = {}
        model = SimpleNamespace(roi_heads=SimpleNamespace(
            box_predictor=SimpleNamespace(cls_score=SimpleNamespace(in_features=1024)),
            mask_predictor=SimpleNamespace(conv5_mask=SimpleNamespace(in_channels=256, out_channels=256))))

        def builder(**kwargs):
            settings.update(kwargs)
            return model

        def box_head(features, classes):
            return SimpleNamespace(features=features, classes=classes)

        def mask_head(features, hidden, classes):
            return SimpleNamespace(features=features, hidden=hidden, classes=classes)

        cfg = backend.validate_config({'max_detections': 37}, training=True)
        result = backend._build_model((None, None, builder, box_head, mask_head), cfg)
        self.assertEqual(settings.get('box_detections_per_img'), 37)
        self.assertIsNone(settings['weights'])
        self.assertIsNone(settings['weights_backbone'])
        self.assertEqual((settings['min_size'], settings['max_size']), (640, 640))
        self.assertEqual(result.roi_heads.box_predictor.classes, 5)
        self.assertEqual(result.roi_heads.box_predictor.features, 1024)
        self.assertEqual(result.roi_heads.mask_predictor.classes, 5)
        self.assertEqual(result.roi_heads.mask_predictor.features, 256)
        self.assertEqual(result.roi_heads.mask_predictor.hidden, 256)
        self.assertEqual(backend.validate_config({}, training=True)['max_detections'], 100)

    @unittest.skipIf(importlib.util.find_spec('torch'), 'checks the dependency-missing path')
    def test_existing_local_checkpoint_reports_missing_dependencies_without_fake_prediction(self):
        with tempfile.NamedTemporaryFile(suffix='.pt') as checkpoint:
            with self.assertRaisesRegex(RuntimeError, 'torch/torchvision'):
                backend.create({'checkpoint': checkpoint.name})


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is unavailable; tensor integration unverified')
class TorchTensorTests(unittest.TestCase):
    def test_negative_target_converts_to_valid_torch_tensors(self):
        import torch
        record = SimpleNamespace(width=5, height=4, instances=[])
        target = backend.target_tensors(record, 3, torch)
        self.assertEqual(tuple(target['boxes'].shape), (0, 4))
        self.assertEqual(tuple(target['masks'].shape), (0, 4, 5))
        self.assertEqual(target['labels'].dtype, torch.int64)


if __name__ == '__main__':
    unittest.main()
