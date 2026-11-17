"""Local MMDetection 3.x RTMDet-Ins-tiny training and inference.

The official resolved tiny config is supplied from a local MMDetection checkout.
Importing this module does not import PyTorch, MMEngine, MMCV or MMDetection.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

from ..contracts import CLASSES


def _local_file(value, name):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError(name + ' must name an existing local file')
    if '://' in str(value):
        raise ValueError(name + ' must be local; remote downloads are disabled')
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(name + ' does not exist: ' + str(path))
    return str(path)


def _settings(config):
    cfg = dict(config)
    if cfg.get('variant', 'rtmdet_ins') != 'rtmdet_ins':
        raise ValueError('this backend supports variant=rtmdet_ins (tiny) only')
    size = cfg.get('image_size', 640)
    if isinstance(size, (list, tuple)) and len(size) == 2 and size[0] == size[1]:
        size = size[0]
    if type(size) is not int or size < 32 or size % 32:
        raise ValueError('image_size must be a square integer size divisible by 32')
    cfg['image_size'] = size
    for key, default in [('epochs', 100), ('batch', 2), ('max_detections', 100)]:
        value = cfg.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(key + ' must be a positive integer')
        cfg[key] = value
    seed = cfg.get('seed', 0)
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    cfg['seed'] = seed
    for key, default in [('confidence', .25), ('mask_threshold', .5)]:
        value = float(cfg.get(key, default))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(key + ' must be finite and in [0,1]')
        cfg[key] = value
    device = str(cfg.get('device', 'cpu'))
    # Plain GPU numbers are accepted consistently with the benchmark CLI.
    if device.isdigit():
        device = 'cuda:' + device
    if device == 'cuda':
        device = 'cuda:0'
    if device != 'cpu' and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise ValueError('device must be cpu, cuda:N, or a GPU index')
    cfg['device'] = device
    return cfg


def _without_pretrained(node):
    """Remove inherited implicit weight loading, keeping ordinary initializers."""
    if isinstance(node, dict):
        if 'pretrained' in node:
            node['pretrained'] = None
        if 'init_cfg' in node:
            init = node['init_cfg']
            if isinstance(init, dict) and (str(init.get('type', '')).split('.')[-1] == 'Pretrained'
                                           or 'checkpoint' in init):
                node['init_cfg'] = None
            elif isinstance(init, (list, tuple)):
                node['init_cfg'] = [item for item in init
                                    if not isinstance(item, dict) or
                                    (str(item.get('type', '')).split('.')[-1] != 'Pretrained'
                                     and 'checkpoint' not in item)] or None
        for value in node.values():
            _without_pretrained(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _without_pretrained(value)


def _prepare_model(base, options):
    import numpy as np
    model = deepcopy(base.get('model', {}))
    backbone = model.get('backbone', {})
    head = model.get('bbox_head', {})
    if (str(model.get('type', '')).split('.')[-1] != 'RTMDet'
            or str(head.get('type', '')).split('.')[-1] != 'RTMDetInsSepBNHead'
            or not math.isclose(float(backbone.get('deepen_factor', -1)), .167)
            or not math.isclose(float(backbone.get('widen_factor', -1)), .375)):
        raise ValueError('config_file must resolve to official RTMDet-Ins-tiny architecture')
    _without_pretrained(model)
    head['num_classes'] = len(CLASSES)
    # The official fp32 head uses strict > internally. Its predecessor cutoff
    # makes that comparison include pixels exactly at the requested threshold.
    cutoff = float(np.nextafter(np.float32(options['mask_threshold']), np.float32(-np.inf)))
    model.setdefault('test_cfg', {}).update(
        score_thr=options['confidence'], mask_thr_binary=cutoff,
        max_per_img=options['max_detections'])
    if options['device'] == 'cpu':
        def ordinary_bn(node):
            if isinstance(node, dict):
                if node.get('type') == 'SyncBN':
                    node['type'] = 'BN'
                for value in node.values():
                    ordinary_bn(value)
            elif isinstance(node, (list, tuple)):
                for value in node:
                    ordinary_bn(value)
        ordinary_bn(model)
    return model


def _adapt_pipeline(pipeline, size, training):
    result = []
    for source in pipeline:
        transform = deepcopy(source)
        kind = str(transform.get('type', '')).split('.')[-1]
        # MMDet 3.3 CachedMixUp flips an unpadded mask against a padded image.
        # Dropping it avoids misaligned visible masks on non-square source images.
        if kind == 'CachedMixUp' or (not training and kind == 'LoadAnnotations'):
            continue
        if kind == 'LoadAnnotations':
            transform.update(with_bbox=True, with_mask=True, poly2mask=True)
        if kind in ('CachedMosaic', 'Mosaic'):
            transform['img_scale'] = (size, size)
        if kind in ('Resize', 'RandomResize'):
            old = transform.get('scale', (640, 640))
            multiple = 2 if training and max(old) > 640 else 1
            transform['scale'] = (size * multiple, size * multiple)
        if kind == 'RandomCrop':
            transform.update(crop_size=(size, size), recompute_bbox=True,
                             allow_negative_crop=True)
        if kind == 'Pad' and 'size' in transform:
            transform['size'] = (size, size)
        if 'transforms' in transform:
            transform['transforms'] = _adapt_pipeline(transform['transforms'], size, training)
        result.append(transform)
    return result


def build_training_config(base, train_annotations, val_annotations, output_dir, config):
    """Pure dict -> dict conversion, independently testable without MM libraries.

    Inputs are a fully resolved official tiny config and exported local COCO files.
    Only train/val datasets are configured; the Runner's test loop is disabled.
    """
    options = _settings(config)
    cfg = deepcopy(dict(base))
    cfg['model'] = _prepare_model(base, options)
    size, epochs = options['image_size'], options['epochs']
    first = base.get('train_pipeline', base['train_dataloader']['dataset'].get('pipeline', []))
    second = base.get('train_pipeline_stage2', [])
    if not first or not second:
        raise ValueError('official tiny config must contain both training pipelines')
    cfg['train_pipeline'] = _adapt_pipeline(first, size, True)
    cfg['train_pipeline_stage2'] = _adapt_pipeline(second, size, True)
    # Explicit inference/validation pipeline, with original-shape metadata retained.
    cfg['test_pipeline'] = _adapt_pipeline(base.get('test_pipeline', []), size, False)
    if not cfg['test_pipeline']:
        raise ValueError('official tiny config must provide a test/validation pipeline')
    for split, annotation, pipeline in [
            ('train', train_annotations, cfg['train_pipeline']),
            ('val', val_annotations, cfg['test_pipeline'])]:
        cfg[split + '_dataloader'] = dict(
            batch_size=options['batch'] if split == 'train' else 1,
            num_workers=0, persistent_workers=False, pin_memory=False,
            sampler=dict(type='DefaultSampler', shuffle=split == 'train'),
            batch_sampler=None,
            dataset=dict(type='CocoDataset', data_root='',
                         ann_file=str(Path(annotation).resolve()), data_prefix=dict(img=''),
                         metainfo=dict(classes=CLASSES), pipeline=deepcopy(pipeline),
                         filter_cfg=dict(filter_empty_gt=False, min_size=1),
                         test_mode=split != 'train', backend_args=None))
    cfg['val_evaluator'] = dict(type='CocoMetric',
                                ann_file=str(Path(val_annotations).resolve()),
                                metric=['bbox', 'segm'], classwise=True,
                                format_only=False, backend_args=None)
    cfg['val_cfg'] = dict(type='ValLoop')
    cfg['test_dataloader'] = cfg['test_evaluator'] = cfg['test_cfg'] = None
    cfg['train_cfg'] = dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=1)
    cfg['max_epochs'] = epochs
    stage2 = min(20, max(1, epochs // 15))
    cfg['stage2_num_epochs'] = stage2
    for hook in cfg.get('custom_hooks', []):
        if str(hook.get('type', '')).split('.')[-1] == 'PipelineSwitchHook':
            hook.update(switch_epoch=max(0, epochs - stage2),
                        switch_pipeline=deepcopy(cfg['train_pipeline_stage2']))
    cfg['param_scheduler'] = []
    warmup_epochs = min(5, epochs // 10)
    if warmup_epochs:
        cfg['param_scheduler'].append(dict(type='LinearLR', start_factor=1.e-5,
                                          by_epoch=True, begin=0, end=warmup_epochs,
                                          convert_to_iter_based=True))
    cosine_begin = epochs // 2
    cfg['param_scheduler'].append(dict(type='CosineAnnealingLR', begin=cosine_begin,
                                      end=epochs, T_max=epochs - cosine_begin,
                                      by_epoch=True, convert_to_iter_based=True,
                                      eta_min=cfg['optim_wrapper']['optimizer']['lr'] * .05))
    cfg.setdefault('default_hooks', {}).setdefault('checkpoint', {}).update(
        type='CheckpointHook', interval=1, save_last=True, max_keep_ckpts=3,
        save_best='coco/segm_mAP', rule='greater', by_epoch=True)
    # Do not auto-resume or inherit the official ImageNet URL.
    init = options.get('init_checkpoint')
    cfg['load_from'] = _local_file(init, 'init_checkpoint') if init else None
    cfg['resume'] = False
    cfg.pop('checkpoint', None)
    cfg['auto_scale_lr'] = dict(enable=False)
    cfg['work_dir'] = str(Path(output_dir).resolve())
    cfg['default_scope'] = 'mmdet'
    cfg['launcher'] = 'none'
    cfg['randomness'] = dict(seed=options['seed'], deterministic=True)
    cfg.setdefault('env_cfg', {}).update(cudnn_benchmark=False)
    return cfg


def validate_class_metadata(metadata):
    """Reject COCO fallback metadata, missing labels, and reordered classes."""
    classes = metadata.get('classes') if isinstance(metadata, dict) else None
    if not isinstance(classes, (list, tuple)) or tuple(classes) != CLASSES:
        raise ValueError('checkpoint dataset_meta classes must be cake, banana, doll, clock in order')


def _numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        value = value.numpy()
    import numpy as np
    return np.asarray(value)


def convert_predictions(result, image_shape, confidence=.25, mask_threshold=.5):
    """Translate MMDet's zero-based labels and rescaled masks to public Instances."""
    import numpy as np
    from ..contracts import Instance
    prediction = result.pred_instances
    scores, labels = _numpy(prediction.scores), _numpy(prediction.labels)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError('inconsistent prediction scores and labels')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('prediction score must be finite in [0,1]')
    if labels.dtype.kind not in 'iu' or np.any((labels < 0) | (labels >= len(CLASSES))):
        raise ValueError('prediction label is outside the four-class mapping')
    if not len(scores):
        return []
    if not hasattr(prediction, 'masks'):
        raise ValueError('RTMDet-Ins prediction is missing instance masks')
    masks = _numpy(prediction.masks)
    if masks.shape != (len(scores), *tuple(image_shape)):
        raise ValueError('instance masks must have original image shape; resized masks are rejected')
    if masks.dtype != np.bool_ and not np.isfinite(masks).all():
        raise ValueError('prediction masks contain non-finite values')
    instances = []
    for score, label, mask in zip(scores, labels, masks):
        if score < confidence:
            continue
        binary = mask.copy() if mask.dtype == np.bool_ else mask >= mask_threshold
        if binary.any():
            instances.append(Instance(CLASSES[int(label)], float(score), binary.astype(bool),
                                      details=dict(backend='rtmdet_ins', label=int(label))))
    return instances


def export_coco(dataset, output_path):
    """Write exact visible pixels as uncompressed COCO RLE, including negative frames."""
    import numpy as np
    from ..data import encode_mask
    if tuple(dataset.classes) != CLASSES:
        raise ValueError('dataset classes do not match four-class mapping')
    if not dataset.records:
        raise ValueError('training/validation split cannot be empty')
    coco = dict(info=dict(description='Original visible pixel masks', version='1'),
                licenses=[], images=[], annotations=[], categories=[dict(id=i + 1, name=name)
                                                     for i, name in enumerate(CLASSES)])
    for image_id, record in enumerate(dataset.records, 1):
        coco['images'].append(dict(id=image_id, file_name=str(Path(record.image_path).resolve()),
                                   width=record.width, height=record.height,
                                   scene_id=record.scene_id, source_image_id=record.image_id))
        for instance in record.instances:
            mask = np.asarray(instance.mask)
            if mask.dtype != np.bool_ or mask.shape != (record.height, record.width) or not mask.any():
                raise ValueError('ground truth must have a nonempty original-image bool mask')
            if instance.category not in CLASSES:
                raise ValueError('unknown ground-truth class')
            ys, xs = np.nonzero(mask)
            coco['annotations'].append(dict(
                id=len(coco['annotations']) + 1, image_id=image_id,
                category_id=CLASSES.index(instance.category) + 1, iscrowd=0,
                segmentation=encode_mask(mask), area=int(mask.sum()),
                bbox=[int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                      int(ys.max() - ys.min() + 1)]))
    Path(output_path).write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding='utf-8')
    return coco


def _load_stack(training=False):
    try:
        import torch
        from mmengine.config import Config
        from mmdet.apis import init_detector, inference_detector
        from mmdet.utils import register_all_modules
        from mmengine.runner import Runner
    except (ImportError, OSError) as error:
        raise RuntimeError('RTMDet requires a compatible local PyTorch/MMDetection 3.x stack '
                           '(MMDetection 3.3: mmcv>=2.0,<2.2 with compiled ops, '
                           'mmengine>=0.7.1,<1.0, pycocotools). '
                           'Install it in a supported Python environment; no weights are downloaded.') from error
    register_all_modules(init_default_scope=True)
    return SimpleNamespace(torch=torch, Config=Config, Runner=Runner,
                           init_detector=init_detector, inference_detector=inference_detector)


def _select_device(stack, device):
    if device.startswith('cuda:'):
        if not stack.torch.cuda.is_available():
            raise RuntimeError('requested CUDA device is unavailable')
        stack.torch.cuda.set_device(int(device[5:]))


class RTMDetBackend:
    def __init__(self, config):
        self.options = _settings(config)
        config_path = _local_file(config.get('config_file'), 'config_file')
        checkpoint = _local_file(config.get('checkpoint'), 'checkpoint')
        self.stack = _load_stack()
        _select_device(self.stack, self.options['device'])
        base = self.stack.Config.fromfile(config_path).to_dict()
        cfg = deepcopy(base)
        cfg['model'] = _prepare_model(base, self.options)
        pipeline = _adapt_pipeline(base.get('test_pipeline', []), self.options['image_size'], False)
        if not pipeline:
            raise ValueError('config_file does not provide an inference pipeline')
        cfg['test_dataloader'] = dict(dataset=dict(type='CocoDataset',
                                                  metainfo=dict(classes=CLASSES), pipeline=pipeline))
        cfg['load_from'] = None
        cfg['resume'] = False
        cfg.pop('checkpoint', None)
        self.model = self.stack.init_detector(self.stack.Config(cfg), checkpoint,
                                             device=self.options['device'], palette='random')
        validate_class_metadata(getattr(self.model, 'dataset_meta', None))

    def predict(self, bgr):
        import numpy as np
        if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3
                or bgr.shape[2] != 3 or not bgr.shape[0] or not bgr.shape[1]):
            raise ValueError('predict requires a nonempty original uint8 BGR image')
        result = self.stack.inference_detector(self.model, bgr)
        return convert_predictions(result, bgr.shape[:2], self.options['confidence'],
                                   self.options['mask_threshold'])

    def synchronize(self):
        if self.options['device'].startswith('cuda:'):
            self.stack.torch.cuda.synchronize(self.options['device'])


def create(config):
    return RTMDetBackend(config)


def train(manifest_path, output_dir, config):
    """Train with the real MMEngine Runner; export train/val only, never test."""
    options = _settings(config)
    config_path = _local_file(config.get('config_file'), 'config_file')
    if options.get('init_checkpoint'):
        _local_file(options['init_checkpoint'], 'init_checkpoint')
    # All dependency errors happen before producing a misleading training directory.
    stack = _load_stack(training=True)
    _select_device(stack, options['device'])
    from ..data import load_dataset
    train_data = load_dataset(manifest_path, 'train')
    val_data = load_dataset(manifest_path, 'val')
    if not train_data.records or not val_data.records:
        raise ValueError('RTMDet training requires nonempty train and val splits')
    if not any(record.instances for record in train_data.records):
        raise ValueError('RTMDet training requires at least one annotated training instance')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    train_annotations, val_annotations = output / 'train.coco.json', output / 'val.coco.json'
    export_coco(train_data, train_annotations)
    export_coco(val_data, val_annotations)
    base = stack.Config.fromfile(config_path).to_dict()
    cfg = stack.Config(build_training_config(base, train_annotations, val_annotations, output, options))
    resolved_path = output / 'rtmdet_four_classes.py'
    cfg.dump(str(resolved_path))
    runner = stack.Runner.from_cfg(cfg)
    # Runner auto-selects an available accelerator; enforce the explicit user device
    # before it constructs the optimizer or begins any training iteration.
    runner.model.to(options['device'])
    runner.train()
    latest = output / 'last_checkpoint'
    if not latest.is_file():
        raise RuntimeError('MMEngine returned without a saved last_checkpoint')
    checkpoint = Path(latest.read_text(encoding='utf-8').strip())
    if not checkpoint.is_absolute():
        checkpoint = output / checkpoint
    if not checkpoint.is_file():
        raise RuntimeError('MMEngine last_checkpoint does not point to an existing checkpoint')
    runtime_info = getattr(getattr(runner, 'message_hub', None), 'runtime_info', {})
    best = runtime_info.get('best_ckpt')
    if not best:
        raise RuntimeError('training produced no validation-selected best checkpoint; '
                           'inspect the validation data and MMEngine logs')
    selected = Path(best)
    if not selected.is_absolute():
        selected = output / selected
    if not selected.is_file():
        raise RuntimeError('validation-selected checkpoint does not exist')
    return dict(backend='rtmdet_ins', checkpoint=str(selected.resolve()),
                last_checkpoint=str(checkpoint.resolve()), selection_metric='coco/segm_mAP',
                config_file=str(resolved_path), output_dir=str(output),
                train_annotations=str(train_annotations), val_annotations=str(val_annotations),
                classes=list(CLASSES), epochs=options['epochs'], device=options['device'],
                status='trained')
