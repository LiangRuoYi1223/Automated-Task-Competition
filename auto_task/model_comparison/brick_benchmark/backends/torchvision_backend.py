"""Local-only Torchvision Mask R-CNN ResNet50 FPN v2 training and inference.

Torch/Torchvision are imported only when create() or train() needs a network.
The helpers below keep image/annotation boundary checks independently testable.
"""
from collections.abc import Mapping
import hashlib
import math
from pathlib import Path
import random
import re
import json
import warnings

import numpy as np

from ..contracts import CLASSES, Instance

MODEL_VARIANT = 'maskrcnn_resnet50_fpn_v2'


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{name} must be finite in [0, 1]')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be finite in [0, 1]')
    return value


def _local_file(value, name):
    if not isinstance(value, (str, Path)) or not str(value).strip() or '://' in str(value):
        raise ValueError(f'{name} must name an explicit local file; downloads are disabled')
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{name} local file does not exist: {path}')
    return str(path)


def validate_config(config, training=False):
    if not isinstance(config, Mapping):
        raise TypeError('Mask R-CNN config must be a mapping')
    cfg = dict(config)
    if cfg.get('variant', 'mask_rcnn') != 'mask_rcnn':
        raise ValueError('torchvision backend requires variant=mask_rcnn (ResNet50 FPN v2)')
    cfg['variant'] = 'mask_rcnn'
    cfg['image_size'] = _integer(cfg.get('image_size', 640), 'image_size')
    cfg['max_detections'] = _integer(cfg.get('max_detections', 100), 'max_detections')
    cfg['confidence'] = _probability(cfg.get('confidence', .25), 'confidence')
    cfg['mask_threshold'] = _probability(cfg.get('mask_threshold', .5), 'mask_threshold')
    cfg['device'] = str(cfg.get('device', 'cpu'))
    if not re.fullmatch(r'cpu|cuda(?::[0-9]+)?|mps', cfg['device']):
        raise ValueError('device must be cpu, cuda, cuda:N or mps')
    if cfg.get('checkpoint') is not None and cfg.get('init_checkpoint') is not None:
        raise ValueError('checkpoint and init_checkpoint are mutually exclusive initialization sources')
    if cfg.get('checkpoint') is not None:
        cfg['checkpoint'] = _local_file(cfg['checkpoint'], 'checkpoint')
    if cfg.get('init_checkpoint') is not None:
        cfg['init_checkpoint'] = _local_file(cfg['init_checkpoint'], 'init_checkpoint')
    if not training:
        if not cfg.get('checkpoint'):
            raise ValueError('inference requires checkpoint: a local trained four-class .pth/.pt file')
        if cfg.get('init_checkpoint'):
            raise ValueError('init_checkpoint is for training, not a four-class inference checkpoint')
    else:
        cfg['epochs'] = _integer(cfg.get('epochs', 10), 'epochs')
        cfg['batch'] = _integer(cfg.get('batch', 2), 'batch')
        cfg['seed'] = _integer(cfg.get('seed', 0), 'seed', 0)
        if cfg['seed'] >= 2**32:
            raise ValueError('seed must be less than 2**32 for NumPy reproducibility')
        lr = cfg.get('learning_rate', 1e-4)
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError('learning_rate must be finite and positive')
        cfg['learning_rate'] = float(lr)
        cfg['initialization'] = ('trained_checkpoint' if cfg.get('checkpoint') else
                                 'local_official_coco' if cfg.get('init_checkpoint') else 'random')
    return cfg


def bgr_to_chw(bgr):
    """Convert original uint8 BGR once to contiguous RGB CHW float32 [0,1]."""
    if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8:
        raise TypeError('image must be a uint8 BGR NumPy array')
    if bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) <= 0:
        raise ValueError('image must have nonempty H x W x 3 dimensions')
    return np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.


def target_arrays(record, index):
    """Derive masks/outer pixel-edge boxes without filling holes or components."""
    height = _integer(record.height, 'record.height')
    width = _integer(record.width, 'record.width')
    index = _integer(index, 'dataset index', 0)
    count = len(record.instances)
    masks = np.zeros((count, height, width), np.uint8)
    boxes = np.zeros((count, 4), np.float32)
    labels = np.zeros(count, np.int64)
    for i, instance in enumerate(record.instances):
        if instance.category not in CLASSES:
            raise ValueError(f'unknown training category: {instance.category}')
        mask = instance.mask
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_:
            raise TypeError('training masks must be original-image bool arrays')
        if mask.shape != (height, width) or not mask.any():
            raise ValueError('training mask must be nonempty and match original image dimensions')
        ys, xs = np.nonzero(mask)
        boxes[i] = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        masks[i] = mask
        labels[i] = CLASSES.index(instance.category) + 1
    return {'boxes': boxes, 'labels': labels, 'masks': masks,
            'image_id': np.asarray([index], np.int64),
            'area': masks.sum(axis=(1, 2)).astype(np.float32),
            'iscrowd': np.zeros(count, np.int64)}


def target_tensors(record, index, torch):
    return {key: torch.from_numpy(value) for key, value in target_arrays(record, index).items()}


def prediction_to_instances(output, image_shape, confidence=.25, mask_threshold=.5):
    confidence = _probability(confidence, 'confidence')
    mask_threshold = _probability(mask_threshold, 'mask_threshold')
    if not isinstance(output, Mapping) or not {'labels', 'scores', 'masks'} <= output.keys():
        raise ValueError('Mask R-CNN must output labels, scores and masks')
    labels, scores, masks = (np.asarray(output[key]) for key in ('labels', 'scores', 'masks'))
    height, width = image_shape
    if labels.ndim != 1 or scores.shape != labels.shape or masks.shape != (len(labels), 1, height, width):
        raise ValueError('Mask R-CNN prediction dimensions must match original H x W image')
    if labels.dtype.kind not in 'iu' or ((labels < 1) | (labels > len(CLASSES))).any():
        raise ValueError('Mask R-CNN labels must be integer 1..4; COCO/background labels are invalid')
    if (not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any() or
            not np.isfinite(masks).all() or ((masks < 0) | (masks > 1)).any()):
        raise ValueError('Mask R-CNN scores and mask probabilities must be finite in [0,1]')
    instances = []
    for label, score, probabilities in zip(labels, scores, masks[:, 0]):
        if score < confidence:
            continue
        mask = np.ascontiguousarray(probabilities >= mask_threshold, dtype=bool)
        # A detection whose thresholded mask is empty cannot satisfy Instance.
        if not mask.any():
            continue
        instances.append(Instance(CLASSES[int(label) - 1], float(score), mask,
                                  details={'model_variant': MODEL_VARIANT}))
    return instances


def validate_checkpoint(payload):
    if not isinstance(payload, Mapping):
        raise ValueError('trained checkpoint must contain state_dict/classes/model_variant metadata')
    classes = payload.get('classes')
    if (not isinstance(classes, (list, tuple)) or tuple(classes) != CLASSES or
            payload.get('model_variant') != MODEL_VARIANT):
        raise ValueError('checkpoint classes/order or model_variant do not match the four-class FPN v2 model')
    state = payload.get('state_dict')
    if not isinstance(state, Mapping) or not state:
        raise ValueError('checkpoint state_dict is missing or empty')
    return state


def _runtime():
    try:
        import torch
        import torchvision
        from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError('Mask R-CNN requires compatible local torch/torchvision packages; '
                           'no packages or weights are downloaded automatically') from error
    return torch, torchvision, maskrcnn_resnet50_fpn_v2, FastRCNNPredictor, MaskRCNNPredictor


def _device(torch, name):
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA device requested but PyTorch CUDA is unavailable')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device does not exist: {name}')
    if device.type == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        raise RuntimeError('MPS device requested but unavailable')
    return device


def _load(torch, path):
    # Explicit weights_only avoids executable pickled modules. Never fall back.
    return torch.load(path, map_location='cpu', weights_only=True)


def _build_model(runtime, cfg, official_state=None):
    _, _, builder, box_predictor, mask_predictor = runtime
    model = builder(weights=None, weights_backbone=None, num_classes=91,
                    min_size=cfg['image_size'], max_size=cfg['image_size'],
                    box_score_thresh=cfg['confidence'],
                    box_detections_per_img=cfg['max_detections'])
    if official_state is not None:
        model.load_state_dict(official_state, strict=True)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = box_predictor(box_features, len(CLASSES) + 1)
    model.roi_heads.mask_predictor = mask_predictor(mask_features, hidden, len(CLASSES) + 1)
    return model


class TorchvisionBackend:
    def __init__(self, model, torch, device, config):
        self.model = model
        self.torch = torch
        self.device = device
        self.config = config

    def predict(self, bgr):
        image = self.torch.from_numpy(bgr_to_chw(bgr)).to(self.device)
        self.model.eval()
        with self.torch.inference_mode():
            outputs = self.model([image])
        if len(outputs) != 1:
            raise ValueError('Mask R-CNN must return exactly one prediction for one image')
        arrays = {key: outputs[0][key].detach().cpu().numpy() for key in ('labels', 'scores', 'masks')}
        return prediction_to_instances(arrays, bgr.shape[:2], self.config['confidence'],
                                       self.config['mask_threshold'])

    def synchronize(self):
        if self.device.type == 'cuda':
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == 'mps':
            self.torch.mps.synchronize()


def create(config):
    cfg = validate_config(config)
    runtime = _runtime()
    torch = runtime[0]
    device = _device(torch, cfg['device'])
    payload = _load(torch, cfg['checkpoint'])
    state = validate_checkpoint(payload)
    model = _build_model(runtime, cfg)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return TorchvisionBackend(model, torch, device, cfg)


class _TrainingDataset:
    def __init__(self, dataset, torch):
        self.dataset = dataset
        self.torch = torch

    def __len__(self):
        return len(self.dataset.records)

    def __getitem__(self, index):
        from ..data import read_image
        record = self.dataset.records[index]
        bgr = read_image(record.image_path)
        if bgr.shape[:2] != (record.height, record.width):
            raise ValueError(f'image dimensions changed: {record.image_path}')
        return self.torch.from_numpy(bgr_to_chw(bgr)), target_tensors(record, index, self.torch)


def _collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def _validation(backend, dataset):
    from ..data import read_image
    from ..metrics import evaluate_frames
    frames = []
    for record in dataset.records:
        bgr = read_image(record.image_path)
        if bgr.shape[:2] != (record.height, record.width):
            raise ValueError(f'image dimensions changed: {record.image_path}')
        predictions = backend.predict(bgr)
        # Validation selection uses class F1, never unmeasured latency or corners.
        frames.append((record.instances, predictions, 0., record.width, record.height))
    report = evaluate_frames(frames)
    report.pop('latency_ms', None)
    report['latency_measured'] = False
    return report


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def train(manifest_path, output_dir, config):
    """Run real optimization on train; select best by separate val class F1.

    checkpoint warm-starts a trained four-class network. init_checkpoint accepts
    the local official 91-output COCO state dict, before replacing both heads.
    With neither source, initialization is explicitly random.
    """
    from ..data import load_dataset
    cfg = validate_config(config, training=True)
    training = load_dataset(manifest_path, 'train')
    validation = load_dataset(manifest_path, 'val')
    for split, dataset in (('train', training), ('val', validation)):
        if tuple(dataset.classes) != CLASSES or not dataset.records:
            raise ValueError(f'{split} must be nonempty with classes cake/banana/doll/clock')
    runtime = _runtime()
    torch, torchvision = runtime[:2]
    device = _device(torch, cfg['device'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    official = None
    if cfg.get('init_checkpoint'):
        official = _load(torch, cfg['init_checkpoint'])
        if not isinstance(official, Mapping) or not official:
            raise ValueError('init_checkpoint must contain the official COCO model state_dict')
    elif cfg['initialization'] == 'random':
        warnings.warn('Mask R-CNN training starts from random weights; no pretrained weights were supplied',
                      RuntimeWarning, stacklevel=2)
    model = _build_model(runtime, cfg, official)
    if cfg.get('checkpoint'):
        model.load_state_dict(validate_checkpoint(_load(torch, cfg['checkpoint'])), strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=cfg['learning_rate'], weight_decay=1e-4)
    generator = torch.Generator().manual_seed(cfg['seed'])
    loader = torch.utils.data.DataLoader(_TrainingDataset(training, torch), batch_size=cfg['batch'],
                                         shuffle=True, num_workers=0, collate_fn=_collate,
                                         generator=generator)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    backend = TorchvisionBackend(model, torch, device, cfg)
    history, best_f1, best_epoch = [], -1., None
    initialization_path = cfg.get('checkpoint') or cfg.get('init_checkpoint')
    metadata = {'model_variant': MODEL_VARIANT, 'classes': list(CLASSES), 'config': cfg,
                'image_size': cfg['image_size'], 'seed': cfg['seed'],
                'initialization': cfg['initialization'],
                'initialization_sha256': _sha256(initialization_path) if initialization_path else None,
                'manifest_path': str(training.manifest_path),
                'manifest_sha256': _sha256(training.manifest_path),
                'train_fingerprint': getattr(training, 'fingerprint', None),
                'val_fingerprint': getattr(validation, 'fingerprint', None),
                'torch_version': str(torch.__version__), 'torchvision_version': str(torchvision.__version__),
                'selection_metric': 'val.classification.f1',
                'reproducibility': 'seeded; exact cross-device determinism is not guaranteed'}
    (output / 'training_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        totals, images_seen = {}, 0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
            losses = model(images, targets)
            if not isinstance(losses, Mapping) or not losses:
                raise RuntimeError('Mask R-CNN training returned no loss dictionary')
            loss = sum(losses.values())
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f'nonfinite training loss at epoch {epoch}')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            count = len(images)
            images_seen += count
            for key, value in losses.items():
                scalar = float(value.detach().cpu().item())
                if not math.isfinite(scalar):
                    raise RuntimeError(f'nonfinite {key} at epoch {epoch}')
                totals[key] = totals.get(key, 0.) + scalar * count
        val_report = _validation(backend, validation)
        row = {'epoch': epoch, 'train_images': images_seen,
               'train_losses': {key: value / images_seen for key, value in totals.items()},
               'train_loss': sum(totals.values()) / images_seen, 'val': val_report}
        history.append(row)
        payload = {**metadata, 'epoch': epoch, 'state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(), 'history': history}
        torch.save(payload, output / 'last.pt')
        f1 = float(val_report['classification']['f1'])
        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            torch.save(payload, output / 'best.pt')
        (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    result = {'status': 'trained', 'variant': 'mask_rcnn', 'model_variant': MODEL_VARIANT,
              'checkpoint': str(output / 'best.pt'), 'last_checkpoint': str(output / 'last.pt'),
              'checkpoint_sha256': _sha256(output / 'best.pt'), 'epochs': cfg['epochs'],
              'seed': cfg['seed'], 'initialization': cfg['initialization'],
              'best_epoch': best_epoch, 'best_val_class_f1': best_f1,
              'history_path': str(output / 'history.json'),
              'train_images': len(training.records), 'val_images': len(validation.records)}
    (output / 'training_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
