"""Local YOLO11n/YOLO26n instance-segmentation adapters.

Importing this module does not import PyTorch or Ultralytics. Training accepts
local pretrained segment checkpoints; inference additionally requires the four
benchmark classes. No model-name shorthand or remote checkpoint is accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from ..contracts import CLASSES, Instance
from ..data import load_dataset, read_image


_VARIANTS = {'yolo11_seg': '11', 'yolo26_seg': '26'}


def _configuration(config, *, training=False):
    values = dict(config)
    variant = values.get('variant')
    if variant not in _VARIANTS:
        raise ValueError('variant must be yolo11_seg or yolo26_seg')
    checkpoint = values.get('init_checkpoint') if training else None
    checkpoint = checkpoint or values.get('checkpoint')
    if not isinstance(checkpoint, (str, Path)) or not str(checkpoint).strip():
        raise ValueError('an explicit local init_checkpoint (train) or checkpoint (inference) is required')
    checkpoint = Path(checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'local checkpoint does not exist: {checkpoint}; automatic download is disabled')
    if checkpoint.suffix.lower() != '.pt':
        raise ValueError('YOLO backend requires a local .pt segment checkpoint')
    values['checkpoint'] = checkpoint
    for key, default in (('image_size', 640), ('epochs', 100), ('batch', 8), ('max_detections', 100)):
        value = values.get(key, default)
        if type(value) is not int or value <= 0:
            raise ValueError(f'{key} must be a positive integer')
        values[key] = value
    seed = values.get('seed', 0)
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    values['seed'] = seed
    for key, default in (('confidence', .25), ('mask_threshold', .5)):
        value = values.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{key} must be finite in [0, 1]')
        values[key] = float(value)
    values['device'] = values.get('device', 'cpu')
    return values


def _load_model(config, *, inference):
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError('YOLO backend requires the optional ultralytics and PyTorch dependencies; install them in its own environment') from error
    model = YOLO(str(config['checkpoint']))
    if getattr(model, 'task', None) != 'segment':
        raise ValueError('checkpoint task must be segment (instance segmentation)')
    architecture = getattr(getattr(model, 'model', None), 'yaml', {})
    if not isinstance(architecture, dict):
        raise ValueError('checkpoint lacks verifiable YOLO architecture metadata')
    yaml_name = re.split(r'[/\\]', str(architecture.get('yaml_file', '')))[-1]
    match = re.fullmatch(r'yolo(11|26)([nslmx]?)-seg\.ya?ml', yaml_name.lower())
    scale = architecture.get('scale') or (match.group(2) if match else '')
    if not match or match.group(1) != _VARIANTS[config['variant']]:
        raise ValueError('checkpoint architecture does not match requested variant')
    if scale != 'n' or match.group(2) not in ('', 'n'):
        raise ValueError('comparison requires nano (n) checkpoints for both YOLO variants')
    if inference:
        names = getattr(model, 'names', None)
        if isinstance(names, dict):
            if set(names) != set(range(len(CLASSES))):
                raise ValueError('inference checkpoint names must map ids 0..3 to the four benchmark classes')
            names = tuple(names[i] for i in range(len(CLASSES)))
        elif isinstance(names, (list, tuple)):
            names = tuple(names)
        if names != CLASSES:
            raise ValueError('inference checkpoint names must be cake, banana, doll, clock in that order; train the model first')
    return model


def _numpy(value):
    for operation in ('detach', 'cpu', 'numpy'):
        method = getattr(value, operation, None)
        if callable(method):
            value = method()
    return np.asarray(value)


class UltralyticsBackend:
    def __init__(self, config):
        self.config = _configuration(config)
        self.model = _load_model(self.config, inference=True)

    def predict(self, bgr):
        if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3
                or bgr.shape[2] != 3 or min(bgr.shape[:2]) <= 0):
            raise ValueError('predict requires a nonempty uint8 BGR H x W x 3 image')
        results = self.model.predict(source=bgr, imgsz=self.config['image_size'],
                                     conf=self.config['confidence'], device=self.config['device'],
                                     retina_masks=True, save=False, verbose=False, stream=False,
                                     max_det=self.config['max_detections'])
        if len(results) != 1:
            raise ValueError('single-image prediction must return exactly one Results object')
        result = results[0]
        shape = tuple(bgr.shape[:2])
        if tuple(result.orig_shape) != shape:
            raise ValueError('prediction original image shape differs from input')
        boxes = getattr(result, 'boxes', None)
        masks = getattr(result, 'masks', None)
        if boxes is None:
            if masks is not None and len(_numpy(masks.data)):
                raise ValueError('prediction has masks without corresponding boxes')
            return []
        categories = _numpy(boxes.cls).reshape(-1)
        scores = _numpy(boxes.conf).reshape(-1)
        if not len(categories) and masks is None and not len(scores):
            return []
        if masks is None:
            raise ValueError('segment prediction returned boxes without masks')
        mask_values = _numpy(masks.data)
        if mask_values.shape != (len(categories), *shape) or len(scores) != len(categories):
            raise ValueError('retina masks must have original H x W shape and match boxes')
        if not np.isfinite(mask_values).all() or (mask_values < 0).any() or (mask_values > 1).any():
            raise ValueError('prediction masks must contain finite values in [0, 1]')
        predictions = []
        for category, score, mask in zip(categories, scores, mask_values):
            if not np.isfinite(category) or category != int(category) or not 0 <= category < len(CLASSES):
                raise ValueError('predicted class id is outside the four-class mapping')
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('predicted confidence must be finite in [0, 1]')
            if score < self.config['confidence']:
                continue
            binary = np.array(mask >= self.config['mask_threshold'], dtype=bool, copy=True)
            if not binary.any():
                raise ValueError('prediction produced an empty instance mask after thresholding')
            predictions.append(Instance(CLASSES[int(category)], float(score), binary,
                                        details={'backend': self.config['variant']}))
        return predictions

    def synchronize(self):
        """Complete accelerator work so the caller can time the entire request."""
        device = str(self.config['device']).lower()
        if device == 'cpu':
            return
        try:
            import torch
        except ImportError as error:
            raise RuntimeError('accelerator synchronization requires PyTorch') from error
        if device == 'mps':
            torch.mps.synchronize()
            return
        if device.isdecimal():
            device = f'cuda:{device}'
        if device.startswith('cuda'):
            torch.cuda.synchronize(device=device)
            return
        raise ValueError('timing requires an explicit single cpu, mps, or CUDA device')


def create(config):
    return UltralyticsBackend(config)


def mask_to_polygon(mask):
    """Return one lossless raster contour, rejecting unsupported mask topology."""
    import cv2
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.size:
        raise ValueError('YOLO export requires a bool H x W mask')
    if not mask.any():
        raise ValueError('YOLO export rejects an empty instance')
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None and np.any(hierarchy[0, :, 3] >= 0):
        raise ValueError('YOLO polygon export cannot represent instance holes; retain COCO/RLE')
    if len(contours) != 1:
        raise ValueError('YOLO polygon export cannot represent multiple connected components; retain COCO/RLE')
    contour = contours[0].reshape(-1, 2)
    if len(contour) < 3 or cv2.contourArea(contours[0]) <= 0:
        raise ValueError('YOLO polygon export rejects a degenerate instance contour')
    restored = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(restored, [contour], 1)
    if not np.array_equal(restored.astype(bool), mask):
        raise ValueError('YOLO polygon conversion would alter the original instance pixels')
    return contour.astype(np.float64) / [mask.shape[1], mask.shape[0]]


def export_yolo_dataset(manifest_path, output_dir):
    """Export all train/val records into a new directory without flattening masks.

    JSON is a YAML subset, so data.yaml needs no additional YAML dependency.
    Images are decoded through the Unicode-safe reader and encoded as lossless
    PNGs under unique ASCII names. The upstream trainer receives no test split.
    """
    import cv2
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'YOLO export destination must not already exist: {output}')
    datasets = {split: load_dataset(manifest_path, split) for split in ('train', 'val')}
    labels = {}
    for split, dataset in datasets.items():
        if tuple(dataset.classes) != CLASSES:
            raise ValueError('YOLO dataset classes must use the fixed four-class mapping')
        if not dataset.records:
            raise ValueError(f'YOLO training requires a nonempty {split} split')
        if split == 'train' and not any(record.instances for record in dataset.records):
            raise ValueError('YOLO training requires at least one labeled training instance')
        labels[split] = []
        for record in dataset.records:
            rows = []
            for index, instance in enumerate(record.instances):
                if instance.category not in CLASSES or instance.mask.shape != (record.height, record.width):
                    raise ValueError('YOLO export instance category or original mask size is invalid')
                try:
                    polygon = mask_to_polygon(instance.mask)
                except ValueError as error:
                    raise ValueError(f'{split} image {record.image_id}, instance {index}: {error}') from error
                rows.append(str(CLASSES.index(instance.category)) + ' ' + ' '.join(f'{v:.10f}' for v in polygon.ravel()))
            labels[split].append('\n'.join(rows) + ('\n' if rows else ''))
    # Create only after every annotation has passed the topology checks.
    output.mkdir(parents=True, exist_ok=False)
    mapping = []
    for split, dataset in datasets.items():
        for folder in ('images', 'labels'):
            (output / folder / split).mkdir(parents=True)
        for index, (record, label) in enumerate(zip(dataset.records, labels[split])):
            stem = f'{index:08d}'
            image = read_image(record.image_path)
            if image.shape[:2] != (record.height, record.width):
                raise ValueError('decoded image dimensions differ from annotation')
            success, encoded = cv2.imencode('.png', image)
            if not success:
                raise ValueError(f'cannot encode training image: {record.image_path}')
            (output / 'images' / split / (stem + '.png')).write_bytes(encoded.tobytes())
            (output / 'labels' / split / (stem + '.txt')).write_text(label, encoding='utf-8')
            mapping.append({'split': split, 'image_id': record.image_id,
                            'source': str(record.image_path), 'exported': f'images/{split}/{stem}.png'})
    descriptor = {'path': str(output), 'train': 'images/train', 'val': 'images/val', 'names': list(CLASSES)}
    data_path = output / 'data.yaml'
    data_path.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'image_mapping.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return {'data': str(data_path), 'counts': {s: len(d.records) for s, d in datasets.items()}}


def train(manifest_path, output_dir, config):
    """Fine-tune a local nano checkpoint using only exported train/val labels."""
    values = _configuration(config, training=True)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'training output must not already exist: {output}')
    model = _load_model(values, inference=False)
    output.mkdir(parents=True, exist_ok=False)
    exported = export_yolo_dataset(manifest_path, output / 'dataset')
    # Explicitly use the supplied local weights for transfer learning. In the
    # current upstream API pretrained=False discards the loaded parameters.
    # Disable AMP's auxiliary-model check, which may fetch unrelated weights.
    settings = dict(data=exported['data'], epochs=values['epochs'], batch=values['batch'],
                    seed=values['seed'], imgsz=values['image_size'], device=values['device'],
                    project=str(output), name='run', exist_ok=False, workers=0,
                    pretrained=str(values['checkpoint']), amp=False, deterministic=True, resume=False,
                    plots=False, save=True, val=True, single_cls=False, classes=None, fraction=1.0,
                    max_det=values['max_detections'])
    request = {'variant': values['variant'], 'manifest': str(Path(manifest_path).resolve()),
               'init_checkpoint': str(values['checkpoint']),
               'init_checkpoint_sha256': hashlib.sha256(values['checkpoint'].read_bytes()).hexdigest(),
               'counts': exported['counts'], 'settings': settings}
    request_path = output / 'training_request.json'
    request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    results = model.train(**settings)
    save_dir = Path(model.trainer.save_dir).resolve()
    checkpoint = save_dir / 'weights' / 'best.pt'
    if not checkpoint.is_file():
        raise RuntimeError('upstream training finished without the expected best.pt checkpoint')
    report = {'variant': values['variant'], 'checkpoint': str(checkpoint),
              'init_checkpoint': str(values['checkpoint']),
              'init_checkpoint_sha256': hashlib.sha256(values['checkpoint'].read_bytes()).hexdigest(),
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'data': exported['data'], 'counts': exported['counts'], 'request': str(request_path),
              'metrics': {str(k): float(v) for k, v in getattr(results, 'results_dict', {}).items()}}
    (output / 'training_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report
