"""COCO masks with explicit scene and content isolation across splits."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .contracts import CLASSES, Dataset, Instance, Record, validate_instance


def read_image(path):
    import cv2
    value = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if value is None:
        raise ValueError('cannot decode image: ' + str(path))
    return value


def image_digest(path):
    image = read_image(path)
    header = '{}x{}x3:'.format(image.shape[0], image.shape[1]).encode('ascii')
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def encode_mask(mask):
    values = np.asarray(mask, dtype=bool).ravel(order='F')
    changes = np.flatnonzero(values[1:] != values[:-1]) + 1
    counts = np.diff(np.concatenate(([0], changes, [len(values)]))).tolist()
    if len(values) and values[0]:
        counts.insert(0, 0)
    return {'size': list(mask.shape), 'counts': counts}


def decode_segmentation(segmentation, height, width):
    if isinstance(segmentation, dict):
        if segmentation.get('size') != [height, width]:
            raise ValueError('RLE dimensions differ from image')
        counts = segmentation.get('counts')
        if isinstance(counts, list):
            if any(type(v) is not int or v < 0 for v in counts) or sum(counts) != height * width:
                raise ValueError('invalid uncompressed RLE counts')
            pixels = np.zeros(height * width, bool)
            position = 0
            for i, count in enumerate(counts):
                if i % 2:
                    pixels[position:position + count] = True
                position += count
            return pixels.reshape((height, width), order='F')
        try:
            from pycocotools import mask as mask_utils
        except ImportError as error:
            raise RuntimeError('compressed COCO RLE requires pycocotools; do not fill holes') from error
        decoded = np.asarray(mask_utils.decode(segmentation), bool)
        if decoded.shape != (height, width):
            raise ValueError('invalid compressed RLE shape')
        return decoded
    if not isinstance(segmentation, list) or not segmentation:
        raise ValueError('segmentation must contain polygon(s) or RLE')
    for polygon in segmentation:
        points = np.asarray(polygon, float)
        if points.ndim != 1 or len(points) < 6 or len(points) % 2:
            raise ValueError('COCO polygon must have at least three coordinate pairs')
        points = points.reshape(-1, 2)
        if (not np.isfinite(points).all() or (points < 0).any() or
                (points[:, 0] > width).any() or (points[:, 1] > height).any()):
            raise ValueError('polygon outside original image')
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise RuntimeError('COCO polygons require pycocotools for canonical pixel rasterization') from error
    # COCO uses subpixel polygon rasterization, not OpenCV rounded inclusive edges.
    rles = mask_utils.frPyObjects(segmentation, height, width)
    return np.asarray(mask_utils.decode(mask_utils.merge(rles)), dtype=bool)


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_dataset(manifest_path, split):
    manifest_path = Path(manifest_path).resolve()
    manifest = _json(manifest_path)
    if manifest.get('schema_version') != 1 or split not in ('train', 'val', 'test'):
        raise ValueError('unsupported dataset schema or split')
    if tuple(manifest.get('classes', CLASSES)) != CLASSES:
        raise ValueError('classes must be cake,banana,doll,clock in that order')
    splits = manifest.get('splits', {})
    if not isinstance(splits, dict) or set(splits) - {'train', 'val', 'test'} or split not in splits:
        raise ValueError('missing or invalid dataset split')
    scenes, contents, selected, identities = {}, {}, [], []
    for name, spec in sorted(splits.items()):
        annotation_path = (manifest_path.parent / spec['annotations']).resolve()
        image_root = (manifest_path.parent / spec['image_root']).resolve()
        coco = _json(annotation_path)
        categories = coco.get('categories', [])
        mapping = {category['id']: category['name'] for category in categories}
        if len(mapping) != 4 or set(mapping.values()) != set(CLASSES):
            raise ValueError('COCO must map four unique category ids to the four classes')
        images = coco.get('images', [])
        by_id = {image['id']: image for image in images}
        if len(by_id) != len(images):
            raise ValueError('duplicate COCO image id')
        annotations = {key: [] for key in by_id}
        annotation_ids = set()
        for annotation in coco.get('annotations', []):
            if annotation['id'] in annotation_ids or annotation['image_id'] not in by_id:
                raise ValueError('duplicate annotation id or missing image reference')
            annotation_ids.add(annotation['id'])
            if annotation.get('iscrowd', 0) or annotation.get('ignore', 0):
                raise ValueError('crowd/ignore annotations need an explicit ignore evaluator; unsupported here')
            annotations[annotation['image_id']].append(annotation)
        for image in images:
            scene = image.get('scene_id')
            if not isinstance(scene, str) or not scene.strip():
                raise ValueError('each COCO image requires a nonempty scene_id')
            if scene in scenes and scenes[scene] != name:
                raise ValueError('scene leakage across dataset splits: ' + scene)
            scenes[scene] = name
            path = (image_root / image['file_name']).resolve()
            if image_root not in path.parents:
                raise ValueError('image file_name escapes image_root')
            pixels = read_image(path)
            height, width = pixels.shape[:2]
            if width != image['width'] or height != image['height']:
                raise ValueError('COCO image dimensions differ from decoded image')
            digest = image_digest(path)
            if digest in contents:
                raise ValueError('duplicate image content / content leakage: ' + str(path))
            contents[digest] = name
            if image.get('pixel_sha256') and image['pixel_sha256'] != digest:
                raise ValueError('image pixel hash mismatch')
            instances = []
            for annotation in annotations[image['id']]:
                if annotation['category_id'] not in mapping:
                    raise ValueError('annotation references an unknown category')
                corners = annotation.get('corners')
                visibility = annotation.get('visibility', 'complete' if corners is not None else 'unknown')
                if visibility not in ('complete', 'partial', 'unknown'):
                    raise ValueError('visibility must be complete/partial/unknown')
                if corners is not None and visibility != 'complete':
                    raise ValueError('only complete reference corners are scored')
                instance = Instance(mapping[annotation['category_id']], 1.,
                    decode_segmentation(annotation['segmentation'], height, width), corners,
                    {'visibility': visibility, 'annotation_id': annotation['id']})
                validate_instance(instance, height, width)
                instances.append(instance)
            identities.append({'split': name, 'image_id': image['id'], 'scene_id': scene,
                               'pixels': digest, 'annotations': annotations[image['id']]})
            if name == split:
                selected.append(Record(image['id'], path, width, height, scene, instances))
    if not selected:
        raise ValueError('selected dataset split is empty; no training or scoring is possible')
    frozen = json.dumps({'mask_decoder': 'canonical_coco_v1', 'images': identities},
                        sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')
    return Dataset(CLASSES, selected, manifest_path, hashlib.sha256(frozen).hexdigest(), split)
