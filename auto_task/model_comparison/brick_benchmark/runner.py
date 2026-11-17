"""Run independent backends, retain evidence, and compare identical eval sets."""
import copy
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import time
from .contracts import Instance, validate_instance
from .data import load_dataset, read_image, image_digest, encode_mask, decode_segmentation
from .geometry import add_geometry
from .metrics import DEFAULT_POLICY, evaluate_frames

VARIANTS = {'yolo11_seg': 'ultralytics_backend', 'yolo26_seg': 'ultralytics_backend',
            'rtmdet_ins': 'rtmdet_backend', 'mask_rcnn': 'torchvision_backend',
            'nontrained': 'nontrained'}
TIMING_SCOPE = 'predict + device synchronize + mask geometry; excludes file decode, model load and warmup'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def read_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    if config.get('variant') not in VARIANTS:
        raise ValueError('unknown variant')
    for key in ('checkpoint', 'init_checkpoint', 'config_file', 'templates'):
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
    return config


def backend_module(config):
    try:
        name = VARIANTS[config['variant']]
    except (KeyError, TypeError) as error:
        raise ValueError('a supported variant is required') from error
    return importlib.import_module('.backends.' + name, __package__)


def environment(config):
    versions = {}
    for name in ('numpy', 'opencv-python', 'pycocotools', 'ultralytics', 'torch', 'torchvision', 'mmdet', 'mmcv', 'mmengine'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    hardware = {'host': platform.node(), 'system': platform.platform(),
                'machine': platform.machine(), 'processor': platform.processor(),
                'device': str(config.get('device', 'cpu'))}
    return {'hardware': hardware, 'python': platform.python_version(), 'versions': versions}


def run_prediction(manifest_path, split, config, output, warmup=2):
    if Path(output).exists():
        raise FileExistsError(str(output))
    if type(warmup) is not int or not 0 <= warmup <= 100:
        raise ValueError('warmup must be an integer in [0,100]')
    dataset = load_dataset(manifest_path, split)
    module = backend_module(config)
    backend = module.create(config)
    sync = getattr(backend, 'synchronize', lambda: None)
    first = read_image(dataset.records[0].image_path)
    for _ in range(warmup):
        backend.predict(first)
        sync()
    frames = []
    for record in dataset.records:
        image = read_image(record.image_path)
        sync()
        start = time.perf_counter()
        instances = list(backend.predict(image))
        sync()
        for instance in instances:
            validate_instance(instance, record.height, record.width)
        if config['variant'] != 'nontrained':
            add_geometry(instances, config.get('geometry'))
        elapsed = (time.perf_counter() - start) * 1000.
        frames.append({'image_id': record.image_id, 'pixel_sha256': image_digest(record.image_path),
            'pipeline_ms': elapsed, 'instances': [dict(category=p.category, score=float(p.score),
                segmentation=encode_mask(p.mask), corners=p.corners, details=p.details) for p in instances]})
    file_hashes = {}
    for key in ('checkpoint', 'init_checkpoint', 'config_file'):
        if config.get(key) and Path(config[key]).is_file():
            file_hashes[key] = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
    source_hashes = {p.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in Path(__file__).parent.rglob('*.py')}
    if config['variant'] == 'nontrained':
        source_hashes.update(module.source_hashes())
    env = environment(config)
    artifacts = copy.deepcopy(getattr(backend, 'artifact_manifests', {}))
    result = {'schema_version': 1, 'status': 'complete', 'variant': config['variant'],
              'split': split, 'dataset_fingerprint': dataset.fingerprint,
              'configuration': copy.deepcopy(config), 'configuration_sha256': digest(config),
              'weight_hashes': file_hashes, 'source_hashes': source_hashes,
              'artifact_manifests': artifacts, 'artifact_manifest_sha256': digest(artifacts),
              'environment': env, 'warmup_frames': warmup, 'timing_scope': TIMING_SCOPE,
              'actionable': False, 'frames': frames}
    write_json(output, result)
    return result


def score_prediction(manifest_path, prediction_path, policy=None):
    prediction = read_json(prediction_path)
    if prediction.get('schema_version') != 1 or prediction.get('status') != 'complete':
        raise ValueError('only complete versioned predictions can be scored')
    dataset = load_dataset(manifest_path, prediction['split'])
    if dataset.fingerprint != prediction['dataset_fingerprint']:
        raise ValueError('dataset fingerprint changed since inference')
    if digest(prediction['configuration']) != prediction['configuration_sha256']:
        raise ValueError('prediction configuration hash mismatch')
    if digest(prediction['artifact_manifests']) != prediction['artifact_manifest_sha256']:
        raise ValueError('prediction artifact manifest hash mismatch')
    entries = prediction['frames']
    lookup = {entry['image_id']: entry for entry in entries}
    if len(lookup) != len(entries) or set(lookup) != {r.image_id for r in dataset.records}:
        raise ValueError('missing, duplicate or unexpected prediction frames')
    frames = []
    for record in dataset.records:
        entry = lookup[record.image_id]
        if entry['pixel_sha256'] != image_digest(record.image_path):
            raise ValueError('prediction refers to different image pixels')
        instances = [Instance(p['category'], p['score'],
                     decode_segmentation(p['segmentation'], record.height, record.width),
                     p.get('corners'), p.get('details', {})) for p in entry['instances']]
        frames.append((record.instances, instances, entry['pipeline_ms'], record.width, record.height))
    policy = copy.deepcopy(policy or DEFAULT_POLICY)
    return {'schema_version': 1, 'variant': prediction['variant'], 'split': dataset.split,
            'dataset_fingerprint': dataset.fingerprint, 'policy': policy,
            'hardware': prediction['environment']['hardware'], 'timing_scope': prediction['timing_scope'],
            'environment': prediction['environment'], 'configuration_sha256': prediction['configuration_sha256'],
            'weight_hashes': prediction['weight_hashes'], 'source_hashes': prediction['source_hashes'],
            'artifact_manifests': prediction['artifact_manifests'],
            'artifact_manifest_sha256': prediction['artifact_manifest_sha256'],
            'metrics': evaluate_frames(frames, policy), 'actionable': False,
            'purpose': 'validation_selection' if dataset.split == 'val' else 'final_confirmation_only'}


def compare_reports(reports):
    if not reports:
        raise ValueError('at least one report is required')
    first, seen = reports[0], set()
    for report in reports:
        if report['variant'] in seen:
            raise ValueError('duplicate variant in comparison')
        seen.add(report['variant'])
        for key in ('dataset_fingerprint', 'split', 'hardware', 'policy'):
            if report[key] != first[key]:
                raise ValueError('reports differ in ' + key)
        if report.get('timing_scope') != first.get('timing_scope'):
            raise ValueError('reports use different timing scopes')
    rows = sorted(reports, key=lambda r: (-r['metrics']['score'],
                  r['metrics'].get('latency_ms', {}).get('p95', float('inf'))))
    ready = (seen == set(VARIANTS) and first['split'] == 'val' and
             all(r['metrics']['selection_ready'] for r in reports))
    return {'schema_version': 1, 'split': first['split'], 'dataset_fingerprint': first['dataset_fingerprint'],
            'hardware': first['hardware'], 'policy': first['policy'],
            'selection_ready': ready, 'missing_variants': sorted(set(VARIANTS) - seen),
            'recommended_offline_candidate': rows[0]['variant'] if ready else None,
            'ranking': [{'variant': r['variant'], **r['metrics']} for r in rows],
            'geometry_ranking': [r['variant'] for r in sorted(reports,
                                   key=lambda r: -r['metrics']['geometry_score'])],
            'limitations': ['score weights are engineering defaults, freeze before evaluation',
                'latency is reported separately and breaks score ties; apply measured match-time limits manually',
                'test split confirms a frozen choice; do not tune against it',
                'offline recommendation is not field acceptance or robot motion permission']}
