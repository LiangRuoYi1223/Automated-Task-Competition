"""A bounded, read-only workbench for image inputs and unverified vision baselines."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time


DEFAULT_CONFIG = {
    'measurement': 'diagnostic_unverified',
    'method': 'contour',
    'line_detection': {},
    'detection': {
        'min_area_px2': 500.0, 'max_area_fraction': 0.95,
        'max_line_error_px': 2.0, 'min_edge_points': 4,
        'min_edge_span_px': 10.0, 'min_corner_sine': 0.15,
        'approx_epsilon_px': 2.0, 'canny_low': 40, 'canny_high': 120,
        'roi': None, 'board_regions': [],
    },
    'classification': {'min_score': 0.75, 'min_margin': 0.08},
    'crop_size_px': 160,
}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def read_config(path=None):
    from .lines import DEFAULTS, validate_config
    config = deepcopy(DEFAULT_CONFIG)
    config['line_detection'] = deepcopy(DEFAULTS)
    if path is not None:
        def reject_constant(value):
            raise ValueError('configuration contains nonfinite JSON number: ' + value)
        data = json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject_constant)
        if not isinstance(data, dict) or set(data) - set(config):
            raise ValueError('unknown or invalid workbench configuration')
        for key, value in data.items():
            if key in ('detection', 'classification', 'line_detection'):
                if not isinstance(value, dict) or set(value) - set(config[key]):
                    raise ValueError('unknown or invalid ' + key + ' configuration')
                config[key].update(value)
            else:
                config[key] = value
    if config['measurement'] != 'diagnostic_unverified':
        raise ValueError('this workbench only supports diagnostic_unverified settings')
    if type(config['crop_size_px']) is not int or not 32 <= config['crop_size_px'] <= 1024:
        raise ValueError('crop_size_px must be an integer from 32 to 1024')
    if config['method'] not in ('contour', 'lsd', 'edgedrawing'):
        raise ValueError('method must be contour, lsd or edgedrawing')
    config['line_detection'] = validate_config(config['line_detection'])
    return config


def _save_json(path, data):
    # Exclusive creation also protects completed output from accidental reruns.
    serialized = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(serialized)


def _save_png(path, bgr):
    import cv2
    ok, encoded = cv2.imencode('.png', bgr)
    if not ok:
        raise ValueError('PNG encoding failed')
    with Path(path).open('xb') as handle:
        handle.write(encoded.tobytes())


def warp_candidate(bgr, corners, size):
    import cv2
    import numpy as np
    points = np.asarray(corners, dtype=np.float32)
    height, width = bgr.shape[:2]
    if (points.shape != (4, 2) or not np.isfinite(points).all()
            or (points[:, 0] < 0).any() or (points[:, 0] >= width).any()
            or (points[:, 1] < 0).any() or (points[:, 1] >= height).any()
            or not cv2.isContourConvex(points.reshape(-1, 1, 2))):
        raise ValueError('candidate must have four convex, finite in-image corners')
    target = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32)
    transform = cv2.getPerspectiveTransform(points, target)
    return cv2.warpPerspective(bgr, transform, (size, size))


def input_frames(path, max_frames, stride):
    from .frames import load_image, iter_video
    source = Path(path)
    if not source.exists():
        raise ValueError('input does not exist: ' + str(source))
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError('input directory contains no supported images')
        for image in images[::stride][:max_frames]:
            yield load_image(image)
    elif source.suffix.lower() in IMAGE_EXTENSIONS:
        yield load_image(source)
    elif source.suffix.lower() in VIDEO_EXTENSIONS:
        yield from iter_video(source, max_frames=max_frames, stride=stride)
    else:
        raise ValueError('unsupported input extension: ' + source.suffix)


def demo_frames():
    import cv2
    import numpy as np
    from .frames import Frame
    bgr = np.full((400, 640, 3), 35, dtype=np.uint8)
    # Artificial polygons only: these are not competition photos or class templates.
    for quad in ([[60, 65], [190, 75], [180, 195], [50, 180]],
                 [[270, 90], [410, 60], [430, 200], [290, 220]],
                 [[450, 260], [570, 250], [580, 350], [440, 360]]):
        cv2.fillConvexPoly(bgr, np.array(quad, np.int32), (50, 190, 235))
    cv2.putText(bgr, 'SYNTHETIC PIXEL TEST', (15, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
    yield Frame(bgr=bgr, metadata={'origin': 'synthetic_demo', 'source_kind': 'synthetic_demo',
                                  'source_description': 'generated polygons; no real objects or class truth'})


def run_inspection(frames, output, config, templates=None, origin='offline_replay'):
    import cv2
    import numpy as np
    from .contours import detect_candidates
    from .lines import capabilities, detect_candidates as detect_lines, validate_config
    from .capture import SessionWriter
    from .templates import TemplateClassifier
    classifier = TemplateClassifier(templates, **config['classification'])
    config = deepcopy(config)
    method = config.setdefault('method', 'contour')
    if method not in ('contour', 'lsd', 'edgedrawing'):
        raise ValueError('unknown detection method')
    config['line_detection'] = validate_config(config.get('line_detection', {}))
    frozen = json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    writer = SessionWriter(output / 'frames')
    report = {
        'schema_version': 1, 'status': 'diagnostic_only', 'origin': origin,
        'actionable': False, 'real_world_validated': False, 'camera_calibrated': False,
        'coordinate_space': 'original_image_pixels', 'configuration': config,
        'configuration_sha256': hashlib.sha256(frozen.encode('utf-8')).hexdigest(),
        'capabilities': capabilities(), 'search_complete': True,
        'templates_ready': classifier.ready, 'template_manifest': classifier.manifest, 'frames': [],
        'limitations': [
            'four-sided candidates are not confirmed bricks or top planes',
            'board regions are manually configured exclusions, not automatic board detections',
            'template similarity is not a calibrated probability',
            'no physical coordinates, target identity, control permission or robot commands',
        ],
    }
    try:
        for index, frame in enumerate(frames):
            frame_started = time.perf_counter()
            saved = writer.write(frame)
            algorithm_started = time.perf_counter()
            result = (detect_candidates(frame.bgr, config['detection']) if method == 'contour' else
                      detect_lines(frame.bgr, config['detection'], config['line_detection'], method=method))
            result.setdefault('timing_ms', (time.perf_counter() - algorithm_started) * 1000.)
            result.setdefault('method', method)
            result.setdefault('unresolved', [])
            result.setdefault('budget_truncated', False)
            if result['budget_truncated']:
                report['search_complete'] = False
                report['status'] = 'diagnostic_incomplete'
            overlay = frame.bgr.copy()
            frame_report = {'index': index, 'source': saved, **result}
            for number, candidate in enumerate(frame_report['candidates']):
                crop = warp_candidate(frame.bgr, candidate['corners'], config['crop_size_px'])
                name = 'crop_{:06d}_{:03d}.png'.format(index, number)
                _save_png(output / name, crop)
                candidate['crop_path'] = name
                candidate['classification'] = classifier.classify(crop)
                candidate['actionable'] = False
                poly = np.rint(candidate['corners']).astype(np.int32)
                cv2.polylines(overlay, [poly], True, (0, 220, 255), 2)
                x, y = poly[0]
                label = '{}: {}'.format(number, candidate['classification']['category'])
                cv2.putText(overlay, label, (int(x), max(15, int(y) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
            _draw_evidence(overlay, frame_report, cv2, np)
            if method != 'contour':
                support_overlay = frame.bgr.copy()
                for group in result.get('groups', []):
                    points = np.rint(group.get('observed_points', [])).astype(np.int32).reshape(-1, 2)
                    for x, y in points:
                        cv2.circle(support_overlay, (int(x), int(y)), 1, (240, 210, 40), -1)
                    if len(points):
                        x, y = points[len(points) // 2]
                        cv2.putText(support_overlay, str(group['id']), (int(x), int(y)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .32, (30, 40, 250), 1)
                note = result.get('no_candidate_reason') or 'observed gradient pixels; labels=group IDs'
                cv2.putText(support_overlay, note, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 255), 1)
                support_name = 'support_{:06d}.png'.format(index)
                _save_png(output / support_name, support_overlay)
                frame_report['support_overlay_path'] = support_name
            for region in config['detection']['board_regions']:
                x, y, w, h = region
                cv2.rectangle(overlay, (round(x), round(y)), (round(x + w), round(y + h)), (255, 80, 80), 2)
            roi = config['detection']['roi']
            if roi is not None:
                x, y, w, h = roi
                cv2.rectangle(overlay, (round(x), round(y)), (round(x + w), round(y + h)), (80, 220, 80), 1)
            overlay_name = 'overlay_{:06d}.png'.format(index)
            _save_png(output / overlay_name, overlay)
            frame_report['overlay_path'] = overlay_name
            frame_report['frame_processing_ms'] = (time.perf_counter() - frame_started) * 1000.
            frame_report['latency_basis'] = 'local save + algorithm + overlay; not camera-to-result latency'
            report['frames'].append(frame_report)
        if not report['frames']:
            raise ValueError('no decodable frames received')
        report['processing_complete'] = True
    except BaseException as error:
        report['processing_complete'] = False
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'error'
        report['error'] = str(error)
        _save_json(output / 'report.json', report)
        raise
    _save_json(output / 'report.json', report)
    return report


def _draw_evidence(overlay, result, cv2, np):
    """Show observed pixels separately from fitted edges; retain unresolved output."""
    for kind, color in (('candidates', (0, 240, 100)), ('unresolved', (255, 120, 220))):
        for number, item in enumerate(result.get(kind, [])):
            corners = np.rint(item['corners']).astype(np.int32)
            cv2.polylines(overlay, [corners], True, color, 1)
            for edge_index, points in enumerate(item['edges']):
                for x, y in np.rint(points).astype(np.int32):
                    cv2.circle(overlay, (int(x), int(y)), 1, color, -1)
                if edge_index < len(item.get('edge_metrics', [])):
                    metric = item['edge_metrics'][edge_index]
                    x, y = corners[edge_index]
                    text = 'e{} s{:.2f} g{:.1f}'.format(edge_index, metric['support_rate'], metric['max_gap_px'])
                    cv2.putText(overlay, text, (int(x), min(overlay.shape[0] - 3, int(y) + 13)),
                                cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1)
            for x, y in corners:
                cv2.circle(overlay, (int(x), int(y)), 3, (255, 255, 255), 1)
            if kind == 'unresolved':
                x, y = corners[0]
                cv2.putText(overlay, 'unresolved {}'.format(number), (int(x), max(12, int(y) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    if result.get('budget_truncated'):
        cv2.putText(overlay, 'SEARCH TRUNCATED - scene undecided', (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 255), 1)


def positive_integer(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def build_parser():
    parser = argparse.ArgumentParser(description='Read-only vision workbench; diagnostic candidates only.')
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('capabilities', help='report local OpenCV frontends; no robot changes')
    evaluate = subs.add_parser('evaluate', help='score frozen diagnostic report against manual references')
    evaluate.add_argument('--report', required=True, help='completed inspection report JSON')
    evaluate.add_argument('--references', required=True, help='manual reference JSON with scene splits and pixel hashes')
    evaluate.add_argument('--output', required=True, help='new metrics JSON file; never overwritten')
    for command in ('inspect', 'demo'):
        sub = subs.add_parser(command)
        sub.add_argument('--output', required=True, help='new session directory; never overwritten')
        sub.add_argument('--config', help='unverified diagnostic configuration JSON')
        sub.add_argument('--method', choices=('contour', 'lsd', 'edgedrawing'), help='override configured method')
        sub.add_argument('--templates', help='directory containing doll/clock/banana/cake subdirectories')
        if command == 'inspect':
            sub.add_argument('--input', required=True, help='image, directory of images, or video')
            sub.add_argument('--max-frames', type=positive_integer, default=100)
            sub.add_argument('--stride', type=positive_integer, default=1)
    capture = subs.add_parser('capture-ros', help='bounded sensor_msgs/Image subscriber; no driver startup')
    capture.add_argument('--topic', required=True)
    capture.add_argument('--output', required=True)
    capture.add_argument('--max-frames', type=positive_integer, default=100)
    capture.add_argument('--timeout-s', type=float, default=30.0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'capabilities':
            from .lines import capabilities
            print(json.dumps(capabilities(), ensure_ascii=False, allow_nan=False))
            return 0
        if args.command == 'evaluate':
            from .evaluation import evaluate_report
            def read_finite_json(path):
                def reject(value):
                    raise ValueError('nonfinite JSON number: ' + value)
                raw = Path(path).read_bytes()
                return json.loads(raw.decode('utf-8-sig'), parse_constant=reject), hashlib.sha256(raw).hexdigest()
            inspection, report_hash = read_finite_json(args.report)
            references, reference_hash = read_finite_json(args.references)
            metrics = evaluate_report(inspection, references)
            metrics['input_report_sha256'] = report_hash
            metrics['input_references_sha256'] = reference_hash
            _save_json(args.output, metrics)
            print(json.dumps({'output': str(Path(args.output).resolve()), 'actionable': False}, ensure_ascii=False))
            return 0
        if args.command == 'capture-ros':
            from .capture import capture_ros
            summary = capture_ros(args.topic, args.output, args.max_frames, args.timeout_s)
            print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
            return 0 if summary.get('accepted_frames', 0) else 2
        config = read_config(args.config)
        if args.method is not None:
            config['method'] = args.method
        frames = demo_frames() if args.command == 'demo' else input_frames(args.input, args.max_frames, args.stride)
        report = run_inspection(frames, args.output, config, args.templates,
                                origin='synthetic_demo' if args.command == 'demo' else 'offline_replay')
        print(json.dumps({'output': str(Path(args.output).resolve()), 'frames': len(report['frames']),
                          'candidates': sum(len(f['candidates']) for f in report['frames']),
                          'templates_ready': report['templates_ready'], 'actionable': False}, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        print('error: interrupted', file=sys.stderr)
        return 130
    except Exception as error:
        print('error: ' + str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
