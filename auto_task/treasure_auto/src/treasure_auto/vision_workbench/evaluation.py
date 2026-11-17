"""Provenance-bound offline candidate metrics against manual quadrilaterals.

References describe independent scenes and tuning/acceptance membership. They
are evaluation annotations, never training data or proof of a top plane. This
module deliberately makes no automatic real-world acceptance or action claim.
"""
import hashlib
import json
import math


def _number(value, name, minimum=0., maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value) or
            value < minimum or (maximum is not None and value > maximum)):
        raise ValueError(name + ' must be a finite number in range')
    return float(value)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer in range')
    return value


def _hash(value):
    if (not isinstance(value, str) or len(value) != 64 or
            any(c not in '0123456789abcdefABCDEF' for c in value)):
        raise ValueError('pixel_sha256 must be a 64-character hexadecimal SHA256')
    return value.lower()


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _quad(points, width, height):
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise ValueError('corners must contain four cyclic convex vertices')
    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError('corner must be [x,y]')
        x, y = (_number(point[0], 'corner x'), _number(point[1], 'corner y'))
        if x >= width or y >= height:
            raise ValueError('corner lies outside original image dimensions')
        result.append((x, y))
    turns = [_cross(result[i], result[(i + 1) % 4], result[(i + 2) % 4]) for i in range(4)]
    if not (all(t > 1e-8 for t in turns) or all(t < -1e-8 for t in turns)):
        raise ValueError('corners must be a nondegenerate cyclic convex quadrilateral')
    return result if turns[0] > 0 else list(reversed(result))


def _area(points):
    if len(points) < 3:
        return 0.
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(points, points[1:] + points[:1]))) / 2.


def _iou(a, b):
    # Sutherland-Hodgman clipping; both inputs have positive cyclic winding.
    polygon = list(a)
    for start, end in zip(b, b[1:] + b[:1]):
        clipped = []
        if not polygon:
            break
        previous = polygon[-1]
        previous_distance = _cross(start, end, previous)
        for current in polygon:
            distance = _cross(start, end, current)
            if (distance >= 0.) != (previous_distance >= 0.):
                fraction = previous_distance / (previous_distance - distance)
                clipped.append((previous[0] + fraction * (current[0] - previous[0]),
                                previous[1] + fraction * (current[1] - previous[1])))
            if distance >= 0.:
                clipped.append(current)
            previous, previous_distance = current, distance
        polygon = clipped
    intersection = _area(polygon)
    union = _area(a) + _area(b) - intersection
    return max(0., min(1., intersection / union)) if union > 0. else 0.


def _matches(candidates, targets, threshold):
    scores = [[_iou(candidate, target) for target in targets] for candidate in candidates]
    adjacency = [sorted((j for j, value in enumerate(row) if value >= threshold),
                        key=lambda j: (-row[j], j)) for row in scores]
    owners = {}

    def augment(candidate, seen):
        for target in adjacency[candidate]:
            if target in seen:
                continue
            seen.add(target)
            if target not in owners or augment(owners[target], seen):
                owners[target] = candidate
                return True
        return False

    # Augmenting paths guarantee maximum cardinality, unlike greedy IoU.
    for candidate in range(len(candidates)):
        augment(candidate, set())
    return sorted((candidate, target, scores[candidate][target])
                  for target, candidate in owners.items())


def _aligned_errors(candidate, reference, width, height):
    alternatives = []
    for shift in range(4):
        aligned = candidate[shift:] + candidate[:shift]
        raw = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(aligned, reference)]
        normalized = [math.hypot((a[0] - b[0]) * 640. / width,
                                 (a[1] - b[1]) * 480. / height)
                      for a, b in zip(aligned, reference)]
        alternatives.append((sum(v * v for v in raw), raw, normalized))
    _, raw, normalized = min(alternatives, key=lambda item: item[0])
    return raw, normalized


def _quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _summary(values):
    return dict(count=len(values), median=_quantile(values, .5), p95=_quantile(values, .95))


def _indexed(items, name):
    if not isinstance(items, list):
        raise ValueError(name + ' must be an array')
    result = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(name + ' must contain objects')
        index = _integer(item.get('index'), name + ' index')
        if index in result:
            raise ValueError('duplicate frame mapping in ' + name)
        result[index] = item
    return result


def _validate(frames, references):
    if (not isinstance(references, dict) or type(references.get('schema_version')) is not int or
            references['schema_version'] != 1 or
            references.get('measurement') != 'manual_reference_unverified'):
        raise ValueError('references require schema_version 1 and manual_reference_unverified measurement')
    actual = _indexed(frames, 'frames')
    expected = _indexed(references.get('frames'), 'references')
    if not actual or set(actual) != set(expected):
        raise ValueError('frame/reference mappings must be nonempty and match exactly')
    scene_splits, pixel_splits, pixel_scenes, validated = {}, {}, {}, []
    for index in sorted(actual):
        frame, reference = actual[index], expected[index]
        source = frame.get('source')
        if not isinstance(source, dict):
            raise ValueError('frame source must contain original image dimensions and pixel_sha256')
        width = _integer(reference.get('width'), 'reference width', 1)
        height = _integer(reference.get('height'), 'reference height', 1)
        if (type(source.get('width')) is not int or source['width'] != width or
                type(source.get('height')) is not int or source['height'] != height):
            raise ValueError('reference dimensions do not match original frame source')
        digest = _hash(reference.get('pixel_sha256'))
        if _hash(source.get('pixel_sha256')) != digest:
            raise ValueError('reference pixel_sha256 does not match frame source')
        scene, split = reference.get('scene_id'), reference.get('split')
        if not isinstance(scene, str) or not scene.strip() or split not in ('acceptance', 'tuning'):
            raise ValueError('reference requires nonempty scene_id and acceptance/tuning split')
        for mapping, key in ((scene_splits, scene), (pixel_splits, digest)):
            if key in mapping and mapping[key] != split:
                raise ValueError('scene or image leakage between tuning and acceptance splits')
            mapping[key] = split
        if digest in pixel_scenes and pixel_scenes[digest] != scene:
            raise ValueError('identical image pixels cannot belong to multiple independent scenes')
        pixel_scenes[digest] = scene
        instances = reference.get('instances')
        if not isinstance(instances, list):
            raise ValueError('reference instances must be an array, empty for true negative images')
        complete, partial = [], 0
        for instance in instances:
            if not isinstance(instance, dict) or instance.get('visibility') not in ('complete', 'partial'):
                raise ValueError('instance visibility must be complete or partial')
            corners = _quad(instance.get('corners'), width, height)
            if instance['visibility'] == 'complete':
                complete.append(corners)
            else:
                partial += 1
        candidates = frame.get('candidates')
        if not isinstance(candidates, list) or any(not isinstance(c, dict) for c in candidates):
            raise ValueError('frame candidates must be an array of qualified candidate objects')
        candidates = [_quad(candidate.get('corners'), width, height) for candidate in candidates]
        unresolved = frame.get('unresolved', [])
        if not isinstance(unresolved, list):
            raise ValueError('unresolved must be an array')
        timing = _number(frame.get('timing_ms'), 'frame timing_ms')
        truncated = False
        for key in ('budget_truncated', 'search_truncated'):
            if key in frame:
                if type(frame[key]) is not bool:
                    raise ValueError(key + ' must be boolean')
                truncated = truncated or frame[key]
        if 'search_complete' in frame:
            if type(frame['search_complete']) is not bool:
                raise ValueError('search_complete must be boolean')
            truncated = truncated or not frame['search_complete']
        validated.append(dict(index=index, scene_id=scene, split=split, width=width, height=height,
                              candidates=candidates, complete=complete, partial=partial,
                              negative=not instances, unresolved=len(unresolved), timing_ms=timing,
                              truncated=truncated))
    return validated


def evaluate_frames(frames, references, iou_threshold=.7):
    """Measure only acceptance frames, while validating all split provenance.

    Every report frame must have exactly one manual reference carrying its
    canonical BGR pixel hash and dimensions. Complete references form the
    recall denominator. All qualified candidates form the precision
    denominator, including unmatched candidates on partial/difficult scenes.
    Unresolved outputs never count as matches. Missing detections on truncated
    searches still count as misses; such a run is explicitly incomplete.
    """
    threshold = _number(iou_threshold, 'iou_threshold', maximum=1.)
    if threshold <= 0.:
        raise ValueError('iou_threshold must exceed zero')
    validated = _validate(frames, references)
    selected = [frame for frame in validated if frame['split'] == 'acceptance']
    matched = candidate_count = complete_count = partial_count = 0
    negative_images = false_positive_images = unresolved_count = truncated_count = 0
    corner_errors, normalized_errors, timings, details = [], [], [], []
    positive_scenes, negative_scenes = set(), set()
    for frame in selected:
        matches = _matches(frame['candidates'], frame['complete'], threshold)
        matched += len(matches)
        candidate_count += len(frame['candidates'])
        complete_count += len(frame['complete'])
        partial_count += frame['partial']
        unresolved_count += frame['unresolved']
        truncated_count += int(frame['truncated'])
        timings.append(frame['timing_ms'])
        if frame['complete']:
            positive_scenes.add(frame['scene_id'])
        if frame['negative']:
            negative_images += 1
            false_positive_images += int(bool(frame['candidates']))
            negative_scenes.add(frame['scene_id'])
        for candidate, target, _ in matches:
            raw, normalized = _aligned_errors(frame['candidates'][candidate], frame['complete'][target],
                                              frame['width'], frame['height'])
            corner_errors.extend(raw)
            normalized_errors.extend(normalized)
        details.append(dict(index=frame['index'], scene_id=frame['scene_id'],
                            matched=len(matches), complete_instances=len(frame['complete']),
                            qualified_candidates=len(frame['candidates']),
                            matches=[dict(candidate_index=c, complete_reference_index=t, iou=iou)
                                     for c, t, iou in matches], budget_truncated=frame['truncated']))
    return dict(schema_version=1, status='offline_metrics_only', actionable=False,
                real_world_validated=False, passed=None,
                acceptance_evaluated=bool(selected) and not truncated_count,
                iou_threshold=threshold, evaluated_frames=len(selected),
                tuning_frames_excluded=len(validated) - len(selected),
                qualified_candidates=candidate_count, complete_instances=complete_count,
                partial_instances=partial_count, matched_instances=matched,
                false_positive_candidates=candidate_count - matched,
                missed_complete_instances=complete_count - matched,
                precision=matched / candidate_count if candidate_count else None,
                recall=matched / complete_count if complete_count else None,
                negative_images=negative_images, negative_images_with_false_positives=false_positive_images,
                negative_image_false_positive_rate=(false_positive_images / negative_images
                                                    if negative_images else None),
                unresolved_candidates=unresolved_count, truncated_frames=truncated_count,
                corner_error_px=_summary(corner_errors),
                corner_error_reference_640x480_px=_summary(normalized_errors),
                algorithm_time_ms=_summary(timings),
                independent_positive_scenes=len(positive_scenes),
                independent_negative_scenes=len(negative_scenes),
                sufficient_data=len(positive_scenes) >= 60 and len(negative_scenes) >= 30,
                suggested_targets=dict(precision=.95, recall=.95, negative_image_false_positive_rate=.05,
                                       corner_median_reference_640x480_px=3.,
                                       corner_p95_reference_640x480_px=6., target_computer_algorithm_p95_ms=200.),
                target_runtime_validated=False, frames=details,
                limitations=[
                    'Manual scene membership and annotations are supplied claims, not independently verified.',
                    'Only complete visible reference quadrilaterals enter recall; unresolved true tops remain misses.',
                    'Matched corner errors exclude misses; precision and recall must be read alongside corner errors.',
                    'Suggested targets are provisional; runtime on this computer does not validate the target computer.',
                    'A matched geometric quadrilateral does not establish a brick, upward plane or action permission.',
                ])


def evaluate_report(report, references, iou_threshold=.7):
    """Evaluate a completed workbench replay and retain its run context."""
    if (not isinstance(report, dict) or type(report.get('schema_version')) is not int or
            report['schema_version'] != 1):
        raise ValueError('report must use workbench schema_version 1')
    if report.get('processing_complete') is not True:
        raise ValueError('report processing must be complete before evaluation')
    result = evaluate_frames(report.get('frames'), references, iou_threshold)
    if report.get('search_complete') is False:
        result['acceptance_evaluated'] = False
    result['report_origin'] = report.get('origin', 'unspecified')
    result['report_status'] = report.get('status', 'unspecified')
    configuration_hash = hashlib.sha256(json.dumps(
        report.get('configuration', {}), sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
    stored_hash = report.get('configuration_sha256')
    if stored_hash is not None and stored_hash != configuration_hash:
        raise ValueError('report configuration_sha256 does not match frozen configuration')
    result['configuration_sha256'] = configuration_hash
    result['configuration_hash_verified'] = stored_hash is not None
    return result
