"""Threshold metrics; these are explicitly NOT COCO AP/mAP."""
import math
import numpy as np
from .contracts import CLASSES, validate_instance

DEFAULT_POLICY = {'iou_threshold': .5, 'corner_tolerance_fraction': .01,
                  'weights': {'geometry_f1': .35, 'class_f1': .25,
                              'mask_quality': .25, 'corner_recall': .15}}


def _divide(a, b):
    return a / b if b else 0.


def _f1(tp, predicted, truth):
    return _divide(2. * tp, predicted + truth)


def _match(truth, predicted, threshold):
    scores = [[_divide(np.logical_and(p.mask, g.mask).sum(),
                       np.logical_or(p.mask, g.mask).sum()) for g in truth] for p in predicted]
    if not truth or not predicted:
        return []
    # Integer lexicographic objective: cardinality, summed IoU (1e-6), then
    # confidence-weighted IoU. The last tie-break avoids duplicate detections
    # displacing the stronger prediction's best overlap. Class is never used.
    rows, real_columns = len(predicted), len(truth)
    columns = real_columns + rows  # individual unmatched dummy columns
    units = 10**6
    big = (min(rows, real_columns) + 1) * units
    costs = []
    for p, overlaps in enumerate(scores):
        row = []
        for iou in overlaps:
            reward = (big*big + round(float(iou)*units)*big +
                      round(float(iou)*predicted[p].score*units)) if iou >= threshold else 0
            row.append(-reward)
        costs.append(row + [0] * rows)
    # Rectangular Hungarian assignment; Python integers retain the objective's
    # priority exactly, avoiding a new SciPy dependency in the baseline env.
    u, v, owner, way = [0]*(rows+1), [0]*(columns+1), [0]*(columns+1), [0]*(columns+1)
    for i in range(1, rows+1):
        owner[0], j0 = i, 0
        minimum, used = [float('inf')]*(columns+1), [False]*(columns+1)
        while True:
            used[j0] = True
            i0, delta, j1 = owner[j0], float('inf'), 0
            for j in range(1, columns+1):
                if not used[j]:
                    current = costs[i0-1][j-1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(columns+1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while True:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(g, owner[g+1]-1, float(scores[owner[g+1]-1][g]))
            for g in range(real_columns) if owner[g+1] and scores[owner[g+1]-1][g] >= threshold]


def corner_errors(predicted, reference):
    p, g = np.asarray(predicted, float), np.asarray(reference, float)
    alternatives = [np.roll(p, i, axis=0) for i in range(4)]
    alternatives += [np.roll(p[::-1], i, axis=0) for i in range(4)]
    return min((np.linalg.norm(a - g, axis=1) for a in alternatives), key=lambda e: float((e*e).sum()))


def evaluate_frames(frames, policy=None):
    policy = policy or DEFAULT_POLICY
    threshold = float(policy['iou_threshold'])
    tolerance = float(policy['corner_tolerance_fraction'])
    weights = policy['weights']
    if (not math.isfinite(threshold) or not 0 < threshold <= 1 or
            not math.isfinite(tolerance) or tolerance <= 0 or
            set(weights) != set(DEFAULT_POLICY['weights']) or
            any(not math.isfinite(float(v)) or v < 0 for v in weights.values()) or
            not math.isclose(sum(weights.values()), 1.)):
        raise ValueError('invalid frozen scoring policy')
    if not frames:
        raise ValueError('cannot score an empty dataset')
    total_gt = total_pred = tp = correct = corner_refs = corner_success = corner_matched = 0
    negative = false_negative_frames = 0
    mask_sum = 0.
    errors, latencies = [], []
    classes = {name: {'truth': 0, 'predicted': 0, 'correct': 0} for name in CLASSES}
    for truth, predicted, elapsed, width, height in frames:
        if not math.isfinite(float(elapsed)) or elapsed < 0:
            raise ValueError('invalid measured pipeline latency')
        latencies.append(float(elapsed))
        for instance in truth + predicted:
            validate_instance(instance, height, width)
        total_gt += len(truth)
        total_pred += len(predicted)
        corner_refs += sum(g.corners is not None for g in truth)
        if not truth:
            negative += 1
            false_negative_frames += bool(predicted)
        for g in truth:
            classes[g.category]['truth'] += 1
        for p in predicted:
            if p.category in classes:
                classes[p.category]['predicted'] += 1
        matches = _match(truth, predicted, threshold)
        tp += len(matches)
        for gi, pi, iou in matches:
            g, p = truth[gi], predicted[pi]
            mask_sum += iou
            if g.category == p.category:
                correct += 1
                classes[g.category]['correct'] += 1
            if g.corners is not None and p.corners is not None:
                values = corner_errors(p.corners, g.corners)
                errors.extend(float(v) for v in values)
                corner_matched += 1
                corner_success += bool(values.max() <= tolerance * math.hypot(width, height))
    per_class = {name: dict(values, f1=_f1(values['correct'], values['predicted'], values['truth']))
                 for name, values in classes.items()}
    components = {'geometry_f1': _f1(tp, total_pred, total_gt),
                  'class_f1': _f1(correct, total_pred, total_gt),
                  'mask_quality': _divide(mask_sum, total_gt),
                  'corner_recall': _divide(corner_success, corner_refs)}
    return {'counts': {'images': len(frames), 'truth': total_gt, 'predicted': total_pred,
                       'tp': tp, 'fp': total_pred - tp, 'fn': total_gt - tp},
            'geometry': {'precision': _divide(tp, total_pred), 'recall': _divide(tp, total_gt),
                         'f1': components['geometry_f1']},
            'segmentation': {'mean_matched_iou': _divide(mask_sum, tp),
                             'quality_per_truth': components['mask_quality']},
            'classification': {'correct': correct, 'f1': components['class_f1'],
                               'accuracy_on_matched': _divide(correct, tp), 'per_class': per_class},
            'corners': {'reference_count': corner_refs, 'matched_with_corners': corner_matched,
                        'success_recall': components['corner_recall'],
                        'error_px_median': float(np.median(errors)) if errors else None,
                        'error_px_p95': float(np.percentile(errors, 95)) if errors else None},
            'negative_frames': {'count': negative, 'false_positive_frames': false_negative_frames},
            'latency_ms': {'mean': float(np.mean(latencies)), 'p95': float(np.percentile(latencies, 95))},
            'components': components,
            'score': 100. * sum(weights[k] * v for k, v in components.items()),
            'geometry_score': 100. * (.5*components['geometry_f1'] + .3*components['mask_quality'] +
                                       .2*components['corner_recall']),
            'selection_ready': bool(total_gt and corner_refs and all(v['truth'] for v in classes.values())),
            'policy': policy,
            'limitations': ['threshold metrics are not COCO mAP',
                'unmatched/missing corners remain in recall denominators',
                'pixel errors do not establish calibrated millimeter accuracy']}
