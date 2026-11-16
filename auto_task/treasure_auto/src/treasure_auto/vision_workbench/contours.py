"""Pixel-based quadrilateral diagnostics, independent of competition control.

An observed outer outline is only a candidate. It does not establish a brick,
an upward plane, or an unoccluded suction surface. Inner contours may be print,
holes or occluders; that ambiguity is retained in the diagnostic output.
"""
import math
from types import SimpleNamespace

from treasure_auto.contracts import core as c, models as m
from treasure_auto.perception.geometry import (
    _intersection, _line, cross, overlaps, segment_distance,
)


_DEFAULTS = dict(min_area_px2=500., max_area_fraction=.8,
                 max_line_error_px=2., min_edge_points=8,
                 min_edge_span_px=12., min_corner_sine=.15,
                 approx_epsilon_px=2., roi=None, board_regions=[],
                 canny_low=50., canny_high=150.)


def _number(value, name, minimum=0., maximum=None, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value) or
            value < minimum or (positive and value <= minimum) or
            (maximum is not None and value > maximum)):
        raise ValueError('{} must be a finite number in the permitted range'.format(name))
    return float(value)


def _rectangle(value, name, width, height):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('{} must be [x,y,width,height]'.format(name))
    x, y, w, h = [_number(v, name, positive=(i >= 2)) for i, v in enumerate(value)]
    if x + w > width or y + h > height:
        raise ValueError('{} extends outside the image'.format(name))
    return x, y, w, h


def _configuration(config, width, height):
    if type(config) is not dict:
        raise ValueError('contour config must be an object')
    values = dict(_DEFAULTS)
    values.update(config)
    for name in ('min_area_px2', 'max_line_error_px', 'min_edge_span_px', 'approx_epsilon_px'):
        values[name] = _number(values[name], name, positive=True)
    for name in ('max_area_fraction', 'min_corner_sine'):
        values[name] = _number(values[name], name, maximum=1., positive=True)
    if type(values['min_edge_points']) is not int or values['min_edge_points'] < 3:
        raise ValueError('min_edge_points must be an integer of at least 3')
    for name in ('canny_low', 'canny_high'):
        values[name] = _number(values[name], name, maximum=255.)
    if values['canny_high'] <= values['canny_low']:
        raise ValueError('canny_high must exceed canny_low')
    if values['roi'] is not None:
        values['roi'] = _rectangle(values['roi'], 'roi', width, height)
    if not isinstance(values['board_regions'], (list, tuple)):
        raise ValueError('board_regions must be an array of rectangles')
    values['board_regions'] = [
        _rectangle(r, 'board_regions', width, height) for r in values['board_regions']]
    # Only a raster-scale endpoint allowance: no missing edge is completed.
    values['max_extrapolation_px'] = 2. * values['max_line_error_px'] + 1.
    return SimpleNamespace(**values)


def _rect_polygon(rect):
    x, y, w, h = rect
    return (m.PixelPoint(x, y), m.PixelPoint(x + w, y),
            m.PixelPoint(x + w, y + h), m.PixelPoint(x, y + h))


def _points(contour):
    return tuple(m.PixelPoint(float(p[0][0]), float(p[0][1])) for p in contour)


def _coordinates(points):
    return [[float(p.u), float(p.v)] for p in points]


def _fit(contour, cv2, config):
    approx = cv2.approxPolyDP(contour, config.approx_epsilon_px, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        raise ValueError('outline is not a convex four-edge contour')
    vertices, points = _points(approx), _points(contour)
    edges = [[] for _ in range(4)]
    for point in points:
        distances = [segment_distance(point, vertices[i], vertices[(i + 1) % 4]) for i in range(4)]
        edge_index = min(range(4), key=distances.__getitem__)
        if distances[edge_index] > config.max_line_error_px:
            raise ValueError('outline deviates from four observed edges')
        edges[edge_index].append(point)
    fitted = [_line(tuple(edge), config) for edge in edges]
    corners = [_intersection(fitted[(i - 1) % 4], fitted[i], config)[0] for i in range(4)]
    turns = [cross(corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]) for i in range(4)]
    if not (all(t > 1e-9 for t in turns) or all(t < -1e-9 for t in turns)):
        raise ValueError('fitted edges do not form a convex quadrilateral')
    area2 = sum(a.u * b.v - b.u * a.v for a, b in zip(corners, corners[1:] + corners[:1]))
    if area2 < 0:
        # Original edge i joins vertex i to vertex i+1. Reverse it together
        # with its vertices; retain edge i between published corners i,i+1.
        corners.reverse()
        edges = list(reversed(edges[:-1])) + [edges[-1]]
    start = min(range(4), key=lambda i: (round(corners[i].v, 8), corners[i].u))
    corners = corners[start:] + corners[:start]
    edges = edges[start:] + edges[:start]
    return corners, edges, abs(area2) / 2.


def _internal_boundaries(root, contours, hierarchy, cv2):
    """Ignore the normal paired Canny rim; retain distinct inner outlines."""
    outer = contours[root]
    outer_area = abs(cv2.contourArea(outer))
    x, y, w, h = cv2.boundingRect(outer)
    count = 0
    pending = [int(hierarchy[root][2])]
    while pending:
        index = pending.pop()
        if index < 0:
            continue
        node = hierarchy[index]
        pending.extend([int(node[0]), int(node[2])])
        inner = contours[index]
        ix, iy, iw, ih = cv2.boundingRect(inner)
        paired_rim = (abs(ix - x) <= 2 and abs(iy - y) <= 2 and
                      abs(ix + iw - x - w) <= 2 and abs(iy + ih - y - h) <= 2 and
                      abs(cv2.contourArea(inner)) >= .9 * outer_area)
        if not paired_rim and abs(cv2.contourArea(inner)) >= 4.:
            count += 1
    return count


def detect_candidates(bgr, config):
    """Return JSON-ready, non-actionable external quadrilateral candidates.

    Coordinates refer to the unchanged full BGR image. ROI/board rectangles are
    manual filters, applied after full-image edge extraction; cropping cannot
    supply an artificial fourth edge. No morphological gap closing is used.
    """
    import cv2
    import numpy as np
    if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or
            bgr.shape[2] != 3 or min(bgr.shape[:2]) < 3):
        raise ValueError('image must be a nonempty uint8 BGR array with three channels')
    height, width = bgr.shape[:2]
    cfg = _configuration(config, width, height)
    result = dict(candidates=[], rejected=[],
                  region_basis='manual' if cfg.roi is not None or cfg.board_regions else 'unconfigured')
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), .8)
    edge_image = cv2.Canny(gray, cfg.canny_low, cfg.canny_high, L2gradient=True)
    contours, tree = cv2.findContours(edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if tree is None:
        return result
    hierarchy = tree[0]
    for index, contour in enumerate(contours):
        if hierarchy[index][3] != -1:
            continue  # Canny paired rims and inner print are not separate bodies.
        bbox = list(map(int, cv2.boundingRect(contour)))
        x, y, w, h = bbox
        if w * h < cfg.min_area_px2:
            continue
        try:
            if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
                raise ValueError('frame-truncated outline has no complete four-edge support')
            points = _points(contour)
            if cfg.roi is not None:
                rx, ry, rw, rh = cfg.roi
                if not all(rx <= p.u < rx + rw and ry <= p.v < ry + rh for p in points):
                    raise ValueError('outline crosses or lies outside manual roi')
            corners, edges, area = _fit(contour, cv2, cfg)
            if area < cfg.min_area_px2 or area > cfg.max_area_fraction * width * height:
                raise ValueError('quadrilateral area outside configured bounds')
            if not all(0 <= p.u < width and 0 <= p.v < height for p in corners):
                raise ValueError('fitted intersection outside image')
            if cfg.roi is not None:
                rx, ry, rw, rh = cfg.roi
                if not all(rx <= p.u < rx + rw and ry <= p.v < ry + rh for p in corners):
                    raise ValueError('fitted intersection outside manual roi')
            if any(overlaps(tuple(corners), _rect_polygon(r)) for r in cfg.board_regions):
                raise ValueError('candidate overlaps manually excluded board region')
            internal = _internal_boundaries(index, contours, hierarchy, cv2)
            result['candidates'].append(dict(
                corners=_coordinates(corners), edges=[_coordinates(e) for e in edges],
                contour=_coordinates(points), bbox=bbox, body_confirmed=False,
                plane_basis='unknown', actionable=False, internal_boundaries=internal,
                occlusion_unresolved=bool(internal)))
        except (ValueError, c.ContractError) as error:
            result['rejected'].append(dict(reason=str(error), bbox=bbox))
    result['candidates'].sort(key=lambda item: (item['bbox'][1], item['bbox'][0]))
    return result
