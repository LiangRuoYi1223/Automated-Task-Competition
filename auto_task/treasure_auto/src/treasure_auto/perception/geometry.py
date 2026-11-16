"""Real-edge fitting in full-image pixels. No rectangle/square corner substitution."""
import math
from dataclasses import dataclass
from typing import Tuple
from treasure_auto.contracts import core as c, models as m


def distance(a, b):
    return math.hypot(a.u - b.u, a.v - b.v)


def cross(a, b, p):
    return (b.u - a.u) * (p.v - a.v) - (b.v - a.v) * (p.u - a.u)


def contains(polygon, point, margin=0.):
    """Convex polygon containment; positive margin requires interior clearance."""
    signs = [cross(a, b, point) / max(distance(a, b), 1e-12)
             for a, b in zip(polygon, polygon[1:] + polygon[:1])]
    return all(x >= margin for x in signs) or all(x <= -margin for x in signs)


def require_convex_polygon(polygon, width, height):
    c.require(type(polygon) is tuple and len(polygon) >= 3 and len(set(polygon)) == len(polygon),
              'region', 'distinct ordered convex polygon vertices required')
    c.require(all(type(p) is m.PixelPoint and 0 <= p.u < width and 0 <= p.v < height for p in polygon),
              'region', 'region outside full-image bounds')
    # All other vertices must lie on the same side of every oriented boundary edge.
    signed = [cross(a, b, p) for a, b in zip(polygon, polygon[1:] + polygon[:1])
              for p in polygon if p != a and p != b]
    c.require(all(v > 1e-9 for v in signed) or all(v < -1e-9 for v in signed),
              'region', 'region must be strictly convex without self intersections')


def overlaps(a, b):
    """Separating axis test for convex polygons, including touching boundaries."""
    for poly in (a, b):
        for p, q in zip(poly, poly[1:] + poly[:1]):
            axis = (p.v - q.v, q.u - p.u)
            ap = [v.u * axis[0] + v.v * axis[1] for v in a]
            bp = [v.u * axis[0] + v.v * axis[1] for v in b]
            if max(ap) < min(bp) or max(bp) < min(ap):
                return False
    return True


def segment_distance(point, a, b):
    d2 = distance(a, b) ** 2
    t = max(0., min(1., ((point.u - a.u) * (b.u - a.u) + (point.v - a.v) * (b.v - a.v)) / max(d2, 1e-12)))
    return distance(point, m.PixelPoint(a.u + t * (b.u - a.u), a.v + t * (b.v - a.v)))


@dataclass(frozen=True)
class EdgeFit:
    center: m.PixelPoint
    direction: Tuple[float, float]
    support_min: float
    support_max: float
    residual_px: float
    sample_count: int


@dataclass(frozen=True)
class CornerFit:
    point: m.PixelPoint
    visibility: m.Visibility
    basis: str
    error_px: float
    extrapolation_px: float
    edge_indices: Tuple[int, int]


@dataclass(frozen=True)
class SurfaceFit:
    top: m.TopSurface
    corners: Tuple[CornerFit, ...]
    edges: Tuple[EdgeFit, ...]


def _line(points, config):
    c.require(len(set(points)) >= config.min_edge_points, 'edge', 'insufficient distinct observed edge points')
    x, y = sum(p.u for p in points) / len(points), sum(p.v for p in points) / len(points)
    xx = sum((p.u - x) ** 2 for p in points)
    yy = sum((p.v - y) ** 2 for p in points)
    xy = sum((p.u - x) * (p.v - y) for p in points)
    theta = .5 * math.atan2(2 * xy, xx - yy)
    dx, dy = math.cos(theta), math.sin(theta)
    projections = [(p.u - x) * dx + (p.v - y) * dy for p in points]
    residual = max(abs((p.u - x) * dy - (p.v - y) * dx) for p in points)
    c.require(max(projections) - min(projections) >= config.min_edge_span_px, 'edge', 'edge support too short')
    c.require(residual <= config.max_line_error_px, 'edge', 'contour inconsistent with straight edge')
    return EdgeFit(m.PixelPoint(x, y), (dx, dy), min(projections), max(projections), residual, len(points))


def _intersection(a, b, config):
    ax, ay = a.direction; bx, by = b.direction
    determinant = ax * by - ay * bx
    c.require(abs(determinant) >= config.min_corner_sine, 'corner', 'parallel or ill-conditioned adjacent edges')
    qx, qy = b.center.u - a.center.u, b.center.v - a.center.v
    t = (qx * by - qy * bx) / determinant
    point = m.PixelPoint(a.center.u + t * ax, a.center.v + t * ay)
    extents = []
    for line in (a, b):
        t = ((point.u - line.center.u) * line.direction[0] +
             (point.v - line.center.v) * line.direction[1])
        extents.append(max(line.support_min - t, t - line.support_max, 0.))
    extra = max(extents)
    c.require(extra <= config.max_extrapolation_px, 'corner', 'unconstrained occlusion/extrapolation')
    # Quantization plus residual amplification. Diagnostic bound, not calibrated metric accuracy.
    error = (1. + a.residual_px + b.residual_px) / abs(determinant)
    error += extra * error / min(a.support_max - a.support_min, b.support_max - b.support_min)
    return point, extra, error


def fit_top(image, contour, edges, config, plane_basis, height_mm, evidence):
    c.require(len(edges) == 4, 'edges', 'exactly four independently supported real edges required')
    c.require(plane_basis is not m.PlaneBasis.UNKNOWN and type(plane_basis) is m.PlaneBasis,
              'plane_basis', 'layer/plane basis unresolved')
    # This is plane z in the calibrated base frame, whose origin may be above
    # the ground. A measured negative z is valid; a missing/nonfinite z is not.
    c.require(type(height_mm) in (int, float) and math.isfinite(height_mm),
              'height_mm', 'explicit finite plane height required, not a default first layer')
    c.require(bool(evidence), 'evidence', 'observed edges require evidence', c.ErrorCode.EVIDENCE_MISSING)
    c.require(image.distortion is not m.Distortion.UNKNOWN, 'distortion', 'pixel space unresolved')
    points = tuple(contour) + tuple(p for edge in edges for p in edge)
    c.require(all(type(p) is m.PixelPoint and 0 <= p.u < image.width and 0 <= p.v < image.height for p in points),
              'points', 'real edges must be full-image coordinates within bounds')
    fitted = tuple(_line(edge, config) for edge in edges)
    corners = []
    for i in range(4):
        point, extra, error = _intersection(fitted[(i - 1) % 4], fitted[i], config)
        corners.append(CornerFit(point, m.Visibility.VISIBLE if extra <= 1e-6 else m.Visibility.PARTIAL,
                                 'intersection_of_observed_edges', error, extra, ((i - 1) % 4, i)))
    poly = tuple(x.point for x in corners)
    turns = [cross(poly[i], poly[(i + 1) % 4], poly[(i + 2) % 4]) for i in range(4)]
    c.require(all(t > 1e-9 for t in turns) or all(t < -1e-9 for t in turns), 'corners', 'nonconvex/crossed edge order')
    area2 = sum(a.u * b.v - b.u * a.v for a, b in zip(poly, poly[1:] + poly[:1]))
    c.require(abs(area2) / 2 >= config.min_area_px2, 'corners', 'top area below configured support')
    c.require(len(contour) >= 4, 'contour', 'visible boundary required')
    c.require(all(min(segment_distance(p, a, b) for a, b in zip(poly, poly[1:] + poly[:1])) <=
                  config.max_line_error_px + 1e-6 for p in contour),
              'contour', 'shadow/occluder/side-face boundary is not consistent with four true edges')
    if area2 < 0:
        corners.reverse()
    start = min(range(4), key=lambda i: (round(corners[i].point.v, 8), corners[i].point.u))
    corners = corners[start:] + corners[:start]
    visibility = m.Visibility.VISIBLE if all(x.visibility is m.Visibility.VISIBLE for x in corners) else m.Visibility.PARTIAL
    top = m.TopSurface(tuple(x.point for x in corners), tuple(contour), image.width, image.height,
                       image.camera_frame, image.distortion, max(x.error_px for x in corners),
                       plane_basis, height_mm, visibility, tuple(evidence))
    return SurfaceFit(top, tuple(corners), fitted)


def top_from_mask(mask, image, config, plane_basis, height_mm, evidence):
    """Extract from a segmented SINGLE entity top. Segmentation/body proof are separate inputs.

    OpenCV findContours/approxPolyDP establish four edge hypotheses, then all supported
    samples are line-fitted. This never calls minAreaRect or invents square corners.
    """
    import cv2
    import numpy as np
    c.require(isinstance(mask, np.ndarray) and mask.dtype == np.uint8 and
              mask.shape == (image.height, image.width), 'mask', 'full-image uint8 mask required')
    c.require(bool(np.all((mask == 0) | (mask == 255))), 'mask', 'binary segmentation required')
    c.require(not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
              'mask', 'clipped top has no complete edge constraint')
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    c.require(len(contours) == 1, 'mask', 'multiple entities or holes/occlusion require separate edge evidence')
    contour = contours[0]
    approx = cv2.approxPolyDP(contour, config.max_line_error_px, True)
    c.require(len(approx) == 4 and cv2.isContourConvex(approx), 'mask', 'not a four-edge convex top')
    vertices = tuple(m.PixelPoint(float(p[0][0]), float(p[0][1])) for p in approx)
    points = tuple(m.PixelPoint(float(p[0][0]), float(p[0][1])) for p in contour)
    groups = [[] for _ in range(4)]
    for p in points:
        i = min(range(4), key=lambda i: segment_distance(p, vertices[i], vertices[(i + 1) % 4]))
        groups[i].append(p)
    return fit_top(image, points, tuple(tuple(g) for g in groups), config, plane_basis, height_mm, evidence)
