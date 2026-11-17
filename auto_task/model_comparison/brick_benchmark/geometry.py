"""Mask-only quadrilateral estimation, never the old full-image edge detector."""
import numpy as np


def extract_quad(mask, epsilon_fraction=.01, max_mask_quad_error=.10, max_edge_error_px=2.):
    import cv2
    import math
    for name, value, maximum in (('epsilon_fraction', epsilon_fraction, .1),
                                  ('max_mask_quad_error', max_mask_quad_error, 1.),
                                  ('max_edge_error_px', max_edge_error_px, 100.)):
        if isinstance(value, bool) or not math.isfinite(float(value)) or not 0 < value <= maximum:
            raise ValueError(name + ' must be finite, positive and within its permitted range')
    binary = np.asarray(mask, np.uint8)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 'empty_mask'
    hierarchy = hierarchy[0]
    outer = [i for i, item in enumerate(hierarchy) if item[3] == -1]
    if len(outer) != 1:
        return None, 'disconnected_mask'
    index = outer[0]
    if hierarchy[index][2] != -1:
        return None, 'mask_has_holes'
    contour = contours[index]
    perimeter = cv2.arcLength(contour, True)
    quad = cv2.approxPolyDP(contour, epsilon_fraction * perimeter, True)
    if len(quad) != 4 or not cv2.isContourConvex(quad) or cv2.contourArea(quad) <= 1:
        return None, 'boundary_is_not_a_supported_quad'
    reconstructed = np.zeros_like(binary)
    cv2.fillConvexPoly(reconstructed, quad, 1)
    union = np.logical_or(binary, reconstructed).sum()
    mismatch = np.logical_xor(binary, reconstructed).sum() / max(1, union)
    if mismatch > max_mask_quad_error:
        return None, 'quad_does_not_follow_mask'
    points = quad.reshape(4, 2).astype(float)
    # Relative area alone can hide a small but important cut corner in a large mask.
    # Every observed boundary point must lie close to one of the four finite edges.
    observed = contour.reshape(-1, 2).astype(float)
    vectors = np.roll(points, -1, axis=0) - points
    delta = observed[:, None, :] - points[None, :, :]
    projection = np.sum(delta * vectors[None, :, :], axis=2) / np.sum(vectors * vectors, axis=1)
    nearest = points[None, :, :] + np.clip(projection, 0., 1.)[:, :, None] * vectors[None, :, :]
    distances = np.linalg.norm(observed[:, None, :] - nearest, axis=2)
    if float(distances.min(axis=1).max()) > max_edge_error_px:
        return None, 'observed_boundary_departs_from_four_edges'
    a, b = points[1] - points[0], points[2] - points[1]
    area = a[0] * b[1] - a[1] * b[0]
    if area < 0:
        points = points[::-1]
    start = min(range(4), key=lambda i: (points[i, 1], points[i, 0]))
    return np.roll(points, -start, axis=0).tolist(), None


def add_geometry(instances, config=None):
    config = config or {}
    for instance in instances:
        if instance.corners is None:
            instance.corners, reason = extract_quad(instance.mask, **config)
            instance.details['geometry_reason'] = reason
        instance.details['geometry_basis'] = 'predicted_mask_pixels'
        instance.details['physical_pose_validated'] = False
    return instances
