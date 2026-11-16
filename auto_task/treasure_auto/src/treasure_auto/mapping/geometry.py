"""Conservative 2D footprint checks; acceptance is not a path or action permit."""
import math
from treasure_auto.contracts import core as c, models as m

EPS = 1e-10


def finite(value, field, minimum=0):
    c.require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
              field, 'finite number above lower bound required')


def reason(error):
    return c.Reason(error.code, error.field, error.detail)


def angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def cross(a, b, p):
    return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)


def edges(polygon):
    return zip(polygon, polygon[1:] + polygon[:1])


def point_segment(p, a, b):
    dx, dy = b.x - a.x, b.y - a.y
    length2 = dx * dx + dy * dy
    t = 0 if length2 == 0 else max(0, min(1, ((p.x-a.x)*dx + (p.y-a.y)*dy) / length2))
    return math.hypot(p.x-a.x-t*dx, p.y-a.y-t*dy)


def intersects(a, b, p, q):
    if min(point_segment(a, p, q), point_segment(b, p, q),
           point_segment(p, a, b), point_segment(q, a, b)) <= EPS:
        return True
    return cross(a, b, p) * cross(a, b, q) < 0 and cross(p, q, a) * cross(p, q, b) < 0


def validate_polygon(polygon, field='polygon'):
    c.require(type(polygon) is tuple and len(polygon) >= 3 and
              all(type(p) is m.Point2 for p in polygon), field, 'immutable polygon of Point2 required')
    c.require(len(set(polygon)) == len(polygon), field, 'duplicate vertex')
    area2 = sum(a.x*b.y - a.y*b.x for a, b in edges(polygon))
    c.require(abs(area2) > EPS, field, 'degenerate polygon')
    segments = list(edges(polygon))
    for i, (a, b) in enumerate(segments):
        for j in range(i + 1, len(segments)):
            if j == i + 1 or (i == 0 and j == len(segments) - 1):
                continue
            c.require(not intersects(a, b, *segments[j]), field, 'self-intersecting polygon')


def inside(p, polygon):
    if any(point_segment(p, a, b) <= EPS for a, b in edges(polygon)):
        return True
    hit = False
    for a, b in edges(polygon):
        if (a.y > p.y) != (b.y > p.y) and p.x < (b.x-a.x)*(p.y-a.y)/(b.y-a.y)+a.x:
            hit = not hit
    return hit


def edge_distance(first, second):
    best = float('inf')
    for a, b in edges(first):
        for p, q in edges(second):
            if intersects(a, b, p, q):
                return 0
            best = min(best, point_segment(a, p, q), point_segment(b, p, q),
                       point_segment(p, a, b), point_segment(q, a, b))
    return best


def contained(inner, outer, margin=0):
    return all(inside(p, outer) for p in inner) and edge_distance(inner, outer) > margin + EPS


def covered(inner, outer):
    """Inclusive polygon containment, splitting edges at every boundary crossing."""
    if not all(inside(p,outer) for p in inner):
        return False
    for a,b in edges(inner):
        cuts=[0.0,1.0]
        dx,dy=b.x-a.x,b.y-a.y
        for p,q in edges(outer):
            ex,ey=q.x-p.x,q.y-p.y
            determinant=dx*ey-dy*ex
            if abs(determinant) > EPS:
                t=((p.x-a.x)*ey-(p.y-a.y)*ex)/determinant
                u=((p.x-a.x)*dy-(p.y-a.y)*dx)/determinant
                if 0 <= t <= 1 and 0 <= u <= 1:
                    cuts.append(t)
        cuts=sorted(set(cuts))
        if any(not inside(m.Point2(a.x+dx*(t+u)/2,a.y+dy*(t+u)/2),outer) for t,u in zip(cuts,cuts[1:])):
            return False
    return True


def collides(first, second, margin):
    return (inside(first[0], second) or inside(second[0], first) or
            edge_distance(first, second) <= margin + EPS)


def transform_polygon(polygon, pose):
    cs, sn = math.cos(pose.yaw), math.sin(pose.yaw)
    return tuple(m.Point2(pose.x + cs*p.x - sn*p.y, pose.y + sn*p.x + cs*p.y) for p in polygon)


def check_parking(bundle, pose, body, space, holding, carried, margin_m=0):
    """Pure geometry only. Caller must check freshness, purpose and measurement.

    Every occupancy, including UNKNOWN and RESERVED, blocks its 2D projection.
    A negative result never removes or mutates a shared occupancy.
    """
    try:
        finite(margin_m, 'margin_m')
        c.require(type(pose) is c.MapPose and pose.frame_id == bundle.metadata.origin.frame_id,
                  'pose.frame_id', 'map frame required')
        c.require(space.envelope.versions.map == bundle.metadata.map_version,
                  'space.map', 'different map version', c.ErrorCode.VERSION_MISMATCH)
        c.require(holding in (c.Holding.EMPTY, c.Holding.HELD), 'holding',
                  'unresolved holding cannot select a safe footprint', c.ErrorCode.UNAVAILABLE)
        c.require(holding is not c.Holding.HELD or carried is not None, 'carried',
                  'held object requires a calibrated current transport envelope', c.ErrorCode.UNCONFIGURED)
        shapes = (body,) if holding is c.Holding.EMPTY else (body, carried)
        for shape in shapes:
            validate_polygon(shape, 'footprint')
            radius = max(math.hypot(p.x, p.y) for p in shape)
            margin = (margin_m + pose.error_m + bundle.metadata.origin.error_m +
                      2*radius*math.sin(min(math.pi, pose.error_rad) / 2))
            map_radius=math.hypot(bundle.raster.width,bundle.raster.height)*bundle.metadata.resolution_m_per_cell
            margin+=2*map_radius*math.sin(min(math.pi,bundle.metadata.origin.error_rad)/2)
            world = transform_polygon(shape, pose)
            c.require(contained(world, bundle.boundary, margin), 'boundary',
                      'complete footprint or uncertainty crosses field boundary', c.ErrorCode.CONFLICT)
            # Unmapped space is not free, even if it is inside the surveyed field.
            grid = tuple(bundle.grid_corner(col, row) for col, row in
                         ((0,0), (bundle.raster.width,0),
                          (bundle.raster.width,bundle.raster.height), (0,bundle.raster.height)))
            c.require(contained(world, grid, margin), 'grid', 'footprint extends beyond map coverage',
                      c.ErrorCode.OCCUPANCY_UNRESOLVED)
            for row in range(bundle.raster.height):
                for col in range(bundle.raster.width):
                    occupancy = (255-bundle.raster.pixels[row*bundle.raster.width+col])/255
                    if occupancy < bundle.free_threshold:
                        continue
                    cell = tuple(bundle.grid_corner(x, y) for x, y in
                                 ((col,row), (col+1,row), (col+1,row+1), (col,row+1)))
                    c.require(not collides(world, cell, margin), 'grid',
                              'occupied or unknown static cell', c.ErrorCode.OCCUPANCY_UNRESOLVED)
            for item in space.occupancies:
                c.require(item.frame_id == pose.frame_id, 'occupancy.frame_id', 'wrong occupancy frame')
                validate_polygon(item.footprint_m, 'occupancy.footprint_m')
                c.require(not collides(world, item.footprint_m, margin + item.error_m),
                          'occupancies.' + item.target_id, 'actual, reserved or unknown occupancy intersects',
                          c.ErrorCode.OCCUPANCY_UNRESOLVED)
        return m.ContractCheck(True, None)
    except c.ContractError as error:
        return m.ContractCheck(False, reason(error))
