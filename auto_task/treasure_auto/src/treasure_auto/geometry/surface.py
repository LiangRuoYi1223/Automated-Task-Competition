"""Horizontal-plane reconstruction, true diagonals and fixed-orientation tool geometry."""
import math
from dataclasses import dataclass, replace
from typing import Tuple
from treasure_auto.contracts import core as c, models as m
from . import planar
from .inputs import PlaneSolution
from .math3d import finite, ray_on_plane, correct_pixel, apply, validate_transform, square_yaw


@dataclass(frozen=True)
class KnownLayerEvidence:
    """Measured ground datum and currently verified horizontal layer in base units.

    The provider must revalidate support and identity for the current image. Layer
    number alone does not establish horizontal contact after a push or collapse.
    """
    solution_id: str
    base_frame: str
    ground_height_mm: float
    cube_edge_mm: float
    confirmed_layer: int
    ground_error_mm: float
    cube_error_mm: float
    horizontal: c.Fact
    support_unchanged: c.Fact
    identity_consistent: c.Fact
    support_ids: Tuple[str, ...]
    evidence: Tuple[c.EvidenceRef, ...]


def known_layer_plane(layer: KnownLayerEvidence, base_frame: str) -> PlaneSolution:
    """Return a P06 plane; never assume a cube edge or a zero base-ground datum."""
    c.require(type(layer) is KnownLayerEvidence, 'known_layer', 'typed layer evidence required')
    c.require(type(base_frame) is str and bool(base_frame.strip()) and layer.base_frame == base_frame,
              'known_layer.base_frame', 'explicit matching base frame required', c.ErrorCode.CONFLICT)
    c.require(type(layer.solution_id) is str and bool(layer.solution_id.strip()),
              'known_layer.solution_id', 'solution identity required')
    ground = finite(layer.ground_height_mm, 'known_layer.ground_height_mm')
    edge = finite(layer.cube_edge_mm, 'known_layer.cube_edge_mm', 0.)
    c.require(edge > 0 and type(layer.confirmed_layer) is int and layer.confirmed_layer >= 1,
              'known_layer.confirmed_layer', 'positive measured edge and confirmed layer required')
    ground_error = finite(layer.ground_error_mm, 'known_layer.ground_error_mm', 0.)
    edge_error = finite(layer.cube_error_mm, 'known_layer.cube_error_mm', 0.)
    c.require(all(value is c.Fact.CONFIRMED for value in
                  (layer.horizontal, layer.support_unchanged, layer.identity_consistent)),
              'known_layer.support', 'tilt, changed support or unresolved identity requires review', c.ErrorCode.CONFLICT)
    c.require(type(layer.support_ids) is tuple and bool(layer.support_ids) and
              all(type(value) is str and bool(value.strip()) for value in layer.support_ids) and
              len(set(layer.support_ids)) == len(layer.support_ids),
              'known_layer.support_ids', 'explicit ground or lower support identities required')
    c.require(type(layer.evidence) is tuple and bool(layer.evidence) and
              all(type(value) is c.EvidenceRef for value in layer.evidence),
              'known_layer.evidence', 'current physical evidence required', c.ErrorCode.EVIDENCE_MISSING)
    height = finite(ground + layer.confirmed_layer * edge, 'known_layer.height_mm')
    error = finite(ground_error + layer.confirmed_layer * edge_error, 'known_layer.error_mm', 0.)
    return PlaneSolution(layer.solution_id, height, error, (0., 0., 1.), m.PlaneBasis.KNOWN_DIMENSIONS,
                         layer.support_ids, layer.evidence)


def resolve_plane(planes,config):
    c.require(bool(planes),'plane','current plane unknown',c.ErrorCode.EVIDENCE_MISSING)
    first=planes[0]
    for p in planes:
        finite(p.height_mm,'plane.height_mm');finite(p.error_mm,'plane.error_mm',0.)
        c.require(p.basis is not m.PlaneBasis.UNKNOWN and type(p.basis) is m.PlaneBasis and bool(p.evidence),
                  'plane.basis','missing independent layer evidence',c.ErrorCode.EVIDENCE_MISSING)
        c.require(p.normal==(0.,0.,1.),'plane.normal','only verified horizontal planes supported; tilt/normal unknown')
        c.require(p.solution_id==first.solution_id and p.support_ids==first.support_ids,'plane','multiple unresolved plane hypotheses',c.ErrorCode.CONFLICT)
        c.require(abs(p.height_mm-first.height_mm)<=config.plane_tolerance_mm,'plane.height','conflicting layer evidence',c.ErrorCode.CONFLICT)
    return replace(first,error_mm=max(abs(p.height_mm-first.height_mm)+p.error_mm for p in planes),
                   evidence=tuple(dict.fromkeys(e for p in planes for e in p.evidence)))


def _add(a,b):return a[0]+b[0],a[1]+b[1]
def _sub(a,b):return a[0]-b[1],a[1]-b[0]
def _mul(a,b):
    v=[x*y for x in a for y in b];return min(v),max(v)
def _div(a,b):
    c.require(b[0]*b[1]>0,'corners','diagonal intersection uncertain/parallel')
    v=[x/y for x in a for y in b];return min(v),max(v)
def _cross(a,b):return _sub(_mul(a[0],b[1]),_mul(a[1],b[0]))


def diagonal_center(points,error):
    planar.convex(points);finite(error,'corner_error',0.)
    a,b,d,e=points
    r=(d[0]-a[0],d[1]-a[1]);s=(e[0]-b[0],e[1]-b[1]);q=(b[0]-a[0],b[1]-a[1])
    det=r[0]*s[1]-r[1]*s[0]
    c.require(abs(det)>1e-12,'corners','parallel diagonals')
    t=(q[0]*s[1]-q[1]*s[0])/det;center=(a[0]+t*r[0],a[1]+t*r[1])
    boxes=tuple(tuple((v-error,v+error) for v in p) for p in points)
    aa,bb,dd,ee=boxes
    rr=tuple(_sub(dd[i],aa[i]) for i in range(2));ss=tuple(_sub(ee[i],bb[i]) for i in range(2));qq=tuple(_sub(bb[i],aa[i]) for i in range(2))
    tt=_div(_cross(qq,ss),_cross(rr,ss))
    bounds=[_add(aa[i],_mul(tt,rr[i])) for i in range(2)]
    return center,math.sqrt(sum(max(abs(x-center[i]) for x in bounds[i])**2 for i in range(2)))


@dataclass(frozen=True)
class SurfaceGeometry:
    corners: tuple
    center: c.ArmPoint
    yaw_rad: float
    yaw_error_rad: float
    normal: tuple


def solve_top(top,calibration,camera_to_base,base_to_arm,plane,history,config,calibration_error_mm):
    config.require_ready(top.evidence[0].origin if top.evidence else c.Origin.SIMULATED)
    validate_transform(camera_to_base,calibration.camera_frame,config.base_frame)
    validate_transform(base_to_arm,config.base_frame,config.arm_frame)
    c.require((top.image_width,top.image_height)==config.image_size and top.frame_id==calibration.camera_frame,
              'image','resolution/camera mismatch')
    c.require(top.height_mm is not None and top.plane_basis is not m.PlaneBasis.UNKNOWN and
              abs(top.height_mm-plane.height_mm)<=config.plane_tolerance_mm,'height_mm','current top plane unknown/conflicting')
    c.require(top.distortion is calibration.distortion,'distortion','pixel/calibration spaces differ')
    c.require(history.coordinate_order=='uv' and history.pixel_space=='full','pixels','full-image u=column v=row required')
    identity=m.PixelTransform(0.,0.,1.,1.,0.,0.)
    c.require(history.roi_restorations==(0 if history.preprocessing==identity else 1),
              'pixels.roi_restorations','preprocessing must be undone exactly once')
    expected_passes=1 if history.input_distortion is m.Distortion.RAW and top.distortion is m.Distortion.CORRECTED else 0
    c.require(history.input_distortion is not m.Distortion.UNKNOWN and history.distortion_passes==expected_passes and
              not (history.input_distortion is m.Distortion.CORRECTED and top.distortion is m.Distortion.RAW),
              'pixels.distortion_passes','double or inconsistent distortion correction')
    c.require(top.error_px<=config.max_pixel_error_px,'error_px','pixel uncertainty outside calibrated domain')
    finite(calibration_error_mm,'calibration_error_mm',0.)
    gain=1.
    if top.distortion is m.Distortion.RAW:
        c.require(history.inverse_error_gain is not None,'distortion.error','measured inverse error gain missing',c.ErrorCode.UNCONFIGURED)
        gain=finite(history.inverse_error_gain,'inverse_error_gain',1.)
    pixels=tuple(correct_pixel(p,calibration) for p in top.corners)
    center_px,center_error=diagonal_center(tuple((p.u,p.v) for p in pixels),top.error_px*gain)
    cal=replace(calibration,distortion=m.Distortion.CORRECTED)
    def project(pixel,error):
        xyz,uncertainty=ray_on_plane(pixel,cal,camera_to_base,plane.height_mm,error,plane.error_mm)
        xyz=apply(base_to_arm,xyz)
        return c.ArmPoint(config.arm_frame,*xyz,'mm',uncertainty+calibration_error_mm)
    corners=tuple(project(p,top.error_px*gain) for p in pixels)
    center=project(m.PixelPoint(*center_px),center_error)
    polygon=planar.convex(tuple((p.x,p.y) for p in corners))
    c.require(center.error_mm<=config.max_error_mm,'error_mm','center error exceeds validated contact margin')
    c.require(planar.clearance(polygon,(center.x,center.y))>config.cup_radius_mm+center.error_mm+max(p.error_mm for p in corners),
              'cup.coverage','insufficient real top clearance')
    lengths=[math.hypot(a.x-b.x,a.y-b.y) for a,b in zip(corners,corners[1:]+corners[:1])]
    for length in lengths:
        c.require(abs(length-config.brick_size_mm[0])<=2*max(p.error_mm for p in corners)+config.plane_tolerance_mm,
                  'top.dimensions','physical top dimensions inconsistent; possible swapped pixels/wrong layer')
    a,b=corners[:2];yaw=square_yaw(math.atan2(b.y-a.y,b.x-a.x))
    uncertainty=min(1.,2*max(p.error_mm for p in corners)/min(lengths))
    return SurfaceGeometry(corners,center,yaw,math.asin(uncertainty),(0.,0.,1.))


def tool_target(center,tool_transform,held_offset,yaw_rad):
    validate_transform(tool_transform,'cup','arm_reference');finite(yaw_rad,'yaw_rad')
    if held_offset is not None:c.require(held_offset.frame_id=='cup','held_offset.frame','offset must be expressed in cup frame')
    offset=(0.,0.,0.) if held_offset is None else (held_offset.x,held_offset.y,held_offset.z)
    x,y,z=apply(tool_transform,offset);co,si=math.cos(yaw_rad),math.sin(yaw_rad)
    return c.ArmPoint(center.frame_id,center.x-(co*x-si*y),center.y-(si*x+co*y),center.z-z,'mm',
                      center.error_mm+(0. if held_offset is None else held_offset.error_mm))
