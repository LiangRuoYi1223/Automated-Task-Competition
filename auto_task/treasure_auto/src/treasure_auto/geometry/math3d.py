"""Rigid transforms in mm; the two map boundary functions alone convert m/mm."""
import math
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m


def finite(value, field, minimum=None):
    c.require(type(value) in (int, float) and math.isfinite(value), field, 'finite number required')
    c.require(minimum is None or value >= minimum, field, 'below permitted minimum')
    return value


def validate_transform(t, source=None, target=None):
    c.require(type(t) is m.RigidTransform, 'transform', 'C0 rigid transform required')
    c.require((source is None or t.source_frame == source) and (target is None or t.target_frame == target),
              'transform.frames', 'wrong matrix direction', c.ErrorCode.INVALID_FIELD)
    a=t.matrix
    c.require(all(abs(x-y)<1e-9 for x,y in zip(a[12:],(0.,0.,0.,1.))), 'transform', 'invalid homogeneous row')
    rows=[a[i:i+3] for i in (0,4,8)]
    for i in range(3):
        for j in range(3):
            c.require(abs(sum(rows[i][k]*rows[j][k] for k in range(3))-(1. if i==j else 0.))<1e-8,
                      'transform.rotation', 'rotation must be orthonormal, not scale/shear')
    det=(a[0]*(a[5]*a[10]-a[6]*a[9])-a[1]*(a[4]*a[10]-a[6]*a[8])+a[2]*(a[4]*a[9]-a[5]*a[8]))
    c.require(abs(det-1.)<1e-8,'transform.rotation','reflection/singular rotation')


def apply(t, xyz):
    validate_transform(t)
    c.require(len(xyz)==3,'point','three coordinates required')
    for x in xyz: finite(x,'point')
    return tuple(sum(t.matrix[4*i+j]*xyz[j] for j in range(3))+t.matrix[4*i+3] for i in range(3))


def rotate(t, xyz):
    validate_transform(t)
    return tuple(sum(t.matrix[4*i+j]*xyz[j] for j in range(3)) for i in range(3))


def inverse(t):
    validate_transform(t)
    a=t.matrix; out=[]
    for i in range(3):
        row=tuple(a[4*j+i] for j in range(3))
        out.extend(row+(-sum(row[j]*a[4*j+3] for j in range(3)),))
    return m.RigidTransform(t.target_frame,t.source_frame,tuple(out)+(0.,0.,0.,1.),'mm',t.version)


def map_to_arm(desired, height_mm, base, base_to_arm):
    c.require(desired.frame_id==base.frame_id,'map.frame','map frames differ')
    finite(height_mm,'height_mm')
    co,si=math.cos(base.yaw),math.sin(base.yaw)
    dx,dy=(desired.x-base.x)*1000.,(desired.y-base.y)*1000.
    return apply(base_to_arm,(co*dx+si*dy,-si*dx+co*dy,height_mm))


def arm_to_map(xyz, base, base_to_arm):
    x,y,z=apply(inverse(base_to_arm),xyz)
    co,si=math.cos(base.yaw),math.sin(base.yaw)
    return (base.x+(co*x-si*y)/1000.,base.y+(si*x+co*y)/1000.,z/1000.)


def _intrinsics(calibration):
    k=calibration.intrinsics
    c.require(k[0]>0 and k[4]>0 and abs(k[3])<1e-12 and k[6:]==(0.,0.,1.),
              'intrinsics','positive focal lengths and pinhole row-major K required')
    return k


def correct_pixel(p, calibration):
    k=_intrinsics(calibration)
    c.require(calibration.distortion is not m.Distortion.UNKNOWN,'distortion','unknown pixel space')
    if calibration.distortion is m.Distortion.CORRECTED:return p
    coeff=calibration.distortion_coefficients
    c.require(len(coeff)==5,'distortion_coefficients','only explicitly calibrated Brown k1,k2,p1,p2,k3 supported')
    yd=(p.v-k[5])/k[4];xd=(p.u-k[2]-k[1]*yd)/k[0]
    k1,k2,p1,p2,k3=coeff
    def distort(x,y):
        r=x*x+y*y;rad=1+k1*r+k2*r*r+k3*r*r*r
        return x*rad+2*p1*x*y+p2*(r+2*x*x),y*rad+p1*(r+2*y*y)+2*p2*x*y
    x,y=xd,yd
    for _ in range(30):
        a,b=distort(x,y);ex,ey=a-xd,b-yd
        if math.hypot(ex,ey)<1e-12:return m.PixelPoint(k[0]*x+k[1]*y+k[2],k[4]*y+k[5])
        h=1e-6;ax,bx=distort(x+h,y);ay,by=distort(x,y+h)
        j00,j10,j01,j11=(ax-a)/h,(bx-b)/h,(ay-a)/h,(by-b)/h
        det=j00*j11-j01*j10
        c.require(abs(det)>1e-12,'distortion','singular inverse distortion')
        x-=(j11*ex-j01*ey)/det;y-=(-j10*ex+j00*ey)/det
        c.require(math.isfinite(x) and math.isfinite(y) and max(abs(x),abs(y))<1e6,'distortion','inverse outside calibrated model')
    raise c.ContractError(c.ErrorCode.INVALID_FIELD,'distortion','inverse did not converge')


def ray_on_plane(pixel, calibration, camera_to_base, height_mm, error_px, height_error_mm):
    """Horizontal base-z plane. Interval bound covers all pixel/height errors.

    Raw distortion inversion is available separately; a measured error amplification
    bound is needed before rectified uncertain pixels may be used for a target.
    """
    validate_transform(camera_to_base,calibration.camera_frame)
    c.require(calibration.distortion is m.Distortion.CORRECTED,'distortion','rectify exactly once with a validated error bound')
    finite(error_px,'error_px',0.);finite(height_error_mm,'height_error_mm',0.);finite(height_mm,'height_mm')
    k=_intrinsics(calibration)
    yn=(pixel.v-k[5])/k[4];xn=(pixel.u-k[2]-k[1]*yn)/k[0]
    direction=rotate(camera_to_base,(xn,yn,1.))
    dy=error_px/k[4];dx=(error_px+abs(k[1])*dy)/k[0]
    radii=tuple(abs(camera_to_base.matrix[4*i])*dx+abs(camera_to_base.matrix[4*i+1])*dy for i in range(3))
    origin=apply(camera_to_base,(0.,0.,0.));den=direction[2]
    c.require(abs(den)>radii[2]+1e-12,'plane','parallel/uncertain ray-plane intersection')
    scale=(height_mm-origin[2])/den
    c.require(scale>0,'plane','plane is behind camera')
    denom=(den-radii[2],den+radii[2]);numer=(height_mm-height_error_mm-origin[2],height_mm+height_error_mm-origin[2])
    scales=[n/d for n in numer for d in denom]
    c.require(min(scales)>0,'plane','uncertain intersection behind camera')
    result=tuple(origin[i]+scale*direction[i] for i in range(3));bounds=[]
    for i in range(3):
        vals=[origin[i]+s*d for s in (min(scales),max(scales)) for d in (direction[i]-radii[i],direction[i]+radii[i])]
        bounds.append(max(abs(v-result[i]) for v in vals))
    return result,math.sqrt(sum(b*b for b in bounds))


def square_yaw(angle):
    return angle % (math.pi/2)


def angle_distance(a,b,period=2*math.pi):
    return abs((a-b+period/2)%period-period/2)
