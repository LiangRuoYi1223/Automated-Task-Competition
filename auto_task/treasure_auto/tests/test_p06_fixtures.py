"""Synthetic P06 vectors. Never a hardware configuration or public contract copy."""
import importlib
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c, models as m
from contract_fixtures.samples import envelope, evidence, stamp, consumer_context


def module(case, name):
    path = 'treasure_auto.geometry.' + name
    case.assertIsNotNone(importlib.util.find_spec(path), 'P06 implementation missing: ' + name)
    return importlib.import_module(path)


def transform(source='base', target='arm', xyz=(0., 0., 0.), version='sim-transform-v1'):
    x, y, z = xyz
    return m.RigidTransform(source, target, (1., 0., 0., x, 0., 1., 0., y,
                          0., 0., 1., z, 0., 0., 0., 1.), 'mm', version)


def camera_transform():
    # Optical x right, optical y down, optical z forward; camera looks down from 1000 mm.
    return m.RigidTransform('camera', 'base', (1., 0., 0., 0., 0., -1., 0., 0.,
                           0., 0., -1., 1000., 0., 0., 0., 1.), 'mm', 'sim-extrinsic-v1')


def camera():
    return m.CameraCalibration('sim-calibration-v1', 'camera',
        (500., 0., 320., 0., 500., 240., 0., 0., 1.), (), m.Distortion.CORRECTED,
        'sim-domain', 0.1, 0.5, (evidence(evidence_id='validation'),))


def pose(x=0., y=0., yaw=0., error=0.):
    return c.MapPose('map', x, y, yaw, 'm', 'rad', error, 0.)


def config_dict():
    return dict(version='sim-config-v1', measurement='simulated',
        map_frame='map', base_frame='base', arm_frame='arm', image_size=[640, 480],
        approach_mm=80., descend_mm=20., lift_mm=100., withdraw_mm=80.,
        speeds_mm_s=[10., 5., 1., 5., 10.], timeouts_s=[60., 60., 60., 60., 60.],
        cup_radius_mm=5., tool_error_mm=0.5, brick_size_mm=[100., 100., 100.],
        max_error_mm=10., plane_tolerance_mm=1., max_pose_age_s=2., fixed_tool_yaw_rad=0.,
        yaw_tolerance_rad=0.001, max_contact_gap_mm=1., max_pixel_error_px=3.)


def square(x=0., y=0., size=.1):
    return tuple(m.Point2(a, b) for a, b in ((x,y),(x+size,y),(x+size,y+size),(x,y+size)))


def occupancy(name='lower', x=0., z0=0., z1=.1, error=0., state=m.OccupancyState.ACTUAL):
    return m.Occupancy(name, state, 'map', square(x), z0, z1, error, (), stamp(110),
                       m.Visibility.VISIBLE, m.Area.B, (evidence(target_id=name),))


def space(items=()):
    return m.SpaceSnapshot(envelope(c.Owner.P08, target_id=None, action_id=None, request_id=None), 0, tuple(items))
