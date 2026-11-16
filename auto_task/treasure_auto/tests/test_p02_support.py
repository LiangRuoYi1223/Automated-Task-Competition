"""Local SIMULATED test vectors; not robot measurements or another C0."""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c, models as m


def stamp(t=10):
    return c.ClockStamp(t, c.ClockBasis.MONOTONIC, 'sim-clock', 'sim-boot')


def env(t=10, event='e1', **kw):
    data = dict(run_id='sim-run', instance_id='sim-instance', control_epoch=1,
                producer=c.Owner.P01, event_id=event, sequence=int(t * 100),
                request_id=None, action_id=None, target_id=None,
                emitted_at=stamp(t), expires_at=stamp(t + 10),
                versions=c.Versions('sim-config', 'map-v1', None, None, None, None, 0),
                origin=c.Origin.SIMULATED, layout_recovery_round=None,
                rebuild_generation=None, base_motion_epoch=0)
    data.update(kw)
    return c.Envelope(**data)


def proof(t=10, name='frame1', **kw):
    data = dict(evidence_id=name, run_id='sim-run', target_id=None, action_id=None,
                kind=c.EvidenceKind.PHYSICAL, origin=c.Origin.SIMULATED,
                captured_at=c.ClockStamp(t, c.ClockBasis.ROS, 'sim-tf', 'sim-boot'),
                received_at=stamp(t), valid_until=stamp(t + 10), uri='sim://' + name, sha256=None)
    data.update(kw)
    return c.EvidenceRef(**data)


def pose(x=2, y=2, yaw=0, frame='map', error=0):
    return c.MapPose(frame, x, y, yaw, 'm', 'rad', error, 0)


def poly(x0, y0, x1, y1):
    return tuple(m.Point2(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


def bundle():
    from treasure_auto.mapping.storage import MapBundle, Raster
    regions = tuple(m.RegionBoundary(a, 'map', poly(i, 0, i + .45, .45),
                                    pose(i + .225, .225), 0)
                    for i, a in enumerate((m.Area.A, m.Area.B, m.Area.C, m.Area.D)))
    metadata = m.MapMetadata(env(producer=c.Owner.P02), 'map-v1', 'grid.pgm', .5,
                             pose(0, 0), regions, m.Measurement.SIMULATED)
    waypoint = m.WaypointResult(env(producer=c.Owner.P02), 'board', 'observe',
                                pose(2, 2), c.Fact.UNKNOWN, None)
    return MapBundle(metadata, Raster(12, 12, (255,) * 144), poly(0, 0, 6, 6),
                     poly(2.5, 2.5, 3.5, 3.5), (waypoint,), .196, .65)


def policy():
    from treasure_auto.mapping.localization import LocalizationPolicy
    return LocalizationPolicy(min_samples=3, confirmation_s=.4, max_gap_s=.3,
                              max_age_s=1, max_skew_s=.02, max_error_m=.1,
                              max_error_rad=.1, jump_m=.4, jump_rad=.4,
                              initial_m=.2, initial_rad=.2, stable_m=.03,
                              stable_rad=.05, measurement=m.Measurement.SIMULATED)


def transforms(t=10, x=2, y=2, yaw=0, correction=None, **kw):
    from treasure_auto.mapping.localization import TransformSample
    e = env(t, 'obs-' + str(t), **kw)
    return (TransformSample(e, correction or pose(0, 0), 'odom', proof(t, 'map-' + str(t)),proof(t).captured_at),
            TransformSample(e, pose(x, y, yaw, 'odom'), 'base_link', proof(t, 'odom-' + str(t)),proof(t).captured_at))


class Clock:
    def __init__(self, t=10):
        self.t = t

    def __call__(self):
        return stamp(self.t)


def occupancy(state=m.OccupancyState.UNKNOWN, box=(1.9, 1.9, 2.1, 2.1)):
    return m.Occupancy('brick1', state, 'map', poly(*box), 0, .1, 0, (),
                       stamp(), m.Visibility.UNKNOWN, None, ())


def space(items=(), t=10, **kw):
    return m.SpaceSnapshot(env(t, 'space', producer=c.Owner.P08, **kw), 0, tuple(items))
