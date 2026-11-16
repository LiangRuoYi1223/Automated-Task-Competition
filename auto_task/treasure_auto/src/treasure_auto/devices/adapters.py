"""Pure stage-speed encoding; this is not an installed vendor SDK driver."""
from dataclasses import dataclass
import math
from treasure_auto.contracts import core as c, models as m


@dataclass(frozen=True)
class ArmDispatch:
    x_mm: float
    y_mm: float
    z_mm: float
    feed_mm_min: float
    phase: m.Phase
    timeout_s: float
    action_id: str


def encode_arm(command, config):
    c.require(config.arm_frame_id is not None and config.arm_bounds_mm is not None,
              'arm_bounds_mm', 'position frame and bounds unmeasured', c.ErrorCode.UNCONFIGURED)
    c.require(command.target.frame_id == config.arm_frame_id, 'frame_id', 'wrong arm coordinate frame')
    for coordinate, bounds in zip((command.target.x, command.target.y, command.target.z), config.arm_bounds_mm):
        c.require(bounds[0] <= coordinate - command.target.error_mm and coordinate + command.target.error_mm <= bounds[1],
                  'target', 'position including uncertainty outside configured bounds', c.ErrorCode.LIMIT_EXCEEDED)
    limits = dict(config.stage_limits).get(command.phase)
    c.require(limits is not None, 'stage_limits', 'phase speed range unmeasured', c.ErrorCode.UNCONFIGURED)
    low, high, timeout = limits
    c.require(low <= command.speed_mm_s <= high and command.timeout_s <= timeout,
              'speed_timeout', 'outside configured stage limits', c.ErrorCode.LIMIT_EXCEEDED)
    feed = command.speed_mm_s * 60.0
    c.require(math.isfinite(feed), 'feed_mm_min', 'conversion overflow', c.ErrorCode.LIMIT_EXCEEDED)
    return ArmDispatch(command.target.x, command.target.y, command.target.z, feed,
                       command.phase, command.timeout_s, command.envelope.action_id)
