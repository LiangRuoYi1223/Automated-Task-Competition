"""Explicit deployment inputs. Example values are not measurements."""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from treasure_auto.contracts.core import require, ErrorCode


@dataclass(frozen=True)
class PersistenceConfig:
    runtime_root: str
    queue_capacity: int
    operation_timeout_s: float
    timer_refresh_period_s: float
    max_keyframe_bytes: int

    def __post_init__(self):
        require(isinstance(self.runtime_root, str) and bool(self.runtime_root),
                'runtime_root', 'explicit dedicated absolute root required', ErrorCode.UNCONFIGURED)
        root = Path(self.runtime_root)
        require(root.is_absolute() and '..' not in root.parts and root != Path(root.anchor),
                'runtime_root', 'absolute non-volume dedicated directory required')
        for field in ('queue_capacity', 'max_keyframe_bytes'):
            value = getattr(self, field)
            require(type(value) is int and value > 0, field, 'positive integer required')
        for field in ('operation_timeout_s', 'timer_refresh_period_s'):
            value = getattr(self, field)
            require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                    field, 'explicit finite positive value required')

    @classmethod
    def from_json(cls, filename):
        return cls(**json.loads(Path(filename).read_text(encoding='utf-8')))
