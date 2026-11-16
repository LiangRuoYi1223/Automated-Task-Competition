"""Local assembly configuration. Offline timings are never measured robot values."""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from treasure_auto.contracts import core as c


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    supervision_period_s: float
    lease_s: float
    heartbeat_timeout_s: float
    required_capabilities: Tuple[str, ...]
    signal_adapter: Optional[str]

    def __post_init__(self):
        c.require(self.mode in ('offline', 'formal_simulation', 'formal'), 'mode', 'unknown execution mode')
        for name in ('supervision_period_s', 'lease_s', 'heartbeat_timeout_s'):
            value = getattr(self, name)
            c.require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                      name, 'finite positive seconds required', c.ErrorCode.UNCONFIGURED)
        c.require(self.supervision_period_s < self.lease_s <= self.heartbeat_timeout_s,
                  'timing', 'require period < lease <= heartbeat timeout')
        c.require(type(self.required_capabilities) is tuple and bool(self.required_capabilities)
                  and all(type(x) is str and x.strip() for x in self.required_capabilities)
                  and len(set(self.required_capabilities)) == len(self.required_capabilities),
                  'required_capabilities', 'nonempty unique names required')
        c.require(self.signal_adapter is None or
                  (type(self.signal_adapter) is str and bool(self.signal_adapter.strip())),
                  'signal_adapter', 'explicit adapter identifier or null')
        # No supplied adapter can prove cross-process fencing under C0 1.0.0 yet.
        # Fail closed even if a caller changes the sample's mode flag.
        c.require(self.mode != 'formal', 'formal',
                  'P01 exclusive fencing/stop integration and measured deployment not delivered; '
                  'see P10A_interface_change.md', c.ErrorCode.UNCONFIGURED)

    @classmethod
    def from_dict(cls, data):
        keys = {'mode', 'supervision_period_s', 'lease_s', 'heartbeat_timeout_s',
                'required_capabilities', 'signal_adapter'}
        c.require(type(data) is dict and set(data) == keys, 'config', 'exact configuration fields required')
        data = dict(data)
        c.require(type(data['required_capabilities']) is list, 'required_capabilities', 'expected list')
        data['required_capabilities'] = tuple(data['required_capabilities'])
        return cls(**data)

    @classmethod
    def load(cls, path):
        def pairs(items):
            result = {}
            for key, value in items:
                c.require(key not in result, key, 'duplicate configuration key')
                result[key] = value
            return result
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs))
