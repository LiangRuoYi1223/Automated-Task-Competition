"""Explicit system clock adapter; import never probes the host or starts work."""
import time
from pathlib import Path
from treasure_auto.contracts import core as c


class LinuxBootClock:
    """CLOCK_BOOTTIME includes suspend; availability/robot behavior still need measurement."""
    def __init__(self):
        c.require(hasattr(time, 'CLOCK_BOOTTIME'), 'clock',
                  'Linux CLOCK_BOOTTIME required; no wall-clock fallback', c.ErrorCode.UNAVAILABLE)
        self.boot_id = Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip()
        c.require(bool(self.boot_id), 'boot_id', 'OS boot identity missing', c.ErrorCode.UNAVAILABLE)

    def now(self):
        return c.ClockStamp(time.clock_gettime(time.CLOCK_BOOTTIME), c.ClockBasis.MONOTONIC,
                            'linux:CLOCK_BOOTTIME', self.boot_id)
