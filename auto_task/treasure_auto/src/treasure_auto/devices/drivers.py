"""Bounded connection decisions; no implicit launch, restart, ROS or serial IO."""
from dataclasses import dataclass
import threading
from .config import positive


@dataclass(frozen=True)
class ConnectionResult:
    state: str
    operation: str
    detail: str
    physical_verified: bool = False


class DriverSession:
    """poll() only consumes cached observations; start_once is an offline decision.

    A future authorized launcher must enforce its own subprocess total timeout.
    Calling poll cannot launch anything. Terminal failures never restart a driver.
    """
    def __init__(self, clock, timeout_s, managed, recovering, startup_effects_verified, origin):
        positive(timeout_s, 'timeout_s')
        if any(type(v) is not bool for v in (managed, recovering, startup_effects_verified)):
            raise ValueError('driver policy flags must be explicit booleans')
        if origin not in ('simulated', 'live'):
            raise ValueError('invalid observation origin')
        self.clock, self.start = clock, clock.now()
        self.timeout_s, self.managed, self.recovering = timeout_s, managed, recovering
        self.can_start = startup_effects_verified and origin == 'simulated'
        self._started, self._terminal = False, None

    def poll(self, process, interface, messages, transforms, launch_error=None):
        if self._terminal is not None:
            return self._terminal
        try:
            elapsed = self.clock.now() - self.start
            if elapsed < 0:
                raise ValueError('clock moved backwards')
        except ValueError as error:
            self._terminal = ConnectionResult('failed', 'none', str(error))
            return self._terminal
        if elapsed >= self.timeout_s:
            self._terminal = ConnectionResult('timed_out', 'none', 'connection total deadline reached')
        elif launch_error:
            self._terminal = ConnectionResult('failed', 'none', str(launch_error))
        elif all(v is True for v in (process, interface, messages, transforms)):
            self._terminal = ConnectionResult('ready', 'started' if self._started else 'reuse', 'interfaces only; physical verification pending')
        if self._terminal:
            return self._terminal
        if process is not True:
            launch = self.managed and not self.recovering and self.can_start and not self._started
            self._started = self._started or launch
            return ConnectionResult('waiting_process', 'start_once' if launch else 'none',
                                    'startup power/reset/pump effects require verified preparation')
        for present, state in ((interface, 'waiting_interface'), (messages, 'waiting_messages'), (transforms, 'waiting_tf')):
            if present is not True:
                return ConnectionResult(state, 'none', 'awaiting actual observation')


class SerialLease:
    """In-process simulation ownership only. OS/driver exclusivity is NOT proven."""
    _guard = threading.Lock()
    _owners = set()

    def __init__(self, resource):
        if not isinstance(resource, str) or not resource.startswith('sim://'):
            raise ValueError('real serial ownership adapter has not been verified')
        self.resource, self.owned, self._controller = resource, False, None

    def bind(self, controller):
        with self._guard:
            if not self.owned or self._controller is not None:
                raise RuntimeError('serial lease absent or already bound to a controller')
            self._controller = controller

    def owns(self, controller):
        with self._guard:
            return self.owned and self._controller is controller

    def acquire(self):
        with self._guard:
            if self.resource in self._owners:
                raise RuntimeError('serial resource already owned')
            self._owners.add(self.resource)
            self.owned = True
        return self

    def release(self):
        with self._guard:
            if self.owned:
                self._owners.remove(self.resource)
                self.owned = False
                self._controller = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()
