"""Explicit watchdog lifecycle. No threads or outputs are started by import."""
import threading
from treasure_auto.contracts import core as c


class IndependentSupervisor:
    """Local watchdog using cached health only; stop sink must return promptly.

    P11 must isolate the final P01 watchdog from process/GIL failure. This thread
    proves independence from waiting business/storage calls, not hardware latency.
    """
    def __init__(self, runtime, stop_sink):
        self.runtime, self.stop_sink = runtime, stop_sink
        self._closed = threading.Event()
        self._thread = None
        self.last_error = None

    def start(self):
        c.require(self._thread is None, 'supervisor', 'start once')
        self._thread = threading.Thread(target=self._loop, name='p10a-supervisor', daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._closed.is_set():
            try:
                decision = self.runtime.supervise(self.runtime.clock.now(), self.runtime.cached_health)
                if decision is not None and decision.stop is not None:
                    self.stop_sink(decision.stop)
            except Exception as error:
                self.last_error = type(error).__name__
                self.runtime.authority.revoke(c.Reason(c.ErrorCode.UNAVAILABLE, 'supervisor', self.last_error), True)
            self._closed.wait(self.runtime.config.supervision_period_s)

    def close(self):
        self.runtime.authority.revoke(c.Reason(c.ErrorCode.CANCELLED, 'supervisor', 'supervision stopped'), True)
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            c.require(not self._thread.is_alive(), 'stop_sink', 'sink blocked; final gateway must fail closed', c.ErrorCode.UNAVAILABLE)
