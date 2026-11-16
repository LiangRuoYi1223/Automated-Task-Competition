"""Nonblocking in-memory event outbox; only P10B persists the C0 events.

A disk call runs outside the queue lock. Unknown writes are retained and latched
for reconciliation, never automatically retried as though nothing happened.
"""
from collections import deque
import threading
from treasure_auto.contracts import core as c, models as m


class EventOutbox:
    def __init__(self, capacity):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('explicit positive outbox capacity required')
        self.capacity = capacity
        self._pending, self._history = deque(), []
        self._lock, self._writer = threading.Lock(), threading.Lock()
        self._failure = None

    @property
    def failure(self):
        with self._lock:
            return self._failure

    @property
    def events(self):
        with self._lock:
            return tuple(self._history)

    @property
    def pending_count(self):
        with self._lock:
            return len(self._pending)

    def append(self, event):
        with self._lock:
            if len(self._pending) >= self.capacity:
                self._failure = c.Reason(c.ErrorCode.STORAGE_FAILURE, 'outbox', 'queue full; output must stop')
                return False
            self._pending.append(event)
            # Bounded diagnostic tail. The durable full history belongs to P10B.
            self._history.append(event)
            if len(self._history) > self.capacity:
                del self._history[0]
            return True

    def check_dispatch_capacity(self):
        """Single gateway producer needs both acceptance and send-result slots."""
        with self._lock:
            if self.capacity - len(self._pending) < 2:
                self._failure = c.Reason(c.ErrorCode.STORAGE_FAILURE, 'outbox', 'insufficient room for acceptance and dispatch result')
                return False
            return self._failure is None

    def flush_one(self, persistence):
        if not self._writer.acquire(blocking=False):
            return False
        try:
            with self._lock:
                if not self._pending or self._failure is not None:
                    return False
                event = self._pending[0]
            try:
                receipt = persistence.append_event(event)
                if type(receipt) is not m.CommitReceipt or receipt.status is not m.CommitStatus.COMMITTED:
                    raise ValueError('event persistence not confirmed')
            except Exception as error:
                with self._lock:
                    self._failure = c.Reason(c.ErrorCode.STORAGE_FAILURE, 'append_event', str(error) or type(error).__name__)
                return False
            with self._lock:
                self._pending.popleft()
            return True
        finally:
            self._writer.release()
