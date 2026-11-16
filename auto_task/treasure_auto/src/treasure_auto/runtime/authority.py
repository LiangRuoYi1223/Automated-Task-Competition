"""Small final-permission lock. Never calls storage, business code or device callbacks."""
import threading
from treasure_auto.contracts import core as c


class PermitAuthority:
    def __init__(self, config, read_clock):
        self.config = config
        self._clock = read_clock
        self._lock = threading.RLock()
        self._run = None
        self._heartbeat = None
        self._last_now = None
        self._permit = None
        self._revision = 0
        self._reason = c.Reason(c.ErrorCode.NOT_AUTHORIZED, 'state', 'not initialized')
        self._terminal = False

    @property
    def ticket(self):
        with self._lock:
            return self._revision

    @property
    def reason(self):
        with self._lock:
            return self._reason

    @property
    def active_permit(self):
        with self._lock:
            return self._permit

    def bind(self, run):
        with self._lock:
            c.require(self._run is None, 'run', 'authority cannot be reused for a new run')
            self._run = run

    def rotate(self, run, ticket, now):
        """Install a reviewed same-run generation while preserving every stop latch."""
        with self._lock:
            now = self._clock()
            self._watch(now)
            c.require(ticket == self._revision and not self._terminal, 'stop_lock', 'concurrent or terminal stop')
            c.require(self._reason is not None and self._run is not None, 'stop_lock', 'rotation requires stopped authority')
            c.require((run.started_at, run.deadline, run.envelope.run_id) ==
                      (self._run.started_at, self._run.deadline, self._run.envelope.run_id),
                      'run', 'rotation must retain original run')
            self._run = run
            self._permit = None
            self._heartbeat = None
            self._revision += 1

    def _revoke(self, reason, terminal=False, new_event=True):
        if new_event or self._reason != reason or self._permit is not None:
            self._revision += 1
        if not self._terminal:
            self._reason = reason
        self._terminal = self._terminal or terminal
        self._permit = None
        return self._reason

    def revoke(self, reason, terminal=False):
        with self._lock:
            return self._revoke(reason, terminal)

    def _watch(self, now):
        if self._run is None:
            return self._reason
        try:
            c.monotonic(now, 'now')
            c.require(now - self._run.started_at >= 0, 'now', 'clock predates start', c.ErrorCode.CLOCK_MISMATCH)
            if self._last_now is not None:
                c.require(now - self._last_now >= 0, 'now', 'clock reversed', c.ErrorCode.CLOCK_MISMATCH)
            self._last_now = now
            if now - self._run.deadline >= 0:
                return self._revoke(c.Reason(c.ErrorCode.DEADLINE, 'deadline', 'original deadline reached'), True, False)
            if self._heartbeat is not None and now - self._heartbeat >= self.config.heartbeat_timeout_s:
                return self._revoke(c.Reason(c.ErrorCode.TIMEOUT, 'heartbeat', 'mission heartbeat expired'), new_event=False)
            if self._permit is not None and now - self._permit.envelope.expires_at >= 0:
                return self._revoke(c.Reason(c.ErrorCode.EXPIRED, 'permit', 'control lease expired'), new_event=False)
        except c.ContractError as error:
            return self._revoke(c.Reason(error.code, error.field, error.detail), True, False)
        return self._reason

    def watch(self, now):
        with self._lock:
            return self._watch(self._clock())

    def heartbeat(self, now):
        with self._lock:
            now = self._clock()
            self._watch(now)
            self._heartbeat = now
            # Receiving data never clears a stop latch.

    def release(self, ticket, now):
        with self._lock:
            now = self._clock()
            self._watch(now)
            c.require(ticket == self._revision and not self._terminal, 'stop_lock',
                      'stop changed while operation was pending', c.ErrorCode.NOT_AUTHORIZED)
            self._reason = None
            self._heartbeat = now

    def install(self, permit, ticket, now):
        with self._lock:
            now = self._clock()
            self._watch(now)
            c.require(ticket == self._revision and self._reason is None, 'stop_lock',
                      'stopped or concurrent revocation', c.ErrorCode.NOT_AUTHORIZED)
            c.require(self._permit is None or self._permit.source is permit.source,
                      'source', 'another source owns control', c.ErrorCode.CONFLICT)
            c.require(permit.envelope.expires_at - now > 0, 'permit', 'expired before installation', c.ErrorCode.EXPIRED)
            self._permit = permit

    def allows(self, permit, envelope, source, now, real_output):
        with self._lock:
            now = self._clock()
            if self._watch(now) is not None or permit is None or permit != self._permit:
                return False
            if real_output:
                # Both delivered assembly modes are simulation; formal is gated
                # by config until P01 fencing and physical validation exist.
                return False
            p = permit.envelope
            try:
                return (source is permit.source and envelope.origin is p.origin
                        and all(getattr(envelope, n) == getattr(p, n) for n in
                                ('run_id', 'instance_id', 'control_epoch', 'request_id', 'action_id',
                                 'target_id', 'versions', 'base_motion_epoch',
                                 'layout_recovery_round', 'rebuild_generation'))
                        and now - envelope.emitted_at >= 0 and envelope.expires_at - now > 0)
            except c.ContractError:
                return False
