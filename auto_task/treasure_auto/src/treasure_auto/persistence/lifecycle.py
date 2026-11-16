"""Conservative lifecycle; physical stops and time policy are owned by P10A."""
import math
import sqlite3
from dataclasses import replace
from treasure_auto.contracts import core as c, models as m, rules as r
from . import filesystem as fs
from .filesystem import remove_run


class Lifecycle:
    def _remaining(self, run, now):
        # P10A supplies the policy. P10B only checks its output shape; C0
        # TimerRecord additionally checks consistency with the frozen deadline.
        result = self.remaining(run, now)
        c.require(type(result) in (int, float) and math.isfinite(result) and 0 <= result <= 600,
                  'remaining', 'time policy returned invalid remaining time', c.ErrorCode.CLOCK_MISMATCH)
        return result

    def _marker_path(self, run_id):
        return fs.checked(self.root / 'ends' / (fs.run_name(run_id) + '.json'), self.root)

    def _marker(self, run_id):
        path = self._marker_path(run_id)
        if not path.exists():
            return None
        marker = c.loads(path.read_text(encoding='utf-8'))
        c.require(isinstance(marker, m.EndMarker) and marker.run_id == run_id,
                  'marker', 'invalid end marker', c.ErrorCode.STORAGE_FAILURE)
        return marker

    def _write_marker(self, marker):
        fs.atomic_write(self._marker_path(marker.run_id), c.dumps(marker).encode('utf-8'))

    def _initialize_run(self, request):
        run, initial = request.run, request.initial_snapshot
        run_id = fs.run_name(run.envelope.run_id)
        c.require(not self._faulted.is_set(), 'storage', 'fault latched', c.ErrorCode.STORAGE_FAILURE)
        c.require(self._marker(run_id) is None, 'run_id', 'run already ended; explicit different run required', c.ErrorCode.NOT_AUTHORIZED)
        fingerprint = fs.digest(c.dumps(request).encode('utf-8'))
        known = self._catalog['runs'].get(run_id)
        if known:
            c.require(known['initial_digest'] == fingerprint and known['phase'] == 'active',
                      'initialize_run', 'run identity conflict', c.ErrorCode.ID_CONFLICT)
            self._snapshot(run_id)
            return m.CommitReceipt('initialize:' + run_id, m.CommitStatus.COMMITTED, 0, known['initial_space'], None)
        c.require(self._catalog['active'] is None, 'run_id', 'another active run exists', c.ErrorCode.CONFLICT)
        c.require(run.start_signal_id not in self._catalog['starts'], 'start_signal_id', 'start already used', c.ErrorCode.DUPLICATE)
        if self._catalog['runs']:
            prior = request.previous_end_marker
            c.require(prior is not None and self._marker(prior.run_id) == prior and
                      prior.sealed is c.Fact.CONFIRMED and prior.writers_closed is c.Fact.CONFIRMED,
                      'previous_end_marker', 'explicit completed prior-run marker required', c.ErrorCode.NOT_AUTHORIZED)
        c.require(initial.run == run and initial.revision == 0 and not initial.deliveries and not run.ever_restarted,
                  'initial_snapshot', 'new run cannot inherit revisions/arrivals/restart history')
        c.require(run.envelope.producer is c.Owner.P10A and
                  run.state in (m.RunState.RUNNING, m.RunState.OFFLINE), 'run', 'P10A must establish active run')
        signal = request.start_signal
        c.require(signal.kind is m.SignalKind.START and signal.referee_authorized is c.Fact.CONFIRMED and
                  signal.envelope.producer is c.Owner.P10A and signal.envelope.event_id == run.start_signal_id,
                  'start_signal', 'authorized exact P10A start required', c.ErrorCode.NOT_AUTHORIZED)
        self._current_envelope(signal.envelope, initial)
        r.require_provenance(initial, run.envelope.origin)
        c.require(initial.space.envelope.run_id == run_id, 'space', 'initial space wrong run')
        remaining = self._remaining(run, self.clock.now())
        c.require(remaining > 0, 'deadline', 'cannot initialize expired run', c.ErrorCode.DEADLINE)
        directory = self._run_path(run_id)
        c.require(not directory.exists(), 'run_id', 'unowned directory exists; preserve it', c.ErrorCode.CONFLICT)
        # Reservation precedes mkdir: a crash cannot turn missing data into a new run.
        self._catalog['active'] = run_id
        self._catalog['starts'][run.start_signal_id] = run_id
        self._catalog['runs'][run_id] = dict(initial_digest=fingerprint, initial_space=initial.space.version,
                                           phase='reserved', ever_restarted=False, end_reason=None)
        self._save_catalog()
        directory.mkdir()
        (directory / 'keyframes').mkdir()
        self._db = sqlite3.connect(str(directory / 'state.sqlite3'), isolation_level=None,
                                   timeout=self.config.operation_timeout_s)
        self._db_run = run_id
        self._configure_db()
        self._db.executescript('''
            PRAGMA user_version=1;
            CREATE TABLE snapshot(id INTEGER PRIMARY KEY CHECK(id=1),payload TEXT NOT NULL,sha256 TEXT NOT NULL);
            CREATE TABLE receipts(kind TEXT NOT NULL,key TEXT NOT NULL,payload TEXT NOT NULL,result TEXT NOT NULL,PRIMARY KEY(kind,key));
            CREATE TABLE history(id INTEGER PRIMARY KEY AUTOINCREMENT,kind TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL,result TEXT);
        ''')
        receipt = m.CommitReceipt('initialize:' + run_id, m.CommitStatus.COMMITTED, 0, initial.space.version, None)
        self._check_frames(initial)
        def work():
            self._write_snapshot(initial)
            self._log('initialize', request, receipt)
        self._atomic(work)
        self._catalog['runs'][run_id]['phase'] = 'active'
        self._save_catalog()
        self._authorized = True
        refreshed = self.clock.now()
        remaining = self._remaining(run, refreshed)
        self._timer_write(m.TimerRecord(1, run_id, run.started_at, run.deadline, refreshed,
                                        remaining, self.config.timer_refresh_period_s, 0, run.state, False))
        if remaining > 0:
            self._blocked.clear()
        return receipt

    def _timer_read(self, run_id):
        path = fs.checked(self._run_path(run_id) / 'timer.txt', self.root)
        record = c.loads(path.read_text(encoding='utf-8'))
        c.require(isinstance(record, m.TimerRecord), 'timer', 'wrong timer record', c.ErrorCode.STORAGE_FAILURE)
        return record

    def _timer_write(self, record):
        fs.atomic_write(self._run_path(record.run_id) / 'timer.txt', c.dumps(record).encode('utf-8'))

    def _refresh_timer(self, record):
        snapshot = self._writable(record.run_id)
        run, old = snapshot.run, self._timer_read(record.run_id)
        c.require(record.started_at == run.started_at and record.deadline == run.deadline and
                  record.state is run.state and record.ever_restarted == run.ever_restarted,
                  'timer', 'timer contradicts snapshot', c.ErrorCode.CONFLICT)
        c.require(record.refresh_period_s == old.refresh_period_s, 'refresh_period_s', 'keep valid configured period')
        c.require(record.remaining_s == self._remaining(run, record.refreshed_at) and
                  self.clock.now() - record.refreshed_at >= 0,
                  'timer', 'invalid or future refresh', c.ErrorCode.CLOCK_MISMATCH)
        if record == old:
            return m.CommitReceipt('timer:' + str(record.sequence), m.CommitStatus.COMMITTED, snapshot.revision, snapshot.space.version, None)
        c.require(record.sequence > old.sequence and record.refreshed_at - old.refreshed_at >= 0,
                  'timer.sequence', 'stale refresh', c.ErrorCode.STALE_REQUEST)
        self._timer_write(record)
        self._log('timer', record)
        return m.CommitReceipt('timer:' + str(record.sequence), m.CommitStatus.COMMITTED, snapshot.revision, snapshot.space.version, None)

    def _stop_confirmed(self, stop, run, now):
        try:
            c.require(stop.envelope.run_id == run.envelope.run_id and stop.envelope.producer is c.Owner.P01 and
                      stop.envelope.instance_id == run.envelope.instance_id and
                      stop.envelope.control_epoch == run.envelope.control_epoch and
                      stop.envelope.origin is run.envelope.origin and
                      stop.physically_stopped is c.Fact.CONFIRMED and stop.old_queue_empty is c.Fact.CONFIRMED and
                      stop.requested is c.Fact.CONFIRMED and stop.submitted is c.Fact.CONFIRMED and
                      bool(stop.evidence), 'stop', 'physical stop and queue evidence required')
            c.require(now - stop.envelope.emitted_at >= 0 and stop.envelope.expires_at - now > 0,
                      'stop', 'stale/future stop report')
            for evidence in stop.evidence:
                r.require_physical_evidence(evidence, stop.envelope, now, run.envelope.origin)
            self._check_frames(stop.evidence)
            return c.Fact.CONFIRMED
        except c.ContractError:
            return c.Fact.UNKNOWN

    def _register_recovery(self, request):
        run_id = fs.run_name(request.envelope.run_id)
        self._blocked.set()
        self._authorized = False
        c.require(run_id in self._catalog['runs'], 'run_id', 'unknown run; do not initialize', c.ErrorCode.STALE_RUN)
        marker = self._marker(run_id)
        if marker is not None:
            # Never reopen a terminal run, including an empty/mid-cleanup directory.
            return m.RecoveryResult(c.Outcome.REJECTED, None, 0.0, None,
                                     c.Reason(c.ErrorCode.DEADLINE, 'marker', 'ended run; only cleanup/retention allowed'))
        entry = self._catalog['runs'][run_id]
        entry['ever_restarted'] = True  # Permanent even if subsequent validation fails.
        self._save_catalog()
        snapshot = self._snapshot(run_id)
        run = snapshot.run
        c.require(run.state not in (m.RunState.ENDED, m.RunState.STOPPING),
                  'snapshot', 'terminal snapshot without complete marker; retain for reconciliation', c.ErrorCode.CONFLICT)
        c.require(entry['phase'] == 'active' and self._catalog['active'] == run_id,
                  'run_id', 'incomplete initialization/ended registry', c.ErrorCode.CONFLICT)
        # Verify all retained events before trusting the snapshot for recovery.
        for row in self._history(run_id):
            if not row['kind'].startswith('rejected:'):
                self._check_frames(c.loads(row['payload']))
        self._check_frames(snapshot)
        old = self._timer_read(run_id)
        c.require(old.run_id == run_id and old.started_at == run.started_at and old.deadline == run.deadline and
                  (not old.ever_restarted or run.ever_restarted), 'timer', 'timer identity/deadline contradiction', c.ErrorCode.CONFLICT)
        c.require(request.original_started_at == run.started_at and request.original_deadline == run.deadline,
                  'original_deadline', 'original run mismatch', c.ErrorCode.CONFLICT)
        now = self.clock.now()
        remaining = self._remaining(run, now)
        c.require(now - old.refreshed_at >= 0, 'timer', 'refresh is in future', c.ErrorCode.CLOCK_MISMATCH)
        c.require(request.envelope.producer is c.Owner.P10A and request.envelope.origin is run.envelope.origin,
                  'recovery', 'P10A registration only', c.ErrorCode.NOT_AUTHORIZED)
        c.require(now - request.envelope.emitted_at >= 0 and request.envelope.expires_at - now > 0 and
                  request.envelope.versions == snapshot.space.envelope.versions,
                  'recovery', 'expired registration or changed versions', c.ErrorCode.VERSION_MISMATCH)
        key = request.envelope.instance_id
        known = self._lookup('recovery', key, request)
        if known is not None:
            c.require(run.envelope.instance_id == key and run.envelope.control_epoch == request.envelope.control_epoch,
                      'control_epoch', 'registration belongs to superseded instance', c.ErrorCode.STALE_CONTROL)
            self._authorized = True
            if remaining > 0:
                return replace(known, snapshot=snapshot, remaining_s=remaining)
        c.require(known is not None or (request.envelope.instance_id != run.envelope.instance_id and
                  request.envelope.control_epoch > run.envelope.control_epoch),
                  'control_epoch', 'new instance and increasing control epoch required', c.ErrorCode.STALE_CONTROL)
        state = m.RunState.ENDED if remaining == 0 else m.RunState.RECOVERING
        updated = replace(run, envelope=request.envelope, state=state, ever_restarted=True)
        proposed = replace(snapshot, run=updated, revision=snapshot.revision + 1)
        receipt = m.CommitReceipt('recovery:' + key, m.CommitStatus.COMMITTED, proposed.revision, proposed.space.version, None)
        reason = c.Reason(c.ErrorCode.DEADLINE if remaining == 0 else c.ErrorCode.NOT_AUTHORIZED,
                          'recovery', 'ended; no movement' if remaining == 0 else 'fresh stop/scene/holding/paths/permission required; never replay')
        result = m.RecoveryResult(c.Outcome.SUCCESS, proposed, remaining, receipt, reason)
        def work():
            self._write_snapshot(proposed)
            self._log('recovery', request, result)
            if known is None:
                self._remember('recovery', key, request, result)
        self._atomic(work)
        self._timer_write(m.TimerRecord(1, run_id, run.started_at, run.deadline, now, remaining,
                                        old.refresh_period_s, old.sequence + 1, state, True))
        self._authorized = True  # Only state/event storage; intent remains blocked by RECOVERING.
        if remaining == 0:
            stop = self._stop_confirmed(request.old_stop, run, now)
            marker = m.EndMarker(run_id, run.deadline, now, True, stop, c.Fact.UNKNOWN,
                                 c.Fact.CONFIRMED, m.CleanupState.RETAINED)
            self._write_marker(marker)
            entry.update(phase='ended', end_reason=c.ErrorCode.DEADLINE.value)
            self._catalog['active'] = None
            self._save_catalog()
            self._close_db()
            self._write_marker(replace(marker, writers_closed=c.Fact.CONFIRMED))
            self._authorized = False
        return result

    def _seal(self, request):
        run_id = fs.run_name(request.envelope.run_id)
        self._blocked.set()
        old_marker = self._marker(run_id)
        if old_marker is not None:
            return old_marker
        c.require(self._authorized and self._catalog['active'] == run_id,
                  'seal', 'not current registered instance', c.ErrorCode.NOT_AUTHORIZED)
        snapshot = self._snapshot(run_id)
        run, now = snapshot.run, self.clock.now()
        self._current_envelope(request.envelope, snapshot)
        c.require(request.envelope.producer is c.Owner.P10A and request.run.state is m.RunState.ENDED and
                  request.run.started_at == run.started_at and request.run.deadline == run.deadline and
                  request.run.start_signal_id == run.start_signal_id and request.run.ever_restarted == run.ever_restarted and
                  request.run.envelope == request.envelope,
                  'seal', 'P10A terminal context must preserve original identity/deadline')
        remaining = self._remaining(run, now)
        c.require(request.end_reason.code is not c.ErrorCode.DEADLINE or remaining == 0,
                  'deadline', 'deadline end cannot occur early', c.ErrorCode.DEADLINE)
        stopped = self._stop_confirmed(request.stop, run, now)
        restarted = run.ever_restarted or self._catalog['runs'][run_id]['ever_restarted']
        eligible = (request.end_reason.code is c.ErrorCode.DEADLINE and remaining == 0 and
                    not restarted and stopped is c.Fact.CONFIRMED and not self._faulted.is_set())
        marker = m.EndMarker(run_id, run.deadline, now, restarted, stopped, c.Fact.UNKNOWN,
                             c.Fact.CONFIRMED, m.CleanupState.PENDING if eligible else m.CleanupState.RETAINED)
        terminal = replace(snapshot, run=replace(request.run, ever_restarted=restarted), revision=snapshot.revision + 1)
        def work():
            self._write_snapshot(terminal)
            self._log('seal', request)
        self._atomic(work)
        old_timer = self._timer_read(run_id)
        self._timer_write(m.TimerRecord(1, run_id, run.started_at, run.deadline, now, remaining,
                                        old_timer.refresh_period_s, old_timer.sequence + 1, m.RunState.ENDED, restarted))
        self._write_marker(marker)  # Before closing and before any deletion.
        entry = self._catalog['runs'][run_id]
        entry.update(phase='ended', ever_restarted=restarted, end_reason=request.end_reason.code.value)
        self._catalog['active'] = None
        self._save_catalog()
        self._authorized = False
        self._close_db()  # Prior jobs drained; subsequent queued writes fail terminal checks.
        marker = replace(marker, writers_closed=c.Fact.CONFIRMED)
        self._write_marker(marker)
        return marker

    def _cleanup(self, request):
        run_id = fs.run_name(request.run_id)
        c.require(request.runtime_root == str(self.root), 'runtime_root', 'must exactly match configured absolute root', c.ErrorCode.NOT_AUTHORIZED)
        fs.checked(self.root)
        marker = self._marker(run_id)
        c.require(marker is not None and marker == request.marker, 'marker', 'caller marker must match durable marker', c.ErrorCode.CONFLICT)
        entry = self._catalog['runs'].get(run_id)
        c.require(entry is not None and entry['phase'] == 'ended' and self._catalog['active'] != run_id and
                  marker.sealed is c.Fact.CONFIRMED and marker.writers_closed is c.Fact.CONFIRMED,
                  'cleanup', 'run must be ended and no longer used', c.ErrorCode.NOT_AUTHORIZED)
        if request.automatic:
            c.require(not self._faulted.is_set() and not entry['ever_restarted'] and not marker.ever_restarted and
                      marker.stop_confirmed is c.Fact.CONFIRMED and entry['end_reason'] == c.ErrorCode.DEADLINE.value and
                      marker.cleanup_state in (m.CleanupState.PENDING, m.CleanupState.COMPLETE),
                      'cleanup', 'automatic cleanup prerequisites not satisfied', c.ErrorCode.NOT_AUTHORIZED)
        if marker.cleanup_state is m.CleanupState.COMPLETE:
            c.require(not self._run_path(run_id).exists(), 'cleanup', 'completed run directory reappeared; retain', c.ErrorCode.CONFLICT)
            return m.CleanupResult(c.Outcome.SUCCESS, marker, None)
        self._close_db()
        fs.scan_tree(self._run_path(run_id), self.root)
        pending = replace(marker, cleanup_state=m.CleanupState.PENDING)
        self._write_marker(pending)
        try:
            remove_run(self._run_path(run_id), self.root)
            complete = replace(pending, cleanup_state=m.CleanupState.COMPLETE)
            self._write_marker(complete)
            return m.CleanupResult(c.Outcome.SUCCESS, complete, None)
        except (OSError, c.ContractError) as error:
            self._fault('cleanup interrupted; terminal marker retained: ' + str(error))
            return m.CleanupResult(c.Outcome.UNKNOWN, pending,
                                   c.Reason(c.ErrorCode.STORAGE_FAILURE, 'cleanup', str(error)))
