"""One bounded queue and one SQLite-owning thread; no ROS or actuator access."""
import json
import queue
import sqlite3
import threading
from concurrent.futures import Future, TimeoutError
from pathlib import Path
from treasure_auto.contracts import core as c, models as m
from . import filesystem as fs
from .operations import Operations
from .lifecycle import Lifecycle


class SQLitePersistence(Operations, Lifecycle):
    def __init__(self, config, clock, *, remaining):
        c.require(callable(getattr(clock, 'now', None)) and callable(remaining),
                  'clock', 'P10A ClockPort and remaining(run, now) policy must be injected', c.ErrorCode.UNCONFIGURED)
        self.config, self.clock, self.remaining = config, clock, remaining
        self.root = fs.checked(Path(config.runtime_root))
        self.root.mkdir(parents=True, exist_ok=True)
        self._lease = fs.RootLease(self.root / '.writer.lock')
        self._db = None
        self._db_run = None
        self._authorized = False
        self._blocked = threading.Event()
        self._blocked.set()
        self._faulted = threading.Event()
        self._faults = queue.SimpleQueue()
        self._queue = queue.Queue(maxsize=config.queue_capacity)
        self._admission = threading.Lock()
        self._accepting = True
        try:
            catalog_path = self.root / 'catalog.json'
            if catalog_path.exists():
                self._catalog = fs.read_json(catalog_path)
                c.require(self._catalog['format'] == 1 and self._catalog['contract'] == c.CONTRACT_VERSION,
                          'catalog', 'unsupported storage/contract version', c.ErrorCode.VERSION_MISMATCH)
            else:
                c.require(set(p.name for p in self.root.iterdir()) <= {'.writer.lock'},
                          'catalog', 'unowned/nonempty root or missing catalog; preserve it', c.ErrorCode.STORAGE_FAILURE)
                self._catalog = dict(format=1, contract=c.CONTRACT_VERSION, active=None, runs={}, starts={})
                self._save_catalog()
            for name in ('runs', 'ends'):
                fs.checked(self.root / name).mkdir(exist_ok=True)
            self._worker = threading.Thread(target=self._work, name='P10B-single-writer', daemon=True)
            self._worker.start()
        except Exception:
            self._lease.close()
            raise

    @property
    def actions_blocked(self):
        """Lock-free observation for P10A; never a positive motion permit."""
        return not self._accepting or self._blocked.is_set() or self._faulted.is_set()

    def poll_fault(self):
        try:
            return self._faults.get_nowait()
        except queue.Empty:
            return None

    def _fault(self, detail):
        self._blocked.set()
        self._faulted.set()
        reason = c.Reason(c.ErrorCode.STORAGE_FAILURE, 'persistence', str(detail))
        self._faults.put(reason)
        return reason

    def _save_catalog(self):
        fs.write_json(self.root / 'catalog.json', self._catalog)

    def _run_path(self, run_id):
        return fs.checked(self.root / 'runs' / fs.run_name(run_id), self.root)

    def _open_run(self, run_id):
        c.require(run_id in self._catalog['runs'], 'run_id', 'unknown run', c.ErrorCode.STALE_RUN)
        path = self._run_path(run_id) / 'state.sqlite3'
        fs.checked(path, self.root)
        for suffix in ('-wal', '-shm', '-journal'):
            fs.checked(Path(str(path) + suffix), self.root)
        c.require(path.is_file(), 'snapshot', 'database missing; never recreate', c.ErrorCode.STORAGE_FAILURE)
        if self._db is not None and self._db_run == run_id:
            return self._db
        self._close_db()
        # mode=rw is essential: reading a missing database must not create one.
        self._db = sqlite3.connect(path.as_uri() + '?mode=rw', uri=True, isolation_level=None,
                                   timeout=self.config.operation_timeout_s)
        self._db_run = run_id
        try:
            self._configure_db()
            c.require(self._db.execute('PRAGMA user_version').fetchone()[0] == 1,
                      'storage_version', 'unsupported database format', c.ErrorCode.VERSION_MISMATCH)
        except Exception:
            self._close_db()
            raise
        return self._db

    def _configure_db(self):
        mode = self._db.execute('PRAGMA journal_mode=WAL').fetchone()[0]
        self._db.execute('PRAGMA synchronous=FULL')
        full = self._db.execute('PRAGMA synchronous').fetchone()[0]
        c.require(mode.lower() == 'wal' and full == 2, 'sqlite', 'WAL/FULL unavailable', c.ErrorCode.STORAGE_FAILURE)
        c.require(self._db.execute('PRAGMA quick_check').fetchone()[0] == 'ok',
                  'sqlite', 'integrity check failed', c.ErrorCode.STORAGE_FAILURE)

    def _close_db(self):
        if self._db is not None:
            self._db.close()
            self._db = self._db_run = None

    def _snapshot(self, run_id):
        db = self._open_run(run_id)
        row = db.execute('SELECT payload, sha256 FROM snapshot WHERE id=1').fetchone()
        c.require(row is not None and fs.digest(row[0].encode('utf-8')) == row[1],
                  'snapshot', 'snapshot missing or hash mismatch', c.ErrorCode.STORAGE_FAILURE)
        result = c.loads(row[0])
        c.require(isinstance(result, m.StateSnapshot) and result.run.envelope.run_id == run_id,
                  'snapshot', 'wrong type/run', c.ErrorCode.STALE_RUN)
        return result

    def _write_snapshot(self, snapshot):
        payload = c.dumps(snapshot)
        self._db.execute('INSERT OR REPLACE INTO snapshot VALUES(1, ?, ?)',
                         (payload, fs.digest(payload.encode('utf-8'))))

    def _log(self, kind, value, result=None):
        payload = c.dumps(value)
        self._db.execute('INSERT INTO history(kind,payload,sha256,result) VALUES(?,?,?,?)',
                         (kind, payload, fs.digest(payload.encode('utf-8')), None if result is None else c.dumps(result)))

    def _lookup(self, kind, key, value):
        row = self._db.execute('SELECT payload,result FROM receipts WHERE kind=? AND key=?', (kind, key)).fetchone()
        if row:
            c.require(row[0] == c.dumps(value), kind, 'identity reused with different payload', c.ErrorCode.ID_CONFLICT)
            return c.loads(row[1])
        return None

    def _remember(self, kind, key, value, result):
        self._db.execute('INSERT INTO receipts VALUES(?,?,?,?)', (kind, key, c.dumps(value), c.dumps(result)))

    def _atomic(self, work):
        self._db.execute('BEGIN IMMEDIATE')
        try:
            result = work()
            self._db.execute('COMMIT')
            return result
        except BaseException:
            if self._db.in_transaction:
                self._db.execute('ROLLBACK')
            raise

    def _failure(self, operation, value, reason, unknown=False):
        status = m.CommitStatus.UNKNOWN if unknown else m.CommitStatus.REJECTED
        if operation in ('save_keyframe', 'seal', 'history', 'storage_settings'):
            raise c.ContractError(reason.code, reason.field, reason.detail)
        if operation == 'read':
            return m.ReadResult(c.Outcome.UNKNOWN, None, reason)
        if operation == 'register_recovery':
            return m.RecoveryResult(c.Outcome.UNKNOWN if unknown else c.Outcome.REJECTED, None, None, None, reason)
        if operation == 'cleanup':
            return m.CleanupResult(c.Outcome.UNKNOWN if unknown else c.Outcome.REJECTED, value.marker, reason)
        transaction_id = getattr(value, 'transaction_id', operation)
        receipt = m.CommitReceipt(transaction_id, status, None, None, reason)
        if operation == 'commit_intent':
            from treasure_auto.contracts.rules import intent_digest
            return m.IntentReceipt(value.envelope.run_id, value.envelope.action_id, intent_digest(value), receipt)
        return receipt

    def submit(self, operation, value=None):
        """Nonblocking adapter for P11. Future acceptance is NOT a commit receipt."""
        allowed = {'initialize_run', 'commit', 'commit_intent', 'append_event', 'append_record', 'read',
                   'save_keyframe', 'refresh_timer', 'register_recovery', 'seal', 'cleanup', 'history', 'storage_settings'}
        c.require(operation in allowed, 'operation', 'unknown persistence operation')
        future = Future()
        with self._admission:
            if not self._accepting:
                reason = c.Reason(c.ErrorCode.UNAVAILABLE, 'writer', 'writer closed')
            else:
                try:
                    self._queue.put_nowait((operation, value, future))
                    return future
                except queue.Full:
                    reason = self._fault('bounded event queue full; new saved-dependent actions paused')
        try:
            future.set_result(self._failure(operation, value, reason))
        except Exception as error:
            future.set_exception(error)
        return future

    def _call(self, operation, value=None):
        future = self.submit(operation, value)
        try:
            return future.result(self.config.operation_timeout_s)
        except TimeoutError:
            # The writer may still commit. Never report failure as proof of no action.
            return self._failure(operation, value, self._fault('writer wait timed out; query before retry'), True)

    def _work(self):
        try:
            while self._accepting or not self._queue.empty():
                try:
                    operation, value, future = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                if not future.set_running_or_notify_cancel():
                    self._queue.task_done()
                    continue
                try:
                    result = getattr(self, '_' + operation)(value)
                except c.ContractError as error:
                    reason = c.Reason(error.code, error.field, error.detail)
                    storage_error = error.code is c.ErrorCode.STORAGE_FAILURE
                    if storage_error:
                        self._fault(str(error))
                    elif self._db is not None:
                        try:
                            audit = value[0] if operation == 'save_keyframe' else value
                            if isinstance(audit, c.Record):
                                self._log('rejected:' + operation, audit, reason)
                        except Exception as audit_error:
                            reason = self._fault('cannot persist rejection: ' + str(audit_error))
                            storage_error = True
                    try:
                        result = self._failure(operation, value, reason, storage_error)
                    except Exception as failure:
                        future.set_exception(failure)
                        self._queue.task_done()
                        continue
                except Exception as error:
                    reason = self._fault(str(error))
                    try:
                        result = self._failure(operation, value, reason, True)
                    except Exception as failure:
                        future.set_exception(failure)
                        self._queue.task_done()
                        continue
                future.set_result(result)
                self._queue.task_done()
        finally:
            self._close_db()
            self._lease.close()

    def close(self):
        """Stop admission, drain accepted work, then release all connections/lease."""
        with self._admission:
            self._accepting = False
            self._blocked.set()
        self._worker.join(self.config.operation_timeout_s)
        if self._worker.is_alive():
            self._fault('writer still in flight; root retained and lease not released')
            return False
        return True

    def _read(self, run_id):
        try:
            marker = self._marker(run_id)
            if marker is not None and not self._run_path(run_id).exists():
                return m.ReadResult(c.Outcome.REJECTED, None,
                                    c.Reason(c.ErrorCode.DEADLINE, 'run_id', 'ended run data removed; marker retained'))
            return m.ReadResult(c.Outcome.SUCCESS, self._snapshot(run_id), None)
        finally:
            if run_id in self._catalog['runs'] and self._catalog['runs'][run_id]['phase'] == 'ended':
                self._close_db()

    def _history(self, run_id):
        db = self._open_run(run_id)
        output = []
        for kind, payload, digest, result in db.execute('SELECT kind,payload,sha256,result FROM history ORDER BY id'):
            c.require(fs.digest(payload.encode('utf-8')) == digest, 'history', 'event hash mismatch', c.ErrorCode.STORAGE_FAILURE)
            c.loads(payload)
            output.append(dict(kind=kind, payload=payload, result=result))
        if self._catalog['runs'][run_id]['phase'] == 'ended':
            self._close_db()
        return tuple(output)

    def _storage_settings(self, unused):
        c.require(self._db is not None, 'sqlite', 'no active connection')
        return dict(journal_mode=self._db.execute('PRAGMA journal_mode').fetchone()[0],
                    synchronous=self._db.execute('PRAGMA synchronous').fetchone()[0])

    def initialize_run(self, request): return self._call('initialize_run', request)
    def commit(self, transaction): return self._call('commit', transaction)
    def commit_intent(self, intent): return self._call('commit_intent', intent)
    def append_event(self, event): return self._call('append_event', event)
    def append_record(self, record): return self._call('append_record', record)
    def read(self, run_id): return self._call('read', run_id)
    def save_keyframe(self, reference, content): return self._call('save_keyframe', (reference, content))
    def refresh_timer(self, record): return self._call('refresh_timer', record)
    def register_recovery(self, request): return self._call('register_recovery', request)
    def seal(self, request): return self._call('seal', request)
    def cleanup(self, request): return self._call('cleanup', request)
    def history(self, run_id): return self._call('history', run_id)
    def storage_settings(self): return self._call('storage_settings')
