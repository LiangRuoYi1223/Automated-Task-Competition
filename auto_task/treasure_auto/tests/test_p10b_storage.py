import hashlib
import json
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from test_p10b_support import StoreCase, c, m, r, s, persistence, initial_request


class StorageTests(StoreCase):
    def test_T04_atomic_delivery_dedup_and_rebuild_keep_original_receipt(self):
        tx = s.release_transaction()
        receipt = self.store.commit(tx)
        self.assertEqual(receipt.status, m.CommitStatus.COMMITTED)
        self.assertEqual(self.store.commit(tx), receipt)
        snap = self.store.read('sim-run-1').snapshot
        self.assertEqual((snap.revision, len(snap.deliveries), snap.holding.state), (1, 1, c.Holding.EMPTY))
        self.assertEqual(snap.space.occupancies[0].state, m.OccupancyState.ACTUAL)
        again = s.replacement_transaction(snap, tx, 'rebuild-tx')
        self.assertEqual(self.store.commit(again).status, m.CommitStatus.COMMITTED)
        new = self.store.read('sim-run-1').snapshot
        self.assertEqual(new.deliveries, snap.deliveries)
        self.assertEqual(new.layouts, ())

    def test_T04_conflict_and_invalid_transaction_do_not_half_update(self):
        tx = replace(s.release_transaction(), holding_update=None)
        self.assertEqual(self.store.commit(tx).status, m.CommitStatus.REJECTED)
        self.assertEqual(self.store.read('sim-run-1').snapshot, self.initial.initial_snapshot)
        good = s.release_transaction()
        self.assertEqual(self.store.commit(good).status, m.CommitStatus.COMMITTED)
        bad = replace(good, budgets=())
        self.assertEqual(self.store.commit(bad).reason.code, c.ErrorCode.ID_CONFLICT)
        stale = replace(good, transaction_id='stale', envelope=replace(good.envelope, event_id='stale'))
        self.assertEqual(self.store.commit(stale).reason.code, c.ErrorCode.CONFLICT)

    def test_T01_wal_full_and_durable_snapshot_reopen(self):
        self.assertEqual(self.store.storage_settings(), {'journal_mode': 'wal', 'synchronous': 2})
        self.store.commit(s.release_transaction())
        self.reopen()
        self.assertEqual(len(self.store.read('sim-run-1').snapshot.deliveries), 1)
        self.assertTrue(self.store.actions_blocked)

    def test_T02_intent_is_not_submission_or_physical_success(self):
        intent = s.arm_intent()
        receipt = self.store.commit_intent(intent)
        r.validate_intent_receipt(intent, receipt)
        self.assertEqual(self.store.commit_intent(intent), receipt)
        records = self.store.history('sim-run-1')
        self.assertEqual(sum(item['kind'] == 'intent' for item in records), 1)
        self.assertFalse(any(item['kind'] == 'event' for item in records))
        changed = replace(intent, arm_plan=replace(intent.arm_plan, path_id='another-path'))
        self.assertEqual(self.store.commit_intent(changed).commit.reason.code, c.ErrorCode.ID_CONFLICT)

    def test_T01_all_event_outcomes_and_versions_survive_with_dedup(self):
        for n, outcome in enumerate(c.Outcome):
            result = replace(s.timeout_result(), status=outcome)
            event = m.ActionEvent(s.envelope(event_id='result-' + str(n), sequence=n), result)
            got = self.store.append_event(event)
            self.assertEqual(got.status, m.CommitStatus.COMMITTED)
            self.assertEqual(self.store.append_event(event), got)
        records = [c.loads(x['payload']) for x in self.store.history('sim-run-1') if x['kind'] == 'event']
        self.assertEqual({x.result.status for x in records}, set(c.Outcome))
        self.assertTrue(all(x.envelope.versions == s.versions() for x in records))
        self.assertEqual(self.store.read('sim-run-1').snapshot, self.initial.initial_snapshot)
        conflict = replace(records[0], result=replace(records[0].result, elapsed_s=99))
        self.assertEqual(self.store.append_event(conflict).reason.code, c.ErrorCode.ID_CONFLICT)

    def test_T05_keyframe_written_and_verified_before_reference(self):
        content = b'simulated frame bytes'
        reference = s.evidence(uri='pending:frame', sha256=hashlib.sha256(content).hexdigest())
        saved = self.store.save_keyframe(reference, content)
        from urllib.request import url2pathname
        from urllib.parse import urlparse
        path = Path(url2pathname(urlparse(saved.uri).path))
        self.assertEqual(path.read_bytes(), content)
        event = m.ActionEvent(s.envelope(event_id='frame-event'), replace(s.timeout_result(), evidence=(saved,)))
        self.assertEqual(self.store.append_event(event).status, m.CommitStatus.COMMITTED)
        path.unlink()
        missing = replace(event, envelope=replace(event.envelope, event_id='missing-frame'))
        self.assertNotEqual(self.store.append_event(missing).status, m.CommitStatus.COMMITTED)
        with self.assertRaises(c.ContractError):
            self.store.save_keyframe(replace(reference, evidence_id='oversize'), b'x' * 4097)

    def test_T05_disk_failure_rolls_back_and_latches_fault(self):
        with patch.object(self.store, '_write_snapshot', side_effect=sqlite3.OperationalError('database or disk is full')):
            got = self.store.commit(s.release_transaction())
        self.assertNotEqual(got.status, m.CommitStatus.COMMITTED)
        self.assertTrue(self.store.actions_blocked)
        self.assertEqual(self.store.poll_fault().code, c.ErrorCode.STORAGE_FAILURE)
        self.assertEqual(self.store.read('sim-run-1').snapshot, self.initial.initial_snapshot)
        self.assertNotEqual(self.store.commit_intent(s.arm_intent()).commit.status, m.CommitStatus.COMMITTED)

    def test_T05_bounded_queue_fault_is_observable_without_writer_lock(self):
        entered, release = threading.Event(), threading.Event()
        original = self.store._append_event
        def blocked(value):
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test barrier timed out')
            return original(value)
        event = m.ActionEvent(s.envelope(), s.timeout_result())
        with patch.object(self.store, '_append_event', side_effect=blocked):
            futures = [self.store.submit('append_event', event)]
            self.assertTrue(entered.wait(1))
            try:
                futures += [self.store.submit('append_event', event) for _ in range(9)]
                self.assertTrue(futures[-1].done())
                self.assertNotEqual(futures[-1].result().status, m.CommitStatus.COMMITTED)
                self.assertTrue(self.store.actions_blocked)
                self.assertIsNotNone(self.store.poll_fault())
                # A stop owner needs neither this queue nor a SQLite connection.
                stopped = threading.Event()
                thread = threading.Thread(target=stopped.set)
                thread.start()
                self.assertTrue(stopped.wait(0.2))
                thread.join()
            finally:
                release.set()
            for future in futures:
                future.result(3)

    def test_T03_duplicate_start_never_resets_state_or_creates_missing_run(self):
        receipt = self.store.initialize_run(self.initial)
        self.assertEqual(receipt.status, m.CommitStatus.COMMITTED)
        self.store.commit(s.release_transaction())
        self.assertEqual(self.store.initialize_run(self.initial), receipt)
        self.assertEqual(self.store.read('sim-run-1').snapshot.revision, 1)
        self.reopen()
        self.assertEqual(self.store.initialize_run(self.initial), receipt)
        self.assertTrue(self.store.actions_blocked)


class ConfigTests(StoreCase):
    def test_T05_unconfigured_and_unsafe_configuration_rejected(self):
        values = dict(runtime_root=str(self.root), queue_capacity=2, operation_timeout_s=1,
                      timer_refresh_period_s=1, max_keyframe_bytes=1024)
        for key, value in [('runtime_root', None), ('runtime_root', '.'), ('queue_capacity', 0),
                           ('queue_capacity', True), ('operation_timeout_s', float('nan')),
                           ('timer_refresh_period_s', 0), ('max_keyframe_bytes', -1)]:
            with self.subTest(key=key, value=value), self.assertRaises((ValueError, c.ContractError)):
                self.p.PersistenceConfig(**dict(values, **{key: value}))

    def test_T07_exclusive_process_writer_ownership(self):
        with self.assertRaises(c.ContractError):
            self.open_store()
