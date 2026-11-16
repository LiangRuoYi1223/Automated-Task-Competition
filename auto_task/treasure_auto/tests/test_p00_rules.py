"""P00-T02/T03: consumer rejection, evidence provenance and atomic shared state."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c
try:
    from treasure_auto.contracts import models as m, rules as r
    from contract_fixtures import samples as s
    from contract_fixtures.fakes import FakeClock, FakeStateStore, ResultConsumer
except ImportError:
    s = None


class RulesTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(s, 'P00 rules/fakes are not implemented')

    def test_T02_old_run_instance_epoch_and_expired_no_update(self):
        for changes in (dict(run_id='old-run'), dict(instance_id='old-instance'),
                        dict(control_epoch=0), dict(expires_at=s.stamp(110)),
                        dict(request_id='old-request'), dict(action_id='old-action')):
            consumer = ResultConsumer(s.consumer_context())
            with self.subTest(changes=changes), self.assertRaises(c.ContractError):
                consumer.consume(replace(s.envelope(), **changes), s.timeout_result())
            self.assertIsNone(consumer.current)

    def test_T02_future_wrong_clock_version_and_recovery_rejected(self):
        for changes in (dict(emitted_at=s.stamp(115)),
                        dict(layout_recovery_round=1), dict(rebuild_generation=1),
                        dict(base_motion_epoch=1),
                        dict(versions=replace(s.versions(), space=2))):
            with self.subTest(changes=changes), self.assertRaises(c.ContractError):
                r.validate_envelope(replace(s.envelope(), **changes), s.consumer_context())
        with self.assertRaises(c.ContractError):
            r.validate_envelope(s.envelope(), replace(s.consumer_context(), now=s.stamp(111, boot_id='new')))

    def test_T02_duplicate_result_does_not_update_twice(self):
        consumer = ResultConsumer(s.consumer_context())
        self.assertTrue(consumer.consume(s.envelope(), s.timeout_result()))
        self.assertFalse(consumer.consume(s.envelope(), s.timeout_result()))
        self.assertEqual(consumer.updates, 1)
        with self.assertRaises(c.ContractError):
            consumer.consume(s.envelope(), replace(s.timeout_result(), elapsed_s=9))

    def test_T02_out_of_order_sequence_rejected(self):
        consumer = ResultConsumer(s.consumer_context())
        consumer.consume(s.envelope(), s.timeout_result())
        with self.assertRaises(c.ContractError):
            consumer.consume(replace(s.envelope(), event_id='other', sequence=0), s.timeout_result())
        self.assertEqual(consumer.updates, 1)

    def test_R04_both_consumers_require_same_space_version(self):
        space = s.space_snapshot()
        r.require_space_version(s.envelope(), space)
        with self.assertRaises(c.ContractError):
            r.require_space_version(replace(s.envelope(), versions=replace(s.versions(), space=2)), space)

    def test_R04_physical_evidence_cannot_be_echo_or_simulated_live(self):
        for evidence, origin in ((replace(s.evidence(), kind=c.EvidenceKind.ECHO), c.Origin.SIMULATED),
                                 (s.evidence(), c.Origin.LIVE)):
            with self.subTest(origin=origin), self.assertRaises(c.ContractError):
                r.require_physical_evidence(evidence, s.envelope(), s.stamp(111), origin)

    def test_R04_expired_wrong_target_and_wrong_action_evidence_rejected(self):
        for changes in (dict(valid_until=s.stamp(110)), dict(target_id='other'),
                        dict(action_id='other'), dict(run_id='other')):
            with self.subTest(changes=changes), self.assertRaises(c.ContractError):
                r.require_physical_evidence(replace(s.evidence(), **changes), s.envelope(),
                                            s.stamp(111), c.Origin.SIMULATED)

    def test_R04_stillness_500ms_is_capture_time_and_needs_coverage(self):
        cert = s.stillness_certificate()
        r.validate_stillness(cert, s.envelope(), s.stamp(111), ('brick-1',), 0)
        for changes in (dict(capture_end=s.sensor_stamp(20.499)),
                        dict(continuous=c.Fact.UNKNOWN), dict(all_stopped=c.Fact.UNKNOWN),
                        dict(covered_target_ids=()), dict(scene_epoch=1)):
            with self.subTest(changes=changes), self.assertRaises(c.ContractError):
                r.validate_stillness(replace(cert, **changes), s.envelope(), s.stamp(111), ('brick-1',), 0)

    def test_T03_duplicate_transaction_returns_same_receipt(self):
        store = FakeStateStore(s.state_snapshot())
        tx = s.release_transaction()
        receipt = store.commit(tx)
        self.assertEqual(store.commit(tx), receipt)
        self.assertEqual(len(store.read_snapshot().deliveries), 1)
        self.assertEqual(store.read_snapshot().revision, 1)

    def test_T03_renumber_and_new_release_event_no_new_arrival(self):
        store = FakeStateStore(s.state_snapshot())
        tx = s.release_transaction()
        store.commit(tx)
        tx2 = s.replacement_transaction(store.read_snapshot(), tx, 'tx-2')
        receipt = store.commit(tx2)
        self.assertEqual(receipt.status, m.CommitStatus.COMMITTED)
        self.assertEqual(len(store.read_snapshot().deliveries), 1)

    def test_T03_unresolved_identity_cannot_create_arrival(self):
        tx = s.release_transaction()
        item = replace(tx.inventory[0], identity_confirmed=c.Fact.UNKNOWN)
        store = FakeStateStore(s.state_snapshot())
        result = store.commit(replace(tx, inventory=(item,)))
        self.assertEqual(result.status, m.CommitStatus.REJECTED)
        self.assertEqual(store.read_snapshot().revision, 0)

    def test_R04_write_failure_and_version_conflict_leave_everything_unchanged(self):
        for fail, revision in ((True, 0), (False, 9)):
            store = FakeStateStore(s.state_snapshot())
            before = store.read_snapshot()
            store.fail_next = fail
            receipt = store.commit(replace(s.release_transaction(), expected_revision=revision))
            self.assertNotEqual(receipt.status, m.CommitStatus.COMMITTED)
            self.assertEqual(store.read_snapshot(), before)

    def test_R04_duplicate_transaction_id_conflict_rejected(self):
        store = FakeStateStore(s.state_snapshot())
        tx = s.release_transaction()
        store.commit(tx)
        conflict = replace(tx, inventory=())
        self.assertEqual(store.commit(conflict).status, m.CommitStatus.REJECTED)
        self.assertEqual(store.read_snapshot().revision, 1)

    def test_R04_occupancy_cannot_clear_without_new_evidence(self):
        store = FakeStateStore(s.state_snapshot())
        tx = s.release_transaction()
        tx = replace(tx, space=replace(tx.space, occupancies=()), release_events=(),
                     clearing_evidence=())
        self.assertEqual(store.commit(tx).status, m.CommitStatus.REJECTED)
        self.assertEqual(len(store.read_snapshot().space.occupancies), 1)

    def test_R04_pump_echo_cannot_create_arrival_or_empty(self):
        store = FakeStateStore(s.state_snapshot())
        tx = s.release_transaction()
        release = replace(tx.release_events[0], evidence=(replace(s.evidence(), kind=c.EvidenceKind.ECHO),))
        self.assertEqual(store.commit(replace(tx, release_events=(release,))).status, m.CommitStatus.REJECTED)
        self.assertEqual(store.read_snapshot().holding.state, c.Holding.HELD)

    def test_R04_release_after_original_deadline_rejected(self):
        tx = s.release_transaction()
        release = replace(tx.release_events[0], released_at=s.stamp(700))
        store = FakeStateStore(s.state_snapshot())
        self.assertEqual(store.commit(replace(tx, release_events=(release,))).status, m.CommitStatus.REJECTED)

    def test_R05_fake_clock_never_sleeps_or_moves_backwards(self):
        clock = FakeClock(s.stamp(100))
        self.assertEqual(clock.advance(600).seconds, 700)
        with self.assertRaises(c.ContractError):
            clock.advance(-1)


if __name__ == '__main__':
    unittest.main()
