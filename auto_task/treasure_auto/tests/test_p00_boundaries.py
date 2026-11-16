"""Additional C0 cross-object and evidence boundary regressions."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c, models as m, rules as r
from contract_fixtures import samples as s
from contract_fixtures.fakes import FakeStateStore, ResultConsumer


class BoundaryTests(unittest.TestCase):
    def reject_transaction(self, tx):
        store = FakeStateStore(s.state_snapshot())
        before = store.read_snapshot()
        receipt = store.commit(tx)
        self.assertEqual(receipt.status, m.CommitStatus.REJECTED)
        self.assertEqual(store.read_snapshot(), before)

    def test_R04_release_cannot_leave_same_object_held(self):
        self.reject_transaction(replace(s.release_transaction(), holding=s.state_snapshot().holding))

    def test_R04_old_release_recovery_and_action_rejected(self):
        tx = s.release_transaction()
        for changes in (dict(layout_recovery_round=8), dict(rebuild_generation=8),
                        dict(action_id='old-action'), dict(request_id='old-work')):
            with self.subTest(changes=changes):
                event = replace(tx.release_events[0], envelope=replace(tx.release_events[0].envelope, **changes))
                self.reject_transaction(replace(tx, release_events=(event,)))

    def test_R04_old_control_space_proposal_rejected(self):
        tx = s.release_transaction()
        space = replace(tx.space, envelope=replace(tx.space.envelope, instance_id='old'))
        self.reject_transaction(replace(tx, space=space))

    def test_R04_occupied_footprint_cannot_shrink_without_new_proof(self):
        tx = s.release_transaction()
        tiny = (m.Point2(0, 0), m.Point2(0.001, 0), m.Point2(0, 0.001))
        occ = replace(tx.space.occupancies[0], footprint_m=tiny, evidence=())
        self.reject_transaction(replace(tx, space=replace(tx.space, occupancies=(occ,))))

    def test_R04_inventory_category_cannot_change_on_rebuild(self):
        tx = s.release_transaction()
        self.reject_transaction(replace(tx, inventory=(replace(tx.inventory[0], category=m.Category.BANANA),),
                                        release_events=(), holding=s.state_snapshot().holding))

    def test_R04_evidence_from_different_provenance_not_consumed(self):
        result = replace(s.timeout_result(), evidence=(replace(s.evidence(), origin=c.Origin.LIVE),))
        consumer = ResultConsumer(s.consumer_context())
        with self.assertRaises(c.ContractError):
            consumer.consume(s.envelope(), result)
        self.assertEqual(consumer.updates, 0)

    def test_R04_stillness_duplicate_capture_time_not_two_new_frames(self):
        cert = s.stillness_certificate()
        cert = replace(cert, evidence=(s.evidence(), replace(s.evidence(), evidence_id='second')))
        with self.assertRaises(c.ContractError):
            r.validate_stillness(cert, s.envelope(), s.stamp(111), ('brick-1',), 0)

    def test_R02_intent_contains_full_action_and_receipt_binds_it(self):
        intent = s.arm_intent()
        self.assertEqual(intent.arm_plan.segments[0].speed_mm_s, 1)
        receipt = s.intent_receipt(intent)
        r.validate_intent_receipt(intent, receipt)
        with self.assertRaises(c.ContractError):
            r.validate_intent_receipt(replace(intent, operation=m.IntentOperation.PUMP,
                                            arm_plan=None, pump_enabled=True), receipt)

    def test_R02_intent_forbids_contradictory_command_bodies(self):
        with self.assertRaises(c.ContractError):
            replace(s.arm_intent(), pump_enabled=True)

    def test_R04_holding_update_requires_P07(self):
        tx = s.release_transaction()
        bad = replace(tx.holding_update, envelope=replace(tx.holding_update.envelope, producer=c.Owner.P09))
        self.reject_transaction(replace(tx, holding_update=bad))

    def test_R04_drop_can_resolve_empty_without_arrival(self):
        tx = s.release_transaction()
        unknown = replace(tx.space.occupancies[0], state=m.OccupancyState.UNKNOWN, area=None)
        item = replace(tx.inventory[0], location=m.InventoryLocation.UNKNOWN, area=None)
        tx = replace(tx, release_events=(), space=replace(tx.space, occupancies=(unknown,)), inventory=(item,),
                     holding_update=replace(tx.holding_update, cause=m.HoldingCause.DROP_RECONCILED))
        store = FakeStateStore(s.state_snapshot())
        self.assertEqual(store.commit(tx).status, m.CommitStatus.COMMITTED)
        self.assertEqual(store.read_snapshot().holding.state, c.Holding.EMPTY)
        self.assertEqual(store.read_snapshot().deliveries, ())


if __name__ == '__main__':
    unittest.main()
