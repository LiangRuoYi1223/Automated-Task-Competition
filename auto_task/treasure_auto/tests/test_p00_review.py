"""Independent review regressions: clock trust and missing cross-package paths."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from treasure_auto.contracts import core as c, models as m, rules as r, ports
from contract_fixtures import samples as s
from contract_fixtures.fakes import FakeStateStore, FakeClock


class ReviewTests(unittest.TestCase):
    def test_transaction_cannot_supply_its_own_now_or_boot(self):
        previous = s.state_snapshot()
        tx = s.release_transaction()
        tx = replace(tx, release_events=(), holding=previous.holding, holding_update=None,
                     inventory=previous.inventory)
        for header in (replace(tx.envelope, emitted_at=s.stamp(1, boot_id='new'),
                               expires_at=s.stamp(2, boot_id='new')),
                       replace(tx.envelope, emitted_at=s.stamp(100), expires_at=s.stamp(105))):
            with self.subTest(header=header):
                store = FakeStateStore(previous)
                self.assertEqual(store.commit(replace(tx, envelope=header)).status, m.CommitStatus.REJECTED)

    def test_timer_rejects_nonmonotonic_or_before_start_refresh(self):
        for basis, refresh in ((c.ClockBasis.UTC, 110), (c.ClockBasis.MONOTONIC, 90)):
            def at(t):
                return c.ClockStamp(t, basis, 'clock', 'boot')
            with self.subTest(basis=basis), self.assertRaises(c.ContractError):
                m.TimerRecord(1, 'run', at(100), at(700), at(refresh), 700-refresh,
                              1, 0, m.RunState.RUNNING, False)

    def test_map_origin_is_pose_not_evidence_origin(self):
        pose = c.MapPose('map', 0, 0, 0, 'm', 'rad', 0, 0)
        metadata = m.MapMetadata(s.envelope(c.Owner.P02), 'sim-map-v1', 'sim://map',
                                 0.01, pose, (), m.Measurement.SIMULATED)
        r.require_simulated(metadata)

    def test_P06_has_map_placement_and_support_geometry_ports(self):
        self.assertTrue(hasattr(ports.GeometryPort, 'plan_placement'))
        self.assertTrue(hasattr(ports.GeometryPort, 'support_geometry'))

    def test_budget_usage_persists_atomically_and_cannot_reset(self):
        store = FakeStateStore(s.state_snapshot())
        tx = s.release_transaction()
        tx = replace(tx, budgets=(replace(tx.budgets[0], attempts_used=1, seconds_used=2),))
        self.assertEqual(store.commit(tx).status, m.CommitStatus.COMMITTED)
        next_tx = s.replacement_transaction(store.read_snapshot(), tx, 'next')
        self.assertEqual(store.commit(replace(next_tx, budgets=s.state_snapshot().budgets)).status,
                         m.CommitStatus.REJECTED)

    def test_run_update_cannot_move_original_deadline_or_erase_restart(self):
        previous = s.state_snapshot()
        tx = s.release_transaction()
        shifted = replace(previous.run, started_at=s.stamp(101), deadline=s.stamp(701))
        self.assertEqual(FakeStateStore(previous).commit(replace(tx, run_update=shifted)).status,
                         m.CommitStatus.REJECTED)
        previous = replace(previous, run=replace(previous.run, ever_restarted=True))
        self.assertEqual(FakeStateStore(previous).commit(replace(tx, run_update=s.run_context())).status,
                         m.CommitStatus.REJECTED)

    def test_initial_mapping_can_be_frozen_atomically(self):
        previous = s.state_snapshot()
        old_versions = replace(s.versions(), mapping=None)
        previous = replace(previous, mapping=None,
                    space=replace(previous.space, envelope=replace(previous.space.envelope, versions=old_versions)))
        tx = s.release_transaction()
        tx = replace(tx, envelope=replace(tx.envelope, versions=old_versions),
                     holding=previous.holding, holding_update=None, inventory=previous.inventory,
                     release_events=())
        self.assertEqual(FakeStateStore(previous).commit(tx).status, m.CommitStatus.COMMITTED)

    def test_held_state_requires_same_inventory_in_hand(self):
        previous = s.state_snapshot()
        tx = s.release_transaction()
        tx = replace(tx, holding=previous.holding, holding_update=None, release_events=(),
                     inventory=(replace(previous.inventory[0], location=m.InventoryLocation.CENTRAL),))
        self.assertEqual(FakeStateStore(previous).commit(tx).status, m.CommitStatus.REJECTED)

    def test_nested_layout_provenance_cannot_be_promoted(self):
        def live(record):
            return c.loads(c.dumps(record).replace('"origin":"simulated"', '"origin":"live"'))
        previous = live(s.state_snapshot())
        tx = live(s.release_transaction())
        tx = replace(tx, layouts=(s.layout_pair(m.LayoutKind.V1)[0],))
        with self.assertRaises(c.ContractError):
            r.validate_transaction(tx, previous, s.stamp(110))

    def test_new_actual_occupancy_requires_physical_evidence(self):
        tx = s.release_transaction()
        extra = replace(tx.space.occupancies[0], target_id='unproved-brick',
                        evidence=(replace(s.evidence(), target_id='unproved-brick', kind=c.EvidenceKind.ECHO),))
        tx = replace(tx, space=replace(tx.space, occupancies=tx.space.occupancies + (extra,)))
        self.assertEqual(FakeStateStore(s.state_snapshot()).commit(tx).status, m.CommitStatus.REJECTED)

    def test_P10B_has_explicit_new_run_initialization_port(self):
        self.assertTrue(hasattr(ports.PersistencePort, 'initialize_run'))

    def test_new_space_or_run_proposal_must_be_fresh_at_commit(self):
        tx = s.release_transaction()
        old_space = replace(tx.space, envelope=replace(tx.space.envelope,
                             emitted_at=s.stamp(100), expires_at=s.stamp(105)))
        old_run = replace(s.run_context(), envelope=replace(s.run_context().envelope,
                             emitted_at=s.stamp(100), expires_at=s.stamp(105)))
        for change in (dict(space=old_space), dict(run_update=old_run)):
            with self.subTest(change=change):
                self.assertEqual(FakeStateStore(s.state_snapshot()).commit(replace(tx, **change)).status,
                                 m.CommitStatus.REJECTED)


if __name__ == '__main__':
    unittest.main()
