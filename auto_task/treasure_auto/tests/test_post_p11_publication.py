"""Actual SQLite publications must be consumable without renaming their owner."""
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from test_p09_support import c, m, r, s, FakeClock, release_case
from publication_support import snapshot_for_space
from treasure_auto.persistence import SQLitePersistence, PersistenceConfig
from treasure_auto.mission.transactions import ReleaseCoordinator


class PublicationIntegrationTests(unittest.TestCase):
    def store(self, clock, snapshot):
        temp = tempfile.TemporaryDirectory(prefix='post-p11-publication-')
        self.addCleanup(temp.cleanup)
        store = SQLitePersistence(PersistenceConfig(str(Path(temp.name) / 'runtime'),
            16, 2., 1., 4096), clock, remaining=r.remaining_seconds)
        self.addCleanup(store.close)
        run = snapshot.run
        signal = m.RuntimeSignal(replace(run.envelope, event_id=run.start_signal_id),
                                 m.SignalKind.START, c.Fact.CONFIRMED)
        receipt = store.initialize_run(m.InitializeRunRequest(run, snapshot, signal, None))
        self.assertEqual(receipt.status, m.CommitStatus.COMMITTED, receipt.reason)
        return store

    def test_actual_sqlite_p08_space_reaches_geometry(self):
        from test_p06_paths import PathFixture
        f = PathFixture(); f.setUp()
        self.addCleanup(f.doCleanups)
        space = replace(f.ctx.space, envelope=replace(f.ctx.space.envelope, producer=c.Owner.P08))
        f.ctx = replace(f.ctx, space=space)
        store = self.store(FakeClock(s.stamp(110)), snapshot_for_space(space))
        f.svc.space_readback = lambda candidate: store.read(candidate.envelope.run_id)
        result = f.svc.plan(f.request())
        self.assertEqual(result.result.status, c.Outcome.SUCCESS, result.result.reason)
        self.assertEqual(store.read(space.envelope.run_id).snapshot.space.envelope.producer, c.Owner.P08)

    def test_actual_sqlite_p08_space_reaches_navigation(self):
        import test_p03_support as p3
        nav, clock, gateway = p3.service()
        req = p3.request()
        req = replace(req, space=replace(req.space, envelope=replace(req.space.envelope, producer=c.Owner.P08)))
        store = self.store(clock, snapshot_for_space(req.space))
        nav.space_readback = lambda candidate: store.read(candidate.envelope.run_id)
        result = nav.navigate(req)
        self.assertEqual(result.result.accepted, c.Fact.CONFIRMED, result.result.reason)
        self.assertIsNotNone(nav.goal_intent)

    def test_actual_sqlite_p08_space_reaches_mapping(self):
        from test_p02_service import ServiceTests
        import test_p02_support as p2
        f = ServiceTests(); f.setUp(); f.competition(); f.ready(); f.purpose_ready()
        self.addCleanup(f.doCleanups)
        space = p2.space(t=f.clock.t)
        space = replace(space, envelope=replace(space.envelope, producer=c.Owner.P08))
        store = self.store(SimpleNamespace(now=f.clock), snapshot_for_space(space))
        f.service.space_readback = lambda candidate: store.read(candidate.envelope.run_id)
        result = f.service.query(f.query(), space)
        self.assertEqual(result.valid, c.Fact.CONFIRMED, result.reason)

    def test_uncommitted_geometry_space_is_rejected_even_with_p08_owner(self):
        from test_p06_paths import PathFixture
        f = PathFixture(); f.setUp(); self.addCleanup(f.doCleanups)
        space = replace(f.ctx.space, envelope=replace(f.ctx.space.envelope, producer=c.Owner.P08))
        f.ctx = replace(f.ctx, space=space)
        f.svc.space_readback = None
        self.assertIsNone(f.svc.plan(f.request()).plan)

    def test_navigation_adopts_actual_committed_next_version_without_resetting_budget(self):
        import test_p03_support as p3
        nav, clock, gateway = p3.service()
        initial = replace(snapshot_for_space(p3.space()), run=nav.run)
        store = self.store(clock, initial)
        next_space = p3.space(version=1)
        next_run = replace(nav.run, envelope=replace(nav.run.envelope, versions=next_space.envelope.versions))
        tx = m.StateTransaction(replace(initial.space.envelope, producer=c.Owner.P09),
            'post-p11-nav-space', 0, 0, next_space, initial.holding, (), (), None, (), (), None, (), next_run)
        self.assertEqual(store.commit(tx).status, m.CommitStatus.COMMITTED)
        nav.space_readback = lambda space: store.read(space.envelope.run_id)
        nav._attempts, nav._recovery_used = 1, .4
        budget, deadline = nav.budget, nav.run.deadline
        check = nav.adopt_committed_space(next_space)
        self.assertTrue(check.accepted, check.reason)
        req = p3.request(versions=next_space.envelope.versions)
        req = replace(req, space=next_space,
            localization=replace(req.localization, envelope=replace(req.localization.envelope, versions=next_space.envelope.versions)),
            readiness=replace(req.readiness, envelope=replace(req.readiness.envelope, versions=next_space.envelope.versions)))
        # Health is an external simulated boundary; no gateway version rewrite.
        old_health = gateway.health
        gateway.health = lambda: replace(old_health(), envelope=replace(old_health().envelope, versions=next_space.envelope.versions))
        self.assertEqual(nav.navigate(req).result.accepted, c.Fact.CONFIRMED)
        self.assertEqual(nav.budget, budget)
        self.assertEqual(nav.run.deadline, deadline)

    def test_navigation_publication_cannot_extend_deadline_or_change_controller(self):
        import test_p03_support as p3
        for change in ('deadline','controller'):
            nav, clock, gateway = p3.service()
            old = nav.run
            space = p3.space(version=1)
            run = replace(old, envelope=replace(old.envelope, versions=space.envelope.versions))
            if change == 'deadline':
                run = replace(run, started_at=replace(run.started_at, seconds=run.started_at.seconds+1),
                              deadline=replace(run.deadline, seconds=run.deadline.seconds+1))
            else:
                run = replace(run, envelope=replace(run.envelope, instance_id='other'))
            snapshot = replace(snapshot_for_space(space), run=run)
            nav.space_readback = lambda candidate: m.ReadResult(c.Outcome.SUCCESS, snapshot, None)
            self.assertFalse(nav.adopt_committed_space(space).accepted)
            self.assertEqual(nav.run, old)

    def test_release_without_runtime_proposal_never_commits(self):
        from contract_fixtures.fakes import FakeStateStore
        before, order, placed, update = release_case()
        clock = FakeClock(s.stamp(110))
        store = FakeStateStore(before, clock)
        result = ReleaseCoordinator(store, clock).commit_release(order, placed, update)
        self.assertEqual(result.receipt.status, m.CommitStatus.REJECTED)
        self.assertEqual(store.read(before.run.envelope.run_id).snapshot, before)

    def test_readback_content_mismatch_blocks_geometry(self):
        from test_p06_paths import PathFixture
        f = PathFixture(); f.setUp(); self.addCleanup(f.doCleanups)
        space = replace(f.ctx.space, envelope=replace(f.ctx.space.envelope, producer=c.Owner.P08))
        f.ctx = replace(f.ctx, space=space)
        published = snapshot_for_space(replace(space, envelope=replace(space.envelope, event_id='other-publication')))
        f.svc.space_readback = lambda candidate: m.ReadResult(c.Outcome.SUCCESS, published, None)
        self.assertIsNone(f.svc.plan(f.request()).plan)

    def test_release_atomically_updates_runtime_space_version_without_extending_deadline(self):
        from test_p10a_support import config_data, signal, initial_snapshot
        from treasure_auto.runtime import RuntimeConfig, RuntimeController
        clock = FakeClock(s.stamp(109))
        ids = iter(['sim-run-1'] + ['post-p11-' + str(i) for i in range(100)])
        rt = RuntimeController(RuntimeConfig.from_dict(config_data()), clock, s.versions(),
            'sim-instance-1', 1, id_factory=lambda: next(ids))
        rt.arm(); rt.signal(signal(rt, clock, event='sim-start-1'))
        before, order, placed, update = release_case()
        before = replace(before, run=rt.run, mapping=initial_snapshot(rt.run).mapping)
        # Runtime initializes through the real sole writer, not a replacement store.
        temp = tempfile.TemporaryDirectory(prefix='post-p11-runtime-')
        self.addCleanup(temp.cleanup)
        store = SQLitePersistence(PersistenceConfig(str(Path(temp.name) / 'runtime'),
            16, 2., 1., 4096), clock, remaining=rt.remaining)
        self.addCleanup(store.close)
        rt.initialize(store, before)
        before = store.read('sim-run-1').snapshot
        clock.advance(1)
        coordinator = ReleaseCoordinator(store, clock)
        coordinator.run_update_provider = lambda previous, space: rt.propose_space_update(previous, space)
        out = coordinator.commit_release(order, placed, update)
        self.assertEqual(out.receipt.status, m.CommitStatus.COMMITTED, out.receipt.reason)
        self.assertEqual(out.snapshot.run.envelope.versions.space, 1)
        self.assertEqual(out.snapshot.run.deadline, before.run.deadline)
        rt.accept_snapshot(out.snapshot, out.receipt)
        self.assertEqual(rt.run.envelope.versions.space, 1)
        self.assertEqual(rt.run.deadline, before.run.deadline)
        self.assertEqual(coordinator.commit_release(order, placed, update), out)


if __name__ == '__main__':
    unittest.main()
