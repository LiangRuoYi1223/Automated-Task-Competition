"""P11 acceptance evidence and real-code boundaries, with simulated I/O only."""
from publication_support import simulated_run_proposal
import importlib
import importlib.util
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS.parent / 'src'))
OUTPUT = TESTS / 'integration' / 'acceptance'

from treasure_auto.contracts import core as c, models as m, rules as r
from contract_fixtures import samples as s


class AcceptanceRunnerTests(unittest.TestCase):
    """A failing/omitted/skipped method must never produce an acceptance PASS."""

    def api(self):
        spec = importlib.util.find_spec('integration.acceptance')
        self.assertIsNotNone(spec, 'P11 acceptance runner must be executable')
        module = importlib.import_module('integration.acceptance')
        self.assertTrue(callable(getattr(module, 'run_acceptance', None)),
                        'P11 acceptance runner must aggregate actual method results')
        return module

    def matrix(self):
        return {'schema_version': 1, 'contract_version': '1.0.0', 'scenarios': [
            {'id': 'E{:02}'.format(i), 'title': 'controlled runner test',
             'requirements': ['P11-R04'], 'packages': ['P11'],
             'cases': [{'id': 'runner_fixture.Case.test_e{:02}'.format(i),
                        'level': 'component', 'real_packages': ['P11'],
                        'proves': 'runner result handling'}],
             'remaining_gaps': ['Formal and hardware validation not performed.']}
            for i in range(1, 17)]}

    def loader(self, statuses, observed):
        class Loader:
            def loadTestsFromName(self, name):
                status = statuses.get(name, 'PASS')
                if status == 'MISSING':
                    return unittest.TestSuite()
                class ControlledCase(unittest.TestCase):
                    def id(self):
                        return name if status != 'WRONG_ID' else name + '_other'

                    def runTest(self):
                        observed.append(name)
                        if status == 'FAIL':
                            self.fail('deliberately failing fixture')
                        if status == 'SKIP':
                            self.skipTest('deliberately unavailable fixture')
                        if status == 'ERROR':
                            raise RuntimeError('deliberate boundary error')
                        if status == 'SUBTEST_FAIL':
                            with self.subTest(boundary='unknown'):
                                self.fail('deliberate failed subtest')
                return unittest.TestSuite([ControlledCase()])
        return Loader()

    def execute(self, matrix, statuses=None):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.TemporaryDirectory(prefix='runner-check-', dir=str(OUTPUT))
        self.addCleanup(tmp.cleanup)
        observed = []
        report = self.api().run_acceptance(
            matrix, Path(tmp.name), loader=self.loader(statuses or {}, observed))
        return report, observed, Path(tmp.name)

    def test_failure_runs_all_sixteen_rows_and_writes_exact_results(self):
        first = 'runner_fixture.Case.test_e01'
        report, observed, folder = self.execute(self.matrix(), {first: 'FAIL'})
        self.assertEqual(len(observed), 16)
        self.assertEqual([r['id'] for r in report['rows']], ['E{:02}'.format(i) for i in range(1, 17)])
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['executed_count'], 16)
        self.assertEqual(report['rows'][0]['results'][0]['id'], first)
        self.assertEqual(report['rows'][0]['results'][0]['status'], 'FAIL')
        self.assertEqual(report['rows'][-1]['status'], 'PASS')
        self.assertEqual(json.loads((folder / 'acceptance_results.json').read_text(encoding='utf-8')), report)
        self.assertIn('deliberately failing fixture', (folder / 'acceptance.log').read_text(encoding='utf-8'))

    def test_missing_or_wrong_collection_never_passes_and_later_rows_run(self):
        for problem in ('MISSING', 'WRONG_ID'):
            with self.subTest(problem=problem):
                report, observed, _ = self.execute(self.matrix(), {'runner_fixture.Case.test_e02': problem})
                self.assertEqual(report['status'], 'FAIL')
                self.assertEqual(report['rows'][1]['status'], 'FAIL')
                self.assertTrue(report['rows'][1]['collection_errors'])
                self.assertEqual(report['executed_count'], 15)
                self.assertIn('runner_fixture.Case.test_e16', observed)

    def test_malformed_case_ids_and_types_fail_collection_without_aborting_later_rows(self):
        invalid_cases = ({}, {'id': None}, {'id': 42}, {'id': []}, {'id': {}}, None,
                         {'id': 'runner_fixture.Case.test_extra', 'level': [], 'proves': 'invalid level'},
                         {'id': 'runner_fixture.Case.test_extra', 'level': {}, 'proves': 'invalid level'})
        for invalid in invalid_cases:
            with self.subTest(invalid=invalid):
                matrix = self.matrix()
                matrix['scenarios'][0]['cases'].append(invalid)
                try:
                    report, observed, folder = self.execute(matrix)
                except Exception as error:
                    self.fail('malformed case aborted the matrix: {}: {}'.format(type(error).__name__, error))
                self.assertEqual(report['status'], 'FAIL')
                self.assertEqual(len(report['rows']), 16)
                self.assertEqual(report['executed_count'], 16)
                self.assertEqual(len(observed), 16)
                self.assertEqual(report['rows'][0]['status'], 'FAIL')
                self.assertTrue(report['rows'][0]['collection_errors'])
                self.assertTrue(all(isinstance(name, str) for name in report['rows'][0]['requested_ids']))
                self.assertEqual(report['rows'][-1]['status'], 'PASS')
                self.assertEqual(json.loads((folder / 'acceptance_results.json').read_text(encoding='utf-8')), report)
                log = (folder / 'acceptance.log').read_text(encoding='utf-8')
                self.assertIn('COLLECTION FAILURE', log)
                self.assertIn('ROW RESULT: E16 PASS', log)

    def test_malformed_remaining_gaps_cannot_abort_or_retain_a_previous_pass_report(self):
        for invalid in (None, 42, 'not a list', {'gap': 'not a list'}, ['valid gap', None]):
            with self.subTest(invalid=invalid):
                matrix = self.matrix()
                matrix['scenarios'][0]['remaining_gaps'] = invalid
                OUTPUT.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='malformed-gaps-', dir=str(OUTPUT)) as folder:
                    folder = Path(folder)
                    (folder / 'acceptance_results.json').write_text('{"status": "PASS"}', encoding='utf-8')
                    (folder / 'acceptance.log').write_text('PREVIOUS PASS', encoding='utf-8')
                    observed = []
                    try:
                        report = self.api().run_acceptance(matrix, folder, loader=self.loader({}, observed))
                    except Exception as error:
                        self.fail('malformed remaining_gaps aborted evidence export: {}: {}'.format(
                            type(error).__name__, error))
                    self.assertEqual(report['status'], 'FAIL')
                    self.assertEqual(len(report['rows']), 16)
                    self.assertEqual(report['executed_count'], 16)
                    self.assertEqual(len(observed), 16)
                    first = report['rows'][0]
                    self.assertEqual(first['status'], 'FAIL')
                    self.assertTrue(any('remaining_gaps' in error for error in first['collection_errors']))
                    self.assertIsInstance(first['remaining_gaps'], list)
                    self.assertTrue(first['remaining_gaps'])
                    self.assertTrue(all(isinstance(gap, str) for gap in first['remaining_gaps']))
                    self.assertEqual(report['rows'][-1]['status'], 'PASS')
                    self.assertEqual(json.loads((folder / 'acceptance_results.json').read_text(encoding='utf-8')), report)
                    log = (folder / 'acceptance.log').read_text(encoding='utf-8')
                    self.assertIn('COLLECTION FAILURE', log)
                    self.assertIn('ROW RESULT: E16 PASS', log)
                    self.assertNotIn('PREVIOUS PASS', log)

    def test_skip_error_and_failed_subtest_are_not_passes(self):
        statuses = {'runner_fixture.Case.test_e01': 'SKIP',
                    'runner_fixture.Case.test_e02': 'ERROR',
                    'runner_fixture.Case.test_e03': 'SUBTEST_FAIL'}
        report, observed, _ = self.execute(self.matrix(), statuses)
        self.assertEqual(len(observed), 16)
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual([row['status'] for row in report['rows'][:3]], ['FAIL'] * 3)
        self.assertEqual([row['results'][0]['status'] for row in report['rows'][:3]], ['SKIP', 'ERROR', 'FAIL'])

    def test_omitted_and_empty_scenarios_fail_instead_of_silently_disappearing(self):
        matrix = self.matrix()
        matrix['scenarios'].pop(3)
        matrix['scenarios'][8]['cases'] = []
        report, observed, _ = self.execute(matrix)
        self.assertEqual(len(report['rows']), 16)
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['rows'][3]['status'], 'FAIL')
        self.assertEqual(report['rows'][9]['status'], 'FAIL')
        self.assertEqual(len(observed), 14)

    def test_green_offline_evidence_preserves_formal_and_hardware_gaps(self):
        report, _, _ = self.execute(self.matrix())
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['unique_executed_count'], 16)
        self.assertEqual(report['formal_integration'], 'UNVERIFIED')
        self.assertEqual(report['hardware'], 'UNVERIFIED')
        self.assertTrue(all(row['remaining_gaps'] for row in report['rows']))
        self.assertTrue(all(row['verification_level'] == 'COMPONENT_ONLY' for row in report['rows']))

    def test_real_missing_method_is_collection_failure_and_cannot_hide_later_test(self):
        matrix = self.matrix()
        matrix['scenarios'][-1]['cases'][0]['id'] = (
            'test_p00_rules.RulesTests.test_T02_old_run_instance_epoch_and_expired_no_update')
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='missing-method-', dir=str(OUTPUT)) as folder:
            report = self.api().run_acceptance(matrix, Path(folder))
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['executed_count'], 1)
        self.assertEqual(report['rows'][0]['results'], [])
        self.assertTrue(report['rows'][0]['collection_errors'])
        self.assertEqual(report['rows'][-1]['status'], 'PASS')

    def test_cli_malformed_input_exports_all_failed_rows_and_returns_failure(self):
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='invalid-manifest-', dir=str(OUTPUT)) as folder:
            folder = Path(folder)
            manifest = folder / 'invalid.json'
            manifest.write_text('{ invalid json', encoding='utf-8')
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                code = self.api().main(['--matrix', str(manifest), '--output-dir', str(folder)])
            report = json.loads((folder / 'acceptance_results.json').read_text(encoding='utf-8'))
            self.assertEqual(code, 1)
            self.assertEqual(report['executed_count'], 0)
            self.assertEqual(len(report['rows']), 16)
            self.assertTrue(all(row['status'] == 'FAIL' for row in report['rows']))
            self.assertIn('Expecting property name', report.get('input_error', ''))


class ActualBoundaryTests(unittest.TestCase):
    """Actual package logic/SQLite; fixture observations and actuator transport.

    None of these tests certify physical motion, complete production adapters,
    process/GIL failure isolation, or real-time deadline latency.
    """

    def store(self, clock, remaining):
        from treasure_auto.persistence import PersistenceConfig, SQLitePersistence
        OUTPUT.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.TemporaryDirectory(prefix='sqlite-boundary-', dir=str(OUTPUT))
        self.addCleanup(tmp.cleanup)
        db = SQLitePersistence(PersistenceConfig(str(Path(tmp.name) / 'runtime'), 16, 3., 1., 4096),
                               clock, remaining=remaining)
        self.addCleanup(lambda: self.assertTrue(db.close(), 'temporary writer must close'))
        return db

    def test_deadline_stops_real_gateway_while_sqlite_writer_and_business_are_blocked(self):
        from test_p10a_support import make_runtime, signal, initial_snapshot, request
        from test_p01_gateway import BoundaryTransport
        from treasure_auto.devices.gateway import DeviceGateway
        from treasure_auto.devices.config import ControlConfig
        from treasure_auto.devices.events import EventOutbox
        from treasure_auto.runtime import IndependentSupervisor

        rt, clock = make_runtime()
        rt.arm()
        rt.signal(signal(rt, clock))
        db = self.store(clock, rt.remaining)
        self.assertIsNone(rt.initialize(db, initial_snapshot(rt.run)).reason)
        rt.heartbeat(replace(rt.run.envelope, producer=c.Owner.P09))
        req = request(rt, clock)
        held = c.HoldingSnapshot(c.Holding.HELD, 'brick-1', None,
                                (replace(req.readiness.holding.evidence[0], target_id='brick-1'),))
        envelope = replace(req.envelope, producer=c.Owner.P03, action_id='p11-durable-navigation')
        intent = m.ActionIntent(envelope, m.IntentOperation.BASE_VELOCITY, None, None, None, (.1, 0.), None)
        receipt = db.commit_intent(intent)
        self.assertEqual(receipt.commit.status, m.CommitStatus.COMMITTED)
        rt.remember_intent(intent, receipt, source=req.source)
        req = replace(req, envelope=envelope, intent_receipt=receipt,
                      readiness=replace(req.readiness, holding=held))
        permit = rt.request_permission(req).permit
        self.assertIsNotNone(permit)
        transport = BoundaryTransport()
        config = ControlConfig('sim-config-v1', .5, 1., 'sim-arm',
                               ((-200., 200.), (-200., 200.), (0., 200.)),
                               tuple((phase, (.5, 10., 5.)) for phase in m.Phase))
        gateway = DeviceGateway(clock, rt.run, config, transport, lambda: req.health,
                                EventOutbox(100), None)
        command = m.BaseCommand(envelope, permit, receipt.commit.transaction_id, .1, 0., req.source)
        gateway.register_intent(command, intent, receipt)
        gateway.install_permit(permit)
        event = gateway.submit_base(command)
        self.assertEqual(event.result.submitted, c.Fact.CONFIRMED)
        self.assertEqual(transport.base, [(.1, 0.)])
        retired = replace(command, envelope=replace(envelope, instance_id='retired-instance', control_epoch=0))
        self.assertEqual(gateway.submit_base(retired).result.accepted, c.Fact.REJECTED)
        self.assertEqual(transport.base, [(.1, 0.)])

        # This separate real SQLite connection owns the write lock. The patched
        # wrapper only observes arrival at BEGIN IMMEDIATE; it does not block or
        # replace SQLite/domain behavior.
        filename = db.root / 'runs' / rt.run.envelope.run_id / 'state.sqlite3'
        lock = sqlite3.connect(str(filename), isolation_level=None)
        entered, business_entered, stopped = threading.Event(), threading.Event(), threading.Event()
        reports, reads, failures = [], [], []
        original_atomic = db._atomic

        def observe_atomic(work):
            entered.set()
            return original_atomic(work)

        def business_read():
            business_entered.set()
            try:
                reads.append(db.read(rt.run.envelope.run_id))
            except Exception as error:
                failures.append(error)

        def stop_sink(stop):
            report = gateway.stop(stop)
            reports.append((stop, report))
            stopped.set()

        business = threading.Thread(target=business_read, name='p11-blocked-business', daemon=True)
        supervisor = IndependentSupervisor(rt, stop_sink)
        future = None
        lock.execute('BEGIN IMMEDIATE')
        try:
            with patch.object(db, '_atomic', side_effect=observe_atomic):
                future = db.submit('append_event', event)
                self.assertTrue(entered.wait(1), 'real writer did not reach locked SQLite transaction')
                self.assertFalse(future.done())
                business.start()
                self.assertTrue(business_entered.wait(1))
                supervisor.start()
                clock.advance(600)
                self.assertTrue(stopped.wait(1), 'independent stop waited for business/storage')
                self.assertFalse(future.done(), 'test must still hold the SQLite writer lock')
                self.assertTrue(business.is_alive(), 'business read must still be queued behind writer')
                stop, report = reports[-1]
                self.assertEqual(stop.reason.code, c.ErrorCode.DEADLINE)
                self.assertEqual(stop.holding.state, c.Holding.HELD)
                self.assertEqual(rt.run.deadline.seconds, 700.)
                self.assertEqual(rt.remaining_s, 0.)
                self.assertEqual(report.submitted, c.Fact.CONFIRMED)
                self.assertEqual(report.physically_stopped, c.Fact.UNKNOWN)
                self.assertEqual(transport.base[-1], (0., 0.))
                self.assertEqual(transport.pump, [])
                self.assertEqual(transport.arm, [])
                for cached in (command, replace(command, envelope=replace(envelope, instance_id='retired-instance', control_epoch=0))):
                    self.assertEqual(gateway.submit_base(cached).result.accepted, c.Fact.REJECTED)
                with self.assertRaises(c.ContractError):
                    gateway.install_permit(permit)
                self.assertFalse(rt.authorize_output(permit, envelope, req.source, real_output=False))
                self.assertEqual([value for value in transport.base if value != (0., 0.)], [(.1, 0.)])
        finally:
            lock.execute('ROLLBACK')
            lock.close()
            if business.ident is not None:
                business.join(3)
            supervisor.close()
        self.assertFalse(business.is_alive())
        self.assertFalse(failures)
        self.assertEqual(future.result(3).status, m.CommitStatus.COMMITTED)
        self.assertEqual(reads[0].snapshot.run.deadline.seconds, 700.)
        self.assertIsNone(supervisor.last_error)

    def release_case(self):
        from test_p07_support import Harness
        from test_p08_support import request, region, metric, config, proof
        from treasure_auto.placement.evidence import ReleaseWindow
        from treasure_auto.placement.space import SpaceProposals
        from treasure_auto.mission.transactions import ReleaseCoordinator

        h = Harness()
        h.reach('transport')
        h.confirm_stage()
        h.service.tick()
        order = h.request('place-root').order
        req = replace(request(), envelope=h.env(c.Owner.P09, action_id='place-root'),
                      order=order, region=region(m.Area.A), holding=h.service.holding)
        before = replace(s.state_snapshot(), holding=h.service.holding)
        db = self.store(h.clock, r.remaining_seconds)
        signal = m.RuntimeSignal(replace(before.run.envelope, event_id=before.run.start_signal_id),
                                 m.SignalKind.START, c.Fact.CONFIRMED)
        initialized = db.initialize_run(m.InitializeRunRequest(before.run, before, signal, None))
        self.assertEqual(initialized.status, m.CommitStatus.COMMITTED, initialized.reason)
        h.service.persistence = db
        window = ReleaseWindow(req, config(), h.clock.now())
        start = h.clock.now().seconds
        for dt, cup in ((0., (0., 0., .1)), (.1, (0., 0., .15)), (.35, (0., 0., .15))):
            now = start + dt
            h.clock.advance(now - h.clock.now().seconds)
            frame = metric(now, cup=cup)
            evidence = proof(now, action='place-root')
            frame = replace(frame, evidence=evidence,
                            objects=(replace(frame.objects[0], area=m.Area.A, evidence=(evidence,)),))
            release = window.push(frame, replace(h.context, now=h.clock.now()), before.run)
        self.assertIsNotNone(release, window.last_reason)
        update = h.service.propose_release(release, h.service.holding)
        space = SpaceProposals(config()).observe(before.space, frame, replace(h.context, now=h.clock.now()))
        layout = m.LayoutSnapshot(replace(release.envelope, target_id=None, action_id=None),
                                  m.Area.A, m.LayoutKind.V1, 0, 0, 0, (), c.Fact.UNKNOWN, c.Fact.UNKNOWN)
        placed = m.PlacementResult(release.envelope, s.timeout_result(), release, layout, space, update.current, None)
        return h, db, before, order, placed, update, ReleaseCoordinator(db, h.clock, run_update_provider=simulated_run_proposal)

    def test_real_release_duplicate_layout_failure_counts_once_and_keeps_empty_unready(self):
        h, db, before, order, placed, update, coordinator = self.release_case()
        self.assertEqual(h.service.holding.state, c.Holding.HELD)
        out = coordinator.commit_release(order, placed, update)
        self.assertEqual(out.receipt.status, m.CommitStatus.COMMITTED, out.receipt.reason)
        h.service.accept_holding_commit(out.receipt)
        snapshot = db.read('sim-run-1').snapshot
        self.assertEqual(len(snapshot.deliveries), 1)
        self.assertEqual(snapshot.deliveries[0].key, m.DeliveryKey('sim-run-1', 'inventory-1'))
        self.assertEqual(snapshot.layouts[0].confirmed_count, 0)
        self.assertEqual(snapshot.layouts[0].quality, c.Fact.UNKNOWN)
        self.assertEqual(snapshot.space.occupancies[0].state, m.OccupancyState.ACTUAL)
        self.assertEqual(snapshot.holding.state, c.Holding.EMPTY)
        self.assertEqual(h.service.holding.state, c.Holding.EMPTY)
        self.assertIsNone(h.service.readiness)
        self.assertEqual(coordinator.commit_release(order, placed, update), out)
        self.assertEqual(db.read('sim-run-1').snapshot, snapshot)
        self.assertEqual(sum(row['kind'] == 'transaction' for row in db.history('sim-run-1')), 1)
        changed = replace(placed, release=replace(placed.release, category=m.Category.DOLL))
        self.assertEqual(coordinator.commit_release(order, changed, update).receipt.status, m.CommitStatus.REJECTED)
        self.assertEqual(db.read('sim-run-1').snapshot, snapshot)

    def test_unknown_release_keeps_held_and_zero_arrivals_and_blocks_second_grasp(self):
        for mode in ('separation_unknown', 'echo_only', 'wrong_area', 'old_instance'):
            with self.subTest(mode=mode):
                h, db, before, order, placed, update, coordinator = self.release_case()
                release = placed.release
                if mode == 'separation_unknown':
                    release = replace(release, separated=c.Fact.UNKNOWN)
                elif mode == 'echo_only':
                    release = replace(release, evidence=tuple(replace(ev, kind=c.EvidenceKind.ECHO) for ev in release.evidence))
                elif mode == 'wrong_area':
                    release = replace(release, area=m.Area.B)
                else:
                    release = replace(release, envelope=replace(release.envelope, instance_id='retired'))
                with self.assertRaises(c.ContractError):
                    h.service.propose_release(release, h.service.holding)
                result = coordinator.commit_release(order, replace(placed, release=release), update)
                self.assertEqual(result.receipt.status, m.CommitStatus.REJECTED)
                self.assertEqual(db.read('sim-run-1').snapshot, before)
                self.assertEqual(h.service.holding.state, c.Holding.HELD)
                sent = len(h.commands)
                h.service.grasp(h.request('second-grasp'))
                self.assertEqual(len(h.commands), sent)
                self.assertIsNone(h.service.readiness)

    def test_unknown_sqlite_commit_reconciles_exact_release_without_replaying_action(self):
        h, db, before, order, placed, update, coordinator = self.release_case()
        lock = sqlite3.connect(str(db.root / 'runs' / 'sim-run-1' / 'state.sqlite3'), isolation_level=None)
        lock.execute('BEGIN IMMEDIATE')
        original_config = db.config
        db.config = replace(db.config, operation_timeout_s=.05)
        sent = len(h.commands)
        try:
            out = coordinator.commit_release(order, placed, update)
            self.assertEqual(out.receipt.status, m.CommitStatus.UNKNOWN)
            self.assertTrue(coordinator.blocked)
            self.assertTrue(db.actions_blocked)
            with self.assertRaises(c.ContractError):
                h.service.accept_holding_commit(out.receipt)
            self.assertEqual(h.service.holding.state, c.Holding.HELD)
            self.assertEqual(c.loads(lock.execute('SELECT payload FROM snapshot WHERE id=1').fetchone()[0]), before)
            self.assertEqual(len(h.commands), sent)
        finally:
            db.config = original_config
            lock.execute('ROLLBACK')
            lock.close()
        reconciled = coordinator.reconcile()
        self.assertEqual(reconciled.receipt.status, m.CommitStatus.COMMITTED, reconciled.receipt.reason)
        h.service.accept_holding_commit(reconciled.receipt)
        self.assertEqual(h.service.holding.state, c.Holding.EMPTY)
        self.assertIsNone(h.service.readiness)
        self.assertEqual(len(reconciled.snapshot.deliveries), 1)
        self.assertEqual(coordinator.commit_release(order, placed, update), reconciled)
        self.assertEqual(len(h.commands), sent)
        self.assertEqual(sum(row['kind'] == 'transaction' for row in db.history('sim-run-1')), 1)
        self.assertTrue(db.actions_blocked, 'ledger reconciliation is not renewed motion authority')


if __name__ == '__main__':
    unittest.main()
