"""Execute the E01--E16 offline evidence matrix; never implies live acceptance.

From auto_task: python -B treasure_auto/tests/integration/acceptance.py
Each entry must name one exact unittest method, not a discovery pattern or class.
"""
import argparse
import contextlib
import io
import json
import platform
import re
import sys
import time
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
SCENARIOS = tuple('E{:02}'.format(i) for i in range(1, 17))
LEVELS = {'component', 'boundary_integration', 'simulated_orchestration'}
METHOD_ID = re.compile(r'^[A-Za-z_]\w*\.[A-Za-z_]\w*\.test_\w+$')


def _flatten(suite):
    if isinstance(suite, unittest.TestSuite):
        for child in suite:
            yield from _flatten(child)
    else:
        yield suite


class EvidenceResult(unittest.TextTestResult):
    """Keep individual results, including skipped tests and failed subtests."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self._by_test = {}

    def startTest(self, test):
        super().startTest(test)
        record = {'id': test.id(), 'status': 'INCOMPLETE', 'detail': ''}
        self.records.append(record)
        self._by_test[test] = record

    def _record(self, test, status, detail=''):
        if test not in self._by_test:  # e.g. setUpClass raises/skips before startTest
            record = {'id': test.id(), 'status': status, 'detail': detail}
            self.records.append(record)
            self._by_test[test] = record
        else:
            record = self._by_test[test]
            if record['status'] not in ('FAIL', 'ERROR'):
                record['status'] = status
            if detail:
                record['detail'] += detail

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, 'PASS')

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._record(test, 'FAIL', self._exc_info_to_string(err, test))

    def addError(self, test, err):
        super().addError(test, err)
        self._record(test, 'ERROR', self._exc_info_to_string(err, test))

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, 'SKIP', reason)

    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test, err)
        self._record(test, 'EXPECTED_FAILURE', self._exc_info_to_string(err, test))

    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test)
        self._record(test, 'UNEXPECTED_SUCCESS')

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            status = 'FAIL' if issubclass(err[0], test.failureException) else 'ERROR'
            self._record(test, status, '{}\n{}'.format(subtest.id(), self._exc_info_to_string(err, test)))


def run_acceptance(matrix, output_dir, *, loader=None):
    """Run every required row and write reviewable evidence, even after failures.

    `loader` is a test-only collection boundary. Tests themselves execute through
    unittest and must have the exact requested method IDs. A green return means
    only that the selected offline evidence passed, not that every E requirement
    or production/hardware integration is complete.
    """
    started = time.monotonic()
    loader = loader or unittest.TestLoader()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log = io.StringIO()
    errors = []
    if matrix.get('input_error'):
        errors.append('Matrix input unavailable: ' + matrix['input_error'])
    if matrix.get('schema_version') != 1 or matrix.get('contract_version') != '1.0.0':
        errors.append('Unsupported matrix schema/contract version; expected schema=1 and C0=1.0.0.')
    definitions = matrix.get('scenarios', [])
    if not isinstance(definitions, list):
        errors.append('scenarios must be a list')
        definitions = []
    for definition in definitions:
        if not isinstance(definition, dict) or definition.get('id') not in SCENARIOS:
            errors.append('Unrecognized scenario: {!r}'.format(definition))
    rows = []
    for scenario_id in SCENARIOS:
        found = [d for d in definitions if isinstance(d, dict) and d.get('id') == scenario_id]
        definition = found[0] if found else {'id': scenario_id, 'cases': []}
        collection_errors = []
        if len(found) != 1:
            collection_errors.append('Expected exactly one definition for {}, found {}.'.format(scenario_id, len(found)))
        gaps = definition.get('remaining_gaps')
        if not isinstance(gaps, list) or not all(isinstance(gap, str) for gap in gaps):
            collection_errors.append('remaining_gaps must be a list of strings; received {!r}.'.format(gaps))
            gaps = ['Invalid remaining_gaps declaration; formal/hardware coverage remains UNVERIFIED.']
        cases = definition.get('cases', [])
        if not isinstance(cases, list):
            collection_errors.append('cases must be a list')
            cases = []
        if not cases:
            collection_errors.append('Zero case collection is a failure.')
        collected, provenance = [], {}
        for case in cases:
            if not isinstance(case, dict):
                collection_errors.append('Malformed case: {!r}'.format(case))
                continue
            name = case.get('id', '')
            if (not isinstance(name, str) or not METHOD_ID.fullmatch(name)
                    or not isinstance(case.get('level'), str)
                    or case['level'] not in LEVELS or not case.get('proves')):
                collection_errors.append('Invalid exact method/evidence declaration: {!r}'.format(case))
                continue
            if name in provenance:
                collection_errors.append('Duplicate method within row: ' + name)
                continue
            provenance[name] = case
            try:
                tests = list(_flatten(loader.loadTestsFromName(name)))
                ids = [test.id() for test in tests]
                if ids != [name]:
                    collection_errors.append('{} collected {!r}; expected one exact method.'.format(name, ids))
                    continue
                collected.extend(tests)
            except Exception:
                collection_errors.append('{} collection failed:\n{}'.format(name, traceback.format_exc()))
        log.write('\n{} {}\n'.format(scenario_id, definition.get('title', 'MISSING')))
        for error in collection_errors:
            log.write('COLLECTION FAILURE: {}\n'.format(error))
        runner = unittest.TextTestRunner(stream=log, verbosity=2, resultclass=EvidenceResult)
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = runner.run(unittest.TestSuite(collected))
        for record in result.records:
            case = provenance.get(record['id'], {})
            record['level'] = case.get('level', 'uncollected_fixture')
            record['real_packages'] = case.get('real_packages', [])
            record['proves'] = case.get('proves', '')
        actual_ids = [record['id'] for record in result.records]
        # Invalid declarations remain collection failures above. Only validated
        # string IDs may enter comparison/reporting: None/list/dict IDs must not
        # abort this row (and prevent all remaining rows/evidence from running).
        declared_ids = [case['id'] for case in cases if isinstance(case, dict)
                        and isinstance(case.get('id'), str) and METHOD_ID.fullmatch(case['id'])]
        if sorted(actual_ids) != sorted(declared_ids) or result.testsRun != len(cases):
            collection_errors.append('Declared, started and completed method counts/IDs differ.')
        ok = (not collection_errors and result.testsRun > 0 and result.wasSuccessful()
              and all(record['status'] == 'PASS' for record in result.records))
        levels = {record['level'] for record in result.records}
        verification = ('NARROW_BOUNDARY_INTEGRATION' if 'boundary_integration' in levels
                        else 'SIMULATED_ORCHESTRATION' if 'simulated_orchestration' in levels
                        else 'COMPONENT_ONLY')
        row = {key: value for key, value in definition.items() if key != 'cases'}
        row.update(id=scenario_id, status='PASS' if ok else 'FAIL',
                   verification_level=verification, requested_ids=declared_ids,
                   collected_count=len(collected), executed_count=result.testsRun,
                   results=result.records, collection_errors=collection_errors,
                   remaining_gaps=gaps)
        rows.append(row)
        log.write('ROW RESULT: {} {}; evidence={}; formal/hardware remain UNVERIFIED.\n'.format(
            scenario_id, row['status'], verification))
        for gap in row['remaining_gaps']:
            log.write('UNRESOLVED: {}\n'.format(gap))
    executed = [record['id'] for row in rows for record in row['results']
                if record['id'] in row['requested_ids']]
    report = {'schema_version': 1, 'contract_version': matrix.get('contract_version'),
              'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.version, 'platform': platform.platform(),
              'status': 'PASS' if not errors and all(row['status'] == 'PASS' for row in rows) else 'FAIL',
              'meaning': 'Selected offline evidence only; unresolved E-row gates remain explicit.',
              'formal_integration': 'UNVERIFIED', 'hardware': 'UNVERIFIED',
              'executed_count': sum(row['executed_count'] for row in rows),
              'unique_executed_count': len(set(executed)),
              'elapsed_s': round(time.monotonic() - started, 6),
              'manifest_errors': errors, 'rows': rows}
    if matrix.get('input_error'):
        report['input_error'] = matrix['input_error']
    log.write('\nMATRIX RESULT: {}; {} method executions / {} unique IDs; {} rows.\n'.format(
        report['status'], report['executed_count'], report['unique_executed_count'], len(rows)))
    for error in errors:
        log.write('MANIFEST FAILURE: {}\n'.format(error))
    (output_dir / 'acceptance_results.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'acceptance.log').write_text(log.getvalue(), encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, default=HERE / 'scenario_matrix.json')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'acceptance' / 'latest')
    args = parser.parse_args(argv)
    sys.path[:0] = [str(TESTS), str(TESTS.parent / 'src')]
    try:
        matrix = json.loads(args.matrix.read_text(encoding='utf-8'))
        if not isinstance(matrix, dict):
            raise ValueError('matrix root must be an object')
    except (OSError, ValueError) as error:
        # Still export all 16 failed rows rather than treating no input as PASS.
        matrix = {'scenarios': [], 'input_error': str(error)}
    report = run_acceptance(matrix, args.output_dir)
    if 'input_error' in matrix:
        print('Matrix input unavailable: ' + matrix['input_error'])
    print('{}: E01--E16, {} executions / {} unique methods; formal/hardware UNVERIFIED'.format(
        report['status'], report['executed_count'], report['unique_executed_count']))
    print(args.output_dir.resolve() / 'acceptance_results.json')
    print(args.output_dir.resolve() / 'acceptance.log')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
