"""Repeat P11 verification and record exact collection, failures and scope.

Run from auto_task: python -B treasure_auto/tests/integration/run_verification.py
For installed-artifact tests set P11_BUILD_PYTHON to Python with setuptools/wheel.
This runs only offline tests, local temporary stores and isolated wheel installs.
"""
import ast
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import unittest
from xml.etree import ElementTree

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
PACKAGE = TESTS.parent
BASE = PACKAGE.parent
ARTIFACTS = HERE / 'artifacts'
sys.path[:0] = [str(TESTS), str(PACKAGE / 'src')]


def flatten(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from flatten(test)
        else:
            yield test


def run_group(name, suite):
    ids = [case.id() for case in flatten(suite)]
    with (ARTIFACTS / (name+'.log')).open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    valid = (bool(ids) and len(ids) == len(set(ids)) == result.testsRun
             and result.wasSuccessful() and not result.skipped and not result.expectedFailures)
    report = dict(collected=len(ids), executed=result.testsRun, passed=valid,
                  failures=[t.id() for t, _ in result.failures], errors=[t.id() for t, _ in result.errors],
                  skipped=[t.id() for t, _ in result.skipped], ids=ids,
                  log=str((ARTIFACTS/(name+'.log')).relative_to(BASE)).replace('\\', '/'))
    print('{}: {}/{} executed, {}'.format(name, result.testsRun, len(ids), 'PASS' if valid else 'FAIL'), flush=True)
    return report


def owned(name):
    exact = {'README.md', 'docs/progress.md', 'docs/handoffs/P11_delivery.md',
             'treasure_auto/setup.py', 'treasure_auto/CMakeLists.txt', 'treasure_auto/package.xml',
             'treasure_auto/src/treasure_auto/__init__.py'}
    return (name in exact or name.startswith(('treasure_auto/src/treasure_auto/integration/',
            'treasure_auto/tests/integration/', 'treasure_auto/tests/test_p11_',
            'treasure_auto/scripts/run_match_', 'treasure_auto/launch/match_')))


def scope_report():
    baseline = json.loads((ARTIFACTS/'baseline_sha256.json').read_text(encoding='utf-8'))
    changed, missing = [], []
    for name, digest in baseline.items():
        path = BASE / name
        if not path.is_file():
            missing.append(name)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(name)
    added = [p.relative_to(BASE).as_posix() for p in BASE.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.relative_to(BASE).as_posix() not in baseline]
    violations = [name for name in changed+added+missing if not owned(name)]
    return dict(baseline_count=len(baseline), changed_originals=changed, missing_originals=missing,
                unchanged_originals=len(baseline)-len(changed)-len(missing),
                added_count=len(added), violations=violations, passed=not violations and not missing)


def static_report():
    files = list((PACKAGE/'src').rglob('*.py')) + list(TESTS.glob('test_p11_*.py'))
    files = [p for p in files if 'verification' not in p.parts and '__pycache__' not in p.parts]
    files += [PACKAGE/'setup.py', PACKAGE/'scripts/run_match_offline.py', HERE/'acceptance.py', Path(__file__)]
    errors = []
    for path in files:
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path), feature_version=(3, 8))
        except (SyntaxError, ValueError) as error:
            errors.append(str(error))
    for path in [PACKAGE/'package.xml']+list((PACKAGE/'launch').glob('*.launch')):
        try:
            ElementTree.parse(str(path))
        except ElementTree.ParseError as error:
            errors.append(str(error))
    return dict(python38_grammar_files=len(files), errors=errors, passed=not errors,
                python38_runtime='unverified', ros_catkin='unverified')


def run_stages():
    """R02: prerequisite regressions in the required integration order."""
    definitions = [
        ('staged_stop_chain', ('test_p10a_*.py', 'test_p10b_*.py', 'test_p01_*.py')),
        ('staged_mapping_navigation', ('test_p02_*.py', 'test_p03_*.py')),
        ('staged_vision_geometry', ('test_p04_*.py', 'test_p05_*.py', 'test_p06_*.py')),
        ('staged_grasp_placement', ('test_p07_*.py', 'test_p08_*.py')),
        ('staged_mission', ('test_p09_*.py',)),
    ]
    groups = {}
    for name, patterns in definitions:
        suite = unittest.TestSuite(unittest.TestLoader().discover(str(TESTS), pattern=p) for p in patterns)
        groups[name] = run_group(name, suite)
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages-only', action='store_true', help='only the original54, C0 and ordered prerequisite regressions')
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    report = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  python=sys.version, executable=sys.executable, platform=platform.platform(), groups={})
    loader = unittest.TestLoader()
    original_name = 'staged_original54' if args.stages_only else 'original54_final'
    report['groups'][original_name] = run_group(original_name, loader.loadTestsFromNames(['test_preflight','test_probe','test_cli']))
    c0_name = 'staged_c0' if args.stages_only else 'c0_final'
    report['groups'][c0_name] = run_group(c0_name, unittest.TestLoader().discover(str(TESTS), pattern='test_p00_*.py'))
    report['groups'].update(run_stages())
    if args.stages_only:
        report['passed'] = all(group['passed'] for group in report['groups'].values())
        (ARTIFACTS/'staged_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        return 0 if report['passed'] else 1
    for name, pattern in [('p11_final','test_p11_*.py'), ('all_final','test*.py')]:
        report['groups'][name] = run_group(name, unittest.TestLoader().discover(str(TESTS), pattern=pattern))
    report['per_module_collected'] = dict(sorted(Counter(x.split('.')[0] for x in report['groups']['all_final']['ids']).items()))
    from integration.acceptance import run_acceptance
    matrix = json.loads((HERE/'scenario_matrix.json').read_text(encoding='utf-8'))
    acceptance = run_acceptance(matrix, HERE/'acceptance/latest')
    report['acceptance'] = dict(status=acceptance['status'], rows=len(acceptance['rows']),
        executed_count=acceptance['executed_count'], unique_executed_count=acceptance['unique_executed_count'])
    report['static'] = static_report()
    report['scope'] = scope_report()
    report['passed'] = (all(g['passed'] for g in report['groups'].values()) and report['static']['passed']
                        and report['scope']['passed'] and acceptance['status']=='PASS')
    (ARTIFACTS/'verification_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('passed','acceptance','static','scope')}, ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
