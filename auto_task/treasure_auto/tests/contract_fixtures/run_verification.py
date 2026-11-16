"""Reproduce P00 tests, final logs, collection counts and this delivery's scope audit.

The baseline applies to this P00 handoff only. Later owners may legitimately
change their files; do not use its hash comparison as a permanent project gate.
"""
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import unittest

FIXTURES = Path(__file__).resolve().parent
PACKAGE = FIXTURES.parents[1]
AUTO_TASK = PACKAGE.parent
WORKSPACE = AUTO_TASK.parent


def run(pattern, logfile):
    command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'treasure_auto/tests']
    if pattern is not None:
        command += ['-p', pattern]
    command += ['-v']
    completed = subprocess.run(command, cwd=str(AUTO_TASK), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, encoding='utf-8', errors='replace', timeout=120)
    (FIXTURES / logfile).write_text(completed.stdout, encoding='utf-8')
    match = re.search(r'Ran (\d+) tests in ([0-9.]+)s', completed.stdout)
    if completed.returncode or not match or not completed.stdout.rstrip().endswith('OK'):
        raise RuntimeError('verification failed; inspect ' + logfile)
    collected = unittest.TestLoader().discover(str(PACKAGE / 'tests'), pattern=pattern or 'test*.py').countTestCases()
    if collected != int(match.group(1)):
        raise RuntimeError('test collection count mismatch')
    return dict(command=command, tests=collected, seconds=float(match.group(2)), result='PASS', log=logfile)


def main():
    p00 = run('test_p00_*.py', 'p00_final.log')
    all_tests = run(None, 'all_final.log')
    baseline = json.loads((FIXTURES / 'baseline_sha256.json').read_text(encoding='utf-8'))
    changed = []
    for name, digest in baseline.items():
        path = WORKSPACE / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            changed.append(name)
    new_files, outside = [], []
    allowed_dirs = ('auto_task/treasure_auto/src/treasure_auto/contracts/',
                    'auto_task/treasure_auto/tests/contract_fixtures/')
    for path in sorted(AUTO_TASK.rglob('*')):
        if not path.is_file():
            continue
        name = path.relative_to(WORKSPACE).as_posix()
        if name in baseline:
            continue
        new_files.append(name)
        allowed = (name.startswith(allowed_dirs) or
                   (path.parent == PACKAGE / 'tests' and path.name.startswith('test_p00_') and path.suffix == '.py') or
                   name == 'auto_task/docs/handoffs/P00_delivery.md')
        if not allowed:
            outside.append(name)
    summary = dict(task='P00', contract_version='1.0.0', delivery_date='2026-11-14',
                   python=sys.version, platform=platform.platform(), baseline_tests=54,
                   p00=p00, all=all_tests, protected_files_checked=len(baseline),
                   protected_changes=changed, out_of_scope_new_files=outside,
                   validation=dict(offline='PASS', cross_package='NOT_RUN', ros='NOT_RUN',
                                   physical_actions='NOT_RUN', python38_grammar='PASS', python38_runtime='NOT_RUN'),
                   new_files=new_files)
    (FIXTURES / 'verification_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if changed or outside:
        raise RuntimeError('scope audit failed; see verification_summary.json')
    if all_tests['tests'] != p00['tests'] + 54:
        raise RuntimeError('baseline count changed; review the collected tests and delivery context')
    print('P00 {}/{} PASS; full {}/{} PASS; collection complete; {} protected files unchanged; scope PASS'.format(
        p00['tests'], p00['tests'], all_tests['tests'], all_tests['tests'], len(baseline)))


if __name__ == '__main__':
    main()
