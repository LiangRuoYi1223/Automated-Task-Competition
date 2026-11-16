"""Reproduce P01/C0/original/full offline tests and read-only scope checks."""
import ast
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
TESTS = ROOT / 'treasure_auto/tests'


def run(name, patterns):
    if len(patterns) == 1:
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'treasure_auto/tests', '-p', patterns[0], '-v']
        result = subprocess.run(command, cwd=str(ROOT), capture_output=True, text=True, encoding='utf-8')
        log, exit_code = result.stdout + result.stderr, result.returncode
    else:
        command = ['TestLoader.discover'] + list(patterns)
        stream = io.StringIO()
        suite = unittest.TestSuite()
        for pattern in patterns:
            suite.addTests(unittest.TestLoader().discover(str(TESTS), pattern=pattern))
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        log, exit_code = stream.getvalue(), int(not result.wasSuccessful())
    counts = re.findall(r'^Ran (\d+) tests? in ', log, re.M)
    failures = re.findall(r'^(?:FAIL|ERROR): (.+)$', log, re.M)
    (HERE / (name + '.log')).write_text(log, encoding='utf-8')
    expected = sum(unittest.TestLoader().discover(str(TESTS), pattern=p).countTestCases() for p in patterns)
    actual = int(counts[-1]) if counts else 0
    return dict(command=command, exit_code=exit_code, tests_run=actual,
                collected_after_run=expected, collection_matches=(expected == actual),
                failures=failures, log=name + '.log')


def main():
    results = {name: run(name, patterns) for name, patterns in (
        ('original_final', ('test_cli.py', 'test_probe.py', 'test_preflight.py')),
        ('c0_final', ('test_p00_*.py',)), ('p01_final', ('test_p01_*.py',)), ('all_final', ('test*.py',)))}
    baseline = json.loads((HERE / 'baseline_sha256.json').read_text(encoding='utf-8'))
    changed = [name for name, digest in baseline.items()
               if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    sources = sorted((HERE.parent).rglob('*.py')) + sorted(TESTS.glob('test_p01_*.py'))
    syntax_errors = []
    for path in sources:
        try:
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 8))
        except SyntaxError as error:
            syntax_errors.append(str(error))
    p01_files = sorted((HERE.parent).rglob('*')) + sorted(TESTS.glob('test_p01_*.py'))
    p01_files += list((ROOT / 'treasure_auto/config/devices').rglob('*'))
    p01_files += list((ROOT / 'docs/handoffs').glob('P01_*.md'))
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in p01_files if p.is_file() and '__pycache__' not in p.parts and p.name != 'verification_summary.json'}
    summary = dict(timestamp=datetime.now(timezone(timedelta(hours=8))).isoformat(), python=sys.version,
                   results=results, baseline_files=len(baseline), changed_baseline_files=changed,
                   python38_syntax_files=len(sources), python38_syntax_errors=syntax_errors,
                   p01_file_sha256=hashes,
                   dependency_handoffs=sorted(p.name for p in (ROOT / 'docs/handoffs').glob('*delivery.md')),
                   real_actions_executed=False)
    (HERE / 'verification_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: {n: v[n] for n in ('tests_run', 'exit_code', 'collection_matches')} for k, v in results.items()}, indent=2))
    print('Baseline changed:', changed, '; Python 3.8 syntax errors:', syntax_errors)
    return int(any(v['exit_code'] or not v['collection_matches'] for v in results.values()) or bool(changed) or bool(syntax_errors))


if __name__ == '__main__':
    raise SystemExit(main())
