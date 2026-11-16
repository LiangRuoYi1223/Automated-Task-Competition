"""Reproduce P04/public regression logs and snapshot concurrent workspace changes.

Run from auto_task: python -B treasure_auto/src/treasure_auto/perception/verification/run_verification.py
Writes only this P04 verification directory. It never runs ROS or devices.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def p04_owned(name):
    return (name.startswith('treasure_auto/src/treasure_auto/perception/') or
            name.startswith('treasure_auto/config/perception/') or
            (name.startswith('treasure_auto/tests/test_p04_') and name.endswith('.py')) or
            name in {'docs/handoffs/P04_delivery.md', 'docs/handoffs/P04_interface_change.md'})


def run(name, pattern):
    args = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'treasure_auto/tests']
    if pattern is not None:
        args.extend(['-p', pattern])
    args.append('-v')
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run(args, cwd=str(ROOT), env=env, capture_output=True, text=True, encoding='utf-8', timeout=180)
    output = result.stdout + result.stderr
    (HERE / (name + '.log')).write_text(output, encoding='utf-8')
    matches = re.findall(r'^Ran (\d+) tests? in ', output, re.MULTILINE)
    ids = re.findall(r'^(test_\S+) \(([^)]+)\) \.\.\.', output, re.MULTILINE)
    unique = {owner + '.' + method for method, owner in ids}
    failures = re.findall(r'^(?:FAIL|ERROR): (.+)$', output, re.MULTILINE)
    summary = dict(command=subprocess.list2cmdline(args), returncode=result.returncode,
                   tests_run=int(matches[-1]) if matches else None,
                   reported_test_lines=len(ids), unique_test_ids=len(unique),
                   failures=failures, log=str((HERE / (name + '.log')).relative_to(ROOT)))
    print(name, json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    summaries = [run('p04_final', 'test_p04_*.py'), run('p00_final', 'test_p00_*.py'), run('all_final', None)]
    baseline = json.loads((HERE / 'baseline_sha256.json').read_text(encoding='utf-8'))
    files = {p.relative_to(ROOT).as_posix(): digest(p) for p in ROOT.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts}
    changed = [name for name, value in baseline.items() if files.get(name) != value]
    new = sorted(set(files) - set(baseline))
    summary = dict(contract_version='1.0.0', python=sys.version,
                   results=summaries, baseline_file_count=len(baseline), changed_baseline_files=changed,
                   p04_files=sorted(name for name in files if p04_owned(name)),
                   concurrent_external_new_files=[name for name in new if not p04_owned(name)],
                   tier=dict(offline='see test results', cross_package='not validated', ros='not run', physical_actions='not run'))
    (HERE / 'verification_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (HERE / 'p04_sha256.json').write_text(json.dumps({k: v for k, v in files.items() if p04_owned(k)
                       and not k.endswith(('p04_sha256.json', 'verification_summary.json'))}, indent=2), encoding='utf-8')
    return 0 if all(x['returncode'] == 0 for x in summaries) else 1


if __name__ == '__main__':
    sys.exit(main())
