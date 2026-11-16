"""Explicit P10B verification entry point; no duplicate unittest discovery cases."""
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path


def main():
    base = Path(__file__).resolve().parents[2]
    package = base / 'treasure_auto/src/treasure_auto/persistence'
    evidence = package / 'verification'
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    results = {}
    for name, pattern in [('p10b_final', 'test_p10b_*.py'), ('all_final', 'test*.py')]:
        command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'treasure_auto/tests', '-p', pattern, '-v']
        collected = unittest.TestLoader().discover(str(base / 'treasure_auto/tests'), pattern=pattern).countTestCases()
        run = subprocess.run(command, cwd=str(base), env=env, capture_output=True, timeout=180)
        text = (run.stdout + run.stderr).decode('utf-8', errors='replace')
        (evidence / (name + '.log')).write_text(text, encoding='utf-8')
        count = re.search(r'Ran (\d+) tests?', text)
        actual = int(count.group(1)) if count else None
        failures = re.findall(r'^(?:FAIL|ERROR): (.+)$', text, re.MULTILINE)
        results[name] = dict(command=command, collected=collected, executed=actual, exit_code=run.returncode,
                             failures=failures, collection_matches=collected == actual,
                             skipped=re.findall(r'^.+\.\.\. skipped .+$', text, re.MULTILINE))
        print(name, 'collected=', collected, 'executed=', actual, 'exit=', run.returncode, flush=True)
    before = json.loads((evidence / 'baseline_sha256.json').read_text(encoding='utf-8'))
    changed = [name for name, sha in before.items() if not (base / name).is_file() or
               hashlib.sha256((base / name).read_bytes()).hexdigest() != sha]
    sources = list(package.glob('*.py')) + list((base / 'treasure_auto/tests').glob('test_p10b_*.py'))
    for file in sources:
        ast.parse(file.read_text(encoding='utf-8'), filename=str(file), feature_version=(3, 8))
    code = """
import sys
class NoDevices:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('rospy','roslib','serial','cv2'):
            raise RuntimeError('device library imported: '+fullname)
sys.meta_path.insert(0,NoDevices())
sys.path.insert(0,sys.argv[1])
import treasure_auto.persistence
print('PASS: import without ROS, serial, or OpenCV')
"""
    imported = subprocess.run([sys.executable, '-B', '-c', code, str(base / 'treasure_auto/src')],
                              cwd=str(base), env=env, capture_output=True, timeout=15)
    (evidence / 'import_check.log').write_bytes(imported.stdout + imported.stderr)
    results['environment'] = dict(python=sys.version, platform=sys.platform)
    results['scope'] = dict(baseline_files=len(before), changed_existing=changed)
    results['python38_grammar'] = dict(files=len(sources), result='PASS', runtime38='not installed/verified')
    results['import_without_devices'] = imported.returncode
    results['owned_files'] = sorted(str(p.relative_to(base)).replace('\\', '/') for p in base.rglob('*')
                                   if p.is_file() and '__pycache__' not in p.parts and
                                   (package in p.parents or 'test_p10b_' in p.name or
                                    p.parent == base / 'treasure_auto/config/persistence' or
                                    p.name in ('P10B_delivery.md', 'P10B_interface_change.md')))
    (evidence / 'verification_summary.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print('scope changed existing:', changed, 'Python 3.8 grammar:', len(sources), 'files', flush=True)
    return 0 if (all(results[name]['exit_code'] == 0 and results[name]['collection_matches']
                     for name in ('p10b_final', 'all_final')) and not changed and imported.returncode == 0) else 1


if __name__ == '__main__':
    raise SystemExit(main())
