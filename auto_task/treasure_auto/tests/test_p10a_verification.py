"""Executable audit, not a discovered test class. Writes only P10A verification artifacts."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest


def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'treasure_auto/src/treasure_auto/runtime/verification'
    baseline=json.loads((out/'baseline_sha256.json').read_text(encoding='utf-8'))
    testdir=root/'treasure_auto/tests'
    results={}
    for name,pattern in (('p10a','test_p10a_*.py'),('p00','test_p00_*.py'),('all','test*.py')):
        suite=unittest.TestLoader().discover(str(testdir),pattern=pattern)
        collected=suite.countTestCases()
        command=[sys.executable,'-B','-m','unittest','discover','-s','treasure_auto/tests','-p',pattern,'-v']
        process=subprocess.run(command,cwd=str(root),capture_output=True,text=True,encoding='utf-8')
        (out/(name+'_final.log')).write_text(process.stdout+process.stderr,encoding='utf-8')
        results[name]={'collected':collected,'returncode':process.returncode,'command':command,
                       'count_matches':('Ran {} tests'.format(collected) in process.stderr),
                       'skipped':('skipped=' in process.stderr)}
    changes=[]
    for rel,digest in baseline.items():
        path=root/rel
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            changes.append(rel)
    all_files=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    added=sorted(str(p.relative_to(root)).replace('\\','/') for p in all_files if str(p.relative_to(root)).replace('\\','/') not in baseline)
    def owned(path):
        return (path.startswith('treasure_auto/src/treasure_auto/runtime/') or path.startswith('treasure_auto/config/runtime/')
                or path.startswith('treasure_auto/tests/test_p10a_')
                or path in ('docs/handoffs/P10A_delivery.md','docs/handoffs/P10A_interface_change.md'))
    unowned=[p for p in added if not owned(p)]
    concurrent_plans={
        'treasure_auto/src/treasure_auto/devices/IMPLEMENTATION_PLAN.md',
        'treasure_auto/src/treasure_auto/perception/IMPLEMENTATION_PLAN.md',
        'treasure_auto/src/treasure_auto/persistence/IMPLEMENTATION_PLAN.md',
    }
    # These plans existed during our initial scan and were completed by their
    # owners in this shared non-Git workspace. Keep the original hashes and the
    # observed differences; never silently replace the baseline.
    unexpected=[path for path in changes if path not in concurrent_plans]
    pyfiles=list((root/'treasure_auto/src/treasure_auto/runtime').rglob('*.py'))+list(testdir.glob('test_p10a_*.py'))
    for path in pyfiles:ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,8))
    summary={'python':sys.version,'contract_version':'1.0.0','results':results,'baseline_files':len(baseline),
             'baseline_changed':changes,'other_package_additions_observed':unowned,
             'unexpected_baseline_changes':unexpected,
             'authored_scope_files':[p for p in added if owned(p)],
             'scope_note':'Shared workspace received concurrent other-package additions; these are observed, not attributed to P10A. Existing baseline hashes are checked separately.',
             'python38_syntax_files':len(pyfiles),'real_devices_executed':False}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('authored_scope_files','other_package_additions_observed')},ensure_ascii=False,indent=2))
    return 1 if unexpected or any(v['returncode'] or not v['count_matches'] or v['skipped'] for v in results.values()) else 0


if __name__=='__main__':sys.exit(main())
