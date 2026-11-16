"""Run from any cwd; retain collected IDs, actual counts, logs and scope hashes."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
TESTS=ROOT/'treasure_auto/tests'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'treasure_auto/src'))
sys.path.insert(0,str(TESTS))

def flatten(suite):
    for test in suite:
        if isinstance(test,unittest.TestSuite):yield from flatten(test)
        else:yield test.id()

def allowed(path):
    return (path.startswith('treasure_auto/src/treasure_auto/navigation/') or
            path.startswith('treasure_auto/config/navigation/') or
            (path.startswith('treasure_auto/tests/test_p03_') and path.endswith('.py')) or
            path in ('docs/handoffs/P03_delivery.md','docs/handoffs/P03_interface_change.md'))

def main():
    os.chdir(ROOT)
    report={'python':sys.version,'contract_version':'1.0.0','suites':{}}
    ok=True
    for label,pattern in (('p03','test_p03_*.py'),('p00','test_p00_*.py'),('all','test*.py')):
        loader=unittest.TestLoader();suite=loader.discover(str(TESTS),pattern=pattern)
        ids=list(flatten(suite))
        with (HERE/(label+'_final.log')).open('w',encoding='utf-8') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        data=dict(collected=len(ids),run=result.testsRun,failures=[t.id() for t,_ in result.failures],
                  errors=[t.id() for t,_ in result.errors],skipped=[t.id() for t,_ in result.skipped],test_ids=ids,
                  passed=result.wasSuccessful() and not result.skipped and len(ids)==result.testsRun and not loader.errors)
        report['suites'][label]=data;ok=ok and data['passed']
        print(label,'collected',len(ids),'run',result.testsRun,'PASS' if data['passed'] else 'FAIL',flush=True)
    base=json.loads((HERE/'baseline_sha256.json').read_text(encoding='utf-8'))
    baseline_ids=[]
    for name in sorted(base):
        path=Path(name)
        if path.parent.as_posix()=='treasure_auto/tests' and path.name.startswith('test') and path.suffix=='.py':
            baseline_ids.extend(flatten(unittest.TestLoader().discover(str(TESTS),pattern=path.name)))
    actual=report['suites']['all']
    bad=set(actual['failures']+actual['errors']+actual['skipped'])
    missing=sorted(set(baseline_ids)-set(actual['test_ids']))
    baseline_bad=sorted(t for t in baseline_ids if any(x==t or x.startswith(t+' ') for x in bad))
    report['baseline_regression']=dict(basis='subset of actual all-suite run',matched=len(baseline_ids),
                                       missing=missing,failed=baseline_bad,passed=not missing and not baseline_bad)
    report['baseline_files']=len(base)
    report['baseline_changed_or_missing']=[p for p,digest in base.items() if not (ROOT/p).is_file() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=digest]
    all_files={p.relative_to(ROOT).as_posix():p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    report['new_outside_scope']=sorted(p for p in all_files if p not in base and not allowed(p))
    report['files_sha256']={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in sorted(all_files.items())
                          if allowed(name) and p.name!='verification_summary.json'}
    report['python38_syntax']=[]
    for name,p in all_files.items():
        if allowed(name) and p.suffix=='.py':
            ast.parse(p.read_text(encoding='utf-8'),filename=name,feature_version=(3,8))
            report['python38_syntax'].append(name)
    report['source_docx_sha256']=hashlib.sha256((ROOT.parent/'自动任务赛代码设计总纲_v2.docx').read_bytes()).hexdigest()
    report['c0_schema_sha256']=hashlib.sha256((ROOT/'treasure_auto/src/treasure_auto/contracts/schema_v1.json').read_bytes()).hexdigest()
    # Shared workspace: additions owned by other packages are observations, not
    # evidence that this package wrote them. Keep the full list in the report.
    ok=ok and not report['baseline_changed_or_missing']
    report['passed']=bool(ok)
    (HERE/'verification_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('unchanged baseline',len(base)-len(report['baseline_changed_or_missing']), '/',len(base),flush=True)
    print('original regression subset',len(baseline_ids),'PASS' if report['baseline_regression']['passed'] else 'FAIL',flush=True)
    print('observed other-package new files',len(report['new_outside_scope']),'Python 3.8 syntax files',len(report['python38_syntax']),flush=True)
    return 0 if ok else 1

if __name__=='__main__':raise SystemExit(main())
