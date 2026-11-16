"""Reproduce P02, original+C0, and current all-package offline test collection."""
import hashlib
import json
import platform
import sys
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
TESTS=ROOT/'treasure_auto/tests'
sys.path.insert(0,str(TESTS))
sys.path.insert(0,str(ROOT/'treasure_auto/src'))


def run(name,suite):
    collected=suite.countTestCases()
    with (HERE/(name+'.log')).open('w',encoding='utf-8') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    data=dict(collected=collected,ran=result.testsRun,failures=[t.id() for t,_ in result.failures],
              errors=[t.id() for t,_ in result.errors],skipped=[t.id() for t,_ in result.skipped],
              passed=result.wasSuccessful() and result.testsRun==collected,log=name+'.log')
    print(name,json.dumps(data,ensure_ascii=False))
    return data


def main():
    suites={}
    suites['p02_final']=run('p02_final',unittest.TestLoader().discover(str(TESTS),pattern='test_p02_*.py'))
    baseline=json.loads((HERE/'baseline_sha256.json').read_text(encoding='utf-8'))
    modules=[Path(path).stem for path in baseline if path.startswith('treasure_auto/tests/test_') and path.endswith('.py')]
    suites['baseline_regression']=run('baseline_regression',unittest.TestLoader().loadTestsFromNames(sorted(modules)))
    suites['all_final']=run('all_final',unittest.TestLoader().discover(str(TESTS)))
    def fingerprint(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    changed=[p for p,h in baseline.items() if not (ROOT/p).is_file() or fingerprint(ROOT/p)!=h]
    now={str(p.relative_to(ROOT)).replace('\\','/'):fingerprint(p) for p in ROOT.rglob('*') if p.is_file()}
    def owned(p):
        return (p.startswith('treasure_auto/src/treasure_auto/mapping/') or p.startswith('treasure_auto/config/mapping/') or
                p.startswith('treasure_auto/tests/test_p02_') or p in ('docs/handoffs/P02_delivery.md','docs/handoffs/P02_interface_change.md'))
    summary=dict(recorded_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),python=platform.python_version(),
        platform=platform.platform(),suites=suites,baseline_count=len(baseline),baseline_changed=changed,
        p02_files=sorted(p for p in now if owned(p)),
        concurrent_other_new_files=sorted(p for p in now if p not in baseline and not owned(p)),
        p02_source_sha256={p:h for p,h in now.items() if owned(p) and '/verification/' not in p},
        dependencies={'C0':'1.0.0','P01_delivery_present':(ROOT/'docs/handoffs/P01_delivery.md').exists()})
    (HERE/'verification_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('baseline_changed:',changed)
    print('concurrent_other_new_files:',len(summary['concurrent_other_new_files']))
    return 0 if all(s['passed'] for s in suites.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
