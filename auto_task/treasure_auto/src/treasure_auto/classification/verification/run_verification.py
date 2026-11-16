"""Reproducible P05, C0, and full regression logs, discovery and scope checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    results={}
    environment=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    baseline=json.loads((HERE/'baseline_sha256.json').read_text(encoding='utf-8'))
    for name,pattern in (('p05','test_p05_*.py'),('p00','test_p00_*.py'),('all','test*.py')):
        command=[sys.executable,'-B','-m','unittest','discover','-s','treasure_auto/tests','-p',pattern,'-v']
        result=subprocess.run(command,cwd=ROOT,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        (HERE/(name+'_final.log')).write_text(result.stdout,encoding='utf-8')
        # Separate loader process matches the discovery entry point without
        # importing a second copy of test classes into this verification runner.
        script="import unittest; s=unittest.TestLoader().discover('treasure_auto/tests',pattern="+repr(pattern)+"); print(s.countTestCases())"
        count=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,env=environment,capture_output=True,text=True)
        import re
        reported=re.search(r'Ran (\d+) tests? in',result.stdout)
        discovered=int(count.stdout.strip()) if count.returncode==0 else None
        executed=int(reported.group(1)) if reported else None
        results[name]={'command':command,'exit_code':result.returncode,'discovered':discovered,
                       'executed':executed,'collected_equals_run':discovered==executed and executed is not None,
                       'log':name+'_final.log'}
        print(name,results[name],flush=True)
    baseline_modules=[Path(p).stem for p in sorted(baseline) if p.startswith('treasure_auto/tests/test') and p.endswith('.py')]
    script="""import sys,unittest
sys.path.insert(0,'treasure_auto/tests')
suite=unittest.TestLoader().loadTestsFromNames(%r)
print('Collected baseline:',suite.countTestCases(),flush=True)
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
""" % baseline_modules
    original=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,env=environment,
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    (HERE/'baseline_final.log').write_text(original.stdout,encoding='utf-8')
    collected=re.search(r'Collected baseline: (\d+)',original.stdout)
    executed=re.search(r'Ran (\d+) tests? in',original.stdout)
    results['baseline']={'exit_code':original.returncode,'discovered':int(collected.group(1)) if collected else None,
                         'executed':int(executed.group(1)) if executed else None,'log':'baseline_final.log',
                         'command':'run_verification.py: load exactly baseline_sha256.json test modules'}
    results['baseline']['collected_equals_run']=bool(collected and executed and collected.group(1)==executed.group(1))
    print('baseline',results['baseline'],flush=True)
    changed=[path for path,sha in baseline.items() if not (ROOT/path).is_file() or digest(ROOT/path)!=sha]
    new=[]
    allowed=[]
    for path in ROOT.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts: continue
        relative=path.relative_to(ROOT).as_posix()
        if relative in baseline: continue
        owned=(relative.startswith('treasure_auto/src/treasure_auto/classification/') or
               relative.startswith('treasure_auto/config/classification/') or
               (relative.startswith('treasure_auto/tests/test_p05_') and relative.endswith('.py')) or
               relative in ('docs/handoffs/P05_delivery.md','docs/handoffs/P05_interface_change.md'))
        (allowed if owned else new).append(relative)
    summary={'python':sys.version,'contract':'1.0.0','runs':results,'protected_files':len(baseline),
             'changed_existing_files':changed,'outside_additions_observed':new,'P05_files':sorted(allowed),
             'scope_note':'Outside additions are reported for attribution; concurrent package work is not removed or claimed by P05.'}
    (HERE/'verification_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Scope:',len(baseline),'protected;',len(changed),'changed;',len(new),'outside additions',flush=True)
    return 0 if not changed and all(r['exit_code']==0 and r['collected_equals_run'] for r in results.values()) else 1


if __name__=='__main__':
    raise SystemExit(run())
