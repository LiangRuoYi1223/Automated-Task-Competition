"""Run task-book commands, count unique collected IDs, and check owned-file scope."""
import hashlib
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
TESTS=ROOT/'treasure_auto/tests'


def ids(suite):
    for test in suite:
        if isinstance(test,unittest.TestSuite):yield from ids(test)
        else:yield test.id()


def owned(path):
    return (path.startswith('treasure_auto/src/treasure_auto/geometry/') or
            path.startswith('treasure_auto/config/geometry/') or
            (path.startswith('treasure_auto/tests/test_p06_') and path.endswith('.py')) or
            path in ('docs/handoffs/P06_delivery.md','docs/handoffs/P06_interface_change.md'))


def main():
    sys.path.insert(0,str(TESTS));sys.path.insert(0,str(ROOT/'treasure_auto/src'))
    summary={'python':sys.version,'contract':'1.0.0','suites':[]}
    for name,pattern in (('p06_final','test_p06_*.py'),('c0_final','test_p00_*.py'),('all_final','test*.py')):
        collected=list(ids(unittest.TestLoader().discover(str(TESTS),pattern=pattern)))
        command=[sys.executable,'-B','-m','unittest','discover','-s','treasure_auto/tests','-v']
        if name!='all_final':command.extend(['-p',pattern])
        log=HERE/(name+'.log')
        with log.open('w',encoding='utf-8') as f:
            result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        text=log.read_text(encoding='utf-8');match=re.search(r'Ran (\d+) tests?',text)
        count=int(match.group(1)) if match else None
        row=dict(name=name,command=command,returncode=result.returncode,collected=len(collected),
                 unique=len(set(collected)),ran=count,test_ids=collected,log=log.name)
        summary['suites'].append(row)
        print(name,'exit',result.returncode,'collected',len(collected),'unique',len(set(collected)),'ran',count,flush=True)
    before=json.loads((HERE/'baseline_sha256.json').read_text(encoding='utf-8'))
    current={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    summary['outside_scope_changed']=[p for p,h in before.items() if not owned(p) and current.get(p)!=h]
    summary['outside_scope_added']=[p for p in current if p not in before and not owned(p)]
    summary['owned_files']=sorted(p for p in current if owned(p))
    summary['source_sha256']=hashlib.sha256((ROOT.parent/'自动任务赛代码设计总纲_v2.docx').read_bytes()).hexdigest()
    summary['passed']=all(s['returncode']==0 and s['ran']==s['collected']==s['unique'] for s in summary['suites'])
    (HERE/'verification_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('outside scope changed:',summary['outside_scope_changed'],flush=True)
    print('outside scope added:',summary['outside_scope_added'],flush=True)
    return 0 if summary['passed'] else 1


if __name__=='__main__':sys.exit(main())
