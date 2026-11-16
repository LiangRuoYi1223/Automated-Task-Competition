"""Reproducible offline checks, logs, collection and ownership evidence for P08."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[5]
OUT=Path(__file__).resolve().parent
TESTS=ROOT/'treasure_auto/tests'
SRC=ROOT/'treasure_auto/src'


def owned(path):
    return (path.startswith('treasure_auto/src/treasure_auto/placement/') or
            path.startswith('treasure_auto/config/placement/') or
            path.startswith('treasure_auto/tests/test_p08_') or
            path in ('docs/handoffs/P08_delivery.md','docs/handoffs/P08_interface_change.md'))


def ids(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from ids(item)
        else:yield item.id()


def main():
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(SRC),str(TESTS)]
    results={};environment=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
    for name,pattern in (('p08','test_p08_*.py'),('p00','test_p00_*.py'),('all','test*.py')):
        collected=list(ids(unittest.TestLoader().discover(str(TESTS),pattern=pattern)))
        command=[sys.executable,'-B','-m','unittest','discover','-s','treasure_auto/tests','-p',pattern,'-v']
        done=subprocess.run(command,cwd=str(ROOT),env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
        (OUT/(name+'_final.log')).write_text(done.stdout,encoding='utf-8')
        match=re.search(r'Ran (\d+) tests?',done.stdout);ran=int(match.group(1)) if match else None
        results[name]=dict(command=command,exit_code=done.returncode,collected=len(collected),ran=ran,
            unique=len(set(collected)),collection_complete=(len(collected)==ran==len(set(collected))),ids=collected)
        print(name, {k:v for k,v in results[name].items() if k not in ('ids','command')},flush=True)
        if done.returncode:print(done.stdout[-6000:],flush=True)
    python_files=list((SRC/'treasure_auto/placement').rglob('*.py'))+list(TESTS.glob('test_p08_*.py'))
    for p in python_files:ast.parse(p.read_text(encoding='utf-8'),filename=str(p),feature_version=(3,8))
    import_code='''import sys,importlib
class NoDeviceImports:
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('rospy','roslib','serial','cv2'):raise AssertionError('unexpected hardware dependency '+fullname)
sys.meta_path.insert(0,NoDeviceImports())
from treasure_auto.contracts.core import _RECORDS
before=set(_RECORDS)
from treasure_auto.contracts import models
before=set(_RECORDS)
for name in ('config','inputs','guards','targets','space','evidence','layout','recovery','execution','service'):
 importlib.import_module('treasure_auto.placement.'+name)
assert set(_RECORDS)==before,'local inputs must not register new public contracts'
print('PASS: no ROS/serial/cv2 imports and no C0 registry mutation')
'''
    safe_env=dict(environment,PYTHONPATH=str(SRC))
    imported=subprocess.run([sys.executable,'-B','-c',import_code],cwd=str(ROOT),env=safe_env,capture_output=True,text=True,encoding='utf-8')
    (OUT/'import_check.log').write_text(imported.stdout+imported.stderr,encoding='utf-8')
    from treasure_auto.placement.config import PlacementConfig
    from treasure_auto.contracts.core import ContractError
    cfg=ROOT/'treasure_auto/config/placement'
    PlacementConfig.from_dict(json.loads((cfg/'offline.example.json').read_text(encoding='utf-8')))
    blocked=False
    try:PlacementConfig.from_dict(json.loads((cfg/'unmeasured.json').read_text(encoding='utf-8')))
    except ContractError:blocked=True
    baseline=json.loads((OUT/'baseline_sha256.json').read_text(encoding='utf-8'))
    actual={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    changed=[p for p,h in baseline.items() if actual.get(p)!=h]
    outside=sorted(p for p in actual if p not in baseline and not owned(p))
    source=ROOT.parent/'自动任务赛代码设计总纲_v2.docx'
    summary=dict(python=sys.version,contract='1.0.0',tests=results,python38_syntax_files=len(python_files),
        import_exit=imported.returncode,unmeasured_rejected=blocked,baseline_files=len(baseline),
        changed_existing=changed,new_outside_scope=outside,owned_files=sorted(p for p in actual if owned(p)),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('checks',dict(python38_syntax_files=len(python_files),import_exit=imported.returncode,unmeasured_rejected=blocked,
                      baseline_files=len(baseline),changed_existing=changed,new_outside_scope=outside),flush=True)
    return 0 if (all(x['exit_code']==0 and x['collection_complete'] for x in results.values()) and imported.returncode==0 and blocked and not changed and not outside) else 1


if __name__=='__main__':sys.exit(main())
