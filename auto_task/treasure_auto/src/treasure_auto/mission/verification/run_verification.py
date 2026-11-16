"""Reproducible local-only P09 verification and trace export."""
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

OUT=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[5]
SRC=ROOT/'treasure_auto/src'
TESTS=ROOT/'treasure_auto/tests'
sys.path[:0]=[str(TESTS),str(SRC)]


def ids(suite):
    for item in suite:
        if isinstance(item,unittest.TestSuite):yield from ids(item)
        else:yield item.id()


def owned(path):
    return (path.startswith('treasure_auto/src/treasure_auto/mission/') or
            path.startswith('treasure_auto/config/mission/') or
            path.startswith('treasure_auto/tests/test_p09_') and path.endswith('.py') or
            path in ('docs/handoffs/P09_delivery.md','docs/handoffs/P09_interface_change.md'))


def main():
    result={'python':sys.version,'contract':'1.0.0','tests':{}}
    for label,pattern in (('p09','test_p09_*.py'),('c0','test_p00_*.py'),('all','test*.py')):
        suite=unittest.TestLoader().discover(str(TESTS),pattern=pattern)
        names=list(ids(suite))
        cmd=[sys.executable,'-B','-m','unittest','discover','-s','treasure_auto/tests','-p',pattern,'-v']
        run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        log=run.stdout+run.stderr;(OUT/(label+'_final.log')).write_text(log,encoding='utf-8')
        match=re.search(r'Ran (\d+) tests? in ([\d.]+)s',log)
        count=int(match.group(1)) if match else 0
        result['tests'][label]=dict(collected=len(names),executed=count,unique=len(set(names)),
            exit_code=run.returncode,seconds=float(match.group(2)) if match else None,command=cmd,ids=names)
        assert run.returncode==0 and count==len(names)==len(set(names)),log[-4000:]
        print(label,count,'PASS',flush=True)
    paths=list((SRC/'treasure_auto/mission').rglob('*.py'))+list(TESTS.glob('test_p09_*.py'))
    for path in paths:ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,8))
    result['python38_grammar_files']=len(paths)
    modules=[p.stem for p in (SRC/'treasure_auto/mission').glob('*.py') if p.stem!='__init__']
    probe='''import sys,importlib,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('rospy','roslib','serial','cv2','numpy'):raise RuntimeError('forbidden '+fullname)
sys.meta_path.insert(0,Block())
from treasure_auto.contracts import core as c,models
before=set(c._RECORDS)
for name in sys.argv[1:]:importlib.import_module('treasure_auto.mission.'+name)
assert set(c._RECORDS)==before
print('PASS: no device imports or C0 registration changes')
'''
    env=dict(os.environ,PYTHONPATH=str(SRC),PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
    run=subprocess.run([sys.executable,'-B','-c',probe]+modules,env=env,capture_output=True,text=True,encoding='utf-8')
    (OUT/'import_check.log').write_text(run.stdout+run.stderr,encoding='utf-8');assert run.returncode==0,run.stderr
    result['isolated_import']='PASS'
    from treasure_auto.mission.config import MissionConfig
    from treasure_auto.contracts.core import ContractError
    MissionConfig.load(ROOT/'treasure_auto/config/mission/offline.example.json')
    try:MissionConfig.load(ROOT/'treasure_auto/config/mission/unmeasured.json')
    except ContractError:result['unmeasured_config']='REJECTED as required'
    else:raise AssertionError('unmeasured config accepted')
    result['simulations']={}
    for layout in ('V1','V2','V3'):
        cmd=[sys.executable,'-B','-m','treasure_auto.mission.simulator','--layout',layout]
        run=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8');assert run.returncode==0,run.stderr
        (OUT/('simulation_'+layout+'.json')).write_text(run.stdout,encoding='utf-8')
        data=json.loads(run.stdout)
        result['simulations'][layout]={k:data[k] for k in ('origin','arrivals','layout_complete','conditional_points','elapsed_s','original_deadline_s')}
    baseline=json.loads((OUT/'baseline_sha256.json').read_text(encoding='utf-8'))
    changed=[]
    for name,digest in baseline.items():
        p=ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:changed.append(name)
    current=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    unexpected=[p for p in current if p not in baseline and not owned(p)]
    result['scope']=dict(baseline_files=len(baseline),changed_existing=changed,new_outside_scope=unexpected)
    result['owned_files']=sorted(p for p in current if owned(p))
    result['ok']=not changed and not unexpected
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('tests','owned_files')},ensure_ascii=False,indent=2))
    assert result['ok'],'scope changed; inspect summary, never revert other package work blindly'


if __name__=='__main__':main()
