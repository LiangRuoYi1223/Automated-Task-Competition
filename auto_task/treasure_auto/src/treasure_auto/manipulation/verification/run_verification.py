"""Reproduce offline suites; no ROS, device, deployment, or source mutation."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


def main():
    output=Path(__file__).resolve().parent
    root=output.parents[4]
    tests=root/'treasure_auto/tests'
    summary={'python':sys.version,'contract_version':'1.0.0','suites':{}}
    commands=[('p07','test_p07_*.py'),('c0','test_p00_*.py'),('all','test*.py')]
    for name,pattern in commands:
        loader=unittest.TestLoader()
        count=loader.discover(str(tests),pattern=pattern).countTestCases()
        command=[sys.executable,'-B','-m','unittest','discover','-s','treasure_auto/tests','-p',pattern,'-v']
        proc=subprocess.run(command,cwd=str(root),capture_output=True,encoding='utf-8',errors='replace')
        log=proc.stdout+proc.stderr
        (output/(name+'_final.log')).write_text(log,encoding='utf-8')
        summary['suites'][name]={'command':command,'collected':count,'returncode':proc.returncode,
                                'tail':log[-1800:]}
        print(name,count,'exit',proc.returncode,flush=True)
    command=[sys.executable,'-B','-m','unittest',
             'test_p10b_boundaries.BoundaryTests.test_T05_slow_commit_returns_unknown_then_preserves_actual_commit','-v']
    env=os.environ.copy(); env['PYTHONPATH']=str(tests)
    proc=subprocess.run(command,cwd=str(root),env=env,capture_output=True,encoding='utf-8',errors='replace')
    (output/'p10b_baseline_recheck.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    summary['baseline_recheck']={'command':command,'returncode':proc.returncode,'output':proc.stdout+proc.stderr}
    owned=[]
    for directory in (root/'treasure_auto/src/treasure_auto/manipulation',root/'treasure_auto/config/manipulation'):
        owned.extend(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    owned.extend(tests.glob('test_p07_*.py'))
    owned.extend((root/'docs/handoffs').glob('P07_*.md'))
    parsed=[]
    for path in owned:
        if path.suffix=='.py':
            ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,8))
            parsed.append(str(path.relative_to(root)))
    summary['python38_syntax_files']=parsed
    baseline=json.loads((output/'baseline_sha256.json').read_text(encoding='utf-8'))
    changed=[]; missing=[]
    for relative,digest in baseline.items():
        path=root/relative
        if not path.is_file(): missing.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=digest: changed.append(relative)
    summary['baseline_files_checked']=len(baseline)
    summary['baseline_changed']=changed; summary['baseline_missing']=missing
    summary['owned_files']={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(owned) if p.name!='verification_summary.json'}
    summary['source_sha256']=hashlib.sha256((root.parent/'自动任务赛代码设计总纲_v2.docx').read_bytes()).hexdigest()
    summary['schema_sha256']=hashlib.sha256((root/'treasure_auto/src/treasure_auto/contracts/schema_v1.json').read_bytes()).hexdigest()
    (output/'verification_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('baseline changed',changed,'missing',missing,'Python 3.8 syntax',len(parsed),flush=True)
    return int(any(suite['returncode'] for suite in summary['suites'].values()) or bool(changed) or bool(missing))


if __name__=='__main__':
    sys.exit(main())
