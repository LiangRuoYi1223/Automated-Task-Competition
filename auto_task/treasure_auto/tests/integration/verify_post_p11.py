"""Reproduce post-P11 offline verification; never starts robot drivers."""
import argparse
import ast
import contextlib
import hashlib
import json
import os
import tempfile
import platform
import shutil
import subprocess
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
TESTS = PACKAGE / 'tests'
sys.path[:0] = [str(TESTS), str(PACKAGE / 'src')]
from integration.acceptance import EvidenceResult, _flatten, run_acceptance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'post_p11' / 'latest')
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    # Keep generated reports separate from package inputs and source files.
    for name in ('src','config','scripts','examples','launch'):
        root = (PACKAGE / name).resolve()
        if out == root or root in out.parents:
            parser.error('Output directory must not be a package input/source directory')
    out.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    labels=('unittest','acceptance','audit','formal_blocked','nominal_V1','nominal_V2','nominal_V3','all_scenarios','structure')
    report = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  python=sys.version, platform=platform.platform(), status='RUNNING',
                  hardware='UNVERIFIED', formal_assembly='INCOMPLETE',
                  checks={label:dict(status='NOT_RUN') for label in labels})
    _publish(out,report)  # Invalidate any previous PASS before touching logs.
    try:
        _perform(out,report)
    except (Exception,KeyboardInterrupt) as error:
        report['error']=type(error).__name__+': '+str(error)
    finally:
        report['status']='PASS' if 'error' not in report and all(item['status']=='PASS' for item in report['checks'].values()) else 'FAIL'
        report['elapsed_s']=round(time.monotonic()-began,3)
        _publish(out,report)
    print(report['status'],out/'verification.json')
    return 0 if report['status']=='PASS' else 1


def _publish(out,report):
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',suffix='.json.tmp',dir=str(out),delete=False) as stream:
        temp=Path(stream.name)
        json.dump(report,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    os.replace(str(temp),str(out/'verification.json'))


def _perform(out,report):
    log = out / 'all_tests.log'
    suite = unittest.TestLoader().discover(str(TESTS))
    ids = [test.id() for test in _flatten(suite)]
    with log.open('w', encoding='utf-8') as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2, resultclass=EvidenceResult).run(suite)
    actual = [item['id'] for item in result.records]
    ok = bool(ids) and len(ids) == len(set(ids)) and sorted(ids) == sorted(actual)
    ok = ok and result.testsRun == len(ids) and all(item['status'] == 'PASS' for item in result.records)
    report['checks']['unittest'] = dict(status='PASS' if ok else 'FAIL',
        collected=len(ids), executed=result.testsRun, unique=len(set(actual)),
        skips=len(result.skipped), results=result.records)
    print('unittest:', report['checks']['unittest']['status'], result.testsRun, 'tests', flush=True)
    matrix = json.loads((HERE / 'scenario_matrix.json').read_text(encoding='utf-8'))
    acceptance = run_acceptance(matrix, out / 'acceptance')
    report['checks']['acceptance'] = {key: acceptance[key] for key in
        ('status','executed_count','unique_executed_count','formal_integration','hardware')}
    commands = [('audit',['--audit'],0), ('formal_blocked',['--mode','formal'],2)]
    commands += [('nominal_'+layout,['--layout',layout,'--scenario','nominal'],0) for layout in ('V1','V2','V3')]
    commands += [('all_scenarios',['--scenario','all'],0)]
    for label, options, expected in commands:
        command = [sys.executable,'-B',str(PACKAGE/'scripts/run_match_offline.py')] + options
        completed = subprocess.run(command, cwd=str(out), capture_output=True, text=True,
                                   encoding='utf-8', errors='replace', timeout=120)
        (out/(label+'.json')).write_text(completed.stdout,encoding='utf-8')
        (out/(label+'.stderr.log')).write_text(completed.stderr,encoding='utf-8')
        parsed = False
        try:
            parsed = isinstance(json.loads(completed.stdout),dict)
        except ValueError:
            pass
        report['checks'][label] = dict(status='PASS' if completed.returncode==expected and parsed else 'FAIL',
            returncode=completed.returncode, expected_returncode=expected)
    errors=[]; py_count=0; json_count=0; xml_count=0
    for path in list((PACKAGE/'src').rglob('*.py'))+list((PACKAGE/'scripts').rglob('*.py')):
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path),feature_version=(3,8))
            py_count+=1
        except (SyntaxError,ValueError) as error:
            errors.append(str(error))
    for root in ('config','examples'):
        for path in (PACKAGE/root).rglob('*.json'):
            try:
                json.loads(path.read_text(encoding='utf-8-sig')); json_count+=1
            except ValueError as error:
                errors.append(str(error))
    for path in list((PACKAGE/'launch').glob('*.launch'))+[PACKAGE/'package.xml']:
        try:
            ET.parse(str(path)); xml_count+=1
        except ET.ParseError as error:
            errors.append(str(error))
    report['checks']['structure'] = dict(status='FAIL' if errors else 'PASS',python38_syntax_files=py_count,
        json_files=json_count,xml_files=xml_count,errors=errors,actual_python38_runtime='UNVERIFIED')
    latest_path=HERE/'packaging/artifacts/latest.json'
    if latest_path.is_file():
        latest=json.loads(latest_path.read_text(encoding='utf-8'))
        work=Path(latest['run_directory']).resolve()
        if (HERE/'packaging/artifacts').resolve() in work.parents:
            target=out/'installation';target.mkdir(exist_ok=True)
            for source in work.glob('*.log'):
                shutil.copy2(str(source),str(target/source.name))
            (target/'summary.json').write_text(json.dumps(latest,indent=2)+'\n',encoding='utf-8')
    fingerprints={}
    for name in ('src','config','scripts','launch','examples'):
        for path in (PACKAGE/name).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                fingerprints[path.relative_to(PACKAGE).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    report['tested_package_files_sha256']=fingerprints


if __name__=='__main__':
    raise SystemExit(main())
