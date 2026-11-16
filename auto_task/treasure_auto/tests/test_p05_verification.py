import ast
import subprocess
import sys
import unittest
from pathlib import Path


class VerificationTests(unittest.TestCase):
    def test_R01_Python38_syntax_and_no_required_external_dependencies(self):
        root=Path(__file__).resolve().parents[1]
        files=list((root/'src/treasure_auto/classification').rglob('*.py'))+list((root/'tests').glob('test_p05_*.py'))
        self.assertTrue(files)
        for path in files:
            with self.subTest(file=path.name): ast.parse(path.read_text(encoding='utf-8-sig'),feature_version=(3,8))

    def test_R01_import_does_not_add_C0_records_or_connect_devices(self):
        root=Path(__file__).resolve().parents[1]
        script='''
import builtins, sys
sys.path.insert(0, sys.argv[1])
from treasure_auto.contracts import core, models
before=set(core._RECORDS)
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'rospy','roslib','serial','cv2','numpy'}:
        raise AssertionError('unexpected import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
import treasure_auto.classification
assert set(core._RECORDS)==before, 'private adapters must not register as new public contracts'
print('P05 import clean; public C0 unchanged')
'''
        result=subprocess.run([sys.executable,'-B','-c',script,str(root/'src')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

