"""P06 import/compatibility and collection checks; no robot access."""
import ast
import subprocess
import sys
import unittest
from pathlib import Path
from test_p06_fixtures import module, c, m


class VerificationTests(unittest.TestCase):
    def test_R01_imports_do_not_require_device_network_or_scientific_packages(self):
        source=Path(__file__).resolve().parents[1]/'src'
        script="""
import importlib.abc, sys
class NoHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'rospy','serial','socket','cv2','numpy'}:
            raise RuntimeError('forbidden import: '+fullname)
sys.meta_path.insert(0, NoHardware())
sys.path.insert(0, sys.argv[1])
from treasure_auto.geometry.service import GeometryService
from treasure_auto.geometry.support import compute_support
from treasure_auto.geometry.calibration import make_report
print('offline imports OK')
"""
        p=subprocess.run([sys.executable,'-B','-c',script,str(source)],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('offline imports OK',p.stdout)

    def test_R01_p06_uses_python38_syntax(self):
        pkg=Path(__file__).resolve().parents[1]
        files=list((pkg/'src/treasure_auto/geometry').rglob('*.py'))+list((pkg/'tests').glob('test_p06_*.py'))
        self.assertGreater(len(files),15)
        for path in files:
            with self.subTest(path=path.name):ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,8))
