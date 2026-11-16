"""C0 artifacts, wire cases, Python baseline and no-device import contract."""
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / 'src'))
from treasure_auto.contracts import core as c, rules as r


class ArtifactTests(unittest.TestCase):
    def test_R06_schema_is_frozen_and_current(self):
        from contract_fixtures.generate_artifacts import schema
        saved = json.loads((PACKAGE / 'src/treasure_auto/contracts/schema_v1.json').read_text(encoding='utf-8'))
        self.assertEqual(saved, schema())

    def test_R05_serialized_normal_and_abnormal_cases(self):
        cases = json.loads((PACKAGE / 'tests/contract_fixtures/cases.json').read_text(encoding='utf-8'))
        self.assertGreaterEqual(len(cases), 30)
        for case in cases:
            with self.subTest(case=case['id']):
                self.assertEqual(case['origin'], 'simulated')
                def check():
                    record = c.loads(json.dumps(case['wire']))
                    r.require_simulated(record)
                    if case['validator'] == 'layout_quality':
                        r.require_layout_quality(record)
                    return record
                if case['expected'] == 'accept':
                    check()
                else:
                    with self.assertRaises(c.ContractError):
                        check()

    def test_T06_import_and_use_with_ros_and_device_imports_blocked(self):
        script = '''import importlib.abc, sys
class NoDevices(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'rospy','roslib','actionlib','sensor_msgs','geometry_msgs','serial','cv2'}:
   raise AssertionError('device dependency imported: ' + fullname)
sys.meta_path.insert(0,NoDevices())
sys.path.insert(0,sys.argv[1]); sys.path.insert(0,sys.argv[2])
from treasure_auto.contracts import core, models, ports, rules
from contract_fixtures.samples import all_samples, state_snapshot, release_transaction
from contract_fixtures.fakes import FakeStateStore
for sample in all_samples().values():
 assert core.loads(core.dumps(sample)) == sample
assert FakeStateStore(state_snapshot()).commit(release_transaction()).status is models.CommitStatus.COMMITTED
print('no-device-import: PASS')
'''
        result = subprocess.run([sys.executable, '-B', '-c', script, str(PACKAGE / 'src'),
                                 str(PACKAGE / 'tests')], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('PASS', result.stdout)

    def test_T06_all_p00_python_uses_38_grammar(self):
        files = list((PACKAGE / 'src/treasure_auto/contracts').glob('*.py'))
        files += list((PACKAGE / 'tests/contract_fixtures').glob('*.py'))
        files += list((PACKAGE / 'tests').glob('test_p00_*.py'))
        for path in files:
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path), feature_version=(3, 8))

    def test_R06_every_public_record_is_in_field_dictionary(self):
        from treasure_auto.contracts import models
        from dataclasses import is_dataclass
        dictionary = (PACKAGE / 'src/treasure_auto/contracts/FIELD_DICTIONARY.md').read_text(encoding='utf-8')
        for module in (c, models):
            for name, value in vars(module).items():
                if isinstance(value, type) and is_dataclass(value) and value.__module__ == module.__name__:
                    self.assertIn('## ' + name + '\n', dictionary)


if __name__ == '__main__':
    unittest.main()
