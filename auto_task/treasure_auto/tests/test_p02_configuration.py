"""P02 unmeasured examples and private file-format validation."""
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace, fields
from test_p02_support import c, m, env, bundle, policy, poly, proof, transforms, Clock
from treasure_auto.mapping import storage, localization as loc

try:
    from treasure_auto.mapping import configuration as config
except ImportError:
    config = None


class ConfigurationTests(unittest.TestCase):
    def test_R02_waypoints_saved_separately_and_reload_bound_to_map(self):
        with tempfile.TemporaryDirectory() as d:
            saved=storage.save_bundle(bundle(),d,'map-v1')
            points=Path(d)/'map-v1/waypoints.json'
            self.assertTrue(points.exists(),'point records need a separate version-bound file')
            self.assertEqual(storage.load_bundle(points.parent/'map.json','map-v1'),saved)
            points.write_text('{}',encoding='utf-8')
            with self.assertRaises(c.ContractError):
                storage.load_bundle(points.parent/'map.json','map-v1')

    def test_R04_localization_expired_envelope_returns_unknown(self):
        clock=Clock()
        tracker=loc.LocalizationTracker(env(),policy(),clock,'base_link')
        from test_p02_support import pose
        tracker.initialize(pose())
        for t in (10,10.2,10.4):
            clock.t=t
            a,b=transforms(t)
            b=replace(b,envelope=replace(b.envelope,expires_at=replace(env().emitted_at,seconds=t+.1)))
            tracker.observe(a,b)
        clock.t=10.6
        self.assertEqual(tracker.snapshot().trusted,c.Fact.UNKNOWN)

    def test_R04_each_chain_capture_must_advance(self):
        clock=Clock()
        tracker=loc.LocalizationTracker(env(),policy(),clock,'base_link')
        from test_p02_support import pose
        tracker.initialize(pose())
        tracker.observe(*transforms())
        a,b=transforms(10.01)
        b=replace(b,evidence=replace(proof(10.01,'replayed-odom'),captured_at=proof(10).captured_at),
                  source_now=proof(10.01).captured_at)
        clock.t=10.01
        result=tracker.observe(a,b)
        self.assertEqual(result.trusted,c.Fact.UNKNOWN)
        self.assertEqual(result.reason.code,c.ErrorCode.DUPLICATE)
        self.assertEqual(len(tracker.samples),0)

    def test_R04_sample_configs_fail_closed_and_points_are_unmeasured(self):
        self.assertIsNotNone(config,'P02 configuration loader missing')
        root=Path(__file__).resolve().parents[1]/'config/mapping'
        with self.assertRaises(c.ContractError) as error:
            config.load_parameters(root/'parameters.unmeasured.json',c.Origin.LIVE)
        self.assertEqual(error.exception.code,c.ErrorCode.UNCONFIGURED)
        b=storage.load_bundle(root/'example-unmeasured-v1/map.json','example-unmeasured-v1')
        self.assertEqual(b.metadata.measurement,m.Measurement.UNMEASURED)
        self.assertTrue(all(p.pose is None for p in b.waypoints))

    def test_R04_simulated_parameters_validate_but_cannot_be_live(self):
        self.assertIsNotNone(config,'P02 configuration loader missing')
        data=dict(format=1,contract_version=c.CONTRACT_VERSION,measurement='simulated',evidence=[],
            base_frame='base_link',mapping_source='sim-mapper',localization_source='sim-localizer',
            localization={f.name:getattr(policy(),f.name) for f in fields(policy()) if f.name not in ('measurement','evidence')},
            footprint=dict(body=c.to_data(poly(-.2,-.2,.2,.2)),carried=None,margin_m=.01))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'parameters.json'
            path.write_text(json.dumps(data),encoding='utf-8')
            result=config.load_parameters(path,c.Origin.SIMULATED)
            self.assertEqual(result.base_frame,'base_link')
            with self.assertRaises(c.ContractError):
                config.load_parameters(path,c.Origin.LIVE)
            data['unknown']=True
            path.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaises(c.ContractError):
                config.load_parameters(path,c.Origin.SIMULATED)

    def test_R04_missing_numeric_or_wrong_unit_is_not_defaulted(self):
        self.assertIsNotNone(config,'P02 configuration loader missing')
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'parameters.json'
            for text in ('{"format":1}', '{"format":1,"format":1}', '{"format":NaN}'):
                path.write_text(text,encoding='utf-8')
                with self.assertRaises(c.ContractError):
                    config.load_parameters(path,c.Origin.SIMULATED)

    def test_R03_python38_syntax_and_import_without_ROS(self):
        root=Path(__file__).resolve().parents[1]
        for path in (root/'src/treasure_auto/mapping').glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,8))
        code="""import sys
class Block:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('rospy','rosgraph','serial','cv2','tf'):
            raise RuntimeError('unexpected device import '+fullname)
sys.meta_path.insert(0,Block())
from treasure_auto.mapping import storage, geometry, localization, service
print('offline imports OK')
"""
        import os
        environment=dict(os.environ,PYTHONPATH=str(root/'src'))
        result=subprocess.run([sys.executable,'-B','-c',code],env=environment,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
