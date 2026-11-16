"""P11-T01/T07: run the actual entrypoint without device connections."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
SCRIPT = PACKAGE / 'scripts/run_match_offline.py'


class CliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-B', str(SCRIPT)] + list(args),
                              cwd=str(PACKAGE.parent), capture_output=True,
                              encoding='utf-8', timeout=30)

    def test_source_entry_runs_complete_flow_and_never_claims_physical_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            result = self.run_cli('--output', str(path), '__name:=p11_test')
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(report['results']['nominal']['arrivals'], 10)
            self.assertEqual(report['status'], 'passed')
            for tier in ('ros', 'device_control', 'physical_action', 'full_match'):
                self.assertEqual(report['tiers'][tier], 'unverified')

    def test_formal_mode_is_blocked_and_cannot_reuse_a_success_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_text('{"status":"passed"}', encoding='utf-8')
            result = self.run_cli('--mode', 'formal', '--output', str(path))
            self.assertEqual(result.returncode, 2)
            report = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'blocked')
            self.assertEqual(report['results'], {})
            self.assertTrue(report['blockers'])

    def test_unknown_scenario_and_ros_topic_remapping_fail(self):
        for args in (('--scenario', 'made_up'), ('/cmd_vel:=/bypass',)):
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 2)

    def test_unmeasured_config_and_output_input_alias_refused(self):
        result = self.run_cli('--config', str(PACKAGE / 'config/mission/unmeasured.json'))
        self.assertEqual(result.returncode, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            original = (PACKAGE / 'config/mission/offline.example.json').read_bytes()
            path.write_bytes(original)
            result = self.run_cli('--config', str(path), '--output', str(path))
            self.assertEqual(result.returncode, 2)
            self.assertEqual(path.read_bytes(), original)

    def test_audit_checks_all_deliveries_without_claiming_scenario_pass(self):
        result = self.run_cli('--audit')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['validation'], 'dependency_audit')
        self.assertEqual(len(report['dependencies']), 12)
        self.assertEqual(report['results'], {})

    def test_missing_configuration_replaces_previous_success_with_blocked_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_text('{"status":"passed"}', encoding='utf-8')
            result = self.run_cli('--config', str(Path(directory) / 'missing.json'), '--output', str(path))
            self.assertEqual(result.returncode, 2)
            report = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'blocked')
            self.assertIn('error', report)

    def test_report_destination_cannot_replace_launch_script_or_build_inputs(self):
        sys.path.insert(0, str(PACKAGE / 'src'))
        from treasure_auto.contracts import core as c
        from treasure_auto.integration.cli import _protect_output
        config = PACKAGE / 'config/mission/offline.example.json'
        # Check the guard directly: never invoke a destructive CLI against originals.
        for name in ('scripts/run_match_offline.py', 'scripts/device_preflight.py',
                     'setup.py', 'CMakeLists.txt', 'package.xml', 'launch/match_offline.launch'):
            with self.subTest(name=name), self.assertRaises(c.ContractError):
                _protect_output(PACKAGE / name, config)

    def test_all_replays_can_run_with_device_and_network_apis_forbidden(self):
        # Blocking real boundaries catches accidental import-time/device setup.
        code = '''
import importlib.abc, sys, socket, threading
sys.path.insert(0, sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('rospy','rosgraph','serial','cv2','actionlib'):
            raise RuntimeError('forbidden device import: '+fullname)
sys.meta_path.insert(0, Block())
def deny(*args, **kwargs): raise RuntimeError('real IO forbidden')
socket.socket.connect = deny
threading.Thread.start = deny
from treasure_auto.integration.cli import main
raise SystemExit(main(['--scenario','all','--output',sys.argv[2]]))
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'all.json'
            result = subprocess.run([sys.executable, '-B', '-c', code, str(PACKAGE / 'src'), str(path)],
                                    capture_output=True, encoding='utf-8', timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(len(report['results']), 24)
            for name in ('field_push','field_stall','field_budget','field_regions','field_recenter'):
                self.assertTrue(report['results'][name]['field_actions'])
            self.assertTrue(all(x['status'] == 'confirmed' for x in report['contract_report']['checks']))


if __name__ == '__main__':
    unittest.main()
