"""Synthetic end-to-end checks, not competition recognition validation."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'vision_workbench.py'


class WorkbenchCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-B', str(SCRIPT)] + list(args),
                              capture_output=True, encoding='utf-8')

    def test_help_does_not_require_ros(self):
        result = self.run_cli('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('inspect', result.stdout)
        self.assertIn('capture-ros', result.stdout)

    def test_demo_creates_auditable_unknown_results_without_templates(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / '中文输出'
            result = self.run_cli('demo', '--output', str(out))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['origin'], 'synthetic_demo')
            self.assertFalse(report['actionable'])
            self.assertFalse(report['real_world_validated'])
            self.assertEqual(len(report['frames']), 1)
            frame = report['frames'][0]
            self.assertTrue(frame['candidates'], frame)
            for candidate in frame['candidates']:
                self.assertEqual(candidate['classification']['category'], 'unknown')
                self.assertFalse(candidate['actionable'])
                self.assertTrue((out / candidate['crop_path']).is_file())
            self.assertTrue((out / frame['overlay_path']).is_file())
            saved = (out / 'report.json').read_bytes()
            again = self.run_cli('demo', '--output', str(out))
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual((out / 'report.json').read_bytes(), saved)

    def test_bad_input_is_error_not_empty_success(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.run_cli('inspect', '--input', str(Path(folder) / 'missing.png'),
                                  '--output', str(Path(folder) / 'out'))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('error', result.stderr.lower())

    def test_inspect_preserves_source_and_accepts_fractional_manual_regions(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / '原图.png'
            bgr = np.zeros((240, 320, 3), np.uint8)
            cv2.rectangle(bgr, (40, 40), (170, 180), (0, 190, 250), -1)
            source.write_bytes(cv2.imencode('.png', bgr)[1].tobytes())
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            config = Path(folder) / 'config.json'
            config.write_text(json.dumps({'detection': {'roi': [10.5, 10.5, 290.0, 210.0],
                                      'board_regions': [[250.5, 20.0, 40.0, 100.0]]}}), encoding='utf-8')
            out = Path(folder) / 'result'
            result = self.run_cli('inspect', '--input', str(source), '--output', str(out), '--config', str(config))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
            self.assertTrue(report['processing_complete'])
            self.assertEqual(report['origin'], 'offline_replay')
            self.assertEqual(report['frames'][0]['region_basis'], 'manual')
            self.assertTrue(report['frames'][0]['candidates'])
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_malformed_config_does_not_create_a_success_report(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'invalid.json'
            config.write_text('{"measurement":"verified"}', encoding='utf-8')
            out = Path(folder) / 'out'
            result = self.run_cli('demo', '--output', str(out), '--config', str(config))
            self.assertEqual(result.returncode, 2)
            self.assertFalse((out / 'report.json').exists())

    def test_nonfinite_json_config_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'invalid.json'
            config.write_text('{"detection":{"canny_low":NaN}}', encoding='utf-8')
            out = Path(folder) / 'out'
            result = self.run_cli('demo', '--output', str(out), '--config', str(config))
            self.assertEqual(result.returncode, 2)
            self.assertFalse(out.exists(), 'bad configuration must not leave a broken report')


if __name__ == '__main__':
    unittest.main()
