"""Method selection and diagnostic provenance, without ROS or actuation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class LineCliTests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/vision_workbench.py')] + list(args),
                              capture_output=True, encoding='utf-8')

    def test_capabilities_are_explicit(self):
        result = self.run_cli('capabilities')
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertIn('opencv_version', data)
        self.assertIs(type(data['lsd']), bool)
        self.assertIs(type(data['edgedrawing']), bool)

    def test_lsd_demo_records_parameters_support_and_incomplete_state(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / '诊断'
            result = self.run_cli('demo', '--method', 'lsd', '--output', str(out))
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((out / 'report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['configuration']['method'], 'lsd')
            frozen = json.dumps(report['configuration'], sort_keys=True, separators=(',', ':'),
                                ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.assertEqual(report['configuration_sha256'], hashlib.sha256(frozen).hexdigest())
            self.assertFalse(report['actionable'])
            self.assertFalse(report['real_world_validated'])
            frame = report['frames'][0]
            self.assertIn('budget_truncated', frame)
            self.assertIn('timing_ms', frame)
            self.assertIn('unresolved', frame)
            self.assertTrue((out / frame['overlay_path']).is_file())
            self.assertTrue((out / frame['support_overlay_path']).is_file())
            for candidate in frame['candidates'] + frame['unresolved']:
                self.assertEqual(len(candidate['edges']), 4)
                self.assertIn('edge_metrics', candidate)
                self.assertFalse(candidate['body_confirmed'])
                self.assertEqual(candidate['plane_basis'], 'unknown')
                self.assertFalse(candidate['actionable'])

    def test_unknown_method_or_line_key_is_rejected_without_output(self):
        from treasure_auto.vision_workbench.cli import read_config
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.json'
            for config in ({'method': 'invented'}, {'line_detection': {'typo': 2}}):
                path.write_text(json.dumps(config), encoding='utf-8')
                with self.assertRaises(ValueError):
                    read_config(path)

    def test_budget_exhaustion_does_not_report_scene_absent(self):
        from treasure_auto.vision_workbench.cli import read_config, run_inspection, demo_frames
        with tempfile.TemporaryDirectory() as folder:
            config = read_config()
            config['method'] = 'lsd'
            config['line_detection']['max_segments'] = 4
            report = run_inspection(demo_frames(), Path(folder) / 'limited', config)
            self.assertTrue(report['frames'][0]['budget_truncated'])
            self.assertFalse(report['search_complete'])
            self.assertEqual(report['status'], 'diagnostic_incomplete')

    def test_evaluate_binds_reference_pixels_and_protects_input(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'report.json'
            references = Path(folder) / 'references.json'
            output = Path(folder) / 'metrics.json'
            digest = 'a' * 64
            frame = dict(index=0, source=dict(width=640, height=480, pixel_sha256=digest),
                         candidates=[], unresolved=[], timing_ms=5., budget_truncated=False)
            source.write_text(json.dumps(dict(schema_version=1, processing_complete=True,
                                             configuration={}, frames=[frame])), encoding='utf-8')
            reference = dict(index=0, width=640, height=480, pixel_sha256=digest,
                             scene_id='empty-floor-1', split='acceptance', instances=[])
            references.write_text(json.dumps(dict(schema_version=1, measurement='manual_reference_unverified',
                                                  frames=[reference])), encoding='utf-8')
            result = self.run_cli('evaluate', '--report', str(source), '--references', str(references),
                                  '--output', str(output))
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(metrics['negative_images'], 1)
            self.assertEqual(metrics['negative_image_false_positive_rate'], 0.)
            self.assertIsNone(metrics['passed'])
            before = source.read_bytes()
            result = self.run_cli('evaluate', '--report', str(source), '--references', str(references),
                                  '--output', str(source))
            self.assertEqual(result.returncode, 2)
            self.assertEqual(before, source.read_bytes())


if __name__ == '__main__':
    unittest.main()
