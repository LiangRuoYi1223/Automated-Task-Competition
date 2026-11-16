"""A failed rerun must not leave an earlier PASS as its latest result."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from integration import verify_post_p11 as verifier


class VerifierEvidenceTests(unittest.TestCase):
    def test_failed_collection_replaces_old_pass_with_this_run_failure(self):
        for fault in (OSError('test discovery unavailable'), KeyboardInterrupt()):
            with self.subTest(fault=type(fault).__name__), tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'verification.json'
                path.write_text(json.dumps({'status':'PASS','old_run':True}),encoding='utf-8')
                try:
                    with patch.object(unittest.TestLoader,'discover',side_effect=fault):
                        code=verifier.main(['--output-dir',folder])
                except BaseException as error:
                    code=error
                self.assertEqual(code,1)
                report=json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(report['status'],'FAIL')
                self.assertNotIn('old_run',report)
                self.assertIn(type(fault).__name__,report['error'])
                self.assertTrue(all(check['status']=='NOT_RUN' for check in report['checks'].values()))


if __name__=='__main__':
    unittest.main()
