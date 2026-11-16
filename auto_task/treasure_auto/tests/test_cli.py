"""真实入口与受控子进程的退出码、报告、超时行为。"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

try:
    from treasure_auto import cli
except ImportError:
    cli = None


class CliTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(cli, "尚未实现自检命令入口")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.profile = PACKAGE / "config/hardware.json"
        self.snapshot = self.directory / "snapshot.json"
        self.snapshot.write_text((PACKAGE / "examples/healthy_interfaces.json").read_text(encoding="utf-8"), encoding="utf-8")
        self.output = self.directory / "report.json"

    def invoke(self, extra, command=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(stdout))
            stack.enter_context(redirect_stderr(stderr))
            if command is not None:
                stack.enter_context(patch.object(cli, "_collector_command", return_value=command))
            code = cli.main(["--profile", str(self.profile), "--output", str(self.output)] + extra)
        report = json.loads(self.output.read_text(encoding="utf-8")) if self.output.exists() else None
        return code, report, stdout.getvalue(), stderr.getvalue()

    def test_offline_success_writes_explicit_offline_report(self):
        code, report, stdout, stderr = self.invoke(["--snapshot", str(self.snapshot)])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(report["mode"], "offline")
        self.assertTrue(report["passed"])
        self.assertIn("离线", stdout)
        self.assertEqual(stderr, "")

    def test_failed_stream_returns_one_with_report(self):
        snapshot = json.loads(self.snapshot.read_text(encoding="utf-8"))
        snapshot["topics"]["/scan"]["received_count"] = 0
        self.snapshot.write_text(json.dumps(snapshot), encoding="utf-8")
        code, report, _, _ = self.invoke(["--snapshot", str(self.snapshot)])
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])

    def test_malformed_snapshot_returns_two_without_success_report(self):
        self.snapshot.write_text("{broken", encoding="utf-8")
        code, report, _, stderr = self.invoke(["--snapshot", str(self.snapshot)])
        self.assertEqual(code, 2)
        self.assertFalse(report["passed"])
        self.assertTrue(stderr)

    def test_invalid_profile_returns_two(self):
        self.profile = self.directory / "invalid_profile.json"
        self.profile.write_text("{}", encoding="utf-8")
        code, report, _, stderr = self.invoke(["--snapshot", str(self.snapshot)])
        self.assertEqual(code, 2)
        self.assertFalse(report["passed"])
        self.assertTrue(stderr)

    def test_missing_snapshot_returns_two(self):
        code, _, _, stderr = self.invoke(["--snapshot", str(self.directory / "missing.json")])
        self.assertEqual(code, 2)
        self.assertTrue(stderr)

    def test_mode_must_be_explicit(self):
        code, report, _, _ = self.invoke([])
        self.assertEqual(code, 2)
        self.assertIsNone(report)

    def test_live_and_snapshot_are_mutually_exclusive(self):
        code, _, _, _ = self.invoke(["--live", "--snapshot", str(self.snapshot)])
        self.assertEqual(code, 2)

    def test_roslaunch_bookkeeping_arguments_do_not_break_the_entry(self):
        code, report, _, stderr = self.invoke([
            "--snapshot", str(self.snapshot), "__name:=treasure_device_preflight_report",
            "__log:=/tmp/treasure_device_preflight.log"])
        self.assertEqual(code, 0, stderr)
        self.assertEqual(report["mode"], "offline")

    def test_unknown_cli_arguments_are_still_rejected(self):
        code, _, _, _ = self.invoke(["--snapshot", str(self.snapshot), "--unknown-option"])
        self.assertEqual(code, 2)

    def test_bad_time_limits_are_rejected(self):
        for options in [["--observe", "nan"], ["--observe", "0"], ["--timeout", "inf"],
                        ["--observe", "3", "--timeout", "3"]]:
            with self.subTest(options=options):
                code, _, _, _ = self.invoke(["--live"] + options)
                self.assertEqual(code, 2)

    def test_hanging_collector_is_killed_and_reported_as_failure(self):
        command = [sys.executable, "-B", "-c", "import time; time.sleep(30)"]
        code, report, _, _ = self.invoke(["--live", "--observe", "0.01", "--timeout", "0.2"], command)
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(report["mode"], "live")
        self.assertIn("collector_timeout", [c["code"] for c in report["checks"]])

    def test_collector_failure_is_environment_error_not_a_pass(self):
        command = [sys.executable, "-B", "-c", "import sys; sys.stderr.write('ROS unavailable'); sys.exit(2)"]
        code, report, _, stderr = self.invoke(["--live"], command)
        self.assertEqual(code, 2)
        self.assertFalse(report["passed"])
        self.assertIn("ROS unavailable", stderr)

    def test_corrupt_collector_stdout_is_not_accepted_as_snapshot(self):
        command = [sys.executable, "-B", "-c", "print('not json')"]
        code, report, _, stderr = self.invoke(["--live"], command)
        self.assertEqual(code, 2)
        self.assertFalse(report["passed"])
        self.assertTrue(stderr)

    def test_error_replaces_a_previous_success_report(self):
        self.assertEqual(self.invoke(["--snapshot", str(self.snapshot)])[0], 0)
        self.snapshot.write_text("{broken", encoding="utf-8")
        code, report, _, _ = self.invoke(["--snapshot", str(self.snapshot)])
        self.assertEqual(code, 2)
        self.assertFalse(report["passed"])
        self.assertIn("tool_error", [c["code"] for c in report["checks"]])

    def test_report_cannot_overwrite_input_snapshot(self):
        before = self.snapshot.read_bytes()
        self.output = self.snapshot
        code, _, _, stderr = self.invoke(["--snapshot", str(self.snapshot)])
        self.assertEqual(code, 2)
        self.assertEqual(self.snapshot.read_bytes(), before)
        self.assertTrue(stderr)

    def test_live_collector_boundary_returns_a_live_scoped_report(self):
        command = [sys.executable, "-B", "-c", "import sys; print(open(sys.argv[1], encoding='utf-8').read())", str(self.snapshot)]
        code, report, _, stderr = self.invoke(["--live"], command)
        self.assertEqual(code, 0, stderr)
        self.assertEqual(report["mode"], "live")
        self.assertEqual(report["scope"], "device_interfaces")
        self.assertIn("physical_checks_pending", [c["code"] for c in report["checks"]])

    def test_script_is_runnable_without_ros_for_offline_mode(self):
        completed = subprocess.run(
            [sys.executable, "-B", str(PACKAGE / "scripts/device_preflight.py"),
             "--profile", str(self.profile), "--snapshot", str(self.snapshot), "--output", str(self.output)],
            capture_output=True, text=True, encoding="utf-8", timeout=5,
            env=dict(os.environ, PYTHONIOENCODING="ascii"))
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("离线快照", completed.stdout)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(report["mode"], "offline")


if __name__ == "__main__":
    unittest.main()
