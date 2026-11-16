"""检查会导致现场误报的输入；预期值由场景手工确定。"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

try:
    from treasure_auto.preflight import evaluate, load_profile
except ModuleNotFoundError:
    evaluate = load_profile = None


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(evaluate, "尚未实现设备自检判断逻辑")
        self.profile = json.loads((PACKAGE / "config/hardware.json").read_text(encoding="utf-8"))
        self.snapshot = json.loads((PACKAGE / "examples/healthy_interfaces.json").read_text(encoding="utf-8"))

    def report(self, mode="live"):
        return evaluate(self.profile, self.snapshot, mode)

    def assert_failed(self, key, code):
        report = self.report()
        self.assertFalse(report["passed"])
        self.assertTrue(any(c["key"] == key and c["status"] == "FAIL" and c["code"] == code
                            for c in report["checks"]), report)

    def test_valid_interfaces_pass_but_keep_physical_verification_warning(self):
        report = self.report()
        self.assertTrue(report["passed"])
        self.assertEqual(report["scope"], "device_interfaces")
        self.assertIn("physical_checks_pending", [c["code"] for c in report["checks"]])
        self.assertTrue(any(c["status"] == "WARN" and c["code"] == "command_echo"
                            for c in report["checks"]))

    def test_missing_stream_fails(self):
        del self.snapshot["topics"]["/scan"]
        self.assert_failed("laser", "stream_missing")

    def test_stream_without_publisher_fails(self):
        self.snapshot["topics"]["/scan"]["publishers"] = []
        self.assert_failed("laser", "stream_missing")

    def test_registered_stream_without_received_messages_fails(self):
        self.snapshot["topics"]["/scan"]["received_count"] = 0
        self.snapshot["topics"]["/scan"]["last_message_age_s"] = None
        self.assert_failed("laser", "stream_silent")

    def test_stale_stream_fails(self):
        self.snapshot["topics"]["/odom"]["last_message_age_s"] = 1.01
        self.assert_failed("odometry", "stream_stale")

    def test_invalid_stream_age_never_passes(self):
        for value in [None, -0.01, float("nan"), float("inf"), "0.1", True, 10**400]:
            with self.subTest(value=value):
                self.snapshot["topics"]["/odom"]["last_message_age_s"] = value
                self.assert_failed("odometry", "invalid_observation")

    def test_boolean_or_negative_message_count_never_passes(self):
        for value in [True, -1, 0.5]:
            with self.subTest(value=value):
                self.snapshot["topics"]["/odom"]["received_count"] = value
                self.assert_failed("odometry", "invalid_observation")

    def test_wrong_stream_message_type_fails(self):
        self.snapshot["topics"]["/scan"]["type"] = "sensor_msgs/Image"
        self.assert_failed("laser", "type_mismatch")

    def test_unrelated_command_subscriber_does_not_replace_driver(self):
        self.snapshot["topics"]["/cmd_vel"]["subscribers"] = ["/debug_echo"]
        self.assert_failed("base_command", "driver_missing")

    def test_missing_command_topic_fails(self):
        del self.snapshot["topics"]["/pump_topic"]
        self.assert_failed("pump_command", "driver_missing")

    def test_wrong_command_message_type_fails(self):
        self.snapshot["topics"]["/pump_topic"]["type"] = "std_msgs/Bool"
        self.assert_failed("pump_command", "type_mismatch")

    def test_wrong_driver_subscription_type_fails_despite_correct_master_type(self):
        self.snapshot["topics"]["/pump_topic"]["subscriber_types"] = {"/swiftpro_write_node": "std_msgs/Bool"}
        self.assert_failed("pump_command", "type_mismatch")

    def test_missing_driver_type_evidence_does_not_pass(self):
        del self.snapshot["topics"]["/pump_topic"]["subscriber_types"]
        self.assert_failed("pump_command", "driver_type_unknown")

    def test_missing_transform_fails(self):
        del self.snapshot["transforms"]["base_footprint->camera_link"]
        self.assert_failed("base_to_camera", "tf_missing")

    def test_unavailable_transform_fails(self):
        self.snapshot["transforms"]["odom->base_footprint"]["available"] = False
        self.assert_failed("odometry_to_base", "tf_missing")

    def test_stale_dynamic_transform_fails(self):
        self.snapshot["transforms"]["odom->base_footprint"]["age_s"] = 2.0
        self.assert_failed("odometry_to_base", "tf_stale")

    def test_dynamic_transform_without_valid_time_fails(self):
        for value in [None, -0.1, float("nan")]:
            with self.subTest(value=value):
                self.snapshot["transforms"]["odom->base_footprint"]["age_s"] = value
                self.assert_failed("odometry_to_base", "invalid_observation")

    def test_static_transform_does_not_need_nonzero_time(self):
        self.snapshot["transforms"]["base_footprint->camera_link"]["age_s"] = None
        self.assertTrue(self.report()["passed"])

    def test_wrong_ros_distro_fails(self):
        self.snapshot["environment"]["ros_distro"] = "foxy"
        self.assert_failed("ros_distro", "ros_distro_mismatch")

    def test_unknown_ros_distro_does_not_pass(self):
        self.snapshot["environment"]["ros_distro"] = ""
        self.assert_failed("ros_distro", "ros_distro_unknown")

    def test_offline_result_is_identified_as_offline(self):
        report = self.report("offline")
        self.assertEqual(report["mode"], "offline")
        self.assertIn("offline_snapshot", [c["code"] for c in report["checks"]])

    def test_invalid_profile_is_rejected(self):
        original = copy.deepcopy(self.profile)
        for bad_age in [0, -1, float("nan"), True, 10**400]:
            with self.subTest(age=bad_age):
                self.profile = copy.deepcopy(original)
                self.profile["streams"][0]["max_age_s"] = bad_age
                with self.assertRaises(ValueError):
                    self.report()

    def test_duplicate_check_key_is_rejected(self):
        self.profile["commands"][0]["key"] = "odometry"
        with self.assertRaises(ValueError):
            self.report()

    def test_missing_transform_age_policy_is_invalid_configuration(self):
        del self.profile["transforms"][0]["max_age_s"]
        self.snapshot["transforms"]["odom->base_footprint"]["age_s"] = 20.0
        with self.assertRaises(ValueError):
            self.report()

    def test_invalid_snapshot_shape_is_rejected(self):
        for invalid in [{}, {"schema_version": 2}, {"schema_version": 1, "topics": []}]:
            with self.subTest(snapshot=invalid), self.assertRaises(ValueError):
                evaluate(self.profile, invalid, "offline")

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.report("pretend-live")

    def test_profile_loader_rejects_malformed_json_and_nonobject(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            for content in ["{broken", "[]"]:
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_profile(str(path))


if __name__ == "__main__":
    unittest.main()
