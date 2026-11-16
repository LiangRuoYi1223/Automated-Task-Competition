"""ROS 边界用最小替身；断言真实采集逻辑产生的报告。"""

from contextlib import contextmanager
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))
from treasure_auto.preflight import evaluate, load_profile

try:
    from treasure_auto.ros1_probe import ReceiptTracker, collect
except ModuleNotFoundError:
    ReceiptTracker = collect = None


@contextmanager
def ros_boundary(profile, ros_now=100.0, transform_stamp=99.9, missing_tf=False,
                 wrong_stream=False, wrong_driver_type=False, unreachable_driver=False):
    """仅替代本机不可用的 ROS graph、订阅和 TF API；不替代判断逻辑。"""
    class Stamp:
        def __init__(self, value):
            self.value = value

        def to_sec(self):
            return self.value

    class Time:
        def __init__(self, value=0):
            self.value = value

        @staticmethod
        def now():
            return Stamp(ros_now)

    class Subscription:
        def unregister(self):
            pass

    def subscribe(topic, message_class, callback, **kwargs):
        actual_type = next(s["type"] for s in profile["streams"] if s["topic"] == topic)
        if wrong_stream and topic == "/scan":
            actual_type = "std_msgs/Empty"
        # 模拟传输边界：通配符会收到错误类型；明确消息类仅接收匹配数据。
        if message_class is object or getattr(message_class, "_type", None) == actual_type:
            callback(types.SimpleNamespace(_type=actual_type, _connection_header={"type": actual_type}))
        return Subscription()

    def forbidden_publish(*args, **kwargs):
        raise AssertionError("只读自检尝试创建执行器发布者")

    rospy = types.SimpleNamespace(
        init_node=lambda *args, **kwargs: None,
        get_name=lambda: "/treasure_device_preflight_test",
        get_param=lambda name, default=None: "noetic" if name == "/rosdistro" else default,
        Subscriber=subscribe, Publisher=forbidden_publish, AnyMsg=object,
        is_shutdown=lambda: False, Time=Time,
    )

    class Master:
        def __init__(self, caller):
            pass

        def getSystemState(self):
            return (
                [(s["topic"], ["/device_source"]) for s in profile["streams"]],
                [(c["topic"], [c["subscriber"]]) for c in profile["commands"]],
                [],
            )

        def getTopicTypes(self):
            return [(s["topic"], s["type"]) for s in profile["streams"] + profile["commands"]]

        def lookupNode(self, name):
            return "http://test.local" + name

    class NodeProxy:
        def __init__(self, uri):
            self.node = uri.split("test.local", 1)[1]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def getSubscriptions(self, caller):
            if unreachable_driver:
                raise OSError("driver unavailable")
            subscriptions = [[c["topic"], "std_msgs/Bool" if wrong_driver_type and c["topic"] == "/pump_topic" else c["type"]]
                             for c in profile["commands"] if c["subscriber"] == self.node]
            return 1, "subscriptions", subscriptions

    class Listener:
        def lookupTransform(self, target, source, stamp):
            if missing_tf:
                raise LookupError("missing transform")
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)

        def getLatestCommonTime(self, target, source):
            return Stamp(transform_stamp)

    tf = types.SimpleNamespace(TransformListener=Listener, Exception=LookupError)
    roslib = types.ModuleType("roslib")
    roslib.__path__ = []
    message_module = types.ModuleType("roslib.message")
    message_module.get_message_class = lambda name: type("ExpectedMessage", (), {"_type": name, "_md5sum": "expected-md5"})
    roslib.message = message_module
    modules = {"rospy": rospy, "rosgraph": types.SimpleNamespace(Master=Master), "tf": tf,
               "roslib": roslib, "roslib.message": message_module}
    with patch.dict(sys.modules, modules), patch("xmlrpc.client.ServerProxy", NodeProxy):
        yield


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(collect, "尚未实现 ROS 只读采集器")
        self.profile = load_profile(str(PACKAGE / "config/hardware.json"))

    def test_receipt_age_uses_monotonic_arrival_time(self):
        clock_values = iter([0.05, 0.25, 0.5])
        tracker = ReceiptTracker(["/odom", "/scan"], clock=lambda: next(clock_values))
        tracker.record("/odom")
        tracker.record("/odom")
        data = tracker.snapshot()
        self.assertEqual(data["/odom"]["received_count"], 2)
        self.assertEqual(data["/odom"]["last_message_age_s"], 0.25)
        self.assertEqual(data["/scan"]["received_count"], 0)
        self.assertIsNone(data["/scan"]["last_message_age_s"])

    def test_collect_uses_subscriber_only_interfaces_and_returns_evaluable_snapshot(self):
        with ros_boundary(self.profile):
            snapshot = collect(self.profile, 0.01)
        report = evaluate(self.profile, snapshot, "live")
        self.assertTrue(report["passed"], report)
        self.assertEqual(snapshot["topics"]["/camera/color/image_raw"]["received_count"], 1)
        self.assertIn("python_version", snapshot["environment"])
        self.assertIn("os_pretty_name", snapshot["environment"])
        self.assertIn("command_echo", [c["code"] for c in report["checks"]])

    def test_missing_ros_is_an_explicit_environment_error(self):
        with patch.dict(sys.modules, {"rospy": None}), self.assertRaisesRegex(RuntimeError, "ROS 1"):
            collect(self.profile, 0.01)

    def test_invalid_observation_duration_is_rejected_before_ros_import(self):
        for value in [0, -1, True, float("nan"), float("inf")]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                collect(self.profile, value)

    def test_zero_dynamic_tf_timestamp_does_not_pass_as_fresh(self):
        with ros_boundary(self.profile, transform_stamp=0.0):
            snapshot = collect(self.profile, 0.01)
        report = evaluate(self.profile, snapshot, "live")
        self.assertFalse(report["passed"])
        self.assertTrue(any(c["key"] == "odometry_to_base" and c["code"] == "invalid_observation"
                            for c in report["checks"]))

    def test_paused_ros_clock_does_not_block_observation_window(self):
        with ros_boundary(self.profile, ros_now=0.0):
            snapshot = collect(self.profile, 0.01)
        self.assertFalse(evaluate(self.profile, snapshot, "live")["passed"])

    def test_missing_transform_is_reported_instead_of_assumed_available(self):
        with ros_boundary(self.profile, missing_tf=True):
            snapshot = collect(self.profile, 0.01)
        report = evaluate(self.profile, snapshot, "live")
        self.assertFalse(report["passed"])
        self.assertEqual(snapshot["transforms"]["base_footprint->camera_link"]["available"], False)

    def test_wrong_actual_stream_type_cannot_borrow_master_type(self):
        with ros_boundary(self.profile, wrong_stream=True):
            snapshot = collect(self.profile, 0.01)
        report = evaluate(self.profile, snapshot, "live")
        self.assertFalse(report["passed"])
        self.assertTrue(any(c["key"] == "laser" and c["code"] == "stream_silent" for c in report["checks"]))

    def test_wrong_actual_driver_subscription_type_fails(self):
        with ros_boundary(self.profile, wrong_driver_type=True):
            snapshot = collect(self.profile, 0.01)
        report = evaluate(self.profile, snapshot, "live")
        self.assertFalse(report["passed"])
        self.assertTrue(any(c["key"] == "pump_command" and c["code"] == "type_mismatch" for c in report["checks"]))

    def test_unreachable_registered_driver_does_not_pass(self):
        with ros_boundary(self.profile, unreachable_driver=True):
            snapshot = collect(self.profile, 0.01)
        report = evaluate(self.profile, snapshot, "live")
        self.assertFalse(report["passed"])
        self.assertIn("driver_type_unknown", [c["code"] for c in report["checks"]])


if __name__ == "__main__":
    unittest.main()
