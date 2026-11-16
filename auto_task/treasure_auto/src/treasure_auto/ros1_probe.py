"""在机器人已有 ROS 1 环境中收集只读接口快照。"""

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import shlex
import sys
import threading
import time
import xmlrpc.client

from .preflight import load_profile, validate_profile


class ReceiptTracker:
    """按单调时钟记录实际收到的消息，不读取指令回显中的位置值。"""

    def __init__(self, topics, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._counts = {topic: 0 for topic in topics}
        self._last = {topic: None for topic in topics}

    def record(self, topic):
        with self._lock:
            self._counts[topic] += 1
            self._last[topic] = self._clock()

    def snapshot(self):
        with self._lock:
            now = self._clock()
            return {topic: {"received_count": count,
                            "last_message_age_s": None if self._last[topic] is None else now - self._last[topic]}
                    for topic, count in self._counts.items()}


def read_environment():
    """读取当前进程所在环境，不推断机器人型号或默认 Ubuntu 版本。"""
    system_name = platform.platform()
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if line.startswith("PRETTY_NAME="):
                values = shlex.split(line.split("=", 1)[1])
                if values:
                    system_name = values[0]
                break
    except (OSError, ValueError, UnicodeError):
        pass
    return {"os_pretty_name": system_name, "python_version": platform.python_version(),
            "machine": platform.machine(), "ros_distro": os.environ.get("ROS_DISTRO", "")}


def collect(profile, observe_s):
    """只读 ROS graph、订阅流与 TF；整体有界性由调用进程的超时保证。"""
    validate_profile(profile)
    if (isinstance(observe_s, bool) or not isinstance(observe_s, (int, float))
            or not math.isfinite(observe_s) or observe_s <= 0):
        raise ValueError("观察时长必须是有限正数")
    try:
        import rospy
        import rosgraph
        import tf
        from roslib.message import get_message_class
    except ImportError as error:
        raise RuntimeError("需要 ROS 1 的 rospy、rosgraph、roslib、tf；请在机器人已加载 ROS 环境的终端运行") from error

    rospy.init_node("treasure_device_preflight", anonymous=True, disable_signals=True)
    environment = read_environment()
    distro = rospy.get_param("/rosdistro", environment["ros_distro"])
    environment["ros_distro"] = distro.strip() if isinstance(distro, str) else ""
    master = rosgraph.Master(rospy.get_name())
    tracker = ReceiptTracker([entry["topic"] for entry in profile["streams"]])
    listener = tf.TransformListener()
    subscriptions = []
    try:
        for entry in profile["streams"]:
            topic = entry["topic"]
            message_class = get_message_class(entry["type"])
            if message_class is None:
                raise RuntimeError("找不到 ROS 消息类 {}，请确认对应消息包已生成并加载".format(entry["type"]))
            subscriptions.append(rospy.Subscriber(
                topic, message_class, lambda message, topic=topic: tracker.record(topic),
                queue_size=1, buff_size=2**24))
        deadline = time.monotonic() + observe_s
        while time.monotonic() < deadline:
            if rospy.is_shutdown():
                raise RuntimeError("ROS 在观察期间关闭")
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

        publishers, subscribers, _ = master.getSystemState()
        types = dict(master.getTopicTypes())
        published_by, subscribed_by = dict(publishers), dict(subscribers)
        topics = {}
        for entry in profile["streams"] + profile["commands"]:
            topic = entry["topic"]
            topics[topic] = {"type": types.get(topic), "publishers": published_by.get(topic, []),
                             "subscribers": subscribed_by.get(topic, [])}

        # Master 的全局话题类型不等于某个驱动自身实际订阅的类型。
        # 直接查询预期驱动，失败时保留空证据，由判断层明确报错。
        driver_types, driver_errors = {}, {}
        for entry in profile["commands"]:
            topic, node = entry["topic"], entry["subscriber"]
            if node not in driver_types and node in subscribed_by.get(topic, []):
                try:
                    with xmlrpc.client.ServerProxy(master.lookupNode(node)) as proxy:
                        code, message, pairs = proxy.getSubscriptions(rospy.get_name())
                    if code != 1:
                        raise RuntimeError(message)
                    driver_types[node] = dict(pairs)
                except Exception as error:
                    driver_types[node] = {}
                    driver_errors[node] = str(error)
            actual_type = driver_types.get(node, {}).get(topic)
            topics[topic]["subscriber_types"] = {node: actual_type} if actual_type else {}
            if node in driver_errors:
                topics[topic]["subscriber_error"] = driver_errors[node]

        transforms = {}
        for entry in profile["transforms"]:
            target, source = entry["target"], entry["source"]
            pair = "{}->{}".format(target, source)
            try:
                listener.lookupTransform(target, source, rospy.Time(0))
                age = None
                if entry.get("max_age_s") is not None:
                    stamp = listener.getLatestCommonTime(target, source).to_sec()
                    now = rospy.Time.now().to_sec()
                    if now > 0 and stamp > 0:
                        age = now - stamp
                transforms[pair] = {"available": True, "age_s": age}
            except Exception as error:
                transforms[pair] = {"available": False, "age_s": None, "error": str(error)}
        for topic, receipt in tracker.snapshot().items():
            topics[topic].update(receipt)
        return {"schema_version": 1, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "environment": environment, "topics": topics, "transforms": transforms}
    finally:
        for subscription in subscriptions:
            subscription.unregister()


def main(argv=None):
    parser = argparse.ArgumentParser(description="ROS 1 只读快照采集器，由自检入口限时调用")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--observe", type=float, default=3.0)
    args = parser.parse_args(argv)
    try:
        profile = load_profile(args.profile)
        # 把 ROS 的普通日志转往 stderr，stdout 仅输出可解析快照。
        with redirect_stdout(sys.stderr):
            snapshot = collect(profile, args.observe)
        print(json.dumps(snapshot, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        print("采集失败：{}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
