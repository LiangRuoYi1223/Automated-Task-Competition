"""ROS 无关的只读设备接口检查；通过不代表真实动作已验收。"""

import json
import math
from pathlib import Path


def _finite_number(value, positive=False):
    try:
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and (value > 0 if positive else value >= 0))
    except OverflowError:
        return False

def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _schema(value, label):
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("{} 必须是 schema_version=1 的对象".format(label))


def validate_profile(profile):
    """拒绝空检查集、重复标识以及无法执行的配置。"""
    _schema(profile, "配置")
    if not _text(profile.get("expected_ros_distro")):
        raise ValueError("配置缺少 expected_ros_distro")
    keys = {"ros_distro", "physical_verification", "offline"}
    topics = set()
    pairs = set()
    for group in ("streams", "commands", "transforms"):
        entries = profile.get(group)
        if not isinstance(entries, list) or not entries:
            raise ValueError("{} 必须是非空列表".format(group))
        for entry in entries:
            if not isinstance(entry, dict) or not _text(entry.get("key")):
                raise ValueError("{} 检查项缺少 key".format(group))
            key = entry["key"]
            if key in keys or ":" in key:
                raise ValueError("检查 key 重复或包含保留字符: {}".format(key))
            keys.add(key)
            if group == "transforms":
                target, source = entry.get("target"), entry.get("source")
                if not _text(target) or not _text(source) or target == source:
                    raise ValueError("TF 需要不同且非空的 target/source")
                pair = (target, source)
                if pair in pairs:
                    raise ValueError("TF 检查重复")
                pairs.add(pair)
                if "max_age_s" not in entry:
                    raise ValueError("TF 必须明确设置 max_age_s；只有显式 null 才允许静态变换")
                age = entry["max_age_s"]
                if age is not None and not _finite_number(age, positive=True):
                    raise ValueError("动态 TF max_age_s 必须是有限正数")
                continue
            topic, message_type = entry.get("topic"), entry.get("type")
            if not _text(topic) or not topic.startswith("/") or any(c.isspace() for c in topic):
                raise ValueError("话题必须是以 / 开头的完整名称")
            if topic in topics:
                raise ValueError("话题检查重复: {}".format(topic))
            topics.add(topic)
            if not _text(message_type) or len(message_type.split("/")) != 2 or not all(message_type.split("/")):
                raise ValueError("消息类型应为 package/Message")
            if group == "streams":
                if not _finite_number(entry.get("max_age_s"), positive=True):
                    raise ValueError("数据流 max_age_s 必须是有限正数")
                if "echo_only" in entry and type(entry["echo_only"]) is not bool:
                    raise ValueError("echo_only 必须是布尔值")
            else:
                subscriber = entry.get("subscriber")
                if not _text(subscriber) or not subscriber.startswith("/"):
                    raise ValueError("控制入口需指定完整驱动节点名 subscriber")
    return profile


def load_profile(path):
    """读取 UTF-8 JSON 配置并校验；不连接设备。"""
    profile = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return validate_profile(profile)


def evaluate(profile, snapshot, mode):
    """判断接口观测。无实际动作、定位精度或真空成功结论。"""
    validate_profile(profile)
    _schema(snapshot, "快照")
    if mode not in ("live", "offline"):
        raise ValueError("mode 必须为 live 或 offline")
    for field in ("environment", "topics", "transforms"):
        if not isinstance(snapshot.get(field), dict):
            raise ValueError("快照 {} 必须是对象".format(field))
    checks = []

    def add(key, status, code, detail):
        checks.append({"key": key, "status": status, "code": code, "detail": detail})

    distro = snapshot["environment"].get("ros_distro")
    if not _text(distro):
        add("ros_distro", "FAIL", "ros_distro_unknown", "未采到 ROS 发行版")
    elif distro.strip().lower() != profile["expected_ros_distro"].strip().lower():
        add("ros_distro", "FAIL", "ros_distro_mismatch", "实际 ROS={}，预期 {}；需核实适配".format(distro, profile["expected_ros_distro"]))
    else:
        add("ros_distro", "PASS", "ros_distro_ok", "ROS={}".format(distro))

    for entry in profile["streams"]:
        key, topic = entry["key"], entry["topic"]
        observed = snapshot["topics"].get(topic)
        if observed is None:
            add(key, "FAIL", "stream_missing", "{} 没有发布数据流".format(topic))
        elif not isinstance(observed, dict):
            add(key, "FAIL", "invalid_observation", "{} 观测格式无效".format(topic))
        elif observed.get("type") != entry["type"]:
            add(key, "FAIL", "type_mismatch", "{} 消息类型与配置不一致".format(topic))
        elif not isinstance(observed.get("publishers"), list) or not all(_text(n) for n in observed["publishers"]):
            add(key, "FAIL", "invalid_observation", "{} 发布者列表无效".format(topic))
        elif not observed["publishers"]:
            add(key, "FAIL", "stream_missing", "{} 没有发布者".format(topic))
        else:
            count, age = observed.get("received_count"), observed.get("last_message_age_s")
            if type(count) is not int or count < 0:
                add(key, "FAIL", "invalid_observation", "{} 接收计数无效".format(topic))
            elif count == 0:
                add(key, "FAIL", "stream_silent", "{} 已注册，但观察期内未收到消息".format(topic))
            elif not _finite_number(age):
                add(key, "FAIL", "invalid_observation", "{} 最近接收时间无效".format(topic))
            elif age > entry["max_age_s"]:
                add(key, "FAIL", "stream_stale", "{} 最近接收已过去 {:.3f} 秒".format(topic, age))
            else:
                add(key, "PASS", "stream_ok", "{} 收到 {} 条，最近 {:.3f} 秒".format(topic, count, age))
        if entry.get("echo_only"):
            add(key + ":feedback", "WARN", "command_echo", "{} 为指令记录流，不能证明机械臂到位或吸附成功".format(topic))

    for entry in profile["commands"]:
        key, topic = entry["key"], entry["topic"]
        observed = snapshot["topics"].get(topic)
        if observed is None:
            add(key, "FAIL", "driver_missing", "{} 未找到驱动订阅端".format(topic))
        elif not isinstance(observed, dict):
            add(key, "FAIL", "invalid_observation", "{} 观测格式无效".format(topic))
        elif observed.get("type") != entry["type"]:
            add(key, "FAIL", "type_mismatch", "{} 消息类型与配置不一致".format(topic))
        elif not isinstance(observed.get("subscribers"), list) or not all(_text(n) for n in observed["subscribers"]):
            add(key, "FAIL", "invalid_observation", "{} 订阅者列表无效".format(topic))
        elif entry["subscriber"] not in observed["subscribers"]:
            add(key, "FAIL", "driver_missing", "{} 未被预期驱动 {} 订阅".format(topic, entry["subscriber"]))
        elif not isinstance(observed.get("subscriber_types"), dict) or not _text(observed["subscriber_types"].get(entry["subscriber"])):
            add(key, "FAIL", "driver_type_unknown", "{} 无法确认驱动 {} 实际订阅的消息类型".format(topic, entry["subscriber"]))
        elif observed["subscriber_types"][entry["subscriber"]] != entry["type"]:
            add(key, "FAIL", "type_mismatch", "{} 的驱动实际订阅类型与配置不一致".format(topic))
        else:
            add(key, "PASS", "driver_registered", "{} 的驱动订阅端存在；未发送指令".format(topic))

    for entry in profile["transforms"]:
        key = entry["key"]
        pair = "{}->{}".format(entry["target"], entry["source"])
        observed = snapshot["transforms"].get(pair)
        if observed is None:
            add(key, "FAIL", "tf_missing", "{} 不可用".format(pair))
        elif not isinstance(observed, dict) or type(observed.get("available")) is not bool:
            add(key, "FAIL", "invalid_observation", "{} TF 观测格式无效".format(pair))
        elif not observed["available"]:
            add(key, "FAIL", "tf_missing", "{} 不可用".format(pair))
        elif entry.get("max_age_s") is None:
            add(key, "PASS", "tf_available", "{} 存在，按配置允许静态变换".format(pair))
        elif not _finite_number(observed.get("age_s")):
            add(key, "FAIL", "invalid_observation", "{} 动态 TF 时间无效".format(pair))
        elif observed["age_s"] > entry["max_age_s"]:
            add(key, "FAIL", "tf_stale", "{} 动态 TF 已过期".format(pair))
        else:
            add(key, "PASS", "tf_available", "{} 存在且时间符合配置".format(pair))

    if mode == "offline":
        add("offline", "WARN", "offline_snapshot", "离线快照判断，不代表当前机器人状态")
    add("physical_verification", "WARN", "physical_checks_pending", "尚未验证实际移动、机械臂到位、吸附、标定精度和急停；接口通过不是比赛启动许可")
    return {"schema_version": 1, "scope": "device_interfaces", "mode": mode,
            "environment": snapshot["environment"], "passed": not any(c["status"] == "FAIL" for c in checks),
            "checks": checks}
