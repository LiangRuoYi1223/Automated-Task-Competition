"""面向用户的只读自检入口；实机采集在限时子进程内执行。"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .preflight import evaluate, load_profile


def _collector_command(profile_path, observe_s):
    return [sys.executable, "-B", "-m", "treasure_auto.ros1_probe", "--profile", profile_path,
            "--observe", str(observe_s)]


def _collect_snapshot(profile_path, observe_s, timeout_s):
    environment = os.environ.copy()
    module_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = module_root + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(_collector_command(profile_path, observe_s), capture_output=True,
                               text=True, encoding="utf-8", env=environment, timeout=timeout_s)
    if completed.returncode != 0:
        raise RuntimeError("ROS 采集器退出 {}：{}".format(completed.returncode, completed.stderr.strip()[-2000:]))
    return json.loads(completed.stdout)


def _failure_report(mode, code, detail):
    return {"schema_version": 1, "scope": "device_interfaces", "mode": mode,
            "environment": {}, "passed": False, "checks": [
                {"key": "collector", "status": "FAIL", "code": code, "detail": detail},
                {"key": "physical_verification", "status": "WARN", "code": "physical_checks_pending",
                 "detail": "尚未验证实际移动、机械臂到位、吸附、标定精度和急停"}]}


def _same_file(first, second):
    first, second = Path(first), Path(second)
    if first.resolve() == second.resolve():
        return True
    return first.exists() and second.exists() and first.samefile(second)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Spark-I 只读设备接口自检，不启动驱动或发送动作")
    parser.add_argument("--profile", required=True, help="硬件接口 JSON 配置")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--live", action="store_true", help="采集当前 ROS 1 接口")
    mode_group.add_argument("--snapshot", help="检查离线 JSON 快照")
    parser.add_argument("--observe", type=float, default=3.0, help="消息观察秒数，默认 3")
    parser.add_argument("--timeout", type=float, default=12.0, help="采集总超时秒数，默认 12")
    parser.add_argument("--output", help="另存本次检查报告为 UTF-8 JSON")
    try:
        args, ros_metadata = parser.parse_known_args(argv)
        # roslaunch 自动追加的进程元数据由 ROS 启动器管理，不是本工具的业务参数。
        # 话题和驱动名称仍以配置文件为准；其他未知参数继续报错。
        allowed_metadata = ("__name:=", "__log:=", "__ns:=")
        if any(not value.startswith(allowed_metadata) or not value.split(":=", 1)[-1]
               for value in ros_metadata):
            parser.error("无法识别的参数：{}".format(" ".join(ros_metadata)))
    except SystemExit as error:
        return error.code
    mode = "live" if args.live else "offline"
    # 不能用报告覆盖配置或观测原件，包括链接到同一个文件的路径。
    try:
        if args.output and any(_same_file(args.output, path) for path in [args.profile, args.snapshot] if path):
            raise ValueError("报告输出路径不能与输入配置或快照相同")
    except (OSError, ValueError) as error:
        print("自检错误：{}".format(error), file=sys.stderr)
        return 2
    try:
        if not all(math.isfinite(v) and v > 0 for v in (args.observe, args.timeout)) or args.timeout <= args.observe:
            raise ValueError("观察与超时须为有限正数，且总超时必须大于观察时长")
        profile = load_profile(args.profile)
        if args.live:
            snapshot = _collect_snapshot(args.profile, args.observe, args.timeout)
        else:
            snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8-sig"))
        report = evaluate(profile, snapshot, mode)
        exit_code = 0 if report["passed"] else 1
    except subprocess.TimeoutExpired:
        report = _failure_report(mode, "collector_timeout", "ROS 采集超过 {} 秒，已终止采集进程".format(args.timeout))
        exit_code = 1
    except (OSError, ValueError, RuntimeError) as error:
        detail = "自检错误：{}".format(error)
        print(detail, file=sys.stderr)
        report = _failure_report(mode, "tool_error", detail)
        exit_code = 2
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError) as error:
        print("报告保存失败：{}".format(error), file=sys.stderr)
        return 2
    print("{}：设备接口检查{}（未验证真实动作）".format(
        "实机采集" if mode == "live" else "离线快照",
        "无失败项" if report["passed"] else "未通过"))
    for check in report["checks"]:
        print("[{}] {}：{}".format(check["status"], check["key"], check["detail"]))
    if args.output:
        print("本次报告：{}".format(Path(args.output).resolve()))
    return exit_code
