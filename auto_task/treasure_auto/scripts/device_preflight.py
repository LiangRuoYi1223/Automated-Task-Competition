#!/usr/bin/env python3
"""支持在源码目录离线运行，也支持 catkin 安装后的 rosrun 入口。"""

from pathlib import Path
import sys

source_root = Path(__file__).resolve().parents[1] / "src"
if source_root.is_dir():
    sys.path.insert(0, str(source_root))

from treasure_auto.cli import main

if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.exit(main())
