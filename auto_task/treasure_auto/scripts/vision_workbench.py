#!/usr/bin/env python3
"""Source and installed entry point for read-only vision preparation."""
from pathlib import Path
import sys

source_root = Path(__file__).resolve().parents[1] / 'src'
if source_root.is_dir():
    sys.path.insert(0, str(source_root))

from treasure_auto.vision_workbench.cli import main

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.exit(main())
