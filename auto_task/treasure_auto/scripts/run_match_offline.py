#!/usr/bin/env python3
"""P11 source/catkin/wheel entry; SIMULATED only, no robot control adapters."""
from pathlib import Path
import sys

source_root = Path(__file__).resolve().parents[1] / 'src'
if (source_root / 'treasure_auto/integration/__init__.py').is_file():
    sys.path.insert(0, str(source_root))

from treasure_auto.integration.cli import main

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.exit(main())
