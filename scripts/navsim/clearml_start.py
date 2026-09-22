#!/usr/bin/env python3
"""Python entry point for ClearML interfaces that require a .py script."""
import os
from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    env = dict(os.environ)
    env.setdefault('RFT_PYTHON', sys.executable)
    script = Path(__file__).resolve().with_suffix('.sh')
    raise SystemExit(subprocess.call(['bash', str(script), *sys.argv[1:]], env=env))
