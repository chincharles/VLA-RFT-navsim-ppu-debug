#!/usr/bin/env python3
"""ClearML entry for the first-version single-node two-CUDA-GPU smoke run."""
import os
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    env=dict(os.environ)
    env.setdefault('RFT_PYTHON',sys.executable)
    raise SystemExit(subprocess.call(['bash',str(Path(__file__).with_suffix('.sh')),*sys.argv[1:]],env=env))
