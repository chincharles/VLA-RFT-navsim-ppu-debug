#!/usr/bin/env python3
"""ClearML Python entry for the supplied ZW810E two-card PPU environment."""
import os
from pathlib import Path
import subprocess
import sys
if __name__=='__main__':
    env=dict(os.environ);env.setdefault('RFT_PYTHON',sys.executable)
    raise SystemExit(subprocess.call(['bash',str(Path(__file__).with_suffix('.sh')),*sys.argv[1:]],env=env))
