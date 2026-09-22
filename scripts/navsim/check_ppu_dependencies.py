#!/usr/bin/env python3
"""Import the real first-version model/evaluator stack in the isolated PPU env."""
import argparse
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    modules=['torch','torchvision','numpy','scipy','pandas','hydra','omegaconf',
        'prismatic.extern.hf.modeling_prismatic','ivideogpt.tokenizer',
        'navsim.common.dataclasses','navsim.common.dataloader','navsim.evaluate.pdm_score',
        'navsim.planning.script.run_metric_caching','navsim.planning.script.run_pdm_score',
        'navsim_rft.agent']
    loaded={name:str(importlib.import_module(name).__file__) for name in modules}
    if metadata.version('nuplan-devkit')!='1.2.0':raise RuntimeError('Wrong nuPlan devkit version')
    import numpy as np
    if np.__version__!='1.26.4' or np.int is not int:raise RuntimeError('NAVSIM NumPy compatibility alias missing')
    Path(a.output).write_text(json.dumps(dict(python=sys.version,imports=loaded,
        nuplan=metadata.version('nuplan-devkit'),passed=True,real_training=False),indent=2))
    print('PPU profile model and official evaluator imports passed')

if __name__=='__main__':main()
