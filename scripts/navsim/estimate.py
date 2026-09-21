#!/usr/bin/env python3
"""Extrapolate measured step cost; never invent an unmeasured GPU-hour estimate."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--logs',nargs='+',required=True);p.add_argument('--steps',type=int,required=True);p.add_argument('--warmup',type=int,default=2)
a=p.parse_args();costs=[];tokens=set();peak=0
for file in a.logs:
    rows=[json.loads(x) for x in Path(file).read_text().splitlines() if x.strip()]
    rows=rows[a.warmup:]
    if not rows:raise ValueError('Need measured steps after warmup')
    costs.append(statistics.median(r['seconds'] for r in rows))
    for r in rows:tokens.update(r['tokens']);peak=max(peak,r['peak_memory_bytes'])
slowest=max(costs)
print(json.dumps(dict(measured_ranks=len(costs),steady_step_seconds=slowest,
    projected_training_hours=slowest*a.steps/3600,peak_rank_GiB=peak/1024**3,
    observed_unique_scenes=len(tokens),excludes='setup, preprocessing, checkpoint IO and evaluation'),indent=2))
