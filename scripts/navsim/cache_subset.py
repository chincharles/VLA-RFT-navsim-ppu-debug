#!/usr/bin/env python3
"""Cache exactly the exported navtrain train/validation tokens using official NAVSIM."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--navsim-root', required=True)
    p.add_argument('--logs', required=True)
    p.add_argument('--manifests', nargs='+', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--worker',choices=['sequential','ray_distributed'],default='sequential')
    p.add_argument('--workers',type=int,default=8)
    a = p.parse_args()
    from navsim_rft.data import verify_navsim
    from navsim_rft import NAVSIM_COMMIT
    verify_navsim(a.navsim_root)
    if not Path(os.environ.get('NUPLAN_MAPS_ROOT', '__MISSING__')).is_dir():
        p.error('Set NUPLAN_MAPS_ROOT to the existing nuPlan maps root before running')
    tokens = set()
    for name in a.manifests:
        manifest = json.loads(Path(name).read_text())
        if (manifest['navsim_commit'] != NAVSIM_COMMIT or manifest['split'] != 'navtrain'
                or manifest['role'] not in ('train', 'val') or not manifest['records']):
            p.error(f'Expected a nonempty pinned-v1.1 navtrain train/val manifest: {name}')
        tokens.update(r['token'] for r in manifest['records'])
    cache = Path(a.cache).resolve()
    cache.mkdir(parents=True, exist_ok=False)
    # A full navtrain token list exceeds OS argument-length limits. Put it in a
    # primary Hydra config instead of interpolating it into the command line.
    from omegaconf import OmegaConf
    config_dir=cache/'selected_config';config_dir.mkdir()
    cfg=OmegaConf.load(Path(a.navsim_root)/'navsim/planning/script/config/metric_caching/default_metric_caching.yaml')
    cfg.train_test_split={'scene_filter':{'tokens':sorted(tokens)}}
    OmegaConf.save(cfg,config_dir/'selected_metric_caching.yaml')
    command = [sys.executable, str(Path(a.navsim_root).resolve()
               / 'navsim/planning/script/run_metric_caching.py'),
               '--config-dir',str(config_dir),'--config-name','selected_metric_caching',
               'train_test_split=navtrain', 'worker='+a.worker,
               'cache.cache_path=' + json.dumps(str(cache)),
               'navsim_log_path=' + json.dumps(str(Path(a.logs).resolve()))]
    if a.worker=='ray_distributed':
        if a.workers<1:p.error('--workers must be positive')
        command.append(f'worker.threads_per_node={a.workers}')
    (cache / 'requested_tokens.json').write_text(json.dumps(sorted(tokens), indent=2))
    print(f'Official metric caching for {len(tokens)} selected tokens', flush=True)
    subprocess.run(command, check=True)
    from navsim.common.dataloader import MetricCacheLoader
    covered=set(MetricCacheLoader(cache).tokens)
    if not tokens.issubset(covered):raise RuntimeError(f'Official metric cache missing {len(tokens-covered)} requested tokens')
    print(f'Cache: {cache}. Verified coverage of {len(tokens)} requested tokens.')


if __name__ == '__main__':
    main()
