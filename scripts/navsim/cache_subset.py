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
    command = [sys.executable, str(Path(a.navsim_root).resolve()
               / 'navsim/planning/script/run_metric_caching.py'),
               'train_test_split=navtrain', 'worker=sequential',
               'cache.cache_path=' + json.dumps(str(cache)),
               'navsim_log_path=' + json.dumps(str(Path(a.logs).resolve())),
               'train_test_split.scene_filter.tokens=' + json.dumps(sorted(tokens))]
    (cache / 'requested_tokens.json').write_text(json.dumps(sorted(tokens), indent=2))
    print(f'Official metric caching for {len(tokens)} selected tokens', flush=True)
    subprocess.run(command, check=True)
    print(f'Cache: {cache}. Evaluator will additionally enforce complete token coverage.')


if __name__ == '__main__':
    main()
