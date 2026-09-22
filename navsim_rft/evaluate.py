"""Run v1.1 official evaluator entrypoint with a runtime-only Hydra configuration."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from .data import verify_navsim,build_loader

def main():
    p=argparse.ArgumentParser()
    for k in ('navsim-root','checkpoint','logs','sensors','metric-cache','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--split',choices=['navtest','navmini','navtrain'],default='navtest')
    p.add_argument('--manifest',help='Restrict to an exported validation/test manifest; no training manifests')
    p.add_argument('--vlm'); a=p.parse_args()
    verify_navsim(a.navsim_root)
    out=Path(a.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    conf=out/'hydra'; (conf/'agent').mkdir(parents=True)
    agent=dict(_target_='navsim_rft.agent.NavsimRFTAgent',checkpoint=str(Path(a.checkpoint).resolve()),
               navsim_root=str(Path(a.navsim_root).resolve()),device='cuda',seed=42,vlm_path=a.vlm)
    # JSON is a valid YAML subset; Hydra loads this .yaml file.
    (conf/'agent/navsim_rft.yaml').write_text(json.dumps(agent,indent=2))
    overrides=[]
    loader=build_loader(a.navsim_root,a.logs,a.sensors,a.split)
    expected=set(loader.tokens)
    if a.manifest:
        manifest=json.loads(Path(a.manifest).read_text())
        if manifest['role']=='train' or manifest['split']!=a.split: raise ValueError('Invalid evaluation manifest')
        tokens=[r['token'] for r in manifest['records']]
        if not set(tokens).issubset(expected):raise ValueError('Evaluation manifest tokens absent from logs')
        expected=set(tokens)
        ck=__import__('torch').load(a.checkpoint,map_location='cpu',weights_only=False)
        if set(tokens)&set(ck['metadata']['stats']['tokens']): raise ValueError('Train/eval overlap')
        overrides.append('train_test_split.scene_filter.tokens='+json.dumps(tokens,separators=(',',':')))
    elif a.split=='navtrain': raise ValueError('navtrain evaluation requires held-out validation manifest')
    ck=__import__('torch').load(a.checkpoint,map_location='cpu',weights_only=False)
    if expected & set(ck['metadata']['stats']['tokens']):raise ValueError('Train/evaluation overlap')
    from navsim.common.dataloader import MetricCacheLoader
    caches=MetricCacheLoader(Path(a.metric_cache))
    if not expected or not expected.issubset(set(caches.tokens)):
        raise ValueError('Missing evaluation metric cache; refusing silent subset scoring')
    command=[sys.executable,str(Path(a.navsim_root)/'navsim/planning/script/run_pdm_score.py'),
        '--config-dir',str(conf),'agent=navsim_rft',f'train_test_split={a.split}','worker=sequential',
        f'navsim_log_path={Path(a.logs).resolve()}',f'sensor_blobs_path={Path(a.sensors).resolve()}',
        f'metric_cache_path={Path(a.metric_cache).resolve()}',f'output_dir={out}',
        'experiment_name=unofficial_vla_rft_navsim_v1_1']+overrides
    (out/'command.json').write_text(json.dumps(command,indent=2))
    subprocess.run(command,check=True)
    # Official evaluator saves scene rows and an aggregate row to timestamped CSV.
    import pandas as pd
    csvs=list(out.glob('*.csv'))
    if not csvs: raise RuntimeError('Official evaluator did not produce a result CSV')
    frame=pd.read_csv(max(csvs,key=lambda p:p.stat().st_mtime))
    scene_rows=frame[frame['token']!='average']
    if set(scene_rows['token'])!=expected or not scene_rows['valid'].all():
        raise RuntimeError('Official evaluator has missing/failed scenes; see CSV, not a successful result')
    (out/'evaluation_status.json').write_text(json.dumps(dict(csv=[str(x) for x in csvs],
        rows=len(frame),official_evaluator=True,benchmark='NAVSIM v1.1 PDMS'),indent=2))

if __name__=='__main__': main()
