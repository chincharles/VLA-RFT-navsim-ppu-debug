#!/usr/bin/env python3
"""Generate reviewable server configs from explicit paths, never start training."""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser()
for k in ('navsim-root','train-manifest','stats','vlm','tokenizer','metric-cache','output'):
    p.add_argument('--'+k,required=True)
p.add_argument('--world-backbone')
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
base=dict(navsim_root=str(Path(a.navsim_root).resolve()),manifest=str(Path(a.train_manifest).resolve()),
    stats=str(Path(a.stats).resolve()),vlm=str(Path(a.vlm).resolve()),tokenizer=str(Path(a.tokenizer).resolve()),
    metric_cache=str(Path(a.metric_cache).resolve()),math_mode='paper',batch_size=1,lr=1e-4,
    policy_arch=dict(hidden=512,depth=8,heads=8,steps=10),world_arch=dict(hidden=768,layers=12,heads=12),
    world_backbone=a.world_backbone,train_encoder=True,candidates=4,reference='expert_rollout',
    reward_aggregate='sum',mse_coef=.1,entropy_coef=.001)
for stage in ('wm','sft','rl_img','rl_drive'):
    c=dict(base,stage=stage)
    if stage.startswith('rl'):c['lr']=1e-6;c['train_encoder']=False
    (out/f'{stage}.json').write_text(json.dumps(c,indent=2))
(out/'sft_source_math.json').write_text(json.dumps(dict(base,stage='sft',math_mode='source'),indent=2))
# Explicit paper-reference and released-code ablations; A default preserves source reference choice.
for name,change in [('rl_img_log_future',dict(reference='log_future')),
                    ('rl_img_source_math',dict(math_mode='source',reward_aggregate='mean'))]:
    c=dict(base,stage='rl_img',lr=1e-6,train_encoder=False,**change)
    (out/f'{name}.json').write_text(json.dumps(c,indent=2))
print(out)
