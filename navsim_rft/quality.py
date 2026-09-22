"""Offline validation: single-step teacher-forced vs free rollout; no training update."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .data import SceneDataset
from .world import TokenWorld
from .checkpoint import load
from .rewards import ImageReward


def main():
    p=argparse.ArgumentParser()
    for k in ('checkpoint','manifest','stats','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--limit',type=int,default=4);p.add_argument('--tokenizer')
    a=p.parse_args(); out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); cfg=ck['metadata']['config']
    ds=SceneDataset(a.manifest,a.stats)
    if ds.manifest['role']=='test': raise ValueError('Use held-out navtrain validation for model diagnostics/tuning')
    if ck['metadata']['stats']!=ds.stats.record: raise ValueError('Statistics mismatch')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    wm=TokenWorld(a.tokenizer or cfg['tokenizer'],ds.stats,cfg.get('world_backbone'),**cfg.get('world_arch',{})).to(device)
    load(a.checkpoint,dict(world=wm));wm.eval().requires_grad_(False)
    reward=ImageReward().to(device); rows=[]
    for i in range(min(len(ds),a.limit)):
        item=ds[i];v=item['video'][None].to(device);d=torch.tensor(item['deltas'],device=device,dtype=torch.float32)[None]
        free=wm.rollout(v[:,0],d);single=wm.rollout(v[:,0],d,teacher_video=v)
        _,f=reward(free,v[:,1:],item['valid'][None].to(device));_,s=reward(single,v[:,1:],item['valid'][None].to(device))
        mse=(free-v[:,1:]).square().mean((-1,-2,-3))
        rows.append(dict(token=item['token'],free_l1=f['l1'].cpu().tolist(),free_lpips=f['lpips'].cpu().tolist(),
            teacher_l1=s['l1'].cpu().tolist(),teacher_lpips=s['lpips'].cpu().tolist(),
            free_psnr=(-10*torch.log10(mse.clamp_min(1e-10))).cpu().tolist()))
        frames=[]
        for t in range(8):
            panel=torch.cat([v[0,t+1],single[0,t],free[0,t]],dim=-1)
            frames.append(Image.fromarray((panel.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype(np.uint8)))
        frames[0].save(out/f'{item["token"]}.gif',save_all=True,append_images=frames[1:],duration=500,loop=0)
    (out/'quality.json').write_text(json.dumps(dict(columns=['truth','teacher_forced_single_step','free_rollout'],rows=rows),indent=2))
if __name__=='__main__':main()
