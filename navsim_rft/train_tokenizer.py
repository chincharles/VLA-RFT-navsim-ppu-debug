"""Optional reimplementation of missing tokenizer pretraining, NOT released original recipe."""
import argparse
import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .data import SceneDataset
from .checkpoint import save_distributed,load
from .train import finite_step

def main():
    p=argparse.ArgumentParser()
    for k in ('manifest','stats','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--resume');p.add_argument('--init');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save-every',type=int,default=0)
    a=p.parse_args()
    if a.steps<1 or a.save_every<0:raise ValueError('Invalid step budget/save interval')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':torch.cuda.set_device(device)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',
        timeout=timedelta(seconds=int(os.environ.get('RFT_DIST_TIMEOUT_SECONDS','7200'))))
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    from ivideogpt.tokenizer import CompressiveVQModelFSQ
    from .rewards import ImageReward
    out=Path(a.output)
    if rank==0:out.mkdir(parents=True,exist_ok=bool(a.resume))
    if world>1:dist.barrier()
    ds=SceneDataset(a.manifest,a.stats,training=True)
    model=CompressiveVQModelFSQ.from_pretrained(a.init,local_files_only=True) if a.init else CompressiveVQModelFSQ(
        down_block_types=('DownEncoderBlock2D',)*4,up_block_types=('UpDecoderBlock2D',)*4,
        block_out_channels=(128,256,256,512),layers_per_block=2,latent_channels=4,sample_size=256,
        resolution=256,patch_size=4,vq_fsq_levels=12,dyn_fsq_levels=12)
    model.to(device)
    # Source LPIPS is frozen but gradients through predicted pixels are required here.
    percept=ImageReward().metric.to(device).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4);start=0
    meta=dict(seed=a.seed,stage='tokenizer_reimplementation',stats=ds.stats.record,world_size=world,
              manifest=str(Path(a.manifest).resolve()),init=a.init,tokenizer_config=dict(model.config))
    torch.manual_seed(a.seed+rank);np.random.seed(a.seed+rank);random.seed(a.seed+rank)
    if a.resume:
        ck=load(a.resume,dict(tokenizer=model),opt,rank)
        if ck['metadata']!=meta:raise ValueError('Resume mismatch')
        start=ck['step']
    for step in range(start,start+a.steps):
        tic=time.perf_counter()
        order=np.random.default_rng(a.seed+(step*world)//len(ds)).permutation(len(ds))
        item=ds[int(order[(step*world+rank)%len(ds)])];video=item['video'].to(device)
        # One future frame per step limits reconstruction memory; sample horizon uniformly.
        idx=int(torch.randint(1,9,()).item());target=video[idx:idx+1];initial=video[:1]
        dec=model(initial,dyn_sample=target,segment_len=1,return_loss=True)
        loss=(dec.sample-target).abs().mean()+(dec.ref_sample-initial).abs().mean()
        loss=loss+percept(dec.sample*2-1,target*2-1).mean()+percept(dec.ref_sample*2-1,initial*2-1).mean()
        gradients=finite_step(loss,opt,dict(tokenizer=model),world>1)
        if device.type=='cuda':torch.cuda.synchronize()
        row=dict(step=step+1,loss=float(loss.detach()),token=item['token'],rank=rank,seconds=time.perf_counter()-tic,
                 peak_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,**gradients)
        with (out/f'metrics.rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if rank==0:
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        if a.save_every and (step+1)%a.save_every==0 and step+1<start+a.steps:
            save_distributed(out/f'step-{step+1:06d}.pt',dict(tokenizer=model),opt,step+1,meta)
    save_distributed(out/f'step-{start+a.steps:06d}.pt',dict(tokenizer=model),opt,start+a.steps,meta)
    if rank==0:
        model.save_pretrained(out/f'pretrained-{start+a.steps:06d}')
        (out/f'hf-export-{start+a.steps:06d}.done.json').write_text(json.dumps(dict(step=start+a.steps)))
    if world>1:dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
