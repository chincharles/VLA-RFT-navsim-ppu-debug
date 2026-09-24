"""Train a shared action-conditioned world model over four camera streams."""
import argparse, json, os, random
from datetime import timedelta
from pathlib import Path
import numpy as np, torch
import torch.distributed as dist
from .data import MultiViewSceneDataset
from .world import TokenWorld
from .checkpoint import save_distributed
from .train import finite_step
from .multiview_resume import prepare

def main():
    p=argparse.ArgumentParser()
    for k in ('manifest','stats','tokenizer','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--save-every',type=int,default=100);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--seed',type=int,default=42)
    p.add_argument("--resume", help="Trusted project checkpoint; --steps is total target")
    a=p.parse_args();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.cuda.set_device(device)
    if world>1: dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(seconds=7200))
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    ds=MultiViewSceneDataset(a.manifest,a.stats,training=True)
    world_model=TokenWorld(a.tokenizer,ds.stats).to(device); opt=torch.optim.AdamW(world_model.parameters(),lr=1e-4)
    out=Path(a.output)
    start, metadata, metrics = prepare(a, {'world':world_model}, opt, ds, rank, world, 'multiview_world')
    if world>1: dist.barrier()
    for step in range(start,a.steps):
        losses=[]
        for b in range(a.batch_size):
            item=ds[(step*world*a.batch_size+rank*a.batch_size+b)%len(ds)]; views=item['views'].to(device)
            for cam in range(views.shape[0]):
                deltas=torch.tensor(item['deltas'][None],dtype=torch.float32,device=device)
                losses.append(world_model.loss(views[cam:cam+1],deltas,item['valid'][None].to(device),camera_id=cam))
        loss=torch.stack(losses).mean(); grad=finite_step(loss,opt,{'world':world_model},world>1)
        if rank==0:
            with metrics.open('a') as f:f.write(json.dumps(dict(step=step+1,loss=float(loss),views=4,**grad))+'\n')
        if a.save_every and (step+1)%a.save_every==0 and step+1<a.steps: save_distributed(out/f'step-{step+1:06d}.pt',{'world':world_model},opt,step+1,metadata)
    if start < a.steps: save_distributed(out/f'step-{a.steps:06d}.pt',{'world':world_model},opt,a.steps,metadata)
    if world>1: dist.barrier();dist.destroy_process_group()
if __name__=='__main__': main()
