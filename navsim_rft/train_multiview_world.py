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

def main():
    p=argparse.ArgumentParser()
    for k in ('manifest','stats','tokenizer','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--save-every',type=int,default=100);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.cuda.set_device(device)
    if world>1: dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(seconds=7200))
    torch.manual_seed(a.seed+rank);np.random.seed(a.seed+rank);random.seed(a.seed+rank)
    ds=MultiViewSceneDataset(a.manifest,a.stats,training=True)
    world_model=TokenWorld(a.tokenizer,ds.stats).to(device); opt=torch.optim.AdamW(world_model.parameters(),lr=1e-4)
    out=Path(a.output)
    if rank==0: out.mkdir(parents=True,exist_ok=False)
    if world>1: dist.barrier()
    for step in range(a.steps):
        item=ds[(step*world+rank)%len(ds)]; views=item['views'].to(device); losses=[]
        # Shared world model; each view supplies an independent temporal stream.
        for cam in range(views.shape[0]):
            deltas=torch.tensor(item['deltas'][None],dtype=torch.float32,device=device)
            losses.append(world_model.loss(views[cam:cam+1],deltas,item['valid'][None].to(device)))
        loss=torch.stack(losses).mean(); grad=finite_step(loss,opt,{'world':world_model},world>1)
        if rank==0:
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,loss=float(loss),views=4,**grad))+'\n')
        if a.save_every and (step+1)%a.save_every==0 and step+1<a.steps: save_distributed(out/f'step-{step+1:06d}.pt',{'world':world_model},opt,step+1,dict(stage='multiview_world',tokenizer=a.tokenizer,views=list(ds.EXPECTED_VIEWS),steps=a.steps))
    save_distributed(out/f'step-{a.steps:06d}.pt',{'world':world_model},opt,a.steps,dict(stage='multiview_world',tokenizer=a.tokenizer,views=list(ds.EXPECTED_VIEWS),steps=a.steps))
    if world>1: dist.barrier();dist.destroy_process_group()
if __name__=='__main__': main()
