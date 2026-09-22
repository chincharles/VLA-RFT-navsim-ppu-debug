"""Train one shared image tokenizer on four camera streams."""
import argparse, json, os, random
from datetime import timedelta
from pathlib import Path
import numpy as np, torch
import torch.distributed as dist
from .data import MultiViewSceneDataset
from .checkpoint import save_distributed
from .train import finite_step

def main():
    p=argparse.ArgumentParser()
    for k in ('manifest','stats','output'): p.add_argument('--'+k,required=True)
    p.add_argument('--steps',type=int,required=True); p.add_argument('--save-every',type=int,default=500); p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--init',required=True); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.cuda.set_device(device)
    if world>1: dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(seconds=7200))
    torch.manual_seed(a.seed+rank); np.random.seed(a.seed+rank); random.seed(a.seed+rank)
    from ivideogpt.tokenizer import CompressiveVQModelFSQ
    from .rewards import ImageReward
    ds=MultiViewSceneDataset(a.manifest,a.stats,training=True)
    model=CompressiveVQModelFSQ.from_pretrained(a.init,local_files_only=True).to(device)
    percept=ImageReward().metric.to(device).eval(); opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    out=Path(a.output)
    if rank==0: out.mkdir(parents=True,exist_ok=False)
    if world>1: dist.barrier()
    for step in range(a.steps):
        losses=[]
        for b in range(a.batch_size):
            item=ds[(step*world*a.batch_size+rank*a.batch_size+b)%len(ds)]; views=item['views'].to(device); cam=int((step+b+rank)%views.shape[0])
            video=views[cam]; idx=int(torch.randint(1,video.shape[0],()).item()); initial=video[:1]; target=video[idx:idx+1]
            dec=model(initial,dyn_sample=target,segment_len=1,return_loss=True)
            l=(dec.sample-target).abs().mean()+(dec.ref_sample-initial).abs().mean()
            losses.append(l+percept(dec.sample*2-1,target*2-1).mean()+percept(dec.ref_sample*2-1,initial*2-1).mean())
        loss=torch.stack(losses).mean()
        grad=finite_step(loss,opt,{'tokenizer':model},world>1)
        if rank==0:
            with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(dict(step=step+1,loss=float(loss),camera=cam,**grad))+'\n')
        if a.save_every and (step+1)%a.save_every==0 and step+1<a.steps: save_distributed(out/f'step-{step+1:06d}.pt',{'tokenizer':model},opt,step+1,dict(stage='multiview_tokenizer',steps=a.steps,views=list(ds.EXPECTED_VIEWS)))
    save_distributed(out/f'step-{a.steps:06d}.pt',{'tokenizer':model},opt,a.steps,dict(stage='multiview_tokenizer',steps=a.steps,views=list(ds.EXPECTED_VIEWS)))
    if rank==0: model.save_pretrained(out/f'pretrained-{a.steps:06d}')
    if world>1: dist.barrier(); dist.destroy_process_group()
if __name__=='__main__': main()
