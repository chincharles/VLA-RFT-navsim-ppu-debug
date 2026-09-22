"""Real CUDA/NCCL startup check on every requested GPU, before model training."""
import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--expected',type=int,default=16)
    a=p.parse_args();rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if not torch.cuda.is_available() or torch.cuda.device_count()<=local:raise RuntimeError('Insufficient assigned CUDA devices')
    torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    if world!=a.expected:raise ValueError(f'Expected {a.expected} workers, got {world}')
    x=torch.randn(256,256,device=f'cuda:{local}');y=x@x
    if not torch.isfinite(y).all():raise FloatingPointError('CUDA matrix multiplication')
    value=torch.tensor([rank+1.],device=x.device)
    dist.all_reduce(value)
    if float(value)!=world*(world+1)/2:raise RuntimeError('NCCL all-reduce verification failed')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    row=dict(rank=rank,local_rank=local,world_size=world,gpu=torch.cuda.get_device_name(local),
             memory_bytes=torch.cuda.get_device_properties(local).total_memory,
             capability=torch.cuda.get_device_capability(local),torch=torch.__version__,cuda=torch.version.cuda,
             all_reduce=float(value),passed=True)
    (out/f'rank-{rank}.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
