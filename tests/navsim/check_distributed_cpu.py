"""Two-process CPU check of the real reducer/checkpoint helpers, not NAVSIM scores."""
import argparse
import copy
import json
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from navsim_rft.train import finite_step
from navsim_rft.checkpoint import fingerprint,save_distributed,load


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layer=torch.nn.Linear(3,2);self.unused=torch.nn.Parameter(torch.ones(2))
    def forward(self,x):return self.layer(x)


def worker(rank,root):
    root=Path(root);torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+str(root/'rendezvous'),rank=rank,world_size=2)
    torch.manual_seed(12);model=Tiny();reference=copy.deepcopy(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    ref_opt=torch.optim.AdamW(reference.parameters(),lr=.01)
    x=torch.arange(12,dtype=torch.float32).reshape(4,3)/10
    target=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[0.,0.]])
    loss=(model(x[rank*2:rank*2+2])-target[rank*2:rank*2+2]).square().mean()
    finite_step(loss,opt,dict(model=model),True)
    finite_step((reference(x)-target).square().mean(),ref_opt,dict(model=reference),False)
    for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
    torch.testing.assert_close(model.unused,torch.ones(2),rtol=0,atol=0)
    hashes=[None,None];dist.all_gather_object(hashes,fingerprint(model));assert len(set(hashes))==1
    random.seed(90+rank);np.random.seed(90+rank);torch.manual_seed(90+rank)
    checkpoint=root/'step-000001.pt'
    save_distributed(checkpoint,dict(model=model),opt,1,dict(world_size=2,synthetic=True))
    expected=(random.random(),float(np.random.rand()),torch.randn(2,3))
    restored=Tiny();restored_opt=torch.optim.AdamW(restored.parameters(),lr=.01)
    state=load(checkpoint,dict(model=restored),restored_opt,rank)
    actual=(random.random(),float(np.random.rand()),torch.randn(2,3))
    assert actual[:2]==expected[:2];torch.testing.assert_close(actual[2],expected[2],rtol=0,atol=0)
    assert state['rng'][rank]['cuda_scope']=='current'
    for current,optimizer in ((model,opt),(restored,restored_opt)):
        finite_step(current(actual[2]).square().mean(),optimizer,dict(model=current),True)
    assert fingerprint(model)==fingerprint(restored)
    save_distributed(root/'step-000002.pt',dict(model=restored),restored_opt,2,dict(world_size=2,synthetic=True))
    (root/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,gradient_matches_global_batch=True,
        unused_parameter_unchanged=True,rank_rng_restored=True,optimizer_resume_exact=True)))
    dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory() as root:
        mp.spawn(worker,args=(root,),nprocs=2,join=True)
        rows=[json.loads((Path(root)/f'rank-{i}.json').read_text()) for i in range(2)]
    report=dict(backend='gloo',processes=2,real_navsim_data=False,cuda=False,rows=rows)
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
