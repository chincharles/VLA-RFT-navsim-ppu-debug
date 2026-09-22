"""Small original-model PPU kernel/gradient/RNG probe; synthetic, not driving validation."""
import argparse
import json
import os
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from .checkpoint import fingerprint,save_distributed,load
from .policy import FlowPolicy,group_advantages,objective
from .train import finite_step

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    device=torch.device(f'cuda:{local}');torch.cuda.set_device(device)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    torch.manual_seed(42);out=Path(a.output)
    if rank==0:out.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    from ivideogpt.tokenizer import CompressiveVQModelFSQ
    from .geometry import ActionStats
    from .world import TokenWorld
    import numpy as np
    tokenizer=CompressiveVQModelFSQ(down_block_types=('DownEncoderBlock2D',)*4,
        up_block_types=('UpDecoderBlock2D',)*4,block_out_channels=(32,32,32,32),layers_per_block=1,
        latent_channels=4,sample_size=256,resolution=256,patch_size=4,vq_fsq_levels=12,dyn_fsq_levels=12).to(device)
    images=torch.rand(1,9,3,256,256,device=device)
    opt=torch.optim.AdamW(tokenizer.parameters(),lr=1e-4)
    result=tokenizer(images[:,0],dyn_sample=images[:,1],segment_len=1,return_loss=True)
    rec=(result.sample-images[:,1]).abs().mean()+(result.ref_sample-images[:,0]).abs().mean()
    finite_step(rec,opt,dict(tokenizer=tokenizer),True)
    if rank==0:tokenizer.save_pretrained(out/'synthetic-tokenizer')
    dist.barrier()
    del result,rec,opt,tokenizer
    poses=np.stack([np.arange(1,9),np.zeros(8),np.zeros(8)],-1)[None]
    stats=ActionStats.fit(poses,['SYNTHETIC-NOT-NAVSIM'],'train')
    wm=TokenWorld(out/'synthetic-tokenizer',stats,hidden=32,layers=1,heads=4).to(device)
    before=fingerprint(wm.tokenizer);opt=torch.optim.AdamW(wm.backbone.parameters(),lr=1e-4)
    delta=torch.tensor(stats.denormalize(np.zeros((1,8,3))),dtype=torch.float32,device=device)
    loss=wm.loss(images,delta,torch.ones(1,8,dtype=torch.bool,device=device))
    finite_step(loss,opt,dict(world=wm),True)
    assert fingerprint(wm.tokenizer)==before
    predicted=wm.rollout(images[:,0],delta[:,:1])
    if not torch.isfinite(predicted).all():raise FloatingPointError('Nonfinite original FSQ/Llama rollout')
    del wm,opt,loss,predicted,images
    policy=FlowPolicy(context_dim=16,hidden=32,depth=2,heads=4,steps=3).to(device)
    opt=torch.optim.AdamW(policy.parameters(),lr=1e-4)
    context=torch.randn(2,1,5,16,device=device);state=torch.randn(2,8,device=device)
    target=torch.randn(2,8,3,device=device)
    finite_step(policy.supervised(context,state,target),opt,dict(policy=policy),True)
    chain=policy.sample(context,state);old=policy.log_prob(context,state,chain)[0].detach()
    lp,entropy=policy.log_prob(context,state,chain)
    torch.testing.assert_close(lp.detach(),old,rtol=0,atol=1e-5)
    rewards=-chain[:,-1].square().mean((1,2)).reshape(1,2)
    loss=objective(lp,old,group_advantages(rewards),entropy)
    finite_step(loss,opt,dict(policy=policy),True)
    save_distributed(out/'synthetic-policy.pt',dict(policy=policy),opt,2,dict(synthetic=True))
    expected=torch.randn(8,device=device)
    load(out/'synthetic-policy.pt',dict(policy=policy),opt,rank)
    torch.testing.assert_close(expected,torch.randn(8,device=device),rtol=0,atol=0)
    row=dict(rank=rank,device=torch.cuda.get_device_name(local),synthetic=True,real_navsim_data=False,
        tokenizer_backward=True,world_mle_backward=True,world_one_frame_rollout=True,
        flow_backward=True,policy_probability_replay=True,rl_optimizer_step=True,
        nccl_gradient_sync=True,checkpoint_rng_restore=True,lpips_executed=False,
        pretrained_vlm_executed=False,passed=True)
    (out/f'rank-{rank}.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
