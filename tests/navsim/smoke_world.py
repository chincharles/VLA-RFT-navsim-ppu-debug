"""CPU-executable source-tokenizer/Llama smoke. Synthetic inputs, NOT driving validation."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ivideogpt.tokenizer import CompressiveVQModelFSQ
from navsim_rft.geometry import ActionStats
from navsim_rft.world import TokenWorld
from navsim_rft.policy import FlowPolicy,group_advantages,objective
from navsim_rft.checkpoint import fingerprint,save,load


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);torch.manual_seed(42);torch.set_num_threads(2)
    start=time.perf_counter()
    tokenizer=CompressiveVQModelFSQ(down_block_types=('DownEncoderBlock2D',)*4,
        up_block_types=('UpDecoderBlock2D',)*4,block_out_channels=(32,32,32,32),layers_per_block=1,
        latent_channels=4,sample_size=256,resolution=256,patch_size=4,vq_fsq_levels=12,dyn_fsq_levels=12)
    video=torch.rand(1,9,3,256,256)
    # Exercise original tokenizer reconstruction training before freezing it.
    result=tokenizer(video[:,0],dyn_sample=video[:,1],segment_len=1,return_loss=True)
    tokenizer_loss=(result.sample-video[:,1]).abs().mean()+(result.ref_sample-video[:,0]).abs().mean()
    tokenizer_loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in tokenizer.parameters())
    tokenizer.zero_grad();tokenizer.save_pretrained(out/'tokenizer')
    poses=np.stack([np.arange(1,9),np.zeros(8),np.zeros(8)],-1)[None]
    stats=ActionStats.fit(poses,['SYNTHETIC-NOT-NAVSIM'],'train');stats.save(out/'stats.json')
    world=TokenWorld(out/'tokenizer',stats,hidden=32,layers=1,heads=4)
    token_hash=fingerprint(world.tokenizer);opt=torch.optim.AdamW(world.backbone.parameters(),lr=1e-3)
    loss=world.loss(video,torch.tensor(stats.denormalize(np.zeros((1,8,3))),dtype=torch.float32),torch.ones(1,8,dtype=torch.bool))
    loss.backward();assert all(torch.isfinite(p.grad).all() for p in world.backbone.parameters() if p.grad is not None)
    opt.step();assert fingerprint(world.tokenizer)==token_hash
    save(out/'world-step-1.pt',{'world':world},opt,1,{'synthetic':True})
    world.eval().requires_grad_(False);world_hash=fingerprint(world)
    policy=FlowPolicy(context_dim=16,hidden=32,depth=2,heads=4,steps=3);po=torch.optim.AdamW(policy.parameters(),lr=1e-3)
    context=torch.randn(1,1,5,16);state=torch.randn(1,8);target=torch.randn(1,8,3)
    for _ in range(2):
        po.zero_grad();policy.supervised(context,state,target).backward();po.step()
    c=context.repeat(2,1,1,1);s=state.repeat(2,1);chain=policy.sample(c,s)
    old=policy.log_prob(c,s,chain)[0].detach();delta=torch.tensor(stats.denormalize(chain[:,-1].numpy()),dtype=torch.float32)
    pred=torch.cat([world.rollout(video[:,0],d[None]) for d in delta])
    # Local smoke only: L1 diagnostic. Full server A requires pretrained LPIPS and never falls back to L1.
    reward=-(pred-video[:,1:].repeat(2,1,1,1,1)).abs().mean((-1,-2,-3)).sum(1).reshape(1,2)
    adv=group_advantages(reward);new,en=policy.log_prob(c,s,chain)
    po.zero_grad();rl=objective(new,old,adv,en)+.1*policy.supervised(context,state,target);rl.backward();po.step()
    assert fingerprint(world)==world_hash
    save(out/'policy-step-3.pt',{'policy':policy},po,3,{'synthetic':True})
    frames=[]
    for t in range(8):
        panel=torch.cat([video[0,t+1],pred[0,t],pred[1,t]],-1)
        frames.append(Image.fromarray((panel.permute(1,2,0).numpy()*255).clip(0,255).astype(np.uint8)))
    frames[0].save(out/'synthetic-comparison.gif',save_all=True,append_images=frames[1:],duration=500,loop=0)
    summary=dict(synthetic=True,real_navsim_data=False,pretrained_weights=False,lpips_executed=False,
        tokenizer_backward=True,world_optimizer_steps=1,policy_sft_steps=2,policy_rl_steps=1,
        candidates=2,physical_frames=8,flow_steps=3,tokenizer_loss=float(tokenizer_loss.detach()),
        world_loss=float(loss.detach()),rl_loss=float(rl.detach()),rewards=reward.tolist(),
        elapsed_seconds=time.perf_counter()-start,frozen_tokenizer_and_world='unchanged',seed=42,
        torch=torch.__version__,device='CPU')
    (out/'result.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
