"""Bounded single GPU / torchrun data-parallel training, no implicit large job."""
import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from .data import SceneDataset,verify_navsim
from .geometry import integrate
from .policy import FlowPolicy,group_advantages,objective
from .checkpoint import fingerprint,save,load,rng_state


def batch(items,device):
    keys=('state','actions','video','valid')
    result={k:torch.stack([v[k] for v in items]).to(device) for k in keys}
    result.update(images=[v['image'] for v in items],texts=[v['text'] for v in items],tokens=[v['token'] for v in items],
                  deltas=torch.tensor(np.stack([v['deltas'] for v in items]),dtype=torch.float32,device=device))
    return result


def finite_step(loss,opt,modules,distributed):
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
    opt.zero_grad(set_to_none=True); loss.backward()
    diagnostics={}
    for name,module in modules.items():
        grads=[]
        for p in module.parameters():
            if p.requires_grad:
                if distributed:
                    if p.grad is None: p.grad=torch.zeros_like(p)
                    dist.all_reduce(p.grad); p.grad.div_(dist.get_world_size())
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all(): raise FloatingPointError(f'{name}: nonfinite gradient')
                    grads.append(float(p.grad.detach().float().square().sum()))
        diagnostics[name+'_grad_norm']=sum(grads)**.5
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],1.)
    opt.step()
    return diagnostics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--steps',type=int,default=2); ap.add_argument('--resume'); ap.add_argument('--init-policy')
    ap.add_argument('--wm-checkpoint'); ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args(); cfg=json.loads(Path(a.config).read_text())
    if a.steps<1: raise ValueError('Positive step bound required')
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1)); local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda': torch.cuda.set_device(device)
    if world>1: dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    # Same init across ranks; separate sampling streams after module construction.
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    verify_navsim(cfg['navsim_root'])
    dataset=SceneDataset(cfg['manifest'],cfg['stats'],training=True)
    stage=cfg['stage']; out=Path(a.output)
    if stage not in ('wm','sft','rl_img','rl_drive'):raise ValueError('Unknown training stage')
    if cfg.get('batch_size',1)<1:raise ValueError('Positive batch size required')
    if rank==0:
        if out.exists() and not a.resume: raise FileExistsError('Use a fresh output directory or --resume')
        out.mkdir(parents=True,exist_ok=True)
    if world>1: dist.barrier()
    from .world import TokenWorld
    modules={}; wm=None; encoder=None
    if stage=='wm' or stage=='rl_img':
        wm=TokenWorld(cfg['tokenizer'],dataset.stats,cfg.get('world_backbone'),**cfg.get('world_arch',{})).to(device)
        modules['world']=wm
    if stage!='wm':
        from .encoder import PrismaticEncoder
        encoder=PrismaticEncoder(cfg['vlm']).to(device)
        policy=FlowPolicy(context_dim=encoder.dim,mode=cfg['math_mode'],**cfg.get('policy_arch',{})).to(device)
        encoder.set_trainable(stage=='sft' and cfg.get('train_encoder',True))
        modules.update(policy=policy,encoder=encoder)
        if stage.startswith('rl') and not (a.init_policy or a.resume): raise ValueError('RL requires supervised initialization')
        if a.init_policy:
            init=load(a.init_policy,dict(policy=policy,encoder=encoder))
            if init['metadata']['config']['math_mode']!=cfg['math_mode']:
                raise ValueError('SFT/RL flow sign and time conventions differ; train matching SFT mode')
            if init['metadata']['stage']!='sft' or init['metadata']['stats']!=dataset.stats.record:
                raise ValueError('Require SFT checkpoint with identical action statistics')
    if stage=='rl_img':
        if not a.wm_checkpoint: raise ValueError('Driving-adapted WM checkpoint required')
        wmck=load(a.wm_checkpoint,dict(world=wm))
        if wmck['metadata']['stage']!='wm' or wmck['step']<1 or wmck['metadata']['stats']!=dataset.stats.record:
            raise ValueError('Require trained driving world model with same statistics')
        wm.eval().requires_grad_(False)
        from .rewards import ImageReward
        reward_fn=ImageReward(aggregate=cfg.get('reward_aggregate','sum')).to(device)
    elif stage=='rl_drive':
        from .rewards import DrivingReward
        reward_fn=DrivingReward(cfg['metric_cache'],[r['token'] for r in dataset.records])
    if stage!='wm':
        policy.sigma.requires_grad_(stage.startswith('rl'))
    train_modules={k:v for k,v in modules.items() if any(p.requires_grad for p in v.parameters())}
    opt=torch.optim.AdamW([p for m in train_modules.values() for p in m.parameters() if p.requires_grad],lr=cfg['lr'])
    metadata=dict(stage=stage,config=cfg,stats=dataset.stats.record,seed=a.seed,world_size=world,
                  data_role='train',navsim_commit=dataset.manifest['navsim_commit'])
    frozen={k:fingerprint(v) for k,v in modules.items() if not any(p.requires_grad for p in v.parameters())}
    tokenizer_before=fingerprint(wm.tokenizer) if wm else None
    sigma_before=fingerprint(policy.sigma) if stage=='sft' else None
    torch.manual_seed(a.seed+rank); random.seed(a.seed+rank); np.random.seed(a.seed+rank)
    start=0
    if a.resume:
        ck=load(a.resume,modules,opt,rank)
        if ck['metadata']!=metadata: raise ValueError('Resume metadata/config/seed/world size mismatch')
        start=ck['step']
        frozen={k:fingerprint(modules[k]) for k in frozen}
        tokenizer_before=fingerprint(wm.tokenizer) if wm else None
        sigma_before=fingerprint(policy.sigma) if stage=='sft' else None
    if rank==0: (out/'run.json').write_text(json.dumps(metadata,indent=2))
    began=time.perf_counter(); seen=set(); rows=[]
    for step in range(start,start+a.steps):
        tic=time.perf_counter()
        # Deterministic global permutation; rank shards one batch, resume driven by saved step.
        bs=cfg.get('batch_size',1)
        generator=np.random.default_rng(a.seed+(step*world*bs)//len(dataset))
        order=generator.permutation(len(dataset))
        indices=[int(order[(step*world*bs+rank*bs+j)%len(order)]) for j in range(bs)]
        b=batch([dataset[i] for i in indices],device); seen.update(b['tokens'])
        metrics={}
        if stage=='wm':
            loss=wm.loss(b['video'],b['deltas'],b['valid'])
        else:
            encoder.eval()
            with torch.set_grad_enabled(encoder.trainable):
                context=encoder(b['images'],b['texts'])
            if stage=='sft':
                loss=policy.supervised(context,b['state'],b['actions'])
            else:
                n=cfg.get('candidates',4)
                c=context.detach().repeat_interleave(n,0); state=b['state'].repeat_interleave(n,0)
                with torch.no_grad():
                    chain=policy.sample(c,state)
                    old_lp,_=policy.log_prob(c,state,chain)
                    # Snapshot log probabilities are old policy; exactly one update per on-policy batch.
                    delta=dataset.stats.denormalize(chain[:,-1].cpu().numpy())
                    pose=integrate(delta)
                    tokens=[t for t in b['tokens'] for _ in range(n)]
                    if stage=='rl_img':
                        d=torch.tensor(delta,dtype=torch.float32,device=device)
                        # Serial candidate microbatches bound KV-cache memory.
                        pred=torch.cat([wm.rollout(b['video'][:,0].repeat_interleave(n,0)[i:i+1],d[i:i+1]) for i in range(len(d))])
                        if cfg['reference']=='expert_rollout':
                            ref=wm.rollout(b['video'][:,0],b['deltas'])
                        elif cfg['reference']=='log_future': ref=b['video'][:,1:]
                        else: raise ValueError('Unknown reference')
                        reward,quality=reward_fn(pred,ref.repeat_interleave(n,0),b['valid'].repeat_interleave(n,0))
                        metrics['wm_action_out_of_range']=float(((d<wm.low)|(d>wm.high)).float().mean())
                    else: reward=reward_fn(tokens,pose).to(device)
                    reward=reward.reshape(bs,n)
                    adv=group_advantages(reward,cfg['math_mode'])
                lp,entropy=policy.log_prob(c,state,chain)
                replay_error=float((lp.detach()-old_lp).abs().max())
                if replay_error>1e-5: raise RuntimeError(f'Old-policy replay mismatch {replay_error}')
                loss=objective(lp,old_lp,adv,entropy,cfg['math_mode'],entropy_coef=cfg.get('entropy_coef',.001))
                aux=policy.supervised(context.detach(),b['state'],b['actions'])
                mse_coef=cfg.get('mse_coef',.1)
                if cfg['math_mode']=='source':
                    gate=(-(lp.detach()-old_lp).mean()/cfg.get('mse_kl_high',.2)).clamp(0,1)
                    mse_coef=mse_coef*gate
                loss=loss+mse_coef*aux
                diversity=float(chain[:,-1].reshape(bs,n,-1).std(1).mean())
                if not torch.isfinite(reward).all() or diversity<=0: raise RuntimeError('Invalid/diversity-free rollout')
                metrics.update(reward=reward.cpu().tolist(),advantage=adv.cpu().tolist(),diversity=diversity,
                               replay_max_error=replay_error,logp_min=float(lp.min().detach()),logp_max=float(lp.max().detach()))
        metrics.update(finite_step(loss,opt,train_modules,world>1))
        if device.type=='cuda': torch.cuda.synchronize()
        metrics.update(step=step+1,loss=float(loss.detach()),seconds=time.perf_counter()-tic,
                       rank=rank,tokens=b['tokens'],peak_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0)
        with (out/f'metrics.rank{rank}.jsonl').open('a') as f: f.write(json.dumps(metrics)+'\n')
        if rank==0: print(json.dumps(metrics),flush=True)
        rows.append(metrics)
    for k,h in frozen.items():
        if fingerprint(modules[k])!=h: raise RuntimeError(f'Frozen {k} changed')
    if stage=='sft' and fingerprint(policy.sigma)!=sigma_before:raise RuntimeError('Frozen SFT sigma changed')
    if wm and fingerprint(wm.tokenizer)!=tokenizer_before: raise RuntimeError('Frozen tokenizer changed')
    states=[None]*world
    if world>1: dist.all_gather_object(states,rng_state())
    else: states=[rng_state()]
    if rank==0:
        save(out/f'step-{start+a.steps:06d}.pt',modules,opt,start+a.steps,metadata,states)
        (out/f'summary-{start+a.steps:06d}.json').write_text(json.dumps(dict(steps_executed=a.steps,
            elapsed_seconds=time.perf_counter()-began,unique_scenes_rank0=len(seen),frozen_checks='passed',
            official_metrics=None,real_data=True),indent=2))
    if world>1: dist.barrier(); dist.destroy_process_group()

if __name__=='__main__': main()
