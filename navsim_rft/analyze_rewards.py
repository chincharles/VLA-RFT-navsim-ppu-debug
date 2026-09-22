"""Training-scene diagnostic only: image reward versus official PDMS, no oracle inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .data import SceneDataset
from .encoder import PrismaticEncoder
from .policy import FlowPolicy
from .world import TokenWorld
from .geometry import integrate
from .rewards import ImageReward,DrivingReward
from .checkpoint import load


def main():
    p=argparse.ArgumentParser()
    for k in ('policy','world','manifest','stats','metric-cache','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--limit',type=int,default=4);p.add_argument('--candidates',type=int,default=4)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    ds=SceneDataset(a.manifest,a.stats,training=True)
    ck=torch.load(a.policy,map_location='cpu',weights_only=False);cfg=ck['metadata']['config'];torch.manual_seed(42)
    wcfg=torch.load(a.world,map_location='cpu',weights_only=False)['metadata']['config']
    encoder=PrismaticEncoder(cfg['vlm']).cuda();encoder.set_trainable(False)
    policy=FlowPolicy(context_dim=encoder.dim,mode=cfg['math_mode'],**cfg.get('policy_arch',{})).cuda()
    load(a.policy,dict(policy=policy,encoder=encoder));policy.eval()
    wm=TokenWorld(wcfg['tokenizer'],ds.stats,wcfg.get('world_backbone'),**wcfg.get('world_arch',{})).cuda()
    load(a.world,dict(world=wm));wm.requires_grad_(False).eval()
    image_reward=ImageReward().cuda();drive=DrivingReward(a.metric_cache,[r['token'] for r in ds.records]);rows=[]
    with torch.no_grad():
        for i in range(min(a.limit,len(ds))):
            d=ds[i];n=a.candidates
            c=encoder([d['image']],[d['text']]).repeat_interleave(n,0);s=d['state'][None].cuda().repeat(n,1)
            chain=policy.sample(c,s);delta=ds.stats.denormalize(chain[:,-1].cpu().numpy());pose=integrate(delta)
            pred=torch.cat([wm.rollout(d['video'][None,0].cuda(),torch.tensor(x[None],device='cuda',dtype=torch.float32)) for x in delta])
            expert=wm.rollout(d['video'][None,0].cuda(),torch.tensor(d['deltas'][None],device='cuda',dtype=torch.float32))
            mask=d['valid'][None].cuda().repeat(n,1)
            ri,_=image_reward(pred,expert.repeat(n,1,1,1,1),mask)
            rl,_=image_reward(pred,d['video'][None,1:].cuda().repeat(n,1,1,1,1),mask)
            pdms=drive([d['token']]*n,pose)
            # Keep rollout diagnostics alongside rewards.  These values make a
            # degenerate frozen world model (identical frames or an empty mask)
            # visible before an RL run is trusted.
            expert_mean=expert.mean().item(); expert_std=expert.std().item()
            for j in range(n):
                mask_count=int(mask[j].sum().item())
                pred_mean=pred[j].mean().item(); pred_std=pred[j].std().item()
                rows.append(dict(token=d['token'],candidate=j,image_reward=float(ri[j]),
                    log_image_reward=float(rl[j]),pdms=float(pdms[j]),
                    mask_count=mask_count,pred_mean=float(pred_mean),pred_std=float(pred_std),
                    expert_mean=float(expert_mean),expert_std=float(expert_std),
                    trajectory=pose[j].tolist()))
    from scipy.stats import spearmanr,pearsonr
    x=[r['image_reward'] for r in rows]; y=[r['pdms'] for r in rows]
    valid=len(rows)>2 and np.std(x)>0 and np.std(y)>0
    def stats(values):
        values=[float(v) for v in values]
        return dict(min=min(values),max=max(values),mean=float(np.mean(values)),
            std=float(np.std(values)),unique=len(set(values))) if values else None
    summary=dict(count=len(rows),pearson=float(pearsonr(x,y)[0]) if valid else None,
        spearman=float(spearmanr(x,y)[0]) if valid else None,role='train_diagnostic_only',
        reward_stats={'image_reward':stats(x),'log_image_reward':stats([r['log_image_reward'] for r in rows]),
            'pdms':stats(y),'mask_count':stats([r['mask_count'] for r in rows]),
            'pred_mean':stats([r['pred_mean'] for r in rows]),'pred_std':stats([r['pred_std'] for r in rows]),
            'expert_mean':stats([r['expert_mean'] for r in rows]),'expert_std':stats([r['expert_std'] for r in rows])})
    # Rank disagreement is diagnostic; never feed selected candidates to evaluation.
    for r in rows:r['rank_gap']=float(abs(np.mean(np.array(x)<=r['image_reward'])-np.mean(np.array(y)<=r['pdms'])))
    summary['largest_disagreements']=sorted(rows,key=lambda r:r['rank_gap'],reverse=True)[:10]
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'candidates.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
