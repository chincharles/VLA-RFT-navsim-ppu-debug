"""Optional reimplementation of missing tokenizer pretraining, NOT released original recipe."""
import argparse
import json
from pathlib import Path
import torch
from .data import SceneDataset
from .checkpoint import save,load

def main():
    p=argparse.ArgumentParser()
    for k in ('manifest','stats','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--steps',type=int,default=2);p.add_argument('--resume');p.add_argument('--init');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();torch.manual_seed(a.seed)
    from ivideogpt.tokenizer import CompressiveVQModelFSQ
    from .rewards import ImageReward
    out=Path(a.output);out.mkdir(parents=True,exist_ok=bool(a.resume));ds=SceneDataset(a.manifest,a.stats,training=True)
    model=CompressiveVQModelFSQ.from_pretrained(a.init,local_files_only=True) if a.init else CompressiveVQModelFSQ(
        down_block_types=('DownEncoderBlock2D',)*4,up_block_types=('UpDecoderBlock2D',)*4,
        block_out_channels=(128,256,256,512),layers_per_block=2,latent_channels=4,sample_size=256,
        resolution=256,patch_size=4,vq_fsq_levels=12,dyn_fsq_levels=12)
    device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    # Source LPIPS is frozen but gradients through predicted pixels are required here.
    percept=ImageReward().metric.to(device).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4);start=0
    meta=dict(seed=a.seed,stage='tokenizer_reimplementation',stats=ds.stats.record)
    if a.resume:
        ck=load(a.resume,dict(tokenizer=model),opt)
        if ck['metadata']!=meta:raise ValueError('Resume mismatch')
        start=ck['step']
    for step in range(start,start+a.steps):
        item=ds[step%len(ds)];video=item['video'].to(device)
        # One future frame per step limits reconstruction memory; sample horizon uniformly.
        idx=int(torch.randint(1,9,()).item());target=video[idx:idx+1];initial=video[:1]
        dec=model(initial,dyn_sample=target,segment_len=1,return_loss=True)
        loss=(dec.sample-target).abs().mean()+(dec.ref_sample-initial).abs().mean()
        loss=loss+percept(dec.sample*2-1,target*2-1).mean()+percept(dec.ref_sample*2-1,initial*2-1).mean()
        opt.zero_grad();loss.backward()
        if not torch.isfinite(loss):raise FloatingPointError('Tokenizer loss')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(step=step+1,loss=float(loss.detach()),token=item['token']))+'\n')
    save(out/f'step-{start+a.steps:06d}.pt',dict(tokenizer=model),opt,start+a.steps,meta)
    model.save_pretrained(out/f'pretrained-{start+a.steps:06d}')
if __name__=='__main__':main()
