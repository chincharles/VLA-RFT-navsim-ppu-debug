"""Inspect tokenizer reconstruction on held-out NAVSIM images, without a world model."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import SceneDataset


def rgb(tensor):
    pixels=(tensor.detach().float().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    return Image.fromarray(pixels)


def scores(pred, truth):
    delta=pred.float()-truth.float()
    mse=float(delta.square().mean())
    return dict(l1=float(delta.abs().mean()), psnr=float(-10*np.log10(max(mse,1e-10))))


def main():
    parser=argparse.ArgumentParser()
    for name in ('tokenizer','manifest','stats','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--limit',type=int,default=8)
    parser.add_argument('--frame',type=int,default=1,help='Future frame index, 1 to 8')
    args=parser.parse_args()
    if args.limit<1 or not 1<=args.frame<=8: raise ValueError('Require positive limit and frame in 1..8')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    dataset=SceneDataset(args.manifest,args.stats)
    if dataset.manifest['role']=='test': raise ValueError('Use held-out navtrain validation, not test')
    from ivideogpt.tokenizer import CompressiveVQModelFSQ
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=CompressiveVQModelFSQ.from_pretrained(args.tokenizer,local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    rows=[]
    with torch.inference_mode():
        for index in range(min(args.limit,len(dataset))):
            item=dataset[index]
            start=item['video'][0:1].to(device)
            future=item['video'][args.frame:args.frame+1].to(device)
            direct=model(start,dyn_sample=future,segment_len=1,return_loss=True)
            tokens_c,tokens_d=model.tokenize(torch.stack([start[0],future[0]])[None])
            decoded=model.detokenize(tokens_c,tokens_d)
            start_direct=direct.ref_sample.clamp(0,1)
            future_direct=direct.sample.clamp(0,1)
            start_tokens=decoded[:,0].clamp(0,1)
            future_tokens=decoded[:,1].clamp(0,1)
            panel=Image.new('RGB',(future.shape[-1]*4,future.shape[-2]*2))
            for col,frame in enumerate((start[0],start_direct[0],start_tokens[0],future[0])):
                panel.paste(rgb(frame),(col*future.shape[-1],0))
            for col,frame in enumerate((future[0],future_direct[0],future_tokens[0],start[0])):
                panel.paste(rgb(frame),(col*future.shape[-1],future.shape[-2]))
            panel.save(out/f"{item['token']}.png")
            rows.append(dict(token=item['token'],frame=args.frame,
                context_direct=scores(start_direct,start),context_tokens=scores(start_tokens,start),
                future_direct=scores(future_direct,future),future_tokens=scores(future_tokens,future),
                direct_vs_tokens=scores(future_direct,future_tokens)))
    summary={key:{metric:float(np.mean([row[key][metric] for row in rows])) for metric in ('l1','psnr')}
             for key in ('context_direct','context_tokens','future_direct','future_tokens','direct_vs_tokens')}
    report=dict(columns_top=['start_truth','start_direct','start_token_roundtrip','future_truth'],
        columns_bottom=['future_truth','future_direct','future_token_roundtrip','start_truth'],
        tokenizer=str(Path(args.tokenizer).resolve()),count=len(rows),summary=summary,rows=rows)
    (out/'reconstruction.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(count=len(rows),summary=summary,output=str(out)),indent=2),flush=True)


if __name__=='__main__': main()
