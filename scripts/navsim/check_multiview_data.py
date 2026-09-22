#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--limit',type=int,default=8)
    a=p.parse_args();m=json.loads(Path(a.manifest).read_text());views=m.get('views')
    expected=['cam_f0','cam_b0','cam_l0','cam_r0']
    if views!=expected: raise ValueError(f'Unexpected view order: {views}')
    rows=[]
    for r in m['records'][:a.limit]:
        with np.load(Path(a.manifest).parent/r['file'],allow_pickle=False) as z:
            x=z['views']
            if x.ndim!=5 or x.shape[0]!=4 or x.shape[2:]!=(256,256,3): raise ValueError(f'{r["token"]}: {x.shape}')
            rows.append(dict(token=r['token'],shape=list(x.shape),mins=x.min(axis=(1,2,3,4)).tolist(),maxs=x.max(axis=(1,2,3,4)).tolist()))
    print(json.dumps(dict(views=views,checked=len(rows),records=rows),indent=2))
if __name__=='__main__': main()
