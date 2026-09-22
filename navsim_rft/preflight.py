"""Fail early on missing resources; never substitute synthetic data for training."""
import argparse
import os
import importlib
import json
from pathlib import Path
import platform
import torch
from .data import verify_navsim,SceneDataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());errors=[];report={}
    for name in ('torch','transformers','diffusers','timm','navsim','nuplan','hydra','pyquaternion'):
        try:
            m=importlib.import_module(name);report[name]=dict(version=getattr(m,'__version__','unknown'),path=getattr(m,'__file__',None))
        except Exception as e: errors.append(f'{name}: {e}')
    try:verify_navsim(c['navsim_root'])
    except Exception as e:errors.append(str(e))
    for name in ('vlm','tokenizer'):
        if name not in c:continue
        root=Path(c[name]);weights=list(root.glob('*.safetensors'))+list(root.glob('*.bin'))
        if not (root/'config.json').is_file() or not weights:errors.append(f'Missing {name} config/weights: {root}')
        for index in root.glob('*.index.json'):
            shard_names=set(json.loads(index.read_text()).get('weight_map',{}).values())
            missing=[x for x in shard_names if not (root/x).is_file()]
            if missing:errors.append(f'Incomplete {name} checkpoint: {missing}')
        report[name+'_weight_files']=[dict(name=x.name,bytes=x.stat().st_size) for x in weights]
    try:
        ds=SceneDataset(c['manifest'],c['stats'],training=True)
        report['scene_count']=len(ds);report['first_token']=ds[0]['token']
    except Exception as e:errors.append(f'Dataset: {e}')
    if c['stage']=='rl_img':
        for key in ('VLA_RFT_LPIPS_PATH','VLA_RFT_VGG16_PATH'):
            if not Path(os.environ.get(key,'__MISSING__')).is_file():errors.append(f'Missing pretrained reward weights: {key}')
    report.update(platform=platform.platform(),cuda_available=torch.cuda.is_available(),
        devices=[dict(name=torch.cuda.get_device_name(i),memory=torch.cuda.get_device_properties(i).total_memory)
                 for i in range(torch.cuda.device_count())],errors=errors)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if errors:raise SystemExit(2)
if __name__=='__main__':main()
