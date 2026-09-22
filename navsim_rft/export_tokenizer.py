"""Recover an interrupted HF tokenizer export from this project's trusted checkpoint."""
import argparse
import json
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if ck['metadata']['stage']!='tokenizer_reimplementation':raise ValueError('Not a tokenizer checkpoint')
    from ivideogpt.tokenizer import CompressiveVQModelFSQ
    model=CompressiveVQModelFSQ.from_config(ck['metadata']['tokenizer_config'])
    model.load_state_dict(ck['modules']['tokenizer'],strict=True);model.save_pretrained(out)
    (out.parent/f'hf-export-{ck["step"]:06d}.done.json').write_text(json.dumps(dict(step=ck['step'])))
if __name__=='__main__':main()
