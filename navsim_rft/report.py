"""Compare the three official CSVs on exactly the same successfully evaluated tokens."""
import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    for name in ('sft','image','driving','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();summary={};token_set=None
    for name in ('sft','image','driving'):
        path=Path(getattr(a,name));df=pd.read_csv(path);df=df[df.token!='average']
        if df.empty or df.token.duplicated().any() or not df.valid.all():raise ValueError(f'{name}: invalid/missing/duplicate scenes')
        if token_set is None:token_set=set(df.token)
        if set(df.token)!=token_set:raise ValueError('Evaluation token sets differ')
        cols=[c for c in df.select_dtypes('number').columns if not c.startswith('Unnamed')]
        summary[name]=dict(scenes=len(df),metrics=df[cols].mean().to_dict(),source=str(path.resolve()))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(summary,indent=2))
    pd.DataFrame({k:v['metrics'] for k,v in summary.items()}).to_csv(out/'comparison.csv')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
