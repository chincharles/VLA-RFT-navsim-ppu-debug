import argparse
import json
from .data import export

def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export')
    for key in ('root','logs','sensors','split','out','role'): e.add_argument('--'+key,required=True)
    e.add_argument('--limit',type=int); e.add_argument('--exclude')
    a=vars(p.parse_args()); a.pop('command'); result=export(**a)
    print(json.dumps(dict(count=len(result['records']),role=result['role'],split=result['split']),indent=2))
if __name__=='__main__': main()
