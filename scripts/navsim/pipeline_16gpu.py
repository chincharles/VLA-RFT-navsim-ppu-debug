#!/usr/bin/env python3
"""Single-node or two-node 16-GPU pipeline with shared CPFS coordination.

The launcher uses only the standard library. Each node gets an isolated venv;
all training subprocesses use the existing source models and official evaluator.
"""
import argparse
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

REPO=Path(__file__).resolve().parents[2]
PIN='0811876c274e8b058ab2be9b3dcd4d37bd23f177'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def source_fingerprint():
    digest=hashlib.sha256()
    for folder in ('navsim_rft','scripts/navsim','configs/navsim','train/verl/ivideogpt','train/verl/vla-adapter/openvla-oft/prismatic'):
        for p in sorted((REPO/folder).rglob('*')):
            if p.is_file() and p.suffix in ('.py','.sh','.json','.txt','.yaml','.yml') and '__pycache__' not in p.parts:
                digest.update(str(p.relative_to(REPO)).encode());digest.update(p.read_bytes())
    return digest.hexdigest()


def latest_checkpoint(directory,total):
    matches=[]
    for p in Path(directory).glob('step-*.pt'):
        m=re.fullmatch(r'step-(\d+)\.pt',p.name)
        if m:matches.append((int(m.group(1)),p))
    if not matches:return 0,None
    step,path=max(matches)
    if step>total:raise ValueError(f'{path} exceeds requested total {total}')
    return step,path


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nnodes',type=int,default=int(os.environ.get('NNODES','1')))
    p.add_argument('--gpus-per-node',type=int,default=None)
    p.add_argument('--node-rank',type=int,default=int(os.environ.get('NODE_RANK','0')))
    p.add_argument('--master-addr',default=os.environ.get('MASTER_ADDR','127.0.0.1'))
    p.add_argument('--master-port',type=int,default=int(os.environ.get('MASTER_PORT','29500')))
    p.add_argument('--run-id',default=os.environ.get('RFT_RUN_ID'))
    p.add_argument('--resume-run',default=os.environ.get('RFT_RESUME_RUN'))
    p.add_argument('--attempt-id',default=os.environ.get('RFT_ATTEMPT_ID'))
    p.add_argument('--profile',choices=['train','smoke'],default='train')
    p.add_argument('--config',default=str(REPO/'configs/navsim/train16.json'))
    p.add_argument('--print-plan',action='store_true',help='Print paths/budgets without downloads or training')
    p.add_argument('--wait-timeout',type=int,default=86400,help='Maximum seconds waiting for a peer/preprocessing phase')
    a=p.parse_args()
    if a.nnodes not in (1,2):p.error('Supported layouts are 1x16 or 2x8 GPUs')
    a.gpus_per_node=a.gpus_per_node or int(os.environ.get('NPROC_PER_NODE',str(16//a.nnodes)))
    if a.nnodes*a.gpus_per_node!=16:p.error('Exactly 16 GPUs required: 1x16 or 2x8')
    if not 0<=a.node_rank<a.nnodes:p.error('node-rank must be in [0, nnodes)')
    if not 1<=a.master_port<=65535 or a.wait_timeout<1:p.error('Invalid port/timeout')
    if a.nnodes>1 and a.master_addr in ('127.0.0.1','localhost','::1') and not a.print_plan:
        p.error('Two nodes require a reachable master node address')
    if a.nnodes>1 and not (a.run_id or a.resume_run) and not a.print_plan:
        p.error('Both nodes must provide the same unique --run-id')
    if a.resume_run and a.nnodes>1 and not a.attempt_id and not a.print_plan:
        p.error('Multi-node resume requires the same new --attempt-id on both nodes')
    for name in ('run_id','attempt_id'):
        value=getattr(a,name)
        if value and not re.fullmatch(r'[A-Za-z0-9_.-]+',value):p.error(f'Invalid {name}')
    return a


class Pipeline:
    def __init__(self,args):
        self.a=args;self.budget=read_json(args.config)
        if args.profile=='smoke':
            self.budget.update(train_limit=32,val_limit=8,tokenizer_steps=2,wm_steps=2,sft_steps=2,
                               rl_img_steps=1,rl_drive_steps=1,save_every=1,quality_limit=2,
                               reward_analysis_limit=2,cache_worker='sequential')
        for key in ('tokenizer_steps','wm_steps','sft_steps','rl_img_steps','rl_drive_steps','save_every','val_limit',
                    'batch_size_per_gpu','cache_workers','quality_limit','reward_analysis_limit'):
            if not isinstance(self.budget[key],int) or self.budget[key]<1:raise ValueError(f'{key} must be positive')
        if self.budget['train_limit']<0 or self.budget['candidates']<2:raise ValueError('Invalid data limit/candidates')
        if self.budget['cache_worker'] not in ('sequential','ray_distributed'):raise ValueError('Unknown cache worker')
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_id=args.run_id or ('train16-'+stamp+'-'+uuid.uuid4().hex[:6])
        self.env=dict(os.environ)
        self.run=Path(args.resume_run).resolve() if args.resume_run else Path(self.env['RFT_OUTPUT_ROOT'])/run_id
        attempt=args.attempt_id or (('resume-'+stamp+'-'+uuid.uuid4().hex[:6]) if args.resume_run else 'initial')
        self.attempt=self.run/'launches'/attempt
        self.node=self.attempt/f'node-{args.node_rank}'
        self.py=self.run/'envs'/f'node-{args.node_rank}'/'bin/python'
        self.stage='starting';self.child=None;self.joined=False;self.run_lock=None
        for key in ('PYTHONHOME','PYTHONPATH','RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE'):
            self.env.pop(key,None)
        # Respect scheduler GPU allocation; every requested local device is used.
        self.env.update(RFT_WORK=str(self.run),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
            HF_HOME=str(Path(self.env['RFT_CACHE_ROOT'])/'huggingface'),
            PIP_CACHE_DIR=str(Path(self.env['RFT_CACHE_ROOT'])/'pip'),
            TORCH_HOME=str(Path(self.env['RFT_CACHE_ROOT'])/'torch'),
            NUPLAN_MAP_VERSION='nuplan-maps-v1.0',NAVSIM_EXP_ROOT=str(self.run/'navsim-exp'))
        self.expected=dict(schema=1,source_fingerprint=source_fingerprint(),budget=self.budget,
            nnodes=args.nnodes,gpus_per_node=args.gpus_per_node,world_size=16,
            data={key:self.env[key] for key in ('OPENSCENE_DATA_ROOT','NUPLAN_MAPS_ROOT','TRAIN_LOGS','TRAIN_SENSORS','RFT_CACHE_ROOT')},
            navsim_commit=PIN)

    def failure_check(self):
        for p in self.attempt.glob('FAILED.node-*.json'):
            raise RuntimeError(f'Peer failure recorded in {p}: {read_json(p).get("stage")}')

    def wait_for(self,predicate,label):
        began=time.monotonic();last=0
        while not predicate():
            self.failure_check()
            elapsed=time.monotonic()-began
            if elapsed>self.a.wait_timeout:raise TimeoutError(label)
            if elapsed-last>=60:print(f'[node {self.a.node_rank}] Waiting: {label}',flush=True);last=elapsed
            time.sleep(2)

    def command(self,cmd,label,env=None):
        self.stage=label
        print(f'\n[node {self.a.node_rank}] {label}',flush=True)
        log=self.node/(label.replace('/','_')+'.log')
        with log.open('a') as stream:
            self.child=subprocess.Popen([str(v) for v in cmd],cwd=REPO,env=env or self.env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
            child=self.child
            def drain():
                for line in child.stdout:
                    stream.write(line);stream.flush();print(line,end='',flush=True)
            reader=threading.Thread(target=drain,daemon=True);reader.start()
            try:
                while child.poll() is None:
                    self.failure_check();time.sleep(1)
                reader.join(timeout=10)
                if child.returncode:raise subprocess.CalledProcessError(child.returncode,cmd)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    reader.join(timeout=5)
                self.child=None

    def leader(self,name,fn):
        self.stage=name;marker=self.attempt/(name+'.done.json')
        if self.a.node_rank==0:
            fn();write_json(marker,dict(done=True))
        else:self.wait_for(marker.exists,name)

    def sync(self,name):
        self.stage=name;write_json(self.attempt/f'{name}.node-{self.a.node_rank}.json',dict(done=True))
        self.wait_for(lambda:all((self.attempt/f'{name}.node-{i}.json').exists() for i in range(self.a.nnodes)),name)

    def distributed(self,module,arguments,name):
        cmd=[self.py,'-m','torch.distributed.run',f'--nnodes={self.a.nnodes}',f'--nproc_per_node={self.a.gpus_per_node}',
             f'--node_rank={self.a.node_rank}',f'--master_addr={self.a.master_addr}',f'--master_port={self.a.master_port}',
             '--max_restarts=0','-m',module,*arguments]
        self.command(cmd,name)
        self.sync(name+'_all_nodes_finished')

    def start(self):
        if sys.platform!='linux' or sys.version_info[:2]!=(3,10):raise RuntimeError('Use a Linux Python 3.10 runtime/image')
        for key in ('OPENSCENE_DATA_ROOT','NUPLAN_MAPS_ROOT','TRAIN_LOGS','TRAIN_SENSORS'):
            if not Path(self.env[key]).is_absolute() or not Path(self.env[key]).is_dir():raise FileNotFoundError(f'{key}={self.env[key]}')
        if self.a.node_rank==0:
            if self.a.resume_run:
                if read_json(self.run/'pipeline.json')!=self.expected:raise ValueError('Resume source/config/topology/data paths changed')
            else:
                self.run.mkdir(parents=True,exist_ok=False);write_json(self.run/'pipeline.json',self.expected)
            self.run_lock=(self.run/'.pipeline.lock').open('a')
            fcntl.flock(self.run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.attempt.mkdir(parents=True,exist_ok=False)
            self.joined=True
            write_json(self.attempt/'start.json',dict(run=str(self.run),pipeline=self.expected))
        else:
            self.wait_for((self.attempt/'start.json').exists,'leader startup')
            self.joined=True
            if read_json(self.attempt/'start.json')['pipeline']!=self.expected:raise ValueError('Node code/config/data/topology differs')
        self.node.mkdir(exist_ok=False)
        print(f'Run directory: {self.run}\nAttempt: {self.attempt}\nNode: {self.a.node_rank}',flush=True)
        self.env['NAVSIM_ROOT']=str(self.run/'deps/navsim')
        self.leader('checkout_navsim',self.checkout)
        self.install()
        self.sync('environments_ready')
        self.leader('verify_node_dependencies',self.verify_dependencies)
        self.distributed('navsim_rft.check_distributed',['--output',str(self.attempt/'nccl'),'--expected','16'],'nccl_check')
        self.leader('prepare_data_and_weights',self.prepare)
        paths=read_json(self.run/'weights.json')
        self.env.update(VLM_DIR=paths['vlm'],VLA_RFT_VGG16_PATH=paths['vgg'],VLA_RFT_LPIPS_PATH=paths['lpips'])
        self.leader('verify_real_vlm',lambda:self.command([self.py,'scripts/navsim/check_vlm.py',
            '--config',self.run/'config/sft.json','--output',self.attempt/'vlm-check.json'],'verify_real_vlm'))
        self.train_stage('tokenizer')
        self.leader('prepare_official_cache',self.cache)
        self.train_stage('wm')
        self.leader('world_quality',lambda:self.command([self.py,'-m','navsim_rft.quality','--checkpoint',self.final('wm'),
            '--manifest',self.run/'data/val/manifest.json','--stats',self.run/'data/train/stats.json',
            '--output',self.attempt/'wm-quality','--limit',str(self.budget['quality_limit'])],'world_quality'))
        self.train_stage('sft')
        self.leader('evaluate_sft',lambda:self.evaluate('sft'))
        self.train_stage('rl_img')
        self.leader('evaluate_rl_img',lambda:self.evaluate('rl_img'))
        self.train_stage('rl_drive')
        self.leader('evaluate_rl_drive',lambda:self.evaluate('rl_drive'))
        self.leader('reports',self.reports)
        self.sync('completed')
        print(f'Pipeline completed: {self.run}',flush=True)

    def checkout(self):
        dest=Path(self.env['NAVSIM_ROOT'])
        if not dest.exists():
            self.command(['git','init','-q',dest],'git_init')
            self.command(['git','-C',dest,'remote','add','origin','https://github.com/autonomousvision/navsim.git'],'git_remote')
        head=subprocess.run(['git','-C',str(dest),'rev-parse','HEAD'],capture_output=True,text=True)
        if head.returncode:
            self.command(['git','-C',dest,'fetch','--depth','1','origin',PIN],'git_fetch')
            self.command(['git','-C',dest,'checkout','--detach','FETCH_HEAD'],'git_checkout')
        actual=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
        if actual!=PIN:raise ValueError('Unexpected NAVSIM checkout')
        if subprocess.check_output(['git','-C',str(dest),'status','--porcelain','--untracked-files=no'],text=True).strip():
            raise ValueError('NAVSIM official checkout contains tracked changes')

    def install(self):
        if not self.py.exists():self.command([sys.executable,'-m','venv',self.py.parent.parent],'create_venv')
        self.env.update(VIRTUAL_ENV=str(self.py.parent.parent),PATH=str(self.py.parent)+os.pathsep+self.env['PATH'],
            RFT_REQUIREMENTS_OUTPUT=str(self.node/'navsim-requirements.txt'),
            PYTHONPATH=os.pathsep.join(map(str,[REPO,REPO/'train/verl',REPO/'train/verl/vla-adapter/openvla-oft',Path(self.env['NAVSIM_ROOT'])])))
        self.command(['bash','scripts/navsim/install_server.sh'],'install_dependencies')
        with (self.node/'requirements-resolved.txt').open('w') as f:
            subprocess.run([str(self.py),'-m','pip','freeze'],env=self.env,stdout=f,check=True)

    def verify_dependencies(self):
        versions=[sorted((self.attempt/f'node-{i}/requirements-resolved.txt').read_text().splitlines())
                  for i in range(self.a.nnodes)]
        if any(value!=versions[0] for value in versions[1:]):
            raise ValueError('Installed dependency versions differ between nodes; see requirements-resolved.txt')
        previous=self.run/'dependencies-resolved.json'
        if previous.exists():
            if read_json(previous)!=versions[0]:raise ValueError('Installed dependencies changed since the original run')
        else:write_json(previous,versions[0])

    def export(self,role,limit):
        out=self.run/'data'/role
        if (out/'manifest.json').exists() and (role!='train' or (out/'stats.json').exists()):return
        if out.exists():
            # Preserve an interrupted partial export; create a new one at the canonical path.
            out.rename(out.with_name(role+'-partial-'+uuid.uuid4().hex[:8]))
        command=[self.py,'-m','navsim_rft.cli','export','--root',self.env['NAVSIM_ROOT'],'--logs',self.env['TRAIN_LOGS'],
                 '--sensors',self.env['TRAIN_SENSORS'],'--split','navtrain','--role',role,'--out',out]
        if limit:command+=['--limit',str(limit)]
        if role=='val':command+=['--exclude',self.run/'data/train/manifest.json']
        self.command(command,'export_'+role)

    def prepare(self):
        self.export('train',self.budget['train_limit']);self.export('val',self.budget['val_limit'])
        if not (self.run/'weights.json').exists():self.command([self.py,'scripts/navsim/clearml_weights.py'],'download_weights')
        paths=read_json(self.run/'weights.json')
        config=self.run/'config'
        config_marker=self.run/'prepared-config.json'
        if config_marker.exists():
            for name,digest in read_json(config_marker).items():
                if hashlib.sha256((config/name).read_bytes()).hexdigest()!=digest:raise ValueError('Prepared configuration modified')
        else:
            if config.exists():config.rename(config.with_name('config-partial-'+uuid.uuid4().hex[:8]))
            self.command([self.py,'scripts/navsim/configure.py','--navsim-root',self.env['NAVSIM_ROOT'],
                '--train-manifest',self.run/'data/train/manifest.json','--stats',self.run/'data/train/stats.json',
                '--vlm',paths['vlm'],'--tokenizer',self.run/f'tokenizer/pretrained-{self.budget["tokenizer_steps"]:06d}',
                '--metric-cache',self.run/'metric-cache','--output',config],'configure')
            for path in config.glob('*.json'):
                value=read_json(path);value.update(batch_size=self.budget['batch_size_per_gpu'],candidates=self.budget['candidates'])
                write_json(path,value)
            write_json(config_marker,{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in config.glob('*.json')})
        train=read_json(self.run/'data/train/manifest.json')
        if len(train['records'])<16*self.budget['batch_size_per_gpu']:raise ValueError('Too few training scenes for 16 ranks')

    def cache(self):
        if (self.run/'metric-cache-success.json').exists():return
        dest=self.run/'metric-cache'
        if dest.exists():dest.rename(dest.with_name('metric-cache-partial-'+uuid.uuid4().hex[:8]))
        self.command([self.py,'scripts/navsim/cache_subset.py','--navsim-root',self.env['NAVSIM_ROOT'],
            '--logs',self.env['TRAIN_LOGS'],'--manifests',self.run/'data/train/manifest.json',self.run/'data/val/manifest.json',
            '--cache',dest,'--worker',self.budget['cache_worker'],'--workers',str(self.budget['cache_workers'])],'metric_cache')
        write_json(self.run/'metric-cache-success.json',dict(done=True))

    def final(self,stage):
        return self.run/stage/f'step-{self.budget[stage+"_steps"]:06d}.pt'

    def train_stage(self,stage):
        total=self.budget[stage+'_steps'];out=self.run/stage
        step,checkpoint=latest_checkpoint(out,total)
        if step==total:
            if stage=='tokenizer' and not (out/f'hf-export-{total:06d}.done.json').exists():
                def recover_export():
                    target=out/f'pretrained-{total:06d}'
                    if target.exists():target.rename(target.with_name(target.name+'-partial-'+uuid.uuid4().hex[:8]))
                    self.command([self.py,'-m','navsim_rft.export_tokenizer','--checkpoint',self.final(stage),
                                  '--output',target],'recover_tokenizer_export')
                self.leader('recover_tokenizer_export',recover_export)
            self.sync(stage+'_already_complete');return
        if checkpoint is None and out.exists():
            self.leader(stage+'_preserve_partial',lambda:out.rename(out.with_name(stage+'-partial-'+uuid.uuid4().hex[:8])))
        module='navsim_rft.train_tokenizer' if stage=='tokenizer' else 'navsim_rft.train'
        args=['--output',str(out),'--steps',str(total-step),'--seed',str(self.budget['seed']),
              '--save-every',str(self.budget['save_every'])]
        if stage=='tokenizer':args+=['--manifest',str(self.run/'data/train/manifest.json'),'--stats',str(self.run/'data/train/stats.json')]
        else:args+=['--config',str(self.run/f'config/{stage}.json')]
        if checkpoint:args+=['--resume',str(checkpoint)]
        elif stage.startswith('rl_'):args+=['--init-policy',str(self.final('sft'))]
        if stage=='rl_img':args+=['--wm-checkpoint',str(self.final('wm'))]
        self.distributed(module,args,'train_'+stage)

    def evaluate(self,stage):
        self.command([self.py,'-m','navsim_rft.evaluate','--navsim-root',self.env['NAVSIM_ROOT'],
            '--checkpoint',self.final(stage),'--logs',self.env['TRAIN_LOGS'],'--sensors',self.env['TRAIN_SENSORS'],
            '--metric-cache',self.run/'metric-cache','--split','navtrain','--manifest',self.run/'data/val/manifest.json',
            '--output',self.attempt/('eval_'+stage)],'evaluate_'+stage)

    def reports(self):
        csvs={}
        for stage in ('sft','rl_img','rl_drive'):
            csvs[stage]=read_json(self.attempt/('eval_'+stage)/'evaluation_status.json')['csv'][-1]
        self.command([self.py,'-m','navsim_rft.report','--sft',csvs['sft'],'--image',csvs['rl_img'],
            '--driving',csvs['rl_drive'],'--output',self.attempt/'comparison'],'compare')
        self.command([self.py,'-m','navsim_rft.analyze_rewards','--policy',self.final('sft'),'--world',self.final('wm'),
            '--manifest',self.run/'data/train/manifest.json','--stats',self.run/'data/train/stats.json',
            '--metric-cache',self.run/'metric-cache','--output',self.attempt/'reward-analysis',
            '--limit',str(self.budget['reward_analysis_limit']),'--candidates',str(self.budget['candidates'])],'reward_analysis')


def main():
    args=parse_args();pipeline=Pipeline(args)
    if args.print_plan:
        print(json.dumps(dict(run=str(pipeline.run),node_rank=args.node_rank,**pipeline.expected),indent=2));return 0
    try:
        pipeline.start();return 0
    except BaseException as exc:
        if pipeline.joined:
            write_json(pipeline.attempt/f'FAILED.node-{args.node_rank}.json',dict(stage=pipeline.stage,error_type=type(exc).__name__))
        print(f'FAILED: node={args.node_rank}, stage={pipeline.stage}, run={pipeline.run}',file=sys.stderr,flush=True)
        raise

if __name__=='__main__':raise SystemExit(main())
