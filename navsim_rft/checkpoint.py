import hashlib
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist


def fingerprint(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=[torch.cuda.get_rng_state()] if torch.cuda.is_available() else [],cuda_scope='current')


def restore_rng(r):
    random.setstate(r['python']); np.random.set_state(r['numpy']); torch.set_rng_state(r['torch'])
    if r['cuda']:
        if r.get('cuda_scope')=='current':torch.cuda.set_rng_state(r['cuda'][0])
        else:torch.cuda.set_rng_state_all(r['cuda']) # Older single-process checkpoints.


def save_distributed(path,modules,optimizer,step,metadata):
    """All ranks participate; only rank zero writes, preserving per-rank RNG."""
    world=dist.get_world_size() if dist.is_initialized() else 1
    rank=dist.get_rank() if dist.is_initialized() else 0
    states=[None]*world
    if world>1:dist.all_gather_object(states,rng_state())
    else:states=[rng_state()]
    if rank==0:save(path,modules,optimizer,step,metadata,states)
    if world>1:dist.barrier()


def save(path,modules,optimizer,step,metadata,rng_states=None):
    path=Path(path)
    if path.exists(): raise FileExistsError(f'Refusing overwrite: {path}')
    payload=dict(schema=1,modules={k:m.state_dict() for k,m in modules.items()},
                 optimizer=optimizer.state_dict(),step=step,metadata=metadata,rng=rng_states or [rng_state()])
    tmp=path.with_suffix('.tmp')
    torch.save(payload,tmp); tmp.replace(path)


def load(path,modules,optimizer=None,rank=0):
    # Only load checkpoints produced/trusted by this project, optimizer/RNG use pickle.
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p['schema']!=1: raise ValueError('Unsupported checkpoint')
    for k,m in modules.items(): m.load_state_dict(p['modules'][k],strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(p['optimizer'])
        if rank>=len(p['rng']): raise ValueError('Resume world size differs')
        restore_rng(p['rng'][rank])
    return p
