"""Export official v1.1 scenes into auditable shards. No future in inference inputs."""
import json
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image
from . import NAVSIM_COMMIT
from .geometry import ActionStats


def verify_navsim(root):
    root = Path(root).resolve()
    rev = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if rev != NAVSIM_COMMIT:
        raise RuntimeError(f'Only NAVSIM v1.1 {NAVSIM_COMMIT} supported; got {rev}')
    import navsim
    if Path(navsim.__file__).resolve().parent != root/'navsim':
        raise RuntimeError('Imported NAVSIM differs from configured checkout')


def legal_input(agent_input):
    e = agent_input.ego_statuses[-1]
    command = np.asarray(e.driving_command,dtype=np.float32).reshape(-1)
    if command.shape != (4,):
        raise ValueError(f'Expected v1.1 4-way command, got {command.shape}')
    # Avoid inventing class ordering; official one-hot is itself a lawful text instruction.
    text = 'Drive according to the official navigation command vector: '+','.join(str(int(v)) for v in command)
    state = np.concatenate([e.ego_velocity, e.ego_acceleration, command]).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError('Unexpected ego status')
    image = agent_input.cameras[-1].cam_f0.image
    if image is None:
        raise ValueError('Missing initial front image')
    return Image.fromarray(image.astype(np.uint8)).convert('RGB'), state, text


def build_loader(root, logs, sensors, split, limit=None):
    verify_navsim(root)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import SceneLoader
    path = Path(root)/'navsim/planning/script/config/common/train_test_split/scene_filter'/f'{split}.yaml'
    cfg = OmegaConf.load(path)
    filt = instantiate(cfg)
    if filt.num_history_frames != 4 or filt.num_future_frames < 8:
        raise ValueError('Unexpected v1.1 temporal specification')
    filt.max_scenes = limit
    sensor = SensorConfig.build_no_sensors()
    sensor.cam_f0 = True # load future frames for supervision; policy uses AgentInput only
    return SceneLoader(Path(logs),Path(sensors),filt,sensor)


def export(root,logs,sensors,split,out,limit=None,role=None,exclude=None):
    if limit is not None and limit<1:raise ValueError('Positive export limit required')
    if split not in ('navtrain','navtest','navmini'):
        raise ValueError('Use official navtrain/navtest/navmini')
    if role not in ('train','val','test') or (role in ('train','val') and split!='navtrain'):
        raise ValueError('Training/validation must be subsets of navtrain')
    if role=='test' and split=='navtrain':
        raise ValueError('Use val for a held-out navtrain subset')
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    loader=build_loader(root,logs,sensors,split,None)
    excluded=set()
    if exclude:
        excluded={r['log_name'] for r in json.loads(Path(exclude).read_text())['records']}
    records=[]; trajectories=[]
    for token in sorted(loader.tokens):
        # Scene token is initial frame token used by evaluator/cache, NOT scene_metadata.scene_token.
        raw=loader.scene_frames_dicts[token]
        log_name=raw[3]['log_name']
        if log_name in excluded:
            continue
        # Stable log-level holdout; overlapping clips cannot cross train/validation.
        is_val=int(hashlib.sha256(log_name.encode()).hexdigest()[:8],16)%10==0
        if split=='navtrain' and is_val != (role=='val'):
            continue
        scene=loader.get_scene_from_token(token)
        h=scene.scene_metadata.num_history_frames
        frames=scene.frames[:h+8]
        times=np.array([f.timestamp for f in frames],dtype=np.int64)
        if len(times)!=h+8 or not np.all(np.abs(np.diff(times)-500000)<=50000):
            raise ValueError(f'{token}: irregular frame timing {np.diff(times)}')
        image,state,text=legal_input(scene.get_agent_input())
        poses=scene.get_future_trajectory(8).poses.astype(np.float64)
        history=[]; future=[]; valid=[]; camera_times=[]
        for i,f in enumerate(frames):
            im=f.cameras.cam_f0.image
            ok=im is not None
            arr=np.asarray(Image.fromarray(im).convert('RGB').resize((256,256))) if ok else np.zeros((256,256,3),np.uint8)
            (history if i<h else future).append(arr)
            if i>=h: valid.append(ok)
            cam=raw[i]['cams'].get('CAM_F0',raw[i]['cams'].get('cam_f0',{}))
            camera_times.append(cam.get('timestamp'))
            if cam.get('timestamp') is not None and abs(int(cam['timestamp'])-int(times[i]))>50000:
                raise ValueError(f'{token}: camera/frame timestamps differ >50ms')
        if not all(valid):
            raise ValueError(f'{token}: missing future camera; first version requires complete clips')
        if token!=scene.scene_metadata.initial_token:
            raise ValueError('Token mismatch')
        np.savez_compressed(out/f'{token}.npz',history=np.stack(history),future=np.stack(future),
            state=state,poses=poses,valid=np.asarray(valid),timestamps=times,
            initial=np.asarray(image))
        records.append(dict(token=token,scene_token=scene.scene_metadata.scene_token,
            log_name=scene.scene_metadata.log_name,text=text,file=f'{token}.npz',
            timestamps=times.tolist(),camera_timestamps=camera_times,
            camera_timestamp_evidence='sensor timestamp' if all(t is not None for t in camera_times) else 'official frame association only; no independent camera timestamp'))
        trajectories.append(poses)
        if limit and len(records)>=limit: break
    if not records: raise ValueError('No scenes selected')
    manifest=dict(navsim_commit=NAVSIM_COMMIT,split=split,role=role,dt=.5,horizon=4.,
                  views=['cam_f0'],records=records)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    if role=='train':
        ActionStats.fit(trajectories,[r['token'] for r in records],role).save(out/'stats.json')
    return manifest


class SceneDataset:
    def __init__(self, manifest, stats, training=False):
        self.path=Path(manifest); self.manifest=json.loads(self.path.read_text())
        self.records=self.manifest['records']; self.stats=ActionStats.load(stats)
        if self.manifest['navsim_commit']!=NAVSIM_COMMIT: raise ValueError('Wrong NAVSIM')
        tokens={r['token'] for r in self.records}
        if training and (self.manifest['role']!='train' or not tokens.issubset(set(self.stats.record['tokens']))):
            raise ValueError('Training data/statistics provenance mismatch')
        if self.manifest['role']!='train' and tokens & set(self.stats.record['tokens']):
            raise ValueError('Held-out data overlap with normalization data')
    def __len__(self): return len(self.records)
    def __getitem__(self,i):
        import torch
        from .geometry import local_deltas
        r=self.records[i]
        with np.load(self.path.parent/r['file'],allow_pickle=False) as z:
            d={k:z[k].copy() for k in z.files}
        delta=local_deltas(d['poses'])
        return dict(token=r['token'],text=r['text'],image=Image.fromarray(d['initial']),
            history=d['history'],state=torch.tensor(d['state']),
            actions=torch.tensor(self.stats.normalize(delta),dtype=torch.float32),
            deltas=delta,poses=d['poses'],valid=torch.tensor(d['valid']),
            video=torch.tensor(np.concatenate([d['history'][-1:],d['future']]),dtype=torch.float32).permute(0,3,1,2)/255)
