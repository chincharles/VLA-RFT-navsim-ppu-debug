"""SE(2), metres/radians, rear axle, +x forward, +y left. No clipping on inverse."""
import hashlib
import json
import numpy as np


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


def local_deltas(poses):
    poses = np.asarray(poses, dtype=np.float64)
    previous = np.concatenate([np.zeros_like(poses[..., :1, :]), poses[..., :-1, :]], axis=-2)
    xy = poses[..., :2] - previous[..., :2]
    c, s = np.cos(previous[..., 2]), np.sin(previous[..., 2])
    return np.stack([c*xy[..., 0]+s*xy[..., 1], -s*xy[..., 0]+c*xy[..., 1],
                     wrap(poses[..., 2]-previous[..., 2])], axis=-1)


def integrate(deltas):
    deltas = np.asarray(deltas, dtype=np.float64)
    result = np.zeros_like(deltas)
    prev = np.zeros_like(deltas[..., 0, :])
    for t in range(deltas.shape[-2]):
        d = deltas[..., t, :]
        c, s = np.cos(prev[..., 2]), np.sin(prev[..., 2])
        prev = np.stack([prev[..., 0]+c*d[..., 0]-s*d[..., 1],
                         prev[..., 1]+s*d[..., 0]+c*d[..., 1], wrap(prev[..., 2]+d[..., 2])], -1)
        result[..., t, :] = prev
    return result


class ActionStats:
    def __init__(self, record):
        if record['role'] != 'train' or record['encoding'] != 'se2_local_delta':
            raise ValueError('Statistics must be fitted on train SE(2) deltas')
        self.record = record
        self.mean = np.asarray(record['mean'])
        self.std = np.asarray(record['std'])
        self.low = np.asarray(record['min'])
        self.high = np.asarray(record['max'])
        if np.any(self.std <= 0) or not np.isfinite(self.std).all():
            raise ValueError('Invalid normalization')

    @classmethod
    def fit(cls, trajectories, tokens, role):
        if role != 'train':
            raise ValueError('Refusing statistics from non-training data')
        x = local_deltas(np.asarray(trajectories)).reshape(-1, 3)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite actions')
        return cls(dict(role=role, encoding='se2_local_delta', mean=x.mean(0).tolist(),
                        std=np.maximum(x.std(0), 1e-6).tolist(), min=x.min(0).tolist(),
                        max=x.max(0).tolist(), tokens=sorted(tokens),
                        token_sha256=hashlib.sha256('\n'.join(sorted(tokens)).encode()).hexdigest()))

    def normalize(self, x):
        return (x-self.mean)/self.std

    def denormalize(self, x):
        return x*self.std+self.mean

    def bins(self, x, n=256):
        return np.floor(np.clip((x-self.low)/np.maximum(self.high-self.low, 1e-6), 0, 1)*n).clip(0,n-1).astype(np.int64)

    def save(self, path):
        from pathlib import Path
        Path(path).write_text(json.dumps(self.record, indent=2))

    @classmethod
    def load(cls, path):
        from pathlib import Path
        return cls(json.loads(Path(path).read_text()))
