import torch
from torch import nn

class ImageReward(nn.Module):
    def __init__(self,l1=1.,lpips=1.,aggregate='sum'):
        super().__init__()
        from ivideogpt.lpips import LPIPS
        self.metric=LPIPS().eval().requires_grad_(False)
        self.l1,self.lpips,self.aggregate=l1,lpips,aggregate
    @torch.no_grad()
    def forward(self,pred,ref,mask):
        if pred.shape!=ref.shape or mask.shape!=pred.shape[:2]: raise ValueError('Image alignment mismatch')
        b,t=mask.shape
        a=(pred-ref).abs().mean((-1,-2,-3))
        p=[]
        for x,y in zip(pred.flatten(0,1).split(8),ref.flatten(0,1).split(8)):
            p.append(self.metric(x*2-1,y*2-1).flatten(1).mean(1))
        percept=torch.cat(p).reshape(b,t)
        cost=(self.l1*a+self.lpips*percept)*mask
        reward=-cost.sum(1)
        if self.aggregate=='mean': reward=reward/mask.sum(1).clamp_min(1)
        return reward,dict(l1=a,lpips=percept)

class DrivingReward:
    def __init__(self,cache_path,allowed_train_tokens):
        from pathlib import Path
        from navsim.common.dataloader import MetricCacheLoader
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
        self.allowed=set(allowed_train_tokens)
        self.cache=MetricCacheLoader(Path(cache_path))
        self.sampling=TrajectorySampling(num_poses=40,interval_length=.1)
        self.simulator=PDMSimulator(self.sampling)
        self.scorer=PDMScorer(self.sampling)
        if not self.allowed.issubset(set(self.cache.tokens)): raise ValueError('Missing training metric caches')
    def __call__(self,tokens,poses):
        from navsim.common.dataclasses import Trajectory
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.evaluate.pdm_score import pdm_score
        values=[]
        for token,trajectory in zip(tokens,poses):
            if token not in self.allowed: raise ValueError('Non-training token requested for RL reward')
            result=pdm_score(self.cache.get_from_token(token),Trajectory(trajectory,TrajectorySampling(num_poses=8,interval_length=.5)),
                self.sampling,self.simulator,self.scorer)
            values.append(float(result.score))
        return torch.tensor(values,dtype=torch.float32)
