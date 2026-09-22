"""Official AbstractAgent: inference accepts AgentInput only, never Scene or metric cache."""
import torch
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig,Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from .data import legal_input,verify_navsim
from .geometry import ActionStats,integrate
from .policy import FlowPolicy
from .checkpoint import load

class NavsimRFTAgent(AbstractAgent):
    def __init__(self,checkpoint,device='cuda',seed=42,vlm_path=None,navsim_root=None):
        super().__init__(requires_scene=False)
        self.checkpoint=checkpoint; self.device_name=device; self.seed=seed; self.vlm_path=vlm_path
        self.navsim_root=navsim_root
    def name(self): return 'unofficial_vla_rft_navsim_v1_1'
    def get_sensor_config(self):
        s=SensorConfig.build_no_sensors(); s.cam_f0=[3]; return s
    def initialize(self):
        from .encoder import PrismaticEncoder
        ck=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        cfg=ck['metadata']['config']; self.stats=ActionStats(ck['metadata']['stats'])
        verify_navsim(self.navsim_root or cfg['navsim_root'])
        self.encoder=PrismaticEncoder(self.vlm_path or cfg['vlm']).to(self.device_name)
        self.encoder.set_trainable(False)
        self.policy=FlowPolicy(context_dim=self.encoder.dim,mode=cfg['math_mode'],**cfg.get('policy_arch',{})).to(self.device_name)
        load(self.checkpoint,dict(policy=self.policy,encoder=self.encoder))
        self.eval()
    @torch.no_grad()
    def compute_trajectory(self,agent_input):
        image,state,text=legal_input(agent_input)
        context=self.encoder([image],[text])
        s=torch.tensor(state,device=self.device_name)[None]
        # Fixed initial Gaussian noise, deterministic ODE. Never select using future truth.
        generator=torch.Generator(device=self.device_name).manual_seed(self.seed)
        noise=torch.randn(1,8,3,generator=generator,device=self.device_name)
        chain=self.policy.sample(context,s,stochastic=False,initial_noise=noise)
        poses=integrate(self.stats.denormalize(chain[0,-1].cpu().numpy()))
        return Trajectory(poses,TrajectorySampling(num_poses=8,interval_length=.5))
