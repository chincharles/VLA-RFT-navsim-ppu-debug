"""Original DiT architecture, NAVSIM dimensions; paper and release-code math explicit."""
import math
import torch
from torch import nn
from .dit import DiT_SingleTokenAction_OneCtx

class Projector(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

class FlowPolicy(nn.Module):
    def __init__(self, context_dim=896, hidden=512, depth=8, heads=8, steps=10, mode='paper', min_std=.08, max_std=.2):
        super().__init__()
        self.steps, self.mode = steps, mode
        self.action_projector = Projector(1, context_dim)
        self.proprio_projector = Projector(8, context_dim)
        kw = dict(in_channels=3*context_dim, out_channels=3, num_actions=8,
                  context_dim=context_dim, hidden_size=hidden, depth=depth, num_heads=heads)
        self.flow = DiT_SingleTokenAction_OneCtx(**kw)
        self.sigma = DiT_SingleTokenAction_OneCtx(**kw)
        self.log_min, self.log_max = math.log(min_std), math.log(max_std)
        if mode not in ('paper', 'source'):
            raise ValueError(mode)
        # Dropout changes conditional densities on replay. Always disabled, including SFT.
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.
            if hasattr(m, "dropout") and isinstance(m.dropout, float):
                m.dropout = 0.

    def transition(self, context, state, x, t):
        obs = self.action_projector(x.unsqueeze(-1)).flatten(-2)
        proprio = self.proprio_projector(state).unsqueeze(1)
        v = self.flow(obs, t, context, proprio)
        raw = self.sigma(obs, t, context, proprio)
        log_std = self.log_min+(self.log_max-self.log_min)*(raw.tanh()+1)*.5
        dt = (1 if self.mode == 'paper' else -1)/self.steps
        return torch.distributions.Normal((x+dt*v).float(), log_std.float().exp())

    def supervised(self, context, state, actions):
        eps = torch.randn_like(actions)
        if self.mode == 'paper':
            t = torch.distributions.Beta(1.5, 1.).sample((actions.shape[0],)).to(actions.device)
            target = actions-eps
        else:
            u = torch.rand(actions.shape[0], device=actions.device).pow(1/1.5)
            v = torch.rand_like(u)
            t = (u/(u+v))*.999+.001
            target = eps-actions
        x = (1-t[:,None,None])*eps+t[:,None,None]*actions
        obs = self.action_projector(x.unsqueeze(-1)).flatten(-2)
        pred = self.flow(obs, t, context, self.proprio_projector(state).unsqueeze(1))
        return (pred-target).square().mean()

    @torch.no_grad()
    def sample(self, context, state, stochastic=True, initial_noise=None):
        b = len(state)
        x = torch.randn(b,8,3,device=state.device) if initial_noise is None else initial_noise
        chain = [x]
        for k in range(self.steps):
            t = torch.full((b,), k/self.steps, device=x.device)
            dist = self.transition(context, state, x, t)
            x = dist.sample() if stochastic else dist.mean
            chain.append(x)
        return torch.stack(chain, dim=1)

    def log_prob(self, context, state, chain):
        # Same stored transitions, detached from rollout; never resample x under new policy.
        chain = chain.detach()
        lp, entropy = [], []
        for k in range(self.steps):
            d = self.transition(context,state,chain[:,k],torch.full((len(state),),k/self.steps,device=state.device))
            lp.append(d.log_prob(chain[:,k+1]))
            entropy.append(d.entropy())
        lp, entropy = torch.stack(lp,1), torch.stack(entropy,1) # B,K,T,D
        if self.mode == 'paper':
            return lp.sum((-1,-2)).mean(1), entropy.sum((-1,-2)).mean(1)
        return lp.sum(1).flatten(1), entropy.sum(1).flatten(1)/(self.steps+1)


def group_advantages(rewards, mode='paper'):
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError('Require B scenes x N>=2 candidates')
    a = rewards.detach()-rewards.detach().mean(1,keepdim=True)
    return a if mode=='paper' else a/(rewards.detach().std(1,keepdim=True)+1e-6)


def objective(new, old, advantages, entropy, mode='paper', clip=.2, entropy_coef=.001):
    delta = new-old.detach()
    if not torch.isfinite(delta).all() or delta.abs().max() > 60:
        raise FloatingPointError('Unsafe likelihood ratio; stop rather than silently clip log ratios')
    ratio = delta.exp()
    a = advantages.detach().flatten()
    if mode == 'source':
        a = a[:,None]
        loss = torch.maximum(-a*ratio, -a*ratio.clamp(1-clip,1+clip))
        loss = torch.where(a<0, torch.minimum(-3*a,loss), loss).mean()
    else:
        loss = -(ratio.clamp(1-clip,1+clip)*a).mean()
    return loss-entropy_coef*entropy.mean()
