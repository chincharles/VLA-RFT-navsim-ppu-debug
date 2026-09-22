import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from navsim_rft.geometry import local_deltas,integrate,ActionStats,wrap
from navsim_rft.policy import FlowPolicy,group_advantages,objective
from navsim_rft.checkpoint import save,load,fingerprint

class Core(unittest.TestCase):
    def setUp(self): torch.manual_seed(42);torch.set_num_threads(2)
    def test_se2_rotation_wrap_normalization(self):
        poses=np.array([[1,0,np.pi/2],[1,2,np.pi-.01],[-1,2,-np.pi+.01]])
        delta=local_deltas(poses)
        np.testing.assert_allclose(delta[1,:2],[2,0],atol=1e-10)
        self.assertAlmostEqual(delta[2,2],.02)
        np.testing.assert_allclose(integrate(delta),poses,atol=1e-10)
        stat=ActionStats.fit([poses],['train-token'],'train')
        np.testing.assert_allclose(stat.denormalize(stat.normalize(delta)),delta,atol=1e-10)
        with self.assertRaises(ValueError):ActionStats.fit([poses],['test'],'test')
    def test_paper_equations_are_not_ppo(self):
        reward=torch.tensor([[1.,3.,2.],[9.,9.,9.]])
        a=group_advantages(reward)
        torch.testing.assert_close(a,torch.tensor([[-1.,1.,0.],[0.,0.,0.]]))
        new=torch.tensor([2.],requires_grad=True)
        loss=objective(new,torch.zeros(1),torch.ones(1,1),torch.zeros(1),entropy_coef=0.)
        loss.backward();self.assertEqual(float(new.grad),0.)
    def test_real_dit_rl_freeze_and_resume(self):
        for mode in ('paper','source'):
            p=FlowPolicy(context_dim=16,hidden=32,depth=2,heads=4,steps=3,mode=mode)
            frozen=torch.nn.Linear(7,16).requires_grad_(False)
            frozen_hash=fingerprint(frozen)
            context=frozen(torch.randn(2,5,7)).unsqueeze(1);state=torch.randn(2,8);target=torch.randn(2,8,3)
            opt=torch.optim.AdamW(p.parameters(),lr=1e-3)
            for _ in range(2):
                loss=p.supervised(context,state,target);loss.backward();opt.step();opt.zero_grad()
            context=context.repeat_interleave(3,0);state=state.repeat_interleave(3,0)
            chain=p.sample(context,state)
            old=p.log_prob(context,state,chain)[0].detach()
            new,entropy=p.log_prob(context,state,chain)
            torch.testing.assert_close(new,old,rtol=0,atol=0)
            self.assertGreater(float(chain[:,-1].std(0).mean()),0)
            # Synthetic reward, explicitly not LPIPS or NAVSIM verification.
            r=-chain[:,-1].square().mean((1,2)).reshape(2,3)
            a=group_advantages(r,mode)
            loss=objective(new,old,a,entropy,mode)
            loss.backward()
            for module in (p.flow,p.sigma,p.action_projector,p.proprio_projector):
                grads=[v.grad for v in module.parameters() if v.grad is not None]
                self.assertTrue(grads);self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.)
            before=fingerprint(p);opt.step();self.assertNotEqual(before,fingerprint(p));self.assertEqual(frozen_hash,fingerprint(frozen))
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'ck.pt';save(path,{'policy':p},opt,2,{'synthetic':True})
                expected=p.sample(context,state)
                q=copy.deepcopy(p);opt2=torch.optim.AdamW(q.parameters(),lr=1e-3)
                load(path,{'policy':q},opt2)
                actual=q.sample(context,state);torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                # Identical next optimizer update after resume, including Adam moments.
                for m,o in ((p,opt),(q,opt2)):
                    o.zero_grad();lp,en=m.log_prob(context,state,actual)
                    objective(lp,old,a,en,mode).backward();o.step()
                self.assertEqual(fingerprint(p),fingerprint(q))
    def test_logprob_reductions(self):
        p=FlowPolicy(context_dim=16,hidden=32,depth=2,heads=4,steps=3)
        c=torch.randn(2,1,5,16);s=torch.randn(2,8);chain=p.sample(c,s)
        direct=[]
        for k in range(3):direct.append(p.transition(c,s,chain[:,k],torch.full((2,),k/3)).log_prob(chain[:,k+1]))
        torch.testing.assert_close(p.log_prob(c,s,chain)[0],torch.stack(direct,1).sum((-1,-2)).mean(1))
if __name__=='__main__':unittest.main(verbosity=2)
