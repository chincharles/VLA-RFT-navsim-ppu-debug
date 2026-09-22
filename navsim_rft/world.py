"""Compressive FSQ tokenizer + causal Llama. Source token layout, 3 driving action tokens."""
import torch
from torch import nn

class TokenWorld(nn.Module):
    def __init__(self, tokenizer_path, stats, backbone_path=None, hidden=768,layers=12,heads=12):
        super().__init__()
        from ivideogpt.tokenizer import CompressiveVQModelFSQ
        from transformers import LlamaConfig, LlamaForCausalLM
        self.tokenizer=CompressiveVQModelFSQ.from_pretrained(tokenizer_path,local_files_only=True).eval()
        self.tokenizer.requires_grad_(False)
        if self.tokenizer.context_length!=1: raise ValueError('Require single-context tokenizer')
        self.dyn_vocab=self.tokenizer.num_dyn_embeddings
        self.ctx_vocab=self.tokenizer.num_vq_embeddings
        self.action_offset=self.dyn_vocab+self.ctx_vocab
        self.vocab=self.action_offset+256
        self.register_buffer('low',torch.tensor(stats.low,dtype=torch.float32))
        self.register_buffer('high',torch.tensor(stats.high,dtype=torch.float32))
        if backbone_path:
            self.backbone=LlamaForCausalLM.from_pretrained(backbone_path,local_files_only=True)
            if self.backbone.config.vocab_size < self.vocab:
                raise ValueError('WM vocabulary incompatible with tokenizer')
        else:
            self.backbone=LlamaForCausalLM(LlamaConfig(vocab_size=self.vocab,hidden_size=hidden,
                intermediate_size=4*hidden,num_hidden_layers=layers,num_attention_heads=heads,
                num_key_value_heads=heads,max_position_embeddings=4096,bos_token_id=None,eos_token_id=None))

    def train(self,mode=True):
        super().train(mode); self.tokenizer.eval(); return self

    def action_ids(self,deltas):
        # Source mechanism: train-set ranges, 256 bins; clipping is logged separately.
        v=((deltas-self.low)/(self.high-self.low).clamp_min(1e-6)).clamp(0,1)
        return (v*256).floor().long().clamp(0,255)+self.action_offset

    @torch.no_grad()
    def encode(self,video):
        # Duplicate current image as source initial dynamics token. Not a future observation.
        pixels=torch.cat([video[:,:1],video],1)
        c,d=self.tokenizer.tokenize(pixels)
        if c.shape[-1]!=1024 or d.shape[-1]!=64:
            raise ValueError('Released tokenizer detokenize hardcodes 32x32/8x8; require 256px matching tokenizer')
        return c.long(),d.long()

    def loss(self,video,deltas,valid):
        if not valid.all(): raise ValueError('Partial clips unsupported for autoregressive supervision')
        c,d=self.encode(video)
        seq=[c.flatten(1)+self.dyn_vocab,d[:,0]]
        labels=[torch.full_like(seq[0],-100),torch.full_like(seq[1],-100)]
        acts=self.action_ids(deltas)
        for t in range(8):
            seq.extend([acts[:,t],d[:,t+1]])
            labels.extend([torch.full_like(acts[:,t],-100),d[:,t+1]])
        ids=torch.cat(seq,1); lab=torch.cat(labels,1)
        # Full ground-truth prefix (teacher forcing); only future visual tokens are targets, Eq.3.
        out=self.backbone(input_ids=ids,labels=lab,use_cache=False)
        return out.loss

    @torch.no_grad()
    def rollout(self,initial,deltas,teacher_video=None):
        self.eval()
        c,d=self.encode(initial[:,None])
        prefix=torch.cat([c.flatten(1)+self.dyn_vocab,d[:,0]],1)
        acts=self.action_ids(deltas)
        truth=None if teacher_video is None else self.encode(teacher_video)[1]
        frames=[]
        # One action block was sampled before entering this function. No policy re-query.
        for t in range(deltas.shape[1]):
            prompt=torch.cat([prefix,acts[:,t]],1)
            generated=[]; cache=None
            for j in range(64):
                o=self.backbone(input_ids=prompt if j==0 else token,past_key_values=cache,use_cache=True)
                cache=o.past_key_values
                # Greedy decoding holds simulator sampling noise fixed across candidates.
                token=o.logits[:,-1,:self.dyn_vocab].argmax(-1,keepdim=True)
                generated.append(token)
            frame=torch.cat(generated,1); frames.append(frame)
            prefix=torch.cat([prompt,frame if truth is None else truth[:,t+1]],1)
        return self.tokenizer.detokenize(c,torch.stack(frames,1))[:,1:].clamp(0,1)
