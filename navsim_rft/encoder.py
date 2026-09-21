"""Use the supplied Prismatic vision backbone/projector/language model directly.
No robot action token IDs, target actions, or future images enter this encoder.
"""
from pathlib import Path
import torch
from torch import nn

class PrismaticEncoder(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        if not Path(checkpoint).is_dir():
            raise FileNotFoundError(f'Local Prismatic/OpenVLA checkpoint required: {checkpoint}')
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.processing_prismatic import PrismaticProcessor, PrismaticImageProcessor
        from transformers import AutoTokenizer
        config=OpenVLAConfig.from_pretrained(checkpoint,local_files_only=True)
        self.model, self.loading_info=OpenVLAForActionPrediction.from_pretrained(checkpoint,config=config,
            torch_dtype=torch.float32,attn_implementation='eager',local_files_only=True,output_loading_info=True)
        missing=[k for k in self.loading_info.get('missing_keys',[]) if not k.startswith('action_queries.')]
        if missing or self.loading_info.get('mismatched_keys') or self.loading_info.get('error_msgs'):
            raise ValueError(f'Incomplete/incompatible VLM checkpoint: {self.loading_info}')
        self.processor=PrismaticProcessor(PrismaticImageProcessor.from_pretrained(checkpoint,local_files_only=True),
            AutoTokenizer.from_pretrained(checkpoint,local_files_only=True))
        self.model.vision_backbone.set_num_images_in_input(1)
        self.dim=self.model.llm_dim
        # Same learned-query mechanism, now 8 poses x 3 dimensions. Robot query count is not reused.
        self.queries=nn.Parameter(torch.zeros(24,self.dim,dtype=torch.float32))
        self.model.requires_grad_(False)
        self.model.eval()
        self.trainable=False

    def set_trainable(self, enabled):
        self.trainable=enabled
        self.model.requires_grad_(enabled)
        # Robot action queries / LM output head do not participate in this encoding path.
        self.model.action_queries.requires_grad_(False)
        self.model.language_model.get_output_embeddings().requires_grad_(False)
        self.queries.requires_grad_(enabled)

    def forward(self, images, texts):
        device=self.queries.device
        outputs=[]
        # Per scene processing avoids variable prompt padding contaminating action context.
        for image,text in zip(images,texts):
            batch=self.processor(text=f'In: {text}\nOut:',images=image,return_tensors='pt')
            ids=batch['input_ids'].to(device)
            pixel=batch['pixel_values'].to(device,dtype=torch.float32)
            emb=self.model.get_input_embeddings()(ids)
            visual=self.model._process_vision_features(pixel,use_film=False)
            combined=torch.cat([emb[:,:1],visual,emb[:,1:],self.queries[None]],dim=1)
            out=self.model.language_model(inputs_embeds=combined,attention_mask=torch.ones(combined.shape[:2],device=device),
                output_hidden_states=True,use_cache=False,return_dict=True)
            last=out.hidden_states[-1]
            outputs.append(torch.cat([last[:,1:1+visual.shape[1]],last[:,-24:]],1))
        return torch.cat(outputs).float().unsqueeze(1)

    def train(self, mode=True):
        # Disable stochastic dropout for exact probability replay; gradients remain enabled.
        super().train(False)
        return self
