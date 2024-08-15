"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from utils import *
from typing import Any, Dict
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import contextlib
from unimol_tools import UniMolRepr
from model.unimol_reaction import UniMolR
from unimol_tools.models import UniMolModel
import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.models.blip2_models.blip2 import (
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput
from model.blip2 import Blip2Base
from model.dist_funs import pl_concat_all_gather
# from transformers import BertModel, BertConfig
from lavis.models.blip2_models.Qformer import BertConfig, BertModel, BertLMHeadModel
from rxnfp.tokenization import SmilesTokenizer
from torchmetrics.regression import R2Score, MeanAbsoluteError, MeanSquaredError
from transformers.modeling_outputs import (
    ModelOutput,
)
from dataclasses import dataclass
from typing import Optional
import ipdb


@dataclass
class BlipOutputYield(ModelOutput):
    # some finetuned models (e.g. BlipVQA) do not compute similarity, thus optional.
    # sims: Optional[BlipSimilarity] = None

    # intermediate_output: BlipIntermediateOutput = None

    loss: Optional[torch.FloatTensor] = None

    loss_rmse: Optional[torch.FloatTensor] = None

    loss_mae: Optional[torch.FloatTensor] = None

    loss_r2: Optional[torch.FloatTensor] = None

    preds: Optional[torch.FloatTensor] = None

    labels: Optional[torch.FloatTensor] = None


class Blip2Stage1(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        # self.rerank_cand_num = args.rerank_cand_num
        self.blip2qformer = Blip2Qformer(args=args)
    
        self.save_hyperparameters(args)

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    @torch.no_grad()
    def validation_step(self, batch, batch_idx=-1):
        batch_size = batch[1]["input_ids"].shape[0]
        blip2_loss = self.blip2qformer.forward_v1(batch)
        ###============== Overall Loss ===================###
        self.log("val_loss_itc", float(blip2_loss.loss_itc), batch_size=batch_size, sync_dist=True)
        self.log("val_loss_itm", float(blip2_loss.loss_itm), batch_size=batch_size, sync_dist=True)
        self.log("val_loss_lm", float(blip2_loss.loss_lm), batch_size=batch_size, sync_dist=True)
        self.log("val_loss", float(blip2_loss.loss), batch_size=batch_size, sync_dist=True)

    def training_step(self, batch, batch_idx=-1):
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = batch[1]["input_ids"].shape[0]
        blip2_loss = self.blip2qformer.forward_v1(batch)
        ###============== Overall Loss ===================###
        self.log("train_loss_itc", float(blip2_loss.loss_itc), batch_size=batch_size, sync_dist=True)
        self.log("train_loss_itm", float(blip2_loss.loss_itm), batch_size=batch_size, sync_dist=True)
        self.log("train_loss_lm", float(blip2_loss.loss_lm), batch_size=batch_size, sync_dist=True)
        self.log("train_loss", float(blip2_loss.loss), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return blip2_loss.loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)


# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        temperature=0.1,
        args=None,
    ):
        super().__init__()
        self.args = args
        # self.tokenizer = self.init_tokenizer()

        # unimol_ = UniMolRepr(data_type='molecule', remove_hs=True, use_gpu=True)

        if args.use_3d:
            self.reaction_graph_encoder = UniMolR(UniMolModel(output_dim=1, data_type='molecule', remove_hs=True), args.cls)
            self.ln_graph = nn.LayerNorm(self.reaction_graph_encoder.model.args.encoder_embed_dim)
        else:
            self.reaction_graph_encoder, self.ln_graph = self.init_graph_encoder(args.gin_num_layers, args.gin_hidden_dim, args.gin_drop_ratio)

        for name, param in self.reaction_graph_encoder.named_parameters():
            param.requires_grad = False
        self.reaction_graph_encoder = self.reaction_graph_encoder.eval()
        self.reaction_graph_encoder.train = disabled_train
        logging.info("freeze graph encoder")
        
        # self.Qformer, self.query_tokens, self.graph_proj, self.text_proj = torch.load("/amax/data/group_0/3D-MoLM/3d.pt", map_location="cuda:0")
        self.Qformer, self.tokenizer, self.query_tokens = self.init_Qformer("bert_pretrained", args.num_query_token, 
                                                                            self.reaction_graph_encoder.model.args.encoder_embed_dim if args.use_3d else args.gin_hidden_dim, 
                                                                            args.cross_attention_freq)

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.graph_proj = nn.Linear(self.Qformer.config.hidden_size, args.embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, args.embed_dim)

        # self.yield_head = nn.Linear(self.Qformer.config.hidden_size, 1)
        self.gtm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temperature = temperature

    @classmethod
    def init_Qformer(self, model, num_query_token, graph_width, cross_attention_freq=2):
        model_path =  os.path.join("/etc/anaconda3/envs/yield_pred/lib/python3.9/site-packages", 
                                   "rxnfp", f"models/transformers/{model}")

        tokenizer_vocab_path = os.path.join("/etc/anaconda3/envs/yield_pred/lib/python3.9/site-packages", 
                                            "rxnfp", f"models/transformers/{model}/vocab.txt")
        encoder_config = BertConfig.from_pretrained(model_path)
        encoder_config.encoder_width = graph_width
        # insert cross-attention layer every other block
        encoder_config.is_decoder = True
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        model = BertLMHeadModel.from_pretrained(model_path, config=encoder_config)
        logging.info(f"load from {model_path}")

        tokenizer = SmilesTokenizer(
            tokenizer_vocab_path
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return model, tokenizer, query_tokens
    
    def contrast(self, features_graph, features_text, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        '''
        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B]

        logits_per_graph = sim_g2t / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph, logits_per_text, loss
        else:
            return loss

    def contrast_global(self, features_graph, features_text, features_graph_all, features_text_all, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        features_text_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, num_qs, D]
        '''
        bs = features_graph.size(0)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text_all.unsqueeze(-1)).squeeze(dim=-1) # shape = [B, 1, num_qs, D]; shape = [B * num_gpus, D, 1]; output shape = [B, B * num_gpus, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B * num_gpus]

        logits_per_graph = sim_g2t / self.temperature
    
        sim_t2q = (features_text.unsqueeze(1).unsqueeze(1) @ features_graph_all.permute(0, 2, 1)).squeeze(dim=-2) # shape = [B, 1, 1, D]; [B*num_gpus, D, num_qs]; output shape = [B, B*num_gpus, 1, num_qs]
        sim_t2g, _ = sim_t2q.max(-1)
        logits_per_text = sim_t2g / self.temperature

        # labels = torch.arange(bs, dtype=torch.long, device=self.device)
        rank = dist.get_rank()
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph[:, rank*bs:rank*bs+bs], logits_per_text[:, rank*bs:rank*bs+bs], loss
        else:
            return loss

    def graph_forward(self, graph):
        if self.args.use_3d:
            batch_node, batch_mask = self.graph_encoder(*graph)
        else:
            batch_node, batch_mask = self.graph_encoder(graph)
        batch_node = self.ln_graph(batch_node)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=False,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        graph_feats = F.normalize(graph_feats, p=2, dim=-1)
        return graph_feats, batch_node, batch_mask

    def text_forward(self, text, mask):
        text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :] )
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        return text_feats
    
    def compute_gtm(self, batch_node, batch_mask, text_ids, text_atts):
        '''
        batch_node shape = [B, N, D]
        batch_mask shape = [B, N]
        text_ids shape = [B, N]
        text_atts shape = [B, N]
        '''
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1) # shape = [B, Nq, D]
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            batch_node.device
        ) # shape = [B, Nq]
        attention_mask = torch.cat([query_atts, text_atts], dim=1) # shape = [B, Nq + N]
        output_gtm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask,
            return_dict=True,
        )
        gl_embeddings = output_gtm.last_hidden_state[:, : query_tokens.size(1), :] # shape = [B, Nq, D]
        gtm_logit = self.gtm_head(gl_embeddings).mean(dim=1) # shape = [B, Nq, 2]
        # gtm_logit = F.softmax(gtm_logit, dim=-1)[:, 1] # select the axis of the positive class
        gtm_logit = gtm_logit[:, 1] # select the axis of the positive class
        return gtm_logit

    def forward_v1(self, batch):
        ## v2: gather results from all gpus
        ###============== Image-text Contrastive ===================###
        (conf, nums), text_tokens, labels_y = batch
        text = text_tokens['input_ids']
        mask = text_tokens['attention_mask']
        # unimol .cpu() the results emmm
        batch_node, batch_mask = self.reaction_graph_encoder(conf, nums)
        batch_node = self.ln_graph(batch_node)
        batch_node, batch_mask = batch_node.to(text.device), batch_mask.to(text.device)
        # batch_node = batch_node.detach()
        batch_size = batch_node.shape[0]

        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=True,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :])
        
        text_feats, graph_feats = F.normalize(text_feats, p=2, dim=-1), F.normalize(graph_feats, p=2, dim=-1)
        text_feats_all, graph_feats_all = pl_concat_all_gather(text_feats), pl_concat_all_gather(graph_feats) # shape = [B * num_gpus, D]
        sim_g2t, sim_t2g, loss_gtc = self.contrast_global(graph_feats, text_feats, graph_feats_all, text_feats_all, return_sim=True)


        ###============== Image-text Matching ===================###
        loss_gtm = 0
        if self.args.gtm:
            ## not aggregate global tensor because of their different shapes
            g_emb_world = batch_node
            g_mask_world = batch_mask
            text_ids_world = text
            text_mask_world = mask
            with torch.no_grad():
                weights_t2g = F.softmax(sim_t2g, dim=1) + 1e-4
                weights_t2g.fill_diagonal_(0)
                weights_g2t = F.softmax(sim_g2t, dim=1) + 1e-4
                weights_g2t.fill_diagonal_(0)

            # select a negative graph for each text
            graph_embeds_neg = []
            graph_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(g_emb_world[neg_idx])
                graph_mask_neg.append(g_mask_world[neg_idx])
            
            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
            graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

            # select a negative text for each image
            text_ids_neg = []
            text_atts_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(text_ids_world[neg_idx])
                text_atts_neg.append(text_mask_world[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat(
                [text, text, text_ids_neg], dim=0
            )  # pos, pos, neg
            text_atts_all = torch.cat(
                [mask, mask, text_atts_neg],
                dim=0,
            )

            query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
            query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long, device=text.device)
            attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

            graph_embeds_all = torch.cat([batch_node, graph_embeds_neg, batch_node], dim=0)  # pos, neg, pos
            graph_atts_all = torch.cat([batch_mask, graph_mask_neg, batch_mask], dim=0)

            output_itm = self.Qformer.bert(
                text_ids_all,
                query_embeds=query_tokens_itm,
                attention_mask=attention_mask_all,
                encoder_hidden_states=graph_embeds_all,
                encoder_attention_mask=graph_atts_all,
                return_dict=True,
            )

            vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :] # keep query tokens only
            vl_output = self.gtm_head(vl_embeddings)
            logits = vl_output.mean(dim=1)

            itm_labels = torch.cat(
                [torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)],
                dim=0,
            ).to(text.device)
            loss_gtm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        loss_lm = 0
        if self.args.lm:
            decoder_input_ids = text.clone()
            decoder_input_ids[:, 0] = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.cls_token_id
            labels = decoder_input_ids.masked_fill(
                decoder_input_ids == self.tokenizer.pad_token_id, -100
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=text.device)
            
            attention_mask = torch.cat([query_atts, mask], dim=1)
            lm_output = self.Qformer(
                decoder_input_ids,
                attention_mask=attention_mask,
                past_key_values=query_output.past_key_values,
                return_dict=True,
                labels=labels,
            )

            loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_gtc + loss_gtm + loss_lm,
            loss_itc=loss_gtc,
            loss_itm=loss_gtm,
            loss_lm=loss_lm,
        )
    

class Blip2Stage1_ft(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
        self.args = args
        # self.rerank_cand_num = args.rerank_cand_num
        self.blip2qformer = Blip2Qformer_ft(args=args)
    
        self.save_hyperparameters(args)

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    @torch.no_grad()
    def validation_step(self, batch, batch_idx=-1):
        batch_size = batch[1]["input_ids"].shape[0]
        blip2_loss = self.blip2qformer.forward_v1(batch)
        ###============== Overall Loss ===================###
        self.log("val_loss_rmse", float(blip2_loss.loss_rmse), batch_size=batch_size, sync_dist=True)
        self.log("val_loss_mae", float(blip2_loss.loss_mae), batch_size=batch_size, sync_dist=True)
        self.log("val_loss_r2", float(blip2_loss.loss_r2), batch_size=batch_size, sync_dist=True)
        self.log("val_loss", float(blip2_loss.loss), batch_size=batch_size, sync_dist=True)

    def training_step(self, batch, batch_idx=-1):
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = batch[1]["input_ids"].shape[0]
        blip2_loss = self.blip2qformer.forward_v1(batch)
        ###============== Overall Loss ===================###
        self.log("train_loss_rmse", float(blip2_loss.loss_rmse), batch_size=batch_size, sync_dist=True)
        self.log("train_loss_mae", float(blip2_loss.loss_mae), batch_size=batch_size, sync_dist=True)
        self.log("train_loss_r2", float(blip2_loss.loss_r2), batch_size=batch_size, sync_dist=True)
        self.log("train_loss", float(blip2_loss.loss), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return blip2_loss.loss
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def forward(self, batch):
        return self.blip2qformer.forward_v1(batch)

# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2Qformer_ft(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        temperature=0.1,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mae = MeanAbsoluteError()
        self.r2 = R2Score()
        # self.tokenizer = self.init_tokenizer()

        # unimol_ = UniMolRepr(data_type='molecule', remove_hs=True, use_gpu=True)

        if args.use_3d:
            self.reaction_graph_encoder = UniMolR(UniMolModel(output_dim=1, data_type='molecule', remove_hs=True), args.cls)
            self.ln_graph = nn.LayerNorm(self.reaction_graph_encoder.model.args.encoder_embed_dim)
        else:
            self.reaction_graph_encoder, self.ln_graph = self.init_graph_encoder(args.gin_num_layers, args.gin_hidden_dim, args.gin_drop_ratio)

        for name, param in self.reaction_graph_encoder.named_parameters():
            param.requires_grad = False
        self.reaction_graph_encoder = self.reaction_graph_encoder.eval()
        self.reaction_graph_encoder.train = disabled_train
        logging.info("freeze graph encoder")
        
        # self.Qformer, self.query_tokens, self.graph_proj, self.text_proj = torch.load("/amax/data/group_0/3D-MoLM/3d.pt", map_location="cuda:0")
        self.Qformer, self.tokenizer, self.query_tokens = self.init_Qformer("bert_pretrained", args.num_query_token, 
                                                                            self.reaction_graph_encoder.model.args.encoder_embed_dim if args.use_3d else args.gin_hidden_dim, 
                                                                            args.cross_attention_freq)

        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.graph_proj = nn.Linear(self.Qformer.config.hidden_size, args.embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, args.embed_dim)

        # self.yield_head = nn.Linear(self.Qformer.config.hidden_size, 1)
        self.yield_head = nn.Sequential(
            nn.Linear(self.Qformer.config.hidden_size*2 if args.ft_type == "merge" else self.Qformer.config.hidden_size, args.predict_hidden_size), 
            nn.PReLU(), nn.Dropout(args.dropout),
            nn.Linear(args.predict_hidden_size, 1)
        )

        self.temperature = temperature

    @classmethod
    def init_Qformer(self, model, num_query_token, graph_width, cross_attention_freq=2):
        model_path =  os.path.join("/etc/anaconda3/envs/yield_pred/lib/python3.9/site-packages", 
                                   "rxnfp", f"models/transformers/{model}")

        tokenizer_vocab_path = os.path.join("/etc/anaconda3/envs/yield_pred/lib/python3.9/site-packages", 
                                            "rxnfp", f"models/transformers/{model}/vocab.txt")
        encoder_config = BertConfig.from_pretrained(model_path)
        encoder_config.encoder_width = graph_width
        # insert cross-attention layer every other block
        encoder_config.is_decoder = True
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token

        model = BertLMHeadModel.from_pretrained(model_path, config=encoder_config)

        tokenizer = SmilesTokenizer(
            tokenizer_vocab_path
        )
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return model, tokenizer, query_tokens
    
    def contrast(self, features_graph, features_text, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        '''
        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B]

        logits_per_graph = sim_g2t / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph, logits_per_text, loss
        else:
            return loss

    def contrast_global(self, features_graph, features_text, features_graph_all, features_text_all, return_sim=False):
        '''
        features_graph: shape = [B, num_qs, D]
        features_text: shape = [B, D]
        features_text_all: shape = [B * num_gpus, D]
        features_graph_all: shape = [B * num_gpus, num_qs, D]
        '''
        bs = features_graph.size(0)

        # cosine similarity as logits
        sim_q2t = (features_graph.unsqueeze(1) @ features_text_all.unsqueeze(-1)).squeeze(dim=-1) # shape = [B, 1, num_qs, D]; shape = [B * num_gpus, D, 1]; output shape = [B, B * num_gpus, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B * num_gpus]

        logits_per_graph = sim_g2t / self.temperature
    
        sim_t2q = (features_text.unsqueeze(1).unsqueeze(1) @ features_graph_all.permute(0, 2, 1)).squeeze(dim=-2) # shape = [B, 1, 1, D]; [B*num_gpus, D, num_qs]; output shape = [B, B*num_gpus, 1, num_qs]
        sim_t2g, _ = sim_t2q.max(-1)
        logits_per_text = sim_t2g / self.temperature

        # labels = torch.arange(bs, dtype=torch.long, device=self.device)
        rank = dist.get_rank()
        labels = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(self.device)

        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        if return_sim:
            return logits_per_graph[:, rank*bs:rank*bs+bs], logits_per_text[:, rank*bs:rank*bs+bs], loss
        else:
            return loss

    def graph_forward(self, conf, nums):
        batch_node, batch_mask = self.reaction_graph_encoder(conf, nums)
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            use_cache=False,
            return_dict=True,
        )
        graph_feats = self.graph_proj(query_output.last_hidden_state) # shape = [B, num_q, D]
        graph_feats = F.normalize(graph_feats, p=2, dim=-1)
        return graph_feats, batch_node, batch_mask

    def text_forward(self, text, mask):
        text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
        text_feats = self.text_proj(text_output.last_hidden_state[:, 0, :] )
        text_feats = F.normalize(text_feats, dim=-1, p=2)
        return text_feats
    
    def compute_gtm(self, batch_node, batch_mask, text_ids, text_atts):
        '''
        batch_node shape = [B, N, D]
        batch_mask shape = [B, N]
        text_ids shape = [B, N]
        text_atts shape = [B, N]
        '''
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1) # shape = [B, Nq, D]
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            batch_node.device
        ) # shape = [B, Nq]
        attention_mask = torch.cat([query_atts, text_atts], dim=1) # shape = [B, Nq + N]
        output_gtm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask,
            return_dict=True,
        )
        gl_embeddings = output_gtm.last_hidden_state[:, : query_tokens.size(1), :] # shape = [B, Nq, D]
        gtm_logit = self.gtm_head(gl_embeddings).mean(dim=1) # shape = [B, Nq, 2]
        # gtm_logit = F.softmax(gtm_logit, dim=-1)[:, 1] # select the axis of the positive class
        gtm_logit = gtm_logit[:, 1] # select the axis of the positive class
        return gtm_logit

    def forward(self, batch):
        return self.forward_v1(batch)

    def forward_v1(self, batch):
        ## v2: gather results from all gpus
        ###============== Image-text Contrastive ===================###
        (conf, nums), text_tokens, labels_y = batch
        text = text_tokens['input_ids']
        mask = text_tokens['attention_mask']
        # unimol .cpu() the results emmm
        batch_node, batch_mask = self.reaction_graph_encoder(conf, nums)
        batch_node = self.ln_graph(batch_node)
        # batch_node, batch_mask = batch_node.to(text.device), batch_mask.to(text.device)
        # batch_node = batch_node.detach()
        batch_size = batch_node.shape[0]

        if self.args.ft_type == "merge":
            """may we use concat?"""
            # query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
            # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=text.device)
            # attention_mask_together = torch.cat([query_atts, mask], dim=1)
            # query_output = self.Qformer.bert(
            #     text,
            #     attention_mask=attention_mask_together,
            #     query_embeds=query_tokens,
            #     encoder_hidden_states=batch_node,
            #     encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            #     use_cache=True,
            #     return_dict=True,
            # )
            # vl_embeddings = query_output.last_hidden_state[:, :query_tokens.size(1), :] # keep query tokens only

            query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=batch_node,
                encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
                use_cache=False,
                return_dict=True,
            )
            vl_query_embeddings = query_output.last_hidden_state[:, :query_tokens.size(1), :] # keep query tokens only

            text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
            vl_text_embeddings = text_output.last_hidden_state[:, 0, :] # keep cls tokens only
            # print(vl_query_embeddings.shape, vl_text_embeddings.shape) # query, B, N_k, D, text, B, D
            vl_text_embeddings = vl_text_embeddings.unsqueeze(1)
            vl_text_embeddings = vl_text_embeddings.repeat(1, vl_query_embeddings.shape[1], 1)
            vl_embeddings = torch.cat([vl_query_embeddings, vl_text_embeddings], dim=-1)

            vl_output = self.yield_head(vl_embeddings)
            logits = vl_output.mean(dim=1)
        elif self.args.ft_type == "conformer":
            query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=batch_node,
                encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
                use_cache=False,
                return_dict=True,
                output_attentions=True
            )
            vl_embeddings = query_output.last_hidden_state[:, :query_tokens.size(1), :] # keep query tokens only
            vl_output = self.yield_head(vl_embeddings)
            logits = vl_output.mean(dim=1)
        else:
            text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
            vl_embeddings = text_output.last_hidden_state[:, 0, :] # keep cls tokens only
            logits = self.yield_head(vl_embeddings)

        loss_y = F.mse_loss(logits, labels_y)
        loss_rmse = torch.sqrt(loss_y)
        loss_mae = self.mae(logits, labels_y)
        loss_r2 = self.r2(logits, labels_y)
        # maybe we should cut to 0-1 after loss?
        logits = torch.clamp(logits, 0., 1.)

        return BlipOutputYield(
            loss=loss_y,
            loss_rmse=loss_rmse,
            loss_mae=loss_mae,
            loss_r2=loss_r2,
            preds=logits,
            labels=labels_y
        )
        
    def get_attention(self, batch):
        ## v2: gather results from all gpus
        ###============== Image-text Contrastive ===================###
        (conf, nums), text_tokens, labels_y = batch
        text = text_tokens['input_ids']
        mask = text_tokens['attention_mask']
        # unimol .cpu() the results emmm
        batch_node, batch_mask = self.reaction_graph_encoder(conf, nums)
        # batch_node, batch_mask = batch_node.to(text.device), batch_mask.to(text.device)
        # batch_node = batch_node.detach()
        batch_size = batch_node.shape[0]

        if self.args.ft_type == "merge":
            """may we use concat?"""
            # query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
            # query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=text.device)
            # attention_mask_together = torch.cat([query_atts, mask], dim=1)
            # query_output = self.Qformer.bert(
            #     text,
            #     attention_mask=attention_mask_together,
            #     query_embeds=query_tokens,
            #     encoder_hidden_states=batch_node,
            #     encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            #     use_cache=True,
            #     return_dict=True,
            # )
            # vl_embeddings = query_output.last_hidden_state[:, :query_tokens.size(1), :] # keep query tokens only

            query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=batch_node,
                encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
                use_cache=False,
                return_dict=True,
            )
            vl_query_embeddings = query_output.last_hidden_state[:, :query_tokens.size(1), :] # keep query tokens only

            text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
            vl_text_embeddings = text_output.last_hidden_state[:, 0, :] # keep cls tokens only
            # print(vl_query_embeddings.shape, vl_text_embeddings.shape) # query, B, N_k, D, text, B, D
            vl_text_embeddings = vl_text_embeddings.unsqueeze(1)
            vl_text_embeddings = vl_text_embeddings.repeat(1, vl_query_embeddings.shape[1], 1)
            vl_embeddings = torch.cat([vl_query_embeddings, vl_text_embeddings], dim=-1)

            vl_output = self.yield_head(vl_embeddings)
            logits = vl_output.mean(dim=1)
        elif self.args.ft_type == "conformer":
            query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=batch_node,
                encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
                use_cache=False,
                return_dict=True,
                output_attentions=True
            )
            vl_embeddings = query_output.last_hidden_state[:, :query_tokens.size(1), :] # keep query tokens only
            vl_output = self.yield_head(vl_embeddings)
            logits = vl_output.mean(dim=1)
        else:
            text_output = self.Qformer.bert(text, attention_mask=mask, return_dict=True) # shape = [B, n_max, D]
            vl_embeddings = text_output.last_hidden_state[:, 0, :] # keep cls tokens only
            logits = self.yield_head(vl_embeddings)

        loss_y = F.mse_loss(logits, labels_y)
        loss_rmse = torch.sqrt(loss_y)
        loss_mae = self.mae(logits, labels_y)
        loss_r2 = self.r2(logits, labels_y)
        # maybe we should cut to 0-1 after loss?
        logits = torch.clamp(logits, 0., 1.)

        return BlipOutputYield(
            loss=loss_y,
            loss_rmse=loss_rmse,
            loss_mae=loss_mae,
            loss_r2=loss_r2,
            preds=logits,
            labels=labels_y
        ),query_output.cross_attentions
    