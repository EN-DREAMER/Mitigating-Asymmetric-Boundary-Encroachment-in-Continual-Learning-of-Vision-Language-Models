import os
import sys
import datetime
import math
import copy
import numpy as np
from tqdm import tqdm
from itertools import chain
import json  

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision import transforms

from clip import clip
from scenario_datasets.build_functions import build_cur_task_data_loader

import argparse
import pandas as pd


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a") 

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_dataset_template(dataset_name, is_seen=True):
    return lambda c: f"a photo of a {c}."

def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()
    model = clip.build_model(state_dict or model.state_dict())
    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        model.float()
    return model

class VisualAdapter(nn.Module):
    def __init__(self, in_dim, alpha=0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, in_dim, bias=False)
        self.alpha = alpha
    def forward(self, x):
        cls_token, patch_tokens = x[0:1, :, :], x[1:, :, :]    
        cls_token = cls_token + self.alpha * self.linear(cls_token)
        return torch.cat([cls_token, patch_tokens], dim=0)

class AdaptFormer(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype))
    def forward(self, x):
        return self.up_proj(self.relu(self.down_proj(self.ln(x)))) * self.scale

class TextConstraintManager:
    def __init__(self):
        self.old_class_info = []
        self.cached_similarities = {}

class Text_Tuner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        n_layers = len(clip_model.transformer.resblocks)
        emb_dim = clip_model.ln_final.weight.shape[0]
        self.adaptformer_list = nn.ModuleList([
            AdaptFormer(in_dim=emb_dim, bottle_dim=cfg.T_ADAPTER_DIM, dtype=clip_model.dtype) 
            if getattr(cfg, 'T_ADAPTFORMER', True) else None for _ in range(n_layers)
        ])
        self.constraint_manager = TextConstraintManager()

class Peft_Text(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.blocks = clip_model.transformer.resblocks
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text, tuner: Text_Tuner=None):
        x = self.token_embedding(text).to(self.dtype) + self.positional_embedding.type(self.dtype)
        _bsz, _seq_len, _emb_dim = x.shape
        for i, block in enumerate(self.blocks):
            adaptformer = tuner.adaptformer_list[i] if tuner is not None else None
            x = x.permute(1, 0, 2)
            identity = x
            x = block.ln_1(x)
            q, k, v = F.linear(x, block.attn.in_proj_weight, block.attn.in_proj_bias).chunk(3, dim=-1)
            q = q.contiguous().view(q.shape[0], q.shape[1] * block.attn.num_heads, _emb_dim // block.attn.num_heads).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * block.attn.num_heads, _emb_dim // block.attn.num_heads).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * block.attn.num_heads, _emb_dim // block.attn.num_heads).transpose(0, 1)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=block.attn_mask.to(dtype=x.dtype, device=x.device) if block.attn_mask is not None else None)
            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)
            x = F.linear(x, block.attn.out_proj.weight, block.attn.out_proj.bias).view(_seq_len, _bsz, _emb_dim) + identity
            identity = x
            x = block.ln_2(x)
            x = block.mlp[2](block.mlp[1](block.mlp[0](x)))
            if adaptformer is not None: x = x + 0.1 * adaptformer(identity)
            x = (x + identity).permute(1, 0, 2)
        x = self.ln_final(x).to(self.dtype)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

class ContinualCLIP(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.cfg = cfg
        self.dtype = clip_model.dtype
        self.image_encoder = clip_model.visual
        self.visual_adapters = nn.ModuleList()
        self.frozen_visual_adapters = nn.ModuleList()
        
        adapter_indices = {int(name.split('.')[-1]) for name in cfg.VISUAL_ADAPTER_LAYERS if "resblocks" in name}
        for i in range(len(self.image_encoder.transformer.resblocks)):
            if i in adapter_indices:
                self.visual_adapters.append(VisualAdapter(cfg.VISUAL_ADAPTER_DIM, alpha=cfg.VISUAL_ALPHA).to(self.dtype))
                self.frozen_visual_adapters.append(VisualAdapter(cfg.VISUAL_ADAPTER_DIM, alpha=cfg.VISUAL_ALPHA).to(self.dtype))
            else:
                self.visual_adapters.append(None)
                self.frozen_visual_adapters.append(None)
                
        self.text_encoder = Peft_Text(clip_model)
        self.text_tuner = Text_Tuner(cfg, clip_model)

    def encode_image(self, image, skip_adapters=False):
        x = self.image_encoder.conv1(image.type(self.dtype)).reshape(image.shape[0], self.image_encoder.conv1.out_channels, -1).permute(0, 2, 1)
        x = torch.cat([self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x).permute(1, 0, 2)

        for i, block in enumerate(self.image_encoder.transformer.resblocks):
            x = block(x)
            if not skip_adapters:
                cls_token, patch_tokens = x[0:1, :, :], x[1:, :, :]
                total_update = 0
                applied = False
                if self.frozen_visual_adapters[i] is not None:
                    total_update += self.frozen_visual_adapters[i].linear(cls_token)
                    applied = True
                if self.visual_adapters[i] is not None:
                    total_update += self.visual_adapters[i].linear(cls_token)
                    applied = True
                if applied:
                    x = torch.cat([cls_token + self.cfg.VISUAL_ALPHA * total_update, patch_tokens], dim=0)
        x = self.image_encoder.ln_post(x.permute(1, 0, 2)[:, 0, :])
        return x @ self.image_encoder.proj if self.image_encoder.proj is not None else x

    def encode_text(self, text, skip_tuner=False):
        return self.text_encoder(text, tuner=None if skip_tuner else self.text_tuner)

    def extract_concat_features(self, images, target_layers=None):
        if target_layers is None:
            target_layers = self.cfg.VISUAL_GRAM_SELECT_LAYERS

        extracted_feats = {}
        x = self.image_encoder.conv1(images.type(self.dtype)).reshape(images.shape[0], self.image_encoder.conv1.out_channels, -1).permute(0, 2, 1)
        x = torch.cat([self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x).permute(1, 0, 2)
        
        for i, block in enumerate(self.image_encoder.transformer.resblocks):
            x = block(x)
            name = f"visual.transformer.resblocks.{i}"
            
            if name in target_layers: 
                extracted_feats[name] = x[0, :, :].clone()
            
            cls_token, patch_tokens = x[0:1, :, :], x[1:, :, :]
            total_update = 0
            if self.frozen_visual_adapters[i]: total_update += self.frozen_visual_adapters[i].linear(cls_token)
            if self.visual_adapters[i]: total_update += self.visual_adapters[i].linear(cls_token)
            if self.frozen_visual_adapters[i] or self.visual_adapters[i]:
                x = torch.cat([cls_token + self.cfg.VISUAL_ALPHA * total_update, patch_tokens], dim=0)

        if "visual" in target_layers:
            x_final = self.image_encoder.ln_post(x.permute(1, 0, 2)[:, 0, :])
            if self.image_encoder.proj is not None: x_final = x_final @ self.image_encoder.proj
            extracted_feats["visual"] = x_final
            
        return torch.cat([extracted_feats[name] for name in target_layers], dim=1)


class LayUPEvaluator:
    def __init__(self, G, c, ridge_lambda, device="cuda"):
        self.regularized_G = G + ridge_lambda * torch.eye(G.size(0), device=G.device)
        
        try: 
            self.G_inv = torch.linalg.inv(self.regularized_G)
            W_temp = torch.linalg.solve(self.regularized_G, c)
        except torch.linalg.LinAlgError: 
            self.G_inv = torch.linalg.pinv(self.regularized_G)
            W_temp = self.G_inv @ c
            
        self.weights = W_temp.T
        self.normalized_weights = F.normalize(self.weights, p=2, dim=1) if self.weights.shape[0] > 0 else self.weights

        if G.shape[0] > 0:
            self.G_scale = self.regularized_G.trace() / G.size(0)
        else:
            self.G_scale = 1.0

    def predict(self, features):
        if self.weights.shape[0] == 0: return torch.zeros(features.shape[0], 0, device=features.device)
        return F.normalize(features, p=2, dim=1) @ self.normalized_weights.T

    def get_ridge_confidence(self, features):
        if self.G_inv.shape[0] == 0:
            return torch.zeros(features.size(0), device=features.device)
        
        features_norm = F.normalize(features, p=2, dim=1)
        raw_m_dist = ((features_norm @ self.G_inv) * features_norm).sum(dim=1)
        
        scaled_dist = raw_m_dist * self.G_scale 
        
        confidence = torch.exp(-scaled_dist / 2.0)
        
        return confidence

class ADBManager:
    def __init__(self): self.class_statistics = {}

class VisualGPMManager:
    def __init__(self): self.bases_history = []

class GlobalInferenceManager:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.global_G = None  
        self.global_c = None  
        self.seen_tasks = [] 
        self.total_seen_classes = 0
        self.all_cached_text_features = [] 
        self.task_adapters_history = []

    def apply_summed_adapters(self, model):
        if not self.task_adapters_history: return
        summed_state = copy.deepcopy(model.visual_adapters.state_dict())
        for key in summed_state: summed_state[key].zero_()
        
        print(f"   [Diagnostic] Loading {len(self.task_adapters_history)} tasks from History...")
        for idx, task_state in enumerate(self.task_adapters_history):
            task_norm_sq = sum(val.float().pow(2).sum().item() for val in task_state.values())
            task_norm = task_norm_sq ** 0.5
            print(f"      -> History Task {idx} (Alone) Adapter Norm: {task_norm:.6f}")
            
            for key, val in task_state.items():
                if key in summed_state: summed_state[key] += val.to(self.device)
        
        model.visual_adapters.load_state_dict(summed_state)

    def evaluate_all_seen_tasks(self, model, dataset_root, preprocess, task_sequence):
        print("\n" + "="*100)
        print(">>> Starting Evaluation on ALL Datasets (Seen + Unseen)...")
        print("="*100)
        
        for p in model.frozen_visual_adapters.parameters():
            p.data.zero_()
            p.requires_grad = False
        
        self.apply_summed_adapters(model)

        total_adapter_norm_sq = sum(p.float().pow(2).sum().item() for p in model.visual_adapters.parameters())
        total_adapter_norm = total_adapter_norm_sq ** 0.5
        print(f"   [Diagnostic] Summed Visual Adapters L2 Norm: {total_adapter_norm:.6f}\n")

        if len(self.all_cached_text_features) == 0:
            print("Warning: No cached text features found.")
            return

        W_seen_cached = F.normalize(torch.cat(self.all_cached_text_features, dim=0), dim=-1)
        num_seen_classes = W_seen_cached.shape[0]

        all_prompts = []
        dataset_prompts_dict = {}
        dataset_global_offsets = {} 
        seen_task_names = {t['name'] for t in self.seen_tasks}
        
        current_offset = 0
        for t_info in self.seen_tasks:
            t_name = t_info['name']
            dataset_global_offsets[t_name] = current_offset
            template_fn = get_dataset_template(t_name, is_seen=True)
            prompts = [template_fn(c.replace("_", " ")) for c in t_info['class_names']]
            all_prompts.extend([template_fn(c.replace("_", " ")) for c in t_info['class_names']])
            dataset_prompts_dict[t_name] = prompts
            current_offset += len(t_info['class_names'])

        transform_test = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        num_unseen_classes = 0
        for t_name in task_sequence.keys():
            if t_name not in seen_task_names:
                dataset_global_offsets[t_name] = current_offset
                _, _, _, class_names = build_cur_task_data_loader(
                    root=dataset_root, dataset_name=t_name, transform_train=transform_test, 
                    transform_test=transform_test, num_shots=-1, batch_size=1, num_workers=0
                )
                template_fn = get_dataset_template(t_name, is_seen=False)
                prompts = [template_fn(c.replace("_", " ")) for c in class_names]
                all_prompts.extend([template_fn(c.replace("_", " ")) for c in class_names])
                dataset_prompts_dict[t_name] = prompts
                current_offset += len(class_names)
                num_unseen_classes += len(class_names)

        print(f"Total Classes for Eval: {len(all_prompts)} (Seen: {num_seen_classes}, Unseen: {num_unseen_classes})")          
        
        W_all_orig_list = []
        batch_size_text = 1000
        with torch.no_grad():
            for i in range(0, len(all_prompts), batch_size_text):
                batch_prompts = all_prompts[i : i + batch_size_text]
                tokens = clip.tokenize(batch_prompts).to(self.device)
                with autocast():
                    features = model.encode_text(tokens, skip_tuner=True)
                W_all_orig_list.append(F.normalize(features.float(), dim=-1))
                
        W_all_orig = torch.cat(W_all_orig_list, dim=0)
        W_seen_orig, W_unseen_orig = W_all_orig[:num_seen_classes], W_all_orig[num_seen_classes:]

        train_layers = self.cfg.VISUAL_GRAM_SELECT_LAYERS
        eval_layers = getattr(self.cfg, 'EVAL_GRAM_SELECT_LAYERS', train_layers)

        layer_start_idx = {}
        layer_dims = {}
        curr_idx = 0
        
        for name in train_layers:
            dim = 512 if name == "visual" else 768
            layer_start_idx[name] = curr_idx
            layer_dims[name] = dim
            curr_idx += dim
            
        indices_to_keep = []
        for name in eval_layers:
            if name not in layer_start_idx:
                raise ValueError(f"The selected layer {name} was not saved in VISUAL_GRAM_SELECT_LAYERS during training!")
            start = layer_start_idx[name]
            dim = layer_dims[name]
            indices_to_keep.extend(range(start, start + dim))
            
        indices_to_keep = torch.tensor(indices_to_keep, device=self.device, dtype=torch.long)
        
        G_eval = self.global_G.index_select(0, indices_to_keep).index_select(1, indices_to_keep)
        c_eval = self.global_c.index_select(0, indices_to_keep)
        
        print(f"   [Gram Slicing] Evaluated Layers: {eval_layers}")
        print(f"   [Gram Slicing] Original G: {self.global_G.shape} -> Sliced G: {G_eval.shape}")

        ridge_evaluator = LayUPEvaluator(G_eval, c_eval, self.cfg.RIDGE_LAMBDA, self.device)

        results_acc, results_filter_err = {}, {}
        model.eval()

        for d_name in task_sequence.keys():
            is_seen_task = (d_name in seen_task_names)
            task_status = "[SEEN]" if is_seen_task else "[UNSEEN]"
            
            _, _, loader, _ = build_cur_task_data_loader(
                root=dataset_root, dataset_name=d_name, transform_train=transform_test, 
                transform_test=transform_test, num_shots=-1, batch_size=self.cfg.BATCH_SIZE, num_workers=self.cfg.NUM_WORKERS
            )
            
            labels_start_idx = dataset_global_offsets[d_name]
            correct_fused = correct_ridge = correct_clip = total = count_pred_as_seen = count_pred_as_unseen = 0
            
            clip_logit_sum = 0.0
            clip_logit_sq_sum = 0.0
            ridge_logit_sum = 0.0
            ridge_logit_sq_sum = 0.0
            logit_elements_count = 0
            
            conf_list = []
            scale_factor_list = []
            
            with torch.no_grad():
                for images, labels in tqdm(loader, desc=f"Eval {d_name} {task_status}", leave=False):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    global_labels = labels + labels_start_idx
                    
                    with autocast():
                        img_feats_tuned = F.normalize(model.encode_image(images, skip_adapters=False), dim=-1)
                        img_feats_orig = F.normalize(model.encode_image(images, skip_adapters=True), dim=-1)
                        
                        img_feats_combine = 0.5 * img_feats_tuned + 0.5 * img_feats_orig
                        text_feats_combine = self.cfg.FILTER_WEIGHT_ORIG_TEXT * W_seen_orig.T + self.cfg.FILTER_WEIGHT_TUNED_TEXT * W_seen_cached.T
                        
                        logits_seen_filter = img_feats_combine @ text_feats_combine
                        
                        if W_unseen_orig.shape[0] > 0:
                            logits_global_filter = torch.cat([logits_seen_filter, (0.5 * img_feats_tuned + 0.5 * img_feats_orig ) @ W_unseen_orig.T], dim=1)
                        else:
                            logits_global_filter = logits_seen_filter
                        
                        first_preds = logits_global_filter.argmax(dim=1)
                        mask_predicted_seen = (first_preds < num_seen_classes)
                    
                    count_pred_as_seen += mask_predicted_seen.sum().item()
                    count_pred_as_unseen += (~mask_predicted_seen).sum().item()

                    final_preds_fused, final_preds_ridge, final_preds_clip = first_preds.clone(), first_preds.clone(), first_preds.clone()
                    
                    if mask_predicted_seen.any():
                        with autocast():
                            extracted_feats = model.extract_concat_features(images, target_layers=eval_layers)
                        
                        extracted_feats = extracted_feats.float() 
                        img_feats_tuned_float = img_feats_tuned.float()
                        W_seen_cached_float = W_seen_cached.float()

                        logits_ridge = ridge_evaluator.predict(extracted_feats)
                        logits_clip_tuned = img_feats_tuned_float @ W_seen_cached_float.T
                        
                        ridge_conf = ridge_evaluator.get_ridge_confidence(extracted_feats) 
                        conf_list.append(ridge_conf.detach().cpu())
                        
                        conf_w = ridge_conf.unsqueeze(1)
                        logits_fused = (1 - conf_w) * logits_clip_tuned + conf_w * logits_ridge
                        
                        masked_logits_ridge_all = logits_ridge[mask_predicted_seen]
                        masked_logits_clip_all = logits_clip_tuned[mask_predicted_seen]
                        
                        top1_logits_ridge = masked_logits_ridge_all.max(dim=1)[0]
                        top1_logits_clip = masked_logits_clip_all.max(dim=1)[0]
                        
                        num_samples = top1_logits_clip.size(0)
                        logit_elements_count += num_samples
                        
                        clip_logit_sum += top1_logits_clip.sum().item()
                        clip_logit_sq_sum += (top1_logits_clip ** 2).sum().item()
                        
                        ridge_logit_sum += top1_logits_ridge.sum().item()
                        ridge_logit_sq_sum += (top1_logits_ridge ** 2).sum().item()

                        final_preds_fused[mask_predicted_seen] = logits_fused.argmax(dim=1)[mask_predicted_seen]
                        final_preds_ridge[mask_predicted_seen] = logits_ridge.argmax(dim=1)[mask_predicted_seen]
                        final_preds_clip[mask_predicted_seen]  = logits_clip_tuned.argmax(dim=1)[mask_predicted_seen]
                    
                    correct_fused += (final_preds_fused == global_labels).sum().item()
                    correct_ridge += (final_preds_ridge == global_labels).sum().item()
                    correct_clip  += (final_preds_clip == global_labels).sum().item()
                    total += images.size(0)

            if logit_elements_count > 0:
                clip_mean = clip_logit_sum / logit_elements_count
                clip_var = max(0.0, (clip_logit_sq_sum / logit_elements_count) - (clip_mean ** 2))
                
                ridge_mean = ridge_logit_sum / logit_elements_count
                ridge_var = max(0.0, (ridge_logit_sq_sum / logit_elements_count) - (ridge_mean ** 2))
            else:
                clip_mean, clip_var, ridge_mean, ridge_var = 0.0, 0.0, 0.0, 0.0
                
            logit_stats_str = f"| CLIP(μ:{clip_mean:5.5f}, σ²:{clip_var:5.5f}) | Ridge(μ:{ridge_mean:5.5f}, σ²:{ridge_var:5.5f})"
            
            if len(conf_list) > 0:
                all_conf = torch.cat(conf_list, dim=0)
                conf_mean = all_conf.mean().item()
                conf_var = all_conf.var(unbiased=False).item() if all_conf.numel() > 1 else 0.0
            else:
                conf_mean, conf_var = 0.0, 0.0
                
            conf_str = f"| Conf(μ:{conf_mean:4.3f}, σ²:{conf_var:4.3f})"
            
            if len(scale_factor_list) > 0:
                scale_mean = sum(scale_factor_list) / len(scale_factor_list)
            else:
                scale_mean = 0.0
            sf_str = f"| Scale(μ:{scale_mean:4.2f})"

            acc_fused = 100.0 * correct_fused / total
            results_acc[d_name] = acc_fused
            filter_err = 100.0 * (count_pred_as_unseen if is_seen_task else count_pred_as_seen) / total
            results_filter_err[d_name] = filter_err

            filter_err_str = f"Filter Err(->{'Unseen' if is_seen_task else 'Seen'}): {filter_err:5.2f}%"
            
            if is_seen_task:
                print(f"Task {d_name:<15} | Status: {task_status:<8} | Acc(Fused): {acc_fused:.2f}% | Acc(Ridge): {100.0*correct_ridge/total:.2f}% | Acc(CLIP): {100.0*correct_clip/total:.2f}% | {filter_err_str} {logit_stats_str} {conf_str} {sf_str}")
            else:
                print(f"Task {d_name:<15} | Status: {task_status:<8} | Acc: {acc_fused:.2f}% | {filter_err_str} {logit_stats_str} {conf_str} {sf_str}")
            
        print("\n" + "="*100)
        print("Evaluation Summary (All Datasets)")
        print(f"{'Task Name':<20} | {'Status':<10} | {'Accuracy':<10} | {'Filter Error (Dir)':<25}")
        print("-" * 75)
        
        avg_seen, avg_unseen = [], []
        for name in task_sequence.keys():
            status = "SEEN" if name in seen_task_names else "UNSEEN"
            acc, f_err = results_acc[name], results_filter_err[name]
            err_msg = f"{f_err:5.2f}% (Pred {'Unseen' if status == 'SEEN' else 'Seen'})"
            (avg_seen if status == "SEEN" else avg_unseen).append(acc)
            print(f"{name:<20} | {status:<10} | {acc:<10.2f} | {err_msg:<25}")
        
        print("-" * 75)
        mean_seen = np.mean(avg_seen) if avg_seen else 0.0
        mean_unseen = np.mean(avg_unseen) if avg_unseen else 0.0
        h_mean = (2 * mean_seen * mean_unseen) / (mean_seen + mean_unseen + 1e-6)
        avg_all = np.mean(list(results_acc.values()))
        
        print(f"{'Avg Seen':<20} | {'-':<10} | {mean_seen:<10.2f}")
        print(f"{'Avg Unseen':<20} | {'-':<10} | {mean_unseen:<10.2f}")
        print(f"{'Avg Harmonic':<20} | {'-':<10} | {h_mean:<10.2f}")
        print(f"{'Avg All':<20} | {'-':<10} | {avg_all:<10.2f}")
        print("="*100 + "\n")

        return {
            'seen': mean_seen,
            'unseen': mean_unseen,
            'harmonic': h_mean,
            'all': avg_all,
            'dataset_accs': results_acc
        }

class ContinualEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.GPU}" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading CLIP: {self.cfg.BACKBONE}")
        clip_model = load_clip_to_cpu(self.cfg.BACKBONE, "fp32").to(self.device)
        self.model = ContinualCLIP(self.cfg, clip_model).to(self.device)
        
        self.inference_manager = GlobalInferenceManager(cfg, self.device)
        self.visual_gpm = VisualGPMManager()
        self.adb_manager = ADBManager()
        self.seen_text_features = None

    def load_task_state(self, task_name):
        load_path = os.path.join(self.cfg.CHECKPOINT_DIR, f"ckpt_{task_name}.pt")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint for {task_name} not found at {load_path}")
        
        print(f"Loading checkpoint from {load_path} ...")
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        inf_state = checkpoint["inference_manager"]
        self.inference_manager.global_G = inf_state["global_G"].to(self.device).float() if inf_state["global_G"] is not None else None
        self.inference_manager.global_c = inf_state["global_c"].to(self.device).float() if inf_state["global_c"] is not None else None
        self.inference_manager.seen_tasks = inf_state["seen_tasks"]
        self.inference_manager.total_seen_classes = inf_state["total_seen_classes"]
        self.inference_manager.all_cached_text_features = [t.to(self.device).float() for t in inf_state["all_cached_text_features"]]
        self.inference_manager.task_adapters_history = inf_state.get("task_adapters_history", [])

        if "gpm_bases_history" in checkpoint: self.visual_gpm.bases_history = checkpoint["gpm_bases_history"]
        self.model.text_tuner.constraint_manager.old_class_info = checkpoint.get("text_constraint_old_class_info", [])
        if checkpoint.get("seen_text_features") is not None: self.seen_text_features = checkpoint["seen_text_features"]
        if "adb_class_statistics" in checkpoint: self.adb_manager.class_statistics = checkpoint["adb_class_statistics"]

        print(f"Successfully restored state after task: {task_name}")

    def run_evaluation(self, task_sequence, target_task=None):
        tasks_list = list(task_sequence.keys())
        start_idx = 0 
        
        if target_task is not None:
            if isinstance(target_task, int):
                if target_task < 0 or target_task >= len(tasks_list):
                    raise ValueError(f"Step index {target_task} out of range (0-{len(tasks_list)-1}).")
                start_idx = target_task
            elif isinstance(target_task, str) and target_task in tasks_list:
                start_idx = tasks_list.index(target_task)
            else:
                raise ValueError(f"Invalid target task '{target_task}'. Valid names: {tasks_list}")

            print(f"\n Evaluating sequentially STARTING FROM Step: {tasks_list[start_idx]} (Task Index: {start_idx}) to the end...")
        else:
            print("\n Evaluating ALL Steps Sequentially from the beginning...")

        all_step_results = []

        for i in range(start_idx, len(tasks_list)):
            task_name = tasks_list[i]
            print(f"\n{'#'*60}")
            print(f"### Evaluating after Task {i}: {task_name}")
            print(f"{'#'*60}")
            try:
                self.load_task_state(task_name)
                step_results = self.inference_manager.evaluate_all_seen_tasks(self.model, self.cfg.DATA_ROOT, None, task_sequence)
                if step_results:
                    all_step_results.append({
                        'step': i,
                        'task_name': task_name,
                        **step_results
                    })
            except FileNotFoundError as e:
                print(f" Checkpoint missing for {task_name}, skipping... ({e})")

        return self._print_final_summary(all_step_results, task_sequence)

    def _print_final_summary(self, results, task_sequence):
        if not results:
            print("No evaluation steps were completed, cannot generate a final summary.")
            return None

        name_mapping = {
            'aircraft': 'air.', 'caltech101': 'cal.', 'dtd': 'dtd.', 'eurosat': 'eur.',
            'flowers': 'flo.', 'food101': 'foo.', 'mnist': 'mni.', 'oxford_pets': 'oxf.',
            'stanford_cars': 'sta.', 'sun397': 'sun.'
        }
        dataset_names = list(task_sequence.keys())
        
        data_dict = { "Dataset": dataset_names }
        
        lines = []
        lines.append("{")
        dataset_json = json.dumps(dataset_names, ensure_ascii=False)
        lines.append(f'  "Dataset": {dataset_json},')

        for i, d_name in enumerate(dataset_names):
            acc_history = []
            for res in results:
                acc = res['dataset_accs'].get(d_name, 0.0)
                acc_history.append(round(acc, 2))
            
            short_key = name_mapping.get(d_name, d_name[:3] + ".")
            
            data_dict[short_key] = acc_history
            
            acc_str = ", ".join([f"{val:.2f}" for val in acc_history])
            comma = "," if i < len(dataset_names) - 1 else ""
            lines.append(f'  "{short_key}": [{acc_str}]{comma}')

        lines.append("}")
        final_json_string = "\n".join(lines)

        print("\n\n" + "=" * 100)
        print("Final Accuracy Matrix (Dataset x Step)")
        print(final_json_string)
        print("=" * 100 + "\n")

        matrix_save_path = os.path.join(self.cfg.EXP_ROOT, "final_matrix_results.txt")
        try:
            with open(matrix_save_path, "w", encoding="utf-8") as f:
                f.write(final_json_string)
            print(f" [Success] Final matrix saved to: {matrix_save_path}")
        except Exception as e:
            print(f" [Error] Failed to save matrix file: {e}")

        return data_dict

def calculate_and_save_metrics(data_dict, save_dir):
    df = pd.DataFrame(data_dict)
    df.set_index('Dataset', inplace=True)

    scores = df.iloc[:, :].astype(float)
    columns = df.columns[:]

    transfer = []
    average = []
    last = []

    for i, col in enumerate(columns):
        if i == 0:
            transfer.append(float("nan"))
        else:
            transfer.append(scores.iloc[:i, i].mean())
        
        average.append(scores[col].mean())
        last.append(scores[col].iloc[-1])

    result = pd.DataFrame([transfer, average, last],
                          index=["Transfer", "Average", "Last"],
                          columns=columns)

    transfer_mean = pd.Series(transfer[1:]).mean()
    average_mean = pd.Series(average).mean()
    last_mean = pd.Series(last).mean()

    print("\n" + "="*80)
    print("Metrics Evaluation (Transfer / Average / Last)")
    print("="*80)
    print(result.to_string())
    print()
    print(f"Transfer Mean: {transfer_mean:.1f}")
    print(f"Average Mean:  {average_mean:.1f}")
    print(f"Last Mean:     {last_mean:.1f}")
    print("="*80 + "\n")

    metrics_save_path = os.path.join(save_dir, "calculated_metrics.txt")
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        f.write(result.to_string() + "\n\n")
        f.write(f"Transfer Mean: {transfer_mean:.1f}\n")
        f.write(f"Average Mean:  {average_mean:.1f}\n")
        f.write(f"Last Mean:     {last_mean:.1f}\n")
        
    print(f" Calculated metrics successfully saved to: {metrics_save_path}")


class Config:
    def __init__(self, exp_root, gpu=0):
        self.DATA_ROOT = "/root/data"
        self.BACKBONE = "ViT-B/16"
        self.BATCH_SIZE = 256
        self.GPU = gpu
        self.NUM_WORKERS = 8
        self.T_ADAPTFORMER = True 
        
        self.EXP_ROOT = exp_root 
        self.CHECKPOINT_DIR = os.path.join(self.EXP_ROOT, "checkpoints") 

        self.VISUAL_ADAPTER_LAYERS = [
            "visual.transformer.resblocks.7", "visual.transformer.resblocks.8",
            "visual.transformer.resblocks.9", "visual.transformer.resblocks.10",
            "visual.transformer.resblocks.11"
        ]
        self.VISUAL_GRAM_SELECT_LAYERS = self.VISUAL_ADAPTER_LAYERS + ["visual"]
        
        self.EVAL_GRAM_SELECT_LAYERS = [
            "visual.transformer.resblocks.7", "visual.transformer.resblocks.8",
            "visual.transformer.resblocks.9", "visual.transformer.resblocks.10",
            "visual.transformer.resblocks.11", "visual"
        ]

        self.VISUAL_ADAPTER_DIM = 768 
        self.VISUAL_ALPHA = 0.1       
        self.T_ADAPTER_DIM = 128
        
        self.RIDGE_LAMBDA = 10 
        
        self.FILTER_WEIGHT_ORIG_TEXT = 0.5
        self.FILTER_WEIGHT_TUNED_TEXT = 0.5


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate trained continual learning checkpoints")
    parser.add_argument("--exp_root", type=str, nargs="+", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--target_step", type=int, default=0)
    parser.add_argument("--task_order", type=str, nargs="+", required=True)
    args = parser.parse_args()

    TARGET_STEP = args.target_step
    seed_folders = args.exp_root

    TASK_SEQUENCE = {task.lower(): (0, 0, 0) for task in args.task_order}

    original_stdout = sys.stdout

    for folder in seed_folders:
        if not os.path.exists(folder):
            continue

        sys.stdout = original_stdout  
        
        cfg = Config(exp_root=folder, gpu=args.gpu)
        log_file = os.path.join(cfg.EXP_ROOT, "eval_log_all_steps.txt")
        sys.stdout = Logger(log_file)
        
        print("\n" + "="*80)
        print(f" Processing Folder: {folder}")
        print(f" Evaluation Started at: {datetime.datetime.now()}")
        print("="*80 + "\n")

        evaluator = ContinualEvaluator(cfg)
        
        data_dict = evaluator.run_evaluation(TASK_SEQUENCE, target_task=TARGET_STEP)
        
        if data_dict is not None:
            calculate_and_save_metrics(data_dict, cfg.EXP_ROOT)
        
    sys.stdout = original_stdout
    print("\n Evaluation and metric computation for all seed folders have been fully completed!")
    
