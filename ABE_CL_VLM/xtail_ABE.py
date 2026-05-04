import os
import time
import datetime
import math
import pickle
from tqdm import tqdm
import sys
import numpy as np
import copy
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms

from clip import clip


from scenario_datasets.build_functions import build_cur_task_data_loader, build_TAIL_testloader

import random

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_custom_template(dataset_name, is_eval=False):
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
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.zeros_(self.linear.weight)

    def forward(self, x):
        cls_token = x[0:1, :, :]      
        patch_tokens = x[1:, :, :]    
        update = self.linear(cls_token)
        cls_token = cls_token + self.alpha * update
        x = torch.cat([cls_token, patch_tokens], dim=0)
        return x

class AdaptFormer(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim, dtype=dtype)
        self.down_proj = nn.Linear(in_dim, bottle_dim, dtype=dtype)
        self.relu = nn.ReLU(inplace=True)
        self.up_proj = nn.Linear(bottle_dim, in_dim, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(1, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
        
        if self.ln.elementwise_affine:
            nn.init.ones_(self.ln.weight)
            nn.init.zeros_(self.ln.bias)
            
        nn.init.ones_(self.scale)

    
    def forward(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        x = x * self.scale
        return x

class Text_Tuner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        n_layers = len(clip_model.transformer.resblocks)
        emb_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        adapter_dim = cfg.T_ADAPTER_DIM
        
        use_adaptformer = getattr(cfg, 'T_ADAPTFORMER', True)

        self.adaptformer_list = nn.ModuleList()
        if use_adaptformer:
            print(f"Initializing Text_Tuner with {n_layers} AdaptFormer blocks.")
            for _ in range(n_layers):
                self.adaptformer_list.append(
                    AdaptFormer(in_dim=emb_dim, bottle_dim=adapter_dim, dtype=dtype)
                )
        else:
            for _ in range(n_layers):
                self.adaptformer_list.append(None)
        
    def reset_parameters(self):
        cpu_rng_state = torch.get_rng_state()
        gpu_rng_state = None
        if next(self.parameters()).is_cuda:
            gpu_rng_state = torch.cuda.get_rng_state(next(self.parameters()).device)

        torch.manual_seed(0)
        if gpu_rng_state is not None:
            torch.cuda.manual_seed(0)

        for adapter in self.adaptformer_list:
            if adapter is not None:
                adapter.reset_parameters()

        torch.set_rng_state(cpu_rng_state)
        if gpu_rng_state is not None:
            torch.cuda.set_rng_state(gpu_rng_state, next(self.parameters()).device)

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
        x = self.token_embedding(text).to(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)

        _bsz, _seq_len, _emb_dim = x.shape
        n_layers = len(self.blocks)

        for i in range(n_layers):
            block = self.blocks[i]

            adaptformer = None
            if tuner is not None:
                adaptformer = tuner.adaptformer_list[i]

            x = x.permute(1, 0, 2)

            _attn = block.attn
            _ln_1 = block.ln_1
            _mlp = block.mlp
            _ln_2 = block.ln_2
            _num_heads = _attn.num_heads
            _head_dim = _emb_dim // _num_heads

            identity = x
            x = _ln_1(x)
            
            q, k, v = F.linear(x, _attn.in_proj_weight, _attn.in_proj_bias).chunk(3, dim=-1)
            q = q.contiguous().view(q.shape[0], q.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], k.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], v.shape[1] * _num_heads, _head_dim).transpose(0, 1)
            
            attn_mask = block.attn_mask.to(dtype=x.dtype, device=x.device) if block.attn_mask is not None else None
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            x = x.transpose(0, 1).contiguous().view(-1, _emb_dim)
            x = F.linear(x, _attn.out_proj.weight, _attn.out_proj.bias)
            x = x.view(_seq_len, _bsz, _emb_dim)
            x = x + identity

            identity = x
            x = _ln_2(x)
            x = _mlp[2](_mlp[1](_mlp[0](x)))

            if adaptformer is not None:
                x = x + 0.1 * adaptformer(identity)
            
            x = x + identity
            x = x.permute(1, 0, 2) 

        x = self.ln_final(x).to(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

class ContinualCLIP(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.cfg = cfg
        self.dtype = clip_model.dtype
        
        self.image_encoder = clip_model.visual
        self.visual_adapters = nn.ModuleList()
        self.frozen_visual_adapters = nn.ModuleList()
        
        self.adapter_indices = set()
        for layer_name in cfg.VISUAL_ADAPTER_LAYERS:
            if "resblocks" in layer_name:
                idx = int(layer_name.split('.')[-1])
                self.adapter_indices.add(idx)

        for i in range(len(self.image_encoder.transformer.resblocks)):
            if i in self.adapter_indices:
                self.visual_adapters.append(
                    VisualAdapter(cfg.VISUAL_ADAPTER_DIM, alpha=cfg.VISUAL_ALPHA).to(self.dtype)
                )
                frozen_adapter = VisualAdapter(cfg.VISUAL_ADAPTER_DIM, alpha=cfg.VISUAL_ALPHA).to(self.dtype)
                for p in frozen_adapter.parameters():
                    p.requires_grad = False
                    p.data.zero_()
                self.frozen_visual_adapters.append(frozen_adapter)
            else:
                self.visual_adapters.append(None)
                self.frozen_visual_adapters.append(None)
        
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.logit_scale = clip_model.logit_scale
        
        self.text_encoder = Peft_Text(clip_model)
        self.text_tuner = Text_Tuner(cfg, clip_model)


    def reset_visual_adapters(self):
        print(">> Resetting Visual Adapters to Initial State (Zeros).")
        for adapter in self.visual_adapters:
            if adapter is not None:
                adapter.reset_parameters()

    def load_frozen_accumulated_adapters(self, accumulated_state_dict):
        print(">> Loading accumulated history weights into Frozen Adapters...")
        my_state = self.frozen_visual_adapters.state_dict()
        for k, v in accumulated_state_dict.items():
            if k in my_state:
                my_state[k].copy_(v)
        
        for p in self.frozen_visual_adapters.parameters():
            p.requires_grad = False
        print(">> Frozen Adapters loaded and locked.")

    def encode_image(self, image, skip_adapters=False):
        x = self.image_encoder.conv1(image.type(self.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1) 
        x = torch.cat([self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2) 

        for i, block in enumerate(self.image_encoder.transformer.resblocks):
            x = block(x)
            
            if not skip_adapters:
                cls_token = x[0:1, :, :]
                patch_tokens = x[1:, :, :]
                
                total_update = 0
                alpha = self.cfg.VISUAL_ALPHA
                applied = False

                if self.frozen_visual_adapters[i] is not None:
                    with torch.no_grad():
                        update_frozen = self.frozen_visual_adapters[i].linear(cls_token)
                    total_update = total_update + update_frozen
                    applied = True

                if self.visual_adapters[i] is not None:
                    update_train = self.visual_adapters[i].linear(cls_token)
                    total_update = total_update + update_train
                    applied = True
                
                if applied:
                    cls_token = cls_token + alpha * total_update
                    x = torch.cat([cls_token, patch_tokens], dim=0)

        x = x.permute(1, 0, 2)
        x = self.image_encoder.ln_post(x[:, 0, :])

        if self.image_encoder.proj is not None:
            x = x @ self.image_encoder.proj

        return x

    def encode_text(self, text, skip_tuner=False):
        if skip_tuner:
            return self.text_encoder(text, tuner=None)
        else:
            return self.text_encoder(text, tuner=self.text_tuner)

    
    def forward(self, image, text_features=None, text_prompts=None):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        
        if text_features is None and text_prompts is not None:
            text_features = self.encode_text(text_prompts)
            text_features = F.normalize(text_features, dim=-1)
        
        logits = image_features @ text_features.T
        return logits, image_features, text_features

    def extract_concat_features(self, images, target_layers=None):
        if target_layers is None:
            target_layers = self.cfg.VISUAL_GRAM_SELECT_LAYERS

        extracted_feats = {}
        
        x = self.image_encoder.conv1(images.type(self.dtype))
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = torch.cat([self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)
        x = self.image_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)

        for i, block in enumerate(self.image_encoder.transformer.resblocks):
            x = block(x)
            
            current_layer_name = f"visual.transformer.resblocks.{i}"
            if current_layer_name in target_layers:
                feat = x[0, :, :].clone()
                extracted_feats[current_layer_name] = feat

            cls_token = x[0:1, :, :]      
            patch_tokens = x[1:, :, :]    
            total_update = 0
            alpha = self.cfg.VISUAL_ALPHA
            applied = False

            if self.frozen_visual_adapters[i] is not None:
                with torch.no_grad():
                    update_frozen = self.frozen_visual_adapters[i].linear(cls_token)
                total_update = total_update + update_frozen
                applied = True

            if self.visual_adapters[i] is not None:
                update_train = self.visual_adapters[i].linear(cls_token)
                total_update = total_update + update_train
                applied = True
            
            if applied:
                cls_token = cls_token + alpha * total_update
                x = torch.cat([cls_token, patch_tokens], dim=0)
            
        if "visual" in target_layers:
            x_final = x.permute(1, 0, 2)
            x_final = self.image_encoder.ln_post(x_final[:, 0, :])
            if self.image_encoder.proj is not None:
                x_final = x_final @ self.image_encoder.proj
            extracted_feats["visual"] = x_final

        features_ordered = []
        for layer_name in target_layers:
            if layer_name in extracted_feats:
                features_ordered.append(extracted_feats[layer_name])
            else:
                raise ValueError(f"Feature for layer {layer_name} was not extracted.")
        
        concat_feat = torch.cat(features_ordered, dim=1)
        return concat_feat

class VisualGPMManager:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.bases_history = [] 

    def add_basis_from_task_gram(self, task_G, layer_dims_list):
        print(f"Calculating GPM Bases and Eigenvalues for current task...")
        current_task_data = [None] * len(layer_dims_list)
        start_idx = 0
        for i, dim in enumerate(layer_dims_list):
            end_idx = start_idx + dim
            layer_name = self.cfg.VISUAL_GRAM_SELECT_LAYERS[i]
            G_layer = task_G[start_idx:end_idx, start_idx:end_idx]
            eigenvalues, eigenvectors = torch.linalg.eigh(G_layer.double())
            idx = eigenvalues.argsort(descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            total_energy = torch.sum(eigenvalues)
            energy_cumsum = torch.cumsum(eigenvalues, dim=0)
            threshold = self.cfg.GPM_THRESHOLD * total_energy
            
            if total_energy > 0:
                k = torch.searchsorted(energy_cumsum, threshold).item() + 1
            else:
                k = 0
            
            if k > 0:
                U_new = eigenvectors[:, :k].float().to(self.device)
                Lam_new = eigenvalues[:k].float().to(self.device)
                current_task_data[i] = (U_new, Lam_new)
            else:
                current_task_data[i] = (None, None)
            
            print(f"    Layer {layer_name} | Basis Dim: {k}")
            start_idx = end_idx
        
        self.bases_history.append(current_task_data)
        print(f"GPM: Added new task bases & eigenvalues. Total historical tasks: {len(self.bases_history)}")

    def compute_orthogonality_loss(self, full_visual_adapters_list):
        if not self.bases_history:
            return torch.tensor(0.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)
        valid_layers_count = 0

        for layer_idx, layer_name in enumerate(self.cfg.VISUAL_GRAM_SELECT_LAYERS):
            adapter = None
            if "resblocks" in layer_name:
                idx = int(layer_name.split('.')[-1])
                if idx < len(full_visual_adapters_list):
                    adapter = full_visual_adapters_list[idx]
            
            if adapter is None:
                continue
            
            W = adapter.linear.weight 
            layer_weighted_bases = []
            for task_data in self.bases_history:
                basis_pair = task_data[layer_idx]
                if basis_pair is not None:
                    U, Lam = basis_pair
                    if U is not None and U.shape[1] > 0:
                        U_weighted = U * torch.sqrt(Lam + 1e-6).unsqueeze(0)
                        layer_weighted_bases.append(U_weighted)
            
            if not layer_weighted_bases:
                continue

            U_all = torch.cat(layer_weighted_bases, dim=1) 
            projection = torch.matmul(W, U_all)
            total_loss += torch.sum(projection ** 2)
            valid_layers_count += 1
                
        if valid_layers_count > 0:
            return total_loss / valid_layers_count
        return total_loss

class ADBManager:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.class_statistics = {} 
        
        self.cached_mus_tensor = None      
        self.cached_P_tensor = None       
        self.cache_valid = False

    def load_buffer_for_training(self, seen_tasks):
        print(f">> ADBManager: Analytic Boundary mode enabled. No physical samples needed.")
        self.cache_valid = False 

    def update_class_stats(self, global_label, G_class_accum, mu_class_accum, num_samples):
        N = float(num_samples)
        if N < 2: return

        mu_c = (mu_class_accum.cpu().double() / N).float()
        G_c = (G_class_accum.cpu().double() / N).float()
        D = mu_c.shape[0]

        term_mu = torch.outer(mu_c, mu_c)
        Sigma_c = (N / (N - 1)) * (G_c - term_mu)

        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma_c.double())
        eigenvalues = eigenvalues.float()
        eigenvectors = eigenvectors.float()

        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        valid_eig = torch.clamp(eigenvalues, min=1e-6)
        total_energy = torch.sum(valid_eig)
        energy_cumsum = torch.cumsum(valid_eig, dim=0)
        
        threshold = self.cfg.ADB_ENERGY_THRESHOLD * total_energy
        K = torch.searchsorted(energy_cumsum, threshold).item() + 1 if total_energy > 1e-6 else 1
        K = min(K, D)

        lambda_trunc = valid_eig[:K]
        U_trunc = eigenvectors[:, :K]
        P_c = U_trunc * torch.sqrt(lambda_trunc).unsqueeze(0)

        self.class_statistics[global_label] = {
            'mu': mu_c.cpu(), 
            'P': P_c.cpu()
        }
        self.cache_valid = False

    def _ensure_tensor_cache(self):
        if self.cache_valid and self.cached_mus_tensor is not None:
            return

        if not self.class_statistics:
            return

        sorted_labels = sorted(self.class_statistics.keys())
        mus_list, Ps_list = [], []
        
        max_K = max(stats['P'].shape[1] for stats in self.class_statistics.values())
        D = next(iter(self.class_statistics.values()))['mu'].shape[0]

        for lbl in sorted_labels:
            stats = self.class_statistics[lbl]
            
            mus_list.append(stats['mu'].to(self.device)) 
            
            p_mat = stats['P'].to(self.device)
            
            k_curr = p_mat.shape[1]
            if k_curr < max_K:
                padding = torch.zeros(D, max_K - k_curr, device=self.device, dtype=p_mat.dtype)
                Ps_list.append(torch.cat([p_mat, padding], dim=1))
            else:
                Ps_list.append(p_mat)
        
        self.cached_mus_tensor = torch.stack(mus_list)      
        self.cached_P_tensor = torch.stack(Ps_list)         
        self.cache_valid = True

    def get_stats_for_boundary(self):
        if not self.class_statistics:
            return None, None
        self._ensure_tensor_cache()
        return self.cached_mus_tensor, self.cached_P_tensor
    

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

    def save_current_task_adapters(self, model):
        print(f"Storing Visual Adapter weights for current task...")
        adapter_state = copy.deepcopy(model.visual_adapters.state_dict())
        for k, v in adapter_state.items():
            adapter_state[k] = v.cpu()
        self.task_adapters_history.append(adapter_state)
        print(f"Stored adapters. Total history count: {len(self.task_adapters_history)}")

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

    def update_state(self, task_name, G_task, c_task, num_samples, class_names, task_text_features):
        print(f"Updating Global Inference State for {task_name}...")
        
        G_normalized = 10000 * G_task / num_samples
        
        if self.global_G is None:
            self.global_G = G_normalized.to(self.device)
        else:
            self.global_G += G_normalized.to(self.device)
            
        if self.global_c is None:
            self.global_c = c_task.to(self.device)
        else:
            self.global_c = torch.cat([self.global_c, c_task.to(self.device)], dim=1)
            
        self.seen_tasks.append({
            'name': task_name,
            'class_names': class_names,
            'labels_start_idx': self.total_seen_classes
        })
        self.total_seen_classes += len(class_names)

        self.all_cached_text_features.append(task_text_features.to(self.device))
        
        print(f"Global G shape: {self.global_G.shape}, Global c shape: {self.global_c.shape}")
        print(f"Total seen classes: {self.total_seen_classes}")


class ContinualTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(f"cuda:{cfg.GPU}" if torch.cuda.is_available() else "cpu")
        
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cfg.GRAM_SAVE_DIR, exist_ok=True) 
        
        resolution = 224
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.transform_train_special = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        self.build_model()

        self.scheduler = None 
        
        self.visual_gpm = VisualGPMManager(cfg, self.device)
        self.inference_manager = GlobalInferenceManager(cfg, self.device)
        self.adb_manager = ADBManager(cfg, self.device)

        self.similarity_matrix = None

        self.cached_protos = None 
        self.seen_text_features = None 
        
        self.build_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()

    def save_model_only(self, task_name):
        print(f"Saving TRAINED MODEL WEIGHTS (Safety Backup - Optimized) for task: {task_name} ...")
        full_state = self.model.state_dict()
        optimized_state_dict = {}
        for k, v in full_state.items():
            if "visual_adapters" in k or "text_tuner" in k:
                optimized_state_dict[k] = v.cpu()

        state = {
            "model_state_dict": optimized_state_dict,
        }
        
        save_path = os.path.join(self.cfg.CHECKPOINT_DIR, f"ckpt_{task_name}_weights_only.pt")
        torch.save(state, save_path)
        print(f"Safety backup saved to {save_path}")

    def save_task_state(self, task_name):
        print(f"Saving optimized checkpoint for task: {task_name}...")
        
        full_state = self.model.state_dict()
        optimized_state_dict = {k: v.cpu() for k, v in full_state.items() if "visual_adapters" in k or "text_tuner" in k}
        
        inf_state = {
            "global_G": self.inference_manager.global_G.cpu().float() if self.inference_manager.global_G is not None else None,
            "global_c": self.inference_manager.global_c.cpu().float() if self.inference_manager.global_c is not None else None,
            "seen_tasks": self.inference_manager.seen_tasks,
            "total_seen_classes": self.inference_manager.total_seen_classes,
            "all_cached_text_features": [t.cpu().float() for t in self.inference_manager.all_cached_text_features],
            "task_adapters_history": self.inference_manager.task_adapters_history 
        }

        optimized_adb_stats = {}
        for k, v in self.adb_manager.class_statistics.items():
            optimized_adb_stats[k] = {
                'mu': v['mu'].cpu().float(),
                'P': v['P'].cpu().float()
            }

        gpm_history_cpu = []
        for task_layers in self.visual_gpm.bases_history:
            gpm_history_cpu.append([(item[0].cpu().float(), item[1].cpu().float()) if item else None for item in task_layers])

        state = {
            "model_state_dict": optimized_state_dict,
            "inference_manager": inf_state,
            "gpm_bases_history": gpm_history_cpu,
            "seen_text_features": self.seen_text_features.cpu().float() if self.seen_text_features is not None else None, 
            "adb_class_statistics": optimized_adb_stats,
            
        }
        
        torch.save(state, os.path.join(self.cfg.CHECKPOINT_DIR, f"ckpt_{task_name}.pt"))

    def load_task_state(self, task_name):
        load_path = os.path.join(self.cfg.CHECKPOINT_DIR, f"ckpt_{task_name}.pt")
        print(f"Loading checkpoint from {load_path} ...")
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        
        inf_state = checkpoint["inference_manager"]
        self.inference_manager.global_G = inf_state["global_G"].to(self.device) if inf_state["global_G"] is not None else None
        self.inference_manager.global_c = inf_state["global_c"].to(self.device) if inf_state["global_c"] is not None else None
        self.inference_manager.seen_tasks = inf_state["seen_tasks"]
        self.inference_manager.total_seen_classes = inf_state["total_seen_classes"]
        self.inference_manager.all_cached_text_features = [t.to(self.device) for t in inf_state["all_cached_text_features"]]
        self.inference_manager.task_adapters_history = inf_state.get("task_adapters_history", [])

        self.visual_gpm.bases_history = [[(item[0].to(self.device), item[1].to(self.device)) if item else None for item in layers] 
                                          for layers in checkpoint["gpm_bases_history"]]
        
        self.adb_manager.class_statistics = checkpoint["adb_class_statistics"]
        
        if checkpoint["seen_text_features"] is not None:
            self.seen_text_features = checkpoint["seen_text_features"].to(self.device)
            


    def set_trainable_mode(self, train_text=True, train_visual=True):
        for p in self.model.visual_adapters.parameters():
            p.requires_grad = train_visual
        
        for p in self.model.text_tuner.parameters():
            p.requires_grad = train_text

        params = []
        if train_text:
            params.append({'params': self.model.text_tuner.parameters()})
        
        if train_visual:
            params.append({'params': self.model.visual_adapters.parameters()})
        
        mode_str = f"Text={train_text}, Visual={train_visual}"

        if len(params) > 0:
            self.optimizer = torch.optim.AdamW(params, lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY)
        else:
            self.optimizer = None
        
        print(f">> Switched Training Mode to: {mode_str}")
    
    @torch.no_grad()
    def compute_visual_prototypes(self, loader, skip_adapters=False):
        self.model.eval()
        
        all_features = []
        all_labels = []
        
        desc_str = f"Computing Prototypes (Fast, Original={skip_adapters})"
        pbar = tqdm(loader, desc=desc_str, file=sys.stdout, leave=False, dynamic_ncols=True)

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            with autocast():
                features = self.model.encode_image(images, skip_adapters=skip_adapters)
                features = F.normalize(features, dim=-1)
            
            all_features.append(features)
            all_labels.append(labels)
        
        if len(all_features) == 0:
             return torch.empty(0, 512, device=self.device)

        X = torch.cat(all_features, dim=0) 
        y = torch.cat(all_labels, dim=0)   
        
        unique_labels, inverse_indices = torch.unique(y, sorted=True, return_inverse=True)
        num_classes = len(unique_labels)
        
        prototypes = torch.zeros(num_classes, 512, device=self.device, dtype=X.dtype)
        class_counts = torch.zeros(num_classes, device=self.device, dtype=X.dtype)
        
        prototypes.index_add_(0, inverse_indices, X)
        
        ones = torch.ones(X.size(0), device=self.device, dtype=X.dtype)
        class_counts.index_add_(0, inverse_indices, ones)
        
        class_counts = class_counts.unsqueeze(1).clamp(min=1.0)
        prototypes = prototypes / class_counts
        prototypes = F.normalize(prototypes, dim=-1)
        
        return prototypes

    def build_model(self):
        print(f"Loading CLIP: {self.cfg.BACKBONE}")
        clip_model = load_clip_to_cpu(self.cfg.BACKBONE, "fp32") 
        clip_model.to(self.device)
        
        for p in clip_model.parameters():
            p.requires_grad = False
        self.model = ContinualCLIP(self.cfg, clip_model).to(self.device)

    def build_optimizer(self):
        params = [
            {'params': self.model.text_tuner.parameters()},
            {'params': self.model.visual_adapters.parameters()}
        ]
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.LR, weight_decay=self.cfg.WEIGHT_DECAY)

    def _get_visual_prototype_indices(self):
        start_idx = 0
        found = False
        target_dim = self.model.ln_final.weight.shape[0] 

        for name in self.cfg.VISUAL_GRAM_SELECT_LAYERS:
            if "visual" == name: 
                dim = target_dim
                found = True
                break 
            elif "resblocks" in name:
                dim = 768 
            else:
                dim = 768 
            
            start_idx += dim
        
        if not found:
            return None, None
            
        end_idx = start_idx + target_dim
        return start_idx, end_idx
    

    def _run_single_epoch(self, loader, tokenized_prompts, epoch_idx, phase_name, train_mode):
        do_train_visual = (train_mode in ['visual', 'joint'])
        do_train_text   = (train_mode in ['text_orig_visual', 'text_tuned_visual', 'joint'])
        use_original_encoder = (train_mode == 'text_orig_visual')

        self.model.train()
        loss_meter = {'Total': 0.0, 'Cls': 0.0, 'Boundary': 0.0, 'Anchor': 0.0, 'Dist': 0.0, 'Reg': 0.0, 'Ortho': 0.0}
        steps = 0
        
        num_seen_classes = 0
        if self.inference_manager.global_c is not None:
            num_seen_classes = self.inference_manager.global_c.shape[1]

        desc_str = f"[{phase_name}] Ep {epoch_idx}"
        pbar = tqdm(loader, desc=desc_str, file=sys.stdout, dynamic_ncols=True)

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            global_labels = labels + num_seen_classes 
            
            with autocast():
                current_text_feats = self.model.encode_text(tokenized_prompts)
                current_text_feats = F.normalize(current_text_feats, dim=-1)
                
                if self.seen_text_features is not None:
                    full_text_feats = torch.cat([self.seen_text_features.to(self.device), current_text_feats], dim=0)
                else:
                    full_text_feats = current_text_feats
                
                raw_image_features = self.model.encode_image(images, skip_adapters=use_original_encoder)
                image_features = F.normalize(raw_image_features, dim=-1)
                
                logits = image_features @ full_text_feats.T
                scaled_logits = logits * self.model.logit_scale.exp()
                cls_loss = self.criterion(scaled_logits, global_labels)
                
                loss_boundary = torch.tensor(0.0, device=self.device)
                if do_train_text and self.seen_text_features is not None:
                    mu_old, P_old = self.adb_manager.get_stats_for_boundary()
                    if mu_old is not None:
                        w_old = self.seen_text_features.to(self.device)
                        w_new = current_text_feats
                        
                        C_old, D = w_old.shape
                        C_curr = w_new.shape[0]

                        delta_w = w_old.unsqueeze(1) - w_new.unsqueeze(0) 
                        
                        mu_delta = (delta_w * mu_old.unsqueeze(1)).sum(dim=-1)

                        proj_delta = torch.bmm(delta_w, P_old)
                        sigma_delta = torch.norm(proj_delta, p=2, dim=-1)

                        boundary_violation = self.cfg.BOUNDARY_GAMMA * sigma_delta - mu_delta + self.cfg.BOUNDARY_MARGIN
                        loss_boundary = F.relu(boundary_violation).mean() * self.cfg.BOUNDARY_LOSS_WEIGHT

                
                ortho_loss = torch.tensor(0.0, device=self.device)

                if do_train_visual:
                    if self.cfg.ORTHO_LOSS_COEFF > 0:
                        ortho_loss = self.cfg.ORTHO_LOSS_COEFF * self.visual_gpm.compute_orthogonality_loss(self.model.visual_adapters)

                loss = cls_loss + loss_boundary + ortho_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            pbar.set_postfix({
                'L_Tot': f"{loss.item():.3f}",
                'L_Cls': f"{cls_loss.item():.3f}",
                'L_Bnd': f"{loss_boundary.item():.3f}",
                'L_Ortho': f"{ortho_loss.item():.3f}"
            })

            steps += 1
            loss_meter['Total'] += loss.item()
            loss_meter['Cls'] += cls_loss.item()
            loss_meter['Boundary'] += loss_boundary.item()
            loss_meter['Ortho'] += ortho_loss.item()

        avg_total = loss_meter['Total'] / steps
        print(f"{desc_str} Done | Avg Loss: {avg_total:.4f}")

    def prepare_frozen_adapters_for_training(self):
        history = self.inference_manager.task_adapters_history
        if not history:
            print(">> No history adapters found. Frozen adapters remain Zero/Identity.")
            for p in self.model.frozen_visual_adapters.parameters():
                p.data.zero_()
                p.requires_grad = False
            return

        print(f">> Accumulating weights from {len(history)} previous tasks for Frozen Adapters...")
        
        summed_state = copy.deepcopy(self.model.visual_adapters.state_dict())
        
        for key in summed_state:
            summed_state[key].zero_()
        
        for task_state in history:
            for key, val in task_state.items():
                if key in summed_state:
                    summed_state[key] = summed_state[key] + val.to(self.device)
        
        self.model.load_frozen_accumulated_adapters(summed_state)
    
    def run_tasks(self, task_sequence):
        start_task = self.cfg.START_FROM_TASK
        should_skip = False
        prev_task_name = None
        
        if start_task is not None and start_task not in task_sequence:
             print(f"Warning: START_FROM_TASK '{start_task}' not in sequence. Starting from scratch.")
             start_task = None

        if start_task is not None and list(task_sequence.keys())[0] == start_task:
             start_task = None

        if start_task is not None:
            should_skip = True
            print(f"!!! Resume Mode Activated !!!")
            print(f"Skipping tasks until: {start_task}")
        
        total_tasks = len(task_sequence)

        for i, (dataset_name, epochs_tuple) in enumerate(task_sequence.items()):
            if should_skip:
                if dataset_name == start_task:
                    should_skip = False
                    if prev_task_name is not None:
                        self.load_task_state(prev_task_name)
                    else:
                        print("Error: Logic error. Start task found but no previous task recorded.")
                else:
                    print(f"Skipping completed task: {dataset_name}")
                    prev_task_name = dataset_name
                    continue 
            
            print(f"\n" + "="*40)
            print(f"[Task {i+1}/{total_tasks}] {dataset_name} | Epochs: {epochs_tuple}")
            print("="*40)
            
            print(">> [Requirement Check] Resetting Text Tuner parameters to BRAND NEW for Phase 1.")
            self.model.text_tuner.reset_parameters()

            self.prepare_frozen_adapters_for_training()

            self.train_task_phases(dataset_name, epochs_tuple)

            is_last = (i == total_tasks - 1)

            self.update_state_after_task(dataset_name, is_last_task=is_last)
            
            self.save_task_state(dataset_name)

    def train_task_phases(self, dataset_name, epoch_tuple):
        if len(epoch_tuple) >= 3:
            e1, e2, e3 = epoch_tuple[:3]
        else:
            e1, e2, e3 = epoch_tuple[0], 0, 0
            
        print(f"\n>>> Starting Training Pipeline for {dataset_name}")
        print(f"    Schedule: Ph1(Joint)={e1} | Ph2(Txt+OrigVis)={e2} | Ph3(Vis+TunedTxt)={e3}")

        self.model.reset_visual_adapters()

        if dataset_name in ["mnist", "eurosat"]:
             cur_transform_train = self.transform_train_special
        else:
             cur_transform_train = self.transform_train

        train_loader, _, _, classnames = build_cur_task_data_loader(
            root=self.cfg.DATA_ROOT,
            dataset_name=dataset_name, 
            transform_train=cur_transform_train, 
            transform_test=self.transform_test, 
            num_shots=-1, 
            batch_size=self.cfg.BATCH_SIZE,
            num_workers=self.cfg.NUM_WORKERS
        )
        
        current_class_names = classnames
        template_func = get_custom_template(dataset_name, is_eval=False)
        current_prompts = [template_func(c.replace("_", " ")) for c in current_class_names]
        train_prompt_path = os.path.join(self.cfg.OUTPUT_DIR, "training_prompts.txt")
        with open(train_prompt_path, "a", encoding="utf-8") as f:
            f.write(f"--- Task: {dataset_name} ---\n")
            for p in current_prompts:
                f.write(p + "\n")
            f.write("\n")
        print(f">> Training prompts for {dataset_name} appended to {train_prompt_path}")
        current_tokenized_prompts = clip.tokenize(current_prompts).to(self.device)
        
        loader = train_loader
        steps_per_epoch = len(loader)
        self.cached_protos = None 
  
        self.adb_manager.load_buffer_for_training(self.inference_manager.seen_tasks)
        
        total_current_epoch = 0

        if e1 > 0:
            print("\n--- Phase 1: Joint Training (Both Tuned) ---")
            self.set_trainable_mode(train_text=True, train_visual=True)
            
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.cfg.LR, epochs=e1, steps_per_epoch=steps_per_epoch
            )
            for ep in range(e1):
                self._run_single_epoch(
                    loader, current_tokenized_prompts, total_current_epoch + 1, 
                    phase_name="Ph1-Joint", train_mode="joint"
                )
                total_current_epoch += 1
            
        if e2 > 0:
            if self.cfg.PHASE2_USE_TUNED_VISUAL:
                phase2_mode = "text_tuned_visual"
                phase2_desc = "Ph2-TextTuned"
                print("\n--- Phase 2: Text Only Training (Using Tuned Visual from Phase 1) ---")
            else:
                phase2_mode = "text_orig_visual"
                phase2_desc = "Ph2-TextOrig"
                print("\n--- Phase 2: Text Only Training (Using Original Visual) ---")
            
            torch.manual_seed(self.cfg.SEED)
            torch.cuda.manual_seed(self.cfg.SEED)
            
            if self.cfg.PHASE2_RESET_TEXT_TUNER:
                print(">> Resetting Text Tuner parameters.")
                self.model.text_tuner.reset_parameters()
            else:
                print(">> Keeping Text Tuner parameters from Phase 1.")

            self.scaler = GradScaler()
            print(">> Resetting AMP GradScaler.")

            self.set_trainable_mode(train_text=True, train_visual=False)
            
            self.cached_protos = None 
            
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.cfg.LR, epochs=e2, steps_per_epoch=steps_per_epoch
            )
            for ep in range(e2):
                self._run_single_epoch(
                    loader, current_tokenized_prompts, total_current_epoch + 1, 
                    phase_name=phase2_desc, train_mode=phase2_mode
                )
                total_current_epoch += 1

        if e3 > 0:
            print("\n--- Phase 3: Visual Only Training (Text Frozen, Tuned Visual) ---")
            self.set_trainable_mode(train_text=False, train_visual=True)
            
            self.cached_protos = None 
            
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=self.cfg.LR, epochs=e3, steps_per_epoch=steps_per_epoch
            )
            for ep in range(e3):
                self._run_single_epoch(
                    loader, current_tokenized_prompts, total_current_epoch + 1, 
                    phase_name="Ph3-Visual", train_mode="visual"
                )
                total_current_epoch += 1


    def update_state_after_task(self, dataset_name, is_last_task=False):
        print(f"Updating State after {dataset_name}. Is Last Task? {is_last_task}")

        if dataset_name in ["mnist", "eurosat"]:
             cur_transform_train = self.transform_train_special
        else:
             cur_transform_train = self.transform_train

        _, train_loader_updating, _, classnames = build_cur_task_data_loader(
            root=self.cfg.DATA_ROOT,
            dataset_name=dataset_name,
            transform_train=cur_transform_train,
            transform_test=self.transform_test,
            num_shots=-1, 
            batch_size=self.cfg.BATCH_SIZE,
            num_workers=self.cfg.NUM_WORKERS
        )

        current_class_names = classnames
        template_func = get_custom_template(dataset_name, is_eval=False)
        current_prompts = [template_func(c.replace("_", " ")) for c in current_class_names]
        current_tokenized_prompts = clip.tokenize(current_prompts).to(self.device)
        
        layer_dims_list = []
        for name in self.cfg.VISUAL_GRAM_SELECT_LAYERS:
            if "resblocks" in name:
                layer_dims_list.append(768)
            elif name == "visual":
                layer_dims_list.append(self.model.ln_final.weight.shape[0]) 
            else:
                layer_dims_list.append(768)
        total_dim = sum(layer_dims_list)
        
        if "visual" in self.cfg.VISUAL_GRAM_SELECT_LAYERS:
            final_dim = self.model.ln_final.weight.shape[0]
            final_layer_start_idx = total_dim - final_dim 
        else:
            final_dim = 0
            final_layer_start_idx = total_dim 

        G_task = torch.zeros(total_dim, total_dim, device=self.device)
        num_classes = len(current_class_names)
        c_task = torch.zeros(total_dim, num_classes, device=self.device) 
        total_samples = 0
        
        if is_last_task:
            print(f"Detecting Last Task ({dataset_name}): Using Fast Global Gram Calculation (No ADB Stats needed)...")
            
            full_loader = train_loader_updating
            
            c_task_sum = torch.zeros(total_dim, num_classes, device=self.device)
            c_task_count = torch.zeros(num_classes, device=self.device)
            
            with torch.no_grad():
                for images, labels in tqdm(full_loader, desc="Global Gram Calc"):
                    images = images.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    feats = self.model.extract_concat_features(images) 
                    
                    G_task += feats.T @ feats
                    c_task_sum.index_add_(1, labels, feats.T)
                    ones = torch.ones(images.size(0), device=self.device)
                    c_task_count.index_add_(0, labels, ones)
                    
                    total_samples += images.size(0)
            
            c_task_count = c_task_count.view(1, -1).clamp(min=1.0) 
            c_task = c_task_sum / c_task_count
            print(f"Global Gram Computed. Total Samples: {total_samples}")

        else:
            print(f"Processing {dataset_name} class-wise (Generating ADB Statistics)...")
            
            print("Collecting data features class-wise...")
            features_by_class = {i: [] for i in range(num_classes)}
            
            with torch.no_grad():
                for images, labels in tqdm(train_loader_updating, desc="Feature Collection"):
                    images = images.to(self.device)
                    feats = self.model.extract_concat_features(images)
                    
                    labels_cpu = labels.cpu().numpy()
                    for i in range(len(labels)):
                        lbl = labels_cpu[i]
                        features_by_class[lbl].append(feats[i:i+1])
            
            current_task_global_offset = self.inference_manager.total_seen_classes

            for cls_idx in tqdm(range(num_classes), desc="Class-wise Processing"):
                if not features_by_class[cls_idx]:
                    continue
                    
                cls_feats = torch.cat(features_by_class[cls_idx], dim=0)
                cls_sample_count = cls_feats.size(0)
                total_samples += cls_sample_count
                
                G_class_big = cls_feats.T @ cls_feats
                c_task[:, cls_idx] = cls_feats.sum(dim=0) / cls_sample_count
                
                G_task += G_class_big
                
                global_label = cls_idx + current_task_global_offset
                
                if final_dim > 0: 
                    G_visual = G_class_big[final_layer_start_idx:, final_layer_start_idx:]
                    mu_visual_mean = c_task[final_layer_start_idx:, cls_idx]
                    mu_visual_accum = mu_visual_mean * cls_sample_count 

                    self.adb_manager.update_class_stats(
                        global_label=global_label,
                        G_class_accum=G_visual,
                        mu_class_accum=mu_visual_accum, 
                        num_samples=cls_sample_count
                    )
                                
                del G_class_big, cls_feats
        
        print(f"Generating final text features for {dataset_name}...")
        with torch.no_grad():
            current_task_text_features = self.model.encode_text(current_tokenized_prompts)

        self.inference_manager.update_state(
            task_name=dataset_name, 
            G_task=G_task,      
            c_task=c_task,      
            num_samples=total_samples,
            class_names=current_class_names,
            task_text_features=current_task_text_features 
        )

        print(f"Updating Visual GPM: Adding basis for task {dataset_name}...")
        self.visual_gpm.add_basis_from_task_gram(G_task, layer_dims_list)

        print(f"Snapshotting Visual Adapters for {dataset_name}...")
        self.inference_manager.save_current_task_adapters(self.model)

        if is_last_task:
            print(">> Last Task Detected: Skipping Text Constraint update and ADB Buffer generation.")
        else:            
            if not is_last_task:
                print(f"GMM Statistics updated for {dataset_name}. ADB ready for next task.")
            
            with torch.no_grad():
                final_text_feats = self.model.encode_text(current_tokenized_prompts)
                final_text_feats = F.normalize(final_text_feats, dim=-1)
                
            if self.seen_text_features is not None:
                self.seen_text_features = torch.cat([self.seen_text_features.to(self.device), final_text_feats], dim=0)
            else:
                self.seen_text_features = final_text_feats

        print(f"State Updated. Total Seen Classes: {self.inference_manager.total_seen_classes}")


class Config:
    def __init__(self, args=None):
        self.SEED = args.seed if args else 0
        
        self.DATA_ROOT = "/root/data"
        self.BACKBONE = "ViT-B/16"
        self.BATCH_SIZE = 128
        self.LR = 0.001
        self.WEIGHT_DECAY = 5e-4
        self.GPU = 0
        self.NUM_WORKERS = 8
        self.T_ADAPTFORMER = True

        base_dir = args.exp_base if args else "./Output_Result_full_shot_order1"
        self.EXP_ROOT = f"{base_dir}_seed_{self.SEED}" 
        
        os.makedirs(self.EXP_ROOT, exist_ok=True)
        print(f"Directory created/located: {self.EXP_ROOT}")

        self.OUTPUT_DIR = self.EXP_ROOT 
        self.GRAM_SAVE_DIR = os.path.join(self.EXP_ROOT, "gram_matrices")
        self.ADB_DIR = os.path.join(self.EXP_ROOT, "adb_features")
        
        self.CHECKPOINT_DIR = os.path.join(self.EXP_ROOT, "checkpoints") 
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)

        os.makedirs(self.GRAM_SAVE_DIR, exist_ok=True)
        os.makedirs(self.ADB_DIR, exist_ok=True)

        self.START_FROM_TASK = None
        self.CONTINUE_TEXT_TUNER = False 

        self.BOUNDARY_LOSS_WEIGHT = 500  
        self.BOUNDARY_GAMMA = 3          
        self.BOUNDARY_MARGIN = 0        

        self.ADB_ENERGY_THRESHOLD = 0.9   

        self.VISUAL_ADAPTER_LAYERS = [
            "visual.transformer.resblocks.7", 
            "visual.transformer.resblocks.8",
            "visual.transformer.resblocks.9", 
            "visual.transformer.resblocks.10",
            "visual.transformer.resblocks.11",
        ]

        self.VISUAL_GRAM_SELECT_LAYERS = [
            "visual.transformer.resblocks.7", 
            "visual.transformer.resblocks.8",
            "visual.transformer.resblocks.9", 
            "visual.transformer.resblocks.10",
            "visual.transformer.resblocks.11",
            "visual"
        ]

        self.VISUAL_ADAPTER_DIM = 768
        self.VISUAL_ALPHA = 0.1
        self.GPM_THRESHOLD = 0.99
        

        self.ORTHO_LOSS_COEFF = 100
        self.T_ADAPTER_DIM = 128
        
        self.PHASE2_USE_TUNED_VISUAL = True
        self.PHASE2_RESET_TEXT_TUNER = False

        
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequential Continual Learning Training')
    parser.add_argument('--dataset', type=str, required=True, help='Current dataset to train')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for this run')
    parser.add_argument('--exp_base', type=str, default='./Output_Results', help='Base directory for output')
    parser.add_argument('--epochs', type=int, nargs='+', default=[5, 0, 0], help='Epochs for Ph1, Ph2, Ph3')
    parser.add_argument('--task_order', type=str, nargs='+', required=True, help='The sequence of tasks')
    args = parser.parse_args()

    TASK_SEQUENCE_LIST = [t.lower() for t in args.task_order]

    cfg = Config(args)

    dataset_name = args.dataset.lower()
    if dataset_name not in TASK_SEQUENCE_LIST:
        raise ValueError(f"Dataset {dataset_name} not in sequence list.")
    
    task_idx = TASK_SEQUENCE_LIST.index(dataset_name)
    
    current_seed = cfg.SEED
    set_seed(current_seed)

    trainer = ContinualTrainer(cfg)
    
    log_file = os.path.join(cfg.EXP_ROOT, f"log_{dataset_name}.txt")
    sys.stdout = Logger(log_file)
    
    print("="*60)
    print(f"Task: {dataset_name} (Index: {task_idx})")
    print(f"Seed: {current_seed}")
    print(f"Time: {datetime.datetime.now()}")
    print("="*60)

    if task_idx > 0:
        prev_dataset = TASK_SEQUENCE_LIST[task_idx - 1]
        print(f">> Automatically attempting to load previous task: {prev_dataset}")
        try:
            trainer.load_task_state(prev_dataset)
            print(f">> Successfully loaded state from {prev_dataset}")
        except FileNotFoundError:
            print(f"!! Error: Checkpoint for {prev_dataset} not found. Ensure previous task finished.")
            sys.exit(1)
    else:
        print(">> First task in sequence. Starting from scratch.")

    epochs_tuple = tuple(args.epochs)
    
    trainer.prepare_frozen_adapters_for_training()
    
    print(f">> Resetting Text Tuner for Phase 1...")
    trainer.model.text_tuner.reset_parameters()

    trainer.train_task_phases(dataset_name, epochs_tuple)

    is_last = (task_idx == len(TASK_SEQUENCE_LIST) - 1)
    trainer.update_state_after_task(dataset_name, is_last_task=is_last)
    
    trainer.save_task_state(dataset_name)
    print(f">> Task {dataset_name} completed and state saved.")
