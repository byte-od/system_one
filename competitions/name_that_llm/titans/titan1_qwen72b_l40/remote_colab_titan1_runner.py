#!/usr/bin/env python3
"""
Remote Google Colab Runner for TITAN 1:
👑 Qwen 2.5 72B MoS Golden-Slice (Layers 1 -> 40)
Executed directly on Colab NVIDIA GPU via Colab CLI.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Configuração de credenciais Kaggle
os.environ["KAGGLE_API_TOKEN"] = "KGAT_51b5f3784b45c50dd053aa54332d3faa"
os.environ["KAGGLE_USERNAME"] = "thetod"
os.environ["KAGGLE_KEY"] = "51b5f3784b45c50dd053aa54332d3faa"
kg_dir = Path.home() / ".kaggle"
kg_dir.mkdir(parents=True, exist_ok=True)
with open(kg_dir / "kaggle.json", "w") as f:
    f.write('{"username":"thetod","key":"51b5f3784b45c50dd053aa54332d3faa"}')
os.chmod(kg_dir / "kaggle.json", 0o600)

print("=" * 80)
print("👑 INICIANDO PIPELINE TITAN 1: QWEN 2.5 72B MoS (CAMADAS 1 -> 40) NO GOOGLE COLAB")
print("=" * 80, flush=True)

# 1. Instalação de dependências essenciais
print("\n[Passo 1/8] Verificando e instalando dependências...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "safetensors", "transformers", "accelerate", "scikit-learn", "kaggle"])

import gc
import json
import re
import glob
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, log_loss

assert torch.cuda.is_available(), "ERRO CRÍTICO: CUDA não disponível no ambiente Colab!"
gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"✓ GPU Ativa: {gpu_name} ({vram_gb:.1f} GB VRAM) | PyTorch {torch.__version__}", flush=True)

# 2. Configura diretórios de trabalho
DATA_DIR = Path("/content/data")
MODEL_DIR = Path("/content/qwen25_72b_mos")
HEAD_DIR = Path("/content/head_artifacts_70b_l40")
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
HEAD_DIR.mkdir(parents=True, exist_ok=True)

# 3. Download do dataset oficial train.csv
print("\n[Passo 2/8] Baixando train.csv da competição Name That LLM...", flush=True)
train_csv_path = DATA_DIR / "train.csv"
if not train_csv_path.exists():
    cmd_dl_train = "kaggle competitions download -c name-that-llm -f train.csv -p /content/data"
    os.system(cmd_dl_train)
    for zf in DATA_DIR.glob("*.zip"):
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(DATA_DIR)
        print(f"  ✓ Extraído: {zf.name}", flush=True)

assert train_csv_path.exists(), f"ERRO: train.csv não foi encontrado em {train_csv_path}"
df_train = pd.read_csv(train_csv_path)
n_samples = len(df_train)
print(f"✓ {n_samples} passagens de treino carregadas com sucesso de {train_csv_path}", flush=True)

# 4. Download dos Shards MoS do Qwen 72B
print("\n[Passo 3/8] Verificando e desaninhando shards MoS 72B...", flush=True)
sub_nested = MODEL_DIR / "qwen25_72b_mos"
if sub_nested.exists():
    for f in list(sub_nested.iterdir()):
        dest = MODEL_DIR / f.name
        if not dest.exists():
            shutil.move(str(f), str(dest))
    print("  ✓ Estrutura de diretórios desaninhada com sucesso!", flush=True)

emb_file = MODEL_DIR / "mos_embeddings_and_head.safetensors"
if not emb_file.exists():
    print("  Baixando do Kaggle via API...", flush=True)
    os.system("kaggle kernels output thetod/qwen-72b-to-mixture-of-scars-mos-converter -p /content/qwen25_72b_mos")
    if sub_nested.exists():
        for f in list(sub_nested.iterdir()):
            dest = MODEL_DIR / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))

# Download de tokenizers oficiais se necessário
if not (MODEL_DIR / "tokenizer.json").exists():
    print("  Baixando tokenizer oficial Qwen 2.5 72B do HuggingFace...", flush=True)
    from huggingface_hub import hf_hub_download
    for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        try:
            hf_hub_download("Qwen/Qwen2.5-72B-Instruct", fn, local_dir=str(MODEL_DIR))
        except Exception:
            pass

shards_found = sorted(list(MODEL_DIR.glob("*.safetensors")))
print(f"✓ {len(shards_found)} shards safetensors MoS prontos em {MODEL_DIR}", flush=True)

# 5. Definições do Motor MoS CUDA
OFFICIAL_LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "kimi-k3", "qwen3.8-max"
]
L2I = {l: i for i, l in enumerate(OFFICIAL_LABELS)}

def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, shape: tuple, dtype: torch.dtype = torch.float16, target_device: Optional[torch.device] = None) -> torch.Tensor:
    dev = target_device or packed.device
    p = packed.to(dev)
    s = scales.to(dev).to(dtype)
    low = (p & 0x0F).to(torch.int16) - 8
    high = ((p >> 4) & 0x0F).to(torch.int16) - 8
    unpacked = torch.empty((p.shape[0], p.shape[1] * 2), dtype=dtype, device=dev)
    unpacked[:, 0::2] = low.to(dtype)
    unpacked[:, 1::2] = high.to(dtype)
    return (unpacked * s).view(shape)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

def precompute_freqs_cis(position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    dim = 128
    theta = 1000000.0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: (dim // 2)] / dim))
    freqs = torch.outer(position_ids.reshape(-1).float(), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    freqs_cis = torch.cat([freqs_cis, freqs_cis], dim=-1)
    cos = freqs_cis.cos().to(dtype).reshape(position_ids.shape[0], 1, position_ids.shape[1], dim)
    sin = freqs_cis.sin().to(dtype).reshape(position_ids.shape[0], 1, position_ids.shape[1], dim)
    return cos, sin

class Qwen72BMoSGroup:
    def __init__(self, group_idx: int, checkpoint_path: Path, device: torch.device, dtype: torch.dtype = torch.float16, scar_scale: float = 0.10):
        self.group_idx = group_idx
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype
        self.scar_scale = scar_scale
        self.start_layer = group_idx * 8
        self.end_layer = self.start_layer + 8
        self.base_gate = None
        self.base_up = None
        self.base_down = None
        self.layers_data = {l: {} for l in range(self.start_layer, self.end_layer)}
        self.is_loaded = False

    def load(self):
        if self.is_loaded: return
        with safe_open(str(self.checkpoint_path), framework="pt", device="cpu") as f:
            bg_p = f.get_tensor(f"mos_base.group_{self.group_idx}.mlp.gate_proj.weight")
            bg_s = f.get_tensor(f"mos_base.group_{self.group_idx}.mlp.gate_proj.weight.scales")
            self.base_gate = dequantize_int4(bg_p, bg_s, (29568, 8192), self.dtype, self.device)

            bu_p = f.get_tensor(f"mos_base.group_{self.group_idx}.mlp.up_proj.weight")
            bu_s = f.get_tensor(f"mos_base.group_{self.group_idx}.mlp.up_proj.weight.scales")
            self.base_up = dequantize_int4(bu_p, bu_s, (29568, 8192), self.dtype, self.device)

            bd_p = f.get_tensor(f"mos_base.group_{self.group_idx}.mlp.down_proj.weight")
            bd_s = f.get_tensor(f"mos_base.group_{self.group_idx}.mlp.down_proj.weight.scales")
            self.base_down = dequantize_int4(bd_p, bd_s, (8192, 29568), self.dtype, self.device)

            for l in range(self.start_layer, self.end_layer):
                d = {}
                d["input_layernorm"] = f.get_tensor(f"model.layers.{l}.input_layernorm.weight").to(self.dtype).to(self.device)
                d["post_attention_layernorm"] = f.get_tensor(f"model.layers.{l}.post_attention_layernorm.weight").to(self.dtype).to(self.device)
                for proj, shape in [("q_proj", (8192, 8192)), ("k_proj", (1024, 8192)), ("v_proj", (1024, 8192)), ("o_proj", (8192, 8192))]:
                    p_p = f.get_tensor(f"model.layers.{l}.self_attn.{proj}.weight")
                    p_s = f.get_tensor(f"model.layers.{l}.self_attn.{proj}.weight.scales")
                    d[proj] = dequantize_int4(p_p, p_s, shape, self.dtype, self.device)
                for proj in ["q_proj", "k_proj", "v_proj"]:
                    bias_k = f"model.layers.{l}.self_attn.{proj}.bias"
                    d[f"{proj}_bias"] = f.get_tensor(bias_k).to(self.dtype).to(self.device) if bias_k in f.keys() else None

                if l > self.start_layer:
                    for proj in ["gate_proj", "up_proj", "down_proj"]:
                        d[f"{proj}_scar_A"] = f.get_tensor(f"model.layers.{l}.mlp.{proj}.expert.A").to(self.dtype).to(self.device)
                        d[f"{proj}_scar_B"] = f.get_tensor(f"model.layers.{l}.mlp.{proj}.expert.B").to(self.dtype).to(self.device)
                self.layers_data[l] = d
        self.is_loaded = True

    def unload(self):
        self.base_gate = None
        self.base_up = None
        self.base_down = None
        self.layers_data = {l: {} for l in range(self.start_layer, self.end_layer)}
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward_layer(self, l: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d = self.layers_data[l]
        normed = x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + 1e-6).to(self.dtype)
        normed = normed * d["input_layernorm"]

        q = F.linear(normed, d["q_proj"], d["q_proj_bias"])
        k = F.linear(normed, d["k_proj"], d["k_proj_bias"])
        v = F.linear(normed, d["v_proj"], d["v_proj_bias"])

        bsz, seq_len, _ = q.shape
        q = q.view(bsz, seq_len, 64, 128).transpose(1, 2)
        k = k.view(bsz, seq_len, 8, 128).transpose(1, 2)
        v = v.view(bsz, seq_len, 8, 128).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k_rep = k.repeat_interleave(8, dim=1)
        v_rep = v.repeat_interleave(8, dim=1)

        is_causal = (q.shape[2] > 1 and q.shape[2] == k_rep.shape[2])
        attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, 8192)
        attn_out = F.linear(attn_out, d["o_proj"])
        x = x + attn_out

        normed_mlp = x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + 1e-6).to(self.dtype)
        normed_mlp = normed_mlp * d["post_attention_layernorm"]

        if l == self.start_layer:
            gate = F.silu(F.linear(normed_mlp, self.base_gate))
            up = F.linear(normed_mlp, self.base_up)
            mlp_out = F.linear(gate * up, self.base_down)
        else:
            gate_base = F.linear(normed_mlp, self.base_gate)
            gate_scar = ((normed_mlp @ d["gate_proj_scar_A"]) @ d["gate_proj_scar_B"]) * self.scar_scale
            gate = F.silu(gate_base + gate_scar)

            up_base = F.linear(normed_mlp, self.base_up)
            up_scar = ((normed_mlp @ d["up_proj_scar_A"]) @ d["up_proj_scar_B"]) * self.scar_scale
            up = up_base + up_scar

            inter = gate * up
            down_base = F.linear(inter, self.base_down)
            down_scar = ((inter @ d["down_proj_scar_A"]) @ d["down_proj_scar_B"]) * self.scar_scale
            mlp_out = down_base + down_scar

        return x + mlp_out

# 6. Extração de Estilometria e Embeddings
def extract_tricompass_style_features(texts: List[str]) -> np.ndarray:
    feats = []
    for t in texts:
        t = str(t)
        n = max(len(t), 1); words = t.split(); nw = max(len(words), 1)
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]; ns = max(len(sents), 1)
        lines = [l for l in t.split("\n") if l.strip()]; nl = max(len(lines), 1)
        feats.append([
            t.count(",") / n, t.count(":") / n, t.count(";") / n, t.count("—") / n, t.count(" — ") / n,
            t.count("-") / n, t.count("--") / n, t.count('"') / n, t.count("'") / n, t.count("`") / n, t.count("```") / nw,
            t.count("(") / n, t.count(")") / n, t.count("[") / n, t.count("]") / n, t.count("{") / n, t.count("}") / n,
            t.count("#") / n, t.count("*") / n, (t.count("\n- ") + t.count("\n* ")) / nl,
            sum(c.isdigit() for c in t) / n, sum(c.isupper() for c in t) / n,
            sum(w[0].isupper() for w in words if len(w) > 0) / nw,
            t.count("\n") / nw, t.count("\n\n") / nw,
            sum(len(w) for w in words) / nw, nw / ns, len(set(w.lower() for w in words)) / nw
        ])
    return np.asarray(feats, dtype=np.float32)

def center_groups_delta_origin(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    Xc = X.copy()
    for g in np.unique(groups):
        m = (groups == g)
        if m.sum() > 1:
            Xc[m] = Xc[m] - Xc[m].mean(axis=0, keepdims=True)
    return Xc

texts = df_train["text"].tolist()
labels = df_train["model"].map(L2I).values
prompts = df_train["prompt_id"].values

print("\n[Passo 4/8] Extraindo 28 assinaturas estilométricas da Bússola 1...", flush=True)
t_s0 = time.time()
style_raw = extract_tricompass_style_features(texts)
print(f"  ✓ Estilometria concluída em {time.time()-t_s0:.2f}s: shape {style_raw.shape}", flush=True)

# 7. Execução dos 5 Grupos MoS (Camadas 0 a 39)
print("\n[Passo 5/8] Extraindo representações neurais da Fatia Dourada (Camadas 1 -> 40)...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cuda:0")
dtype = torch.float16
batch_size = 16
max_len = 128
num_batches = (n_samples + batch_size - 1) // batch_size

enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
input_ids = enc.input_ids
attn_mask = enc.attention_mask.unsqueeze(-1)

with safe_open(str(emb_file), framework="pt", device="cpu") as f:
    tk = "model.embed_tokens.weight" if "model.embed_tokens.weight" in f.keys() else "embed_tokens.weight"
    emb_p = f.get_tensor(tk)
    emb_s = f.get_tensor(f"{tk}.scales")

bpt = 8192 // 128
boff = torch.arange(bpt, dtype=torch.long)
print(f"  Alocando buffer de embeddings X_cpu ({n_samples}, {max_len}, 8192)...", flush=True)
X_cpu = torch.empty((n_samples, max_len, 8192), dtype=dtype)

batch_emb = 256
t_emb = time.time()
for i in range(0, n_samples, batch_emb):
    end_i = min(i + batch_emb, n_samples)
    cur_ids = input_ids[i:end_i]
    bidx = (cur_ids.unsqueeze(-1) * bpt + boff).view(-1)
    cur_p = emb_p[bidx]
    cur_s = emb_s[bidx]
    X_cpu[i:end_i] = dequantize_int4(cur_p, cur_s, (end_i - i, max_len, 8192), dtype=dtype, target_device=torch.device("cpu"))
    del bidx, cur_p, cur_s

del emb_p, emb_s
gc.collect()
print(f"  ✓ Buffer de embeddings inicializado com sucesso em {time.time()-t_emb:.2f}s: {X_cpu.shape}", flush=True)

pos = torch.arange(max_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
cos, sin = precompute_freqs_cis(pos, device, dtype)

t_groups_all = time.time()
for g in range(5):
    tg = time.time()
    grp = Qwen72BMoSGroup(g, MODEL_DIR / f"mos_group_{g}.safetensors", device, dtype=dtype, scar_scale=0.10)
    grp.load()

    with torch.no_grad():
        for b in range(num_batches):
            if b % 35 == 0 or b == num_batches - 1:
                print(f"    [Grupo {g} Camadas {grp.start_layer}..{grp.end_layer-1}] Lote {b+1}/{num_batches}...", flush=True)
            s_idx = b * batch_size
            e_idx = min(s_idx + batch_size, n_samples)
            cur_bsz = e_idx - s_idx
            x_b = X_cpu[s_idx:e_idx].to(device)

            cur_cos = cos[:cur_bsz]
            cur_sin = sin[:cur_bsz]
            for l in range(grp.start_layer, grp.end_layer):
                x_b = grp.forward_layer(l, x_b, cur_cos, cur_sin)

            X_cpu[s_idx:e_idx] = x_b.cpu()

    grp.unload()
    torch.cuda.empty_cache()
    print(f"  ✓ Grupo {g} (Camadas {g*8:02d}..{g*8+7:02d}) concluído em {time.time()-tg:.2f}s", flush=True)

print(f"⚡ Todas as 40 camadas MoS processadas em {time.time()-t_groups_all:.2f}s!", flush=True)

# Pooling com RMSNorm
print("  Aplicando pooling ponderado na camada 40 em lotes...", flush=True)
features_70b_l40 = np.zeros((n_samples, 8192), dtype=np.float32)
for b in range(num_batches):
    s_idx = b * batch_size
    e_idx = min(s_idx + batch_size, n_samples)
    x_chunk = X_cpu[s_idx:e_idx].float()
    m_chunk = attn_mask[s_idx:e_idx].float()
    den_chunk = m_chunk.sum(dim=1).clamp(min=1)
    p_chunk = (x_chunk * m_chunk).sum(dim=1) / den_chunk
    rms_chunk = torch.rsqrt(p_chunk.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    features_70b_l40[s_idx:e_idx] = (p_chunk * rms_chunk).numpy()

del X_cpu
gc.collect()

np.save(str(HEAD_DIR / "features_70b_l40_train_full.npy"), features_70b_l40)
print(f"✓ Matriz de representações 70B Camada 40 salva: shape {features_70b_l40.shape}", flush=True)

# 8. Validação Cruzada 5-Fold & Soft Sinkhorn
def sinkhorn(P: np.ndarray, iters: int = 300) -> np.ndarray:
    N, C = P.shape
    r = np.ones(N) / N
    c = np.ones(C) / C
    K = P.copy() + 1e-12
    u = np.ones(N)
    for _ in range(iters):
        v = c / (K.T @ u).clip(min=1e-12)
        u = r / (K @ v).clip(min=1e-12)
    P_cal = u[:, None] * K * v[None, :]
    return P_cal / P_cal.sum(axis=1, keepdims=True)

print("\n[Passo 6/8] Executando Validação Cruzada 5-Fold Stratified Group K-Fold & Soft Sinkhorn...", flush=True)
scaler_style_global = StandardScaler()
style_normed = scaler_style_global.fit_transform(style_raw)

Zc = center_groups_delta_origin(features_70b_l40, prompts)
Sc = center_groups_delta_origin(style_normed, prompts)

X_plain_raw = np.hstack([features_70b_l40, style_normed])
X_centered_raw = np.hstack([Zc, Sc])

sgkf = StratifiedGroupKFold(n_splits=5)
oof_ensemble_probs = np.zeros((n_samples, 18), dtype=np.float32)
fold_accs = []
fold_sinkhorn_accs = []

for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_plain_raw, labels, groups=prompts)):
    t_fold = time.time()
    scaler_p = StandardScaler()
    X_tr_p = scaler_p.fit_transform(X_plain_raw[trn_idx])
    X_va_p = scaler_p.transform(X_plain_raw[val_idx])

    clf_p = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    clf_p.fit(X_tr_p, labels[trn_idx])
    p_val_plain = clf_p.predict_proba(X_va_p)

    scaler_c = StandardScaler()
    X_tr_c = scaler_c.fit_transform(X_centered_raw[trn_idx])
    X_va_c = scaler_c.transform(X_centered_raw[val_idx])

    clf_c = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    clf_c.fit(X_tr_c, labels[trn_idx])
    p_val_twin = clf_c.predict_proba(X_va_c)

    p_val_ens = 0.8 * p_val_plain + 0.2 * p_val_twin
    p_val_ens = p_val_ens / p_val_ens.sum(axis=1, keepdims=True)
    oof_ensemble_probs[val_idx] = p_val_ens

    acc_raw = accuracy_score(labels[val_idx], np.argmax(p_val_ens, axis=1))
    p_val_sink = sinkhorn(p_val_ens)
    acc_sink = accuracy_score(labels[val_idx], np.argmax(p_val_sink, axis=1))

    fold_accs.append(acc_raw)
    fold_sinkhorn_accs.append(acc_sink)
    print(f"  Fold {fold+1}/5: Raw Acc = {acc_raw*100:.2f}% | Soft Sinkhorn Acc = {acc_sink*100:.2f}% ({time.time()-t_fold:.2f}s)", flush=True)

overall_raw = accuracy_score(labels, np.argmax(oof_ensemble_probs, axis=1))
oof_sinkhorn = sinkhorn(oof_ensemble_probs)
overall_sink = accuracy_score(labels, np.argmax(oof_sinkhorn, axis=1))
overall_logloss = log_loss(labels, oof_sinkhorn)

print("\n" + "=" * 80)
print("🏆 RESULTADOS GLOBAIS OOF TITAN 1 (QWEN 72B MoS CAMADA 40):")
print(f"   • Acurácia OOF Raw (Ensemble):        {overall_raw*100:.2f}%")
print(f"   • Acurácia OOF Calibrada (Sinkhorn):  {overall_sink*100:.2f}%")
print(f"   • Log Loss Calibrado:                 {overall_logloss:.4f}")
print("=" * 80, flush=True)

# 9. Refit no dataset 100% oficial e compilação dos pesos
print("\n[Passo 7/8] Ajustando pesos Plain e Centered em 100% dos dados...", flush=True)
scaler_plain_final = StandardScaler()
X_plain_sc = scaler_plain_final.fit_transform(X_plain_raw)
clf_plain_final = LogisticRegression(C=0.1, max_iter=500, random_state=42)
clf_plain_final.fit(X_plain_sc, labels)

scaler_centered_final = StandardScaler()
X_centered_sc = scaler_centered_final.fit_transform(X_centered_raw)
clf_centered_final = LogisticRegression(C=0.1, max_iter=500, random_state=42)
clf_centered_final.fit(X_centered_sc, labels)

train_acc = accuracy_score(labels, clf_plain_final.predict(X_plain_sc))
print(f"✓ Acurácia de Treino Final (Plain): {train_acc*100:.2f}%", flush=True)

head_weights = {
    "style_scaler.mean": torch.from_numpy(scaler_style_global.mean_.astype(np.float32)),
    "style_scaler.scale": torch.from_numpy(scaler_style_global.scale_.astype(np.float32)),
    "scaler_plain.mean": torch.from_numpy(scaler_plain_final.mean_.astype(np.float32)),
    "scaler_plain.scale": torch.from_numpy(scaler_plain_final.scale_.astype(np.float32)),
    "plain.weight": torch.from_numpy(clf_plain_final.coef_.astype(np.float32)),
    "plain.bias": torch.from_numpy(clf_plain_final.intercept_.astype(np.float32)),
    "scaler_centered.mean": torch.from_numpy(scaler_centered_final.mean_.astype(np.float32)),
    "scaler_centered.scale": torch.from_numpy(scaler_centered_final.scale_.astype(np.float32)),
    "centered.weight": torch.from_numpy(clf_centered_final.coef_.astype(np.float32)),
    "centered.bias": torch.from_numpy(clf_centered_final.intercept_.astype(np.float32)),
}

save_file(head_weights, str(HEAD_DIR / "head_tricompass_70b_l40.safetensors"))
save_file(head_weights, str(HEAD_DIR / "head_tricompass_70b.safetensors"))
print("✓ Safetensors salvos: head_tricompass_70b_l40.safetensors e head_tricompass_70b.safetensors", flush=True)

meta = {
    "architecture": "Titan1_Qwen72B_MoS_Layer40",
    "golden_slice": "Layers 1 -> 40 (Half Depth)",
    "features_dim": int(X_plain_raw.shape[1]),
    "hidden_dim": 8192,
    "style_dim": 28,
    "labels": OFFICIAL_LABELS,
    "overall_oof_raw_acc": float(overall_raw),
    "overall_oof_sinkhorn_acc": float(overall_sink),
    "overall_oof_logloss": float(overall_logloss),
    "fold_accs": [float(x) for x in fold_accs],
    "fold_sinkhorn_accs": [float(x) for x in fold_sinkhorn_accs],
    "train_acc": float(train_acc),
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
}
with open(HEAD_DIR / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)
with open(HEAD_DIR / "meta_tricompass_70b.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"✓ Metadatas salvos em {HEAD_DIR}", flush=True)

# 10. Atualização do Kaggle Dataset
print("\n[Passo 8/8] Empacotando e atualizando Kaggle Dataset thetod/name-that-llm-70b-mos-tricompass-head...", flush=True)
zip_path = Path("/content/head_artifacts_70b_l40.zip")
shutil.make_archive("/content/head_artifacts_70b_l40", "zip", str(HEAD_DIR))
print(f"  ✓ Arquivo ZIP gerado: {zip_path} ({zip_path.stat().st_size / (1024**2):.1f} MB)", flush=True)

dataset_metadata = {
    "title": "Name That LLM - 70B MoS TriCompass Head",
    "id": "thetod/name-that-llm-70b-mos-tricompass-head",
    "licenses": [{"name": "apache-2.0"}]
}
with open(HEAD_DIR / "dataset-metadata.json", "w") as f:
    json.dump(dataset_metadata, f, indent=2)
with open(HEAD_DIR / "datasets-metadata.json", "w") as f:
    json.dump(dataset_metadata, f, indent=2)

cmd_kg_push = f"kaggle datasets version -p {HEAD_DIR} -m 'Titan 1: Qwen 72B MoS Layer 40 Golden Slice' -r zip"
os.system(cmd_kg_push)
print("\n✨ TITAN 1 EXECUTADO E FINALIZADO COM SUCESSO ABSOLUTO!", flush=True)
