#!/usr/bin/env python3
"""
Colab Remote Execution Script: Tri-Compass + 70B MoS Scar Classifier Training.
Runs directly on the remote Tesla T4 GPU VM via colab CLI.
"""

import os
import builtins
_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    return _orig_print(*args, **kwargs)
import sys
import time
import gc
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, log_loss

# 1. INT4 Dequantizer
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

# 2. MoS Group Loader with Calibrated Scar Scale (gamma = 0.10)
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
        with safe_open(self.checkpoint_path, framework="pt", device="cpu") as f:
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

                # Scars apenas se for camada adaptada (l > start_layer)
                if l > self.start_layer:
                    for proj in ["gate_proj", "up_proj", "down_proj"]:
                        d[f"{proj}_scar_A"] = f.get_tensor(f"model.layers.{l}.mlp.{proj}.expert.A").to(self.dtype).to(self.device)
                        d[f"{proj}_scar_B"] = f.get_tensor(f"model.layers.{l}.mlp.{proj}.expert.B").to(self.dtype).to(self.device)
                self.layers_data[l] = d
        self.is_loaded = True

    def unload(self):
        if not self.is_loaded: return
        self.base_gate = None
        self.base_up = None
        self.base_down = None
        self.layers_data = {l: {} for l in range(self.start_layer, self.end_layer)}
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward_layer(self, l: int, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        d = self.layers_data[l]
        bsz, seq_len, _ = x.shape
        # Attention RMSNorm
        normed = x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + 1e-6).to(self.dtype)
        normed = normed * d["input_layernorm"]
        q = F.linear(normed, d["q_proj"], d["q_proj_bias"])
        k = F.linear(normed, d["k_proj"], d["k_proj_bias"])
        v = F.linear(normed, d["v_proj"], d["v_proj_bias"])

        q = q.view(bsz, seq_len, 64, 128).transpose(1, 2)
        k = k.view(bsz, seq_len, 8, 128).transpose(1, 2)
        v = v.view(bsz, seq_len, 8, 128).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_rep = k.repeat_interleave(64 // 8, dim=1)
        v_rep = v.repeat_interleave(64 // 8, dim=1)

        is_causal = (q.shape[2] > 1 and q.shape[2] == k_rep.shape[2])
        attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, 8192)
        attn_proj = F.linear(attn_out, d["o_proj"])
        x = x + attn_proj

        # MoS SwiGLU with Calibrated Scar Scaling
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
            down_scar = (((inter @ d["down_proj_scar_A"]) @ d["down_proj_scar_B"])) * self.scar_scale
            mlp_out = down_base + down_scar

        x = x + mlp_out
        return x

# 3. Complete Engine
class Qwen72BMoSEngine:
    def __init__(self, model_dir: Union[str, Path], device: str = "cuda", dtype: torch.dtype = torch.float16, scar_scale: float = 0.10):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.dtype = dtype
        self.scar_scale = scar_scale

        print(f"🚀 Inicializando Qwen 2.5 72B MoS Engine no dispositivo: {str(self.device).upper()} ({dtype}, scar_scale={scar_scale})...")
        emb_file = self.model_dir / "mos_embeddings_and_head.safetensors"
        if not emb_file.exists():
            emb_file = self.model_dir / "mos_embeddings.safetensors"

        print(f"📖 Carregando embeddings de {emb_file.name}...")
        with safe_open(emb_file, framework="pt", device="cpu") as f:
            self.emb_packed = f.get_tensor("model.embed_tokens.weight")
            self.emb_scales = f.get_tensor("model.embed_tokens.weight.scales")
            if "model.norm.weight" in f.keys():
                self.final_norm_weight = f.get_tensor("model.norm.weight").to(self.dtype).to(self.device)

        if not hasattr(self, "final_norm_weight"):
            with safe_open(self.model_dir / "mos_final_norm.safetensors", framework="pt", device="cpu") as f:
                self.final_norm_weight = f.get_tensor("model.norm.weight").to(self.dtype).to(self.device)

        self.groups = [Qwen72BMoSGroup(g, self.model_dir / f"mos_group_{g}.safetensors", self.device, self.dtype, scar_scale=self.scar_scale) for g in range(10)]
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("✓ Tokenizer e 10 Grupos MoS registrados com sucesso!")

    def precompute_freqs_cis(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = 128; theta = 1000000.0
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=self.device)[: (dim // 2)] / dim))
        freqs = torch.outer(position_ids.reshape(-1).float(), freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        freqs_cis = torch.cat([freqs_cis, freqs_cis], dim=-1)
        cos = freqs_cis.cos().to(self.dtype).reshape(position_ids.shape[0], 1, position_ids.shape[1], dim)
        sin = freqs_cis.sin().to(self.dtype).reshape(position_ids.shape[0], 1, position_ids.shape[1], dim)
        return cos, sin

OFFICIAL_LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
L2I = {l: i for i, l in enumerate(OFFICIAL_LABELS)}

def extract_tricompass_style_features(texts: List[str]) -> np.ndarray:
    feats = []
    for t in texts:
        t = str(t)
        n = max(len(t), 1); words = t.split(); nw = max(len(words), 1)
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]; ns = max(len(sents), 1)
        lines = [l for l in t.split("\n") if l.strip()]; nl = max(len(lines), 1)
        feats.append([
            t.count(",")/n, t.count(":")/n, t.count(";")/n, t.count("—")/n, t.count(" — ")/n,
            t.count("-")/n, t.count("--")/n, t.count('"')/n, t.count("'")/n, t.count("`")/n, t.count("```")/nw,
            t.count("(")/n, t.count(")")/n, t.count("[")/n, t.count("]")/n, t.count("{")/n, t.count("}")/n,
            t.count("#")/n, t.count("*")/n, (t.count("\n- ") + t.count("\n* "))/nl,
            sum(c.isdigit() for c in t)/n, sum(c.isupper() for c in t)/n,
            sum(w[0].isupper() for w in words if len(w) > 0)/nw,
            t.count("\n")/nw, t.count("\n\n")/nw,
            sum(len(w) for w in words)/nw, nw/ns, len(set(w.lower() for w in words))/nw
        ])
    return np.asarray(feats, dtype=np.float32)

def center_groups_delta_origin(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    Xc = X.copy()
    for g in np.unique(groups):
        m = (groups == g)
        if m.sum() > 1:
            Xc[m] = Xc[m] - Xc[m].mean(axis=0, keepdims=True)
    return Xc

def main():
    print("=" * 70)
    print("🏆 TREINAMENTO TRI-COMPASS + 70B MoS SCAR CLASSIFIER NO COLAB GPU")
    print("=" * 70)

    train_path = "/content/train.csv"
    if not os.path.exists(train_path):
        train_path = "/content/data/train.csv"
    df_train = pd.read_csv(train_path)
    print(f"✓ {len(df_train)} passagens de treino carregadas!")
    train_texts = df_train["text"].tolist()
    train_labels = df_train["model"].map(L2I).values
    train_prompts = df_train["prompt_id"].values

    # Bússola 1: Estilometria Local (28D)
    print("\n[Bússola 1/3] Extraindo 28 features estilométricas...")
    style_raw = extract_tricompass_style_features(train_texts)
    scaler_style = StandardScaler()
    style_normed = scaler_style.fit_transform(style_raw)

    # Localiza o diretório dos pesos (trata subpasta se criada pelo Kaggle CLI)
    model_dir = Path("/content/qwen25_72b_mos")
    if (model_dir / "qwen25_72b_mos").exists() and (model_dir / "qwen25_72b_mos" / "mos_group_0.safetensors").exists():
        model_dir = model_dir / "qwen25_72b_mos"
    print(f"✓ Diretório do modelo 70B MoS localizado: {model_dir}")

    # Baixa arquivos auxiliares do tokenizer se necessário
    if not (model_dir / "tokenizer.json").exists():
        print("📥 Baixando arquivos do tokenizer Qwen 2.5 72B...")
        from huggingface_hub import hf_hub_download
        for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
            try:
                hf_hub_download("Qwen/Qwen2.5-72B-Instruct", fn, local_dir=str(model_dir))
            except Exception:
                pass

    # Inicia Engine
    engine = Qwen72BMoSEngine(model_dir, device="cuda", scar_scale=0.10)

    # Bússola 3: Embeddings 70B MoS (8.192D)
    print("\n[Bússola 3/3] Extraindo representações 70B MoS na GPU CUDA (batch_size=32)...")
    n_samples = len(train_texts)
    batch_size = 32
    max_length = 256
    features_70b = np.zeros((n_samples, 8192), dtype=np.float32)
    t0 = time.time()

    for b in range(0, n_samples, batch_size):
        tb_start = time.time()
        b_texts = train_texts[b : b + batch_size]
        act_bs = len(b_texts)
        inputs = engine.tokenizer(b_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = inputs.input_ids.to(engine.device)
        attn_mask = inputs.attention_mask.to(engine.device).unsqueeze(-1)
        bsz, seq_len = input_ids.shape

        bpt = 8192 // 128
        boff = torch.arange(bpt, dtype=torch.long)
        all_bidx = (input_ids.cpu().unsqueeze(-1) * bpt + boff).view(-1)
        x = dequantize_int4(engine.emb_packed[all_bidx], engine.emb_scales[all_bidx], (bsz, seq_len, 8192), engine.dtype, engine.device)
        pos = torch.arange(seq_len, dtype=torch.long, device=engine.device).unsqueeze(0).expand(bsz, -1)
        cos, sin = engine.precompute_freqs_cis(pos)

        with torch.no_grad():
            for g in engine.groups:
                g.load()
                for l in range(g.start_layer, g.end_layer):
                    x = g.forward_layer(l, x, cos, sin)
                g.unload()
                torch.cuda.empty_cache()
            den = attn_mask.sum(dim=1).clamp(min=1)
            pooled = (x.float() * attn_mask.float()).sum(dim=1) / den
            rms = torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            features_70b[b : b + act_bs] = (pooled * rms).cpu().numpy()

        done = min(b + act_bs, n_samples)
        rate = done / max(time.time() - t0, 0.001)
        eta_min = (n_samples - done) / max(rate, 0.001) / 60
        print(f"  [{done:4d}/{n_samples}] Lote concluído em {time.time()-tb_start:.1f}s | Velocidade: {rate:.1f} txt/s | ETA: {eta_min:.1f} min")

    np.save("/content/features_70b_train_full.npy", features_70b)
    print("\n✓ Todas as 2.699 representações de 70B MoS extraídas e salvas!")

    # Bússola 2: Delta Origin Prompt Centering
    print("\n[Bússola 2/3] Calculando Prompt Centering Delta Origin...")
    style_centered = center_groups_delta_origin(style_normed, train_prompts)
    features_70b_centered = center_groups_delta_origin(features_70b, train_prompts)

    X_plain_raw = np.hstack([features_70b, style_normed])
    X_centered_raw = np.hstack([features_70b_centered, style_centered])

    scaler_plain = StandardScaler()
    X_plain = scaler_plain.fit_transform(X_plain_raw)
    scaler_centered = StandardScaler()
    X_centered = scaler_centered.fit_transform(X_centered_raw)

    print(f"\n🏆 Executando 5-Fold Stratified GroupKFold por prompt_id ({X_plain.shape[1]} dims)...\n")
    sgkf = StratifiedGroupKFold(n_splits=5)
    cv_accs, cv_losses = [], []

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_plain, train_labels, groups=train_prompts)):
        clf_p = LogisticRegression(C=0.1, max_iter=300, random_state=42)
        clf_p.fit(X_plain[trn_idx], train_labels[trn_idx])

        clf_c = LogisticRegression(C=0.1, max_iter=300, random_state=42)
        clf_c.fit(X_centered[trn_idx], train_labels[trn_idx])

        prob = 0.7 * clf_p.predict_proba(X_plain[val_idx]) + 0.3 * clf_c.predict_proba(X_centered[val_idx])
        preds = np.argmax(prob, axis=1)
        acc = accuracy_score(train_labels[val_idx], preds)
        loss = log_loss(train_labels[val_idx], prob)
        cv_accs.append(acc)
        cv_losses.append(loss)
        print(f"  Fold {fold+1}: Acurácia = {acc*100:.2f}% | Log-Loss = {loss:.4f}")

    mean_acc = float(np.mean(cv_accs))
    print(f"\n⭐ ACURÁCIA MÉDIA CV TRI-COMPASS + 70B MoS: {mean_acc*100:.2f}%")

    # Treino Final no Dataset Completo
    print("\n[Treino Final] Ajustando pesos no dataset 100% oficial...")
    final_p = LogisticRegression(C=0.1, max_iter=400, random_state=42)
    final_p.fit(X_plain, train_labels)

    final_c = LogisticRegression(C=0.1, max_iter=400, random_state=42)
    final_c.fit(X_centered, train_labels)

    head_tensors = {
        "plain.weight": torch.tensor(final_p.coef_, dtype=torch.float32).contiguous(),
        "plain.bias": torch.tensor(final_p.intercept_, dtype=torch.float32).contiguous(),
        "centered.weight": torch.tensor(final_c.coef_, dtype=torch.float32).contiguous(),
        "centered.bias": torch.tensor(final_c.intercept_, dtype=torch.float32).contiguous(),
        "scaler_plain.mean": torch.tensor(scaler_plain.mean_, dtype=torch.float32).contiguous(),
        "scaler_plain.scale": torch.tensor(scaler_plain.scale_, dtype=torch.float32).contiguous(),
        "scaler_centered.mean": torch.tensor(scaler_centered.mean_, dtype=torch.float32).contiguous(),
        "scaler_centered.scale": torch.tensor(scaler_centered.scale_, dtype=torch.float32).contiguous(),
        "style_scaler.mean": torch.tensor(scaler_style.mean_, dtype=torch.float32).contiguous(),
        "style_scaler.scale": torch.tensor(scaler_style.scale_, dtype=torch.float32).contiguous(),
    }

    head_dir = Path("/content/head_dataset")
    head_dir.mkdir(parents=True, exist_ok=True)
    out_path = head_dir / "head_tricompass_70b.safetensors"
    save_file(head_tensors, str(out_path))

    meta = {
        "architecture": "TriCompass_70B_MoS",
        "labels": OFFICIAL_LABELS,
        "features_dim": X_plain.shape[1],
        "cv_accuracy": mean_acc,
        "scar_scale": 0.10,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(head_dir / "meta_tricompass_70b.json", "w") as f:
        json.dump(meta, f, indent=2)

    sz_mb = out_path.stat().st_size / (1024**2)
    print(f"\n💾 Arquivo Oficial Compilado: {out_path} ({sz_mb:.2f} MB)")

    # Dataset Metadata para Kaggle
    kg_meta = {
        "title": "Name That LLM - 70B MoS TriCompass Head",
        "id": f"{os.environ.get('KAGGLE_USERNAME', 'thetod')}/name-that-llm-70b-mos-tricompass-head",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True
    }
    with open(head_dir / "dataset-metadata.json", "w") as f:
        json.dump(kg_meta, f, indent=2)

    print("\n🚀 Publicando dataset privado diretamente no Kaggle via KaggleApi...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        try:
            api.dataset_create_new(str(head_dir), dir_mode="zip", quiet=False)
            print("✓ Dataset privado criado com sucesso no Kaggle!")
        except Exception as e:
            print(f"Dataset create info: {e}. Tentando nova versão...")
            api.dataset_create_version(str(head_dir), version_notes="Update 70B MoS TriCompass Head", dir_mode="zip", quiet=False)
            print("✓ Nova versão do dataset criada com sucesso no Kaggle!")
    except Exception as e:
        print(f"Aviso publicação Kaggle: {e}")

    print("\n🎉 TREINAMENTO E EXPORTAÇÃO FINALIZADOS COM SUCESSO NO COLAB GPU!")

if __name__ == "__main__":
    main()
