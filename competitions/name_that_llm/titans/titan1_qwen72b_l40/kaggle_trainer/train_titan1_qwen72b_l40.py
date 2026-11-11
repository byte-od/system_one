#!/usr/bin/env python3
"""
Kaggle Cloud Job: Titan 1 - Qwen 2.5 72B MoS Golden-Slice (Layers 1 -> 40)
Runs on Kaggle Cloud GPU with High-Speed CUDA:
1. Mounts MoS 72B checkpoint from kernel 'thetod/qwen-72b-to-mixture-of-scars-mos-converter'.
2. Mounts competition dataset 'name-that-llm'.
3. Streams Groups 0..4 (Layers 0..39 = 40 layers) in batch on CUDA GPU.
4. Extracts 8,192D authorial latent vectors + 28D Tri-Compass stylometrics (8,220D total).
5. Executes 5-Fold Stratified Group K-Fold by prompt_id (zero leakage).
6. Outputs calibrated head artifacts:
   - features_70b_l40_train_full.npy
   - head_tricompass_70b_l40.safetensors
   - meta.json
"""

import os
import sys
import time
import gc
import glob
import json
import re
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

# Competition 18 classes in strict alphabetical order
OFFICIAL_LABELS = [
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
]
L2I = {l: i for i, l in enumerate(OFFICIAL_LABELS)}
I2L = {i: l for i, l in enumerate(OFFICIAL_LABELS)}


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
        if self.is_loaded:
            return
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


def extract_tricompass_style_features(texts: List[str]) -> np.ndarray:
    feats = []
    for t in texts:
        t = str(t)
        n = max(len(t), 1)
        words = t.split()
        nw = max(len(words), 1)
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        ns = max(len(sents), 1)
        lines = [l for l in t.split("\n") if l.strip()]
        nl = max(len(lines), 1)

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


def find_file(pattern: str) -> Optional[str]:
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def main():
    print("=" * 80)
    print("👑 TITAN 1: QWEN 2.5 72B MoS GOLDEN-SLICE EXTRACTION & TRAINING (L1-40)")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    scar_scale = 0.10
    print(f"Device: {device} | Precision: {dtype} | Scar Scale: {scar_scale}")

    # 1. Resolve Paths
    train_file = find_file("/kaggle/input/**/train.csv") or "train.csv"
    assert os.path.exists(train_file), f"train.csv not found! Looked for {train_file}"
    df_train = pd.read_csv(train_file)
    n_samples = len(df_train)
    print(f"✓ Loaded {n_samples} training passages ({train_file}) across {df_train['prompt_id'].nunique()} prompts.")

    mos_dir_cand = find_file("/kaggle/input/**/mos_group_0.safetensors")
    if mos_dir_cand:
        mos_dir = Path(os.path.dirname(mos_dir_cand))
    else:
        mos_dir = Path("/kaggle/input/qwen-72b-to-mixture-of-scars-mos-converter")
    print(f"✓ MoS 72B Model Directory: {mos_dir}")

    texts = df_train["text"].tolist()
    labels = df_train["model"].map(L2I).values
    prompts = df_train["prompt_id"].values

    # 2. Extract Stylometrics
    print("\n[Compass 1/3] Extracting 28D Tri-Compass stylometric signatures...")
    t_style0 = time.time()
    style_raw = extract_tricompass_style_features(texts)
    print(f"✓ Stylometrics computed in {time.time()-t_style0:.2f}s: shape {style_raw.shape}")

    # 3. MoS 40 Layers Extraction on CUDA
    print("\n[Compass 3/3] Extracting 8,192D representations from Layers 0..39 (Groups 0..4)...")
    tokenizer = AutoTokenizer.from_pretrained(str(mos_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cache_dir = Path("/kaggle/temp_cache_titan1")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    emb_file = mos_dir / "mos_embeddings_and_head.safetensors"
    with safe_open(str(emb_file), framework="pt", device="cpu") as f:
        emb_p = f.get_tensor("model.embed_tokens.weight")
        emb_s = f.get_tensor("model.embed_tokens.weight.scales")

    batch_size = 32
    max_length = 256
    num_batches = (n_samples + batch_size - 1) // batch_size
    bpt = 8192 // 128
    boff = torch.arange(bpt, dtype=torch.long)

    t0_emb = time.time()
    print(f"  Tokenizing and projecting embeddings for {num_batches} batches...")
    for b in range(num_batches):
        s_idx = b * batch_size
        e_idx = min(s_idx + batch_size, n_samples)
        b_texts = texts[s_idx:e_idx]
        enc = tokenizer(b_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = enc.input_ids
        attn_mask = enc.attention_mask.unsqueeze(-1)
        bsz, seq_len = input_ids.shape

        all_bidx = (input_ids.unsqueeze(-1) * bpt + boff).view(-1)
        x = dequantize_int4(emb_p[all_bidx], emb_s[all_bidx], (bsz, seq_len, 8192), dtype=dtype, target_device=torch.device("cpu"))
        torch.save({"x": x, "mask": attn_mask, "bsz": bsz, "seq_len": seq_len}, cache_dir / f"b_{b}.pt")

    del emb_p, emb_s
    gc.collect()
    print(f"  ✓ Initial embeddings cached in {time.time()-t0_emb:.2f}s!")

    # Stream Groups 0..4 (Layers 0..39)
    t_groups_start = time.time()
    for g_idx in range(5):
        tg0 = time.time()
        g_path = mos_dir / f"mos_group_{g_idx}.safetensors"
        group = Qwen72BMoSGroup(g_idx, g_path, device, dtype, scar_scale=scar_scale)
        group.load()

        with torch.no_grad():
            for b in range(num_batches):
                data = torch.load(cache_dir / f"b_{b}.pt")
                x = data["x"].to(device)
                bsz, seq_len = data["bsz"], data["seq_len"]

                pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)
                cos, sin = precompute_freqs_cis(pos, device, dtype)

                for l in range(group.start_layer, group.end_layer):
                    x = group.forward_layer(l, x, cos, sin)

                torch.save({"x": x.cpu(), "mask": data["mask"], "bsz": bsz, "seq_len": seq_len}, cache_dir / f"b_{b}_next.pt")
                del x, data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        group.unload()
        for b in range(num_batches):
            os.replace(cache_dir / f"b_{b}_next.pt", cache_dir / f"b_{b}.pt")

        print(f"  ✓ Group {g_idx} (Layers {g_idx*8}..{g_idx*8+7}) completed in {time.time()-tg0:.2f}s", flush=True)

    print(f"  ✓ All 40 layers processed in {time.time()-t_groups_start:.2f}s!", flush=True)

    # Pooling at Layer 40
    print("  Applying masked mean pooling at Layer 40...")
    features_70b_l40 = np.zeros((n_samples, 8192), dtype=np.float32)
    for b in range(num_batches):
        s_idx = b * batch_size
        e_idx = min(s_idx + batch_size, n_samples)
        data = torch.load(cache_dir / f"b_{b}.pt")
        x = data["x"].float()
        mask = data["mask"].float()
        den = mask.sum(dim=1).clamp(min=1)
        pooled = (x * mask).sum(dim=1) / den
        rms = torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        features_70b_l40[s_idx:e_idx] = (pooled * rms).numpy()

    output_dir = Path("/kaggle/working")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features_70b_l40_train_full.npy", features_70b_l40)
    shutil.rmtree(cache_dir, ignore_errors=True)
    print(f"✓ Saved 40-layer features ({features_70b_l40.shape})")

    # 4. Construct Matrices
    X_plain_raw = np.hstack([features_70b_l40, style_raw])
    Zc = center_groups_delta_origin(features_70b_l40, prompts)
    Sc = center_groups_delta_origin(style_raw, prompts)
    X_centered_raw = np.hstack([Zc, Sc])

    # 5. 5-Fold Stratified Group K-Fold
    print("\n" + "=" * 80)
    print("📊 5-FOLD STRATIFIED GROUP K-FOLD CROSS-VALIDATION (BY PROMPT_ID)")
    print("=" * 80)

    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_ensemble_probs = np.zeros((n_samples, 18), dtype=np.float32)
    fold_accs = []
    fold_sinkhorn_accs = []

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_plain_raw, labels, groups=prompts)):
        t_fold = time.time()
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_plain_raw[trn_idx])
        X_va_sc = scaler.transform(X_plain_raw[val_idx])

        clf_p = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        clf_p.fit(X_tr_sc, labels[trn_idx])
        p_val_plain = clf_p.predict_proba(X_va_sc)

        scaler_c = StandardScaler()
        X_tr_c_sc = scaler_c.fit_transform(X_centered_raw[trn_idx])
        X_va_c_sc = scaler_c.transform(X_centered_raw[val_idx])

        clf_c = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        clf_c.fit(X_tr_c_sc, labels[trn_idx])
        p_val_twin = clf_c.predict_proba(X_va_c_sc)

        p_val_ens = 0.8 * p_val_plain + 0.2 * p_val_twin
        p_val_ens = p_val_ens / p_val_ens.sum(axis=1, keepdims=True)
        oof_ensemble_probs[val_idx] = p_val_ens

        acc_raw = accuracy_score(labels[val_idx], np.argmax(p_val_ens, axis=1))
        p_val_sink = sinkhorn(p_val_ens)
        acc_sink = accuracy_score(labels[val_idx], np.argmax(p_val_sink, axis=1))

        fold_accs.append(acc_raw)
        fold_sinkhorn_accs.append(acc_sink)
        print(f"Fold {fold+1}/5: Raw Acc = {acc_raw*100:.2f}% | Sinkhorn Acc = {acc_sink*100:.2f}% ({time.time()-t_fold:.2f}s)", flush=True)

    overall_raw = accuracy_score(labels, np.argmax(oof_ensemble_probs, axis=1))
    oof_sinkhorn = sinkhorn(oof_ensemble_probs)
    overall_sink = accuracy_score(labels, np.argmax(oof_sinkhorn, axis=1))
    overall_logloss = log_loss(labels, oof_sinkhorn)

    print("\n" + "-" * 80)
    print(f"🏆 OVERALL TITAN 1 OOF RESULTS:")
    print(f"   • Raw Ensemble OOF Accuracy:      {overall_raw*100:.2f}%")
    print(f"   • Calibrated Sinkhorn OOF Acc:    {overall_sink*100:.2f}%")
    print(f"   • Calibrated Log Loss:            {overall_logloss:.4f}")
    print("-" * 80)

    # 6. Refit on 100% Data and Save
    print("\n[Exportação] Refazendo ajuste final em 100% dos dados para inferência de campeonato...")
    final_scaler = StandardScaler()
    X_full_sc = final_scaler.fit_transform(X_plain_raw)

    final_clf = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    final_clf.fit(X_full_sc, labels)

    train_acc = accuracy_score(labels, final_clf.predict(X_full_sc))
    print(f"✓ Final Model Train Accuracy: {train_acc*100:.2f}%")

    head_weights = {
        "scaler.mean": torch.from_numpy(final_scaler.mean_.astype(np.float32)),
        "scaler.scale": torch.from_numpy(final_scaler.scale_.astype(np.float32)),
        "head.weight": torch.from_numpy(final_clf.coef_.astype(np.float32)),
        "head.bias": torch.from_numpy(final_clf.intercept_.astype(np.float32)),
    }
    save_head_path = output_dir / "head_tricompass_70b_l40.safetensors"
    save_file(head_weights, str(save_head_path))
    print(f"✓ Safetensors head saved: {save_head_path} ({save_head_path.stat().st_size / 1024:.1f} KB)")

    meta = {
        "architecture": "Titan1_Qwen72B_MoS_Layer40",
        "golden_slice": "Layers 1 -> 40 (Half Depth)",
        "features_dim": X_plain_raw.shape[1],
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
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Metadata saved: {output_dir / 'meta.json'}")

    print("\n" + "=" * 80)
    print("✨ TITAN 1 CONCLUÍDO COM SUCESSO ABSOLUTO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
