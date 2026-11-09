#!/usr/bin/env python3
"""
TITAN 1: Qwen 2.5 72B MoS Golden-Slice (Layers 1 -> 40)
Extracts 8,192D latent representations from Layers 0..39 of Qwen 2.5 72B MoS.
Combines with 28D Tri-Compass stylometric front-end (8,220D total).
Trains and evaluates 5-Fold Stratified Group K-Fold (grouped by prompt_id).
Saves production-ready head artifacts for Kaggle inference.
"""

from __future__ import annotations

import os
import sys
import time
import gc
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

# Add scar-cache root to path
ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scar_cache.qwen72b_mos_pipeline import Qwen72BMoSGroup, dequantize_int4

# Official 18 competition labels in strict alphabetical order
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

MODEL_DIR = ROOT_DIR / "models" / "qwen25_72b_mos"
DATA_PATH = ROOT_DIR / "competitions" / "name_that_llm" / "data" / "train.csv"
OUTPUT_DIR = ROOT_DIR / "competitions" / "name_that_llm" / "head_artifacts_70b_l40"
CACHE_DIR = ROOT_DIR / "competitions" / "name_that_llm" / "scratch" / "titan1_cache"


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


def extract_tricompass_style_features(texts: List[str]) -> np.ndarray:
    """Extracts 28 stylometric, typographical, and structural features."""
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


def main():
    print("=" * 80)
    print("👑 TITAN 1: QWEN 2.5 72B MoS GOLDEN-SLICE EXTRACTION & TRAINING (L1-40)")
    print("=" * 80)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16
    scar_scale = 0.10
    print(f"Device: {device} | Precision: {dtype} | Scar Scale: {scar_scale}")

    # 1. Load Train Data
    df_train = pd.read_csv(DATA_PATH)
    n_samples = len(df_train)
    print(f"✓ Loaded {n_samples} training passages across {df_train['prompt_id'].nunique()} prompts.")

    texts = df_train["text"].tolist()
    labels = df_train["model"].map(L2I).values
    prompts = df_train["prompt_id"].values

    # 2. Extract Tri-Compass Stylometric Features (28D)
    print("\n[Compass 1/3] Extracting 28D Tri-Compass stylometric signatures...")
    t_style0 = time.time()
    style_raw = extract_tricompass_style_features(texts)
    print(f"✓ Stylometrics computed in {time.time()-t_style0:.2f}s: shape {style_raw.shape}")

    # 3. Check for existing cached features or extract
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    feat_save_path = OUTPUT_DIR / "features_70b_l40_train_full.npy"

    if feat_save_path.exists():
        print(f"\n[Compass 3/3] Found pre-extracted features at {feat_save_path}!")
        features_70b_l40 = np.load(feat_save_path)
        print(f"✓ Loaded features shape: {features_70b_l40.shape}")
    else:
        print("\n[Compass 3/3] Extracting 8,192D representations from Layers 0..39 (Groups 0..4)...")
        tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Batch tokenization and embedding initialization
        emb_file = MODEL_DIR / "mos_embeddings_and_head.safetensors"
        with safe_open(emb_file, framework="pt", device="cpu") as f:
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
            enc = tok(b_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            input_ids = enc.input_ids
            attn_mask = enc.attention_mask.unsqueeze(-1)
            bsz, seq_len = input_ids.shape

            all_bidx = (input_ids.unsqueeze(-1) * bpt + boff).view(-1)
            x = dequantize_int4(emb_p[all_bidx], emb_s[all_bidx], (bsz, seq_len, 8192), dtype=dtype, target_device=torch.device("cpu"))
            torch.save({"x": x, "mask": attn_mask, "bsz": bsz, "seq_len": seq_len}, CACHE_DIR / f"b_{b}.pt")

        del emb_p, emb_s
        gc.collect()
        print(f"  ✓ Initial embeddings cached in {time.time()-t0_emb:.2f}s!")

        # Execute Groups 0..4 (Layers 0..39)
        t_groups_start = time.time()
        for g_idx in range(5):
            tg0 = time.time()
            g_path = MODEL_DIR / f"mos_group_{g_idx}.safetensors"
            group = Qwen72BMoSGroup(g_idx, g_path, device, dtype, scar_scale=scar_scale)
            group.load()

            with torch.no_grad():
                for b in range(num_batches):
                    data = torch.load(CACHE_DIR / f"b_{b}.pt", map_location=device)
                    x = data["x"].to(device)
                    bsz, seq_len = data["bsz"], data["seq_len"]

                    pos = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)
                    cos, sin = precompute_freqs_cis(pos, device, dtype)

                    for l in range(group.start_layer, group.end_layer):
                        x = group.forward_layer(l, x, cos, sin)

                    torch.save({"x": x.cpu(), "mask": data["mask"], "bsz": bsz, "seq_len": seq_len}, CACHE_DIR / f"b_{b}_next.pt")

            group.unload()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            for b in range(num_batches):
                os.replace(CACHE_DIR / f"b_{b}_next.pt", CACHE_DIR / f"b_{b}.pt")

            print(f"  ✓ Group {g_idx} (Layers {g_idx*8}..{g_idx*8+7}) completed in {time.time()-tg0:.2f}s")

        print(f"  ✓ All 40 layers processed in {time.time()-t_groups_start:.2f}s!")

        # Final Masked Mean Pooling
        print("  Applying masked mean pooling at Layer 40...")
        features_70b_l40 = np.zeros((n_samples, 8192), dtype=np.float32)
        for b in range(num_batches):
            s_idx = b * batch_size
            e_idx = min(s_idx + batch_size, n_samples)
            data = torch.load(CACHE_DIR / f"b_{b}.pt")
            x = data["x"].float()
            mask = data["mask"].float()
            den = mask.sum(dim=1).clamp(min=1)
            pooled = (x * mask).sum(dim=1) / den
            rms = torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            features_70b_l40[s_idx:e_idx] = (pooled * rms).numpy()

        np.save(feat_save_path, features_70b_l40)
        print(f"✓ Saved 40-layer features: {feat_save_path} ({features_70b_l40.shape})")

        # Cleanup cache
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    # 4. Construct Tri-Compass Feature Matrices
    print("\n[Treinamento da Cabeça] Montando matrizes Tri-Compass (8,220D)...")
    X_plain_raw = np.hstack([features_70b_l40, style_raw])
    print(f"✓ X_plain_raw: {X_plain_raw.shape}")

    # 5. Delta-Origin Prompt Centering
    print("[Compass 2/3] Calculando Prompt-Centering Delta Origin...")
    Zc = center_groups_delta_origin(features_70b_l40, prompts)
    Sc = center_groups_delta_origin(style_raw, prompts)
    X_centered_raw = np.hstack([Zc, Sc])
    print(f"✓ X_centered_raw: {X_centered_raw.shape}")

    # 6. 5-Fold Stratified Group K-Fold Cross-Validation (Zero Leakage)
    print("\n" + "=" * 80)
    print("📊 5-FOLD STRATIFIED GROUP K-FOLD CROSS-VALIDATION (BY PROMPT_ID)")
    print("=" * 80)

    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_plain_probs = np.zeros((n_samples, 18), dtype=np.float32)
    oof_twin_probs = np.zeros((n_samples, 18), dtype=np.float32)
    oof_ensemble_probs = np.zeros((n_samples, 18), dtype=np.float32)

    fold_accs = []
    fold_sinkhorn_accs = []

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_plain_raw, labels, groups=prompts)):
        t_fold = time.time()
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_plain_raw[trn_idx])
        X_va_sc = scaler.transform(X_plain_raw[val_idx])

        # Plain Classifier
        clf_p = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        clf_p.fit(X_tr_sc, labels[trn_idx])
        p_val_plain = clf_p.predict_proba(X_va_sc)
        oof_plain_probs[val_idx] = p_val_plain

        # Centered Classifier
        scaler_c = StandardScaler()
        X_tr_c_sc = scaler_c.fit_transform(X_centered_raw[trn_idx])
        X_va_c_sc = scaler_c.transform(X_centered_raw[val_idx])

        clf_c = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        clf_c.fit(X_tr_c_sc, labels[trn_idx])
        p_val_twin = clf_c.predict_proba(X_va_c_sc)
        oof_twin_probs[val_idx] = p_val_twin

        # Ensemble: 0.8 Plain + 0.2 Centered
        p_val_ens = 0.8 * p_val_plain + 0.2 * p_val_twin
        p_val_ens = p_val_ens / p_val_ens.sum(axis=1, keepdims=True)
        oof_ensemble_probs[val_idx] = p_val_ens

        # Fold Metrics
        acc_raw = accuracy_score(labels[val_idx], np.argmax(p_val_ens, axis=1))
        p_val_sink = sinkhorn(p_val_ens)
        acc_sink = accuracy_score(labels[val_idx], np.argmax(p_val_sink, axis=1))

        fold_accs.append(acc_raw)
        fold_sinkhorn_accs.append(acc_sink)
        print(f"Fold {fold+1}/5 ({len(val_idx)} samples): Raw Acc = {acc_raw*100:.2f}% | Sinkhorn Acc = {acc_sink*100:.2f}% ({time.time()-t_fold:.2f}s)")

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

    # 7. Final Model Refit on All 2,699 Samples
    print("\n[Exportação] Refazendo ajuste final em 100% dos dados para inferência de campeonato...")
    final_scaler = StandardScaler()
    X_full_sc = final_scaler.fit_transform(X_plain_raw)

    final_clf = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    final_clf.fit(X_full_sc, labels)

    train_acc = accuracy_score(labels, final_clf.predict(X_full_sc))
    print(f"✓ Final Model Train Accuracy: {train_acc*100:.2f}%")

    # Save safetensors
    head_weights = {
        "scaler.mean": torch.from_numpy(final_scaler.mean_.astype(np.float32)),
        "scaler.scale": torch.from_numpy(final_scaler.scale_.astype(np.float32)),
        "head.weight": torch.from_numpy(final_clf.coef_.astype(np.float32)),
        "head.bias": torch.from_numpy(final_clf.intercept_.astype(np.float32)),
    }
    save_head_path = OUTPUT_DIR / "head_tricompass_70b_l40.safetensors"
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
    with open(OUTPUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Metadata saved: {OUTPUT_DIR / 'meta.json'}")

    print("\n" + "=" * 80)
    print("✨ TITAN 1 CONCLUÍDO COM SUCESSO ABSOLUTO!")
    print("=" * 80)


if __name__ == "__main__":
    main()
