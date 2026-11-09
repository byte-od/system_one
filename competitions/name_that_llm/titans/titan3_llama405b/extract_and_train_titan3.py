#!/usr/bin/env python3
"""
TITAN 3: Llama 3.1 405B MoS Golden-Slice (Layers 35 -> 75)
Architecture: Meta Llama 3.1 405B (126 Layers Total).
Golden Slice: Layers 35 -> 75 (Core Stylistic Slices), hidden dimension d = 16,384.
Tri-Compass Dimension: 16,384 + 28 = 16,412D.
"""

from __future__ import annotations

import os
import sys
import time
import gc
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, log_loss

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
    print("👑 TITAN 3: LLAMA 3.1 405B MoS GOLDEN-SLICE EXTRACTION & TRAINING (L35-75)")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Paths resolution
    data_path = "/kaggle/input/name-that-llm/train.csv"
    if not os.path.exists(data_path):
        data_path = str(Path(__file__).resolve().parents[2] / "data" / "train.csv")

    df_train = pd.read_csv(data_path)
    n_samples = len(df_train)
    print(f"✓ Loaded {n_samples} training passages ({data_path}) across {df_train['prompt_id'].nunique()} prompts.")

    texts = df_train["text"].tolist()
    labels = df_train["model"].map(L2I).values
    prompts = df_train["prompt_id"].values

    # 1. Stylometrics
    print("\n[Compass 1/3] Extraindo 28D Tri-Compass stylometric signatures...")
    style_raw = extract_tricompass_style_features(texts)

    # 2. Llama 405B MoS Slices Extraction
    output_dir = Path("/kaggle/working") if os.path.exists("/kaggle") else Path(__file__).resolve().parent / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_file = output_dir / "features_405b_slices_train_full.npy"

    if feat_file.exists():
        Z_405b = np.load(feat_file)
        print(f"✓ Features carregadas de cache: {Z_405b.shape}")
    else:
        print("ℹ️ MoS Slice Streaming: extraindo fatias 35..75 do Llama 3.1 405B...")
        # Placeholder for Kaggle runner execution
        return

    # 3. Construct Tri-Compass Matrices
    X_plain_raw = np.hstack([Z_405b, style_raw])
    Zc = center_groups_delta_origin(Z_405b, prompts)
    Sc = center_groups_delta_origin(style_raw, prompts)
    X_centered_raw = np.hstack([Zc, Sc])

    # 4. 5-Fold Stratified Group K-Fold
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_ensemble_probs = np.zeros((n_samples, 18), dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_plain_raw, labels, groups=prompts)):
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
        oof_ensemble_probs[val_idx] = p_val_ens / p_val_ens.sum(axis=1, keepdims=True)

    overall_raw = accuracy_score(labels, np.argmax(oof_ensemble_probs, axis=1))
    oof_sinkhorn = sinkhorn(oof_ensemble_probs)
    overall_sink = accuracy_score(labels, np.argmax(oof_sinkhorn, axis=1))
    overall_logloss = log_loss(labels, oof_sinkhorn)

    print("-" * 80)
    print(f"🏆 OVERALL TITAN 3 (405B MoS Slices) OOF RESULTS:")
    print(f"   • Raw Ensemble OOF Accuracy:      {overall_raw*100:.2f}%")
    print(f"   • Calibrated Sinkhorn OOF Acc:    {overall_sink*100:.2f}%")
    print(f"   • Calibrated Log Loss:            {overall_logloss:.4f}")
    print("-" * 80)

    # 5. Full Refit & Save
    final_scaler = StandardScaler()
    X_full_sc = final_scaler.fit_transform(X_plain_raw)
    final_clf = LogisticRegression(C=0.1, max_iter=500, random_state=42)
    final_clf.fit(X_full_sc, labels)

    head_weights = {
        "scaler.mean": torch.from_numpy(final_scaler.mean_.astype(np.float32)),
        "scaler.scale": torch.from_numpy(final_scaler.scale_.astype(np.float32)),
        "head.weight": torch.from_numpy(final_clf.coef_.astype(np.float32)),
        "head.bias": torch.from_numpy(final_clf.intercept_.astype(np.float32)),
    }
    save_file(head_weights, str(output_dir / "head_tricompass_405b_slices.safetensors"))

    meta = {
        "architecture": "Titan3_Llama3.1_405B_MoS_Slices",
        "golden_slice": "Layers 35 -> 75 (Core Slices)",
        "features_dim": X_plain_raw.shape[1],
        "hidden_dim": 16384,
        "style_dim": 28,
        "labels": OFFICIAL_LABELS,
        "overall_oof_raw_acc": float(overall_raw),
        "overall_oof_sinkhorn_acc": float(overall_sink),
        "overall_oof_logloss": float(overall_logloss),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("✨ TITAN 3 CONCLUÍDO COM SUCESSO!")


if __name__ == "__main__":
    main()
