#!/usr/bin/env python3
"""
👑 TITAN 2: Yi-34B MoS Golden-Slice (Layers 1 -> 30)
Architecture: 01-ai Yi-34B (60 Layers, d = 7,168, INT4 BitsAndBytes).
Golden Slice: Layers 1 -> 30 (Half-Depth Representation, avoiding vocabulary convergence collapse).
Tri-Compass Vector: 7,168 + 28 = 7,196D.
Validation: 5-Fold Stratified Group K-Fold by prompt_id + Continuous Soft Sinkhorn (300 iterations).
Refit: 100% full training fit, saving head_tricompass_yi34b_l30.safetensors.
"""

from __future__ import annotations

import os
import sys
import time
import gc
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, log_loss

# Canonical official competition labels (matching Kaggle submission ordering exactly)
LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for i, l in enumerate(LABELS)}
K = len(LABELS)


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
    c = np.ones(K)
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= target / R.sum(0)
    R = P * c
    return R / R.sum(1, keepdims=True)


def main():
    print("=" * 80)
    print("👑 TITAN 2: YI-34B MoS GOLDEN-SLICE EXTRACTION & TRAINING (L1-30)")
    print("   Total Layers: 60 | Truncation: Layer 30 | Hidden Dim: 7,168")
    print("   Tri-Compass Dim: 7,168 + 28 = 7,196D | Classes: 18 Canonical")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    data_path = "train.csv"
    if not os.path.exists(data_path):
        data_path = "/content/train.csv"
    if not os.path.exists(data_path):
        data_path = "/kaggle/input/name-that-llm/train.csv"

    df_train = pd.read_csv(data_path)
    n_samples = len(df_train)
    print(f"✓ Loaded {n_samples} training passages ({data_path}) across {df_train['prompt_id'].nunique()} prompts.")

    texts = df_train["text"].tolist()
    labels = df_train["model"].map(L2I).values
    prompts = df_train["prompt_id"].values

    # Check for unmapped labels
    if np.isnan(labels).any():
        raise ValueError(f"Unknown labels found in training set: {df_train['model'][np.isnan(labels)].unique()}")

    # 1. Stylometrics (Compass 1)
    print("\n[Compass 1/3] Extraindo 28D Tri-Compass stylometric signatures...")
    style_raw = extract_tricompass_style_features(texts)
    style_scaler = StandardScaler()
    style_normed = style_scaler.fit_transform(style_raw)
    print(f"✓ Stylometrics extraídos: {style_normed.shape}")

    # 2. Yi-34B Golden-Slice Feature Extraction (Layers 1-30)
    output_dir = Path("artifacts_yi34b_l30")
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_file = output_dir / "features_yi34b_l30_train_full.npy"

    if feat_file.exists():
        print(f"\n[Compass 3/3] Carregando features pré-extraídas de {feat_file}...")
        Z_yi = np.load(feat_file)
        print(f"✓ Features carregadas: {Z_yi.shape}")
    else:
        model_id = "unsloth/yi-34b-bnb-4bit"
        print(f"\n[Compass 3/3] Carregando {model_id} (Truncado na Camada 30)...")
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        cfg = AutoConfig.from_pretrained(model_id)
        cfg.num_hidden_layers = 30  # Golden-Slice: Half of 60 layers
        cfg.use_cache = False

        print("Instanciando backbone Yi-34B na GPU A100...")
        t_load0 = time.time()
        model = AutoModel.from_pretrained(
            model_id,
            config=cfg,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        ).eval()
        print(f"✓ Yi-34B Layer 30 carregado em {time.time()-t_load0:.1f}s! VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        batch_size = 8
        all_reps = []
        t_fwd0 = time.time()

        print(f"Iniciando forward pass em batches de {batch_size} para {n_samples} amostras...")
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                b_texts = texts[i : i + batch_size]
                inp = tok(b_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                out = model(**inp)
                mask = inp.attention_mask.unsqueeze(-1).float()
                # RMSNorm unit-scaled mean pooling
                pooled = (out.last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                rms = torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                pooled_norm = (pooled * rms).cpu().numpy()
                all_reps.append(pooled_norm)

                if (i // batch_size) % 30 == 0 or (i + batch_size >= n_samples):
                    elapsed = time.time() - t_fwd0
                    speed = (i + len(b_texts)) / max(elapsed, 1e-3)
                    print(f"  Batch {i//batch_size + 1}/{(n_samples+batch_size-1)//batch_size} ({i+len(b_texts)}/{n_samples}) - {speed:.1f} amostras/s")

        Z_yi = np.concatenate(all_reps, axis=0)
        del model
        torch.cuda.empty_cache()
        gc.collect()

        print(f"✓ Yi-34B Layer 30 features extraídas em {time.time()-t_fwd0:.1f}s: {Z_yi.shape}")
        np.save(feat_file, Z_yi)

    # 3. Construct Tri-Compass Matrices (7,168 + 28 = 7,196D)
    X_plain_raw = np.hstack([Z_yi, style_normed])
    Zc = center_groups_delta_origin(Z_yi, prompts)
    Sc = center_groups_delta_origin(style_normed, prompts)
    X_centered_raw = np.hstack([Zc, Sc])

    print(f"\n[Tri-Compass] Matriz Plain: {X_plain_raw.shape} | Matriz Centered: {X_centered_raw.shape}")

    # 4. 5-Fold Stratified Group K-Fold Validation
    print("\n[Validação 5-Fold Stratified Group K-Fold por prompt_id]...")
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_ensemble_probs = np.zeros((n_samples, K), dtype=np.float32)
    fold_accs = []
    fold_sinkhorn_accs = []

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_plain_raw, labels, groups=prompts)):
        scaler_p = StandardScaler()
        X_tr_p = scaler_p.fit_transform(X_plain_raw[trn_idx])
        X_va_p = scaler_p.transform(X_plain_raw[val_idx])

        clf_p = LogisticRegression(C=0.1, max_iter=600, random_state=42)
        clf_p.fit(X_tr_p, labels[trn_idx])
        p_val_plain = clf_p.predict_proba(X_va_p)

        scaler_c = StandardScaler()
        X_tr_c = scaler_c.fit_transform(X_centered_raw[trn_idx])
        X_va_c = scaler_c.transform(X_centered_raw[val_idx])

        clf_c = LogisticRegression(C=0.1, max_iter=600, random_state=42)
        clf_c.fit(X_tr_c, labels[trn_idx])
        p_val_twin = clf_c.predict_proba(X_va_c)

        # Ensemble: 80% Plain + 20% Prompt-Centered Twin
        p_val_ens = 0.80 * p_val_plain + 0.20 * p_val_twin
        p_val_ens = p_val_ens / p_val_ens.sum(axis=1, keepdims=True)
        oof_ensemble_probs[val_idx] = p_val_ens

        acc_raw = accuracy_score(labels[val_idx], np.argmax(p_val_ens, axis=1))
        p_val_sink = sinkhorn(p_val_ens)
        acc_sink = accuracy_score(labels[val_idx], np.argmax(p_val_sink, axis=1))

        fold_accs.append(acc_raw)
        fold_sinkhorn_accs.append(acc_sink)
        print(f"  Fold {fold+1}/5: Raw Acc = {acc_raw*100:.2f}% | Sinkhorn Acc = {acc_sink*100:.2f}%")

    overall_raw = accuracy_score(labels, np.argmax(oof_ensemble_probs, axis=1))
    oof_sinkhorn = sinkhorn(oof_ensemble_probs)
    overall_sink = accuracy_score(labels, np.argmax(oof_sinkhorn, axis=1))
    overall_logloss = log_loss(labels, oof_sinkhorn)

    print("\n" + "=" * 80)
    print(f"🏆 OVERALL TITAN 2 (Yi-34B L30) OOF RESULTS:")
    print(f"   • Raw Ensemble OOF Accuracy:      {overall_raw*100:.2f}%")
    print(f"   • Calibrated Soft Sinkhorn Acc:   {overall_sink*100:.2f}%")
    print(f"   • Calibrated Log Loss:            {overall_logloss:.4f}")
    print("=" * 80)

    # 5. Full 100% Fit & Save Head Artifacts
    print("\n[Refit 100%] Compilando pesos finais Dual-Head...")
    final_scaler_p = StandardScaler()
    X_full_p = final_scaler_p.fit_transform(X_plain_raw)
    final_clf_p = LogisticRegression(C=0.1, max_iter=600, random_state=42)
    final_clf_p.fit(X_full_p, labels)

    final_scaler_c = StandardScaler()
    X_full_c = final_scaler_c.fit_transform(X_centered_raw)
    final_clf_c = LogisticRegression(C=0.1, max_iter=600, random_state=42)
    final_clf_c.fit(X_full_c, labels)

    head_weights = {
        "style_scaler.mean": torch.from_numpy(style_scaler.mean_.astype(np.float32)),
        "style_scaler.scale": torch.from_numpy(style_scaler.scale_.astype(np.float32)),
        "scaler_plain.mean": torch.from_numpy(final_scaler_p.mean_.astype(np.float32)),
        "scaler_plain.scale": torch.from_numpy(final_scaler_p.scale_.astype(np.float32)),
        "plain.weight": torch.from_numpy(final_clf_p.coef_.astype(np.float32)),
        "plain.bias": torch.from_numpy(final_clf_p.intercept_.astype(np.float32)),
        "scaler_centered.mean": torch.from_numpy(final_scaler_c.mean_.astype(np.float32)),
        "scaler_centered.scale": torch.from_numpy(final_scaler_c.scale_.astype(np.float32)),
        "centered.weight": torch.from_numpy(final_clf_c.coef_.astype(np.float32)),
        "centered.bias": torch.from_numpy(final_clf_c.intercept_.astype(np.float32)),
    }
    save_file(head_weights, str(output_dir / "head_tricompass_yi34b_l30.safetensors"))

    meta = {
        "architecture": "Titan2_Yi34B_Layer30",
        "golden_slice": "Layers 1 -> 30 (Half Depth)",
        "features_dim": X_plain_raw.shape[1],
        "hidden_dim": 7168,
        "style_dim": 28,
        "labels": LABELS,
        "overall_oof_raw_acc": float(overall_raw),
        "overall_oof_sinkhorn_acc": float(overall_sink),
        "overall_oof_logloss": float(overall_logloss),
        "fold_sinkhorn_accs": [float(x) for x in fold_sinkhorn_accs],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    with open(output_dir / "meta_tricompass_yi34b.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✓ Artefatos salvos em {output_dir}:")
    print(f"  - head_tricompass_yi34b_l30.safetensors ({os.path.getsize(output_dir / 'head_tricompass_yi34b_l30.safetensors')/1e6:.2f} MB)")
    print(f"  - meta_tricompass_yi34b.json")
    print(f"  - features_yi34b_l30_train_full.npy ({os.path.getsize(feat_file)/1e6:.2f} MB)")
    print("\n✨ TITAN 2 CONCLUÍDO COM SUCESSO!")


if __name__ == "__main__":
    main()
