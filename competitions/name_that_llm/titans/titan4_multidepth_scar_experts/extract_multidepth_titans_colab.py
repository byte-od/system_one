#!/usr/bin/env python3
"""
👑 SYSTEM ONE — MILESTONE 3 (COLAB A100):
8-DEPTH COVERING-WINDOW MULTI-SLICE EXTRACTOR & CO-TRAINER FOR TITAN 2 (34B) & TITAN 3 (123B)
=============================================================================================
Extrai simultaneamente 8 fatias de profundidade sem truncamento (`max_len=512, stride=384`)
usando exatamente o motor `extract_mos_nf4_streaming`, eliminando a perda dos parágrafos finais
e capturando ortografia rasa + sintaxe intermediária + retórica Golden-Slice num único passe:
- Titan 2v2 (`Yi-34B` L30): `capture_layers = [4, 8, 12, 16, 20, 24, 28, 30]`
- Titan 3v2 (`Mistral Large 2 123B` L44): `capture_layers = [6, 12, 18, 24, 30, 36, 40, 44]`
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

THIS_DIR = Path(__file__).resolve().parent
COMP_DIR = THIS_DIR.parent.parent
sys.path.insert(0, str(COMP_DIR))

from forensic_scouts_120d import extract_forensic_compass1_120d
from mos_streaming_engine import (
    aggregate_window_embeddings,
    build_covering_windows,
    extract_mos_nf4_streaming,
)

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
L2I = {l: i for i, l in enumerate(LABELS)}


def main():
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description="8-Depth Covering-Window Multi-Slice Extractor (Colab A100)")
    parser.add_argument("--backbone-dir", required=True, help="Diretório do backbone NF4 fatiado (Yi-34B L30 ou Mistral 123B L44)")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layers", default="6,12,18,24,30,36,40,44", help="Camadas 1-based separadas por vírgula")
    parser.add_argument("--pca-per-slice", type=int, default=384)
    parser.add_argument("--tag", default="Titan-8Depth")
    args = parser.parse_args()

    capture_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.train_csv)
    texts = df["text"].fillna("").astype(str).tolist()
    y = df["model"].map(L2I).values
    prompts = df["prompt_id"].astype(str).values

    tok = AutoTokenizer.from_pretrained(args.backbone_dir, use_fast=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    raw_ids = tok(texts, truncation=False, padding=False)["input_ids"]
    win_ids, doc_idx, win_wts = build_covering_windows(raw_ids, max_len=512, stride=384)
    print(f"[{args.tag}] {len(texts)} docs -> {len(win_ids)} covering windows | Layers={capture_layers}")

    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
    Z_win = extract_mos_nf4_streaming(
        args.backbone_dir,
        win_ids,
        pad_id=pad_id,
        devices=devices,
        padding_side=tok.padding_side,
        tag=args.tag,
        micro_bsz=16,
        max_tokens_per_dev=120000,
        capture_layers=capture_layers,
    )
    Z_doc = aggregate_window_embeddings(Z_win, doc_idx, win_wts, len(texts), unit_rms=False)
    np.save(out_dir / f"features_{args.tag.lower()}_8depth_train.npy", Z_doc)

    # Compacta cada fatia por PCA para caber < 25 MB em `.safetensors` e valida 5-Fold OOF
    n_slices = len(capture_layers)
    d_model = Z_doc.shape[1] // n_slices
    F_120 = extract_forensic_compass1_120d(texts)

    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_logits = np.zeros((len(texts), len(LABELS)), dtype=np.float32)
    for tr_i, va_i in sgkf.split(Z_doc, y, groups=prompts):
        tr_parts, va_parts = [], []
        for s_i in range(n_slices):
            sl_tr = Z_doc[tr_i, s_i * d_model : (s_i + 1) * d_model]
            sl_va = Z_doc[va_i, s_i * d_model : (s_i + 1) * d_model]
            pca = PCA(n_components=min(args.pca_per_slice, len(tr_i) - 1), random_state=42 + s_i).fit(sl_tr)
            tr_parts.append(pca.transform(sl_tr))
            va_parts.append(pca.transform(sl_va))
        sc_f = StandardScaler().fit(F_120[tr_i])
        X_tr = np.hstack(tr_parts + [sc_f.transform(F_120[tr_i]) * 0.5])
        X_va = np.hstack(va_parts + [sc_f.transform(F_120[va_i]) * 0.5])

        # Plain + Prompt-Centered Twin
        sp = StandardScaler().fit(X_tr)
        cp = LogisticRegression(C=0.06, max_iter=800, random_state=42).fit(sp.transform(X_tr), y[tr_i])

        def center(X, g):
            Xc = np.zeros_like(X)
            for u in np.unique(g):
                m = g == u
                if m.sum() > 1:
                    Xc[m] = X[m] - X[m].mean(0, keepdims=True)
            return Xc

        X_tr_c = center(X_tr, prompts[tr_i])
        X_va_c = center(X_va, prompts[va_i])
        sc_c = StandardScaler().fit(X_tr_c)
        cc = LogisticRegression(C=0.06, max_iter=800, random_state=42).fit(sc_c.transform(X_tr_c), y[tr_i])
        oof_logits[va_i] = 0.4 * (sp.transform(X_va) @ cp.coef_.T + cp.intercept_) + 0.6 * (sc_c.transform(X_va_c) @ cc.coef_.T + cc.intercept_)

    oof_acc = balanced_accuracy_score(y, oof_logits.argmax(axis=1))
    print(f"🏆 [{args.tag}] 8-Depth + Covering Windows 5-Fold OOF Balanced Accuracy: {oof_acc*100:.2f}%")

    # Ajusta em 100% de train.csv e exporta safetensors compacto
    tensors = {}
    full_parts = []
    for s_i in range(n_slices):
        sl_full = Z_doc[:, s_i * d_model : (s_i + 1) * d_model]
        pca = PCA(n_components=min(args.pca_per_slice, len(texts) - 1), random_state=42 + s_i).fit(sl_full)
        tensors[f"slice{s_i}.pca.mean"] = torch.from_numpy(pca.mean_.astype(np.float32))
        tensors[f"slice{s_i}.pca.components"] = torch.from_numpy(pca.components_.astype(np.float32))
        full_parts.append(pca.transform(sl_full))

    sc_f_full = StandardScaler().fit(F_120)
    tensors["f120.mean"] = torch.from_numpy(sc_f_full.mean_.astype(np.float32))
    tensors["f120.scale"] = torch.from_numpy(sc_f_full.scale_.astype(np.float32))
    X_full = np.hstack(full_parts + [sc_f_full.transform(F_120) * 0.5]).astype(np.float32)

    sp_f = StandardScaler().fit(X_full)
    cp_f = LogisticRegression(C=0.06, max_iter=800, random_state=42).fit(sp_f.transform(X_full), y)
    X_full_c = np.zeros_like(X_full)
    for u in np.unique(prompts):
        m = prompts == u
        if m.sum() > 1:
            X_full_c[m] = X_full[m] - X_full[m].mean(0, keepdims=True)
    sc_cf = StandardScaler().fit(X_full_c)
    cc_f = LogisticRegression(C=0.06, max_iter=800, random_state=42).fit(sc_cf.transform(X_full_c), y)

    for k, arr in [
        ("plain.scaler_mean", sp_f.mean_), ("plain.scaler_scale", sp_f.scale_),
        ("plain.weight", cp_f.coef_), ("plain.bias", cp_f.intercept_),
        ("twin.scaler_mean", sc_cf.mean_), ("twin.scaler_scale", sc_cf.scale_),
        ("twin.weight", cc_f.coef_), ("twin.bias", cc_f.intercept_),
    ]:
        tensors[k] = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

    st_path = out_dir / f"head_{args.tag.lower()}_8depth.safetensors"
    save_file(tensors, str(st_path))
    (out_dir / f"meta_{args.tag.lower()}_8depth.json").write_text(
        json.dumps({"tag": args.tag, "capture_layers": capture_layers, "oof_bal_acc": float(oof_acc)}, indent=2)
    )
    print(f"✓ Saved {st_path} ({st_path.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
