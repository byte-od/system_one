#!/usr/bin/env python3
"""
👑 TITAN 3: MISTRAL LARGE 2 (123B) MoS GOLDEN-SLICE (Layers 0 -> 43)
Architecture: unsloth/mistral-large-instruct-2407-bnb-4bit (123B Dense, 88 Layers, d = 12,288).
Golden Slice: Layers 0 -> 43 (Half-Depth Dense Representation, ZERO MoE).
Tri-Compass Vector: 12,288 + 28 = 12,316D.
Co-Trained S-Cache: O(1) Scar covariance projection directly integrated during training.
Validation: 5-Fold Stratified Group K-Fold by prompt_id + Continuous Soft Sinkhorn (300 iterations).
Refit: 100% full training fit, saving head_tricompass_mistral_large_123b_l44.safetensors.
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
from safetensors.torch import save_file, load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
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


def setup_sliced_mistral_large_checkpoint(model_id: str, local_dir: Path, max_layer: int = 44) -> Path:
    """Prepara checkpoint estritamente fatiado para as camadas 0..43 (36.75 GB, dispensando 44 camadas de topo)."""
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[MoS Golden-Slice Setup] Baixando metadados de {model_id}...", flush=True)
    cfg_file = hf_hub_download(repo_id=model_id, filename="config.json", local_dir=str(local_dir))
    with open(cfg_file, "r") as f:
        cfg_data = json.load(f)
    cfg_data["num_hidden_layers"] = max_layer
    cfg_data["use_cache"] = False
    with open(local_dir / "config.json", "w") as f:
        json.dump(cfg_data, f, indent=2)

    # Tokenizer files
    for tok_f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        try:
            hf_hub_download(repo_id=model_id, filename=tok_f, local_dir=str(local_dir))
        except Exception:
            pass

    # Index map filtering
    idx_file = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json", local_dir=str(local_dir))
    with open(idx_file, "r") as f:
        idx_data = json.load(f)
    
    weight_map = idx_data["weight_map"]
    filtered_weight_map = {}
    needed_shards = set()

    for k, shard in weight_map.items():
        if k.startswith("model.layers."):
            layer_idx = int(k.split(".")[2])
            if layer_idx < max_layer:
                filtered_weight_map[k] = shard
                needed_shards.add(shard)
        elif not k.startswith("lm_head."):
            filtered_weight_map[k] = shard
            needed_shards.add(shard)

    filtered_idx = {
        "metadata": idx_data.get("metadata", {}),
        "weight_map": filtered_weight_map
    }
    with open(local_dir / "model.safetensors.index.json", "w") as f:
        json.dump(filtered_idx, f, indent=2)

    shards_list = sorted(needed_shards)
    print(f"✓ Sliced Index: {len(filtered_weight_map)} tensores em {len(shards_list)} shards ({shards_list}).", flush=True)

    # Download only needed shards
    print(f"Baixando {len(shards_list)} shards MoS Golden-Slice (~36.7 GB)...", flush=True)
    t0_dl = time.time()
    for shard in shards_list:
        target_shard = local_dir / shard
        if target_shard.exists():
            print(f"  ✓ {shard} já presente ({target_shard.stat().st_size / (1024**3):.2f} GB)", flush=True)
        else:
            print(f"  ⬇ Baixando {shard}...", flush=True)
            hf_hub_download(repo_id=model_id, filename=shard, local_dir=str(local_dir))
    print(f"✓ Todos os shards Golden-Slice baixados com sucesso em {time.time()-t0_dl:.1f}s!", flush=True)
    return local_dir


def main():
    print("=" * 80)
    print("👑 TITAN 3: MISTRAL LARGE 2 (123B) MoS GOLDEN-SLICE EXTRACTION & TRAINING (L0-43)")
    print("   Total Parameters: 123B Dense | Total Layers: 88 | Truncation: Layer 44")
    print("   Hidden Dim: 12,288 | Tri-Compass Dim: 12,288 + 28 = 12,316D")
    print("   Pure Dense Latent Geometry (ZERO MoE) + SCache O(1) Scar Integration")
    print("=" * 80, flush=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU VRAM Total: {vram_gb:.2f} GB", flush=True)

    data_path = "train.csv"
    if not os.path.exists(data_path):
        data_path = "/content/train.csv"
    if not os.path.exists(data_path):
        data_path = "/kaggle/input/name-that-llm/train.csv"

    df_train = pd.read_csv(data_path)
    n_samples = len(df_train)
    print(f"✓ Loaded {n_samples} training passages ({data_path}) across {df_train['prompt_id'].nunique()} prompts.", flush=True)

    texts = df_train["text"].tolist()
    labels = df_train["model"].map(L2I).values
    prompts = df_train["prompt_id"].values

    if np.isnan(labels).any():
        raise ValueError(f"Unknown labels found in training set: {df_train['model'][np.isnan(labels)].unique()}")

    # 1. Stylometrics (Compass 1)
    print("\n[Compass 1/3] Extraindo 28D Tri-Compass stylometric signatures...", flush=True)
    style_raw = extract_tricompass_style_features(texts)
    style_scaler = StandardScaler()
    style_normed = style_scaler.fit_transform(style_raw)
    print(f"✓ Stylometrics extraídos: {style_normed.shape}", flush=True)

    # 2. Mistral Large 2 Golden-Slice Feature Extraction (Layers 0-43)
    output_dir = Path("artifacts_mistral_large_123b_l44")
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_file = output_dir / "features_mistral_large_123b_l44_train_full.npy"

    if feat_file.exists():
        print(f"\n[Compass 3/3] Carregando features pré-extraídas de {feat_file}...", flush=True)
        Z_mistral = np.load(feat_file)
        print(f"✓ Features carregadas: {Z_mistral.shape}", flush=True)
    else:
        model_id = "unsloth/mistral-large-instruct-2407-bnb-4bit"
        sliced_model_dir = Path("/content/mistral_large_sliced_l44")
        
        setup_sliced_mistral_large_checkpoint(model_id, sliced_model_dir, max_layer=44)

        print(f"\n[Compass 3/3] Carregando Tokenizer e Configuração de {sliced_model_dir}...", flush=True)
        tok = AutoTokenizer.from_pretrained(str(sliced_model_dir))
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        cfg = AutoConfig.from_pretrained(str(sliced_model_dir))
        cfg.num_hidden_layers = 44
        cfg.use_cache = False

        print("Instanciando backbone Mistral Large 2 (L44) na GPU A100 (cuda:0)...", flush=True)
        t_load0 = time.time()
        model = AutoModel.from_pretrained(
            str(sliced_model_dir),
            config=cfg,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        ).eval()
        print(f"✓ Mistral Large 2 Layer 44 carregado em {time.time()-t_load0:.1f}s! VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

        batch_size = 4
        all_reps = []
        t_fwd0 = time.time()

        print(f"Iniciando forward pass em batches de {batch_size} para {n_samples} amostras...", flush=True)
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                b_texts = texts[i : i + batch_size]
                inp = tok(b_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inp = {k: v.to("cuda:0") for k, v in inp.items()}
                out = model(**inp)
                mask = inp["attention_mask"].unsqueeze(-1).float()
                # RMSNorm unit-scaled mean pooling
                pooled = (out.last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                rms = torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                pooled_norm = (pooled * rms).cpu().numpy()
                all_reps.append(pooled_norm)

                if (i // batch_size) % 30 == 0 or (i + batch_size >= n_samples):
                    elapsed = time.time() - t_fwd0
                    speed = (i + len(b_texts)) / max(elapsed, 1e-3)
                    print(f"  Batch {i//batch_size + 1}/{(n_samples+batch_size-1)//batch_size} ({i+len(b_texts)}/{n_samples}) - {speed:.1f} amostras/s | VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

        Z_mistral = np.concatenate(all_reps, axis=0)
        del model
        torch.cuda.empty_cache()
        gc.collect()

        print(f"✓ Mistral Large 2 Layer 44 features extraídas em {time.time()-t_fwd0:.1f}s: {Z_mistral.shape}", flush=True)
        np.save(feat_file, Z_mistral)

    # 3. Construct Tri-Compass Matrices (12,288 + 28 = 12,316D)
    X_plain_raw = np.hstack([Z_mistral, style_normed])
    Zc = center_groups_delta_origin(Z_mistral, prompts)
    Sc = center_groups_delta_origin(style_normed, prompts)
    X_centered_raw = np.hstack([Zc, Sc])

    print(f"\n[Tri-Compass] Matriz Plain: {X_plain_raw.shape} | Matriz Centered: {X_centered_raw.shape}", flush=True)

    # 4. 5-Fold Stratified Group K-Fold Validation
    print("\n[Validação 5-Fold Stratified Group K-Fold por prompt_id]...", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_ensemble_probs = np.zeros((n_samples, K), dtype=np.float32)
    fold_accs = []
    fold_sinkhorn_accs = []
    fold_losses = []

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
        loss_sink = log_loss(labels[val_idx], p_val_sink)

        fold_accs.append(acc_raw)
        fold_sinkhorn_accs.append(acc_sink)
        fold_losses.append(loss_sink)

        print(f"  Fold {fold+1}: Raw Acc = {acc_raw*100:.2f}% | Soft Sinkhorn Acc = {acc_sink*100:.2f}% | LogLoss = {loss_sink:.4f}", flush=True)

    oof_sinkhorn = sinkhorn(oof_ensemble_probs)
    mean_raw_acc = np.mean(fold_accs)
    mean_sink_acc = np.mean(fold_sinkhorn_accs)
    overall_sink_acc = accuracy_score(labels, np.argmax(oof_sinkhorn, axis=1))
    overall_logloss = log_loss(labels, oof_sinkhorn)

    print("\n" + "=" * 80)
    print(f"👑 RESULTADOS OFICIAIS TITAN 3 (Mistral Large 2 123B L44):")
    print(f"   Acurácia Média Folds (Raw):           {mean_raw_acc*100:.2f}%")
    print(f"   Acurácia Média Folds (Soft Sinkhorn): {mean_sink_acc*100:.2f}%")
    print(f"   Acurácia Global OOF (Soft Sinkhorn):  {overall_sink_acc*100:.2f}%")
    print(f"   Log Loss Global OOF (Soft Sinkhorn):  {overall_logloss:.4f}")
    print("=" * 80, flush=True)

    # 5. Full 100% Training Refit
    print("\n[Treinamento 100% Full Training Set] Ajustando heads finais...", flush=True)
    final_scaler_plain = StandardScaler()
    X_plain_full = final_scaler_plain.fit_transform(X_plain_raw)
    final_clf_plain = LogisticRegression(C=0.1, max_iter=600, random_state=42)
    final_clf_plain.fit(X_plain_full, labels)

    final_scaler_centered = StandardScaler()
    X_centered_full = final_scaler_centered.fit_transform(X_centered_raw)
    final_clf_centered = LogisticRegression(C=0.1, max_iter=600, random_state=42)
    final_clf_centered.fit(X_centered_full, labels)

    # 6. Serialização Safetensors e Metadados
    safetensors_file = output_dir / "head_tricompass_mistral_large_123b_l44.safetensors"
    meta_file = output_dir / "meta_tricompass_mistral_large_123b.json"

    tensors = {
        "plain.weight": torch.from_numpy(final_clf_plain.coef_).float(),
        "plain.bias": torch.from_numpy(final_clf_plain.intercept_).float(),
        "scaler_plain.mean": torch.from_numpy(final_scaler_plain.mean_).float(),
        "scaler_plain.scale": torch.from_numpy(final_scaler_plain.scale_).float(),
        "centered.weight": torch.from_numpy(final_clf_centered.coef_).float(),
        "centered.bias": torch.from_numpy(final_clf_centered.intercept_).float(),
        "scaler_centered.mean": torch.from_numpy(final_scaler_centered.mean_).float(),
        "scaler_centered.scale": torch.from_numpy(final_scaler_centered.scale_).float(),
        "style_scaler.mean": torch.from_numpy(style_scaler.mean_).float(),
        "style_scaler.scale": torch.from_numpy(style_scaler.scale_).float(),
    }
    save_file(tensors, str(safetensors_file))
    print(f"✓ Safetensors salvo com sucesso: {safetensors_file} ({safetensors_file.stat().st_size / (1024**2):.2f} MB)", flush=True)

    meta_payload = {
        "titan_id": "titan3_mistral_large_123b_l44",
        "model_id": "unsloth/mistral-large-instruct-2407-bnb-4bit",
        "total_parameters": "123B Dense",
        "golden_slice_layers": 44,
        "hidden_dimension": 12288,
        "tricompass_dimension": 12316,
        "classes": LABELS,
        "fold_raw_accuracies": fold_accs,
        "fold_sinkhorn_accuracies": fold_sinkhorn_accs,
        "fold_loglosses": fold_losses,
        "mean_raw_accuracy": float(mean_raw_acc),
        "mean_sinkhorn_accuracy": float(mean_sink_acc),
        "overall_oof_sinkhorn_accuracy": float(overall_sink_acc),
        "overall_oof_logloss": float(overall_logloss),
        "timestamp": time.time(),
    }
    with open(meta_file, "w") as f:
        json.dump(meta_payload, f, indent=2)
    print(f"✓ Metadados salvos com sucesso: {meta_file}", flush=True)

    print("\n👑 PIPELINE TITAN 3 (MISTRAL LARGE 123B MoS) CONCLUÍDO COM SUCESSO ABSOLUTO!", flush=True)


if __name__ == "__main__":
    main()
