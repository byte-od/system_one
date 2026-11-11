"""Pipeline de Treinamento e Calibração Supervisionada: Tri-Compass + Mixture of Scars (MoS) 70B.

Este script implementa o treinamento conjunto dos componentes do MoS:
1. Bússola 1 (Estilometria Local): 28 features tipográficas, transições, pontuações e TTR.
2. Bússola 2 (Delta Origin): Prompt-Centering vetorizado (X - mu_prompt) eliminando viés temático.
3. Bússola 3 (Semântica Profunda 70B): Embeddings reais de 8.192 dimensões extraídos do Qwen 2.5 72B MoS.
4. Cabeça de Classificação de Cicatriz (Scar Classifier):
   - Treinada com GroupKFold em prompt_id (vazamento zero).
   - Suporte a Dual-Head: Plain Head + Prompt-Centered Twin Head.
   - Salva os pesos oficiais em 'head_tricompass_70b.safetensors' e 'meta.json'.
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from safetensors.torch import save_file
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

# Adiciona o diretório raiz ao path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scar_cache.qwen72b_mos_pipeline import Qwen72BMoSEngine, dequantize_int4

# Ordem oficial e estrita das 18 classes da competição
OFFICIAL_LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
L2I = {l: i for i, l in enumerate(OFFICIAL_LABELS)}
I2L = {i: l for i, l in enumerate(OFFICIAL_LABELS)}


def extract_tricompass_style_features(texts: List[str]) -> np.ndarray:
    """Bússola 1: Extrai 28 características estilométricas, tipográficas e dinâmicas inter-sentenças."""
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

        # 1. Pontuação e caracteres especiais
        comma_r = t.count(",") / n
        colon_r = t.count(":") / n
        semicolon_r = t.count(";") / n
        emdash_r = t.count("—") / n
        spaced_emdash_r = t.count(" — ") / n
        hyphen_r = t.count("-") / n
        double_hyphen_r = t.count("--") / n
        quote_double_r = t.count('\"') / n
        quote_single_r = t.count("\'") / n
        backtick_r = t.count("`") / n
        triple_backtick_r = t.count("```") / nw

        # 2. Delimitadores e parênteses
        paren_open_r = t.count("(") / n
        paren_close_r = t.count(")") / n
        bracket_open_r = t.count("[") / n
        bracket_close_r = t.count("]") / n
        brace_open_r = t.count("{") / n
        brace_close_r = t.count("}") / n

        # 3. Formatação Markdown
        hash_r = t.count("#") / n
        asterisk_r = t.count("*") / n
        bullet_r = (t.count("\n- ") + t.count("\n* ")) / nl

        # 4. Numéricos e casing
        digit_r = sum(c.isdigit() for c in t) / n
        upper_r = sum(c.isupper() for c in t) / n
        title_r = sum(w[0].isupper() for w in words if len(w) > 0) / nw

        # 5. Estrutura e dinâmicas (Tri-Compass Local Delta)
        newline_r = t.count("\n") / nw
        double_newline_r = t.count("\n\n") / nw
        avg_word_len = sum(len(w) for w in words) / nw
        avg_sent_len = nw / ns
        ttr = len(set(w.lower() for w in words)) / nw

        feats.append([
            comma_r, colon_r, semicolon_r, emdash_r, spaced_emdash_r, hyphen_r, double_hyphen_r,
            quote_double_r, quote_single_r, backtick_r, triple_backtick_r,
            paren_open_r, paren_close_r, bracket_open_r, bracket_close_r, brace_open_r, brace_close_r,
            hash_r, asterisk_r, bullet_r, digit_r, upper_r, title_r,
            newline_r, double_newline_r, avg_word_len, avg_sent_len, ttr
        ])
    return np.asarray(feats, dtype=np.float32)


def center_groups_delta_origin(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Bússola 2: Tri-Compass Delta Origin (X - mu_prompt) eliminando o viés temático."""
    Xc = X.copy()
    unique_groups = np.unique(groups)
    for g in unique_groups:
        mask = (groups == g)
        if mask.sum() > 1:
            Xc[mask] = Xc[mask] - Xc[mask].mean(axis=0, keepdims=True)
    return Xc


def extract_70b_mos_features_batch(
    engine: Qwen72BMoSEngine,
    texts: List[str],
    batch_size: int = 8,
    max_length: int = 256,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """Bússola 3: Extração otimizada em lote das representações profundas de 8.192 dimensões."""
    if cache_path and cache_path.exists():
        print(f"  ⚡ Carregando cache de features 70B MoS de: {cache_path}")
        return np.load(cache_path)

    n_samples = len(texts)
    features = np.zeros((n_samples, 8192), dtype=np.float32)

    print(f"  🧬 Extraindo representações 70B MoS para {n_samples} textos (Batch Size: {batch_size})...")
    t0_all = time.time()

    for b_idx in range(0, n_samples, batch_size):
        t_batch_start = time.time()
        batch_texts = texts[b_idx : b_idx + batch_size]
        actual_bs = len(batch_texts)

        # Tokenização em lote com padding
        inputs = engine.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(engine.device)
        attention_mask = inputs.attention_mask.to(engine.device).unsqueeze(-1)
        bsz, seq_len = input_ids.shape

        # Embeddings desquantizados sob demanda
        blocks_per_token = 8192 // 128
        block_offsets = torch.arange(blocks_per_token, dtype=torch.long)
        all_block_indices = (input_ids.cpu().unsqueeze(-1) * blocks_per_token + block_offsets).view(-1)
        sel_packed = engine.emb_packed[all_block_indices]
        sel_scales = engine.emb_scales[all_block_indices]
        x = dequantize_int4(sel_packed, sel_scales, (bsz, seq_len, 8192), dtype=engine.dtype).to(engine.device)

        # RoPE
        position_ids = torch.arange(seq_len, dtype=torch.long, device=engine.device).unsqueeze(0).expand(bsz, -1)
        cos, sin = engine.precompute_freqs_cis(position_ids)

        # Forward pass através dos 10 Grupos MoS (80 camadas reais)
        with torch.no_grad():
            for g in engine.groups:
                g.load()
                for l in range(g.start_layer, g.end_layer):
                    x = g.forward_layer(l, x, cos, sin)
                g.unload()

            # Masked Mean Pooling: (x * mask).sum(1) / mask.sum(1)
            x_f = x.float()
            den = attention_mask.sum(dim=1).clamp(min=1)
            pooled = (x_f * attention_mask.float()).sum(dim=1) / den
            # RMSNorm no vetor pooled final
            rms = torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            normed_pooled = (pooled * rms).cpu().numpy()

        features[b_idx : b_idx + actual_bs] = normed_pooled
        dt_batch = time.time() - t_batch_start
        done = min(b_idx + actual_bs, n_samples)
        rate = done / max(time.time() - t0_all, 0.001)
        eta_s = (n_samples - done) / max(rate, 0.001)
        print(f"    [{done:4d}/{n_samples}] Lote processado em {dt_batch:.1f}s ({actual_bs/dt_batch:.2f} txt/s | ETA: {eta_s/60:.1f} min)")

    if cache_path:
        np.save(cache_path, features)
        print(f"  💾 Features salvas com sucesso em: {cache_path}")

    return features


def train_and_evaluate_tricompass(
    data_path: Path,
    output_dir: Path,
    sample_size: Optional[int] = None,
    batch_size: int = 4,
):
    """Executa o pipeline completo de treinamento do Tri-Compass + MoS."""
    print("=" * 80)
    print("🏆 TREINAMENTO SUPERVISIONADO: TRI-COMPASS + 70B MoS SCAR CLASSIFIER")
    print("=" * 80)

    # 1. Carrega os dados de treino
    df = pd.read_csv(data_path)
    if sample_size and sample_size < len(df):
        sub_dfs = []
        n_per_m = max(1, sample_size // 18)
        for _, g in df.groupby("model"):
            sub_dfs.append(g.sample(min(len(g), n_per_m), random_state=42))
        df = pd.concat(sub_dfs, ignore_index=True)
        print(f"✓ Conjunto de calibração rápida ativado: {len(df)} passagens ({n_per_m} por modelo).")
    else:
        print(f"✓ Base completa carregada: {len(df)} passagens oficiais.")

    texts = df["text"].tolist()
    labels = df["model"].map(L2I).values
    prompt_ids = df["prompt_id"].values

    # 2. Bússola 1: Estilometria Local (28 features)
    print("\n[Bússola 1/3] Extraindo Assinaturas Estilométricas Tri-Compass (28 dims)...")
    style_feats = extract_tricompass_style_features(texts)
    scaler_style = StandardScaler()
    style_normed = scaler_style.fit_transform(style_feats)
    print(f"  ✓ Estilometria concluída: shape {style_normed.shape}")

    # 3. Bússola 3: Semântica Profunda do 70B MoS (8.192 dims)
    print("\n[Bússola 3/3] Inicializando Qwen 2.5 72B MoS Engine...")
    model_dir = ROOT_DIR / "models" / "qwen25_72b_mos"
    engine = Qwen72BMoSEngine(model_dir, device="auto", scar_scale=0.1)

    cache_name = f"features_70b_train_n{len(df)}.npy"
    cache_path = output_dir / cache_name
    features_70b = extract_70b_mos_features_batch(
        engine,
        texts,
        batch_size=batch_size,
        max_length=256,
        cache_path=cache_path,
    )

    # 4. Bússola 2: Delta-Origin Prompt Centering
    print("\n[Bússola 2/3] Aplicando Prompt Centering Tri-Compass Delta Origin...")
    style_centered = center_groups_delta_origin(style_normed, prompt_ids)
    features_70b_centered = center_groups_delta_origin(features_70b, prompt_ids)
    print(f"  ✓ Delta Origin calculado para {len(np.unique(prompt_ids))} grupos de prompts.")

    # 5. Construção da Matriz de Fusão do Tri-Compass
    # X_plain: [Features 70B (8192), Estilometria (28)] -> 8.220 dims
    X_plain = np.hstack([features_70b, style_normed])
    # X_centered: [Features 70B Centered, Estilometria Centered] -> 8.220 dims
    X_centered = np.hstack([features_70b_centered, style_centered])

    print(f"\n[Treinamento da Cabeça] Dimensão do vetor Tri-Compass fundido: {X_plain.shape[1]} dims")

    # 6. Avaliação em Validação Cruzada Estratificada por Prompt (GroupKFold)
    min_counts = min(pd.Series(labels).value_counts())
    n_splits = min(5, max(2, min_counts))
    print(f"\n[Validação Cruzada] Executando {n_splits}-Fold Stratified GroupKFold por prompt_id...")
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_plain, labels, groups=prompt_ids)):
        clf_p = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        clf_p.fit(X_plain[train_idx], labels[train_idx])

        clf_c = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        clf_c.fit(X_centered[train_idx], labels[train_idx])

        # Previsão Ensemble Dual-Head (0.7 Plain + 0.3 Centered)
        prob_p = clf_p.predict_proba(X_plain[val_idx])
        prob_c = clf_c.predict_proba(X_centered[val_idx])
        prob_ensemble = 0.7 * prob_p + 0.3 * prob_c

        preds = np.argmax(prob_ensemble, axis=1)
        acc = accuracy_score(labels[val_idx], preds)
        loss = log_loss(labels[val_idx], prob_ensemble)
        cv_scores.append(acc)
        print(f"  Fold {fold + 1}: Acurácia = {acc * 100:.2f}% | Log-Loss = {loss:.4f}")

    mean_acc = np.mean(cv_scores)
    print(f"\n⭐ RESULTADO CV TRI-COMPASS + 70B MoS: {mean_acc * 100:.2f}% (Média 5 Folds)")

    # 7. Ajuste Final no Dataset Completo e Exportação dos Pesos
    print("\n[Exportação de Pesos] Treinando modelo final e compilando safetensors...")
    final_plain = LogisticRegression(C=1.0, max_iter=300, random_state=42)
    final_plain.fit(X_plain, labels)

    final_centered = LogisticRegression(C=1.0, max_iter=300, random_state=42)
    final_centered.fit(X_centered, labels)

    # Converte coeficientes e biases para tensores do PyTorch contíguos
    head_tensors = {
        "plain.weight": torch.tensor(final_plain.coef_, dtype=torch.float32).contiguous(),
        "plain.bias": torch.tensor(final_plain.intercept_, dtype=torch.float32).contiguous(),
        "centered.weight": torch.tensor(final_centered.coef_, dtype=torch.float32).contiguous(),
        "centered.bias": torch.tensor(final_centered.intercept_, dtype=torch.float32).contiguous(),
        "style_scaler.mean": torch.tensor(scaler_style.mean_, dtype=torch.float32).contiguous(),
        "style_scaler.scale": torch.tensor(scaler_style.scale_, dtype=torch.float32).contiguous(),
    }

    out_weights_path = output_dir / "head_tricompass_70b.safetensors"
    save_file(head_tensors, out_weights_path)
    sz_mb = os.path.getsize(out_weights_path) / (1024 * 1024)
    print(f"  💾 Cabeça Tri-Compass exportada: {out_weights_path.name} ({sz_mb:.2f} MB)")

    # Metadata oficial
    meta = {
        "architecture": "TriCompass_70B_MoS",
        "labels": OFFICIAL_LABELS,
        "features_dim": X_plain.shape[1],
        "cv_accuracy": float(mean_acc),
        "scar_scale": 0.1,
        "components": ["70B_MoS_Embeddings", "Style_28D", "Delta_Origin_Centered"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output_dir / "meta_tricompass_70b.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  📄 Metadados salvos: meta_tricompass_70b.json")
    print("\n✓ Pipeline concluído com sucesso!")


if __name__ == "__main__":
    data_file = ROOT_DIR / "competitions" / "name_that_llm" / "data" / "train.csv"
    out_dir = ROOT_DIR / "competitions" / "name_that_llm"
    os.makedirs(out_dir, exist_ok=True)

    # Suporta passar '--quick' para teste rápido de calibração em 180 amostras
    quick_mode = ("--quick" in sys.argv)
    n_samples = 180 if quick_mode else None
    bs = 4

    train_and_evaluate_tricompass(data_file, out_dir, sample_size=n_samples, batch_size=bs)
