#!/usr/bin/env python3
"""
Local Validation & Inference Test for Tri-Compass 70B MoS Head.

Tests:
1. Integrity of head_tricompass_70b.safetensors & meta_tricompass_70b.json.
2. 28-dimensional stylometric feature extraction & scaling.
3. Plain and Centered inference on real 70B MoS feature representations (features_70b_train_full.npy).
4. Topic clustering and Delta-Origin centering logic.
5. Softmax, vote weighting, Sinkhorn equalization, and submission schema validation.
"""

import sys
import json
import time
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

OFFICIAL_LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
K = len(OFFICIAL_LABELS)

def extract_tricompass_style_features(texts):
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

def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def sinkhorn(P, iters=300):
    c = np.ones(K)
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        R = P * c
        R /= R.sum(axis=1, keepdims=True)
        c *= target / R.sum(axis=0)
    R = P * c
    return R / R.sum(axis=1, keepdims=True)

def main():
    print("=" * 70)
    print("🧪 LOCAL INFERENCE & VALIDATION SUITE: TRI-COMPASS 70B MoS")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parent
    head_path = base_dir / "head_artifacts" / "head_tricompass_70b.safetensors"
    meta_path = base_dir / "head_artifacts" / "meta_tricompass_70b.json"
    train_csv_path = base_dir / "data" / "train.csv"
    test_csv_path = base_dir / "data" / "test.csv"
    feats_npy_path = base_dir / "head_artifacts" / "features_70b_train_full.npy"

    # Test 1: File Existence
    print("\n[Teste 1/6] Verificando existência de todos os arquivos...")
    for p in [head_path, meta_path, train_csv_path, test_csv_path, feats_npy_path]:
        assert p.exists(), f"Arquivo ausente: {p}"
        print(f"  ✓ {p.name} ({p.stat().st_size / 1e6:.2f} MB)")
    print("✓ Todos os arquivos de artefato e dados presentes!")

    # Test 2: Head Safetensors & Metadata
    print("\n[Teste 2/6] Validando pesos e metadados safetensors...")
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["labels"] == OFFICIAL_LABELS, "Ordem das classes difere do oficial Kaggle!"
    assert meta["features_dim"] == 8220, f"Dimensão incorreta: {meta['features_dim']}"
    print(f"  ✓ Metadados conferem: {len(meta['labels'])} classes, {meta['features_dim']} dimensões.")

    head = {}
    with safe_open(head_path, framework="pt") as f:
        for k in f.keys():
            head[k] = f.get_tensor(k).numpy()
    required_keys = [
        "plain.weight", "plain.bias", "centered.weight", "centered.bias",
        "scaler_plain.mean", "scaler_plain.scale",
        "scaler_centered.mean", "scaler_centered.scale",
        "style_scaler.mean", "style_scaler.scale"
    ]
    for rk in required_keys:
        assert rk in head, f"Chave obrigatória ausente no safetensors: {rk}"
    assert head["plain.weight"].shape == (18, 8220)
    assert head["plain.bias"].shape == (18,)
    assert head["centered.weight"].shape == (18, 8220)
    assert head["centered.bias"].shape == (18,)
    assert head["style_scaler.mean"].shape == (28,)
    print(f"  ✓ Todos os tensores verificados com shapes exatos: [18, 8220] e [28].")

    # Test 3: Inference on 100 Real Passages from Train
    print("\n[Teste 3/6] Testando inferência em 100 passagens reais com features 70B MoS...")
    df_train = pd.read_csv(train_csv_path)
    sample_texts = df_train["text"].iloc[:100].tolist()
    sample_labels = df_train["model"].iloc[:100].values

    feats_70b_full = np.load(feats_npy_path)
    feats_70b_sample = feats_70b_full[:100]

    # Estilometria
    style_raw = extract_tricompass_style_features(sample_texts)
    style_normed = (style_raw - head["style_scaler.mean"]) / head["style_scaler.scale"]
    assert style_normed.shape == (100, 28)

    # Plain Head
    X_plain_raw = np.hstack([feats_70b_sample, style_normed])
    X_plain = (X_plain_raw - head["scaler_plain.mean"]) / head["scaler_plain.scale"]
    logits_plain = X_plain @ head["plain.weight"].T + head["plain.bias"]
    prob_plain = softmax(logits_plain)
    assert prob_plain.shape == (100, 18)
    assert np.allclose(prob_plain.sum(axis=1), 1.0)
    print("  ✓ Plain Head calculou probabilidades finitas com soma 1.0.")

    # Centered Head
    X_c_raw = np.hstack([feats_70b_sample, style_normed])
    X_c = (X_c_raw - head["scaler_centered.mean"]) / head["scaler_centered.scale"]
    logits_c = X_c @ head["centered.weight"].T + head["centered.bias"]
    prob_c = softmax(logits_c)

    # Blended Dual-Head
    prob_dual = 0.7 * prob_plain + 0.3 * prob_c
    preds_idx = prob_dual.argmax(axis=1)
    preds_labels = [OFFICIAL_LABELS[i] for i in preds_idx]
    acc = np.mean(preds_labels == sample_labels)
    print(f"  ✓ Acurácia amostral nos dados de treino: {acc*100:.2f}%")

    # Test 4: Topic Clustering on test.csv (3 rows)
    print("\n[Teste 4/6] Validando clustering temático em test.csv (3 linhas)...")
    df_test = pd.read_csv(test_csv_path)
    N_test = len(df_test)
    assert N_test == 3

    Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(df_test["text"].tolist())
    D = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
    np.fill_diagonal(D, 0)
    groups = fcluster(linkage(squareform(D, checks=False), method="average"), 0.85, criterion="distance")
    size = np.bincount(groups)[groups]
    twin_rows = (size >= 2) & (size <= 18)
    print(f"  ✓ Passagens isoladas: {(size == 1).sum()} | Em clusters twin: {twin_rows.sum()}")

    # Test 5: Ensemble Simulator with Baseline Voting & Sinkhorn
    print("\n[Teste 5/6] Simulando ensemble completo (SVC + LoRA + DeBERTa + 70B MoS)...")
    LOGIT = np.zeros((N_test, K))

    # Mock baseline logits from public 0.622 model
    mock_svc_prob = np.full((N_test, K), 1.0 / K)
    mock_lora_prob = np.full((N_test, K), 1.0 / K)
    mock_deberta_prob = np.full((N_test, K), 1.0 / K)

    def vote(name, P, w, rows=slice(None)):
        LOGIT[rows] += w * np.log(np.clip(P, 1e-9, 1))
        print(f"  ✓ Voto registrado: {name} (peso {w})")

    vote("svc", mock_svc_prob, 1.0)
    vote("lora_qwen3_14b", mock_lora_prob, 0.4)
    vote("deberta", mock_deberta_prob, 0.25)

    # 70B MoS vote
    # Mock Z_70b for the 3 test rows
    mock_Z_70b = np.tile(feats_70b_full.mean(axis=0), (N_test, 1))
    test_style_raw = extract_tricompass_style_features(df_test["text"].tolist())
    test_style_normed = (test_style_raw - head["style_scaler.mean"]) / head["style_scaler.scale"]
    X_test_plain_raw = np.hstack([mock_Z_70b, test_style_normed])
    X_test_plain = (X_test_plain_raw - head["scaler_plain.mean"]) / head["scaler_plain.scale"]
    prob_70b_test = softmax(X_test_plain @ head["plain.weight"].T + head["plain.bias"])

    W_70B = 0.35
    vote("qwen72b_mos_tricompass", prob_70b_test, W_70B)

    # Test 6: Final Submission Parquet Generation
    print("\n[Teste 6/6] Verificando geração do arquivo de submissão...")
    assert np.isfinite(LOGIT).all(), "Logits não finitos detectados!"
    P_final = softmax(LOGIT)
    if N_test >= 50:
        P_final = sinkhorn(P_final)
    assert np.isfinite(P_final).all()
    assert P_final.shape == (N_test, K)

    sub = pd.DataFrame({
        "id": df_test["id"].astype(str),
        "model": np.array(OFFICIAL_LABELS)[P_final.argmax(axis=1)]
    })
    assert len(sub) == N_test
    assert sub["model"].isin(OFFICIAL_LABELS).all()
    sub_out = base_dir / "submission_test.parquet"
    sub.to_parquet(sub_out, index=False)
    assert sub_out.exists()
    print("  ✓ submission_test.parquet gerado com sucesso:")
    print(sub)

    # Limpeza
    sub_out.unlink()
    print("\n" + "=" * 70)
    print("🎯 TODOS OS 6 TESTES LOCAIS PASSARAM COM 100% DE SUCESSO!")
    print("=" * 70)

if __name__ == "__main__":
    main()
