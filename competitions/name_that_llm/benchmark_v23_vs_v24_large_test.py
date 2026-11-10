#!/usr/bin/env python3
"""
🧪 BENCHMARK COMPARATIVO EM TEST SET EXPANDIDO: V23 vs V24
Testa ambas as versões em um conjunto de teste held-out de 540 passagens reais
pertencentes a 30 grupos de prompts completamente inéditos (Strict Group Split):

- V23: Composição Pura (LinearSVC 1.0 + Titan 2 Plain 0.60 + Centered Twin 0.20 + Soft Sinkhorn)
- V24: System One Enhanced (Confidence-Gated Dynamic Voting + Test-Time S-Cache 4.1 KB + Soft Sinkhorn)
"""

from __future__ import annotations

import os
import sys
import time
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, log_loss, top_k_accuracy_score

LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)

SMALL_WORDS = set(
    "a an the and or but if while although because as so than that which "
    "who whom whose this these those it its they them their theirs he him his she her hers "
    "we us our ours you your yours in on at by for with about against between into through "
    "during before after above below to from up down in out of off over under again further "
    "then once here there when where why how all any both each few more most other some "
    "such no nor not only own same so than too very can will just should now".split()
)


def extract_tricompass_style_features(texts: list[str]) -> np.ndarray:
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


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def sinkhorn(P: np.ndarray, iters: int = 300) -> np.ndarray:
    c = np.ones(K)
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= target / R.sum(0)
    R = P * c
    return R / R.sum(1, keepdims=True)


def compute_confidence_weights(P: np.ndarray, base_weight: float) -> np.ndarray:
    sorted_P = np.sort(P, axis=1)
    margin = sorted_P[:, -1] - sorted_P[:, -2]
    logK = np.log(float(P.shape[1]))
    entropy = -np.sum(P * np.log(np.clip(P, 1e-9, 1.0)), axis=1) / logK
    certainty = (1.0 - entropy) * (0.4 + 0.6 * margin)
    eff_weight = base_weight * (0.80 + 0.45 * certainty)
    return eff_weight[:, None]


class TestTimeSCache:
    def __init__(self, slot_dim: int = 32, resonance_scale: float = 0.20):
        self.slot_dim = slot_dim
        self.resonance_scale = resonance_scale
        self.proj = None

    def fit_projection(self, in_dim: int):
        np.random.seed(42)
        Q, _ = np.linalg.qr(np.random.randn(in_dim, self.slot_dim))
        self.proj = Q.astype(np.float32)

    def apply_scache_resonance(self, X: np.ndarray, groups: np.ndarray) -> np.ndarray:
        if self.proj is None:
            self.fit_projection(X.shape[1])
        Z = X @ self.proj
        X_out = X.copy()
        for g in np.unique(groups):
            m = (groups == g)
            if m.sum() > 1:
                Z_g = Z[m]
                z_bar = Z_g.mean(axis=0, keepdims=True)
                delta_z = Z_g - z_bar
                W_scar = (delta_z.T @ delta_z) / float(m.sum())
                res_z = delta_z @ W_scar.T
                res_full = res_z @ self.proj.T
                X_out[m] += self.resonance_scale * res_full
        return X_out


def main():
    print("=" * 80)
    print("🔬 TEST SET EXPANDIDO (540 PASSAGENS): V23 vs V24 HEAD-TO-HEAD")
    print("   Divisão Estrita: 2.159 Treino (120 prompts) | 540 Teste Inédito (30 prompts)")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    titan2_dir = base_dir / "titans" / "titan2_yi34b_l30"

    df = pd.read_csv(data_dir / "train.csv")
    y_all = df["model"].map(L2I).values
    prompts_all = df["prompt_id"].values
    texts_all = df["text"].tolist()

    head_yi = load_file(str(titan2_dir / "head_tricompass_yi34b_l30.safetensors"))
    feats_all = np.load(titan2_dir / "features_yi34b_l30_train_full.npy")

    # Estilometria
    style_raw = extract_tricompass_style_features(texts_all)
    style_scaler_mean = head_yi["style_scaler.mean"].float().numpy()
    style_scaler_scale = head_yi["style_scaler.scale"].float().numpy()
    style_normed = (style_raw - style_scaler_mean) / style_scaler_scale
    X_tri_all = np.hstack([feats_all, style_normed])

    # LinearSVC Matriz
    def shape_encode(t):
        return "".join("U" if c.isupper() else "l" if c.islower() else "d" if c.isdigit() else " " if c.isspace() else "." for c in t)
    
    print("\n[1/4] Extraindo features de 4 visões do LinearSVC...")
    c_word = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, max_features=30000).fit_transform(texts_all)
    c_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", sublinear_tf=True, max_features=50000).fit_transform(texts_all)
    shapes = [shape_encode(t) for t in texts_all]
    c_shape = TfidfVectorizer(ngram_range=(2, 6), analyzer="char", sublinear_tf=True, max_features=25000).fit_transform(shapes)
    small_texts = [" ".join(w for w in t.lower().split() if w in SMALL_WORDS) for t in texts_all]
    c_small = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, max_features=10000).fit_transform(small_texts)
    X_svc_all = hstack([c_word, c_char, c_shape, c_small]).tocsr()

    # Divisão de Grupos (Strict 80/20 Group Split)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(X_tri_all, y_all, groups=prompts_all))

    print(f"\n[2/4] Conjunto de Treino: {len(train_idx)} passagens ({len(np.unique(prompts_all[train_idx]))} prompts)")
    print(f"      Conjunto de Teste:  {len(test_idx)} passagens ({len(np.unique(prompts_all[test_idx]))} prompts inéditos)")

    y_tr, y_te = y_all[train_idx], y_all[test_idx]
    p_tr, p_te = prompts_all[train_idx], prompts_all[test_idx]

    # Treinamento nos 80%
    print("\n[3/4] Treinando modelos nos 2.159 exemplos de treino...")
    t0_fit = time.time()
    
    # LinearSVC
    clf_svc = LinearSVC(C=0.1, max_iter=2000, random_state=42)
    clf_svc.fit(X_svc_all[train_idx], y_tr)
    p_svc_te = softmax(clf_svc.decision_function(X_svc_all[test_idx]))

    # Titan 2 Plain
    scaler_p = (X_tri_all[train_idx].mean(axis=0), X_tri_all[train_idx].std(axis=0) + 1e-8)
    X_tr_p = (X_tri_all[train_idx] - scaler_p[0]) / scaler_p[1]
    X_te_p = (X_tri_all[test_idx] - scaler_p[0]) / scaler_p[1]
    clf_plain = LogisticRegression(C=0.1, max_iter=600, random_state=42).fit(X_tr_p, y_tr)
    p_plain_te = softmax(clf_plain.decision_function(X_te_p))

    # Titan 2 Twin Centered
    X_cent_tr = X_tri_all[train_idx].copy()
    for g in np.unique(p_tr):
        m = (p_tr == g)
        if m.sum() > 1:
            X_cent_tr[m] -= X_cent_tr[m].mean(axis=0, keepdims=True)
    scaler_c = (X_cent_tr.mean(axis=0), X_cent_tr.std(axis=0) + 1e-8)
    X_tr_c = (X_cent_tr - scaler_c[0]) / scaler_c[1]
    
    # No teste, centramento por tópicos de teste
    X_cent_te = X_tri_all[test_idx].copy()
    for g in np.unique(p_te):
        m = (p_te == g)
        if m.sum() > 1:
            X_cent_te[m] -= X_cent_te[m].mean(axis=0, keepdims=True)
    X_te_c = (X_cent_te - scaler_c[0]) / scaler_c[1]

    clf_twin = LogisticRegression(C=0.1, max_iter=600, random_state=42).fit(X_tr_c, y_tr)
    p_twin_te = softmax(clf_twin.decision_function(X_te_c))
    print(f"✓ Modelos treinados em {time.time()-t0_fit:.1f}s!")

    # ==========================================================================
    # PIPELINE V23: Composição Pura (Pesos Fixos)
    # ==========================================================================
    t0_v23 = time.time()
    log_v23 = (
        1.00 * np.log(np.clip(p_svc_te, 1e-9, 1))
        + 0.60 * np.log(np.clip(p_plain_te, 1e-9, 1))
        + 0.20 * np.log(np.clip(p_twin_te, 1e-9, 1))
    )
    p_v23 = sinkhorn(softmax(log_v23))
    t_v23 = time.time() - t0_v23

    acc_v23 = accuracy_score(y_te, np.argmax(p_v23, axis=1))
    top3_v23 = top_k_accuracy_score(y_te, p_v23, k=3)
    ll_v23 = log_loss(y_te, p_v23)

    # ==========================================================================
    # PIPELINE V24: System One Enhanced (Confidence Gating + Test-Time S-Cache)
    # ==========================================================================
    t0_v24 = time.time()
    
    # 1. Test-Time S-Cache aplicado ao teste
    scache = TestTimeSCache(slot_dim=32, resonance_scale=0.15)
    X_te_scache = scache.apply_scache_resonance(X_tri_all[test_idx], p_te)
    X_te_scache_norm = (X_te_scache - scaler_p[0]) / scaler_p[1]
    p_plain_scache_te = softmax(clf_plain.decision_function(X_te_scache_norm))

    # 2. Confidence-Gated Dynamic Voting
    w_svc_dyn = compute_confidence_weights(p_svc_te, 1.00)
    w_plain_dyn = compute_confidence_weights(p_plain_scache_te, 0.60)
    w_twin_dyn = compute_confidence_weights(p_twin_te, 0.20)

    log_v24 = (
        w_svc_dyn * np.log(np.clip(p_svc_te, 1e-9, 1))
        + w_plain_dyn * np.log(np.clip(p_plain_scache_te, 1e-9, 1))
        + w_twin_dyn * np.log(np.clip(p_twin_te, 1e-9, 1))
    )
    p_v24 = sinkhorn(softmax(log_v24))
    t_v24 = time.time() - t0_v24

    acc_v24 = accuracy_score(y_te, np.argmax(p_v24, axis=1))
    top3_v24 = top_k_accuracy_score(y_te, p_v24, k=3)
    ll_v24 = log_loss(y_te, p_v24)

    # ==========================================================================
    # RESULTADOS CONSOLIDADOS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("🏆 RESULTADOS CONSOLIDADOS NO TEST SET DE 540 EXEMPLOS INÉDITOS:")
    print("=" * 80)
    print(f"| Métrica                 | V23 (Composição Pura) | V24 (System 1 Enhanced) | Diferencial |")
    print(f"| :---------------------- | :-------------------- | :---------------------- | :---------- |")
    print(f"| Acurácia Top-1          | {acc_v23*100:6.2f}%               | {acc_v24*100:6.2f}%                 | {acc_v24*100 - acc_v23*100:+5.2f}%     |")
    print(f"| Acurácia Top-3          | {top3_v23*100:6.2f}%               | {top3_v24*100:6.2f}%                 | {top3_v24*100 - top3_v23*100:+5.2f}%     |")
    print(f"| Log Loss Calibrado      | {ll_v23:6.4f}                | {ll_v24:6.4f}                 | {ll_v24 - ll_v23:+6.4f}    |")
    print(f"| Latência de Inferência  | {t_v23*1000:6.2f} ms             | {t_v24*1000:6.2f} ms              | {t_v24*1000 - t_v23*1000:+5.2f} ms   |")
    print("=" * 80)

    # Análise de Concordância
    preds_v23 = np.argmax(p_v23, axis=1)
    preds_v24 = np.argmax(p_v24, axis=1)
    agreement = np.mean(preds_v23 == preds_v24)
    print(f"\n📊 Concordância entre V23 e V24: {agreement*100:.2f}% dos casos.")
    diff_mask = (preds_v23 != preds_v24)
    v23_correct_on_diff = np.sum((preds_v23[diff_mask] == y_te[diff_mask]))
    v24_correct_on_diff = np.sum((preds_v24[diff_mask] == y_te[diff_mask]))
    print(f"   Nos {diff_mask.sum()} exemplos onde divergiram:")
    print(f"   • V24 acertou: {v24_correct_on_diff} passagens")
    print(f"   • V23 acertou: {v23_correct_on_diff} passagens")
    if v24_correct_on_diff > v23_correct_on_diff:
        print(f"   🔥 V24 superou V23 por +{v24_correct_on_diff - v23_correct_on_diff} decisões corretas líquidas!")
    print("=" * 80)


if __name__ == "__main__":
    main()
