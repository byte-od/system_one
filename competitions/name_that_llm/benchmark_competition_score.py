#!/usr/bin/env python3
"""
Benchmark Oficial de Competição: Name That LLM.
Mensa o resultado local rigoroso (Zero-Leakage Stratified GroupKFold por prompt_id)
comparando:
  1. Baseline N-Grams + Estilometria (sem 70B)
  2. Modelo Real 70B MoS (Qwen 2.5 72B Instruct - 8.192D)
  3. Ensemble Tri-Compass SOTA (N-Grams + 70B MoS + Prompt Centering + Hungarian)
"""

import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)

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

def capped_agglo(S: np.ndarray, cap: int = 18, thr: float = 0.04) -> np.ndarray:
    n = len(S)
    A = S.astype(np.float32).copy()
    np.fill_diagonal(A, -np.inf)
    size = np.ones(n, np.int32)
    alive = np.ones(n, bool)
    lab = np.arange(n)
    while True:
        ok = (size[:, None] + size[None, :] <= cap) & alive[:, None] & alive[None, :]
        M = np.where(ok, A, -np.inf)
        k = int(np.argmax(M))
        i, j = divmod(k, n)
        if not np.isfinite(M[i, j]) or M[i, j] < thr:
            break
        new = (size[i] * A[i] + size[j] * A[j]) / (size[i] + size[j])
        A[i] = new
        A[:, i] = new
        A[i, i] = -np.inf
        A[j] = -np.inf
        A[:, j] = -np.inf
        size[i] += size[j]
        alive[j] = False
        lab[lab == j] = i
    return lab

def center_groups(X: np.ndarray, g: np.ndarray) -> np.ndarray:
    X = X.copy()
    for u in np.unique(g):
        idx = np.where(g == u)[0]
        if len(idx) > 1:
            X[idx] -= X[idx].mean(axis=0)
    return X

def apply_hungarian(log_probs: np.ndarray, groups: np.ndarray) -> np.ndarray:
    preds = log_probs.argmax(axis=1)
    preds_hung = preds.copy()
    for g in np.unique(groups):
        idx_g = np.where(groups == g)[0]
        if len(idx_g) > 1 and len(idx_g) <= 18:
            cost = -log_probs[idx_g]
            r, c = linear_sum_assignment(cost)
            preds_hung[idx_g[r]] = c
    return preds_hung

def main():
    print("=" * 80)
    print("🏆 BENCHMARK OFICIAL DE COMPETIÇÃO: NAME THAT LLM")
    print("   Validação Rigorosa StratifiedGroupKFold por prompt_id (Zero Leakage)")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent
    train_csv = base_dir / "data" / "train.csv"
    feats_70b_file = base_dir / "head_artifacts" / "features_70b_train_full.npy"

    df = pd.read_csv(train_csv)
    y = df["model"].map(L2I).values
    prompts = df["prompt_id"].values
    texts = df["text"].tolist()
    N = len(df)
    print(f"Dataset Oficial: {N} passagens rotuladas | {len(np.unique(prompts))} prompts únicos.")

    # 1. Extração de Features N-Gram e Estilo
    print("\n[1/4] Extraindo Assinaturas N-Gram (Char + Word) e 28 Estilométricas...")
    t0 = time.time()
    v_word = TfidfVectorizer(ngram_range=(1, 2), max_features=12000, sublinear_tf=True)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=25000, sublinear_tf=True)
    X_w = v_word.fit_transform(texts)
    X_c = v_char.fit_transform(texts)
    style_raw = extract_tricompass_style_features(texts)
    style_normed = StandardScaler().fit_transform(style_raw)
    X_ngram_style = hstack([X_w, X_c, csr_matrix(style_normed)]).tocsr()
    print(f"✓ Matriz N-Gram + Estilo: {X_ngram_style.shape[1]} dimensões ({time.time()-t0:.1f}s).")

    # 2. Carrega Features 70B MoS Reais
    print("\n[2/4] Carregando Features Reais do Modelo 70B MoS (8.192D)...")
    Z_70b = np.load(feats_70b_file)
    print(f"✓ Matriz 70B MoS: shape {Z_70b.shape} (8.192 dimensões oficiais).")

    # Matriz para clusterização temática cega (TF-IDF de tópicos)
    v_topic = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.08)
    X_topic = v_topic.fit_transform(texts)

    # 3. Execução dos Benchmarks em 5 Folds
    print("\n[3/4] Executando Validação Cruzada 5-Fold Stratified GroupKFold...")
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    accs_ngram = []
    accs_70b = []
    accs_ensemble_raw = []
    accs_ensemble_blind_hung = []
    accs_ensemble_oracle = []

    for fold, (trn, val) in enumerate(sgkf.split(Z_70b, y, groups=prompts)):
        y_trn, y_val = y[trn], y[val]
        g_trn, g_val = prompts[trn], prompts[val]

        # --- A. Modelo N-Grams + Estilo (Ridge Classifier calibrado) ---
        clf_ng = RidgeClassifier(alpha=100.0)
        clf_ng.fit(X_ngram_style[trn], y_trn)
        df_ng = clf_ng.decision_function(X_ngram_style[val])
        e_ng = np.exp(df_ng - df_ng.max(axis=1, keepdims=True))
        p_ng = e_ng / e_ng.sum(axis=1, keepdims=True)
        acc_ng = accuracy_score(y_val, p_ng.argmax(axis=1))
        accs_ngram.append(acc_ng)

        # --- B. Modelo 70B MoS (Features 8.192D com LogisticRegression C=0.1) ---
        scaler_70b = StandardScaler()
        Z_trn = scaler_70b.fit_transform(Z_70b[trn])
        Z_val = scaler_70b.transform(Z_70b[val])

        # Centering no treino
        Z_trn_cnt = center_groups(Z_trn, g_trn)

        clf_70b_plain = LogisticRegression(C=0.1, max_iter=200, random_state=42)
        clf_70b_plain.fit(Z_trn, y_trn)
        p_70b_plain = clf_70b_plain.predict_proba(Z_val)

        # Centering cego na validação: infere clusters de tópicos
        X_top_val = X_topic[val]
        sim_val = (X_top_val @ X_top_val.T).toarray()
        g_val_inferred = capped_agglo(sim_val, cap=18, thr=0.04)

        Z_val_cnt_inf = center_groups(Z_val, g_val_inferred)
        clf_70b_cnt = LogisticRegression(C=0.1, max_iter=200, random_state=42)
        clf_70b_cnt.fit(Z_trn_cnt, y_trn)
        p_70b_cnt = clf_70b_cnt.predict_proba(Z_val_cnt_inf)

        p_70b = 0.7 * p_70b_plain + 0.3 * p_70b_cnt
        acc_70b = accuracy_score(y_val, p_70b.argmax(axis=1))
        accs_70b.append(acc_70b)

        # --- C. Ensemble Ponderado (N-Gram + 70B MoS) ---
        log_p_ens = 0.60 * np.log(np.clip(p_ng, 1e-9, 1.0)) + 0.40 * np.log(np.clip(p_70b, 1e-9, 1.0))
        acc_ens_raw = accuracy_score(y_val, log_p_ens.argmax(axis=1))
        accs_ensemble_raw.append(acc_ens_raw)

        # Hungarian nos clusters cegos inferidos
        preds_blind_hung = apply_hungarian(log_p_ens, g_val_inferred)
        acc_bh = accuracy_score(y_val, preds_blind_hung)
        accs_ensemble_blind_hung.append(acc_bh)

        # Oracle (com clusters perfeitos de prompt)
        preds_oracle = apply_hungarian(log_p_ens, g_val)
        acc_oh = accuracy_score(y_val, preds_oracle)
        accs_ensemble_oracle.append(acc_oh)

        print(f"  Fold {fold+1}: N-Gram={acc_ng*100:.2f}% | 70B MoS={acc_70b*100:.2f}% | Ensemble={acc_ens_raw*100:.2f}% | +Hungarian={acc_bh*100:.2f}% | Oracle={acc_oh*100:.2f}%")

    # 4. Relatório Consolidado
    print("\n" + "=" * 80)
    print("📈 RESULTADOS OFICIAIS DO BENCHMARK DE VALIDAÇÃO LOCAL")
    print("=" * 80)
    m_ng = np.mean(accs_ngram) * 100
    m_70b = np.mean(accs_70b) * 100
    m_ens = np.mean(accs_ensemble_raw) * 100
    m_bh = np.mean(accs_ensemble_blind_hung) * 100
    m_oh = np.mean(accs_ensemble_oracle) * 100

    print(f"1. Baseline N-Gram + Estilo:             {m_ng:.2f}% (±{np.std(accs_ngram)*100:.2f}%)")
    print(f"2. Modelo 70B MoS Perceptual:            {m_70b:.2f}% (±{np.std(accs_70b)*100:.2f}%)")
    print(f"3. Ensemble N-Gram + 70B MoS:            {m_ens:.2f}% (±{np.std(accs_ensemble_raw)*100:.2f}%)")
    print(f"4. SOTA Tri-Compass (+Hungarian Cego):   {m_bh:.2f}% (±{np.std(accs_ensemble_blind_hung)*100:.2f}%)")
    print(f"5. Teto Teórico Oracle (Prompts Reais):  {m_oh:.2f}% (±{np.std(accs_ensemble_oracle)*100:.2f}%)")
    print("-" * 80)
    print(f"Referência Kaggle Public Leaderboard:")
    print(f"  • Random Baseline:                      5.55%")
    print(f"  • Nosso Score Verificado Anterior:     62.20%")
    print(f"  • Top 1 Kaggle Líder Atual:            75.20%")
    print("=" * 80)

if __name__ == "__main__":
    main()
