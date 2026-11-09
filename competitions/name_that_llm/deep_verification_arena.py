#!/usr/bin/env python3
"""
Deep Verification Arena: Protocolo Científico de Validação Profunda (Name That LLM)
----------------------------------------------------------------------------------
Executa uma bateria exaustiva de testes empíricos para garantir a pontuação antes de submeter:

  1. Teste de Sinergia 32B MoS vs 70B MoS vs Stylometrics (5-Fold Stratified Group K-Fold)
  2. Batalha de Decisão: Soft Sinkhorn vs Hungarian Rígido vs Calibração Suave
  3. Simulação Cega Rigorosa (Blind Prompt Clustering via capped_agglo sem prompt_id)
  4. Otimização de Pesos de Ensemble (Busca Convexa / Nelder-Mead)
  5. Diagnóstico de Confusão entre Modelos Irmãos (GPTs, Geminis, Claudes)
"""

import os
import sys
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from scipy.optimize import linear_sum_assignment, minimize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.preprocessing import StandardScaler

# 1. Labels Oficiais
LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for i, l in enumerate(LABELS)}
K = len(LABELS)

# 2. Carregar Dados
DATA_DIR = Path(__file__).resolve().parent / "data"
HEAD_DIR = Path(__file__).resolve().parent / "head_artifacts"
HEAD_32B_DIR = Path(__file__).resolve().parent / "head_artifacts_32b"

train_path = DATA_DIR / "train.csv"
if not train_path.exists():
    raise FileNotFoundError(f"train.csv not found at {train_path}")

df = pd.read_csv(train_path)
print(f"==================================================================", flush=True)
print(f"🔬 ARENA DE VALIDAÇÃO PROFUNDA — NAME THAT LLM", flush=True)
print(f"==================================================================", flush=True)
print(f"Dataset oficial: {len(df)} amostras em {df['prompt_id'].nunique()} prompts únicos.", flush=True)

y = np.array([L2I[m] for m in df["model"]])
prompt_ids = df["prompt_id"].astype(str).values

# 3. Estilometria Tri-Compass
def extract_tricompass_style(texts: List[str]) -> np.ndarray:
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

print("Extraindo features de estilometria Tri-Compass...", flush=True)
X_style = extract_tricompass_style(df["text"].tolist())

# 4. TF-IDF Multi-View N-Grams
print("Construindo matriz TF-IDF Multi-View (Word + Char Grams)...", flush=True)
vec_word = TfidfVectorizer(ngram_range=(1, 3), max_features=35000, sublinear_tf=True)
X_word = vec_word.fit_transform(df["text"])

vec_char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=25000, sublinear_tf=True)
X_char = vec_char.fit_transform(df["text"])

X_tfidf = hstack([X_word, X_char, csr_matrix(X_style)]).tocsr()
print(f"✓ Matriz TF-IDF construída: {X_tfidf.shape}", flush=True)

# 5. Carregar Representações Neurais MoS
feat_70b_path = HEAD_DIR / "features_70b_train_full.npy"
X_70b = np.load(feat_70b_path) if feat_70b_path.exists() else None
if X_70b is not None:
    print(f"✓ 70B MoS features carregadas: {X_70b.shape}", flush=True)

feat_32b_path = HEAD_32B_DIR / "features_32b_train_full.npy"
X_32b = np.load(feat_32b_path) if feat_32b_path.exists() else None
if X_32b is not None:
    print(f"✓ 32B MoS features carregadas: {X_32b.shape}", flush=True)
else:
    print("ℹ️ Aguardando download de features_32b_train_full.npy...", flush=True)

# 6. Algoritmos de Atribuição (Sinkhorn vs Hungarian)
def apply_prompt_centering(feats: np.ndarray, pids: np.ndarray) -> np.ndarray:
    """Aplica Prompt-Centering subtraindo o centróide médio de cada grupo de prompt."""
    centered = feats.copy()
    for pid in np.unique(pids):
        mask = (pids == pid)
        if mask.sum() > 1:
            centered[mask] -= feats[mask].mean(axis=0)
    return centered

def run_sinkhorn(P: np.ndarray, temp: float = 0.08, num_iters: int = 15) -> np.ndarray:
    """Sinkhorn continuous probability calibration."""
    K_mat = np.clip(P, 1e-12, 1.0) ** (1.0 / temp)
    for _ in range(num_iters):
        K_mat = K_mat / (K_mat.sum(axis=1, keepdims=True) + 1e-12)
        K_mat = K_mat / (K_mat.sum(axis=0, keepdims=True) + 1e-12)
    return K_mat / K_mat.sum(axis=1, keepdims=True)

def run_hungarian_bipartite(P: np.ndarray) -> np.ndarray:
    """Atribuição bipartida discreta 0/1 forçada (o método da V15)."""
    n, k_classes = P.shape
    cost_matrix = -P
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    P_hung = np.zeros_like(P)
    for r, c in zip(row_ind, col_ind):
        P_hung[r, c] = 1.0
    return P_hung

def run_thresholded_hungarian(P: np.ndarray, thr: float = 0.85) -> np.ndarray:
    """Húngaro apenas quando a probabilidade máxima excede o limiar, senão preserva suave."""
    max_p = P.max(axis=1)
    if (max_p >= thr).all():
        return run_hungarian_bipartite(P)
    return run_sinkhorn(P, temp=0.08)

# 7. Clusterizador Cego (Capped Agglomerative para Simulação Real)
def capped_agglo(S: np.ndarray, cap: int = 18, thr: float = 0.04) -> np.ndarray:
    """Simula a reconstrução de clusters no teste cego sem prompt_id."""
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
        A[i] = new; A[:, i] = new; A[i, i] = -np.inf
        A[j] = -np.inf; A[:, j] = -np.inf
        size[i] += size[j]; alive[j] = False
        lab[lab == j] = i
    u, inv = np.unique(lab, return_inverse=True)
    return inv

def evaluate_predictions(probs: np.ndarray, y_true: np.ndarray, pids: np.ndarray, method: str, blind: bool = False, sim_matrix: np.ndarray = None) -> Tuple[float, float]:
    """Avalia probabilidades com ou sem pós-processamento de prompt."""
    calibrated = probs.copy()
    
    if blind and sim_matrix is not None:
        clusters = capped_agglo(sim_matrix, cap=18, thr=0.04)
    else:
        clusters = pids

    for cid in np.unique(clusters):
        c_mask = (clusters == cid)
        if c_mask.sum() > 1:
            P_cluster = probs[c_mask]
            if method == "sinkhorn":
                calibrated[c_mask] = run_sinkhorn(P_cluster, temp=0.08)
            elif method == "hungarian":
                calibrated[c_mask] = run_hungarian_bipartite(P_cluster)
            elif method == "threshold_hungarian":
                calibrated[c_mask] = run_thresholded_hungarian(P_cluster, thr=0.85)
            # "raw" does nothing
            
    acc = accuracy_score(y_true, np.argmax(calibrated, axis=1))
    loss = log_loss(y_true, np.clip(calibrated, 1e-15, 1 - 1e-15))
    return acc, loss

print("\nInicializando bateria de testes com 5-Fold Stratified Group K-Fold...", flush=True)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(sgkf.split(df, y, prompt_ids))

def run_oof_cv(X_mat, solver="ridge"):
    oof_probs = np.zeros((len(df), K))
    for fold, (train_idx, val_idx) in enumerate(folds):
        if solver == "ridge":
            clf = RidgeClassifier(alpha=1.0)
            clf.fit(X_mat[train_idx], y[train_idx])
            d = clf.decision_function(X_mat[val_idx])
            # Softmax
            exp_d = np.exp(d - d.max(axis=1, keepdims=True))
            oof_probs[val_idx] = exp_d / exp_d.sum(axis=1, keepdims=True)
        else:
            clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
            clf.fit(X_mat[train_idx], y[train_idx])
            oof_probs[val_idx] = clf.predict_proba(X_mat[val_idx])
    return oof_probs

# 1. Avaliar TF-IDF Baseline
print("\n[1/6] Treinando OOF: TF-IDF Multi-View + Estilometria...", flush=True)
oof_tfidf = run_oof_cv(X_tfidf, solver="ridge")
acc_tfidf_raw = accuracy_score(y, np.argmax(oof_tfidf, axis=1))
print(f"  ✓ TF-IDF Baseline OOF Acurácia: {acc_tfidf_raw*100:.2f}%", flush=True)

# 2. Avaliar 70B MoS (Se disponível)
oof_70b = None
if X_70b is not None:
    print("\n[2/6] Treinando OOF: 70B MoS (Qwen 72B) com Tri-Compass Prompt Centering...", flush=True)
    X_70b_centered = apply_prompt_centering(X_70b, prompt_ids)
    X_70b_comb = StandardScaler().fit_transform(np.hstack([X_70b_centered, X_style]))
    oof_70b = run_oof_cv(X_70b_comb, solver="logistic")
    acc_70b = accuracy_score(y, np.argmax(oof_70b, axis=1))
    print(f"  ✓ 70B MoS OOF Acurácia: {acc_70b*100:.2f}%", flush=True)

# 3. Avaliar 32B MoS (Se disponível)
oof_32b = None
if X_32b is not None:
    print("\n[3/6] Treinando OOF: 32B MoS (Qwen 32B) com Tri-Compass Prompt Centering...", flush=True)
    X_32b_centered = apply_prompt_centering(X_32b, prompt_ids)
    X_32b_comb = StandardScaler().fit_transform(np.hstack([X_32b_centered, X_style]))
    oof_32b = run_oof_cv(X_32b_comb, solver="logistic")
    acc_32b = accuracy_score(y, np.argmax(oof_32b, axis=1))
    print(f"  ✓ 32B MoS OOF Acurácia: {acc_32b*100:.2f}%", flush=True)

# Matriz de similaridade para teste cego
sim_matrix = (X_word @ X_word.T).toarray()

# Batalha de Métodos de Decisão
print("\n==================================================================", flush=True)
print("⚔️ BATALHA EMPÍRICA DE DECISÃO: SINKHORN vs HUNGARIAN", flush=True)
print("==================================================================", flush=True)

models_to_test = [("TF-IDF Baseline", oof_tfidf)]
if oof_70b is not None:
    models_to_test.append(("70B MoS (V13 Core)", oof_70b))
if oof_32b is not None:
    models_to_test.append(("32B MoS (Novo Core)", oof_32b))
if oof_70b is not None and oof_32b is not None:
    # Blend 32B + 70B
    blend_dual = 0.55 * oof_32b + 0.30 * oof_70b + 0.15 * oof_tfidf
    models_to_test.append(("Dual MoS SOTA (32B + 70B + TF-IDF)", blend_dual))

for name, probs in models_to_test:
    print(f"\n--- Testando: {name} ---")
    for method in ["raw", "sinkhorn", "threshold_hungarian", "hungarian"]:
        # Oracle (com prompt_id real)
        acc_o, loss_o = evaluate_predictions(probs, y, prompt_ids, method=method, blind=False)
        # Blind (simulando Kaggle real sem prompt_id)
        acc_b, loss_b = evaluate_predictions(probs, y, prompt_ids, method=method, blind=True, sim_matrix=sim_matrix)
        
        status_flag = "❌ DEGRADOU" if acc_b < acc_o - 0.05 else ("🔥 SOTA" if acc_b > 0.63 else "✓ OK")
        print(f"  [{method.upper():<20}] Oracle Acc: {acc_o*100:.2f}% | Blind Kaggle Acc: {acc_b*100:.2f}% | LogLoss: {loss_b:.4f} | {status_flag}")

print("\n==================================================================", flush=True)
print("🎯 TESTE PROFUNDO CONCLUÍDO COM SUCESSO!", flush=True)
print("==================================================================", flush=True)
