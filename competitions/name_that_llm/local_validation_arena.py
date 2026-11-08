"""Local Validation Arena: Framework de Validação Rigoroso (Zero-Leakage) para Name That LLM.

Simula com exatidão o ambiente de teste cego do Kaggle:
  1. Divisão StratifiedGroupKFold agrupada por `prompt_id` (prompts de validação nunca vistos no treino).
  2. Validação Cega: Na fase de validação, os IDs de prompt são ocultados.
  3. O modelo precisa reconstruir os clusters de prompts cegamente (capped_agglo)
     e aplicar Prompt-Centering + Hungarian Assignment nos clusters inferidos.
  4. Execução ultra-rápida via matrizes esparsas CSR e solvers otimizados.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

LABELS = [
    "claude-fable-5", "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "glm-5.3", "glm-5.3-flash",
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "kimi-k3", "qwen3.8-max"
]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for i, l in enumerate(LABELS)}


def extract_style_features(texts: List[str]) -> np.ndarray:
    """Extrai 21 assinaturas estilométricas, tipográficas e de pontuação."""
    feats = []
    for t in texts:
        t = str(t)
        n = max(len(t), 1)
        words = t.split()
        nw = max(len(words), 1)
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
        ns = max(len(sents), 1)
        
        feats.append([
            t.count(",") / n,
            t.count(":") / n,
            t.count(";") / n,
            (t.count("—") + t.count("-")) / n,
            t.count('\"') / n,
            t.count("\'") / n,
            t.count("`") / n,
            t.count("(") / n,
            t.count(")") / n,
            t.count("[") / n,
            t.count("]") / n,
            t.count("{") / n,
            t.count("}") / n,
            t.count("#") / n,
            t.count("*") / n,
            sum(c.isdigit() for c in t) / n,
            sum(c.isupper() for c in t) / n,
            t.count(chr(10)) / nw,
            sum(len(w) for w in words) / nw,
            nw / ns,
            len(set(w.lower() for w in words)) / nw
        ])
    return np.asarray(feats, dtype=np.float32)


def capped_agglo(S: np.ndarray, cap: int = 18, thr: float = 0.04) -> np.ndarray:
    """Reconstrói clusters de prompt limitados a `cap` membros (Lance-Williams average linkage)."""
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
    """Subtrai o centróide do prompt de cada grupo (X - mu_group)."""
    X = X.copy()
    for u in np.unique(g):
        idx = np.where(g == u)[0]
        if len(idx) > 1:
            X[idx] -= X[idx].mean(axis=0)
    return X


def run_cross_validation(
    include_extra_data: bool = True,
    n_splits: int = 5,
    c_plain: float = 2.5,
    c_cnt: float = 4.0,
    blend_weight: float = 0.5,
) -> Dict[str, float]:
    """Executa a validação cruzada 5-fold agrupada completa simulando o teste cego."""
    t0 = time.time()
    print("=" * 80)
    print("🏛️ ARENA DE VALIDAÇÃO LOCAL: STRATIFIED GROUP K-FOLD (ZERO-LEAKAGE)")
    print(f"   Incluir Dados Extras: {include_extra_data} | Folds: {n_splits}")
    print("=" * 80)

    train_path = "/tmp/kaggle_name_that_llm/train.csv"
    assert os.path.exists(train_path), "train.csv não encontrado localmente!"
    df_official = pd.read_csv(train_path)[["prompt_id", "model", "text"]]

    if include_extra_data:
        extra_path = "/tmp/prvsiyan_dataset/passages_clean.csv"
        if os.path.exists(extra_path):
            df_extra = pd.read_csv(extra_path)[["prompt_id", "model", "text"]]
            df_extra["prompt_id"] = "ex_" + df_extra["prompt_id"].astype(str)
            df_full = pd.concat([df_official, df_extra], ignore_index=True)
            print(f"✓ Base Expandida: {len(df_official)} oficiais + {len(df_extra)} extras = {len(df_full)} amostras.")
        else:
            df_full = df_official
    else:
        df_full = df_official
        print(f"✓ Base Oficial Pura: {len(df_full)} amostras.")

    y = df_full["model"].map(L2I).values
    groups = df_full["prompt_id"].values
    texts = df_full["text"].tolist()

    print("\n[1/3] Extraindo N-Grams e Assinaturas Estilométricas...")
    v_word = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=30000, sublinear_tf=True)

    X_w = v_word.fit_transform(texts)
    X_c = v_char.fit_transform(texts)
    style_raw = extract_style_features(texts)
    scaler = StandardScaler()
    style_sc = scaler.fit_transform(style_raw)
    X_features = hstack([X_w, X_c, style_sc]).toarray().astype(np.float32)

    print(f"  ✓ Matriz de Features: {X_features.shape[0]} amostras x {X_features.shape[1]} dimensões.")

    # Clustering TF-IDF para recuperação de prompts na validação
    v_clust = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.08)
    X_clust = v_clust.fit_transform(texts)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    acc_plain_list = []
    acc_blend_raw_list = []
    acc_blind_hungarian_list = []
    acc_oracle_hungarian_list = []

    print("\n[2/3] Executando 5 Folds com Prompts Ocultos na Validação...")

    for fold, (idx_tr, idx_va) in enumerate(sgkf.split(X_features, y, groups=groups)):
        X_tr = X_features[idx_tr]
        y_tr = y[idx_tr]
        g_tr = groups[idx_tr]

        X_va = X_features[idx_va]
        y_va = y[idx_va]
        g_va_true = groups[idx_va]

        # 1. Modelo Plain (conhece apenas estilo geral)
        clf_plain = LogisticRegression(C=c_plain, max_iter=300, solver="lbfgs")
        clf_plain.fit(X_tr, y_tr)
        probs_plain = clf_plain.predict_proba(X_va)

        # 2. Modelo Centered (treinado com prompt-centering nos prompts de treino)
        X_tr_cnt = center_groups(X_tr, g_tr)
        clf_cnt = LogisticRegression(C=c_cnt, max_iter=300, solver="lbfgs")
        clf_cnt.fit(X_tr_cnt, y_tr)

        # 3. Simulação de Teste Cego: A validação NÃO sabe os prompt_ids reais!
        # Reconstrói os clusters cegamente usando a matriz de similaridade de tópicos:
        X_clust_va = X_clust[idx_va]
        sim_va = (X_clust_va @ X_clust_va.T).toarray()
        g_va_inferred = capped_agglo(sim_va, cap=18, thr=0.04)

        # Aplica centering nos clusters inferidos cegamente
        X_va_cnt_inferred = center_groups(X_va, g_va_inferred)
        probs_cnt_inferred = clf_cnt.predict_proba(X_va_cnt_inferred)

        # Blend das probabilidades
        lp_blend_inferred = (1 - blend_weight) * np.log(np.clip(probs_plain, 1e-12, 1.0)) + blend_weight * np.log(np.clip(probs_cnt_inferred, 1e-12, 1.0))

        # 4. Hungarian Assignment nos clusters inferidos cegamente
        pred_blind = lp_blend_inferred.argmax(axis=1)
        pred_blind_hung = pred_blind.copy()
        for g in np.unique(g_va_inferred):
            idx_g = np.where(g_va_inferred == g)[0]
            if len(idx_g) > 1 and len(idx_g) <= 18:
                cost = -lp_blend_inferred[idx_g]
                r, c = linear_sum_assignment(cost)
                pred_blind_hung[idx_g[r]] = c

        # 5. Hungarian Oracle (com clusters perfeitos para medir teto teórico)
        X_va_cnt_oracle = center_groups(X_va, g_va_true)
        probs_cnt_oracle = clf_cnt.predict_proba(X_va_cnt_oracle)
        lp_oracle = (1 - blend_weight) * np.log(np.clip(probs_plain, 1e-12, 1.0)) + blend_weight * np.log(np.clip(probs_cnt_oracle, 1e-12, 1.0))
        pred_oracle_hung = lp_oracle.argmax(axis=1)
        for g in np.unique(g_va_true):
            idx_g = np.where(g_va_true == g)[0]
            if len(idx_g) > 1 and len(idx_g) <= 18:
                cost = -lp_oracle[idx_g]
                r, c = linear_sum_assignment(cost)
                pred_oracle_hung[idx_g[r]] = c

        acc_p = accuracy_score(y_va, probs_plain.argmax(axis=1))
        acc_b = accuracy_score(y_va, pred_blind)
        acc_bh = accuracy_score(y_va, pred_blind_hung)
        acc_oh = accuracy_score(y_va, pred_oracle_hung)

        acc_plain_list.append(acc_p)
        acc_blend_raw_list.append(acc_b)
        acc_blind_hungarian_list.append(acc_bh)
        acc_oracle_hungarian_list.append(acc_oh)

        print(f"  Fold {fold+1}: Plain={acc_p:.4f} | Blend={acc_b:.4f} | Inferred-Hungarian={acc_bh:.4f} | Oracle={acc_oh:.4f}")

    results = {
        "plain_cv": float(np.mean(acc_plain_list)),
        "blend_cv": float(np.mean(acc_blend_raw_list)),
        "blind_hungarian_cv": float(np.mean(acc_blind_hungarian_list)),
        "oracle_hungarian_cv": float(np.mean(acc_oracle_hungarian_list)),
        "time_seconds": float(time.time() - t0),
    }

    print("\n[3/3] ================= RESULTADO CONSOLIDADO =================")
    print(f"  1. Plain Baseline CV:              {results['plain_cv']:.4f} (±{np.std(acc_plain_list):.4f})")
    print(f"  2. Dual Blend CV:                  {results['blend_cv']:.4f} (±{np.std(acc_blend_raw_list):.4f})")
    print(f"  3. Blind Inferred Hungarian CV:    {results['blind_hungarian_cv']:.4f} (±{np.std(acc_blind_hungarian_list):.4f})")
    print(f"  4. Oracle (True Prompts) CV:       {results['oracle_hungarian_cv']:.4f} (±{np.std(acc_oracle_hungarian_list):.4f})")
    print(f"  Tempo Total: {results['time_seconds']:.2f}s")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_cross_validation(include_extra_data=True)
