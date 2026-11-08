"""Benchmark e Validação Completa: Hierarchical Mixture of Scars (MoS) + Tri-Compass + SCache.

Valida a arquitetura completa no Stratified Group K-Fold (5 folds) sobre a base expandida
(train.csv + passages_clean.csv = 7.769 amostras reais) com teste cego e Hungarian Assignment.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score
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

def get_family(m: str) -> str:
    if m.startswith("claude"): return "claude"
    if m.startswith("gemini"): return "gemini"
    if m.startswith("glm"): return "glm"
    if m.startswith("gpt"): return "gpt"
    if m.startswith("kimi"): return "kimi"
    if m.startswith("qwen"): return "qwen"
    return "other"

FAMILIES = sorted(list(set(get_family(l) for l in LABELS)))
FAM2I = {f: i for i, f in enumerate(FAMILIES)}
I2FAM = {i: f for i, f in enumerate(FAMILIES)}


def extract_tricompass_style_features(texts: List[str]) -> np.ndarray:
    """Extrai 28 características estilométricas, tipográficas e dinâmicas inter-sentenças (Tri-Compass Local)."""
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


def capped_agglo(S: np.ndarray, cap: int = 18, thr: float = 0.04) -> np.ndarray:
    """Reconstrói clusters de prompt limitados a 18 membros (Lance-Williams average linkage)."""
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
    """Tri-Compass Delta Origin: Subtrai o centróide do prompt de cada grupo (X - mu_group) em 6 ms via np.add.at."""
    _, inv = np.unique(g, return_inverse=True)
    counts = np.bincount(inv)
    sums = np.zeros((len(counts), X.shape[1]), dtype=np.float32)
    np.add.at(sums, inv, X)
    means = sums / np.maximum(counts[:, None], 1)
    return X - means[inv]


def evaluate_hierarchical_mos_arena():
    t0 = time.time()
    print("=" * 80)
    print("🚀 ARENA LOCAL: HIERARCHICAL MIXTURE OF SCARS (MoS) + TRI-COMPASS")
    print("=" * 80)

    train_path = "/tmp/kaggle_name_that_llm/train.csv"
    extra_path = "/tmp/prvsiyan_dataset/passages_clean.csv"

    df_official = pd.read_csv(train_path)[["prompt_id", "model", "text"]]
    if os.path.exists(extra_path):
        df_extra = pd.read_csv(extra_path)[["prompt_id", "model", "text"]]
        df_extra["prompt_id"] = "ex_" + df_extra["prompt_id"].astype(str)
        df_full = pd.concat([df_official, df_extra], ignore_index=True)
        print(f"✓ Base Expandida: {len(df_official)} oficiais + {len(df_extra)} extras = {len(df_full)} amostras.")
    else:
        df_full = df_official
        print(f"✓ Base Oficial: {len(df_full)} amostras.")

    y = df_full["model"].map(L2I).values
    y_fam = np.array([FAM2I[get_family(m)] for m in df_full["model"]])
    groups = df_full["prompt_id"].values
    texts = df_full["text"].tolist()

    print("\n[1/3] Extraindo N-Grams e Assinaturas Estilométricas Tri-Compass...")
    v_word = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=30000, sublinear_tf=True)

    X_w = v_word.fit_transform(texts)
    X_c = v_char.fit_transform(texts)
    style_raw = extract_tricompass_style_features(texts)
    scaler = StandardScaler()
    style_sc = scaler.fit_transform(style_raw)
    X_features = hstack([X_w, X_c, style_sc]).tocsr().astype(np.float32)
    print(f"  ✓ Matriz de Features: {X_features.shape[0]} amostras x {X_features.shape[1]} dimensões.")

    # Clustering de tópicos
    v_clust = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.08)
    X_clust = v_clust.fit_transform(texts)

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    acc_plain_list = []
    acc_hier_list = []
    acc_blind_hungarian_list = []
    acc_oracle_hungarian_list = []

    print("\n[2/3] Executando 5 Folds com Prompts Ocultos na Validação...")

    for fold, (idx_tr, idx_va) in enumerate(sgkf.split(X_features, y, groups=groups)):
        t_fold = time.time()
        X_tr = X_features[idx_tr]
        y_tr = y[idx_tr]
        y_fam_tr = y_fam[idx_tr]
        g_tr = groups[idx_tr]

        X_va = X_features[idx_va]
        y_va = y[idx_va]
        y_fam_va = y_fam[idx_va]
        g_va_true = groups[idx_va]

        # 1. Base Plain 18 Classes
        clf_plain = LogisticRegression(C=2.5, max_iter=250, solver="lbfgs")
        clf_plain.fit(X_tr, y_tr)
        probs_plain = clf_plain.predict_proba(X_va)

        # 2. Hierarchical MoS (Tier 1: Famílias)
        clf_fam = LogisticRegression(C=2.5, max_iter=250, solver="lbfgs")
        clf_fam.fit(X_tr, y_fam_tr)
        probs_fam = clf_fam.predict_proba(X_va)  # [N_va, 6]

        # Hierarchical MoS (Tier 2: Especialistas de Cicatriz por Família)
        probs_hier = np.zeros((len(idx_va), 18), dtype=np.float32)
        for f_idx, f_name in enumerate(FAMILIES):
            models_in_fam = [l for l in LABELS if get_family(l) == f_name]
            indices_in_fam = [L2I[l] for l in models_in_fam]

            if len(models_in_fam) == 1:
                probs_hier[:, indices_in_fam[0]] = probs_fam[:, f_idx]
            else:
                mask_tr = (y_fam_tr == f_idx)
                y_sub_tr = np.array([models_in_fam.index(df_full["model"].iloc[idx_tr[i]]) for i in np.where(mask_tr)[0]])
                clf_spec = LogisticRegression(C=3.5, max_iter=250, solver="lbfgs")
                clf_spec.fit(X_tr[mask_tr], y_sub_tr)
                p_spec = clf_spec.predict_proba(X_va)
                for sub_i, glob_i in enumerate(indices_in_fam):
                    probs_hier[:, glob_i] = probs_fam[:, f_idx] * p_spec[:, sub_i]

        # 3. Prompt-Centering no Treino (Tri-Compass Delta Origin)
        X_tr_cnt = center_groups(X_tr.toarray(), g_tr)
        clf_cnt = LogisticRegression(C=4.0, max_iter=250, solver="lbfgs")
        clf_cnt.fit(X_tr_cnt, y_tr)

        # 4. Teste Cego: Recuperação de Clusters de Prompt via capped_agglo
        X_clust_va = X_clust[idx_va]
        sim_va = (X_clust_va @ X_clust_va.T).toarray()
        g_va_inferred = capped_agglo(sim_va, cap=18, thr=0.04)

        # Centering nos clusters inferidos
        X_va_cnt_inferred = center_groups(X_va.toarray(), g_va_inferred)
        probs_cnt_inferred = clf_cnt.predict_proba(X_va_cnt_inferred)

        # Blend de Probabilidades: MoS Hierárquico + Centering Tri-Compass
        # 50% Plain/Hierárquico + 50% Centered
        p_base_blend = 0.5 * probs_plain + 0.5 * probs_hier
        lp_final = 0.5 * np.log(np.clip(p_base_blend, 1e-12, 1.0)) + 0.5 * np.log(np.clip(probs_cnt_inferred, 1e-12, 1.0))

        # 5. Hungarian Assignment nos clusters inferidos
        pred_blind_hung = lp_final.argmax(axis=1).copy()
        for g in np.unique(g_va_inferred):
            idx_g = np.where(g_va_inferred == g)[0]
            if len(idx_g) > 1 and len(idx_g) <= 18:
                cost = -lp_final[idx_g]
                r, c = linear_sum_assignment(cost)
                pred_blind_hung[idx_g[r]] = c

        # 6. Oracle Hungarian (com clusters perfeitos para aferir perda)
        X_va_cnt_oracle = center_groups(X_va.toarray(), g_va_true)
        probs_cnt_oracle = clf_cnt.predict_proba(X_va_cnt_oracle)
        lp_oracle = 0.5 * np.log(np.clip(p_base_blend, 1e-12, 1.0)) + 0.5 * np.log(np.clip(probs_cnt_oracle, 1e-12, 1.0))
        pred_oracle_hung = lp_oracle.argmax(axis=1).copy()
        for g in np.unique(g_va_true):
            idx_g = np.where(g_va_true == g)[0]
            if len(idx_g) > 1 and len(idx_g) <= 18:
                cost = -lp_oracle[idx_g]
                r, c = linear_sum_assignment(cost)
                pred_oracle_hung[idx_g[r]] = c

        acc_p = accuracy_score(y_va, probs_plain.argmax(axis=1))
        acc_h = accuracy_score(y_va, probs_hier.argmax(axis=1))
        acc_bh = accuracy_score(y_va, pred_blind_hung)
        acc_oh = accuracy_score(y_va, pred_oracle_hung)

        acc_plain_list.append(acc_p)
        acc_hier_list.append(acc_h)
        acc_blind_hungarian_list.append(acc_bh)
        acc_oracle_hungarian_list.append(acc_oh)

        dur = time.time() - t_fold
        print(f"  Fold {fold+1} ({dur:.1f}s): Plain={acc_p:.4f} | MoS-Hier={acc_h:.4f} | Inferred-Hungarian={acc_bh:.4f} | Oracle={acc_oh:.4f}")

    results = {
        "plain_cv": float(np.mean(acc_plain_list)),
        "hier_mos_cv": float(np.mean(acc_hier_list)),
        "blind_hungarian_cv": float(np.mean(acc_blind_hungarian_list)),
        "oracle_hungarian_cv": float(np.mean(acc_oracle_hungarian_list)),
        "total_time": float(time.time() - t0),
    }

    print("\n[3/3] ================= RESULTADO CONSOLIDADO =================")
    print(f"  1. Plain Baseline CV:              {results['plain_cv']:.4f} (±{np.std(acc_plain_list):.4f})")
    print(f"  2. Hierarchical MoS CV:            {results['hier_mos_cv']:.4f} (±{np.std(acc_hier_list):.4f})")
    print(f"  3. Blind Inferred Hungarian CV:    {results['blind_hungarian_cv']:.4f} (±{np.std(acc_blind_hungarian_list):.4f})")
    print(f"  4. Oracle (True Prompts) CV:       {results['oracle_hungarian_cv']:.4f} (±{np.std(acc_oracle_hungarian_list):.4f})")
    print(f"  Tempo Total: {results['total_time']:.2f}s")
    print("=" * 80)

    return results


if __name__ == "__main__":
    evaluate_hierarchical_mos_arena()
