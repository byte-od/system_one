#!/usr/bin/env python3
"""
👑 SYSTEM ONE APEX V34.1 — DUAL-TITAN (157B MoS) + DUAL-SCALE BESSEL TWIN + WOODBURY SCACHE O(1)
================================================================================================
Validação rigorosa 5-Fold StratifiedGroupKFold (métrica oficial: Balanced Accuracy Score)
avaliada SIMULTANEAMENTE em TODOS os regimes de multiplicidade de prompt do teste oculto:
  m in {1 (singleton), 2 (arena pair), 3 (triple), 4, 6, 18 (full prompt block)}!

Lições da Auditoria V34.0 incorporadas para dominância estrita (`V34.1 >= V33` em TODOS os `m`):
1. `Zero Poluição Qwen-72B Legado` (`max_len=128`, 20.8% OOF removido) -> Dual-Titan puro
   (`Yi-34B L30` 7.168D + `Mistral Large 2 123B L44` 12.288D + `120D Forensic Scouts`).
2. `Preservação Exata da Escala SVM (T_svc = 0.85)`: evita o super-aguçamento `T=0.35` que
   abafava os Titãs em `m >= 4`, mantendo Cell 2 100% idêntica byte-a-byte!
3. `Dual-Scale Bessel Twin Heads` (`centered` puro `m=18` + `centered_arena` especializado `m in {2,3,4}`):
   - Em `m=2, 3, 4`: usa `centered_arena` (+5.2% sobre V33 em pares/trios!).
   - Em `m >= 7` (`m=18`): usa 100% `centered` puro (zero diluição em blocos completos!).
   - `Normalized Multiplicity Gate`: `gate(1) = 0.0`, transição suave em `m=2..6`, e `gate(18) = 1.000`!
4. `SCacheBayesianWoodburyHead` (`192D` PCA sobre `[Z_yi, Z_mis, F_120]`):
   - `m = 1`: Gaussiana Marginal exata `(Sigma_P + Sigma_E)^{-1}`.
   - `m = 2, 3`: Marginalização Analítica Injetiva EXATA via Woodbury (`18x18` e `18x18x18`).
   - `m >= 4`: Cavity Kalman EM idêntico à V33 (preserva 100% da performance em `m=18`!).
5. `UniversalS1ReflexiveMLP` Multi-Regime (`625D`) + `S1->S2 Intra-Clan Verifiers` + `Damped Sinkhorn`.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.sparse import hstack
from scipy.spatial.distance import squareform
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from forensic_scouts_120d import extract_forensic_compass1_120d

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)

CLANS = {
    "openai": [0, 1, 2, 3, 4],
    "gemini": [5, 6, 7, 8],
    "claude": [9, 10, 11, 12, 13],
    "glm": [15, 16],
}

SMALL_WORDS_C2 = set(
    "a an the and or but if while although because as so than that which who whom whose "
    "this these those it its they them their theirs he him his she her hers we us our ours "
    "you your yours i me my mine in on at to from with without for of by over under between "
    "through into about is are was were be been being have has had do does did can could "
    "will would shall should may might must not no nor all any some each every both either "
    "neither other another such only also just very too more most less least here there "
    "when where why how what whether then now still yet even much many few little".split()
)
SMALL_WORDS_EXT = SMALL_WORDS_C2 | set(
    "however moreover furthermore therefore thus instead rather overall finally first second".split()
)


@dataclass(frozen=True)
class ReflexDecision:
    action: np.ndarray
    confidence: np.ndarray
    uncertainty: np.ndarray
    margin: np.ndarray
    is_uncertain: np.ndarray
    latency_ms: float


def build_reflex_decision(P: np.ndarray, margin_thresh: float = 0.35, latency_ms: float = 0.0) -> ReflexDecision:
    sp = np.sort(P, axis=1)[:, ::-1]
    conf = sp[:, 0]
    margin = np.clip(sp[:, 0] - sp[:, 1], 0.0, 1.0)
    p_safe = np.clip(P, 1e-12, 1.0)
    unc = np.clip(-np.sum(p_safe * np.log(p_safe), axis=1) / np.log(P.shape[1]), 0.0, 1.0)
    return ReflexDecision(
        action=P.argmax(axis=1),
        confidence=conf,
        uncertainty=unc,
        margin=margin,
        is_uncertain=(margin < margin_thresh),
        latency_ms=latency_ms,
    )


# ==============================================================================
# 1. Preprocessors & Sparse Views (100% fiéis ao Notebook Cell 2 & Cell 4)
# ==============================================================================
def shape_text_cell2(t: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def small_words_cell2(t: str) -> str:
    return " ".join(
        w.lower() if w.lower() in SMALL_WORDS_C2 or not w.isalnum() else "CONTENT"
        for w in re.findall(r"\w+|[^\w\s]", t)
    )


def unicode_skeleton_text(t: str) -> str:
    t = re.sub(r"[A-Z][a-z]+", "Tt", t)
    t = re.sub(r"[A-Z]{2,}", "UU", t)
    t = re.sub(r"[a-z]+", "w", t)
    t = re.sub(r"\d+", "0", t)
    return t


def host_distortion_text(t: str) -> str:
    toks = re.findall(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|\d+|[^\w\s]|\n", t)
    out = []
    for tok in toks:
        if tok == "\n":
            out.append("NL")
        elif not tok[0].isalnum():
            out.append(f"P_{ord(tok[0])}")
        elif tok[0].isdigit():
            out.append("NUM")
        else:
            has_curly = "\u2019" in tok
            has_str = "'" in tok
            low = tok.lower().replace("\u2019", "'")
            if low in SMALL_WORDS_EXT or "'" in low:
                tag = low + ("_CURLY" if has_curly else ("_STR" if has_str else ""))
                if tok[0].isupper():
                    tag = "CAP_" + tag
                out.append(tag)
            else:
                L = len(tok)
                b = "S" if L <= 4 else ("M" if L <= 7 else "L")
                out.append(("C" if tok[0].isupper() else "c") + b)
    return " ".join(out)


def build_cell2_views_exact(texts: List[str], vecs: List[TfidfVectorizer] | None = None):
    views = [
        ("char", 3, 5, 250000, 1.0),
        ("word", 1, 3, 180000, 0.7),
        ("shape", 2, 6, 60000, 0.5),
        ("small", 1, 3, 50000, 0.5),
    ]
    mats, fitted = [], []
    for idx, (view, lo, hi, max_f, w) in enumerate(views):
        prep = shape_text_cell2 if view == "shape" else small_words_cell2 if view == "small" else (lambda s: s)
        proc = [prep(t) for t in texts]
        if vecs is None:
            analyzer = "char" if view in ("char", "shape") else "word"
            v = TfidfVectorizer(
                analyzer=analyzer, ngram_range=(lo, hi), max_features=max_f,
                sublinear_tf=True, lowercase=False, min_df=2,
            )
            m = v.fit_transform(proc) * w
            fitted.append(v)
        else:
            v = vecs[idx]
            m = v.transform(proc) * w
        mats.append(m)
    return hstack(mats, format="csr"), (fitted if vecs is None else vecs)


def build_c2_views_fresh(texts: List[str], vecs: List[TfidfVectorizer] | None = None):
    views = [
        ("char", 3, 5, 250000, 1.0),
        ("word", 1, 3, 180000, 0.7),
        ("shape", 2, 6, 60000, 0.5),
        ("small", 1, 3, 50000, 0.5),
    ]
    mats, fitted = [], []
    for idx, (view, lo, hi, max_f, w) in enumerate(views):
        prep = shape_text_cell2 if view == "shape" else small_words_cell2 if view == "small" else (lambda s: s)
        proc = [prep(t) for t in texts]
        if vecs is None:
            analyzer = "char" if view in ("char", "shape") else "word"
            v = TfidfVectorizer(analyzer=analyzer, ngram_range=(lo, hi), max_features=max_f, sublinear_tf=True)
            m = v.fit_transform(proc) * w
            fitted.append(v)
        else:
            v = vecs[idx]
            m = v.transform(proc) * w
        mats.append(m)
    return hstack(mats, format="csr"), (fitted if vecs is None else vecs)


def build_host_exact_views(texts: List[str], vecs: List[TfidfVectorizer] | None = None):
    specs = [
        ("skel_char", unicode_skeleton_text, "char", (1, 5), 65000, r"(?u)\b\w\w+\b", 1.0),
        ("distort_tok", host_distortion_text, "word", (1, 3), 90000, r"\S+", 0.9),
        ("char_wb", lambda s: s, "char_wb", (2, 5), 110000, r"(?u)\b\w\w+\b", 0.75),
        ("word_punct", lambda s: s, "word", (1, 2), 80000, r"\w+|[^\w\s]", 0.65),
    ]
    mats, fitted = [], []
    for i, (_, fn, analyzer, ngr, mf, pat, w) in enumerate(specs):
        proc = [fn(t) for t in texts]
        if vecs is None:
            kw = dict(analyzer=analyzer, ngram_range=ngr, max_features=mf, sublinear_tf=True, lowercase=False)
            if analyzer == "word":
                kw["token_pattern"] = pat
            v = TfidfVectorizer(**kw)
            m = v.fit_transform(proc) * w
            fitted.append(v)
        else:
            v = vecs[i]
            m = v.transform(proc) * w
        mats.append(m)
    return hstack(mats, format="csr"), (fitted if vecs is None else vecs)


# ==============================================================================
# 2. Clustering, Bessel Centering & Normalized Multiplicity Gate
# ==============================================================================
def compute_tfidf_sim_and_clusters(
    texts: List[str], base_thresh: float = 0.89, min_size: int = 2, max_size: int = 18
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(texts)
    if n < 2:
        return np.ones(n, dtype=np.int32), np.zeros(n, dtype=bool), np.ones(n, dtype=np.int32), np.eye(n, dtype=np.float32)

    Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(texts)
    S_mat = (Xt @ Xt.T).toarray().astype(np.float32)
    del Xt
    D = np.clip(1.0 - S_mat, 0.0, None)
    np.fill_diagonal(D, 0.0)

    Z = linkage(squareform(D, checks=False), method="average")
    g = fcluster(Z, base_thresh, criterion="distance")
    next_id = int(g.max()) + 1
    out = g.copy()

    for cid in np.unique(g):
        idx = np.where(g == cid)[0]
        if len(idx) > max_size:
            subD = D[np.ix_(idx, idx)]
            subZ = linkage(squareform(subD, checks=False), method="average")
            for th in np.linspace(base_thresh - 0.005, 0.05, 165):
                sg = fcluster(subZ, th, criterion="distance")
                if np.bincount(sg).max() <= max_size:
                    for sc in np.unique(sg):
                        out[idx[sg == sc]] = next_id
                        next_id += 1
                    break
            else:
                k_split = int(np.ceil(len(idx) / max_size))
                sg = fcluster(subZ, k_split, criterion="maxclust")
                for sc in np.unique(sg):
                    out[idx[sg == sc]] = next_id
                    next_id += 1

    sizes = np.bincount(out)[out].astype(np.int32)
    twin_rows = (sizes >= min_size) & (sizes <= max_size)
    return out.astype(np.int32), twin_rows, sizes, S_mat


def group_center_raw(X: np.ndarray, groups: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    Xc = np.zeros_like(X, dtype=np.float32)
    ugs = np.unique(groups) if mask is None else np.unique(groups[mask])
    for g in ugs:
        m = groups == g
        if m.sum() > 1:
            Xc[m] = X[m] - X[m].mean(axis=0, keepdims=True)
    return Xc


def bessel_group_center(X: np.ndarray, groups: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    Xc = np.zeros_like(X, dtype=np.float32)
    ugs = np.unique(groups) if mask is None else np.unique(groups[mask])
    for g in ugs:
        m = groups == g
        cnt = int(m.sum())
        if cnt > 1:
            bessel = float(np.sqrt(cnt / float(cnt - 1)))
            Xc[m] = (X[m] - X[m].mean(axis=0, keepdims=True)) * bessel
    return Xc


def compute_normalized_twin_gate(sizes: np.ndarray, kappa: float = 2.5) -> np.ndarray:
    """
    Gate normalizado em [0, 1] tal que:
      - `gate(1) = 0.0` (zero ruído em singletons)
      - `gate(m)` cresce suavemente em `m = 2, 3, 4, 6`
      - `gate(18) = 1.000` EXATO (preserva 100% da potência Twin em blocos completos!)
    """
    s = np.asarray(sizes, dtype=np.float64)
    gate = np.zeros_like(s, dtype=np.float64)
    active = s >= 2
    dm = s[active] - 1.0
    kp = max(float(kappa), 1e-4)
    norm_18 = (17.0 + kp) / 17.0
    gate[active] = np.minimum(1.0, norm_18 * (dm / (dm + kp)))
    return gate


def build_arena_small_m_training(
    X: np.ndarray, y: np.ndarray, prompts: np.ndarray, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Treina especificamente o especialista `centered_arena` para subgrupos pequenos `m in {2, 3, 4}`!
    Separado do `centered` de `m=18` para ZERO diluição em grupos grandes!
    """
    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    uniq_p = np.unique(prompts)
    for draw in range(2):
        for m_sub in (2, 3, 4):
            sub_idx_all, sub_grp_all = [], []
            gid = 1
            for p in uniq_p:
                perm = rng.permutation(np.where(prompts == p)[0])
                for s in range(0, len(perm) - m_sub + 1, m_sub):
                    chunk = perm[s : s + m_sub]
                    sub_idx_all.append(chunk)
                    sub_grp_all.append(np.full(m_sub, gid, dtype=np.int32))
                    gid += 1
            idx_cat = np.concatenate(sub_idx_all)
            grp_cat = np.concatenate(sub_grp_all)
            X_parts.append(bessel_group_center(X[idx_cat], grp_cat))
            y_parts.append(y[idx_cat])
    return np.vstack(X_parts).astype(np.float32), np.concatenate(y_parts)


class TestTimeSCache:
    """Memória associativa O(1) Scar-Cache (32x32 slots constantes)."""
    def __init__(self, slot_dim: int = 32, resonance_scale: float = 0.15):
        self.slot_dim = slot_dim
        self.resonance_scale = resonance_scale
        self.proj = None

    def apply_scache_resonance(self, X: np.ndarray, groups: np.ndarray, twin_mask: np.ndarray | None = None) -> np.ndarray:
        if self.proj is None:
            rng = np.random.RandomState(42)
            Q, _ = np.linalg.qr(rng.randn(X.shape[1], self.slot_dim))
            self.proj = Q.astype(np.float32)
        Z = X @ self.proj
        X_out = X.copy()
        ugs = np.unique(groups) if twin_mask is None else np.unique(groups[twin_mask])
        for g in ugs:
            m = groups == g
            cnt = int(m.sum())
            if cnt > 1:
                gate_m = min(1.0, (19.0 / 17.0) * (cnt - 1.0) / (cnt - 1.0 + 2.0))
                dz = Z[m] - Z[m].mean(axis=0, keepdims=True)
                W_scar = (dz.T @ dz) / float(cnt)
                X_out[m] += (self.resonance_scale * gate_m) * ((dz @ W_scar.T) @ self.proj.T)
        return X_out


# ==============================================================================
# 3. UniversalS1ReflexiveMLP (625D Multi-Regime)
# ==============================================================================
class UniversalS1ReflexiveMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 320, num_classes: int = 18):
        super().__init__()
        self.front_ln = nn.LayerNorm(in_dim)
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc_res = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(0.35)
        self.policy_head = nn.Linear(hidden_dim, num_classes)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.silu(self.ln1(self.fc_in(self.front_ln(x))))
        h_res = F.silu(self.ln2(self.fc_res(self.drop(h))))
        h = h + 0.5 * h_res
        logits = self.policy_head(self.drop(h))
        val = torch.sigmoid(self.value_head(h)).squeeze(-1)
        return logits, val


def build_s1_features(
    Z_pca: np.ndarray, F_120: np.ndarray, groups: np.ndarray, twin_mask: np.ndarray | None = None, kappa: float = 2.5
) -> np.ndarray:
    """`[Z_pca (192), gate*bessel(Z_pca) (192), F_120 (120), gate*bessel(F_120) (120), gate (1)]` = 625D."""
    _, inv = np.unique(groups, return_inverse=True)
    sizes = np.bincount(inv)[inv]
    if twin_mask is not None:
        sizes = np.where(twin_mask, sizes, 1)
    gate = compute_normalized_twin_gate(sizes, kappa=kappa).astype(np.float32)[:, None]
    zc = bessel_group_center(Z_pca, groups, twin_mask) * gate
    fc = bessel_group_center(F_120, groups, twin_mask) * gate
    return np.hstack([Z_pca, zc, F_120, fc, gate]).astype(np.float32)


def train_s1_reflex_mlp_multiregime(
    Z_pca_tr: np.ndarray,
    F_120_tr: np.ndarray,
    y_tr: np.ndarray,
    prompts_tr: np.ndarray,
    seed: int = 42,
    hidden_dim: int = 320,
    epochs: int = 95,
) -> Tuple[UniversalS1ReflexiveMLP, StandardScaler]:
    rng = np.random.default_rng(seed)
    X_m18 = build_s1_features(Z_pca_tr, F_120_tr, prompts_tr)
    # 2x peso em m=18 + 1x m=1 + 1x m=2 + 1x m=4
    X_views = [X_m18, X_m18]
    y_views = [y_tr, y_tr]

    singleton_grps = np.arange(len(y_tr), dtype=np.int32)
    X_views.append(build_s1_features(Z_pca_tr, F_120_tr, singleton_grps))
    y_views.append(y_tr)

    uniq_p = np.unique(prompts_tr)
    for m_sub in (2, 4):
        g_sim = np.arange(len(y_tr), dtype=np.int32)
        nxt = len(y_tr) + 1
        for p in uniq_p:
            p_idx = rng.permutation(np.where(prompts_tr == p)[0])
            for s in range(0, len(p_idx) - m_sub + 1, m_sub):
                g_sim[p_idx[s : s + m_sub]] = nxt
                nxt += 1
        X_views.append(build_s1_features(Z_pca_tr, F_120_tr, g_sim))
        y_views.append(y_tr)

    X_all = np.vstack(X_views)
    y_all = np.concatenate(y_views)
    scaler = StandardScaler().fit(X_all)
    Xs = np.clip(scaler.transform(X_all), -6.0, 6.0).astype(np.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = UniversalS1ReflexiveMLP(Xs.shape[1], hidden_dim=hidden_dim, num_classes=K)
    opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=3e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    xt = torch.from_numpy(Xs)
    yt = torch.from_numpy(y_all.astype(np.int64))
    model.train()
    bsz = 512
    for _ in range(epochs):
        perm = torch.randperm(len(xt))
        for s in range(0, len(xt), bsz):
            idx = perm[s : s + bsz]
            xb, yb = xt[idx], yt[idx]
            opt.zero_grad()
            logits, val = model(xb + 0.035 * torch.randn_like(xb))
            ce = F.cross_entropy(logits, yb, label_smoothing=0.06)
            with torch.no_grad():
                correct = (logits.argmax(dim=-1) == yb).float()
            loss = ce + 0.15 * F.mse_loss(val, correct)
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    return model, scaler


def predict_s1_reflex_mlp(model: UniversalS1ReflexiveMLP, scaler: StandardScaler, X: np.ndarray, temp: float = 1.5) -> np.ndarray:
    Xs = np.clip(scaler.transform(X), -6.0, 6.0).astype(np.float32)
    model.eval()
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(Xs))
        return F.softmax(logits / temp, dim=-1).numpy()


# ==============================================================================
# 4. SCache O(1) Bayesian Shared-Prompt + Exact Pair/Triple Woodbury Head (192D)
# ==============================================================================
class SCacheBayesianWoodburyHead:
    """
    Combina o subespaço 192D PCA comprovado da V33 (`[Z_yi, Z_mis, F_120]`) com:
      - `m = 1`: Gaussiana Marginal exata `(Sigma_P + Sigma_E)^{-1}`
      - `m = 2`: Marginalização Analítica Injetiva EXATA `18 x 18` via Woodbury
      - `m = 3`: Marginalização Analítica Injetiva EXATA `18 x 18 x 18` via Woodbury
      - `m >= 4`: Cavity Kalman Filter 3-iter idêntico à V33 (`temp = 2.2`)!
    """

    def __init__(self, n_components: int = 192):
        self.d = n_components
        self.sc_t2 = StandardScaler()
        self.sc_t3 = StandardScaler()
        self.sc_f = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        self.mu_c: np.ndarray | None = None
        self.cov_eps: np.ndarray | None = None
        self.cov_prompt: np.ndarray | None = None
        self.prec_eps: np.ndarray | None = None
        self.prec_prompt: np.ndarray | None = None

    def _project_fit(self, Z_t2: np.ndarray, Z_t3: np.ndarray, F_120: np.ndarray) -> np.ndarray:
        a = self.sc_t2.fit_transform(Z_t2) / np.sqrt(Z_t2.shape[1])
        b = self.sc_t3.fit_transform(Z_t3) / np.sqrt(Z_t3.shape[1])
        c = self.sc_f.fit_transform(F_120) / np.sqrt(F_120.shape[1]) * 0.85
        return self.pca.fit_transform(np.hstack([a, b, c]).astype(np.float32)).astype(np.float32)

    def project(self, Z_t2: np.ndarray, Z_t3: np.ndarray, F_120: np.ndarray) -> np.ndarray:
        a = self.sc_t2.transform(Z_t2) / np.sqrt(Z_t2.shape[1])
        b = self.sc_t3.transform(Z_t3) / np.sqrt(Z_t3.shape[1])
        c = self.sc_f.transform(F_120) / np.sqrt(F_120.shape[1]) * 0.85
        return self.pca.transform(np.hstack([a, b, c]).astype(np.float32)).astype(np.float32)

    def fit(
        self, Z_t2: np.ndarray, Z_t3: np.ndarray, F_120: np.ndarray, y: np.ndarray, prompts: np.ndarray
    ) -> "SCacheBayesianWoodburyHead":
        Z = self._project_fit(Z_t2, Z_t3, F_120)
        self.mu_c = np.zeros((K, self.d), dtype=np.float64)
        for k in range(K):
            self.mu_c[k] = Z[y == k].mean(axis=0)

        R = Z.astype(np.float64) - self.mu_c[y]
        within_res, group_means, group_sizes = [], [], []
        for g in np.unique(prompts):
            m = prompts == g
            cnt = int(m.sum())
            if cnt >= 2:
                rg = R[m]
                gbar = rg.mean(axis=0, keepdims=True)
                within_res.append((rg - gbar) * np.sqrt(cnt / float(cnt - 1)))
                group_means.append(gbar[0])
                group_sizes.append(cnt)

        cov_eps = LedoitWolf().fit(np.vstack(within_res)).covariance_
        cov_gbar = LedoitWolf().fit(np.vstack(group_means)).covariance_
        cov_prompt_raw = cov_gbar - (cov_eps / float(np.mean(group_sizes)))
        eigvals, eigvecs = np.linalg.eigh(cov_prompt_raw)
        eigvals = np.maximum(eigvals, 1e-4 * np.trace(cov_eps) / self.d)
        cov_prompt = (eigvecs * eigvals) @ eigvecs.T

        reg = 1e-4 * np.eye(self.d)
        self.cov_eps = cov_eps + reg
        self.cov_prompt = cov_prompt + reg
        self.prec_eps = np.linalg.inv(self.cov_eps)
        self.prec_prompt = np.linalg.inv(self.cov_prompt)
        return self

    def predict_log_proba(
        self,
        Z_t2: np.ndarray,
        Z_t3: np.ndarray,
        F_120: np.ndarray,
        groups: np.ndarray,
        prior_probs: np.ndarray | None = None,
        temp: float = 2.2,
        inj_strength: float = 0.85,
    ) -> np.ndarray:
        Z = self.project(Z_t2, Z_t3, F_120).astype(np.float64)
        N = len(Z)
        mu = self.mu_c

        prec0 = np.linalg.inv(self.cov_prompt + self.cov_eps)
        diff0 = Z[:, None, :] - mu[None, :, :]
        ll0 = -0.5 * np.einsum("nkd,de,nke->nk", diff0, prec0, diff0) / temp
        u_all = -0.5 * np.einsum("nkd,de,nke->nk", diff0, self.prec_eps, diff0) / temp

        prior_log = (
            np.log(np.clip(prior_probs, 1e-9, 1.0)) if prior_probs is not None else np.zeros((N, K), dtype=np.float64)
        )
        prior_log = prior_log - logsumexp(prior_log, axis=1, keepdims=True)

        woodbury_L: Dict[int, np.ndarray] = {}
        wiener_ops: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        def get_woodbury_L(m: int) -> np.ndarray:
            if m not in woodbury_L:
                post_cov = np.linalg.inv(self.prec_prompt + float(m) * self.prec_eps)
                W_m = self.prec_eps @ post_cov @ self.prec_eps
                ev, eu = np.linalg.eigh(W_m)
                ev = np.maximum(ev, 0.0)
                woodbury_L[m] = eu * np.sqrt(ev / temp)
            return woodbury_L[m]

        def get_wiener(m_other: int) -> Tuple[np.ndarray, np.ndarray]:
            if m_other not in wiener_ops:
                post_cov = np.linalg.inv(self.prec_prompt + float(m_other) * self.prec_eps)
                wiener_ops[m_other] = (post_cov @ self.prec_eps, np.linalg.inv(self.cov_eps + post_cov))
            return wiener_ops[m_other]

        ll_out = ll0.copy()
        same_pen = float(np.log(max(1.0 - inj_strength, 1e-4)))
        large_groups = []

        for g in np.unique(groups):
            idx = np.where(groups == g)[0]
            m = len(idx)
            if m <= 1:
                ll_out[idx] = ll0[idx]
            elif m == 2:
                i0, i1 = idx[0], idx[1]
                L2 = get_woodbury_L(2)
                v0 = diff0[i0] @ L2
                v1 = diff0[i1] @ L2
                s0 = u_all[i0] + 0.5 * np.sum(v0 * v0, axis=1)
                s1 = u_all[i1] + 0.5 * np.sum(v1 * v1, axis=1)
                coup = v0 @ v1.T
                J0 = s0[:, None] + (s1 + 0.75 * prior_log[i1])[None, :] + coup
                np.fill_diagonal(J0, np.diag(J0) + same_pen)
                J1 = (s0 + 0.75 * prior_log[i0])[:, None] + s1[None, :] + coup
                np.fill_diagonal(J1, np.diag(J1) + same_pen)
                ll_out[i0] = logsumexp(J0, axis=1)
                ll_out[i1] = logsumexp(J1, axis=0)
            elif m == 3:
                i0, i1, i2 = idx[0], idx[1], idx[2]
                L3 = get_woodbury_L(3)
                v0 = diff0[i0] @ L3
                v1 = diff0[i1] @ L3
                v2 = diff0[i2] @ L3
                s0 = u_all[i0] + 0.5 * np.sum(v0 * v0, axis=1)
                s1 = u_all[i1] + 0.5 * np.sum(v1 * v1, axis=1)
                s2 = u_all[i2] + 0.5 * np.sum(v2 * v2, axis=1)
                c01 = v0 @ v1.T
                c02 = v0 @ v2.T
                c12 = v1 @ v2.T
                eye_pen = np.eye(K, dtype=np.float64) * same_pen
                base_3 = (
                    c01[:, :, None] + c02[:, None, :] + c12[None, :, :]
                    + eye_pen[:, :, None] + eye_pen[:, None, :] + eye_pen[None, :, :]
                )
                t0 = base_3 + s0[:, None, None] + (s1 + 0.75 * prior_log[i1])[None, :, None] + (s2 + 0.75 * prior_log[i2])[None, None, :]
                t1 = base_3 + (s0 + 0.75 * prior_log[i0])[:, None, None] + s1[None, :, None] + (s2 + 0.75 * prior_log[i2])[None, None, :]
                t2 = base_3 + (s0 + 0.75 * prior_log[i0])[:, None, None] + (s1 + 0.75 * prior_log[i1])[None, :, None] + s2[None, None, :]
                ll_out[i0] = logsumexp(t0, axis=(1, 2))
                ll_out[i1] = logsumexp(t1, axis=(0, 2))
                ll_out[i2] = logsumexp(t2, axis=(0, 1))
            else:
                large_groups.append(idx)

        if large_groups:
            Q = softmax(ll0, axis=1) if prior_probs is None else (0.65 * prior_probs + 0.35 * softmax(ll0, axis=1))
            for _ in range(3):
                res = Z - (Q @ mu)
                for idx in large_groups:
                    W_eta, prec_m = get_wiener(len(idx) - 1)
                    cav_sum = res[idx].sum(axis=0, keepdims=True) - res[idx]
                    z_clean = Z[idx] - (cav_sum @ W_eta.T)
                    diff = z_clean[:, None, :] - mu[None, :, :]
                    ll_out[idx] = -0.5 * np.einsum("mkd,de,mke->mk", diff, prec_m, diff) / temp
                P_like = softmax(ll_out, axis=1)
                Q = P_like if prior_probs is None else (0.50 * prior_probs + 0.50 * P_like)

        return ll_out - logsumexp(ll_out, axis=1, keepdims=True)


# ==============================================================================
# 5. Cavity Repulsion, Intra-Clan Verifiers & Damped Soft Sinkhorn
# ==============================================================================
def apply_similarity_weighted_cavity_repulsion(
    log_P: np.ndarray, groups: np.ndarray, S_mat: np.ndarray, gamma: float = 0.70, sim_floor: float = 0.12, sim_span: float = 0.30, iters: int = 5
) -> np.ndarray:
    if gamma <= 1e-6 or len(log_P) < 2:
        return softmax(log_P, axis=1)
    P0_log = log_P - logsumexp(log_P, axis=1, keepdims=True)
    Q = np.exp(P0_log)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        m = len(idx)
        if 2 <= m <= 18:
            sub_S = S_mat[np.ix_(idx, idx)]
            W = np.clip((sub_S - sim_floor) / max(sim_span, 1e-4), 0.0, 1.0)
            np.fill_diagonal(W, 0.0)
            if W.max() > 1e-6:
                W_eff = W / np.maximum(W.sum(axis=1, keepdims=True) / min(float(m - 1), 6.0), 1.0)
                lp_g = P0_log[idx]
                q_g = Q[idx].copy()
                for _ in range(iters):
                    excl = np.log(np.clip(1.0 - gamma * q_g, 1e-4, 1.0))
                    q_g = 0.65 * softmax(lp_g + (W_eff @ excl), axis=1) + 0.35 * q_g
                Q[idx] = q_g
    return Q / Q.sum(axis=1, keepdims=True)


def sinkhorn_damped(P: np.ndarray, target_mix: float = 0.55, iters: int = 300) -> np.ndarray:
    if target_mix <= 1e-6 or len(P) < 50:
        return P / P.sum(axis=1, keepdims=True)
    P_safe = np.clip(P, 1e-12, 1.0)
    P_safe = P_safe / P_safe.sum(axis=1, keepdims=True)
    size_scale = min(1.0, len(P_safe) / 360.0)
    eff_mix = target_mix * size_scale
    emp = P_safe.sum(axis=0)
    unif = np.full(K, len(P_safe) / float(K))
    target = np.maximum((1.0 - eff_mix) * emp + eff_mix * unif, 1e-6)
    c = np.ones(K, dtype=np.float64)
    for _ in range(iters):
        R = P_safe * c
        R /= R.sum(axis=1, keepdims=True)
        c *= target / np.maximum(R.sum(axis=0), 1e-12)
    R = P_safe * c
    return R / R.sum(axis=1, keepdims=True)


class IntraClanVerifierSuite:
    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        self.clfs: Dict[str, LogisticRegression] = {}

    def fit(self, X_feat: np.ndarray, y: np.ndarray) -> "IntraClanVerifierSuite":
        for clan_name, members in CLANS.items():
            mask = np.isin(y, members)
            Xc = X_feat[mask]
            yc_raw = y[mask]
            re_map = {m: i for i, m in enumerate(members)}
            yc = np.array([re_map[val] for val in yc_raw], dtype=np.int64)
            sc = StandardScaler().fit(Xc)
            clf = LogisticRegression(C=0.06, max_iter=800, random_state=42).fit(sc.transform(Xc), yc)
            self.scalers[clan_name] = sc
            self.clfs[clan_name] = clf
        return self

    def refine_probabilities(
        self, P_s1: np.ndarray, X_feat: np.ndarray, margin_tau: float = 0.35, beta_max: float = 0.25, temp: float = 2.0
    ) -> np.ndarray:
        if beta_max <= 1e-6:
            return P_s1
        dec = build_reflex_decision(P_s1, margin_thresh=margin_tau)
        P_out = P_s1.copy()
        unc_idx = np.where(dec.is_uncertain)[0]
        if len(unc_idx) == 0:
            return P_out
        beta = beta_max * np.clip(1.0 - dec.margin[unc_idx] / max(margin_tau, 1e-4), 0.0, 1.0)
        X_u = X_feat[unc_idx]
        for clan_name, members in CLANS.items():
            clan_mass = P_out[np.ix_(unc_idx, members)].sum(axis=1, keepdims=True)
            active = clan_mass[:, 0] > 0.25
            if not active.any():
                continue
            sub_u = unc_idx[active]
            b_a = beta[active][:, None]
            cm_a = clan_mass[active]
            Xs = self.scalers[clan_name].transform(X_u[active])
            logits_s2 = (Xs @ self.clfs[clan_name].coef_.T + self.clfs[clan_name].intercept_) / temp
            if logits_s2.shape[1] == 1:
                p1 = 1.0 / (1.0 + np.exp(-logits_s2[:, 0]))
                p_s2 = np.stack([1.0 - p1, p1], axis=1)
            else:
                p_s2 = softmax(logits_s2, axis=1)
            p_s1_cond = np.clip(P_out[np.ix_(sub_u, members)] / np.maximum(cm_a, 1e-12), 1e-9, 1.0)
            fused_cond = softmax((1.0 - b_a) * np.log(p_s1_cond) + b_a * np.log(np.clip(p_s2, 1e-9, 1.0)), axis=1)
            P_out[np.ix_(sub_u, members)] = cm_a * fused_cond
        return P_out / P_out.sum(axis=1, keepdims=True)


def build_regime_subsamples(
    folds: List[Tuple[np.ndarray, np.ndarray]],
    prompts_train: np.ndarray,
    regimes: Sequence[int] = (1, 2, 3, 4, 6, 18),
    draws_per_small_m: int = 3,
    seed: int = 2026,
) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    out: Dict[int, List[Tuple[int, np.ndarray]]] = {m: [] for m in regimes}
    for m in regimes:
        n_draws = 1 if m == 18 else draws_per_small_m
        for _ in range(n_draws):
            for fold_i, (_, va_idx) in enumerate(folds):
                if m == 18:
                    out[m].append((fold_i, va_idx.copy()))
                else:
                    va_p = prompts_train[va_idx]
                    picked = [rng.choice(va_idx[va_p == pid], size=min(m, int((va_p == pid).sum())), replace=False) for pid in np.unique(va_p)]
                    out[m].append((fold_i, np.concatenate(picked)))
    return out


def main():
    t_start = time.time()
    print("=" * 94)
    print("👑 SYSTEM ONE APEX V34.1 — DUAL-TITAN + DUAL-SCALE TWIN + WOODBURY SCACHE O(1)")
    print("=" * 94)

    data_dir = THIS_DIR / "data"
    t2_dir = THIS_DIR / "titans" / "titan2_yi34b_l30"
    t3_dir = THIS_DIR / "titans" / "titan3_mistral_large_123b_l44"
    scratch_root = Path("/Users/juliocesarreisfilho/.gemini/antigravity/brain/2b3990be-2635-4b7d-98c1-2d81f1cbd3e0/scratch")
    ext_dir = scratch_root / "ext_data"
    svc_cache_path = scratch_root / "v34_1_svc_oof_cache.npz"

    df_train = pd.read_csv(data_dir / "train.csv")
    df_train["text"] = df_train["text"].fillna("").astype(str)
    y_train = df_train["model"].map(L2I).values
    prompts_train = df_train["prompt_id"].astype(str).values
    tr_texts = df_train["text"].tolist()
    n_train = len(df_train)

    Z_yi = np.load(t2_dir / "features_yi34b_l30_train_full.npy").astype(np.float32)
    Z_mis = np.load(t3_dir / "features_mistral_large_123b_l44_train_full.npy").astype(np.float32)
    F_120 = extract_forensic_compass1_120d(tr_texts)
    print(f"✓ Host Train: {n_train} rows | T2 Yi34B={Z_yi.shape} | T3 Mistral123B={Z_mis.shape} | F120={F_120.shape}")

    sgkf = StratifiedGroupKFold(n_splits=5)
    folds = list(sgkf.split(df_train, y_train, groups=prompts_train))
    regimes = [1, 2, 3, 4, 6, 18]
    regime_splits = build_regime_subsamples(folds, prompts_train, regimes=regimes, draws_per_small_m=3, seed=2026)

    # 1. Carrega ou computa e faz cache das margens 5-Fold exatas dos 3 LinearSVCs!
    if svc_cache_path.exists():
        print(f"\n[1/5] Carregando margens OOF LinearSVC cacheadas de {svc_cache_path.name}...")
        npz = np.load(svc_cache_path)
        oof_m_c2 = npz["m_c2"]
        oof_m_he = npz["m_he"]
        oof_m_px = npz["m_px"]
    else:
        print("\n[1/5] Computando margens 5-Fold Leak-Free dos 3 LinearSVCs (100% fiéis às Células 2 e 4)...")
        pv4 = pd.read_parquet(ext_dir / "pool_v4.parquet")
        pv4 = pv4[pv4["model"].isin(L2I)].copy()
        pv4["text"] = pv4["text"].fillna("").astype(str)
        pv4_y = pv4["model"].map(L2I).values

        pa = pd.read_csv(ext_dir / "passages.csv")
        pa = pa[
            (pa["passes_funnel"] == True)
            & (pa["is_refusal"] == False)
            & (pa["is_empty"] == False)
            & (pa["split"] != "pilot_range")
            & (~((pa["split"] == "new") & (pa["model"] == "kimi-k3")))
            & (pa["model"].isin(L2I))
        ].copy()
        pa["text"] = pa["text"].fillna("").astype(str)
        pa["y"] = pa["model"].map(L2I).values
        pa["src_pid"] = pa["source_prompt_id"].fillna("").astype(str)

        X_pv4_c2, vecs_c2 = build_cell2_views_exact(pv4["text"].tolist(), vecs=None)
        X_tr_c2, _ = build_cell2_views_exact(tr_texts, vecs=vecs_c2)

        oof_m_c2 = np.zeros((n_train, K), dtype=np.float32)
        oof_m_he = np.zeros((n_train, K), dtype=np.float32)
        oof_m_px = np.zeros((n_train, K), dtype=np.float32)

        for fold_i, (tr_idx, va_idx) in enumerate(folds):
            ts0 = time.time()
            val_pids = set(prompts_train[va_idx])
            tr_texts_f = [tr_texts[i] for i in tr_idx]
            va_texts_f = [tr_texts[i] for i in va_idx]
            y_tr_f = y_train[tr_idx]

            X_c2_fit = hstack([X_pv4_c2.T, X_tr_c2[tr_idx].T], format="csr").T
            y_c2_fit = np.concatenate([pv4_y, y_tr_f])
            svc_c2 = LinearSVC(C=0.5, class_weight="balanced", max_iter=1500, random_state=42).fit(X_c2_fit, y_c2_fit)
            oof_m_c2[va_idx] = svc_c2.decision_function(X_tr_c2[va_idx]).astype(np.float32)

            X_he_tr, vecs_he = build_host_exact_views(tr_texts_f, vecs=None)
            X_he_va, _ = build_host_exact_views(va_texts_f, vecs=vecs_he)
            svc_he = LinearSVC(C=0.25, max_iter=2000, random_state=42).fit(X_he_tr, y_tr_f)
            oof_m_he[va_idx] = svc_he.decision_function(X_he_va).astype(np.float32)

            pa_lf = pa[~pa["src_pid"].isin(val_pids)]
            px_texts = tr_texts_f + pa_lf["text"].tolist()
            px_y = np.concatenate([y_tr_f, pa_lf["y"].values])
            px_w = np.concatenate([np.full(len(tr_idx), 2.5, dtype=np.float32), np.ones(len(pa_lf), dtype=np.float32)])
            X_px1_tr, vp1 = build_c2_views_fresh(px_texts, vecs=None)
            X_px1_va, _ = build_c2_views_fresh(va_texts_f, vecs=vp1)
            X_px2_tr, vp2 = build_host_exact_views(px_texts, vecs=None)
            X_px2_va, _ = build_host_exact_views(va_texts_f, vecs=vp2)
            svc_px = LinearSVC(C=0.35, max_iter=2000, random_state=42).fit(
                hstack([X_px1_tr, X_px2_tr * 0.7], format="csr"), px_y, sample_weight=px_w
            )
            oof_m_px[va_idx] = svc_px.decision_function(hstack([X_px1_va, X_px2_va * 0.7], format="csr")).astype(np.float32)
            print(f"  ✓ Sparse SVCs Fold {fold_i + 1}/5 ready in {time.time() - ts0:.1f}s")

        np.savez_compressed(svc_cache_path, m_c2=oof_m_c2, m_he=oof_m_he, m_px=oof_m_px)
        print(f"  ✓ Cached sparse SVC OOF margins to {svc_cache_path}")

    # 2. Treina módulos System One (Titans Dual-Scale + Forensic 120D + Woodbury SCache + S1 MLP + S2) em ~25s!
    print("\n[2/5] Treinando módulos System One 5-Fold Leak-Free (Dual-Titan + Dual-Scale Twin + Woodbury SCache)...")
    fold_models = []
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        tf0 = time.time()
        y_tr_f = y_train[tr_idx]
        p_tr_f = prompts_train[tr_idx]

        sc120 = StandardScaler().fit(F_120[tr_idx])
        f120_tr = np.clip(sc120.transform(F_120[tr_idx]), -6.0, 6.0).astype(np.float32)

        titan_heads = {}
        for t_key, Z_all in [("t2", Z_yi), ("t3", Z_mis)]:
            Xtr_t = TestTimeSCache(32, 0.15).apply_scache_resonance(np.hstack([Z_all[tr_idx], f120_tr * 0.5]), p_tr_f)
            sp_t = StandardScaler().fit(Xtr_t)
            cp_t = LogisticRegression(C=0.08, max_iter=700, random_state=42).fit(sp_t.transform(Xtr_t), y_tr_f)

            # 1) Full-Group Bessel Twin Head (`m = 18` puro — zero diluição!)
            Xtr_c18 = bessel_group_center(Xtr_t, p_tr_f)
            sc_c18 = StandardScaler().fit(Xtr_c18)
            cc_c18 = LogisticRegression(C=0.08, max_iter=700, random_state=42).fit(sc_c18.transform(Xtr_c18), y_tr_f)

            # 2) Arena Small-Group Bessel Twin Head (`m in {2, 3, 4}` puro!)
            Xtr_carena, y_carena = build_arena_small_m_training(Xtr_t, y_tr_f, p_tr_f, seed=200 + fold_i)
            sc_carena = StandardScaler().fit(Xtr_carena)
            cc_carena = LogisticRegression(C=0.05, max_iter=700, random_state=42).fit(sc_carena.transform(Xtr_carena), y_carena)

            # 3) Legado V33 sem Bessel para benchmark lado-a-lado exato
            Xtr_v33 = group_center_raw(Xtr_t, p_tr_f)
            sc_v33 = StandardScaler().fit(Xtr_v33)
            cc_v33 = LogisticRegression(C=0.08, max_iter=700, random_state=42).fit(sc_v33.transform(Xtr_v33), y_tr_f)

            titan_heads[t_key] = (sp_t, cp_t, sc_c18, cc_c18, sc_carena, cc_carena, sc_v33, cc_v33)

        # Forensic 120D Scouts (2x m=18 + 1x m=1 para preservar m=18 e calibrar m=1!)
        f120_tr_dual = np.hstack([F_120[tr_idx], bessel_group_center(F_120[tr_idx], p_tr_f)])
        f120_tr_solo = np.hstack([F_120[tr_idx], np.zeros_like(F_120[tr_idx])])
        f120_tr_aug = np.vstack([f120_tr_dual, f120_tr_dual, f120_tr_solo])
        y_f120_aug = np.concatenate([y_tr_f, y_tr_f, y_tr_f])
        sc_fdual = StandardScaler().fit(f120_tr_aug)
        clf_flr = LogisticRegression(C=0.15, max_iter=1000, random_state=42).fit(
            np.clip(sc_fdual.transform(f120_tr_aug), -6, 6), y_f120_aug
        )
        hgb_f120 = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=5.0, random_state=42
        ).fit(f120_tr_aug, y_f120_aug)

        # SCache Bayesian Woodbury 192D (Dual-Titan limpo!)
        scb = SCacheBayesianWoodburyHead(n_components=192).fit(
            Z_yi[tr_idx], Z_mis[tr_idx], F_120[tr_idx], y_tr_f, p_tr_f
        )
        Z_pca_tr = scb.project(Z_yi[tr_idx], Z_mis[tr_idx], F_120[tr_idx])

        # UniversalS1ReflexiveMLP (625D Multi-Regime) + S2 Verifiers
        s1_mlp, s1_sc = train_s1_reflex_mlp_multiregime(Z_pca_tr, F_120[tr_idx], y_tr_f, p_tr_f, seed=42 + fold_i)
        X_s1_tr = build_s1_features(Z_pca_tr, F_120[tr_idx], p_tr_f, kappa=2.5)
        s2_ver = IntraClanVerifierSuite().fit(X_s1_tr, y_tr_f)

        # Também treina S1 legado V33 (624D m=18-only) para comparação fiel V33 vs V34.1!
        f120_v33_tr = np.hstack([F_120[tr_idx], group_center_raw(F_120[tr_idx], p_tr_f)])
        X_s1_v33_tr = np.hstack([Z_pca_tr, group_center_raw(Z_pca_tr, p_tr_f), f120_v33_tr])
        s1_v33_sc = StandardScaler().fit(X_s1_v33_tr)
        torch.manual_seed(42 + fold_i)
        s1_v33_mlp = UniversalS1ReflexiveMLP(X_s1_v33_tr.shape[1], hidden_dim=320, num_classes=K)
        opt_v33 = torch.optim.AdamW(s1_v33_mlp.parameters(), lr=1.8e-3, weight_decay=3e-2)
        xt_v33 = torch.from_numpy(np.clip(s1_v33_sc.transform(X_s1_v33_tr), -6, 6).astype(np.float32))
        yt_v33 = torch.from_numpy(y_tr_f.astype(np.int64))
        s1_v33_mlp.train()
        for _ in range(95):
            opt_v33.zero_grad()
            lg_v, vl_v = s1_v33_mlp(xt_v33 + 0.04 * torch.randn_like(xt_v33))
            loss_v = F.cross_entropy(lg_v, yt_v33, label_smoothing=0.06) + 0.15 * F.mse_loss(vl_v, (lg_v.argmax(-1) == yt_v33).float())
            loss_v.backward()
            opt_v33.step()
        s1_v33_mlp.eval()

        fold_models.append({
            "va_idx": va_idx,
            "sc120": sc120,
            "titan_heads": titan_heads,
            "sc_fdual": sc_fdual,
            "clf_flr": clf_flr,
            "hgb_f120": hgb_f120,
            "scb": scb,
            "s1_mlp": s1_mlp,
            "s1_sc": s1_sc,
            "s1_v33_mlp": s1_v33_mlp,
            "s1_v33_sc": s1_v33_sc,
            "s2_ver": s2_ver,
        })
        print(f"  ✓ Fold {fold_i + 1}/5 System One modules trained in {time.time() - tf0:.1f}s")

    # 3. Avalia TODOS os regimes m in {1, 2, 3, 4, 6, 18}
    print("\n[3/5] Avaliando canais em TODOS os regimes de multiplicidade (`m = 1, 2, 3, 4, 6, 18`)...")

    def evaluate_subset(fold_i: int, sub_idx: np.ndarray) -> Dict[str, object]:
        fm = fold_models[fold_i]
        sub_texts = [tr_texts[int(r)] for r in sub_idx]
        groups, twin_rows, sizes, S_mat = compute_tfidf_sim_and_clusters(sub_texts, base_thresh=0.89, min_size=2, max_size=18)
        eff_sizes = np.where(twin_rows, sizes, 1)

        m_c2 = oof_m_c2[sub_idx]
        m_he = oof_m_he[sub_idx]
        m_px = oof_m_px[sub_idx]
        m_he_raw_c = group_center_raw(m_he, groups, twin_rows)
        m_px_raw_c = group_center_raw(m_px, groups, twin_rows)
        m_he_bc = bessel_group_center(m_he, groups, twin_rows)
        m_px_bc = bessel_group_center(m_px, groups, twin_rows)

        f120_raw = F_120[sub_idx]
        f120_half = np.clip(fm["sc120"].transform(f120_raw), -6.0, 6.0).astype(np.float32) * 0.5

        titans_out = {}
        for t_key, Z_all in [("t2", Z_yi), ("t3", Z_mis)]:
            sp_t, cp_t, sc_c18, cc_c18, sc_carena, cc_carena, sc_v33, cc_v33 = fm["titan_heads"][t_key]
            X_sc = TestTimeSCache(32, 0.15).apply_scache_resonance(np.hstack([Z_all[sub_idx], f120_half]), groups, twin_rows)
            lg_p = (sp_t.transform(X_sc) @ cp_t.coef_.T + cp_t.intercept_).astype(np.float32)

            X_bc = bessel_group_center(X_sc, groups, twin_rows)
            lg_c18 = (sc_c18.transform(X_bc) @ cc_c18.coef_.T + cc_c18.intercept_).astype(np.float32)
            lg_carena = (sc_carena.transform(X_bc) @ cc_carena.coef_.T + cc_carena.intercept_).astype(np.float32)

            X_v33 = group_center_raw(X_sc, groups, twin_rows)
            lg_v33 = (sc_v33.transform(X_v33) @ cc_v33.coef_.T + cc_v33.intercept_).astype(np.float32)
            titans_out[t_key] = (lg_p, lg_c18, lg_carena, lg_v33)

        gate_f = compute_normalized_twin_gate(eff_sizes, kappa=2.5).astype(np.float32)[:, None]
        f120_dual = np.hstack([f120_raw, bessel_group_center(f120_raw, groups, twin_rows) * gate_f])
        lg_f120 = (np.clip(fm["sc_fdual"].transform(f120_dual), -6, 6) @ fm["clf_flr"].coef_.T + fm["clf_flr"].intercept_).astype(np.float32)
        pr_hgb = fm["hgb_f120"].predict_proba(f120_dual).astype(np.float32)

        Z_pca = fm["scb"].project(Z_yi[sub_idx], Z_mis[sub_idx], f120_raw)
        X_s1 = build_s1_features(Z_pca, f120_raw, groups, twin_rows, kappa=2.5)
        pr_s1 = predict_s1_reflex_mlp(fm["s1_mlp"], fm["s1_sc"], X_s1, temp=1.5)

        X_s1_v33 = np.hstack([
            Z_pca, group_center_raw(Z_pca, groups, twin_rows),
            f120_raw, group_center_raw(f120_raw, groups, twin_rows),
        ])
        pr_s1_v33 = predict_s1_reflex_mlp(fm["s1_v33_mlp"], fm["s1_v33_sc"], X_s1_v33, temp=1.5)

        p_prior = softmax(
            1.0 * np.log(np.clip(softmax(m_c2 / 0.85, axis=1), 1e-9, 1.0))
            + 0.6 * np.log(np.clip(softmax(m_px / 0.85, axis=1), 1e-9, 1.0))
            + 0.3 * np.log(np.clip(pr_s1, 1e-9, 1.0)),
            axis=1,
        )
        logp_scb = fm["scb"].predict_log_proba(
            Z_yi[sub_idx], Z_mis[sub_idx], f120_raw, groups, prior_probs=p_prior, temp=2.2
        ).astype(np.float32)

        return {
            "fold_i": fold_i,
            "y": y_train[sub_idx],
            "groups": groups,
            "twin_rows": twin_rows,
            "sizes": eff_sizes,
            "S_mat": S_mat,
            "m_c2": m_c2,
            "m_he": m_he, "m_he_raw_c": m_he_raw_c, "m_he_bc": m_he_bc,
            "m_px": m_px, "m_px_raw_c": m_px_raw_c, "m_px_bc": m_px_bc,
            "lg_f120": lg_f120, "pr_hgb": pr_hgb,
            "titans": titans_out,
            "pr_s1": pr_s1, "pr_s1_v33": pr_s1_v33,
            "logp_scb": logp_scb,
            "X_s1": X_s1,
        }

    cached_evals: Dict[int, List[Dict[str, object]]] = {
        m: [evaluate_subset(fi, s_idx) for fi, s_idx in regime_splits[m]] for m in regimes
    }

    # 4. Construção de canais V34.1 (Preserva T_svc = 0.85 e Dual-Scale Twin Blend!)
    ch_names = [
        "c2_svc", "host_svc", "proxy_svc", "forensic120",
        "t2_plain", "t2_twin", "t3_plain", "t3_twin",
        "s1_mlp", "scache_bayes",
    ]
    T_SVC = 0.85
    T_TITAN = 4.2

    def build_v34_channels(ev: Dict[str, object], kappa: float = 2.5) -> np.ndarray:
        sz = ev["sizes"].astype(np.float64)
        gate = compute_normalized_twin_gate(sz, kappa=kappa)[:, None]  # (n, 1): 0 em m=1, 1.0 em m=18!
        alpha_large = np.clip((sz - 2.0) / 5.0, 0.0, 1.0)[:, None]    # (n, 1): 0 em m=2, 1.0 em m>=7!
        # Fator inverso de Bessel `sqrt(17/18)` mantém amplitude de margem idêntica à V33 em m=18!
        b_norm = np.sqrt(17.0 / 18.0)

        p_c2 = np.log(np.clip(softmax(ev["m_c2"] / T_SVC, axis=1), 1e-9, 1.0))
        p_he = np.log(np.clip(softmax((ev["m_he"] + 0.45 * gate * (ev["m_he_bc"] * b_norm)) / T_SVC, axis=1), 1e-9, 1.0))
        p_px = np.log(np.clip(softmax((ev["m_px"] + 0.55 * gate * (ev["m_px_bc"] * b_norm)) / T_SVC, axis=1), 1e-9, 1.0))
        p_f120 = np.log(np.clip(0.45 * softmax(ev["lg_f120"] / 2.5, axis=1) + 0.55 * ev["pr_hgb"], 1e-9, 1.0))

        t2_p, t2_c18, t2_carena, _ = ev["titans"]["t2"]
        p_t2p = np.log(np.clip(softmax(t2_p / T_TITAN, axis=1), 1e-9, 1.0))
        pr_t2_twin = (1.0 - alpha_large) * softmax(t2_carena / T_TITAN, axis=1) + alpha_large * softmax(t2_c18 / T_TITAN, axis=1)
        p_t2c = gate * np.log(np.clip(pr_t2_twin, 1e-9, 1.0))

        t3_p, t3_c18, t3_carena, _ = ev["titans"]["t3"]
        p_t3p = np.log(np.clip(softmax(t3_p / T_TITAN, axis=1), 1e-9, 1.0))
        pr_t3_twin = (1.0 - alpha_large) * softmax(t3_carena / T_TITAN, axis=1) + alpha_large * softmax(t3_c18 / T_TITAN, axis=1)
        p_t3c = gate * np.log(np.clip(pr_t3_twin, 1e-9, 1.0))

        p_s1 = np.log(np.clip(ev["pr_s1"], 1e-9, 1.0))
        p_scb = np.clip(ev["logp_scb"], -20.0, 0.0)
        return np.stack([p_c2, p_he, p_px, p_f120, p_t2p, p_t2c, p_t3p, p_t3c, p_s1, p_scb], axis=0)

    print("\n[4/5] Otimizando pesos Nested-OOF Multi-Regime (`m = 1, 2, 3, 4, 6, 18` com forte âncora em m>=6)...")
    best_kappa = 2.5
    ch_cache = {m: [build_v34_channels(ev, kappa=best_kappa) for ev in cached_evals[m]] for m in regimes}

    # Pesos de objetivo equilibrando grupos grandes (m=18, m=6) e pequenos (m=1,2,3,4)!
    regime_loss_weights = {1: 0.12, 2: 0.16, 3: 0.16, 4: 0.14, 6: 0.14, 18: 0.28}
    w_prior = np.array([0.95, 0.22, 0.55, 0.16, 0.16, 0.35, 0.26, 1.25, 0.16, 0.26])
    bounds_v34 = [
        (0.60, 1.35),  # c2_svc
        (0.10, 0.65),  # host_svc
        (0.25, 0.85),  # proxy_svc
        (0.05, 0.45),  # forensic120
        (0.14, 0.55),  # t2_plain (nunca silenciado!)
        (0.15, 0.90),  # t2_twin
        (0.22, 0.70),  # t3_plain (nunca silenciado!)
        (0.65, 1.75),  # t3_twin (preserva autoridade total em m=18!)
        (0.08, 0.45),  # s1_mlp
        (0.15, 0.60),  # scache_bayes
    ]

    fold_weights = []
    for holdout_fold in range(5):
        def loss_nested(w: np.ndarray) -> float:
            tot = 0.015 * float(np.sum((w - w_prior) ** 2))
            for m, rw in regime_loss_weights.items():
                tr_evs = [(ev, ch) for ev, ch in zip(cached_evals[m], ch_cache[m]) if ev["fold_i"] != holdout_fold]
                m_loss = sum(
                    log_loss(ev["y"], np.clip(softmax(np.tensordot(w, ch, axes=(0, 0)), axis=1), 1e-9, 1.0), labels=list(range(K)))
                    for ev, ch in tr_evs
                ) / len(tr_evs)
                tot += rw * m_loss
            return tot

        res = minimize(loss_nested, w_prior, method="L-BFGS-B", bounds=bounds_v34)
        fold_weights.append(res.x)

    mean_w = np.mean(fold_weights, axis=0)

    # Pesos exatos da V33 na submissão em produção (Public LB 0.585):
    w_v33_prod = {
        "c2_svc": 1.0441, "host_svc": 0.1584, "proxy_svc": 0.5403, "forensic120": 0.1583,
        "t2_plain": 0.0188, "t2_twin": 0.2712, "t3_plain": 0.0342, "t3_twin": 1.4179,
        "s1_mlp": 0.0985, "scache_bayes": 0.2383,
    }
    print("\n  Pesos médios Nested-OOF Multi-Regime aprendidos (V34.1 vs V33 Prod):")
    for nm, val in zip(ch_names, mean_w):
        print(f"    {nm:<15}: {val:.4f}  (V33 Prod: {w_v33_prod[nm]:.4f})")

    best_gamma, best_s2_beta, best_sink_mix = 0.70, 0.20, 0.55
    best_post_score = -1.0
    for gm in [0.55, 0.70, 0.82]:
        for bt in [0.0, 0.15, 0.25]:
            for sm in [0.50, 0.60]:
                reg_scores = []
                for m, rw in regime_loss_weights.items():
                    preds_m, ys_m = [], []
                    for ev, ch in zip(cached_evals[m], ch_cache[m]):
                        lp = np.tensordot(fold_weights[ev["fold_i"]], ch, axes=(0, 0))
                        P_cur = apply_similarity_weighted_cavity_repulsion(lp, ev["groups"], ev["S_mat"], gamma=gm)
                        P_cur = fold_models[ev["fold_i"]]["s2_ver"].refine_probabilities(P_cur, ev["X_s1"], beta_max=bt)
                        P_cur = sinkhorn_damped(P_cur, target_mix=sm, iters=300)
                        preds_m.append(P_cur.argmax(axis=1))
                        ys_m.append(ev["y"])
                    reg_scores.append(rw * balanced_accuracy_score(np.concatenate(ys_m), np.concatenate(preds_m)))
                sc_tot = float(np.sum(reg_scores))
                if sc_tot > best_post_score:
                    best_post_score = sc_tot
                    best_gamma, best_s2_beta, best_sink_mix = gm, bt, sm

    print(f"\n  ✓ Optimal Multi-Regime Post-Processing: cavity_gamma={best_gamma:.2f}, s2_beta={best_s2_beta:.2f}, sinkhorn_mix={best_sink_mix:.2f}")

    # Tabela Comparativa Estrita: V33 Prod vs V34.1 Dual-Titan Apex em TODOS os 6 regimes!
    print("\n" + "=" * 96)
    print("📊 PROVA EMPÍRICA MULTI-REGIME FINAL: V33 (0.585 LB) vs V34.1 DUAL-TITAN DUAL-SCALE APEX")
    print("=" * 96)
    print(f"  {'Regime':<15} | {'Titan3 V33':>11} | {'Titan3 V34.1':>12} | {'SCache V34.1':>12} | {'V33 Prod OOF':>13} | {'V34.1 Apex OOF':>14} | {'Gain V34.1-V33':>14}")
    print("-" * 96)

    summary_regimes = {}
    for m in regimes:
        y_all_m = []
        p_t3_33, p_t3_34, p_scb_34, p_e33, p_e34 = [], [], [], [], []

        for ev, ch in zip(cached_evals[m], ch_cache[m]):
            fi = ev["fold_i"]
            y_all_m.append(ev["y"])
            tw = ev["twin_rows"]

            # Titan 3 isolado V33 vs V34.1
            lp_t3_v33 = w_v33_prod["t3_plain"] * np.log(np.clip(softmax(ev["titans"]["t3"][0] / 5.0, axis=1), 1e-9, 1.0))
            if tw.any():
                lp_t3_v33[tw] += w_v33_prod["t3_twin"] * np.log(np.clip(softmax(ev["titans"]["t3"][3][tw] / 5.0, axis=1), 1e-9, 1.0))
            p_t3_33.append(lp_t3_v33.argmax(axis=1))

            lp_t3_v34 = fold_weights[fi][6] * ch[6] + fold_weights[fi][7] * ch[7]
            p_t3_34.append(lp_t3_v34.argmax(axis=1))
            p_scb_34.append(ev["logp_scb"].argmax(axis=1))

            # V33 Prod Ensemble fiel (exatamente como rodou no notebook V33!)
            lp_v33 = (
                w_v33_prod["c2_svc"] * np.log(np.clip(softmax(ev["m_c2"] / 0.85, axis=1), 1e-9, 1.0))
                + w_v33_prod["host_svc"] * np.log(np.clip(softmax((ev["m_he"] + 0.45 * ev["m_he_raw_c"]) / 0.85, axis=1), 1e-9, 1.0))
                + w_v33_prod["proxy_svc"] * np.log(np.clip(softmax((ev["m_px"] + 0.55 * ev["m_px_raw_c"]) / 0.85, axis=1), 1e-9, 1.0))
                + w_v33_prod["forensic120"] * ch[3]
                + w_v33_prod["t2_plain"] * np.log(np.clip(softmax(ev["titans"]["t2"][0] / 5.0, axis=1), 1e-9, 1.0))
                + w_v33_prod["t3_plain"] * np.log(np.clip(softmax(ev["titans"]["t3"][0] / 5.0, axis=1), 1e-9, 1.0))
                + w_v33_prod["s1_mlp"] * np.log(np.clip(ev["pr_s1_v33"], 1e-9, 1.0))
                + w_v33_prod["scache_bayes"] * ch[9]
            )
            if tw.any():
                lp_v33[tw] += (
                    w_v33_prod["t2_twin"] * np.log(np.clip(softmax(ev["titans"]["t2"][3][tw] / 5.0, axis=1), 1e-9, 1.0))
                    + w_v33_prod["t3_twin"] * np.log(np.clip(softmax(ev["titans"]["t3"][3][tw] / 5.0, axis=1), 1e-9, 1.0))
                )
            P_v33 = apply_similarity_weighted_cavity_repulsion(lp_v33, ev["groups"], ev["S_mat"], gamma=0.82)
            if len(P_v33) >= 50:
                # Na V33 sinkhorn_damped nao tinha size_scale!
                P_s = np.clip(P_v33, 1e-12, 1.0)
                P_s /= P_s.sum(1, keepdims=True)
                tgt = np.maximum(0.35 * P_s.sum(0) + 0.65 * np.full(K, len(P_s) / float(K)), 1e-6)
                c_sk = np.ones(K, dtype=np.float64)
                for _ in range(300):
                    R_sk = P_s * c_sk
                    R_sk /= R_sk.sum(1, keepdims=True)
                    c_sk *= tgt / np.maximum(R_sk.sum(0), 1e-12)
                P_v33 = (P_s * c_sk) / (P_s * c_sk).sum(1, keepdims=True)
            p_e33.append(P_v33.argmax(axis=1))

            # V34.1 Apex Ensemble
            lp_v34 = np.tensordot(fold_weights[fi], ch, axes=(0, 0))
            P_v34 = apply_similarity_weighted_cavity_repulsion(lp_v34, ev["groups"], ev["S_mat"], gamma=best_gamma)
            P_v34 = fold_models[fi]["s2_ver"].refine_probabilities(P_v34, ev["X_s1"], beta_max=best_s2_beta)
            P_v34 = sinkhorn_damped(P_v34, target_mix=best_sink_mix, iters=300)
            p_e34.append(P_v34.argmax(axis=1))

        yt = np.concatenate(y_all_m)
        s_t33 = balanced_accuracy_score(yt, np.concatenate(p_t3_33))
        s_t34 = balanced_accuracy_score(yt, np.concatenate(p_t3_34))
        s_scb = balanced_accuracy_score(yt, np.concatenate(p_scb_34))
        s_33 = balanced_accuracy_score(yt, np.concatenate(p_e33))
        s_34 = balanced_accuracy_score(yt, np.concatenate(p_e34))
        gain = s_34 - s_33
        summary_regimes[str(m)] = {
            "titan3_v33": float(s_t33),
            "titan3_v34_1": float(s_t34),
            "scache_v34_1": float(s_scb),
            "v33_prod_oof": float(s_33),
            "v34_1_apex_oof": float(s_34),
            "gain_vs_v33": float(gain),
        }
        lbl = f"m = {m:<2} ({'full' if m==18 else 'solo' if m==1 else 'arena'})"
        print(
            f"  {lbl:<15} | {s_t33*100:10.2f}% | {s_t34*100:11.2f}% | {s_scb*100:11.2f}% | "
            f"{s_33*100:12.2f}% | {s_34*100:13.2f}% | {gain*100:+13.2f}%"
        )
    print("=" * 96)

    # 5. Treinamento Final 100% `train.csv` e Exportação `.safetensors` V34.1
    print("\n[5/5] Treinando Heads Finais V34.1 em 100% de `train.csv` e exportando `.safetensors`...")
    tensors_out: Dict[str, torch.Tensor] = {}

    def put_np(key: str, arr: np.ndarray, dtype=torch.float32):
        tensors_out[key] = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)

    sc120_full = StandardScaler().fit(F_120)
    f120_full = np.clip(sc120_full.transform(F_120), -6.0, 6.0).astype(np.float32)
    put_np("f120_scaler.mean", sc120_full.mean_)
    put_np("f120_scaler.scale", sc120_full.scale_)

    for t_key, Z_all in [("t2", Z_yi), ("t3", Z_mis)]:
        X_t_full = TestTimeSCache(32, 0.15).apply_scache_resonance(np.hstack([Z_all, f120_full * 0.5]), prompts_train)
        sp_t = StandardScaler().fit(X_t_full)
        cp_t = LogisticRegression(C=0.08, max_iter=800, random_state=42).fit(sp_t.transform(X_t_full), y_train)

        X_c18 = bessel_group_center(X_t_full, prompts_train)
        sc_c18 = StandardScaler().fit(X_c18)
        cc_c18 = LogisticRegression(C=0.08, max_iter=800, random_state=42).fit(sc_c18.transform(X_c18), y_train)

        X_carena, y_carena = build_arena_small_m_training(X_t_full, y_train, prompts_train, seed=777)
        sc_carena = StandardScaler().fit(X_carena)
        cc_carena = LogisticRegression(C=0.05, max_iter=800, random_state=42).fit(sc_carena.transform(X_carena), y_carena)

        put_np(f"{t_key}.scaler_plain.mean", sp_t.mean_)
        put_np(f"{t_key}.scaler_plain.scale", sp_t.scale_)
        put_np(f"{t_key}.plain.weight", cp_t.coef_)
        put_np(f"{t_key}.plain.bias", cp_t.intercept_)

        put_np(f"{t_key}.scaler_centered.mean", sc_c18.mean_)
        put_np(f"{t_key}.scaler_centered.scale", sc_c18.scale_)
        put_np(f"{t_key}.centered.weight", cc_c18.coef_)
        put_np(f"{t_key}.centered.bias", cc_c18.intercept_)

        put_np(f"{t_key}.scaler_centered_arena.mean", sc_carena.mean_)
        put_np(f"{t_key}.scaler_centered_arena.scale", sc_carena.scale_)
        put_np(f"{t_key}.centered_arena.weight", cc_carena.coef_)
        put_np(f"{t_key}.centered_arena.bias", cc_carena.intercept_)

    f120_dual_full = np.hstack([F_120, bessel_group_center(F_120, prompts_train)])
    f120_solo_full = np.hstack([F_120, np.zeros_like(F_120)])
    f120_aug_full = np.vstack([f120_dual_full, f120_dual_full, f120_solo_full])
    y_aug_full = np.concatenate([y_train, y_train, y_train])
    sc_fd_full = StandardScaler().fit(f120_aug_full)
    clf_flr_full = LogisticRegression(C=0.15, max_iter=1000, random_state=42).fit(
        np.clip(sc_fd_full.transform(f120_aug_full), -6, 6), y_aug_full
    )
    put_np("f120_dual.mean", sc_fd_full.mean_)
    put_np("f120_dual.scale", sc_fd_full.scale_)
    put_np("f120_lr.weight", clf_flr_full.coef_)
    put_np("f120_lr.bias", clf_flr_full.intercept_)

    scb_full = SCacheBayesianWoodburyHead(n_components=192).fit(Z_yi, Z_mis, F_120, y_train, prompts_train)
    put_np("scb.sc_t2.mean", scb_full.sc_t2.mean_)
    put_np("scb.sc_t2.scale", scb_full.sc_t2.scale_)
    put_np("scb.sc_t3.mean", scb_full.sc_t3.mean_)
    put_np("scb.sc_t3.scale", scb_full.sc_t3.scale_)
    put_np("scb.sc_f.mean", scb_full.sc_f.mean_)
    put_np("scb.sc_f.scale", scb_full.sc_f.scale_)
    put_np("scb.pca.mean", scb_full.pca.mean_)
    put_np("scb.pca.components", scb_full.pca.components_)
    put_np("scb.mu_c", scb_full.mu_c)
    put_np("scb.cov_eps", scb_full.cov_eps)
    put_np("scb.cov_prompt", scb_full.cov_prompt)
    put_np("scb.prec_eps", scb_full.prec_eps)
    put_np("scb.prec_prompt", scb_full.prec_prompt)

    Z_pca_full = scb_full.project(Z_yi, Z_mis, F_120)
    for s_idx, seed_val in enumerate([42, 137, 2026]):
        mlp_m, sc_m = train_s1_reflex_mlp_multiregime(Z_pca_full, F_120, y_train, prompts_train, seed=seed_val, hidden_dim=320, epochs=95)
        if s_idx == 0:
            put_np("s1.scaler.mean", sc_m.mean_)
            put_np("s1.scaler.scale", sc_m.scale_)
        put_np(f"s1.mlp{s_idx}.scaler.mean", sc_m.mean_)
        put_np(f"s1.mlp{s_idx}.scaler.scale", sc_m.scale_)
        for k_param, v_param in mlp_m.state_dict().items():
            tensors_out[f"s1.mlp{s_idx}.{k_param}"] = v_param.detach().cpu().float()

    X_s1_full = build_s1_features(Z_pca_full, F_120, prompts_train, kappa=2.5)
    s2_full = IntraClanVerifierSuite().fit(X_s1_full, y_train)
    for clan_name in CLANS:
        put_np(f"s2.{clan_name}.mean", s2_full.scalers[clan_name].mean_)
        put_np(f"s2.{clan_name}.scale", s2_full.scalers[clan_name].scale_)
        put_np(f"s2.{clan_name}.weight", s2_full.clfs[clan_name].coef_)
        put_np(f"s2.{clan_name}.bias", s2_full.clfs[clan_name].intercept_)

    upload_dir = THIS_DIR / "scratch" / "titan3_head_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)
    out_st = upload_dir / "head_milestone1_system_one_apex.safetensors"
    save_file(tensors_out, str(out_st))

    meta_apex = {
        "schema": "system-one-apex-v34-dual-scale-woodbury",
        "labels": LABELS,
        "learned_weights": {nm: float(val) for nm, val in zip(ch_names, mean_w)},
        "temperatures": {"svc": T_SVC, "titan": T_TITAN, "forensic_lr": 2.5, "s1_mlp": 1.5, "scache_bayes": 2.2},
        "best_twin_kappa": float(best_kappa),
        "best_cavity_gamma": float(best_gamma),
        "best_s2_beta": float(best_s2_beta),
        "best_sinkhorn_target_mix": float(best_sink_mix),
        "cluster_base_thresh": 0.89,
        "multi_regime_oof_summary": summary_regimes,
        "oof_v33_apex_bal_acc": float(summary_regimes["18"]["v34_1_apex_oof"]),
        "safetensors_bytes": int(out_st.stat().st_size),
    }
    meta_path = upload_dir / "meta_milestone1_system_one_apex.json"
    meta_path.write_text(json.dumps(meta_apex, indent=2))
    save_file(tensors_out, str(t3_dir / "head_milestone1_system_one_apex.safetensors"))
    (t3_dir / "meta_milestone1_system_one_apex.json").write_text(json.dumps(meta_apex, indent=2))

    print(f"✓ Saved V34.1 Safetensors: {out_st} ({out_st.stat().st_size / (1024*1024):.2f} MB)")
    print(f"✓ Saved V34.1 Metadata:    {meta_path}")
    print(f"✓ Total V34.1 Validation & Export Time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
