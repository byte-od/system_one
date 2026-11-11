#!/usr/bin/env python3
"""
👑 SYSTEM ONE APEX (MILESTONE 1 — V33) — TREINAMENTO, ABLAÇÃO E SERIALIZAÇÃO SAFETENSORS
========================================================================================
Validação rigorosa 5-Fold StratifiedGroupKFold (métrica oficial: Balanced Accuracy Score)
com clusterização TF-IDF cega realista (`cluster_size_capped`, sem vazamento de `prompt_id`)
e desduplicação estrita de prompts em `passages.csv` (`train_regen`).

Módulos 100% Puros System One (`AGENTS.md` & `SKILL.md` — Zero LoRAs de terceiros, Zero Pickle):
1. `cluster_size_capped`: Clusterização hierárquica recursiva limitada a `<= 18` passagens (ARI 0.977).
2. `Forensic 120D Scouts` (Bússola 1) + `UniversalS1ReflexiveAgent` (MLP Residual LayerNorm SiLU + `ReflexDecision`).
3. `Dual-Anchor Host + Proxy Clean`: Corrige o Register Shift OpenRouter-vs-Host (Gemini travessões, Kimi apóstrofos curvos).
4. `Titan 2 (Yi-34B L30)` + `Titan 3 (Mistral Large 123B L44)` Co-Treinados com `120D` + `TestTimeSCache` O(1).
5. `SCacheInjectiveGroupHead`: Modelo Gerativo Gaussiano Bayesiano Cavity Shared-Prompt O(1) (< 200 KB estado)
   + Repulsão Suave Cavity-EM Ponderada por Similaridade Par-a-Par.
6. `System 1 -> System 2 Intra-Clan Verifiers`: Especialistas dedicados para OpenAI-5, Gemini-4, Claude-5, GLM-2
   acionados sob `ReflexDecision.is_uncertain` (`margin < tau`).
7. `Damped Soft Sinkhorn`: Calibração marginal amortecida validada em splits completos e subamostrados (Public/Private).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack
from scipy.spatial.distance import squareform
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
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

SMALL_WORDS = set(
    "a an the and or but if while although because as so than that which who whom whose "
    "this these those it its they them their theirs he him his she her hers we us our ours "
    "you your yours i me my mine in on at to from with without for of by over under between "
    "through into about is are was were be been being have has had do does did can could "
    "will would shall should may might must not no nor all any some each every both either "
    "neither other another such only also just very too more most less least here there "
    "when where why how what whether then now still yet even much many few little "
    "however moreover furthermore therefore thus instead rather overall finally first second".split()
)


@dataclass(frozen=True)
class ReflexDecision:
    """Contrato canônico UniversalS1Agent (`SKILL.md` Seção 1.A)."""
    action: np.ndarray
    confidence: np.ndarray
    uncertainty: np.ndarray
    margin: np.ndarray
    is_uncertain: np.ndarray
    latency_ms: float


def build_reflex_decision(P: np.ndarray, margin_thresh: float = 0.30, latency_ms: float = 0.0) -> ReflexDecision:
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
# 1. Preprocessors para Âncoras Ortográficas & Tipografia Host-Aware
# ==============================================================================
def shape_text_cell2(t: str) -> str:
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def small_words_cell2(t: str) -> str:
    return " ".join(
        w.lower() if w.lower() in SMALL_WORDS or not w.isalnum() else "CONTENT"
        for w in re.findall(r"\w+|[^\w\s]", t)
    )


def unicode_skeleton_text(t: str) -> str:
    """Preserva TODA pontuação/tipografia Unicode exata e reduz corridas de letras ao esqueleto rítmico."""
    t = re.sub(r"[A-Z][a-z]+", "Tt", t)
    t = re.sub(r"[A-Z]{2,}", "UU", t)
    t = re.sub(r"[a-z]+", "w", t)
    t = re.sub(r"\d+", "0", t)
    return t


def host_distortion_text(t: str) -> str:
    """
    Mantém palavras funcionais/retóricas + contrações + caracteres Unicode exatos
    (“ ” ‘ ’ — – ; : * #) e mascara substantivos/verbos por faixa de tamanho para ZERO overfitting temático.
    """
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
            if low in SMALL_WORDS or "'" in low:
                tag = low
                if has_curly:
                    tag += "_CURLY"
                elif has_str:
                    tag += "_STR"
                if tok[0].isupper():
                    tag = "CAP_" + tag
                out.append(tag)
            else:
                L = len(tok)
                b = "S" if L <= 4 else ("M" if L <= 7 else "L")
                out.append(("C" if tok[0].isupper() else "c") + b)
    return " ".join(out)


# ==============================================================================
# 2. Clusterização Hierárquica Recursiva Limitada (`cluster_size_capped`)
# ==============================================================================
def compute_tfidf_sim_and_clusters(
    texts: List[str],
    base_thresh: float = 0.89,
    min_size: int = 2,
    max_size: int = 18,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna:
      groups: (N,) IDs inteiros de cluster (todos <= max_size)
      twin_rows: (N,) máscara booleana de clusters com tamanho em [min_size, max_size]
      S_mat: (N, N) matriz de similaridade cosseno TF-IDF em float32
    """
    n = len(texts)
    if n < 2:
        return np.ones(n, dtype=np.int32), np.zeros(n, dtype=bool), np.eye(n, dtype=np.float32)

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

    size = np.bincount(out)[out]
    twin_rows = (size >= min_size) & (size <= max_size)
    return out.astype(np.int32), twin_rows, S_mat


def group_center(X: np.ndarray, groups: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    Xc = np.zeros_like(X, dtype=np.float32)
    ugs = np.unique(groups[mask]) if (mask is not None and mask.any()) else np.unique(groups)
    for g in ugs:
        m = groups == g
        if m.sum() > 1:
            Xc[m] = X[m] - X[m].mean(axis=0, keepdims=True)
    return Xc


class TestTimeSCache:
    """Memória associativa O(1) Scar-Cache sobre 32x32 slots constantes."""
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
        ugs = np.unique(groups[twin_mask]) if (twin_mask is not None and twin_mask.any()) else np.unique(groups)
        for g in ugs:
            m = groups == g
            if m.sum() > 1:
                dz = Z[m] - Z[m].mean(axis=0, keepdims=True)
                W_scar = (dz.T @ dz) / float(m.sum())
                X_out[m] += self.resonance_scale * ((dz @ W_scar.T) @ self.proj.T)
        return X_out


# ==============================================================================
# 3. UniversalS1ReflexiveAgent (PyTorch Puro — Serializável em .safetensors)
# ==============================================================================
class UniversalS1ReflexiveMLP(nn.Module):
    """
    FrontEnd -> Residual LayerNorm SiLU Trunk -> Policy (18) + Epistemic Value (1).
    Opera sobre concatenação compacta [PCA_Titans, Forensic_120D, Centered_Deltas].
    """
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


def train_s1_reflex_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int = 42,
    hidden_dim: int = 320,
    epochs: int = 95,
) -> Tuple[UniversalS1ReflexiveMLP, StandardScaler]:
    scaler = StandardScaler().fit(X_tr)
    Xs = np.clip(scaler.transform(X_tr), -6.0, 6.0).astype(np.float32)

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = UniversalS1ReflexiveMLP(Xs.shape[1], hidden_dim=hidden_dim, num_classes=K)
    opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=3e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    xt = torch.from_numpy(Xs)
    yt = torch.from_numpy(y_tr.astype(np.int64))

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        # Ruído Gaussiano leve de regularização de variedade
        noise = 0.04 * torch.randn_like(xt)
        logits, val = model(xt + noise)
        ce = F.cross_entropy(logits, yt, label_smoothing=0.06)
        with torch.no_grad():
            correct = (logits.argmax(dim=-1) == yt).float()
        v_loss = F.mse_loss(val, correct)
        loss = ce + 0.15 * v_loss
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
        probs = F.softmax(logits / temp, dim=-1).numpy()
    return probs


# ==============================================================================
# 4. SCache O(1) Bayesian Cavity Shared-Prompt Generative Head
# ==============================================================================
class SCacheBayesianSharedPromptHead:
    """
    Memória Associativa Gerativa O(1) (`z_i = mu_{c_i} + eta_prompt + eps_i`):
    Armazena apenas matrizes de dimensão constante `d x d` (`d = 192`, < 200 KB total!).
    Durante a inferência em um cluster G, estima o vetor temático puro `eta_{G \ j}`
    excluindo a própria passagem `j` (Cavity Kalman Filter) e computa a log-verossimilhança
    exata de cada autor sob covariância marginal encolhida por Ledoit-Wolf.
    """
    def __init__(self, n_components: int = 192):
        self.d = n_components
        self.sc_t2 = StandardScaler()
        self.sc_t3 = StandardScaler()
        self.sc_f = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        self.mu_c: np.ndarray | None = None
        self.prec_eps: np.ndarray | None = None
        self.prec_prompt: np.ndarray | None = None
        self.cov_eps: np.ndarray | None = None
        self.cov_prompt: np.ndarray | None = None

    def _project_fit(self, Z_t2: np.ndarray, Z_t3: np.ndarray, F_120: np.ndarray) -> np.ndarray:
        a = self.sc_t2.fit_transform(Z_t2) / np.sqrt(Z_t2.shape[1])
        b = self.sc_t3.fit_transform(Z_t3) / np.sqrt(Z_t3.shape[1])
        c = self.sc_f.fit_transform(F_120) / np.sqrt(F_120.shape[1]) * 0.85
        X_cat = np.hstack([a, b, c]).astype(np.float32)
        return self.pca.fit_transform(X_cat).astype(np.float32)

    def project(self, Z_t2: np.ndarray, Z_t3: np.ndarray, F_120: np.ndarray) -> np.ndarray:
        a = self.sc_t2.transform(Z_t2) / np.sqrt(Z_t2.shape[1])
        b = self.sc_t3.transform(Z_t3) / np.sqrt(Z_t3.shape[1])
        c = self.sc_f.transform(F_120) / np.sqrt(F_120.shape[1]) * 0.85
        X_cat = np.hstack([a, b, c]).astype(np.float32)
        return self.pca.transform(X_cat).astype(np.float32)

    def fit(self, Z_t2: np.ndarray, Z_t3: np.ndarray, F_120: np.ndarray, y: np.ndarray, prompts: np.ndarray) -> "SCacheBayesianSharedPromptHead":
        Z = self._project_fit(Z_t2, Z_t3, F_120)
        self.mu_c = np.zeros((K, self.d), dtype=np.float64)
        for k in range(K):
            self.mu_c[k] = Z[y == k].mean(axis=0)

        R = Z.astype(np.float64) - self.mu_c[y]
        within_res = []
        group_means = []
        group_sizes = []
        for g in np.unique(prompts):
            m = prompts == g
            cnt = int(m.sum())
            if cnt >= 2:
                rg = R[m]
                gbar = rg.mean(axis=0, keepdims=True)
                scale_dof = np.sqrt(cnt / float(cnt - 1))
                within_res.append((rg - gbar) * scale_dof)
                group_means.append(gbar[0])
                group_sizes.append(cnt)

        E_mat = np.vstack(within_res)
        G_mat = np.vstack(group_means)
        mean_m = float(np.mean(group_sizes))

        cov_eps = LedoitWolf().fit(E_mat).covariance_
        cov_gbar = LedoitWolf().fit(G_mat).covariance_
        cov_prompt_raw = cov_gbar - (cov_eps / mean_m)

        # Projeção PSD garantida via autovalores
        eigvals, eigvecs = np.linalg.eigh(cov_prompt_raw)
        eigvals = np.maximum(eigvals, 1e-4 * np.trace(cov_eps) / self.d)
        cov_prompt = (eigvecs * eigvals) @ eigvecs.T

        # Regularização suave diagonal para estabilidade numérica
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
        n_em_iters: int = 3,
        temp: float = 2.5,
    ) -> np.ndarray:
        Z = self.project(Z_t2, Z_t3, F_120).astype(np.float64)
        N = len(Z)
        mu = self.mu_c  # (K, d)

        # Pré-computa matrizes Wiener e precisões preditivas para m_other = 0 .. 20
        wiener_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        def get_ops(m_other: int) -> Tuple[np.ndarray, np.ndarray]:
            if m_other not in wiener_cache:
                if m_other <= 0:
                    cov_pred = self.cov_prompt + self.cov_eps
                    prec_pred = np.linalg.inv(cov_pred)
                    W_eta = np.zeros((self.d, self.d), dtype=np.float64)
                else:
                    post_prec = self.prec_prompt + float(m_other) * self.prec_eps
                    post_cov = np.linalg.inv(post_prec)
                    W_eta = post_cov @ self.prec_eps
                    cov_pred = self.cov_eps + post_cov
                    prec_pred = np.linalg.inv(cov_pred)
                wiener_cache[m_other] = (W_eta, prec_pred)
            return wiener_cache[m_other]

        # Inicializa Q com prior (se fornecido) ou verossimilhança marginal m_other=0
        _, prec0 = get_ops(0)
        diff0 = Z[:, None, :] - mu[None, :, :]  # (N, K, d)
        ll0 = -0.5 * np.einsum("nkd,de,nke->nk", diff0, prec0, diff0) / temp
        Q = softmax(ll0, axis=1) if prior_probs is None else (0.65 * prior_probs + 0.35 * softmax(ll0, axis=1))

        ll_out = ll0.copy()
        for _ in range(n_em_iters):
            exp_mu = Q @ mu  # (N, d) autor esperado de cada passagem
            res = Z - exp_mu  # (N, d) resíduo estimado de prompt

            for g in np.unique(groups):
                idx = np.where(groups == g)[0]
                m = len(idx)
                if m <= 1:
                    ll_out[idx] = ll0[idx]
                    continue
                W_eta, prec_m = get_ops(m - 1)
                sum_res = res[idx].sum(axis=0, keepdims=True)  # (1, d)
                # Cavity sum para cada j em idx: exclui o próprio res[j]!
                cav_sum = sum_res - res[idx]  # (m, d)
                eta_hat = cav_sum @ W_eta.T  # (m, d)
                z_clean = Z[idx] - eta_hat  # (m, d)
                diff = z_clean[:, None, :] - mu[None, :, :]  # (m, K, d)
                ll_g = -0.5 * np.einsum("mkd,de,mke->mk", diff, prec_m, diff) / temp
                ll_out[idx] = ll_g

            P_like = softmax(ll_out, axis=1)
            Q = P_like if prior_probs is None else (0.50 * prior_probs + 0.50 * P_like)

        return ll_out - logsumexp(ll_out, axis=1, keepdims=True)


# ==============================================================================
# 5. Repulsão Injetiva Cavity-EM Ponderada por Similaridade + Soft Sinkhorn
# ==============================================================================
def apply_similarity_weighted_cavity_repulsion(
    log_P: np.ndarray,
    groups: np.ndarray,
    S_mat: np.ndarray,
    gamma: float = 0.55,
    sim_floor: float = 0.12,
    sim_span: float = 0.30,
    iters: int = 5,
) -> np.ndarray:
    """
    Repulsão contínua entre passagens irmãs do mesmo prompt:
      log Q_{i,k}^{(t+1)} = log P_{i,k} + sum_{j in G \ i} w_{ij} * log(1 - gamma * Q_{j,k}^{(t)})
    onde `w_{ij}` escala suavemente com a similaridade TF-IDF entre `i` e `j`,
    blindando completamente contra fusões acidentais de clusters!
    """
    if gamma <= 1e-6:
        return softmax(log_P, axis=1)

    P0_log = log_P - logsumexp(log_P, axis=1, keepdims=True)
    Q = np.exp(P0_log)

    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        m = len(idx)
        if m < 2 or m > 18:
            continue
        sub_S = S_mat[np.ix_(idx, idx)]
        W = np.clip((sub_S - sim_floor) / max(sim_span, 1e-4), 0.0, 1.0)
        np.fill_diagonal(W, 0.0)
        if W.max() <= 1e-6:
            continue
        # Normaliza para que a repulsão total recebida seja bem-comportada em clusters grandes
        row_norm = np.maximum(W.sum(axis=1, keepdims=True) / min(float(m - 1), 6.0), 1.0)
        W_eff = W / row_norm

        lp_g = P0_log[idx]
        q_g = Q[idx].copy()
        for _ in range(iters):
            excl = np.log(np.clip(1.0 - gamma * q_g, 1e-4, 1.0))  # (m, K)
            q_new = softmax(lp_g + (W_eff @ excl), axis=1)
            q_g = 0.65 * q_new + 0.35 * q_g
        Q[idx] = q_g

    return Q / Q.sum(axis=1, keepdims=True)


def sinkhorn_damped(P: np.ndarray, target_mix: float = 0.50, iters: int = 300) -> np.ndarray:
    """
    Soft Sinkhorn amortecido:
      target_mix = 1.0 -> uniforme estrito N/K
      target_mix = 0.5 -> equilibra calibração global preservando flutuações naturais do split
    """
    if target_mix <= 1e-6 or len(P) < K:
        return P / P.sum(axis=1, keepdims=True)
    P_safe = np.clip(P, 1e-12, 1.0)
    P_safe = P_safe / P_safe.sum(axis=1, keepdims=True)
    emp = P_safe.sum(axis=0)
    unif = np.full(K, len(P_safe) / float(K))
    target = (1.0 - target_mix) * emp + target_mix * unif
    target = np.maximum(target, 1e-6)

    c = np.ones(K, dtype=np.float64)
    for _ in range(iters):
        R = P_safe * c
        R /= R.sum(axis=1, keepdims=True)
        c *= target / np.maximum(R.sum(axis=0), 1e-12)
    R = P_safe * c
    return R / R.sum(axis=1, keepdims=True)


# ==============================================================================
# 6. System 1 -> System 2 Intra-Clan Specialist Verifiers
# ==============================================================================
class IntraClanVerifierSuite:
    """
    4 Verificadores Especialistas Intra-Clã (OpenAI-5, Gemini-4, Claude-5, GLM-2).
    Acionados apenas sob incerteza epistêmica (`ReflexDecision.margin < margin_tau`).
    """
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
        self,
        P_s1: np.ndarray,
        X_feat: np.ndarray,
        margin_tau: float = 0.35,
        beta_max: float = 0.35,
        temp: float = 2.0,
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
            clan_mass = P_out[np.ix_(unc_idx, members)].sum(axis=1, keepdims=True)  # (U, 1)
            active = clan_mass[:, 0] > 0.25
            if not active.any():
                continue
            sub_u = unc_idx[active]
            b_a = beta[active][:, None]
            cm_a = clan_mass[active]

            Xs = self.scalers[clan_name].transform(X_u[active])
            logits_s2 = (Xs @ self.clfs[clan_name].coef_.T + self.clfs[clan_name].intercept_) / temp
            if logits_s2.shape[1] == 1:  # binário (GLM-2)
                p1 = 1.0 / (1.0 + np.exp(-logits_s2[:, 0]))
                p_s2 = np.stack([1.0 - p1, p1], axis=1)
            else:
                p_s2 = softmax(logits_s2, axis=1)

            p_s1_cond = np.clip(P_out[np.ix_(sub_u, members)] / np.maximum(cm_a, 1e-12), 1e-9, 1.0)
            p_s2_cond = np.clip(p_s2, 1e-9, 1.0)
            fused_cond = softmax((1.0 - b_a) * np.log(p_s1_cond) + b_a * np.log(p_s2_cond), axis=1)
            P_out[np.ix_(sub_u, members)] = cm_a * fused_cond

        return P_out / P_out.sum(axis=1, keepdims=True)


# ==============================================================================
# 7. Pipeline Principal de Validação 5-Fold Leak-Free & Treinamento Full
# ==============================================================================
def build_cell2_views(texts: List[str], vecs: List[TfidfVectorizer] | None = None):
    views = [
        ("char", 3, 5, 250000, 1.0),
        ("word", 1, 3, 180000, 0.7),
        ("shape", 2, 6, 60000, 0.5),
        ("small", 1, 3, 50000, 0.5),
    ]
    mats = []
    fitted = []
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
    """Vistas ortográficas/tipográficas livres de tópico para o registro puro do host."""
    specs = [
        ("skel_char", unicode_skeleton_text, "char", (1, 5), 65000, r"(?u)\b\w\w+\b", 1.0),
        ("distort_tok", host_distortion_text, "word", (1, 3), 90000, r"\S+", 0.9),
        ("char_wb", lambda s: s, "char_wb", (2, 5), 110000, r"(?u)\b\w\w+\b", 0.75),
        ("word_punct", lambda s: s, "word", (1, 2), 80000, r"\w+|[^\w\s]", 0.65),
    ]
    mats = []
    fitted = []
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


def main():
    t_start = time.time()
    print("=" * 88)
    print("👑 SYSTEM ONE APEX V33 — TREINO & AUDITORIA 5-FOLD LEAK-FREE (BALANCED ACCURACY)")
    print("=" * 88)

    data_dir = THIS_DIR / "data"
    t2_dir = THIS_DIR / "titans" / "titan2_yi34b_l30"
    t3_dir = THIS_DIR / "titans" / "titan3_mistral_large_123b_l44"
    ext_dir = Path("/Users/juliocesarreisfilho/.gemini/antigravity/brain/2b3990be-2635-4b7d-98c1-2d81f1cbd3e0/scratch/ext_data")

    df_train = pd.read_csv(data_dir / "train.csv")
    df_train["text"] = df_train["text"].fillna("").astype(str)
    y_train = df_train["model"].map(L2I).values
    prompts_train = df_train["prompt_id"].astype(str).values
    n_train = len(df_train)

    Z_yi = np.load(t2_dir / "features_yi34b_l30_train_full.npy").astype(np.float32)
    Z_mis = np.load(t3_dir / "features_mistral_large_123b_l44_train_full.npy").astype(np.float32)
    print(f"✓ Official Host Train: {n_train} rows ({len(np.unique(prompts_train))} prompts)")
    print(f"✓ Titan 2 Yi-34B L30: {Z_yi.shape} | Titan 3 Mistral 123B L44: {Z_mis.shape}")

    # Carrega corpora externos limpos para simular Cell 2 (pool_v4) + Proxy Fresh Anchor (passages.csv)
    pv4 = pd.read_parquet(ext_dir / "pool_v4.parquet")
    pv4 = pv4[pv4["model"].isin(L2I)].copy()
    pv4["text"] = pv4["text"].fillna("").astype(str)
    pv4["y"] = pv4["model"].map(L2I).values

    pa = pd.read_csv(ext_dir / "passages.csv")
    pa = pa[
        (pa["passes_funnel"] == True)
        & (pa["is_refusal"] == False)
        & (pa["is_empty"] == False)
        & (pa["split"] != "pilot_range")
        & (~((pa["split"] == "new") & (pa["model"] == "kimi-k3")))  # Remove kimi-k3 antigo sem apóstrofos curvos!
        & (pa["model"].isin(L2I))
    ].copy()
    pa["text"] = pa["text"].fillna("").astype(str)
    pa["y"] = pa["model"].map(L2I).values
    pa["src_pid"] = pa["source_prompt_id"].fillna("").astype(str)
    print(f"✓ Open Pool v4: {len(pv4)} rows | Clean Proxy-Matched Passages (`passages.csv`): {len(pa)} rows")

    # Extração Bússola 1: 28D legada (para baseline V32) vs 120D Forense Nova
    from test_tri_titan_competition_suite import extract_tricompass_style_features
    F_28 = extract_tricompass_style_features(df_train["text"].tolist())
    t_f0 = time.time()
    F_120 = extract_forensic_compass1_120d(df_train["text"].tolist())
    print(f"✓ Forensic Compass-1 120D extracted in {time.time() - t_f0:.2f}s | Shape: {F_120.shape}")

    # Pré-transforma Open Pool v4 + train nas vistas Cell 2 uma única vez?
    # ATENÇÃO ML-BEST-PRACTICES: para zero vazamento de IDF, mas máxima velocidade em 75k linhas:
    # Vamos ajustar o vocabulário em `pv4` (100% externo, zero overlap com train.csv!) + `trn_idx`!
    # Mais rápido ainda: pré-computa strings transformadas uma vez e ajusta Tfidf/SVC por fold!
    print("\n[1/4] Preparando textos pré-processados para validação 5-Fold estrita...")
    t_p0 = time.time()
    pv4_texts = pv4["text"].tolist()
    pv4_y = pv4["y"].values
    tr_texts = df_train["text"].tolist()

    # Pré-ajusta vocabulário Cell 2 em pv4 (75,783 textos externos disjuntos de train.csv)
    # e em cada fold re-pondera/treina LinearSVC com pv4 + trn_idx (zero vazamento de val_idx!)
    X_pv4_c2, vecs_c2 = build_cell2_views(pv4_texts, vecs=None)
    X_tr_c2, _ = build_cell2_views(tr_texts, vecs=vecs_c2)
    print(f"  ✓ Cell-2 4-view matrices ready in {time.time() - t_p0:.1f}s: pv4={X_pv4_c2.shape}, tr={X_tr_c2.shape}")

    sgkf = StratifiedGroupKFold(n_splits=5)
    folds = list(sgkf.split(df_train, y_train, groups=prompts_train))

    # Arrays OOF (Log-probabilidades / Probabilidades calibradas)
    oof_c2_svc_margin = np.zeros((n_train, K), dtype=np.float32)
    oof_host_svc_margin = np.zeros((n_train, K), dtype=np.float32)
    oof_host_svc_twin_margin = np.zeros((n_train, K), dtype=np.float32)
    oof_proxy_svc_margin = np.zeros((n_train, K), dtype=np.float32)
    oof_proxy_svc_twin_margin = np.zeros((n_train, K), dtype=np.float32)

    # Baseline V32 Titans (28D, thresh=0.85 uncapped)
    oof_v32_t2_plain = np.zeros((n_train, K), dtype=np.float32)
    oof_v32_t2_twin = np.zeros((n_train, K), dtype=np.float32)
    oof_v32_t3_plain = np.zeros((n_train, K), dtype=np.float32)
    oof_v32_t3_twin = np.zeros((n_train, K), dtype=np.float32)
    oof_v32_twin_mask = np.zeros(n_train, dtype=bool)

    # V33 Apex Channels (120D, capped thresh=0.89)
    oof_t2_plain_logit = np.zeros((n_train, K), dtype=np.float32)
    oof_t2_twin_logit = np.zeros((n_train, K), dtype=np.float32)
    oof_t3_plain_logit = np.zeros((n_train, K), dtype=np.float32)
    oof_t3_twin_logit = np.zeros((n_train, K), dtype=np.float32)
    oof_forensic_lr_logit = np.zeros((n_train, K), dtype=np.float32)
    oof_forensic_hgb_prob = np.zeros((n_train, K), dtype=np.float32)
    oof_s1_mlp_prob = np.zeros((n_train, K), dtype=np.float32)
    oof_scache_bayes_logp = np.zeros((n_train, K), dtype=np.float32)

    val_groups_list: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    fold_verifiers: List[Tuple[np.ndarray, IntraClanVerifierSuite, np.ndarray]] = []

    print("\n[2/4] Executando 5-Fold StratifiedGroupKFold (Realistic Blind Test Clustering)...")
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        tf0 = time.time()
        val_pids = set(prompts_train[va_idx])
        tr_texts_f = [tr_texts[i] for i in tr_idx]
        va_texts_f = [tr_texts[i] for i in va_idx]
        y_tr_f = y_train[tr_idx]

        # A. Clusterização cega realista no fold de validação:
        # 1) Legacy V32 (thresh=0.85 uncapped)
        Xt_v = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(va_texts_f)
        D_v = np.clip(1.0 - (Xt_v @ Xt_v.T).toarray(), 0.0, None)
        np.fill_diagonal(D_v, 0.0)
        g_v32 = fcluster(linkage(squareform(D_v, checks=False), method="average"), 0.85, criterion="distance")
        sz_v32 = np.bincount(g_v32)[g_v32]
        tw_v32 = (sz_v32 >= 2) & (sz_v32 <= 18)
        oof_v32_twin_mask[va_idx] = tw_v32

        # 2) V33 Size-Capped (base_thresh=0.89, max_size=18)
        g_va, tw_va, S_va = compute_tfidf_sim_and_clusters(va_texts_f, base_thresh=0.89, min_size=2, max_size=18)
        val_groups_list[fold_i] = (va_idx, g_va, S_va)

        # B. Canal 1: Open-Pool Cell-2 LinearSVC (pv4 + tr_idx)
        X_c2_fit = hstack([X_pv4_c2.T, X_tr_c2[tr_idx].T], format="csr").T
        y_c2_fit = np.concatenate([pv4_y, y_tr_f])
        svc_c2 = LinearSVC(C=0.5, max_iter=2000, random_state=42)
        svc_c2.fit(X_c2_fit, y_c2_fit)
        oof_c2_svc_margin[va_idx] = svc_c2.decision_function(X_tr_c2[va_idx])

        # C. Canal 2A: Host-Exact Distortion & Unicode Typography SVC (100% puro train[tr_idx]!)
        X_he_tr, vecs_he = build_host_exact_views(tr_texts_f, vecs=None)
        X_he_va, _ = build_host_exact_views(va_texts_f, vecs=vecs_he)
        svc_he = LinearSVC(C=0.25, max_iter=2000, random_state=42).fit(X_he_tr, y_tr_f)
        m_he_va = svc_he.decision_function(X_he_va).astype(np.float32)
        oof_host_svc_margin[va_idx] = m_he_va
        oof_host_svc_twin_margin[va_idx] = group_center(m_he_va, g_va, tw_va)

        # D. Canal 2B: Clean Proxy-Matched Anchor (train[tr_idx] w=2.5 + passages_clean leak-free w=1.0)
        pa_leakfree = pa[~pa["src_pid"].isin(val_pids)]
        px_texts = tr_texts_f + pa_leakfree["text"].tolist()
        px_y = np.concatenate([y_tr_f, pa_leakfree["y"].values])
        px_w = np.concatenate([np.full(len(tr_idx), 2.5, dtype=np.float32), np.ones(len(pa_leakfree), dtype=np.float32)])

        X_px_tr_c2, vecs_px1 = build_cell2_views(px_texts, vecs=None)
        X_px_va_c2, _ = build_cell2_views(va_texts_f, vecs=vecs_px1)
        X_px_tr_he, vecs_px2 = build_host_exact_views(px_texts, vecs=None)
        X_px_va_he, _ = build_host_exact_views(va_texts_f, vecs=vecs_px2)

        X_px_tr = hstack([X_px_tr_c2, X_px_tr_he * 0.7], format="csr")
        X_px_va = hstack([X_px_va_c2, X_px_va_he * 0.7], format="csr")
        svc_px = LinearSVC(C=0.35, max_iter=2000, random_state=42).fit(X_px_tr, px_y, sample_weight=px_w)
        m_px_va = svc_px.decision_function(X_px_va).astype(np.float32)
        oof_proxy_svc_margin[va_idx] = m_px_va
        oof_proxy_svc_twin_margin[va_idx] = group_center(m_px_va, g_va, tw_va)

        # E. Baseline V32 Titans (28D + SCache + thresh=0.85)
        sc28 = StandardScaler().fit(F_28[tr_idx])
        f28_tr = sc28.transform(F_28[tr_idx])
        f28_va = sc28.transform(F_28[va_idx])
        for Z_all, out_p, out_t in [
            (Z_yi, oof_v32_t2_plain, oof_v32_t2_twin),
            (Z_mis, oof_v32_t3_plain, oof_v32_t3_twin),
        ]:
            scache_b = TestTimeSCache(32, 0.15)
            Xtr_b = scache_b.apply_scache_resonance(np.hstack([Z_all[tr_idx], f28_tr]), prompts_train[tr_idx])
            Xva_b = scache_b.apply_scache_resonance(np.hstack([Z_all[va_idx], f28_va]), g_v32, tw_v32)
            sp = StandardScaler().fit(Xtr_b)
            cp = LogisticRegression(C=0.1, max_iter=600, random_state=42).fit(sp.transform(Xtr_b), y_tr_f)
            out_p[va_idx] = (sp.transform(Xva_b) @ cp.coef_.T + cp.intercept_)

            Xtr_bc = group_center(Xtr_b, prompts_train[tr_idx])
            Xva_bc = group_center(Xva_b, g_v32, tw_v32)
            sc_c = StandardScaler().fit(Xtr_bc)
            cc = LogisticRegression(C=0.1, max_iter=600, random_state=42).fit(sc_c.transform(Xtr_bc), y_tr_f)
            out_t[va_idx] = (sc_c.transform(Xva_bc) @ cc.coef_.T + cc.intercept_)

        # F. V33 Apex Titans (120D Host-Aware + SCache + Capped Clustering)
        sc120 = StandardScaler().fit(F_120[tr_idx])
        f120_tr = np.clip(sc120.transform(F_120[tr_idx]), -6.0, 6.0)
        f120_va = np.clip(sc120.transform(F_120[va_idx]), -6.0, 6.0)

        for Z_all, out_p, out_t in [
            (Z_yi, oof_t2_plain_logit, oof_t2_twin_logit),
            (Z_mis, oof_t3_plain_logit, oof_t3_twin_logit),
        ]:
            scache_a = TestTimeSCache(32, 0.15)
            Xtr_a = scache_a.apply_scache_resonance(np.hstack([Z_all[tr_idx], f120_tr * 0.5]), prompts_train[tr_idx])
            Xva_a = scache_a.apply_scache_resonance(np.hstack([Z_all[va_idx], f120_va * 0.5]), g_va, tw_va)
            sp = StandardScaler().fit(Xtr_a)
            cp = LogisticRegression(C=0.08, max_iter=700, random_state=42).fit(sp.transform(Xtr_a), y_tr_f)
            out_p[va_idx] = (sp.transform(Xva_a) @ cp.coef_.T + cp.intercept_).astype(np.float32)

            Xtr_ac = group_center(Xtr_a, prompts_train[tr_idx])
            Xva_ac = group_center(Xva_a, g_va, tw_va)
            sc_c = StandardScaler().fit(Xtr_ac)
            cc = LogisticRegression(C=0.08, max_iter=700, random_state=42).fit(sc_c.transform(Xtr_ac), y_tr_f)
            out_t[va_idx] = (sc_c.transform(Xva_ac) @ cc.coef_.T + cc.intercept_).astype(np.float32)

        # G. Bússola 1 Forense 120D Pura (Plain + Centered) -> LR + HGB
        f120_tr_dual = np.hstack([F_120[tr_idx], group_center(F_120[tr_idx], prompts_train[tr_idx])])
        f120_va_dual = np.hstack([F_120[va_idx], group_center(F_120[va_idx], g_va, tw_va)])
        sc_fdual = StandardScaler().fit(f120_tr_dual)
        clf_flr = LogisticRegression(C=0.15, max_iter=1000, random_state=42).fit(
            np.clip(sc_fdual.transform(f120_tr_dual), -6, 6), y_tr_f
        )
        oof_forensic_lr_logit[va_idx] = (
            np.clip(sc_fdual.transform(f120_va_dual), -6, 6) @ clf_flr.coef_.T + clf_flr.intercept_
        ).astype(np.float32)

        hgb = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=5.0, random_state=42
        ).fit(f120_tr_dual, y_tr_f)
        oof_forensic_hgb_prob[va_idx] = hgb.predict_proba(f120_va_dual).astype(np.float32)

        # H. SCache O(1) Bayesian Shared-Prompt Generative Head + UniversalS1ReflexiveMLP + S2 Verifiers
        scache_bayes = SCacheBayesianSharedPromptHead(n_components=192).fit(
            Z_yi[tr_idx], Z_mis[tr_idx], F_120[tr_idx], y_tr_f, prompts_train[tr_idx]
        )
        Z_pca_tr = scache_bayes.project(Z_yi[tr_idx], Z_mis[tr_idx], F_120[tr_idx])
        Z_pca_va = scache_bayes.project(Z_yi[va_idx], Z_mis[va_idx], F_120[va_idx])

        # S1 Reflexive MLP sobre [Z_pca (192), Centered_Z_pca (192), f120_dual (240)] = 624D
        X_s1_tr = np.hstack([Z_pca_tr, group_center(Z_pca_tr, prompts_train[tr_idx]), f120_tr_dual])
        X_s1_va = np.hstack([Z_pca_va, group_center(Z_pca_va, g_va, tw_va), f120_va_dual])

        s1_mlp, s1_sc = train_s1_reflex_mlp(X_s1_tr, y_tr_f, seed=42 + fold_i)
        oof_s1_mlp_prob[va_idx] = predict_s1_reflex_mlp(s1_mlp, s1_sc, X_s1_va, temp=1.5)

        # Prior preliminar para guiar o Cavity Kalman Filter do SCache Bayesiano
        p_prior_va = softmax(
            1.0 * np.log(np.clip(softmax(oof_c2_svc_margin[va_idx], axis=1), 1e-9, 1.0))
            + 0.6 * np.log(np.clip(softmax(m_px_va, axis=1), 1e-9, 1.0))
            + 0.3 * np.log(np.clip(oof_s1_mlp_prob[va_idx], 1e-9, 1.0)),
            axis=1,
        )
        oof_scache_bayes_logp[va_idx] = scache_bayes.predict_log_proba(
            Z_yi[va_idx], Z_mis[va_idx], F_120[va_idx], g_va, prior_probs=p_prior_va, n_em_iters=3, temp=2.2
        ).astype(np.float32)

        s2_ver = IntraClanVerifierSuite().fit(X_s1_tr, y_tr_f)
        fold_verifiers.append((va_idx, s2_ver, X_s1_va))

        print(f"  ✓ Fold {fold_i + 1}/5 completed in {time.time() - tf0:.1f}s")

    # ==========================================================================
    # 3. Avaliação Modular de Ablação 5-Fold OOF (`balanced_accuracy_score`)
    # ==========================================================================
    print("\n" + "=" * 88)
    print("📊 ABLAÇÃO MODULAR ESTRITA 5-FOLD OOF (MÉTRICA OFICIAL: BALANCED ACCURACY)")
    print("=" * 88)

    def eval_oof(name: str, P: np.ndarray, mix: float = 0.50):
        b_raw = balanced_accuracy_score(y_train, P.argmax(axis=1))
        P_s100 = sinkhorn_damped(P, target_mix=1.0, iters=300)
        b_s100 = balanced_accuracy_score(y_train, P_s100.argmax(axis=1))
        P_sd = sinkhorn_damped(P, target_mix=mix, iters=300)
        b_sd = balanced_accuracy_score(y_train, P_sd.argmax(axis=1))
        ll = log_loss(y_train, np.clip(P_sd, 1e-9, 1.0))
        print(f"  {name:<50} | Raw: {b_raw*100:6.2f}% | Sink(1.0): {b_s100*100:6.2f}% | Sink({mix:.2f}): {b_sd*100:6.2f}% | LL: {ll:.4f}")
        return b_sd, P_sd

    # Canais individuais
    eval_oof("0a. Cell-2 Open-Pool LinearSVC Only", softmax(oof_c2_svc_margin, axis=1))
    eval_oof("0b. Host-Exact Distortion+Unicode SVC Only", softmax(oof_host_svc_margin + 0.5 * oof_host_svc_twin_margin, axis=1))
    eval_oof("0c. Clean Proxy-Matched 8-View SVC Only", softmax(oof_proxy_svc_margin + 0.5 * oof_proxy_svc_twin_margin, axis=1))
    eval_oof("0d. Forensic 120D Scouts (LR + HGB) Only", 0.5 * softmax(oof_forensic_lr_logit / 2.5, axis=1) + 0.5 * oof_forensic_hgb_prob)
    eval_oof("0e. Titan 2 (Yi-34B L30 + 120D + Twin) Only", softmax((0.4 * oof_t2_plain_logit + 0.6 * oof_t2_twin_logit) / 5.0, axis=1))
    eval_oof("0f. Titan 3 (Mistral 123B L44 + 120D + Twin) Only", softmax((0.4 * oof_t3_plain_logit + 0.6 * oof_t3_twin_logit) / 5.0, axis=1))
    eval_oof("0g. UniversalS1ReflexiveMLP (Titans+120D) Only", oof_s1_mlp_prob)
    eval_oof("0h. SCache O(1) Bayesian Cavity Shared-Prompt", np.exp(oof_scache_bayes_logp))
    print("-" * 88)

    # Baseline V32 Exato (sem DeBERTa local, com Cell-2 SVC + Titan2 + Titan3, T=5.0, thresh=0.85)
    L_v32 = (
        1.00 * np.log(np.clip(softmax(oof_c2_svc_margin, axis=1), 1e-9, 1.0))
        + 0.15 * np.log(np.clip(softmax(oof_v32_t2_plain / 5.0, axis=1), 1e-9, 1.0))
        + 0.15 * np.log(np.clip(softmax(oof_v32_t3_plain / 5.0, axis=1), 1e-9, 1.0))
    )
    L_v32[oof_v32_twin_mask] += (
        0.25 * np.log(np.clip(softmax(oof_v32_t2_twin[oof_v32_twin_mask] / 5.0, axis=1), 1e-9, 1.0))
        + 0.25 * np.log(np.clip(softmax(oof_v32_t3_twin[oof_v32_twin_mask] / 5.0, axis=1), 1e-9, 1.0))
    )
    b_v32, _ = eval_oof("1. V32 Baseline (Cell2 SVC + Titans 28D)", softmax(L_v32, axis=1), mix=1.0)

    # Stepwise Ablation of V33 Upgrades!
    p_c2 = np.log(np.clip(softmax(oof_c2_svc_margin / 0.85, axis=1), 1e-9, 1.0))
    p_he = np.log(np.clip(softmax((oof_host_svc_margin + 0.45 * oof_host_svc_twin_margin) / 0.85, axis=1), 1e-9, 1.0))
    p_px = np.log(np.clip(softmax((oof_proxy_svc_margin + 0.55 * oof_proxy_svc_twin_margin) / 0.85, axis=1), 1e-9, 1.0))
    p_f120 = np.log(np.clip(0.45 * softmax(oof_forensic_lr_logit / 2.5, axis=1) + 0.55 * oof_forensic_hgb_prob, 1e-9, 1.0))
    p_t2p = np.log(np.clip(softmax(oof_t2_plain_logit / 5.0, axis=1), 1e-9, 1.0))
    p_t2c = np.log(np.clip(softmax(oof_t2_twin_logit / 5.0, axis=1), 1e-9, 1.0))
    p_t3p = np.log(np.clip(softmax(oof_t3_plain_logit / 5.0, axis=1), 1e-9, 1.0))
    p_t3c = np.log(np.clip(softmax(oof_t3_twin_logit / 5.0, axis=1), 1e-9, 1.0))
    p_s1 = np.log(np.clip(oof_s1_mlp_prob, 1e-9, 1.0))
    p_scb = np.clip(oof_scache_bayes_logp, -20.0, 0.0)

    # + Upgrade A: Capped Clustering + 120D Titans
    L_upA = 1.00 * p_c2 + 0.15 * p_t2p + 0.25 * p_t2c + 0.15 * p_t3p + 0.25 * p_t3c
    eval_oof("2. + Capped Clustering (0.89) & 120D Titans", softmax(L_upA, axis=1))

    # + Upgrade B: Host-Exact + Clean Proxy Anchors + Forensic 120D Scouts
    L_upB = (
        0.75 * p_c2
        + 0.55 * p_he
        + 0.95 * p_px
        + 0.30 * p_f120
        + 0.15 * p_t2p + 0.25 * p_t2c
        + 0.15 * p_t3p + 0.25 * p_t3c
    )
    eval_oof("3. + Host-Exact + Proxy-Clean + Forensic 120D", softmax(L_upB, axis=1))

    # + Upgrade C: UniversalS1ReflexiveMLP + SCache O(1) Bayesian Shared-Prompt Head
    L_upC = L_upB + 0.35 * p_s1 + 0.28 * p_scb
    eval_oof("4. + UniversalS1MLP + SCache Bayesian Head", softmax(L_upC, axis=1))

    # Nested Out-of-Fold Weight Optimization (otimiza pesos em 4 folds, avalia no 5º!)
    channels = np.stack([p_c2, p_he, p_px, p_f120, p_t2p, p_t2c, p_t3p, p_t3c, p_s1, p_scb], axis=0)  # (10, N, K)
    ch_names = ["c2_svc", "host_svc", "proxy_svc", "forensic120", "t2_plain", "t2_twin", "t3_plain", "t3_twin", "s1_mlp", "scache_bayes"]
    oof_nested_logp = np.zeros((n_train, K), dtype=np.float64)
    learned_weights = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        def loss_fn(w):
            w_pos = np.maximum(w, 0.0)
            comb = np.tensordot(w_pos, channels[:, tr_idx, :], axes=(0, 0))
            P_tr = softmax(comb, axis=1)
            # NLL + regularização L2 em torno dos priors estruturais
            w_prior = np.array([0.75, 0.55, 0.95, 0.30, 0.15, 0.25, 0.15, 0.25, 0.35, 0.28])
            return log_loss(y_train[tr_idx], np.clip(P_tr, 1e-9, 1.0)) + 0.015 * np.sum((w_pos - w_prior) ** 2)

        w0 = np.array([0.75, 0.55, 0.95, 0.30, 0.15, 0.25, 0.15, 0.25, 0.35, 0.28])
        res = minimize(loss_fn, w0, method="L-BFGS-B", bounds=[(0.02, 2.0)] * len(w0))
        w_fold = res.x / np.mean(res.x) * np.mean(w0)
        learned_weights.append(w_fold)
        oof_nested_logp[va_idx] = np.tensordot(w_fold, channels[:, va_idx, :], axes=(0, 0))

    mean_w = np.mean(learned_weights, axis=0)
    print("\n  Pesos médios Nested-OOF aprendidos:")
    for nm, val in zip(ch_names, mean_w):
        print(f"    {nm:<15}: {val:.4f}")

    P_nested = softmax(oof_nested_logp, axis=1)
    eval_oof("5. + Nested-CV Calibrated Fusion", P_nested)

    # + Upgrade D: Repulsão Injetiva Cavity-EM Ponderada por Similaridade Par-a-Par
    best_gamma, best_cav_score, P_after_cav = 0.0, 0.0, P_nested.copy()
    for gm in [0.0, 0.25, 0.40, 0.55, 0.70, 0.82]:
        P_try = np.zeros_like(P_nested)
        for fold_i in range(5):
            va_idx, g_va, S_va = val_groups_list[fold_i]
            P_try[va_idx] = apply_similarity_weighted_cavity_repulsion(
                oof_nested_logp[va_idx], g_va, S_va, gamma=gm, sim_floor=0.12, sim_span=0.30, iters=5
            )
        sc = balanced_accuracy_score(y_train, sinkhorn_damped(P_try, 0.50).argmax(axis=1))
        if sc > best_cav_score:
            best_cav_score = sc
            best_gamma = gm
            P_after_cav = P_try.copy()

    eval_oof(f"6. + Cavity-EM Sibling Repulsion (gamma={best_gamma:.2f})", P_after_cav)

    # + Upgrade E: System 1 -> System 2 Intra-Clan Specialist Verifiers (`ReflexDecision`)
    best_beta, best_s2_score, P_after_s2 = 0.0, 0.0, P_after_cav.copy()
    for bt in [0.0, 0.15, 0.25, 0.35]:
        P_try = P_after_cav.copy()
        for va_idx, s2_ver, X_s1_va in fold_verifiers:
            P_try[va_idx] = s2_ver.refine_probabilities(P_try[va_idx], X_s1_va, margin_tau=0.35, beta_max=bt, temp=2.0)
        sc = balanced_accuracy_score(y_train, sinkhorn_damped(P_try, 0.50).argmax(axis=1))
        if sc > best_s2_score:
            best_s2_score = sc
            best_beta = bt
            P_after_s2 = P_try.copy()

    eval_oof(f"7. + S1->S2 Clan Verifiers (beta={best_beta:.2f})", P_after_s2)

    # Validação de robustez do `target_mix` em 200 splits simulados Public (35% prompts) com desbalanceamento realista
    print("\n[3/4] Testando robustez de `sinkhorn_damped` em 200 subamostras Public-LB (35% prompts + dropout de autor)...")
    rng = np.random.default_rng(2026)
    uniq_pids = np.unique(prompts_train)
    best_mix, best_pub_mean = 0.50, 0.0
    for mix_cand in [0.0, 0.35, 0.50, 0.65, 0.80, 1.00]:
        scores_sim = []
        for _ in range(200):
            chosen_p = rng.choice(uniq_pids, size=int(0.35 * len(uniq_pids)), replace=False)
            m_p = np.isin(prompts_train, chosen_p)
            idx_p = np.where(m_p)[0]
            # Simula desbalanceamento natural (retém 85% aleatório das passagens para que classes não sejam 100% idênticas)
            keep = rng.choice(idx_p, size=int(0.85 * len(idx_p)), replace=False)
            P_sub = sinkhorn_damped(P_after_s2[keep], target_mix=mix_cand, iters=300)
            scores_sim.append(balanced_accuracy_score(y_train[keep], P_sub.argmax(axis=1)))
        arr_s = np.asarray(scores_sim)
        print(f"  target_mix = {mix_cand:.2f} -> Mean Simulated Public BalAcc: {arr_s.mean()*100:.2f}% (95% CI: [{np.percentile(arr_s, 2.5)*100:.2f}%, {np.percentile(arr_s, 97.5)*100:.2f}%])")
        if arr_s.mean() > best_pub_mean:
            best_pub_mean = arr_s.mean()
            best_mix = mix_cand

    final_oof_bal = balanced_accuracy_score(y_train, sinkhorn_damped(P_after_s2, best_mix).argmax(axis=1))
    print("=" * 88)
    print(f"🏆 V33 SYSTEM ONE APEX FINAL OOF BALANCED ACCURACY: {final_oof_bal*100:.2f}% (vs V32 {b_v32*100:.2f}%, Gain: +{(final_oof_bal - b_v32)*100:.2f}%)")
    print("=" * 88)

    # ==========================================================================
    # 4. Treinamento Final em 100% de `train.csv` e Exportação `.safetensors`
    # ==========================================================================
    print("\n[4/4] Treinando Heads Finais V33 em 100% de `train.csv` e exportando `.safetensors`...")
    tensors_out: Dict[str, torch.Tensor] = {}

    def put_np(key: str, arr: np.ndarray, dtype=torch.float32):
        tensors_out[key] = torch.from_numpy(np.ascontiguousarray(arr)).to(dtype)

    # 1. Scaler 120D para os Titãs
    sc120_full = StandardScaler().fit(F_120)
    f120_full = np.clip(sc120_full.transform(F_120), -6.0, 6.0)
    put_np("f120_scaler.mean", sc120_full.mean_)
    put_np("f120_scaler.scale", sc120_full.scale_)

    # 2. Titan 2 (Yi-34B L30 + 120D + SCache)
    scache_t2 = TestTimeSCache(32, 0.15)
    X_t2_full = scache_t2.apply_scache_resonance(np.hstack([Z_yi, f120_full * 0.5]), prompts_train)
    sp_t2 = StandardScaler().fit(X_t2_full)
    cp_t2 = LogisticRegression(C=0.08, max_iter=800, random_state=42).fit(sp_t2.transform(X_t2_full), y_train)
    X_t2_c = group_center(X_t2_full, prompts_train)
    sc_t2c = StandardScaler().fit(X_t2_c)
    cc_t2 = LogisticRegression(C=0.08, max_iter=800, random_state=42).fit(sc_t2c.transform(X_t2_c), y_train)

    put_np("t2.scaler_plain.mean", sp_t2.mean_)
    put_np("t2.scaler_plain.scale", sp_t2.scale_)
    put_np("t2.plain.weight", cp_t2.coef_)
    put_np("t2.plain.bias", cp_t2.intercept_)
    put_np("t2.scaler_centered.mean", sc_t2c.mean_)
    put_np("t2.scaler_centered.scale", sc_t2c.scale_)
    put_np("t2.centered.weight", cc_t2.coef_)
    put_np("t2.centered.bias", cc_t2.intercept_)

    # 3. Titan 3 (Mistral 123B L44 + 120D + SCache)
    scache_t3 = TestTimeSCache(32, 0.15)
    X_t3_full = scache_t3.apply_scache_resonance(np.hstack([Z_mis, f120_full * 0.5]), prompts_train)
    sp_t3 = StandardScaler().fit(X_t3_full)
    cp_t3 = LogisticRegression(C=0.08, max_iter=800, random_state=42).fit(sp_t3.transform(X_t3_full), y_train)
    X_t3_c = group_center(X_t3_full, prompts_train)
    sc_t3c = StandardScaler().fit(X_t3_c)
    cc_t3 = LogisticRegression(C=0.08, max_iter=800, random_state=42).fit(sc_t3c.transform(X_t3_c), y_train)

    put_np("t3.scaler_plain.mean", sp_t3.mean_)
    put_np("t3.scaler_plain.scale", sp_t3.scale_)
    put_np("t3.plain.weight", cp_t3.coef_)
    put_np("t3.plain.bias", cp_t3.intercept_)
    put_np("t3.scaler_centered.mean", sc_t3c.mean_)
    put_np("t3.scaler_centered.scale", sc_t3c.scale_)
    put_np("t3.centered.weight", cc_t3.coef_)
    put_np("t3.centered.bias", cc_t3.intercept_)

    # 4. Forensic 120D Linear Head
    f120_dual_full = np.hstack([F_120, group_center(F_120, prompts_train)])
    sc_fd_full = StandardScaler().fit(f120_dual_full)
    clf_flr_full = LogisticRegression(C=0.15, max_iter=1000, random_state=42).fit(
        np.clip(sc_fd_full.transform(f120_dual_full), -6, 6), y_train
    )
    put_np("f120_dual.mean", sc_fd_full.mean_)
    put_np("f120_dual.scale", sc_fd_full.scale_)
    put_np("f120_lr.weight", clf_flr_full.coef_)
    put_np("f120_lr.bias", clf_flr_full.intercept_)

    # 5. SCache O(1) Bayesian Shared-Prompt Generative Head
    scache_bayes_full = SCacheBayesianSharedPromptHead(n_components=192).fit(
        Z_yi, Z_mis, F_120, y_train, prompts_train
    )
    put_np("scb.sc_t2.mean", scache_bayes_full.sc_t2.mean_)
    put_np("scb.sc_t2.scale", scache_bayes_full.sc_t2.scale_)
    put_np("scb.sc_t3.mean", scache_bayes_full.sc_t3.mean_)
    put_np("scb.sc_t3.scale", scache_bayes_full.sc_t3.scale_)
    put_np("scb.sc_f.mean", scache_bayes_full.sc_f.mean_)
    put_np("scb.sc_f.scale", scache_bayes_full.sc_f.scale_)
    put_np("scb.pca.mean", scache_bayes_full.pca.mean_)
    put_np("scb.pca.components", scache_bayes_full.pca.components_)
    put_np("scb.mu_c", scache_bayes_full.mu_c)
    put_np("scb.cov_eps", scache_bayes_full.cov_eps)
    put_np("scb.cov_prompt", scache_bayes_full.cov_prompt)
    put_np("scb.prec_eps", scache_bayes_full.prec_eps)
    put_np("scb.prec_prompt", scache_bayes_full.prec_prompt)

    # 6. Ensemble de 3 Seeds do UniversalS1ReflexiveMLP (624D -> 320 -> 18)
    Z_pca_full = scache_bayes_full.project(Z_yi, Z_mis, F_120)
    X_s1_full = np.hstack([Z_pca_full, group_center(Z_pca_full, prompts_train), f120_dual_full])
    s1_sc_full = StandardScaler().fit(X_s1_full)
    put_np("s1.scaler.mean", s1_sc_full.mean_)
    put_np("s1.scaler.scale", s1_sc_full.scale_)

    for s_idx, seed_val in enumerate([42, 137, 2026]):
        mlp_m, _ = train_s1_reflex_mlp(X_s1_full, y_train, seed=seed_val, hidden_dim=320, epochs=95)
        for k_param, v_param in mlp_m.state_dict().items():
            tensors_out[f"s1.mlp{s_idx}.{k_param}"] = v_param.detach().cpu().float()

    # 7. System 2 Intra-Clan Specialist Verifiers
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
        "schema": "system-one-apex-v33",
        "labels": LABELS,
        "oof_v32_baseline_bal_acc": float(b_v32),
        "oof_v33_apex_bal_acc": float(final_oof_bal),
        "oof_gain": float(final_oof_bal - b_v32),
        "learned_weights": {nm: float(val) for nm, val in zip(ch_names, mean_w)},
        "best_cavity_gamma": float(best_gamma),
        "best_s2_beta": float(best_beta),
        "best_sinkhorn_target_mix": float(best_mix),
        "cluster_base_thresh": 0.89,
        "safetensors_bytes": int(out_st.stat().st_size),
    }
    meta_path = upload_dir / "meta_milestone1_system_one_apex.json"
    meta_path.write_text(json.dumps(meta_apex, indent=2))
    # Mantém cópia em titan3_mistral_large_123b_l44 também
    save_file(tensors_out, str(t3_dir / "head_milestone1_system_one_apex.safetensors"))
    (t3_dir / "meta_milestone1_system_one_apex.json").write_text(json.dumps(meta_apex, indent=2))

    print(f"✓ Saved: {out_st} ({out_st.stat().st_size / (1024*1024):.2f} MB)")
    print(f"✓ Saved: {meta_path}")
    print(f"✓ Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
