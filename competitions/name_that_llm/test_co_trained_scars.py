#!/usr/bin/env python3
"""
🧪 TESTE DE CO-TREINAMENTO DE HEADS COM CICATRIZES (System One S-Cache)
Treina as cabeças (Plain + Centered Twin) DIRETAMENTE sobre as representações
transformadas pelo S-Cache, avaliando em 5-Fold Stratified Group K-Fold (2.699 passagens).
"""

import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, log_loss

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

def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def sinkhorn(P, iters=300):
    c = np.ones(K)
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= target / R.sum(0)
    R = P * c
    return R / R.sum(1, keepdims=True)

class TestTimeSCache:
    def __init__(self, slot_dim=32, resonance_scale=0.15):
        self.slot_dim = slot_dim
        self.resonance_scale = resonance_scale
        self.proj = None

    def fit_projection(self, in_dim):
        np.random.seed(42)
        Q, _ = np.linalg.qr(np.random.randn(in_dim, self.slot_dim))
        self.proj = Q.astype(np.float32)

    def apply_scache_resonance(self, X, groups):
        if self.proj is None:
            self.fit_projection(X.shape[1])
        Z = X @ self.proj
        X_out = X.copy()
        for g in np.unique(groups):
            m = (groups == g)
            if m.sum() > 1:
                Z_g = Z[m]
                z_bar = Z_g.mean(axis=0, keepdims=True)
                dz = Z_g - z_bar
                W_scar = (dz.T @ dz) / float(m.sum())
                res = (dz @ W_scar.T) @ self.proj.T
                X_out[m] += self.resonance_scale * res
        return X_out

def main():
    print("=" * 80)
    print("👑 CO-TREINAMENTO DE HEAD COM CICATRIZES / S-CACHE (5-FOLD GROUP CV)")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    titan2_dir = base_dir / "titans" / "titan2_yi34b_l30"

    df = pd.read_csv(data_dir / "train.csv")
    y = df["model"].map(L2I).values
    prompts = df["prompt_id"].values
    n = len(df)

    # Features
    feats = np.load(titan2_dir / "features_yi34b_l30_train_full.npy")
    head_yi = load_file(str(titan2_dir / "head_tricompass_yi34b_l30.safetensors"))

    # Estilometria
    from test_clean_composition import extract_tricompass_style_features
    style_raw = extract_tricompass_style_features(df["text"].tolist())
    style_mean = head_yi["style_scaler.mean"].float().numpy()
    style_scale = head_yi["style_scaler.scale"].float().numpy()
    style_normed = (style_raw - style_mean) / style_scale
    X_tri = np.hstack([feats, style_normed])

    # LinearSVC Matriz
    def shape_encode(t):
        return "".join("U" if c.isupper() else "l" if c.islower() else "d" if c.isdigit() else " " if c.isspace() else "." for c in t)
    
    c_word = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, max_features=30000).fit_transform(df["text"])
    c_char = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", sublinear_tf=True, max_features=50000).fit_transform(df["text"])
    shapes = [shape_encode(t) for t in df["text"]]
    c_shape = TfidfVectorizer(ngram_range=(2, 6), analyzer="char", sublinear_tf=True, max_features=25000).fit_transform(shapes)
    small_texts = [" ".join(w for w in t.lower().split() if w in SMALL_WORDS) for t in df["text"]]
    c_small = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, max_features=10000).fit_transform(small_texts)
    X_svc = hstack([c_word, c_char, c_shape, c_small]).tocsr()

    # Aplicação do S-Cache no dataset
    scache = TestTimeSCache(slot_dim=32, resonance_scale=0.15)
    X_scache = scache.apply_scache_resonance(X_tri, prompts)

    # Centramento Delta-Origem
    X_cent_raw = X_tri.copy()
    X_cent_scache = X_scache.copy()
    for g in np.unique(prompts):
        m = (prompts == g)
        if m.sum() > 1:
            X_cent_raw[m] -= X_cent_raw[m].mean(axis=0, keepdims=True)
            X_cent_scache[m] -= X_cent_scache[m].mean(axis=0, keepdims=True)

    sgkf = StratifiedGroupKFold(n_splits=5)
    
    oof_base = np.zeros((n, K), dtype=np.float32)
    oof_cotrained = np.zeros((n, K), dtype=np.float32)

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_tri, y, groups=prompts)):
        # LinearSVC
        clf_svc = LinearSVC(C=0.1, max_iter=2000, random_state=42).fit(X_svc[trn_idx], y[trn_idx])
        p_svc = softmax(clf_svc.decision_function(X_svc[val_idx]))

        # --- A) MODELO BASE (SEM CO-TREINO COM CICATRIZES) ---
        scaler_p_base = (X_tri[trn_idx].mean(0), X_tri[trn_idx].std(0) + 1e-8)
        X_tr_pb = (X_tri[trn_idx] - scaler_p_base[0]) / scaler_p_base[1]
        X_va_pb = (X_tri[val_idx] - scaler_p_base[0]) / scaler_p_base[1]
        clf_pb = LogisticRegression(C=0.1, max_iter=600, random_state=42).fit(X_tr_pb, y[trn_idx])
        p_pb = softmax(clf_pb.decision_function(X_va_pb))

        scaler_c_base = (X_cent_raw[trn_idx].mean(0), X_cent_raw[trn_idx].std(0) + 1e-8)
        X_tr_cb = (X_cent_raw[trn_idx] - scaler_c_base[0]) / scaler_c_base[1]
        X_va_cb = (X_cent_raw[val_idx] - scaler_c_base[0]) / scaler_c_base[1]
        clf_cb = LogisticRegression(C=0.1, max_iter=600, random_state=42).fit(X_tr_cb, y[trn_idx])
        p_cb = softmax(clf_cb.decision_function(X_va_cb))

        oof_base[val_idx] = (
            1.00 * np.log(np.clip(p_svc, 1e-9, 1))
            + 0.60 * np.log(np.clip(p_pb, 1e-9, 1))
            + 0.20 * np.log(np.clip(p_cb, 1e-9, 1))
        )

        # --- B) CO-TREINAMENTO (HEADS TREINADAS DIRETAMENTE SOBRE O S-CACHE) ---
        scaler_p_sc = (X_scache[trn_idx].mean(0), X_scache[trn_idx].std(0) + 1e-8)
        X_tr_psc = (X_scache[trn_idx] - scaler_p_sc[0]) / scaler_p_sc[1]
        X_va_psc = (X_scache[val_idx] - scaler_p_sc[0]) / scaler_p_sc[1]
        clf_psc = LogisticRegression(C=0.1, max_iter=600, random_state=42).fit(X_tr_psc, y[trn_idx])
        p_psc = softmax(clf_psc.decision_function(X_va_psc))

        scaler_c_sc = (X_cent_scache[trn_idx].mean(0), X_cent_scache[trn_idx].std(0) + 1e-8)
        X_tr_csc = (X_cent_scache[trn_idx] - scaler_c_sc[0]) / scaler_c_sc[1]
        X_va_csc = (X_cent_scache[val_idx] - scaler_c_sc[0]) / scaler_c_sc[1]
        clf_csc = LogisticRegression(C=0.1, max_iter=600, random_state=42).fit(X_tr_csc, y[trn_idx])
        p_csc = softmax(clf_csc.decision_function(X_va_csc))

        oof_cotrained[val_idx] = (
            1.00 * np.log(np.clip(p_svc, 1e-9, 1))
            + 0.60 * np.log(np.clip(p_psc, 1e-9, 1))
            + 0.20 * np.log(np.clip(p_csc, 1e-9, 1))
        )
        print(f"  Fold {fold+1}/5 concluído!")

    p_base_final = sinkhorn(softmax(oof_base))
    p_cotrained_final = sinkhorn(softmax(oof_cotrained))

    acc_base = accuracy_score(y, np.argmax(p_base_final, 1))
    ll_base = log_loss(y, p_base_final)

    acc_co = accuracy_score(y, np.argmax(p_cotrained_final, 1))
    ll_co = log_loss(y, p_cotrained_final)

    print("\n" + "=" * 80)
    print("🏆 RESULTADO DO CO-TREINAMENTO COM CICATRIZES:")
    print("=" * 80)
    print(f"  • Base (Head treinada sem cicatrizes):         {acc_base*100:.2f}% acurácia | Log Loss: {ll_base:.4f}")
    print(f"  • 🔥 Co-Treinado (Head treinada com S-Cache):  {acc_co*100:.2f}% acurácia | Log Loss: {ll_co:.4f}")
    print(f"  • Diferencial:                                 {acc_co*100 - acc_base*100:+.2f}% acurácia | {ll_co - ll_base:+.4f} Log Loss")
    print("=" * 80)

if __name__ == "__main__":
    main()
