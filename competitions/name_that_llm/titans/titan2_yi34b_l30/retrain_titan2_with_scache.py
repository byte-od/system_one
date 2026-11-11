#!/usr/bin/env python3
"""
👑 RE-TREINO DO TITAN 2 (YI-34B MoS L30) CO-TREINADO COM S-CACHE & CICATRIZES
Ajusta os classificadores (Plain Head + Centered Twin) DIRETAMENTE sobre as
representações projetadas pelo SCache O(1), alinhando o Titan 2 com a mesma
formulação matemática do Titan 3 para a máxima sinergia na Opção C.
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file, load_file
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, log_loss

# Canonical official competition labels
LABELS = [
    'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
    'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
    'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
    'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'
]
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)

def sinkhorn(P: np.ndarray, iters: int = 300) -> np.ndarray:
    c = np.ones(K)
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= target / R.sum(0)
    R = P * c
    return R / R.sum(1, keepdims=True)

class TestTimeSCache:
    """Memória associativa O(1) Scar-Cache para projeção e subtração de viés intra-prompt."""
    def __init__(self, slot_dim: int = 32, resonance_scale: float = 0.15):
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
                dz = Z_g - z_bar
                W_scar = (dz.T @ dz) / float(m.sum())
                res = (dz @ W_scar.T) @ self.proj.T
                X_out[m] += self.resonance_scale * res
        return X_out

def center_groups_delta_origin(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    Xc = X.copy()
    for g in np.unique(groups):
        m = (groups == g)
        if m.sum() > 1:
            Xc[m] -= Xc[m].mean(axis=0, keepdims=True)
    return Xc

def main():
    print("=" * 80)
    print("👑 RE-TREINO DO TITAN 2 (YI-34B L30) CO-TREINADO COM S-CACHE & CICATRIZES")
    print("   • Alinhando tensores de 7,196D com a formulação SCache O(1)")
    print("   • Otimização para máxima sinergia na Opção C (Dual-Titan)")
    print("=" * 80)

    base_dir = Path("/Users/juliocesarreisfilho/Projects/scar-cache/competitions/name_that_llm")
    data_file = base_dir / "data" / "train.csv"
    titan2_dir = base_dir / "titans" / "titan2_yi34b_l30"

    df_train = pd.read_csv(data_file)
    n_samples = len(df_train)
    y_train = df_train["model"].map(L2I).values
    prompts = df_train["prompt_id"].values
    print(f"✓ Carregadas {n_samples} amostras de treino de {data_file}")

    # 1. Carrega features pré-extraídas de 7,168D
    feat_file = titan2_dir / "features_yi34b_l30_train_full.npy"
    assert feat_file.exists(), f"Arquivo de features {feat_file} não encontrado!"
    Z_yi = np.load(feat_file)
    print(f"✓ Features Yi-34B L30 carregadas: {Z_yi.shape}")

    # 2. Carrega ou extrai estilometria 28D
    head_orig_file = titan2_dir / "head_tricompass_yi34b_l30.safetensors"
    head_orig = load_file(str(head_orig_file))
    
    # Importa extrator de estilometria
    sys.path.insert(0, str(base_dir))
    from test_clean_composition import extract_tricompass_style_features
    style_raw = extract_tricompass_style_features(df_train["text"].tolist())
    style_mean = head_orig["style_scaler.mean"].float().numpy()
    style_scale = head_orig["style_scaler.scale"].float().numpy()
    style_normed = (style_raw - style_mean) / style_scale

    # 3. Constrói Tri-Compass Matrix (7,168 + 28 = 7,196D)
    X_tri_raw = np.hstack([Z_yi, style_normed])
    print(f"✓ Matriz Tri-Compass bruta: {X_tri_raw.shape}")

    # 4. Aplica S-Cache de Cicatrizes (memória associativa O(1))
    print("\n[SCache] Aplicando ressonância de cicatrizes O(1) no espaço Tri-Compass...")
    t0_sc = time.time()
    scache = TestTimeSCache(slot_dim=32, resonance_scale=0.15)
    X_scache_plain = scache.apply_scache_resonance(X_tri_raw, prompts)
    X_scache_centered = center_groups_delta_origin(X_scache_plain, prompts)
    print(f"✓ S-Cache aplicado em {time.time()-t0_sc:.2f}s!")

    # 5. Validação 5-Fold Stratified Group K-Fold
    print("\n[Validação 5-Fold Stratified Group K-Fold por prompt_id]...")
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_probs = np.zeros((n_samples, K), dtype=np.float32)
    fold_accs = []
    fold_sink_accs = []
    fold_losses = []

    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X_scache_plain, y_train, groups=prompts)):
        # Plain Head sobre S-Cache
        scaler_p = StandardScaler()
        X_tr_p = scaler_p.fit_transform(X_scache_plain[trn_idx])
        X_va_p = scaler_p.transform(X_scache_plain[val_idx])

        clf_p = LogisticRegression(C=0.1, max_iter=600, random_state=42)
        clf_p.fit(X_tr_p, y_train[trn_idx])
        p_val_plain = clf_p.predict_proba(X_va_p)

        # Centered Twin Head sobre S-Cache
        scaler_c = StandardScaler()
        X_tr_c = scaler_c.fit_transform(X_scache_centered[trn_idx])
        X_va_c = scaler_c.transform(X_scache_centered[val_idx])

        clf_c = LogisticRegression(C=0.1, max_iter=600, random_state=42)
        clf_c.fit(X_tr_c, y_train[trn_idx])
        p_val_twin = clf_c.predict_proba(X_va_c)

        # Ensemble Co-Treinado: 80% Plain + 20% Centered Twin
        p_val_ens = 0.80 * p_val_plain + 0.20 * p_val_twin
        p_val_ens /= p_val_ens.sum(axis=1, keepdims=True)
        oof_probs[val_idx] = p_val_ens

        acc_raw = accuracy_score(y_train[val_idx], np.argmax(p_val_ens, axis=1))
        p_val_sink = sinkhorn(p_val_ens)
        acc_sink = accuracy_score(y_train[val_idx], np.argmax(p_val_sink, axis=1))
        loss_sink = log_loss(y_train[val_idx], p_val_sink)

        fold_accs.append(acc_raw)
        fold_sink_accs.append(acc_sink)
        fold_losses.append(loss_sink)

        print(f"  Fold {fold+1}: Raw Acc = {acc_raw*100:.2f}% | Soft Sinkhorn Acc = {acc_sink*100:.2f}% | LogLoss = {loss_sink:.4f}")

    oof_sinkhorn = sinkhorn(oof_probs)
    mean_raw = np.mean(fold_accs)
    mean_sink = np.mean(fold_sink_accs)
    overall_sink_acc = accuracy_score(y_train, np.argmax(oof_sinkhorn, axis=1))
    overall_loss = log_loss(y_train, oof_sinkhorn)

    print("\n" + "=" * 80)
    print("👑 RESULTADOS TITAN 2 CO-TREINADO COM S-CACHE:")
    print(f"   Acurácia Média Folds (Raw):           {mean_raw*100:.2f}%")
    print(f"   Acurácia Média Folds (Soft Sinkhorn): {mean_sink*100:.2f}%")
    print(f"   Acurácia Global OOF (Soft Sinkhorn):  {overall_sink_acc*100:.2f}%")
    print(f"   Log Loss Global OOF (Soft Sinkhorn):  {overall_loss:.4f}")
    print("=" * 80)

    # 6. Refit 100% Full Training Set
    print("\n[Refit 100% Full Dataset] Ajustando classificadores finais alinhados com S-Cache...")
    final_scaler_p = StandardScaler()
    X_p_full = final_scaler_p.fit_transform(X_scache_plain)
    final_clf_p = LogisticRegression(C=0.1, max_iter=600, random_state=42)
    final_clf_p.fit(X_p_full, y_train)

    final_scaler_c = StandardScaler()
    X_c_full = final_scaler_c.fit_transform(X_scache_centered)
    final_clf_c = LogisticRegression(C=0.1, max_iter=600, random_state=42)
    final_clf_c.fit(X_c_full, y_train)

    # 7. Serialização Safetensors e Metadados
    out_safetensors = titan2_dir / "head_tricompass_yi34b_l30_scache.safetensors"
    out_meta = titan2_dir / "meta_tricompass_yi34b_scache.json"

    tensors = {
        "plain.weight": torch.from_numpy(final_clf_p.coef_).float().contiguous(),
        "plain.bias": torch.from_numpy(final_clf_p.intercept_).float().contiguous(),
        "scaler_plain.mean": torch.from_numpy(final_scaler_p.mean_).float().contiguous(),
        "scaler_plain.scale": torch.from_numpy(final_scaler_p.scale_).float().contiguous(),
        "centered.weight": torch.from_numpy(final_clf_c.coef_).float().contiguous(),
        "centered.bias": torch.from_numpy(final_clf_c.intercept_).float().contiguous(),
        "scaler_centered.mean": torch.from_numpy(final_scaler_c.mean_).float().contiguous(),
        "scaler_centered.scale": torch.from_numpy(final_scaler_c.scale_).float().contiguous(),
        "style_scaler.mean": torch.from_numpy(style_mean).float().contiguous(),
        "style_scaler.scale": torch.from_numpy(style_scale).float().contiguous(),
    }
    save_file(tensors, str(out_safetensors))
    print(f"✓ Safetensors S-Cache salvo: {out_safetensors} ({out_safetensors.stat().st_size / (1024**2):.2f} MB)")

    meta_payload = {
        "titan_id": "titan2_yi34b_l30_scache",
        "model_id": "unsloth/yi-34b-bnb-4bit",
        "golden_slice_layers": 30,
        "hidden_dimension": 7168,
        "tricompass_dimension": 7196,
        "classes": LABELS,
        "scache_slot_dim": 32,
        "scache_resonance_scale": 0.15,
        "mean_raw_accuracy": float(mean_raw),
        "mean_sinkhorn_accuracy": float(mean_sink),
        "overall_oof_sinkhorn_accuracy": float(overall_sink_acc),
        "overall_oof_logloss": float(overall_loss),
        "timestamp": time.time(),
    }
    with open(out_meta, "w") as f:
        json.dump(meta_payload, f, indent=2)
    print(f"✓ Metadados S-Cache salvos: {out_meta}")
    print("\n👑 TITAN 2 RE-TREINADO COM S-CACHE CONCLUÍDO COM SUCESSO ABSOLUTO!")

if __name__ == "__main__":
    main()
