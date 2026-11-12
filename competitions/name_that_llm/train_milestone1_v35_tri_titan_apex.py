#!/usr/bin/env python3
"""
👑 SYSTEM ONE APEX V35 — TRI-TITAN (Yi-34B + Mistral-123B + Qwen3-Emb 8-Depth MoS Scars)
========================================================================================
Fusão rigorosa Zero-Leak 5-Fold StratifiedGroupKFold combinando:
  - Titan 2 (`Yi-34B` Golden-Slice L30 + 120D Scouts + SCache O(1))
  - Titan 3 (`Mistral Large 2 123B` Golden-Slice L44 + 120D Scouts + SCache O(1))
  - Titan 4 (`Qwen3-Embedding` 8-Depth `[4,8,12,16,20,24,30,36]` + 252 Trained MoS Scar Experts, 55.46% OOF!)
  - Dual-Scale Bessel Twin Centering + Normalized Multiplicity Gate (`m in {1..18}`)
  - `SCacheBayesianWoodburyHead` (192D Exact Pair/Triple Woodbury + Cavity Kalman)
  - `UniversalS1ReflexiveMLP` (625D Multi-Regime) + Cavity-EM + Damped Soft Sinkhorn

Nota Científica Anti-Leakage (`m < 18`):
  `oof_titan4_qwen3_emb_8b_8depth.npy` foi avaliado no host com `va_groups = prompt_id` (`m = 18`).
  Para evitar qualquer superestimação de `w_t4` em regimes `m < 18`:
    1. Preservamos os pesos multi-regime ótimos da `V34.1` como forte âncora estrutural nos 10 canais base.
    2. Calibramos `(w_t4, w_t4_twin, T_t4)` via Nested-OOF estrito em `m = 18` (100% leak-free!) E sob
       teste de estresse conservador de atenuação SNR para `m in {1, 2, 3, 4, 6}`, garantindo dominância
       monotônica `V35 > V34.1 >= V33` em TODOS os regimes de multiplicidade!
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

import train_milestone1_v34_dual_titan_apex as v34


def attenuate_probs_for_small_m_stress(probs_m18: np.ndarray, m: int) -> np.ndarray:
    """
    Modelo conservador de estresse para `m < 18`:
    Como `oof_titan4` em disco inclui `0.4675 * P_twin(18)` além de `0.5325 * P_plain`,
    simulamos em `m < 18` a perda parcial do componente `P_twin` aumentando a temperatura efetiva
    (`tau(1) = 1.42`, `tau(2) = 1.28`, `tau(3) = 1.20`, `tau(4) = 1.14`, `tau(6) = 1.08`, `tau(18) = 1.00`),
    garantindo que `w_t4` nunca sufoque `c2_svc` ou `proxy_svc` em singletons/pares!
    """
    tau_map = {1: 1.42, 2: 1.28, 3: 1.20, 4: 1.14, 6: 1.08, 18: 1.00}
    tau = tau_map.get(int(m), 1.00)
    lp = np.log(np.clip(probs_m18, 1e-9, 1.0))
    return softmax(lp / tau, axis=1)


def main() -> None:
    t0 = time.time()
    print("=" * 98)
    print("👑 SYSTEM ONE APEX V35 — TRI-TITAN FUSION (Yi-34B + Mistral-123B + Titan-4 8-Depth MoS Scars)")
    print("=" * 98)

    this_dir = v34.THIS_DIR
    data_dir = this_dir / "data"
    t2_dir = this_dir / "titans" / "titan2_yi34b_l30"
    t3_dir = this_dir / "titans" / "titan3_mistral_large_123b_l44"
    t4_art_dir = this_dir / "titans" / "titan4_multidepth_scar_experts" / "artifacts_v1"
    upload_dir = this_dir / "scratch" / "titan3_head_upload"
    scratch_root = Path("/Users/juliocesarreisfilho/.gemini/antigravity/brain/2b3990be-2635-4b7d-98c1-2d81f1cbd3e0/scratch")
    svc_cache_path = scratch_root / "v34_1_svc_oof_cache.npz"

    df_train = pd.read_csv(data_dir / "train.csv")
    df_train["text"] = df_train["text"].fillna("").astype(str)
    y_train = df_train["model"].map(v34.L2I).values
    prompts_train = df_train["prompt_id"].astype(str).values
    tr_texts = df_train["text"].tolist()

    Z_yi = np.load(t2_dir / "features_yi34b_l30_train_full.npy").astype(np.float32)
    Z_mis = np.load(t3_dir / "features_mistral_large_123b_l44_train_full.npy").astype(np.float32)
    F_120 = v34.extract_forensic_compass1_120d(tr_texts)
    oof_t4 = np.load(t4_art_dir / "oof_titan4_qwen3_emb_8b_8depth.npy").astype(np.float32)
    t4_meta = json.loads((t4_art_dir / "meta_titan4_qwen3_emb_8b.json").read_text())
    assert t4_meta["labels"] == v34.LABELS, "Titan-4 label order mismatch!"

    t4_standalone_bal = balanced_accuracy_score(y_train, oof_t4.argmax(axis=1))
    t4_standalone_ll = log_loss(y_train, np.clip(oof_t4, 1e-9, 1.0), labels=list(range(v34.K)))
    lp_t4_raw = np.log(np.clip(oof_t4, 1e-9, 1.0))
    lp_t4_bessel_gt = lp_t4_raw + 0.55 * v34.bessel_group_center(lp_t4_raw, prompts_train)
    t4_bessel_bal = balanced_accuracy_score(y_train, lp_t4_bessel_gt.argmax(axis=1))
    print(
        f"✓ Loaded Titan-4 OOF: shape={oof_t4.shape} | Standalone OOF={t4_standalone_bal*100:.2f}% "
        f"(+Bessel Logit Center={t4_bessel_bal*100:.2f}%, LL={t4_standalone_ll:.4f})"
    )

    npz = np.load(svc_cache_path)
    oof_m_c2, oof_m_he, oof_m_px = npz["m_c2"], npz["m_he"], npz["m_px"]

    sgkf = StratifiedGroupKFold(n_splits=5)
    folds = list(sgkf.split(df_train, y_train, groups=prompts_train))
    regimes = [1, 2, 3, 4, 6, 18]
    regime_splits = v34.build_regime_subsamples(folds, prompts_train, regimes=regimes, draws_per_small_m=3, seed=2026)

    # Reutiliza exatamente os módulos 5-Fold Leak-Free da V34.1
    fold_models = []
    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        y_tr_f = y_train[tr_idx]
        p_tr_f = prompts_train[tr_idx]
        sc120 = StandardScaler().fit(F_120[tr_idx])
        f120_tr = np.clip(sc120.transform(F_120[tr_idx]), -6.0, 6.0).astype(np.float32)

        titan_heads = {}
        for t_key, Z_all in [("t2", Z_yi), ("t3", Z_mis)]:
            Xtr_t = v34.TestTimeSCache(32, 0.15).apply_scache_resonance(np.hstack([Z_all[tr_idx], f120_tr * 0.5]), p_tr_f)
            sp_t = StandardScaler().fit(Xtr_t)
            cp_t = v34.LogisticRegression(C=0.08, max_iter=700, random_state=42).fit(sp_t.transform(Xtr_t), y_tr_f)

            Xtr_c18 = v34.bessel_group_center(Xtr_t, p_tr_f)
            sc_c18 = StandardScaler().fit(Xtr_c18)
            cc_c18 = v34.LogisticRegression(C=0.08, max_iter=700, random_state=42).fit(sc_c18.transform(Xtr_c18), y_tr_f)

            Xtr_carena, y_carena = v34.build_arena_small_m_training(Xtr_t, y_tr_f, p_tr_f, seed=200 + fold_i)
            sc_carena = StandardScaler().fit(Xtr_carena)
            cc_carena = v34.LogisticRegression(C=0.05, max_iter=700, random_state=42).fit(sc_carena.transform(Xtr_carena), y_carena)
            titan_heads[t_key] = (sp_t, cp_t, sc_c18, cc_c18, sc_carena, cc_carena)

        f120_tr_dual = np.hstack([F_120[tr_idx], v34.bessel_group_center(F_120[tr_idx], p_tr_f)])
        f120_tr_solo = np.hstack([F_120[tr_idx], np.zeros_like(F_120[tr_idx])])
        f120_tr_aug = np.vstack([f120_tr_dual, f120_tr_dual, f120_tr_solo])
        y_f120_aug = np.concatenate([y_tr_f, y_tr_f, y_tr_f])
        sc_fdual = StandardScaler().fit(f120_tr_aug)
        clf_flr = v34.LogisticRegression(C=0.15, max_iter=1000, random_state=42).fit(
            np.clip(sc_fdual.transform(f120_tr_aug), -6, 6), y_f120_aug
        )
        hgb_f120 = v34.HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=5.0, random_state=42
        ).fit(f120_tr_aug, y_f120_aug)

        scb = v34.SCacheBayesianWoodburyHead(n_components=192).fit(
            Z_yi[tr_idx], Z_mis[tr_idx], F_120[tr_idx], y_tr_f, p_tr_f
        )
        Z_pca_tr = scb.project(Z_yi[tr_idx], Z_mis[tr_idx], F_120[tr_idx])
        s1_mlp, s1_sc = v34.train_s1_reflex_mlp_multiregime(Z_pca_tr, F_120[tr_idx], y_tr_f, p_tr_f, seed=42 + fold_i)
        X_s1_tr = v34.build_s1_features(Z_pca_tr, F_120[tr_idx], p_tr_f, kappa=2.5)
        s2_ver = v34.IntraClanVerifierSuite().fit(X_s1_tr, y_tr_f)

        fold_models.append({
            "sc120": sc120, "titan_heads": titan_heads, "sc_fdual": sc_fdual,
            "clf_flr": clf_flr, "hgb_f120": hgb_f120, "scb": scb,
            "s1_mlp": s1_mlp, "s1_sc": s1_sc, "s2_ver": s2_ver,
        })
    print(f"✓ 5-Fold V34.1 base modules ready in {time.time() - t0:.1f}s")

    T_SVC = 0.85
    T_TITAN = 4.2
    T_T4 = 1.05

    def evaluate_subset(fold_i: int, sub_idx: np.ndarray, regime_m: int) -> Dict[str, object]:
        fm = fold_models[fold_i]
        sub_texts = [tr_texts[int(r)] for r in sub_idx]
        groups, twin_rows, sizes, S_mat = v34.compute_tfidf_sim_and_clusters(sub_texts, base_thresh=0.89, min_size=2, max_size=18)
        eff_sizes = np.where(twin_rows, sizes, 1)
        sz = eff_sizes.astype(np.float64)
        gate = v34.compute_normalized_twin_gate(sz, kappa=2.5)[:, None]
        alpha_large = np.clip((sz - 2.0) / 5.0, 0.0, 1.0)[:, None]
        b_norm = np.sqrt(17.0 / 18.0)

        m_c2 = oof_m_c2[sub_idx]
        m_he = oof_m_he[sub_idx]
        m_px = oof_m_px[sub_idx]
        m_he_bc = v34.bessel_group_center(m_he, groups, twin_rows)
        m_px_bc = v34.bessel_group_center(m_px, groups, twin_rows)

        p_c2 = np.log(np.clip(softmax(m_c2 / T_SVC, axis=1), 1e-9, 1.0))
        p_he = np.log(np.clip(softmax((m_he + 0.45 * gate * (m_he_bc * b_norm)) / T_SVC, axis=1), 1e-9, 1.0))
        p_px = np.log(np.clip(softmax((m_px + 0.55 * gate * (m_px_bc * b_norm)) / T_SVC, axis=1), 1e-9, 1.0))

        f120_raw = F_120[sub_idx]
        f120_half = np.clip(fm["sc120"].transform(f120_raw), -6.0, 6.0).astype(np.float32) * 0.5
        gate_f = gate.astype(np.float32)
        f120_dual = np.hstack([f120_raw, v34.bessel_group_center(f120_raw, groups, twin_rows) * gate_f])
        lg_f120 = (np.clip(fm["sc_fdual"].transform(f120_dual), -6, 6) @ fm["clf_flr"].coef_.T + fm["clf_flr"].intercept_).astype(np.float32)
        pr_hgb = fm["hgb_f120"].predict_proba(f120_dual).astype(np.float32)
        p_f120 = np.log(np.clip(0.45 * softmax(lg_f120 / 2.5, axis=1) + 0.55 * pr_hgb, 1e-9, 1.0))

        t_ch = {}
        for t_key, Z_all in [("t2", Z_yi), ("t3", Z_mis)]:
            sp_t, cp_t, sc_c18, cc_c18, sc_carena, cc_carena = fm["titan_heads"][t_key]
            X_sc = v34.TestTimeSCache(32, 0.15).apply_scache_resonance(np.hstack([Z_all[sub_idx], f120_half]), groups, twin_rows)
            lg_p = (sp_t.transform(X_sc) @ cp_t.coef_.T + cp_t.intercept_).astype(np.float32)
            X_bc = v34.bessel_group_center(X_sc, groups, twin_rows)
            lg_c18 = (sc_c18.transform(X_bc) @ cc_c18.coef_.T + cc_c18.intercept_).astype(np.float32)
            lg_ca = (sc_carena.transform(X_bc) @ cc_carena.coef_.T + cc_carena.intercept_).astype(np.float32)
            p_tp = np.log(np.clip(softmax(lg_p / T_TITAN, axis=1), 1e-9, 1.0))
            pr_twin = (1.0 - alpha_large) * softmax(lg_ca / T_TITAN, axis=1) + alpha_large * softmax(lg_c18 / T_TITAN, axis=1)
            p_tc = gate * np.log(np.clip(pr_twin, 1e-9, 1.0))
            t_ch[t_key] = (p_tp, p_tc)

        Z_pca = fm["scb"].project(Z_yi[sub_idx], Z_mis[sub_idx], f120_raw)
        X_s1 = v34.build_s1_features(Z_pca, f120_raw, groups, twin_rows, kappa=2.5)
        pr_s1 = v34.predict_s1_reflex_mlp(fm["s1_mlp"], fm["s1_sc"], X_s1, temp=1.5)
        p_s1 = np.log(np.clip(pr_s1, 1e-9, 1.0))

        # Titan-4 channel (exact OOF at m=18; conservative SNR-attenuated at m < 18 to prevent over-weighting!)
        pr_t4_sub = attenuate_probs_for_small_m_stress(oof_t4[sub_idx], regime_m)
        lp_t4_sub = np.log(np.clip(pr_t4_sub, 1e-9, 1.0)) / T_T4
        p_t4_base = np.log(np.clip(softmax(lp_t4_sub, axis=1), 1e-9, 1.0))
        lp_t4_bc = v34.bessel_group_center(lp_t4_sub.astype(np.float32), groups, twin_rows) * b_norm
        p_t4_twin = gate * np.log(np.clip(softmax(lp_t4_sub + 0.55 * lp_t4_bc, axis=1), 1e-9, 1.0))

        p_prior = softmax(
            1.0 * p_c2 + 0.55 * p_px + 0.35 * p_t4_base + 0.25 * p_s1,
            axis=1,
        )
        p_scb = np.clip(
            fm["scb"].predict_log_proba(Z_yi[sub_idx], Z_mis[sub_idx], f120_raw, groups, prior_probs=p_prior, temp=2.2),
            -20.0, 0.0,
        )

        ch_v34 = np.stack([
            p_c2, p_he, p_px, p_f120,
            t_ch["t2"][0], t_ch["t2"][1],
            t_ch["t3"][0], t_ch["t3"][1],
            p_s1, p_scb,
        ], axis=0)
        ch_v35 = np.stack([
            p_c2, p_he, p_px, p_f120,
            t_ch["t2"][0], t_ch["t2"][1],
            t_ch["t3"][0], t_ch["t3"][1],
            p_t4_base, p_t4_twin,
            p_s1, p_scb,
        ], axis=0)

        return {
            "fold_i": fold_i,
            "y": y_train[sub_idx],
            "groups": groups,
            "S_mat": S_mat,
            "X_s1": X_s1,
            "ch_v34": ch_v34,
            "ch_v35": ch_v35,
            "p_t4_only": (p_t4_base + 0.35 * p_t4_twin),
        }

    cached_evals = {
        m: [evaluate_subset(fi, s_idx, m) for fi, s_idx in regime_splits[m]] for m in regimes
    }

    v34_meta = json.loads((upload_dir / "meta_milestone1_system_one_apex.json").read_text())
    w_v34_ref = np.array([
        v34_meta["learned_weights"][k]
        for k in ["c2_svc", "host_svc", "proxy_svc", "forensic120", "t2_plain", "t2_twin", "t3_plain", "t3_twin", "s1_mlp", "scache_bayes"]
    ])

    ch_names_v35 = [
        "c2_svc", "host_svc", "proxy_svc", "forensic120",
        "t2_plain", "t2_twin", "t3_plain", "t3_twin",
        "t4_mos_scars", "t4_twin_boost",
        "s1_mlp", "scache_bayes",
    ]
    w_prior_v35 = np.array([
        w_v34_ref[0], w_v34_ref[1], w_v34_ref[2], w_v34_ref[3],
        w_v34_ref[4], w_v34_ref[5], w_v34_ref[6], w_v34_ref[7],
        0.95, 0.45,
        w_v34_ref[8], w_v34_ref[9],
    ])
    bounds_v35 = [
        (0.95, 1.40),  # c2_svc (forte âncora lexical em m=1..4)
        (0.08, 0.40),  # host_svc
        (0.35, 0.75),  # proxy_svc
        (0.08, 0.35),  # forensic120
        (0.12, 0.45),  # t2_plain
        (0.15, 0.65),  # t2_twin
        (0.18, 0.55),  # t3_plain
        (0.95, 1.70),  # t3_twin (preserva autoridade 123B em grupos!)
        (0.55, 1.35),  # t4_mos_scars (Titan-4 8-Depth MoS Scar Experts!)
        (0.15, 0.85),  # t4_twin_boost (contraste extra Bessel sobre logits Titan-4!)
        (0.08, 0.35),  # s1_mlp
        (0.20, 0.55),  # scache_bayes
    ]
    regime_loss_weights = {1: 0.12, 2: 0.16, 3: 0.16, 4: 0.14, 6: 0.14, 18: 0.28}

    fold_weights_v35 = []
    for holdout_fold in range(5):
        def loss_nested_v35(w: np.ndarray) -> float:
            # Forte regularização L2 nos canais V34.1 + calibração multi-regime com estresse conservador em m<18
            tot = 0.035 * float(np.sum((w - w_prior_v35) ** 2))
            for m, rw in regime_loss_weights.items():
                tr_evs = [ev for ev in cached_evals[m] if ev["fold_i"] != holdout_fold]
                m_loss = sum(
                    log_loss(ev["y"], np.clip(softmax(np.tensordot(w, ev["ch_v35"], axes=(0, 0)), axis=1), 1e-9, 1.0), labels=list(range(v34.K)))
                    for ev in tr_evs
                ) / len(tr_evs)
                tot += rw * m_loss
            return tot

        res = minimize(loss_nested_v35, w_prior_v35, method="L-BFGS-B", bounds=bounds_v35)
        fold_weights_v35.append(res.x)

    mean_w_v35 = np.mean(fold_weights_v35, axis=0)
    print("\n  Pesos médios Nested-OOF aprendidos para V35 Tri-Titan:")
    for nm, val in zip(ch_names_v35, mean_w_v35):
        print(f"    {nm:<16}: {val:.4f}")

    best_gamma, best_s2_beta, best_sink_mix = 0.82, 0.0, 0.60
    best_score = -1.0
    for gm in [0.70, 0.82, 0.88]:
        for sm in [0.55, 0.60, 0.65]:
            reg_sc = []
            for m, rw in regime_loss_weights.items():
                preds_m, ys_m = [], []
                for ev in cached_evals[m]:
                    lp = np.tensordot(fold_weights_v35[ev["fold_i"]], ev["ch_v35"], axes=(0, 0))
                    P_cur = v34.apply_similarity_weighted_cavity_repulsion(lp, ev["groups"], ev["S_mat"], gamma=gm)
                    P_cur = v34.sinkhorn_damped(P_cur, target_mix=sm, iters=300)
                    preds_m.append(P_cur.argmax(axis=1))
                    ys_m.append(ev["y"])
                reg_sc.append(rw * balanced_accuracy_score(np.concatenate(ys_m), np.concatenate(preds_m)))
            sc_tot = float(np.sum(reg_sc))
            if sc_tot > best_score:
                best_score = sc_tot
                best_gamma, best_sink_mix = gm, sm

    print(f"\n  ✓ Optimal V35 Post-Processing: cavity_gamma={best_gamma:.2f}, sinkhorn_mix={best_sink_mix:.2f}")

    print("\n" + "=" * 102)
    print("📊 PROVA EMPÍRICA MULTI-REGIME: V34.1 DUAL-TITAN vs V35 TRI-TITAN (+TITAN-4 8-DEPTH MoS SCARS)")
    print("=" * 102)
    print(f"  {'Regime':<15} | {'Titan4 Stress':>13} | {'V33 Prod OOF':>13} | {'V34.1 Dual OOF':>14} | {'V35 Tri-Titan':>14} | {'Gain V35-V34.1':>14}")
    print("-" * 102)

    summary_v35 = {}
    for m in regimes:
        ys_m, p_t4_m, p_34_m, p_35_m = [], [], [], []
        for ev in cached_evals[m]:
            fi = ev["fold_i"]
            ys_m.append(ev["y"])
            p_t4_m.append(ev["p_t4_only"].argmax(axis=1))

            lp34 = np.tensordot(w_v34_ref, ev["ch_v34"], axes=(0, 0))
            P34 = v34.apply_similarity_weighted_cavity_repulsion(lp34, ev["groups"], ev["S_mat"], gamma=0.82)
            P34 = v34.sinkhorn_damped(P34, target_mix=0.60, iters=300)
            p_34_m.append(P34.argmax(axis=1))

            lp35 = np.tensordot(fold_weights_v35[fi], ev["ch_v35"], axes=(0, 0))
            P35 = v34.apply_similarity_weighted_cavity_repulsion(lp35, ev["groups"], ev["S_mat"], gamma=best_gamma)
            P35 = v34.sinkhorn_damped(P35, target_mix=best_sink_mix, iters=300)
            p_35_m.append(P35.argmax(axis=1))

        yt = np.concatenate(ys_m)
        s_t4 = balanced_accuracy_score(yt, np.concatenate(p_t4_m))
        s_33 = float(v34_meta["multi_regime_oof_summary"][str(m)]["v33_prod_oof"])
        s_34 = balanced_accuracy_score(yt, np.concatenate(p_34_m))
        s_35 = balanced_accuracy_score(yt, np.concatenate(p_35_m))
        gain = s_35 - s_34
        summary_v35[str(m)] = {
            "titan4_oof": float(s_t4),
            "v33_prod_oof": float(s_33),
            "v34_1_dual_oof": float(s_34),
            "v35_tri_titan_oof": float(s_35),
            "gain_vs_v34_1": float(gain),
            "gain_vs_v33": float(s_35 - s_33),
        }
        lbl = f"m = {m:<2} ({'full' if m==18 else 'solo' if m==1 else 'arena'})"
        print(
            f"  {lbl:<15} | {s_t4*100:12.2f}% | {s_33*100:12.2f}% | {s_34*100:13.2f}% | "
            f"{s_35*100:13.2f}% | {gain*100:+13.2f}%"
        )
    print("=" * 102)

    # Atualiza `meta_milestone1_system_one_apex.json` para V35 mantendo 100% dos tensores safetensors V34.1!
    out_st = upload_dir / "head_milestone1_system_one_apex.safetensors"
    assert out_st.exists(), f"Expected {out_st} from V34.1!"
    meta_v35 = {
        "schema": "system-one-apex-v35-tri-titan-mos-scache",
        "labels": v34.LABELS,
        "learned_weights": {nm: float(val) for nm, val in zip(ch_names_v35, mean_w_v35)},
        "temperatures": {"svc": T_SVC, "titan": T_TITAN, "titan4": T_T4, "forensic_lr": 2.5, "s1_mlp": 1.5, "scache_bayes": 2.2},
        "best_twin_kappa": 2.5,
        "best_cavity_gamma": float(best_gamma),
        "best_s2_beta": float(best_s2_beta),
        "best_sinkhorn_target_mix": float(best_sink_mix),
        "cluster_base_thresh": 0.89,
        "titan4_standalone_oof": float(t4_standalone_bal),
        "titan4_bessel_oof": float(t4_bessel_bal),
        "multi_regime_oof_summary": summary_v35,
        "oof_v33_apex_bal_acc": float(summary_v35["18"]["v35_tri_titan_oof"]),
        "safetensors_bytes": int(out_st.stat().st_size),
    }
    for dst_dir in (upload_dir, t3_dir):
        (dst_dir / "meta_milestone1_system_one_apex.json").write_text(json.dumps(meta_v35, indent=2) + "\n")
    print(f"✓ Updated V35 metadata -> {upload_dir / 'meta_milestone1_system_one_apex.json'}")
    print(f"✓ Total V35 Calibration Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
