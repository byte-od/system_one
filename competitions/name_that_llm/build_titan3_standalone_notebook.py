import json
from pathlib import Path

from build_dual_titan_hybrid_notebook import CELL2_SVC_PREFIT_VERBATIM

THIS_DIR = Path(__file__).resolve().parent


def _lines(code: str) -> list[str]:
    return [ln + "\n" for ln in code.strip("\n").split("\n")]


def build_titan3_standalone_notebook(sync_default: bool = False):
    mos_src = "\n".join(
        ln for ln in (THIS_DIR / "mos_streaming_engine.py").read_text().splitlines() if "from __future__" not in ln
    )

    nb = {
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": [],
    }

    c0 = {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(
            """# 👑 Mixture of Scars (MoS) Titan 3 Standalone: Mistral Large 2 (123B) L44 + SCache O(1)
Zero-leak pure-PyTorch FP16 layer-streaming (`micro_bsz=8`, `60k` tokens/GPU, `< 4.2 GiB` peak VRAM on T4)."""
        ),
    }

    c1_code = (
        """import gc, glob, json, math, os, re, shutil, time
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

T0 = time.time()
INPUT_DIR = "/kaggle/input"
W_SVC = 1.0
W_DEBERTA = 0.25
TEMP_TITAN3 = 5.0
W_TITAN3_PLAIN = 0.25
W_TITAN3_TWIN = 0.35
TWIN = dict(thresh=0.89, min_size=2, max_size=18)

LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
K = len(LABELS)

def find(pattern):
    hits = sorted(glob.glob(f"{INPUT_DIR}/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]

def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def vote(name, P, w, rows=slice(None)):
    LOGIT[rows] += w * np.log(np.clip(P, 1e-9, 1.0))
    print(f"{name}: weight {w:.3f}, done after {time.time() - T0:.0f}s", flush=True)

def extract_tricompass_style_features(texts):
    feats = []
    for t in texts:
        t = str(t)
        n = max(len(t), 1); words = t.split(); nw = max(len(words), 1)
        sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]; ns = max(len(sents), 1)
        lines = [l for l in t.split("\\n") if l.strip()]; nl = max(len(lines), 1)
        feats.append([
            t.count(",")/n, t.count(":")/n, t.count(";")/n, t.count("—")/n, t.count(" — ")/n,
            t.count("-")/n, t.count("--")/n, t.count('"')/n, t.count("'")/n, t.count("`")/n, t.count("```")/nw,
            t.count("(")/n, t.count(")")/n, t.count("[")/n, t.count("]")/n, t.count("{")/n, t.count("}")/n,
            t.count("#")/n, t.count("*")/n, (t.count("\\n- ") + t.count("\\n* "))/nl,
            sum(c.isdigit() for c in t)/n, sum(c.isupper() for c in t)/n,
            sum(w[0].isupper() for w in words if len(w) > 0)/nw,
            t.count("\\n")/nw, t.count("\\n\\n")/nw,
            sum(len(w) for w in words)/nw, nw/ns, len(set(w.lower() for w in words))/nw
        ])
    return np.asarray(feats, dtype=np.float32)

class TestTimeSCache:
    def __init__(self, slot_dim=32, resonance_scale=0.15):
        self.slot_dim = slot_dim
        self.resonance_scale = resonance_scale
        self.proj = None

    def apply_scache_resonance(self, X, groups, twin_mask=None):
        if self.proj is None:
            rng = np.random.RandomState(42)
            Q, _ = np.linalg.qr(rng.randn(X.shape[1], self.slot_dim))
            self.proj = Q.astype(np.float32)
        Z = X @ self.proj
        X_out = X.copy()
        ugs = np.unique(groups[twin_mask]) if (twin_mask is not None and twin_mask.any()) else []
        for g in ugs:
            m = (groups == g)
            if m.sum() > 1:
                dz = Z[m] - Z[m].mean(axis=0, keepdims=True)
                W_scar = (dz.T @ dz) / float(m.sum())
                X_out[m] += self.resonance_scale * ((dz @ W_scar.T) @ self.proj.T)
        return X_out
"""
        + "\n"
        + mos_src
        + """
train = pd.read_csv(find("train.csv"))
test = pd.read_csv(find("test.csv"))
test["text"] = test["text"].fillna("").astype(str)
y = train["model"].map({l: i for i, l in enumerate(LABELS)}).values
LOGIT = np.zeros((len(test), K))
print("train", train.shape, "test", test.shape, "gpus", torch.cuda.device_count(), flush=True)
"""
    )

    c1 = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(c1_code)}
    c2 = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CELL2_SVC_PREFIT_VERBATIM}

    c3_code = """from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

if len(test) >= 2:
    Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(test.text.tolist())
    S_mat = (Xt @ Xt.T).toarray().astype(np.float32)
    del Xt
    D = np.clip(1.0 - S_mat, 0.0, None)
    np.fill_diagonal(D, 0.0)
    Z_lk = linkage(squareform(D, checks=False), method="average")
    g_raw = fcluster(Z_lk, TWIN["thresh"], criterion="distance")
    next_id = int(g_raw.max()) + 1
    groups = g_raw.copy()
    for cid in np.unique(g_raw):
        idx = np.where(g_raw == cid)[0]
        if len(idx) > TWIN["max_size"]:
            subD = D[np.ix_(idx, idx)]
            subZ = linkage(squareform(subD, checks=False), method="average")
            for th in np.linspace(TWIN["thresh"] - 0.005, 0.05, 165):
                sg = fcluster(subZ, th, criterion="distance")
                if np.bincount(sg).max() <= TWIN["max_size"]:
                    for sc in np.unique(sg):
                        groups[idx[sg == sc]] = next_id; next_id += 1
                    break
            else:
                sg = fcluster(subZ, int(np.ceil(len(idx) / TWIN["max_size"])), criterion="maxclust")
                for sc in np.unique(sg):
                    groups[idx[sg == sc]] = next_id; next_id += 1
    del D, Z_lk
else:
    groups = np.ones(len(test), dtype=int)

size = np.bincount(groups)[groups]
twin_rows = (size >= TWIN["min_size"]) & (size <= TWIN["max_size"])
print("topic clusters:", (size == 1).sum(), "passages alone,", (twin_rows).sum(), "in clusters of 2 to 18", flush=True)
"""
    c3 = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(c3_code)}

    c4_code = """from transformers import AutoModel, AutoTokenizer

ck = os.path.dirname(find("deberta_v3_large/meta.json"))
if ck.startswith("/kaggle/input"):
    shutil.copytree(ck, "/tmp/deberta_v3_large", dirs_exist_ok=True)
    ck = "/tmp/deberta_v3_large"
meta = json.load(open(os.path.join(ck, "meta.json")))
assert meta["labels"] == LABELS, "label order differs from checkpoint head"
etok = AutoTokenizer.from_pretrained(os.path.join(ck, "encoder"))
enc = AutoModel.from_pretrained(os.path.join(ck, "encoder"), dtype=torch.float32).to("cuda:0").eval()
head = load_file(os.path.join(ck, "head.safetensors"))
W, b = head["weight"].float().numpy(), head["bias"].float().numpy()
ids = etok(test.text.tolist(), truncation=True, max_length=meta["max_len"])["input_ids"]
order = np.argsort([len(x) for x in ids], kind="stable")
reps = [None] * len(ids)
with torch.no_grad():
    for i in range(0, len(ids), 32):
        b_idx = order[i:i + 32]
        b_tok = etok.pad({"input_ids": [ids[j] for j in b_idx]}, padding=True, return_tensors="pt").to("cuda:0")
        h = enc(**b_tok).last_hidden_state
        m = b_tok["attention_mask"][..., None]
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        for j, r in zip(b_idx, pooled.cpu().numpy()):
            reps[j] = r
        del b_tok, h, m, pooled
vote("deberta", softmax(np.stack(reps) @ W.T + b), W_DEBERTA)
del enc, head, reps
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""
    c4 = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(c4_code)}

    c5_code = """head_mistral_file = find("head_tricompass_mistral_large_123b_l44_scache.safetensors")
head_mistral = load_file(head_mistral_file)
mistral_backbone_dir = os.path.dirname(find("model-00001-of-00013.safetensors"))
tok_mistral = AutoTokenizer.from_pretrained(mistral_backbone_dir, use_fast=True)
pad_id_m = tok_mistral.pad_token_id if tok_mistral.pad_token_id is not None else tok_mistral.eos_token_id
raw_ids_m = tok_mistral(test.text.tolist(), truncation=True, max_length=512, padding=False)["input_ids"]
mos_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]

Z_mistral = extract_mos_nf4_streaming(
    mistral_backbone_dir, raw_ids_m, pad_id_m, mos_devices,
    padding_side=tok_mistral.padding_side, tag="Titan3-Mistral123B",
    micro_bsz=8, max_tokens_per_dev=60000,
)
assert np.isfinite(Z_mistral).all(), "Non-finite features in Z_mistral"

style_raw = extract_tricompass_style_features(test.text.tolist())
style_normed = (style_raw - head_mistral["style_scaler.mean"].float().numpy()) / head_mistral["style_scaler.scale"].float().numpy()
X_tri_mistral = np.hstack([Z_mistral, style_normed])
X_sc_mistral = TestTimeSCache(slot_dim=32, resonance_scale=0.15).apply_scache_resonance(X_tri_mistral, groups, twin_rows)

X_plain = (X_sc_mistral - head_mistral["scaler_plain.mean"].float().numpy()) / head_mistral["scaler_plain.scale"].float().numpy()
logits_plain = (X_plain @ head_mistral["plain.weight"].float().numpy().T + head_mistral["plain.bias"].float().numpy()) / TEMP_TITAN3
vote("titan3_mistral_123b_plain", softmax(logits_plain), W_TITAN3_PLAIN)

if twin_rows.any():
    X_c_raw = X_sc_mistral.copy()
    for g in np.unique(groups[twin_rows]):
        m = (groups == g)
        if m.sum() > 1:
            X_c_raw[m] -= X_c_raw[m].mean(axis=0, keepdims=True)
    X_c = (X_c_raw - head_mistral["scaler_centered.mean"].float().numpy()) / head_mistral["scaler_centered.scale"].float().numpy()
    logits_c = (X_c[twin_rows] @ head_mistral["centered.weight"].float().numpy().T + head_mistral["centered.bias"].float().numpy()) / TEMP_TITAN3
    vote("titan3_mistral_123b_twin", softmax(logits_c), W_TITAN3_TWIN, rows=twin_rows)

del Z_mistral, X_tri_mistral, X_sc_mistral, X_plain, head_mistral
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""
    c5 = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(c5_code)}

    c6_code = """def sinkhorn_damped(P, target_mix=0.50, iters=300):
    P_s = np.clip(P, 1e-12, 1.0)
    P_s = P_s / P_s.sum(1, keepdims=True)
    emp = P_s.sum(0)
    unif = np.full(K, len(P_s) / float(K))
    target = np.maximum((1.0 - target_mix) * emp + target_mix * unif, 1e-6)
    c = np.ones(K, dtype=np.float64)
    for _ in range(iters):
        R = P_s * c
        R /= R.sum(1, keepdims=True)
        c *= target / np.maximum(R.sum(0), 1e-12)
    R = P_s * c
    return R / R.sum(1, keepdims=True)

assert np.isfinite(LOGIT).all(), "Nonfinite ensemble logits"
P = softmax(LOGIT)
if len(test) >= 50:
    P = sinkhorn_damped(P, target_mix=0.50, iters=300)
assert np.isfinite(P).all(), "Nonfinite ensemble probability"
sub = pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[P.argmax(1)]})
assert len(sub) == len(test) and sub["model"].isin(LABELS).all()
sub.to_parquet("submission.parquet", index=False)
print(sub["model"].value_counts().to_string())
print(f"done in {time.time() - T0:.0f}s")
"""
    c6 = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(c6_code)}

    nb["cells"] = [c0, c1, c2, c3, c4, c5, c6]
    targets = ["name_that_llm_submission_titan3.ipynb"]
    if sync_default:
        targets.append("name_that_llm_submission.ipynb")
    for fname in targets:
        p = THIS_DIR / fname
        p.write_text(json.dumps(nb, indent=1))
        print("✓ Generated:", p)


if __name__ == "__main__":
    build_titan3_standalone_notebook()
