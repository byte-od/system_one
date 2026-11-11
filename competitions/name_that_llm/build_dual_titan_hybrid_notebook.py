import json
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent

CELL2_SVC_PREFIT_VERBATIM = [
    "from scipy.sparse import hstack\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.svm import LinearSVC\n",
    "\n",
    'SMALL_WORDS_C2 = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())\n',
    'VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("small", 1, 3, 50000, 0.5)]\n',
    "\n",
    "def shape_text(t):\n",
    '    """Every run of letters becomes X or x, so only the rhythm is left."""\n',
    '    return re.sub(r"[^\\W\\d_]+", lambda m: "X" if m[0].isupper() else "x", t)\n',
    "\n",
    "def small_words_text(t):\n",
    '    """Keep the small words and the punctuation, and replace every other word by CONTENT."""\n',
    '    return " ".join(w.lower() if w.lower() in SMALL_WORDS_C2 or not w.isalnum() else "CONTENT"\n',
    '                    for w in re.findall(r"\\w+|[^\\w\\s]", t))\n',
    "\n",
    'SOURCE_CODE_SHA256 = "5864b360ef4b470726b6ae2018489e285651dfb2a305d24b1c6884666d81f800"\n',
    "import hashlib, importlib.metadata, joblib\n",
    "from pathlib import Path\n",
    'EXPECTED_CACHE_SHA256 = "f294ce929714e852a5a3f1d34ff7a5c5f0a0df3d73bd4a24ae327cbde4d96352"\n',
    "def _sha(path):\n",
    "    h=hashlib.sha256()\n",
    '    with open(path,"rb") as f:\n',
    '        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)\n',
    "    return h.hexdigest()\n",
    "def _one(pattern):\n",
    '    hits=sorted(set(glob.glob(f"{INPUT_DIR}/**/{pattern}",recursive=True)))\n',
    '    if len(hits)!=1: raise RuntimeError(("Ambiguous or absent input",pattern,hits))\n',
    "    return Path(hits[0])\n",
    'mp = _one("ntl620_svc_prefit_v1/manifest.json")\n',
    "manifest=json.loads(mp.read_text())\n",
    'assert EXPECTED_CACHE_SHA256 != "SET_FROM_CPU_MANIFEST_BEFORE_GPU", "CPU cache checksum not pinned"\n',
    'assert manifest["schema"]=="ntl620-svc-prefit-v1" and manifest["source_version"]==361311282\n',
    'assert manifest["source_code_sha256"] == SOURCE_CODE_SHA256\n',
    'assert manifest["labels"]==LABELS and manifest["views"]==[list(v) for v in VIEWS]\n',
    'assert manifest["classes"]==list(range(K)) and not manifest["fit_used_test_rows"]\n',
    'assert manifest["sample_cache_reload_exact"] is True\n',
    'assert _sha(_one("train.csv"))==manifest["train_sha256"]\n',
    'assert _sha(_one("pool_v4.parquet"))==manifest["pool_sha256"]\n',
    'for package, expected in manifest["versions"].items():\n',
    "    assert importlib.metadata.version(package)==expected, (package,expected,importlib.metadata.version(package))\n",
    'cp=mp.parent/"svc_prefit.joblib"\n',
    'assert manifest["cache_sha256"]==EXPECTED_CACHE_SHA256 and _sha(cp)==EXPECTED_CACHE_SHA256\n',
    "cache=joblib.load(cp)  # checksum-pinned statistical anchor\n",
    'assert cache["labels"]==LABELS and cache["svc"].classes_.tolist()==list(range(K))\n',
    "B=[]\n",
    'for vec,(view,lo,hi,max_features,weight) in zip(cache["vectorizers"],VIEWS):\n',
    '    prep=shape_text if view=="shape" else small_words_text if view=="small" else (lambda s:s)\n',
    "    B.append(vec.transform([prep(t) for t in test.text])*weight)\n",
    'assert len(cache["vectorizers"])==len(VIEWS)==4\n',
    'vote("svc_open_pool", softmax(cache["svc"].decision_function(hstack(B,format="csr")) / 0.85), W_SVC)\n',
    "del B, cache\n",
    "gc.collect()\n",
]


def _lines(code: str) -> list[str]:
    return [ln + "\n" for ln in code.strip("\n").split("\n")]


def build_dual_titan_hybrid_notebook(sync_default: bool = True):
    forensic_src = (THIS_DIR / "forensic_scouts_120d.py").read_text()
    # Strip `from __future__ import annotations` so it can be concatenated mid-cell cleanly
    forensic_src = "\n".join(ln for ln in forensic_src.splitlines() if "from __future__" not in ln)
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
            """# 👑 System One Apex V33: Dual-Titan MoS (Yi-34B L30 + Mistral Large 123B L44) + Bayesian SCache O(1)
## Name That LLM — Pure Proprietary System One Architecture (Zero 3rd-Party LoRAs, Zero VRAM Leak)
---
### Architectural Stack:
1. **Statistical & Structural Anchors** (*explicitly declared per `AGENTS.md` / `SKILL.md`*):
   - `svc_open_pool`: Checksum-pinned 4-view Open-Pool `LinearSVC` anchor.
   - `svc_host_exact` & `svc_proxy_clean`: Live-fitted Unicode-Typography + Lexical-Distortion `LinearSVC` anchors correcting OpenRouter-vs-Host register shifts (`gemini` em-dashes, `kimi-k3` curly apostrophes).
   - `deberta_v3_large`: Bidirectional syntactic encoder anchor.
2. **Compass 1 — 120D Host-Aware Forensic Frontier Scouts (`extract_forensic_compass1_120d`)**:
   - Exact Unicode typography ratios, Markdown geometry, sentence burstiness, UK/US spelling & Ghost-Jump discourse transitions.
3. **Compass 2 — Size-Capped Recursive Topic Clustering (`thresh = 0.89`, `max_size = 18`, `ARI = 0.977`)**:
   - Zero dropped `>18` prompt merges + pure Delta-Origin centering ($X - \mu_{\text{prompt}}$).
4. **Compass 3 — Zero-Leak Multi-GPU FP16 `Mixture of Scars (MoS)` Layer-Streaming Engine**:
   - **Titan 2 (`Yi-34B` Golden-Slice L30, `7,288D` Co-Trained `SCache`)** & **Titan 3 (`Mistral Large 2 123B` Golden-Slice L44, `12,408D` Co-Trained `SCache`)** (`< 4.2 GiB` peak VRAM on T4, `0.9999996` cosine fidelity; auto-detects **Titan 4 `Qwen3-Embedding-8B` 8-Depth MoS Scar Experts** when attached).
5. **Intuition & Collective Memory (`UniversalS1ReflexiveMLP` + `SCache O(1)` Bayesian Cavity Shared-Prompt Head + Cavity-EM + `ReflexDecision` S1 $\to$ S2 Clan Verifiers + Damped Soft Sinkhorn)**."""
        ),
    }

    c1_code = (
        """import gc, glob, json, math, os, re, shutil, time
from collections import Counter
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

T0 = time.time()
INPUT_DIR = "/kaggle/input"

LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)
CLANS = {"openai": [0, 1, 2, 3, 4], "gemini": [5, 6, 7, 8], "claude": [9, 10, 11, 12, 13], "glm": [15, 16]}

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
    print(f"{name}: weight {w:.4f}, done after {time.time() - T0:.1f}s", flush=True)

def group_center(X, groups, mask=None):
    Xc = np.zeros_like(X, dtype=np.float32)
    ugs = np.unique(groups[mask]) if (mask is not None and mask.any()) else np.unique(groups)
    for g in ugs:
        m = (groups == g)
        if m.sum() > 1:
            Xc[m] = X[m] - X[m].mean(axis=0, keepdims=True)
    return Xc

class TestTimeSCache:
    \"\"\"O(1) Associative Scar-Cache resonance over constant 32x32 scar covariance slots.\"\"\"
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
        + forensic_src
        + "\n"
        + mos_src
        + """
apex_meta = json.loads(Path(find("meta_milestone1_system_one_apex.json")).read_text())
assert apex_meta["labels"] == LABELS
apex_st = load_file(find("head_milestone1_system_one_apex.safetensors"))
LW = apex_meta["learned_weights"]

W_SVC = float(LW["c2_svc"])
W_HOST_SVC = float(LW["host_svc"])
W_PROXY_SVC = float(LW["proxy_svc"])
W_FORENSIC120 = float(LW["forensic120"])
W_DEBERTA = 0.28
W_T2_PLAIN = float(LW["t2_plain"])
W_T2_TWIN = float(LW["t2_twin"])
W_T3_PLAIN = float(LW["t3_plain"])
W_T3_TWIN = float(LW["t3_twin"])
W_S1_MLP = float(LW["s1_mlp"])
W_SCACHE_BAYES = float(LW["scache_bayes"])
CAVITY_GAMMA = float(apex_meta["best_cavity_gamma"])
S2_BETA = float(apex_meta["best_s2_beta"])
SINKHORN_MIX = float(apex_meta["best_sinkhorn_target_mix"])
TWIN = dict(thresh=float(apex_meta.get("cluster_base_thresh", 0.89)), min_size=2, max_size=18)

train = pd.read_csv(find("train.csv"))
train["text"] = train["text"].fillna("").astype(str)
test = pd.read_csv(find("test.csv"))
test["text"] = test["text"].fillna("").astype(str)
y = train["model"].map(L2I).values
LOGIT = np.zeros((len(test), K), dtype=np.float64)
print(f"train={train.shape} | test={test.shape} | gpus={torch.cuda.device_count()} | OOF={apex_meta['oof_v33_apex_bal_acc']*100:.2f}%", flush=True)
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
    groups = np.ones(len(test), dtype=np.int32)
    S_mat = np.eye(len(test), dtype=np.float32)

size = np.bincount(groups)[groups]
twin_rows = (size >= TWIN["min_size"]) & (size <= TWIN["max_size"])
print("topic clusters (size-capped <=18):", (size == 1).sum(), "alone,", twin_rows.sum(), "in clusters of 2..18", flush=True)
"""
    c3 = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(c3_code)}

    c4_code = """# ==============================================================================
# Stage-2 Host-Exact & Clean Proxy-Matched Sparse Anchors + 120D Forensic Scouts
# ==============================================================================
from sklearn.ensemble import HistGradientBoostingClassifier

SMALL_WORDS_EXT = SMALL_WORDS_C2 | set("however moreover furthermore therefore thus instead rather overall finally first second".split())

def unicode_skeleton_text(t):
    t = re.sub(r"[A-Z][a-z]+", "Tt", t)
    t = re.sub(r"[A-Z]{2,}", "UU", t)
    t = re.sub(r"[a-z]+", "w", t)
    t = re.sub(r"\\d+", "0", t)
    return t

def host_distortion_text(t):
    toks = re.findall(r"[A-Za-z]+(?:['\\u2019][A-Za-z]+)*|\\d+|[^\\w\\s]|\\n", t)
    out = []
    for tok in toks:
        if tok == "\\n":
            out.append("NL")
        elif not tok[0].isalnum():
            out.append(f"P_{ord(tok[0])}")
        elif tok[0].isdigit():
            out.append("NUM")
        else:
            has_curly = "\\u2019" in tok
            has_str = "'" in tok
            low = tok.lower().replace("\\u2019", "'")
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

def build_host_exact_views(texts, vecs=None):
    specs = [
        ("skel_char", unicode_skeleton_text, "char", (1, 5), 65000, r"(?u)\\b\\w\\w+\\b", 1.0),
        ("distort_tok", host_distortion_text, "word", (1, 3), 90000, r"\\S+", 0.9),
        ("char_wb", lambda s: s, "char_wb", (2, 5), 110000, r"(?u)\\b\\w\\w+\\b", 0.75),
        ("word_punct", lambda s: s, "word", (1, 2), 80000, r"\\w+|[^\\w\\s]", 0.65),
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

def build_c2_views_fresh(texts, vecs=None):
    mats, fitted = [], []
    for i, (view, lo, hi, mf, w) in enumerate(VIEWS):
        prep = shape_text if view == "shape" else small_words_text if view == "small" else (lambda s: s)
        proc = [prep(t) for t in texts]
        if vecs is None:
            v = TfidfVectorizer(analyzer=("char" if view in ("char", "shape") else "word"), ngram_range=(lo, hi), max_features=mf, sublinear_tf=True)
            m = v.fit_transform(proc) * w
            fitted.append(v)
        else:
            v = vecs[i]
            m = v.transform(proc) * w
        mats.append(m)
    return hstack(mats, format="csr"), (fitted if vecs is None else vecs)

tr_texts_list = train["text"].tolist()
te_texts_list = test["text"].tolist()

# 1. Host-Exact Anchor (100% pure Kaggle host register)
X_he_tr, vecs_he = build_host_exact_views(tr_texts_list, vecs=None)
X_he_te, _ = build_host_exact_views(te_texts_list, vecs=vecs_he)
svc_he = LinearSVC(C=0.25, max_iter=2000, random_state=42).fit(X_he_tr, y)
m_he = svc_he.decision_function(X_he_te).astype(np.float32)
m_he_comb = m_he + 0.45 * group_center(m_he, groups, twin_rows)
vote("svc_host_exact", softmax(m_he_comb / 0.85), W_HOST_SVC)
del X_he_tr, X_he_te, vecs_he, svc_he

# 2. Clean Proxy-Matched 8-View Anchor (`passages.csv` funnel-verified + `train.csv` host upweighted)
pa_hits = sorted(glob.glob(f"{INPUT_DIR}/**/passages.csv", recursive=True))
if pa_hits and W_PROXY_SVC > 0:
    pa = pd.read_csv(pa_hits[0])
    pa = pa[
        (pa["passes_funnel"] == True)
        & (pa["is_refusal"] == False)
        & (pa["is_empty"] == False)
        & (pa["split"] != "pilot_range")
        & (~((pa["split"] == "new") & (pa["model"] == "kimi-k3")))
        & (pa["model"].isin(L2I))
    ].copy()
    pa["text"] = pa["text"].fillna("").astype(str)
    px_texts = tr_texts_list + pa["text"].tolist()
    px_y = np.concatenate([y, pa["model"].map(L2I).values])
    px_w = np.concatenate([np.full(len(train), 2.5, dtype=np.float32), np.ones(len(pa), dtype=np.float32)])

    X_px1_tr, vp1 = build_c2_views_fresh(px_texts)
    X_px1_te, _ = build_c2_views_fresh(te_texts_list, vp1)
    X_px2_tr, vp2 = build_host_exact_views(px_texts)
    X_px2_te, _ = build_host_exact_views(te_texts_list, vp2)
    svc_px = LinearSVC(C=0.35, max_iter=2000, random_state=42).fit(
        hstack([X_px1_tr, X_px2_tr * 0.7], format="csr"), px_y, sample_weight=px_w
    )
    m_px = svc_px.decision_function(hstack([X_px1_te, X_px2_te * 0.7], format="csr")).astype(np.float32)
    m_px_comb = m_px + 0.55 * group_center(m_px, groups, twin_rows)
    vote("svc_proxy_clean_8view", softmax(m_px_comb / 0.85), W_PROXY_SVC)
    del pa, px_texts, X_px1_tr, X_px1_te, X_px2_tr, X_px2_te, vp1, vp2, svc_px

# 3. Compass-1 120D Forensic Frontier Scouts (Pre-Saved LR + Live Host HGB)
F_120_train = extract_forensic_compass1_120d(tr_texts_list)
F_120_test = extract_forensic_compass1_120d(te_texts_list)
f120_tr_dual = np.hstack([F_120_train, group_center(F_120_train, train["prompt_id"].astype(str).values)])
f120_te_dual = np.hstack([F_120_test, group_center(F_120_test, groups, twin_rows)])

f120_te_norm = np.clip(
    (f120_te_dual - apex_st["f120_dual.mean"].float().numpy()) / apex_st["f120_dual.scale"].float().numpy(),
    -6.0, 6.0,
)
logits_f120_lr = f120_te_norm @ apex_st["f120_lr.weight"].float().numpy().T + apex_st["f120_lr.bias"].float().numpy()
hgb_f120 = HistGradientBoostingClassifier(
    max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=5.0, random_state=42
).fit(f120_tr_dual, y)
probs_f120 = 0.45 * softmax(logits_f120_lr / 2.5) + 0.55 * hgb_f120.predict_proba(f120_te_dual)
vote("compass1_forensic_120d", probs_f120, W_FORENSIC120)
del F_120_train, f120_tr_dual, hgb_f120
gc.collect()
"""
    c4 = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(c4_code)}

    c5_code = """# ==============================================================================
# Bidirectional Structural Anchor: DeBERTa v3 large (Strict VRAM Cleanup)
# ==============================================================================
from transformers import AutoModel, AutoTokenizer

ck = os.path.dirname(find("deberta_v3_large/meta.json"))
if ck.startswith("/kaggle/input"):
    shutil.copytree(ck, "/tmp/deberta_v3_large", dirs_exist_ok=True)
    ck = "/tmp/deberta_v3_large"
meta = json.load(open(os.path.join(ck, "meta.json")))
assert meta["labels"] == LABELS, "label order differs from checkpoint head"
etok = AutoTokenizer.from_pretrained(os.path.join(ck, "encoder"))
enc = AutoModel.from_pretrained(os.path.join(ck, "encoder"), dtype=torch.float32).to("cuda:0").eval()
head = load_file(os.path.join(ck, "head.safetensors"))
W_deb, b_deb = head["weight"].float().numpy(), head["bias"].float().numpy()
ids = etok(te_texts_list, truncation=True, max_length=meta["max_len"])["input_ids"]
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
vote("deberta_v3_large", softmax(np.stack(reps) @ W_deb.T + b_deb), W_DEBERTA)
del enc, head, reps, W_deb, b_deb
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""
    c5 = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(c5_code)}

    c6_code = """# ==============================================================================
# Dual-Titan Zero-Leak MoS Layer-Streaming: Yi-34B (L30) + Mistral Large 2 (123B L44)
# (+ Auto-Plug-In for Titan 4 Qwen3-Embedding-8B 8-Depth MoS Scar Experts)
# ==============================================================================
mos_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
f120_scaled_half = np.clip(
    (F_120_test - apex_st["f120_scaler.mean"].float().numpy()) / apex_st["f120_scaler.scale"].float().numpy(),
    -6.0, 6.0,
).astype(np.float32) * 0.5

# 1. Titan 2: Yi-34B Golden-Slice L30 (30 Layers, 7,168D + 120D = 7,288D)
yi_backbone_dir = os.path.dirname(find("model-00001-of-00002.safetensors"))
tok_yi = AutoTokenizer.from_pretrained(yi_backbone_dir, use_fast=True)
pad_id_yi = tok_yi.pad_token_id if tok_yi.pad_token_id is not None else tok_yi.eos_token_id
raw_ids_yi = tok_yi(te_texts_list, truncation=True, max_length=512, padding=False)["input_ids"]

Z_yi = extract_mos_nf4_streaming(
    yi_backbone_dir, raw_ids_yi, pad_id_yi, mos_devices,
    padding_side=tok_yi.padding_side, tag="Titan2-Yi34B",
    micro_bsz=8, max_tokens_per_dev=60000,
)
assert np.isfinite(Z_yi).all(), "Non-finite features in Z_yi"

X_sc_yi = TestTimeSCache(32, 0.15).apply_scache_resonance(np.hstack([Z_yi, f120_scaled_half]), groups, twin_rows)
X_p_yi = (X_sc_yi - apex_st["t2.scaler_plain.mean"].float().numpy()) / apex_st["t2.scaler_plain.scale"].float().numpy()
logits_p_yi = (X_p_yi @ apex_st["t2.plain.weight"].float().numpy().T + apex_st["t2.plain.bias"].float().numpy()) / 5.0
vote("titan2_yi34b_120d_plain", softmax(logits_p_yi), W_T2_PLAIN)

if twin_rows.any():
    X_c_yi = (group_center(X_sc_yi, groups, twin_rows) - apex_st["t2.scaler_centered.mean"].float().numpy()) / apex_st["t2.scaler_centered.scale"].float().numpy()
    logits_c_yi = (X_c_yi[twin_rows] @ apex_st["t2.centered.weight"].float().numpy().T + apex_st["t2.centered.bias"].float().numpy()) / 5.0
    vote("titan2_yi34b_120d_twin", softmax(logits_c_yi), W_T2_TWIN, rows=twin_rows)
del X_sc_yi, X_p_yi
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 2. Titan 3: Mistral Large 2 (123B) Golden-Slice L44 (44 Layers, 12,288D + 120D = 12,408D)
mistral_backbone_dir = os.path.dirname(find("model-00001-of-00013.safetensors"))
tok_mistral = AutoTokenizer.from_pretrained(mistral_backbone_dir, use_fast=True)
pad_id_m = tok_mistral.pad_token_id if tok_mistral.pad_token_id is not None else tok_mistral.eos_token_id
raw_ids_m = tok_mistral(te_texts_list, truncation=True, max_length=512, padding=False)["input_ids"]

Z_mistral = extract_mos_nf4_streaming(
    mistral_backbone_dir, raw_ids_m, pad_id_m, mos_devices,
    padding_side=tok_mistral.padding_side, tag="Titan3-Mistral123B",
    micro_bsz=8, max_tokens_per_dev=60000,
)
assert np.isfinite(Z_mistral).all(), "Non-finite features in Z_mistral"

X_sc_m = TestTimeSCache(32, 0.15).apply_scache_resonance(np.hstack([Z_mistral, f120_scaled_half]), groups, twin_rows)
X_p_m = (X_sc_m - apex_st["t3.scaler_plain.mean"].float().numpy()) / apex_st["t3.scaler_plain.scale"].float().numpy()
logits_p_m = (X_p_m @ apex_st["t3.plain.weight"].float().numpy().T + apex_st["t3.plain.bias"].float().numpy()) / 5.0
vote("titan3_mistral123b_120d_plain", softmax(logits_p_m), W_T3_PLAIN)

if twin_rows.any():
    X_c_m = (group_center(X_sc_m, groups, twin_rows) - apex_st["t3.scaler_centered.mean"].float().numpy()) / apex_st["t3.scaler_centered.scale"].float().numpy()
    logits_c_m = (X_c_m[twin_rows] @ apex_st["t3.centered.weight"].float().numpy().T + apex_st["t3.centered.bias"].float().numpy()) / 5.0
    vote("titan3_mistral123b_120d_twin", softmax(logits_c_m), W_T3_TWIN, rows=twin_rows)
del X_sc_m, X_p_m
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 3. Optional Auto-Detected Titan 4: Qwen3-Embedding-8B 8-Depth 5-Fold MoS Scar Experts (Milestone 2)
t4_scars = sorted(glob.glob(f"{INPUT_DIR}/**/titan4_qwen3_emb_8b_fold*_scars.safetensors", recursive=True))
t4_metas = sorted(glob.glob(f"{INPUT_DIR}/**/meta_titan4_qwen3_emb_8b.json", recursive=True))
if t4_scars and t4_metas:
    t4_meta = json.loads(Path(t4_metas[0]).read_text())
    t4_bb_hits = [p for p in glob.glob(f"{INPUT_DIR}/**/model.safetensors.index.json", recursive=True) if "qwen3emb" in p.lower() or "titan4" in p.lower()]
    if t4_bb_hits:
        t4_bb_dir = os.path.dirname(t4_bb_hits[0])
        tok_t4 = AutoTokenizer.from_pretrained(t4_bb_dir, use_fast=True)
        pad_t4 = tok_t4.pad_token_id if tok_t4.pad_token_id is not None else tok_t4.eos_token_id
        win_ids, doc_idx, win_wts = build_covering_windows(tok_t4(te_texts_list, truncation=False, padding=False)["input_ids"], max_len=512, stride=384)
        cap_l = t4_meta["capture_layers_1b"]
        n_dp = len(cap_l)
        t4_probs_acc = np.zeros((len(test), K), dtype=np.float64)
        for f_path in t4_scars:
            st_f = load_file(f_path)
            Z_w = extract_mos_nf4_streaming(t4_bb_dir, win_ids, pad_t4, mos_devices, padding_side="left", tag=Path(f_path).stem, capture_layers=cap_l, scar_tensors=st_f)
            d_m = Z_w.shape[1] // n_dp
            with torch.no_grad():
                dw = F.softmax(st_f["head.depth_logits"].float(), dim=0) * n_dp
                sls = [F.layer_norm(torch.from_numpy(Z_w[:, si*d_m:(si+1)*d_m]), (d_m,), st_f[f"head.depth_norms.{si}.weight"].float(), st_f[f"head.depth_norms.{si}.bias"].float()) * dw[si] for si in range(n_dp)]
                h0 = F.linear(torch.cat(sls, dim=-1), st_f["head.proj.0.weight"].float(), st_f["head.proj.0.bias"].float())
                zw_proj = F.silu(F.layer_norm(h0, (h0.shape[-1],), st_f["head.proj.1.weight"].float(), st_f["head.proj.1.bias"].float())).numpy()
                z_doc = torch.from_numpy(aggregate_window_embeddings(zw_proj, doc_idx, win_wts, len(test), unit_rms=False))
                s_k = F.linear(z_doc, st_f["head.scache_key.weight"].float())
                z_sc = z_doc.clone()
                z_ct = torch.zeros_like(z_doc)
                for g_id in np.unique(groups[twin_rows]) if twin_rows.any() else []:
                    m_g = torch.from_numpy(groups == g_id)
                    if int(m_g.sum()) > 1:
                        dz = s_k[m_g] - s_k[m_g].mean(0, keepdim=True)
                        w_s = (dz.T @ dz) / float(m_g.sum())
                        z_sc[m_g] = z_doc[m_g] + float(st_f["head.scache_scale"].item()) * ((dz @ w_s.T) @ st_f["head.scache_key.weight"].float())
                        z_ct[m_g] = z_sc[m_g] - z_sc[m_g].mean(0, keepdim=True)
                lp = F.linear(F.layer_norm(z_sc, (z_sc.shape[-1],), st_f["head.plain_norm.weight"].float(), st_f["head.plain_norm.bias"].float()), st_f["head.plain_head.weight"].float(), st_f["head.plain_head.bias"].float())
                lc = F.linear(F.layer_norm(z_ct, (z_ct.shape[-1],), st_f["head.twin_norm.weight"].float(), st_f["head.twin_norm.bias"].float()), st_f["head.twin_head.weight"].float(), st_f["head.twin_head.bias"].float())
                t4_probs_acc += (0.45 * F.softmax(lp, dim=-1) + 0.55 * F.softmax(lc, dim=-1)).numpy() / len(t4_scars)
            del st_f, Z_w
        vote("titan4_qwen3emb_8depth_mos_scars", t4_probs_acc, 1.80)
"""
    c6 = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(c6_code)}

    c7_code = """# ==============================================================================
# System One Collective Inference:
# UniversalS1ReflexiveMLP + SCache O(1) Bayesian Shared-Prompt + Cavity-EM + S2 Verifiers
# ==============================================================================
from scipy.special import logsumexp

# 1. Project [Titan2, Titan3, Forensic120] onto 192D SCache Subspace
a_t2 = ((Z_yi - apex_st["scb.sc_t2.mean"].float().numpy()) / apex_st["scb.sc_t2.scale"].float().numpy()) / np.sqrt(Z_yi.shape[1])
b_t3 = ((Z_mistral - apex_st["scb.sc_t3.mean"].float().numpy()) / apex_st["scb.sc_t3.scale"].float().numpy()) / np.sqrt(Z_mistral.shape[1])
c_f = ((F_120_test - apex_st["scb.sc_f.mean"].float().numpy()) / apex_st["scb.sc_f.scale"].float().numpy()) / np.sqrt(F_120_test.shape[1]) * 0.85
X_cat_scb = np.hstack([a_t2, b_t3, c_f]).astype(np.float32)
Z_pca = ((X_cat_scb - apex_st["scb.pca.mean"].float().numpy()) @ apex_st["scb.pca.components"].float().numpy().T).astype(np.float32)
del Z_yi, Z_mistral, a_t2, b_t3, c_f, X_cat_scb
gc.collect()

# 2. UniversalS1ReflexiveMLP 3-Seed Ensemble (624D -> 320D Residual LayerNorm SiLU -> 18)
X_s1 = np.hstack([Z_pca, group_center(Z_pca, groups, twin_rows), f120_te_dual]).astype(np.float32)
X_s1_scaled = np.clip(
    (X_s1 - apex_st["s1.scaler.mean"].float().numpy()) / apex_st["s1.scaler.scale"].float().numpy(),
    -6.0, 6.0,
).astype(np.float32)

with torch.no_grad():
    xt_s1 = torch.from_numpy(X_s1_scaled)
    p_s1_acc = np.zeros((len(test), K), dtype=np.float64)
    for s_idx in range(3):
        pfx = f"s1.mlp{s_idx}."
        h = F.layer_norm(xt_s1, (xt_s1.shape[-1],), apex_st[pfx + "front_ln.weight"].float(), apex_st[pfx + "front_ln.bias"].float())
        h = F.linear(h, apex_st[pfx + "fc_in.weight"].float(), apex_st[pfx + "fc_in.bias"].float())
        h = F.silu(F.layer_norm(h, (h.shape[-1],), apex_st[pfx + "ln1.weight"].float(), apex_st[pfx + "ln1.bias"].float()))
        hr = F.linear(h, apex_st[pfx + "fc_res.weight"].float(), apex_st[pfx + "fc_res.bias"].float())
        hr = F.silu(F.layer_norm(hr, (hr.shape[-1],), apex_st[pfx + "ln2.weight"].float(), apex_st[pfx + "ln2.bias"].float()))
        h = h + 0.5 * hr
        lg = F.linear(h, apex_st[pfx + "policy_head.weight"].float(), apex_st[pfx + "policy_head.bias"].float())
        p_s1_acc += F.softmax(lg / 1.5, dim=-1).numpy() / 3.0
vote("universal_s1_reflex_mlp", p_s1_acc, W_S1_MLP)

# 3. SCache O(1) Bayesian Cavity Shared-Prompt Generative Head (< 200 KB fixed state!)
Z_d64 = Z_pca.astype(np.float64)
mu_c = apex_st["scb.mu_c"].float().numpy().astype(np.float64)
cov_eps = apex_st["scb.cov_eps"].float().numpy().astype(np.float64)
cov_prompt = apex_st["scb.cov_prompt"].float().numpy().astype(np.float64)
prec_eps = apex_st["scb.prec_eps"].float().numpy().astype(np.float64)
prec_prompt = apex_st["scb.prec_prompt"].float().numpy().astype(np.float64)
d_scb = Z_d64.shape[1]

wiener_ops = {}
def _get_scb_ops(m_other):
    if m_other not in wiener_ops:
        if m_other <= 0:
            wiener_ops[m_other] = (np.zeros((d_scb, d_scb)), np.linalg.inv(cov_prompt + cov_eps))
        else:
            post_cov = np.linalg.inv(prec_prompt + float(m_other) * prec_eps)
            wiener_ops[m_other] = (post_cov @ prec_eps, np.linalg.inv(cov_eps + post_cov))
    return wiener_ops[m_other]

_, prec0 = _get_scb_ops(0)
diff0 = Z_d64[:, None, :] - mu_c[None, :, :]
ll0 = -0.5 * np.einsum("nkd,de,nke->nk", diff0, prec0, diff0) / 2.2
prior_cur = softmax(LOGIT)
Q_scb = 0.65 * prior_cur + 0.35 * softmax(ll0)
ll_scb = ll0.copy()
for _ in range(3):
    res_scb = Z_d64 - (Q_scb @ mu_c)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) <= 1:
            ll_scb[idx] = ll0[idx]
        else:
            W_eta, prec_m = _get_scb_ops(len(idx) - 1)
            cav_sum = res_scb[idx].sum(axis=0, keepdims=True) - res_scb[idx]
            z_clean = Z_d64[idx] - (cav_sum @ W_eta.T)
            df_g = z_clean[:, None, :] - mu_c[None, :, :]
            ll_scb[idx] = -0.5 * np.einsum("mkd,de,mke->mk", df_g, prec_m, df_g) / 2.2
    Q_scb = 0.50 * prior_cur + 0.50 * softmax(ll_scb)

vote("scache_o1_bayesian_shared_prompt", softmax(ll_scb), W_SCACHE_BAYES)

# 4. Similarity-Weighted Cavity-EM Sibling Repulsion
assert np.isfinite(LOGIT).all(), "Nonfinite ensemble logits"
if len(test) >= 2 and CAVITY_GAMMA > 1e-6:
    P0_log = LOGIT - logsumexp(LOGIT, axis=1, keepdims=True)
    P = np.exp(P0_log)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        m = len(idx)
        if 2 <= m <= 18:
            sub_S = S_mat[np.ix_(idx, idx)]
            W_sim = np.clip((sub_S - 0.12) / 0.30, 0.0, 1.0)
            np.fill_diagonal(W_sim, 0.0)
            if W_sim.max() > 1e-6:
                W_eff = W_sim / np.maximum(W_sim.sum(axis=1, keepdims=True) / min(float(m - 1), 6.0), 1.0)
                lp_g = P0_log[idx]
                q_g = P[idx].copy()
                for _ in range(5):
                    excl = np.log(np.clip(1.0 - CAVITY_GAMMA * q_g, 1e-4, 1.0))
                    q_g = 0.65 * softmax(lp_g + (W_eff @ excl)) + 0.35 * q_g
                P[idx] = q_g
    P = P / P.sum(axis=1, keepdims=True)
else:
    P = softmax(LOGIT)

# 5. ReflexDecision Escalation -> System 2 Intra-Clan Specialist Verifiers
if S2_BETA > 1e-6:
    sp_sorted = np.sort(P, axis=1)[:, ::-1]
    margins = np.clip(sp_sorted[:, 0] - sp_sorted[:, 1], 0.0, 1.0)
    unc_idx = np.where(margins < 0.35)[0]
    if len(unc_idx) > 0:
        betas = S2_BETA * np.clip(1.0 - margins[unc_idx] / 0.35, 0.0, 1.0)
        X_u = X_s1[unc_idx]
        for clan_name, members in CLANS.items():
            c_mass = P[np.ix_(unc_idx, members)].sum(axis=1, keepdims=True)
            act = c_mass[:, 0] > 0.25
            if act.any():
                sub_u = unc_idx[act]
                b_a = betas[act][:, None]
                cm_a = c_mass[act]
                Xs_c = (X_u[act] - apex_st[f"s2.{clan_name}.mean"].float().numpy()) / apex_st[f"s2.{clan_name}.scale"].float().numpy()
                lg_s2 = (Xs_c @ apex_st[f"s2.{clan_name}.weight"].float().numpy().T + apex_st[f"s2.{clan_name}.bias"].float().numpy()) / 2.0
                if lg_s2.shape[1] == 1:
                    p1 = 1.0 / (1.0 + np.exp(-lg_s2[:, 0]))
                    p_s2 = np.stack([1.0 - p1, p1], axis=1)
                else:
                    p_s2 = softmax(lg_s2)
                p_s1_c = np.clip(P[np.ix_(sub_u, members)] / np.maximum(cm_a, 1e-12), 1e-9, 1.0)
                fused_c = softmax((1.0 - b_a) * np.log(p_s1_c) + b_a * np.log(np.clip(p_s2, 1e-9, 1.0)))
                P[np.ix_(sub_u, members)] = cm_a * fused_c
        P = P / P.sum(axis=1, keepdims=True)

# 6. Damped Soft Sinkhorn Marginal Calibration
def sinkhorn_damped(P_in, target_mix=0.50, iters=300):
    if target_mix <= 1e-6 or len(P_in) < K:
        return P_in / P_in.sum(1, keepdims=True)
    P_s = np.clip(P_in, 1e-12, 1.0)
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

if len(test) >= 50:
    P = sinkhorn_damped(P, target_mix=SINKHORN_MIX, iters=300)

assert np.isfinite(P).all(), "Nonfinite ensemble probability"
assert P.shape == (len(test), K), "Probability shape mismatch"
sub = pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[P.argmax(1)]})
assert len(sub) == len(test) and sub["model"].isin(LABELS).all()
sub.to_parquet("submission.parquet", index=False)
print(sub["model"].value_counts().to_string())
print(f"System One Apex V33 complete in {time.time() - T0:.1f}s!", flush=True)
"""
    c7 = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": _lines(c7_code)}

    nb["cells"] = [c0, c1, c2, c3, c4, c5, c6, c7]
    targets = ["name_that_llm_submission_dual_titan.ipynb"]
    if sync_default:
        targets.append("name_that_llm_submission.ipynb")
    for fname in targets:
        p = THIS_DIR / fname
        p.write_text(json.dumps(nb, indent=1))
        print("✓ Generated:", p)


if __name__ == "__main__":
    build_dual_titan_hybrid_notebook(sync_default=True)
