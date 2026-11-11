#!/usr/bin/env python3
"""
👑 SYSTEM ONE — MILESTONE 2 & 3 (KAGGLE 2xT4 / COLAB A100 / LOCAL VALIDATION):
TWO-STAGE PROPRIETARY MoS RESIDUAL SCAR EXPERTS + 8-DEPTH MULTI-SLICE ENGINE
============================================================================
Alvo de Leaderboard Público: >= 0.740 - 0.755+ (Top 10 em `name-that-llm`).
100% Arquitetura Proprietária System One (`AGENTS.md` & `SKILL.md`):
- ZERO adaptadores LoRA comunitários / de terceiros.
- Especialistas de Cicatriz Residuais MoS (`dW_scar = scale * B @ A`, rank=16) em FP32 master
  injetados diretamente em `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`,
  exportados no formato `.scar_A` / `.scar_B` nativamente suportado pelo `extract_mos_nf4_streaming`!
- Suporta execução universal sem dependências frágeis:
  * Kaggle `2x T4` (offline, sem bitsandbytes): Pipeline Parallel `device_map="auto"` em `float16`
    com parâmetros Scar + Head em `float32` puro (zero overflow/underflow em SM 7.5!).
  * Colab `A100` (`--quant nf4` ou `--quant bf16`): NF4 4-bit ou BF16 nativo.
  * Local `CPU/MPS` (`--quant fp32` / `fp16`): validação determinística fim-a-fim e teste de paridade!
- Centralização Bessel `sqrt(m / (m - 1))` + Multiplicity Gate `w_twin(m) = 0.55 * (m - 1) / (m + 2)`
  + Amostragem Multi-Multiplicidade `m in {1, 2, 3, 6, 16}` no Estágio 2 (imune a grupos pequenos!).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

LABELS = [
    "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra",
    "gemini-3.1-pro-preview", "gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.8-flash",
    "claude-opus-4-6", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5",
    "qwen3.8-max", "glm-5.3", "glm-5.3-flash", "kimi-k3",
]
L2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)

CAPTURE_LAYERS_QWEN3_8B = [4, 8, 12, 16, 20, 24, 30, 36]
SCAR_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def resolve_capture_layers(n_layers: int, spec: str = "auto") -> List[int]:
    if spec != "auto":
        vals = [int(x.strip()) for x in spec.split(",") if x.strip()]
        return [v for v in vals if 1 <= v <= n_layers]
    if n_layers == 36:
        return list(CAPTURE_LAYERS_QWEN3_8B)
    k = min(8, n_layers)
    layers = sorted(set(max(1, int(round(i * n_layers / k))) for i in range(1, k + 1)))
    return layers


# ==============================================================================
# 1. Camada Proprietária MoS Residual Scar Expert (FP32 Master Precision)
# ==============================================================================
class MoSResidualScarLinear(nn.Module):
    """
    Envolve uma projeção base congelada (NF4, FP16 ou BF16, em qualquer GPU do `device_map`)
    com um Especialista de Cicatriz Residual de posto baixo mantido em FP32 Master:
      `y = base(x) + (alpha / r) * (drop(x.float()) @ A.T @ B.T)`
    Zero underflow em T4 (SM 7.5) sem necessidade de GradScaler, e fusão exata
    `w_out.add_((sB @ sA) * 1.0)` em `extract_mos_nf4_streaming`!
    """

    def __init__(self, base_linear: nn.Module, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)

        in_features = int(base_linear.in_features)
        out_features = int(base_linear.out_features)
        self.rank = rank
        self.scale = float(alpha) / float(rank)
        self.drop = nn.Dropout(dropout)

        dev = next(base_linear.parameters()).device
        self.scar_A = nn.Parameter(torch.empty((rank, in_features), device=dev, dtype=torch.float32))
        self.scar_B = nn.Parameter(torch.zeros((out_features, rank), device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.scar_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_f32 = self.drop(x.to(device=self.scar_A.device, dtype=torch.float32))
        delta = F.linear(F.linear(x_f32, self.scar_A), self.scar_B) * self.scale
        return base_out + delta.to(device=base_out.device, dtype=base_out.dtype)


def inject_mos_scar_experts(
    backbone: nn.Module, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05
) -> Dict[str, MoSResidualScarLinear]:
    for p in backbone.parameters():
        p.requires_grad_(False)
    injected: Dict[str, MoSResidualScarLinear] = {}
    base_has_model_prefix = hasattr(backbone, "model") and hasattr(backbone.model, "layers")
    for mod_name, mod in list(backbone.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in SCAR_TARGET_MODULES and hasattr(child, "in_features") and hasattr(child, "out_features"):
                raw_prefix = f"{mod_name}.{child_name}" if mod_name else child_name
                # Alinha com chaves de `model.safetensors.index.json` (que em Qwen/Llama possuem `model.layers...` ou `layers...`)
                scar_mod = MoSResidualScarLinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(mod, child_name, scar_mod)
                injected[raw_prefix] = scar_mod
    return injected


# ==============================================================================
# 2. Cabeça Multi-Depth + SCache O(1) + Bessel Multiplicity-Gated Readout
# ==============================================================================
class MultiDepthMoSSCacheClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        capture_layers_1b: Sequence[int],
        head_device: str = "cpu",
        proj_dim: int = 512,
        scache_dim: int = 32,
        num_classes: int = 18,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        self.capture_layers_1b = list(capture_layers_1b)
        self.n_depths = len(self.capture_layers_1b)
        self.head_device = torch.device(head_device)

        self.depth_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.n_depths)]).to(self.head_device)
        self.depth_logits = nn.Parameter(torch.zeros(self.n_depths, device=self.head_device))
        self.proj = nn.Sequential(
            nn.Linear(self.n_depths * d_model, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
        ).to(self.head_device)
        self.scache_key = nn.Linear(proj_dim, scache_dim, bias=False).to(self.head_device)
        self.scache_scale = nn.Parameter(torch.tensor(0.15, device=self.head_device))

        self.plain_norm = nn.LayerNorm(proj_dim).to(self.head_device)
        self.plain_head = nn.Linear(proj_dim, num_classes).to(self.head_device)

        self.twin_norm = nn.LayerNorm(proj_dim).to(self.head_device)
        self.twin_head = nn.Linear(proj_dim, num_classes).to(self.head_device)

    def encode_passages(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        inp_dev = next(self.backbone.parameters()).device
        out = self.backbone(
            input_ids=input_ids.to(inp_dev),
            attention_mask=attention_mask.to(inp_dev),
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states
        m = attention_mask.to(self.head_device).unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp(min=1.0)

        pooled_slices = []
        d_weights = F.softmax(self.depth_logits, dim=0) * self.n_depths
        n_tot_layers = int(self.backbone.config.num_hidden_layers)
        eps = float(getattr(self.backbone.config, "rms_norm_eps", 1e-6))
        for idx, l_1b in enumerate(self.capture_layers_1b):
            h_l = hs[l_1b].to(self.head_device, dtype=torch.float32)
            if l_1b == n_tot_layers:
                h_use = h_l
            else:
                h_use = h_l * torch.rsqrt(h_l.pow(2).mean(dim=-1, keepdim=True) + eps)
            p = (h_use * m).sum(dim=1) / denom
            p_unit = p * torch.rsqrt(p.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            p_norm = self.depth_norms[idx](p_unit) * d_weights[idx]
            pooled_slices.append(p_norm)

        z_cat = torch.cat(pooled_slices, dim=-1)
        return self.proj(z_cat)

    def apply_scache_and_heads(
        self, z: torch.Tensor, prompt_ids: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retorna `(logits_plain, logits_twin, twin_gate)` onde:
          - `z_centered` usa correção exata de Bessel `sqrt(m / (m - 1))`.
          - `twin_gate_i = 0.55 * (m_i - 1) / (m_i - 1 + 3.0)` (`0.0` em singletons `m_i = 1`!).
        """
        z = z.to(self.head_device, dtype=torch.float32)
        z_sc = z
        z_centered = torch.zeros_like(z)
        twin_gate = torch.zeros((z.shape[0], 1), device=self.head_device, dtype=torch.float32)

        if prompt_ids is not None:
            pids = prompt_ids.to(self.head_device)
            s_proj = self.scache_key(z)
            z_sc_list = z.clone()
            for g in torch.unique(pids):
                mask = pids == g
                cnt = int(mask.sum().item())
                if cnt > 1:
                    bessel = math.sqrt(cnt / float(cnt - 1))
                    gate_m = (cnt - 1.0) / (cnt - 1.0 + 3.0)
                    sg = s_proj[mask]
                    dz = (sg - sg.mean(dim=0, keepdim=True)) * bessel
                    w_scar = (dz.transpose(0, 1) @ dz) / float(cnt)
                    res = (dz @ w_scar.transpose(0, 1)) @ self.scache_key.weight
                    z_sc_list[mask] = z[mask] + (self.scache_scale * gate_m) * res
                    twin_gate[mask, 0] = 0.55 * gate_m
            z_sc = z_sc_list
            for g in torch.unique(pids):
                mask = pids == g
                cnt = int(mask.sum().item())
                if cnt > 1:
                    bessel = math.sqrt(cnt / float(cnt - 1))
                    z_centered[mask] = (z_sc[mask] - z_sc[mask].mean(dim=0, keepdim=True)) * bessel

        logits_plain = self.plain_head(self.plain_norm(z_sc))
        logits_twin = self.twin_head(self.twin_norm(z_centered))
        return logits_plain, logits_twin, twin_gate


# ==============================================================================
# 3. Corpus Desduplicado 2-Estágios & Amostrador Multi-Multiplicidade
# ==============================================================================
def load_clean_stage1_corpus(train_csv: Path, pool_v4_path: Path, passages_csv: Path):
    import pandas as pd
    df_tr = pd.read_csv(train_csv)
    tr_texts = set(df_tr["text"].fillna("").astype(str).str.strip())
    tr_pids = set(df_tr["prompt_id"].astype(str))

    frames = []
    if pool_v4_path.exists():
        pv4 = pd.read_parquet(pool_v4_path)
        pv4 = pv4[pv4["model"].isin(L2I)].copy()
        pv4["text_clean"] = pv4["text"].fillna("").astype(str).str.strip()
        pv4 = pv4[(~pv4["text_clean"].isin(tr_texts)) & (~pv4["prompt_id"].astype(str).isin(tr_pids))]
        frames.append(
            pd.DataFrame({
                "prompt_id": "pv4_" + pv4["prompt_id"].astype(str),
                "text": pv4["text"].astype(str),
                "y": pv4["model"].map(L2I).values,
                "src_pid": "",
            })
        )

    if passages_csv.exists():
        pa = pd.read_csv(passages_csv)
        pa = pa[
            (pa["passes_funnel"] == True)
            & (pa["is_refusal"] == False)
            & (pa["is_empty"] == False)
            & (pa["split"] != "pilot_range")
            & (pa["split"] != "new")
            & (pa["model"].isin(L2I))
        ].copy()
        pa["text_clean"] = pa["text"].fillna("").astype(str).str.strip()
        existing = set(frames[0]["text"].str.strip()) if frames else set()
        pa = pa[(~pa["text_clean"].isin(tr_texts)) & (~pa["text_clean"].isin(existing))]
        frames.append(
            pd.DataFrame({
                "prompt_id": "pa_" + pa["prompt_id"].astype(str),
                "text": pa["text"].astype(str),
                "y": pa["model"].map(L2I).values,
                "src_pid": pa["source_prompt_id"].fillna("").astype(str),
            })
        )

    full_ext = pd.concat(frames, ignore_index=True)
    print(f"✓ Stage-1 Clean Deduplicated Corpus: {len(full_ext)} passages across {full_ext['prompt_id'].nunique()} prompts")
    return full_ext


def build_prompt_grouped_batches(
    df, batch_size: int = 16, seed: int = 42, multiscale_m: bool = False
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Retorna lista de `(row_indices, group_ids_in_batch)`.
    Quando `multiscale_m=True` (Estágio 2), particiona aleatoriamente cada prompt em subgrupos
    `m in {1, 2, 3, 6, 16}` para calibrar nativamente a cabeça `Twin` em grupos pequenos!
    """
    rng = np.random.default_rng(seed)
    by_prompt: Dict[str, List[int]] = defaultdict(list)
    for idx, pid in enumerate(df["prompt_id"].astype(str).values):
        by_prompt[pid].append(idx)

    pids = list(by_prompt.keys())
    rng.shuffle(pids)

    chunks_rows: List[Tuple[int, int]] = []
    gid = 1
    for pid in pids:
        rows = by_prompt[pid].copy()
        rng.shuffle(rows)
        if not multiscale_m:
            for r in rows:
                chunks_rows.append((r, gid))
            gid += 1
        else:
            pos = 0
            while pos < len(rows):
                m_sub = int(rng.choice([1, 2, 3, 4, 8, 16], p=[0.10, 0.25, 0.25, 0.15, 0.10, 0.15]))
                seg = rows[pos : pos + m_sub]
                for r in seg:
                    chunks_rows.append((r, gid))
                gid += 1
                pos += len(seg)

    batches: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in range(0, len(chunks_rows), batch_size):
        seg = chunks_rows[s : s + batch_size]
        if len(seg) >= 2:
            r_arr = np.array([x[0] for x in seg], dtype=np.int64)
            g_arr = np.array([x[1] for x in seg], dtype=np.int64)
            batches.append((r_arr, g_arr))
    return batches


# ==============================================================================
# 4. Exportador `.safetensors` Compatível com `extract_mos_nf4_streaming`
# ==============================================================================
def detect_safetensors_key_prefix(backbone_dir: Path | None) -> str:
    """Detecta se `model.safetensors.index.json` ou `model.safetensors` usa prefixo `model.layers.` ou `layers.`."""
    if backbone_dir is not None:
        idx_file = backbone_dir / "model.safetensors.index.json"
        single_file = backbone_dir / "model.safetensors"
        if idx_file.exists():
            keys = list(json.loads(idx_file.read_text()).get("weight_map", {}).keys())
        elif single_file.exists():
            from safetensors import safe_open
            with safe_open(str(single_file), framework="pt", device="cpu") as f_st:
                keys = list(f_st.keys())
        else:
            keys = []
        if any(k.startswith("model.layers.") for k in keys):
            return "model."
        if any(k.startswith("layers.") for k in keys):
            return ""
    return ""


def export_fold_safetensors(
    model: MultiDepthMoSSCacheClassifier,
    injected: Dict[str, MoSResidualScarLinear],
    out_path: Path,
    key_prefix: str = "",
):
    tensors: Dict[str, torch.Tensor] = {}
    for raw_prefix, scar_mod in injected.items():
        # Normaliza prefixo (`layers.0...` vs `model.layers.0...`) conforme o checkpoint no disco!
        clean_p = raw_prefix[len("model.") :] if raw_prefix.startswith("model.") else raw_prefix
        final_p = f"{key_prefix}{clean_p}"
        tensors[f"{final_p}.scar_A"] = (scar_mod.scar_A.detach().float() * scar_mod.scale).to(torch.float16).cpu()
        tensors[f"{final_p}.scar_B"] = scar_mod.scar_B.detach().float().to(torch.float16).cpu()

    for k, v in model.state_dict().items():
        if not k.startswith("backbone."):
            tensors[f"head.{k}"] = v.detach().float().cpu()

    save_file(tensors, str(out_path))


def get_autocast_ctx(device_str: str, quant_mode: str):
    if not device_str.startswith("cuda") or not torch.cuda.is_available():
        return nullcontext()
    cc_major = torch.cuda.get_device_capability(0)[0]
    if quant_mode in ("bf16", "nf4") and cc_major >= 8:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cuda", dtype=torch.float16)


@torch.no_grad()
def evaluate_covering_windows(
    model: MultiDepthMoSSCacheClassifier,
    tokenizer,
    texts: List[str],
    prompt_groups: np.ndarray,
    max_len: int = 512,
    stride: int = 384,
    batch_size: int = 24,
    device: str = "cuda:0",
    quant_mode: str = "fp16",
) -> np.ndarray:
    model.eval()
    enc_all = tokenizer(texts, truncation=False, padding=False)["input_ids"]
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    win_ids, doc_idx, win_wts = [], [], []
    for i, seq in enumerate(enc_all):
        L = len(seq)
        if L <= max_len:
            win_ids.append(seq if L > 0 else [eos_id])
            doc_idx.append(i)
            win_wts.append(float(max(L, 1)))
        else:
            starts = list(range(0, L - max_len + 1, stride))
            if starts[-1] + max_len < L:
                starts.append(L - max_len)
            for st in starts:
                ch = seq[st : st + max_len]
                win_ids.append(ch)
                doc_idx.append(i)
                win_wts.append(float(len(ch)))

    order = np.argsort([len(s) for s in win_ids], kind="stable")
    proj_dim = model.plain_head.in_features
    Z_win = np.zeros((len(win_ids), proj_dim), dtype=np.float32)

    for b_s in range(0, len(win_ids), batch_size):
        b_i = order[b_s : b_s + batch_size]
        tok_b = tokenizer.pad({"input_ids": [win_ids[j] for j in b_i]}, padding=True, return_tensors="pt")
        with get_autocast_ctx(device, quant_mode):
            z_b = model.encode_passages(tok_b["input_ids"], tok_b["attention_mask"])
        Z_win[b_i] = z_b.float().cpu().numpy()

    Z_doc = np.zeros((len(texts), proj_dim), dtype=np.float32)
    w_sum = np.zeros((len(texts), 1), dtype=np.float32)
    doc_idx_arr = np.asarray(doc_idx, dtype=np.int64)
    win_wts_arr = np.asarray(win_wts, dtype=np.float32)
    np.add.at(Z_doc, doc_idx_arr, Z_win * win_wts_arr[:, None])
    np.add.at(w_sum[:, 0], doc_idx_arr, win_wts_arr)
    Z_doc /= np.maximum(w_sum, 1e-6)

    z_t = torch.from_numpy(Z_doc)
    _, u_inv = np.unique(prompt_groups, return_inverse=True)
    g_t = torch.from_numpy(u_inv)
    logits_p, logits_c, twin_gate = model.apply_scache_and_heads(z_t, g_t)
    probs = (1.0 - twin_gate) * F.softmax(logits_p, dim=-1) + twin_gate * F.softmax(logits_c, dim=-1)
    return probs.cpu().numpy()


# ==============================================================================
# 5. Entrypoint CLI Universal (Kaggle 2xT4 / Colab A100 / Local Smoke Test)
# ==============================================================================
def main():
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    parser = argparse.ArgumentParser(description="System One Two-Stage MoS Residual Scar Experts")
    parser.add_argument("--model-id", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--pool-v4", required=True)
    parser.add_argument("--passages-csv", required=True)
    parser.add_argument("--out-dir", default="./titan4_qwen3_emb_8b_mos_artifacts")
    parser.add_argument("--quant", choices=["auto", "nf4", "fp16", "bf16", "fp32"], default="auto")
    parser.add_argument("--save-backbone", action="store_true", help="Salva cópia do backbone quantizado NF4")
    parser.add_argument("--capture-layers", default="auto")
    parser.add_argument("--stage1-max-rows", type=int, default=27000)
    parser.add_argument("--stage1-epochs", type=int, default=1)
    parser.add_argument("--stage2-epochs", type=int, default=4)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-folds-run", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    has_cuda = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if has_cuda else 0
    head_device = f"cuda:{n_gpus - 1}" if n_gpus > 0 else "cpu"

    quant_mode = args.quant
    if quant_mode == "auto":
        if not has_cuda:
            quant_mode = "fp32"
        elif torch.cuda.get_device_capability(0)[0] >= 8:
            try:
                import bitsandbytes  # noqa: F401
                quant_mode = "nf4"
            except Exception:
                quant_mode = "bf16"
        else:
            quant_mode = "fp16"

    print(f"🚀 Launching Two-Stage MoS Scar-Expert Training | GPUs={n_gpus} | Mode={quant_mode} | Model={args.model_id}", flush=True)

    df_train = pd.read_csv(args.train_csv)
    df_train["text"] = df_train["text"].fillna("").astype(str)
    df_train["y"] = df_train["model"].map(L2I).values
    prompts_train = df_train["prompt_id"].astype(str).values

    df_ext = load_clean_stage1_corpus(Path(args.train_csv), Path(args.pool_v4), Path(args.passages_csv))
    if len(df_ext) > args.stage1_max_rows:
        per_cls = max(1, args.stage1_max_rows // K)
        df_ext = (
            df_ext.groupby("y", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), per_cls), random_state=42))
            .reset_index(drop=True)
        )
        print(f"✓ Balanced Stage-1 Subset: {len(df_ext)} rows (~{per_cls}/class)", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_src_path = Path(args.model_id) if Path(args.model_id).exists() else None
    if quant_mode == "nf4":
        from transformers import BitsAndBytesConfig
        comp_dt = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=comp_dt,
        )
        base_backbone = AutoModel.from_pretrained(args.model_id, quantization_config=bnb_cfg, device_map="auto")
    elif quant_mode in ("fp16", "bf16") and has_cuda:
        dt = torch.float16 if quant_mode == "fp16" else torch.bfloat16
        base_backbone = AutoModel.from_pretrained(args.model_id, dtype=dt, device_map="auto")
    else:
        base_backbone = AutoModel.from_pretrained(args.model_id, dtype=torch.float32).to("cpu")

    base_backbone.config.use_cache = False
    if has_cuda and hasattr(base_backbone, "gradient_checkpointing_enable"):
        base_backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(base_backbone, "enable_input_require_grads"):
            base_backbone.enable_input_require_grads()

    bb_save_dir = out_dir / "backbone_nf4_mos"
    if args.save_backbone and not (bb_save_dir / "config.json").exists():
        print(f"💾 Exportando backbone MoS para {bb_save_dir}...", flush=True)
        base_backbone.save_pretrained(bb_save_dir, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(bb_save_dir)
        model_src_path = bb_save_dir

    key_prefix = detect_safetensors_key_prefix(model_src_path)
    n_tot_layers = int(base_backbone.config.num_hidden_layers)
    cap_layers = resolve_capture_layers(n_tot_layers, args.capture_layers)
    d_model = int(base_backbone.config.hidden_size)

    injected = inject_mos_scar_experts(base_backbone, rank=args.rank, alpha=2.0 * args.rank, dropout=0.05)
    model = MultiDepthMoSSCacheClassifier(
        base_backbone, d_model=d_model, capture_layers_1b=cap_layers, head_device=head_device
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(
        f"✓ Injected {len(injected)} MoS Scar projections | Capture={cap_layers} | "
        f"KeyPrefix='{key_prefix}' | Trainable={sum(p.numel() for p in trainable_params):,}",
        flush=True,
    )

    def run_epoch(df_ep: pd.DataFrame, opt: torch.optim.Optimizer, lr_sched, seed: int, multiscale: bool) -> float:
        model.train()
        batches = build_prompt_grouped_batches(df_ep, batch_size=args.batch_size, seed=seed, multiscale_m=multiscale)
        tot_loss, n_steps = 0.0, 0
        texts_all = df_ep["text"].tolist()
        y_all = df_ep["y"].values

        for b_idx, g_idx in batches:
            b_texts = [texts_all[i] for i in b_idx]
            tok = tokenizer(b_texts, truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            yt = torch.from_numpy(y_all[b_idx]).long().to(head_device)
            gt = torch.from_numpy(g_idx).long().to(head_device)

            opt.zero_grad(set_to_none=True)
            with get_autocast_ctx(head_device, quant_mode):
                z = model.encode_passages(tok["input_ids"], tok["attention_mask"])
            lp, lc, twin_gate = model.apply_scache_and_heads(z.float(), gt)
            loss_p = F.cross_entropy(lp, yt, label_smoothing=0.05)
            has_sib = (twin_gate[:, 0] > 1e-5)
            if bool(has_sib.any()):
                loss_c = F.cross_entropy(lc[has_sib], yt[has_sib], label_smoothing=0.05)
                loss = 0.50 * loss_p + 0.50 * loss_c
            else:
                loss = loss_p

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            if lr_sched is not None:
                lr_sched.step()
            tot_loss += float(loss.item())
            n_steps += 1
            if n_steps % 100 == 0:
                vram = (torch.cuda.max_memory_allocated(0) / (1024**3)) if has_cuda else 0.0
                print(f"      step {n_steps}/{len(batches)} | avg_loss={tot_loss/n_steps:.4f} | gpu0_peak={vram:.2f} GiB", flush=True)
        return tot_loss / max(n_steps, 1)

    # ==========================================================================
    # ESTÁGIO 1: Pré-Adaptação Forense no Open Pool Limpo
    # ==========================================================================
    df_ext_stage1 = df_ext[df_ext["src_pid"] == ""].reset_index(drop=True)
    if args.stage1_epochs > 0 and len(df_ext_stage1) >= 4:
        print("\n=== [STAGE 1] Pre-Adapting MoS Residual Scar Experts on Clean Open Pool ===", flush=True)
        total_s1_steps = max(len(build_prompt_grouped_batches(df_ext_stage1, args.batch_size)) * args.stage1_epochs, 1)
        opt_s1 = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=0.02)
        sched_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s1, T_max=total_s1_steps, eta_min=2e-5)
        for ep in range(args.stage1_epochs):
            t0 = time.time()
            l_ep = run_epoch(df_ext_stage1, opt_s1, sched_s1, seed=100 + ep, multiscale=False)
            print(f"  Stage-1 Epoch {ep+1}/{args.stage1_epochs} | Loss={l_ep:.4f} | Time={time.time()-t0:.1f}s", flush=True)

    stage1_snapshot = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if any(t in k for t in ("scar_A", "scar_B")) or not k.startswith("backbone.")
    }

    # ==========================================================================
    # ESTÁGIO 2: Calibração Fina Multi-Multiplicidade no Registro do Host
    # ==========================================================================
    print("\n=== [STAGE 2] Multi-Multiplicity Host Calibration on Official `train.csv` ===", flush=True)
    sgkf = StratifiedGroupKFold(n_splits=args.n_folds)
    oof_probs = np.zeros((len(df_train), K), dtype=np.float32)
    evaluated_mask = np.zeros(len(df_train), dtype=bool)

    for fold_i, (tr_idx, va_idx) in enumerate(sgkf.split(df_train, df_train["y"].values, groups=prompts_train)):
        if fold_i >= args.max_folds_run:
            break
        tf0 = time.time()
        cur_sd = model.state_dict()
        for k, v in stage1_snapshot.items():
            cur_sd[k].copy_(v.to(cur_sd[k].device))

        df_fold_tr = df_train.iloc[tr_idx][["prompt_id", "text", "y"]].reset_index(drop=True)
        est_batches = build_prompt_grouped_batches(df_fold_tr, args.batch_size, seed=500 + fold_i, multiscale_m=True)
        total_s2_steps = max(len(est_batches) * args.stage2_epochs, 1)
        opt_s2 = torch.optim.AdamW(trainable_params, lr=8e-5, weight_decay=0.03)
        sched_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s2, T_max=total_s2_steps, eta_min=5e-6)

        for ep in range(args.stage2_epochs):
            l_s2 = run_epoch(df_fold_tr, opt_s2, sched_s2, seed=500 + fold_i * 10 + ep, multiscale=True)
            print(f"    Fold {fold_i+1} Epoch {ep+1}/{args.stage2_epochs} | Loss={l_s2:.4f}", flush=True)

        va_texts = df_train.iloc[va_idx]["text"].tolist()
        va_groups = prompts_train[va_idx]
        p_va = evaluate_covering_windows(
            model, tokenizer, va_texts, va_groups, max_len=args.max_len, device=head_device, quant_mode=quant_mode
        )
        oof_probs[va_idx] = p_va
        evaluated_mask[va_idx] = True

        fold_acc = balanced_accuracy_score(df_train["y"].values[va_idx], p_va.argmax(axis=1))
        fold_path = out_dir / f"titan4_qwen3_emb_8b_fold{fold_i}_scars.safetensors"
        export_fold_safetensors(model, injected, fold_path, key_prefix=key_prefix)
        print(
            f"  ✓ Fold {fold_i+1}/{args.n_folds} | Host Val BalAcc = {fold_acc*100:.2f}% | "
            f"Saved {fold_path.name} ({time.time()-tf0:.1f}s)",
            flush=True,
        )
        gc.collect()
        if has_cuda:
            torch.cuda.empty_cache()

    eval_y = df_train["y"].values[evaluated_mask]
    eval_p = oof_probs[evaluated_mask]
    oof_bal = balanced_accuracy_score(eval_y, eval_p.argmax(axis=1))
    oof_ll = log_loss(eval_y, np.clip(eval_p, 1e-9, 1.0), labels=list(range(K)))
    np.save(out_dir / "oof_titan4_qwen3_emb_8b_8depth.npy", oof_probs)
    meta = {
        "schema": "system-one-titan4-mos-scars-v2",
        "model_id": args.model_id,
        "capture_layers_1b": cap_layers,
        "key_prefix": key_prefix,
        "rank": args.rank,
        "n_folds_completed": int(min(args.n_folds, args.max_folds_run)),
        "oof_balanced_accuracy": float(oof_bal),
        "oof_log_loss": float(oof_ll),
        "labels": LABELS,
    }
    (out_dir / "meta_titan4_qwen3_emb_8b.json").write_text(json.dumps(meta, indent=2))
    print("=" * 84, flush=True)
    print(f"🏆 TITAN 4 OOF BALANCED ACC: {oof_bal*100:.2f}% | LL: {oof_ll:.4f}", flush=True)
    print("=" * 84, flush=True)


if __name__ == "__main__":
    main()
