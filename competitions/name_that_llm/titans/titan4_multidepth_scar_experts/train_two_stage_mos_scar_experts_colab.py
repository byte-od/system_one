#!/usr/bin/env python3
"""
👑 SYSTEM ONE — MILESTONE 2 & 3 (COLAB A100):
TWO-STAGE PROPRIETARY MoS RESIDUAL SCAR EXPERTS + 8-DEPTH MULTI-SLICE ENGINE
============================================================================
Alvo de Leaderboard Público: >= 0.740 - 0.755+ (Top 10 em `name-that-llm`).
100% Arquitetura Proprietária System One (`AGENTS.md` & `SKILL.md`):
- ZERO adaptadores LoRA comunitários / de terceiros.
- Especialistas de Cicatriz Residuais MoS (`dW_scar = scale * B @ A`, rank=16) injetados
  diretamente em `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`, exportados
  no formato `.scar_A` / `.scar_B` nativamente suportado pelo `extract_mos_nf4_streaming`!
- Extração simultânea de 8 Profundidades (`[4, 8, 12, 16, 20, 24, 30, 36]` em `Qwen/Qwen3-Embedding-8B`)
  com pesos aprendidos por camada + Janelas Deslizantes Sem Truncamento (`max_len=512, stride=384`).
- Treinamento Fim-a-Fim em Mini-Batches Agrupados por Prompt: retropropaga simultaneamente
  pelo `Plain Head`, pelo `Delta-Origin Prompt-Centered Twin Head (z - mu_prompt)` e pela
  memória associativa `SCache O(1)`, ensinando as camadas internas a anular o assunto do prompt!

Estágio 1 (Pré-Adaptação Forense em Larga Escala):
  `pool_v4.parquet` + `passages.csv` limpo (desduplicado contra `train.csv` e sem `kimi-k3` legado).
Estágio 2 (Calibração Fina de Registro do Host Kaggle):
  5-Fold `StratifiedGroupKFold` por `prompt_id` sobre `train.csv` (2.699 linhas oficiais),
  salvando 5 estados de cicatriz (`fold0..fold4_scar_experts.safetensors`) + `oof_preds.npy`
  + backbone NF4 MoS pronto para upload direto ao Kaggle!
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer, BitsAndBytesConfig

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


# ==============================================================================
# 1. Camada Proprietária MoS Residual Scar Expert (Compatível com Streaming NF4)
# ==============================================================================
class MoSResidualScarLinear(nn.Module):
    """
    Envolve uma projeção base quantizada NF4 congelada com um Especialista de Cicatriz
    Residual de posto baixo (`y = base(x) + (alpha / r) * (drop(x) @ A.T @ B.T)`).
    Serializa `prefix.scar_A` (r, in_features) e `prefix.scar_B` (out_features, r) em FP16
    para fusão in-place `w_out.addmm_(sB, sA, alpha=scar_scale)` no motor de streaming MoS!
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
        self.scar_A = nn.Parameter(torch.empty((rank, in_features), device=dev, dtype=torch.bfloat16))
        self.scar_B = nn.Parameter(torch.zeros((out_features, rank), device=dev, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(self.scar_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_bf = self.drop(x.to(self.scar_A.dtype))
        delta = F.linear(F.linear(x_bf, self.scar_A), self.scar_B) * self.scale
        return base_out + delta.to(base_out.dtype)


def inject_mos_scar_experts(backbone: nn.Module, rank: int = 16, alpha: float = 32.0, dropout: float = 0.05) -> Dict[str, MoSResidualScarLinear]:
    """Congela 100% do backbone base (incluindo embeddings/layernorms) e injeta `MoSResidualScarLinear`."""
    for p in backbone.parameters():
        p.requires_grad_(False)
    injected: Dict[str, MoSResidualScarLinear] = {}
    for mod_name, mod in list(backbone.named_modules()):
        for child_name, child in list(mod.named_children()):
            if child_name in SCAR_TARGET_MODULES and hasattr(child, "in_features") and hasattr(child, "out_features"):
                full_prefix = f"{mod_name}.{child_name}" if mod_name else child_name
                scar_mod = MoSResidualScarLinear(child, rank=rank, alpha=alpha, dropout=dropout)
                setattr(mod, child_name, scar_mod)
                injected[full_prefix] = scar_mod
    return injected


# ==============================================================================
# 2. Cabeça Multi-Depth de 8 Camadas + SCache O(1) + Dual Plain/Prompt-Twin Readout
# ==============================================================================
class MultiDepthMoSSCacheClassifier(nn.Module):
    """
    Combina 8 fatias de profundidade (`[4,8,12,16,20,24,30,36]`) extraídas num único forward:
      1. Normalização RMS unitária por fatia + LayerNorm aprendida por profundidade
      2. Ponderação softmax aprendida entre profundidades + projeção residual `8*D -> proj_dim`
      3. Modulação associativa `SCache O(1)` intra-prompt (`32x32` covariância de cicatriz)
      4. Cabeças gêmeas simultâneas: `Plain Head` + `Prompt-Centered Twin Head (z - mu_prompt)`
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        capture_layers_1b: Sequence[int] = CAPTURE_LAYERS_QWEN3_8B,
        proj_dim: int = 512,
        scache_dim: int = 32,
        num_classes: int = 18,
    ):
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model
        self.capture_layers_1b = list(capture_layers_1b)
        self.n_depths = len(self.capture_layers_1b)

        self.depth_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.n_depths)])
        self.depth_logits = nn.Parameter(torch.zeros(self.n_depths))
        self.proj = nn.Sequential(
            nn.Linear(self.n_depths * d_model, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
        )
        self.scache_key = nn.Linear(proj_dim, scache_dim, bias=False)
        self.scache_scale = nn.Parameter(torch.tensor(0.15))

        self.plain_norm = nn.LayerNorm(proj_dim)
        self.plain_head = nn.Linear(proj_dim, num_classes)

        self.twin_norm = nn.LayerNorm(proj_dim)
        self.twin_head = nn.Linear(proj_dim, num_classes)

    def encode_passages(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        # hidden_states[0] = embeddings; hidden_states[l_1b] = saída da camada 1-based `l_1b`
        hs = out.hidden_states
        m = attention_mask.unsqueeze(-1).float()
        denom = m.sum(dim=1).clamp(min=1.0)

        pooled_slices = []
        d_weights = F.softmax(self.depth_logits, dim=0) * self.n_depths
        n_tot_layers = int(self.backbone.config.num_hidden_layers)
        eps = float(getattr(self.backbone.config, "rms_norm_eps", 1e-6))
        for idx, l_1b in enumerate(self.capture_layers_1b):
            h_l = hs[l_1b].float()
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aplica ressonância `SCache O(1)` e centralização por grupo quando `prompt_ids` possui irmãos.
        """
        z_sc = z
        z_centered = torch.zeros_like(z)

        if prompt_ids is not None:
            s_proj = self.scache_key(z)
            z_sc_list = z.clone()
            for g in torch.unique(prompt_ids):
                mask = prompt_ids == g
                cnt = int(mask.sum().item())
                if cnt > 1:
                    sg = s_proj[mask]
                    dz = sg - sg.mean(dim=0, keepdim=True)
                    w_scar = (dz.transpose(0, 1) @ dz) / float(cnt)
                    res = (dz @ w_scar.transpose(0, 1)) @ self.scache_key.weight
                    z_sc_list[mask] = z[mask] + self.scache_scale * res
            z_sc = z_sc_list
            for g in torch.unique(prompt_ids):
                mask = prompt_ids == g
                if int(mask.sum().item()) > 1:
                    z_centered[mask] = z_sc[mask] - z_sc[mask].mean(dim=0, keepdim=True)

        logits_plain = self.plain_head(self.plain_norm(z_sc))
        logits_twin = self.twin_head(self.twin_norm(z_centered))
        return logits_plain, logits_twin


# ==============================================================================
# 3. Construtor de Corpus Desduplicado 2-Estágios & Amostrador Agrupado por Prompt
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
            & (~((pa["split"] == "new") & (pa["model"] == "kimi-k3")))
            & (pa["model"].isin(L2I))
        ].copy()
        pa["text_clean"] = pa["text"].fillna("").astype(str).str.strip()
        # Evita duplicar linhas de `pa` que já estão em `pv4`
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


def build_prompt_grouped_batches(df, batch_size: int = 16, seed: int = 42) -> List[np.ndarray]:
    """Agrupa índices do mesmo `prompt_id` no mesmo mini-batch para ativar `z - mu_prompt` fim-a-fim."""
    rng = np.random.default_rng(seed)
    by_prompt: Dict[str, List[int]] = defaultdict(list)
    for idx, pid in enumerate(df["prompt_id"].astype(str).values):
        by_prompt[pid].append(idx)

    pids = list(by_prompt.keys())
    rng.shuffle(pids)
    batches: List[np.ndarray] = []
    cur: List[int] = []
    for pid in pids:
        rows = by_prompt[pid].copy()
        rng.shuffle(rows)
        for r in rows:
            cur.append(r)
            if len(cur) == batch_size:
                batches.append(np.asarray(cur, dtype=np.int64))
                cur = []
    if len(cur) >= 2:
        batches.append(np.asarray(cur, dtype=np.int64))
    return batches


# ==============================================================================
# 4. Exportador de Pesos `.safetensors` Compatível com `extract_mos_nf4_streaming`
# ==============================================================================
def export_fold_safetensors(
    model: MultiDepthMoSSCacheClassifier,
    injected: Dict[str, MoSResidualScarLinear],
    out_path: Path,
):
    tensors: Dict[str, torch.Tensor] = {}
    for prefix, scar_mod in injected.items():
        # Salva scar_A já multiplicado por `scale` para fusão direta `w_out.addmm_(sB, sA, alpha=1.0)`!
        tensors[f"{prefix}.scar_A"] = (scar_mod.scar_A.detach().float() * scar_mod.scale).to(torch.float16).cpu()
        tensors[f"{prefix}.scar_B"] = scar_mod.scar_B.detach().to(torch.float16).cpu()

    # Salva parâmetros da cabeça Multi-Depth + SCache
    for k, v in model.state_dict().items():
        if not k.startswith("backbone."):
            tensors[f"head.{k}"] = v.detach().float().cpu()

    save_file(tensors, str(out_path))


@torch.no_grad()
def evaluate_covering_windows(
    model: MultiDepthMoSSCacheClassifier,
    tokenizer,
    texts: List[str],
    prompt_groups: np.ndarray,
    max_len: int = 512,
    stride: int = 384,
    batch_size: int = 32,
    device: str = "cuda:0",
) -> np.ndarray:
    """Inferência completa sem truncamento via janelas deslizantes ponderadas por tokens."""
    model.eval()
    enc_all = tokenizer(texts, truncation=False, padding=False)["input_ids"]

    win_ids, doc_idx, win_wts = [], [], []
    for i, seq in enumerate(enc_all):
        L = len(seq)
        if L <= max_len:
            win_ids.append(seq if L > 0 else [tokenizer.eos_token_id])
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
        tok_b = tokenizer.pad({"input_ids": [win_ids[j] for j in b_i]}, padding=True, return_tensors="pt").to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z_b = model.encode_passages(tok_b["input_ids"], tok_b["attention_mask"])
        Z_win[b_i] = z_b.float().cpu().numpy()

    Z_doc = np.zeros((len(texts), proj_dim), dtype=np.float32)
    w_sum = np.zeros((len(texts), 1), dtype=np.float32)
    doc_idx_arr = np.asarray(doc_idx, dtype=np.int64)
    win_wts_arr = np.asarray(win_wts, dtype=np.float32)
    np.add.at(Z_doc, doc_idx_arr, Z_win * win_wts_arr[:, None])
    np.add.at(w_sum[:, 0], doc_idx_arr, win_wts_arr)
    Z_doc /= np.maximum(w_sum, 1e-6)

    # Aplica SCache + Plain + Twin sobre os documentos agregados
    z_t = torch.from_numpy(Z_doc).to(device)
    _, u_inv = np.unique(prompt_groups, return_inverse=True)
    g_t = torch.from_numpy(u_inv).to(device)
    logits_p, logits_c = model.apply_scache_and_heads(z_t, g_t)
    probs = 0.45 * F.softmax(logits_p, dim=-1) + 0.55 * F.softmax(logits_c, dim=-1)
    return probs.cpu().numpy()


# ==============================================================================
# 5. Entrypoint CLI para Execução Turn-Key no Google Colab A100
# ==============================================================================
def main():
    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score, log_loss
    from sklearn.model_selection import StratifiedGroupKFold
    parser = argparse.ArgumentParser(description="System One Two-Stage MoS Residual Scar Experts (Colab A100)")
    parser.add_argument("--model-id", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--train-csv", required=True, help="Caminho para train.csv oficial (2.699 linhas)")
    parser.add_argument("--pool-v4", required=True, help="Caminho para pool_v4.parquet")
    parser.add_argument("--passages-csv", required=True, help="Caminho para passages.csv")
    parser.add_argument("--out-dir", default="./titan4_qwen3_emb_8b_mos_artifacts")
    parser.add_argument("--stage1-max-rows", type=int, default=36000, help="Amostragem balanceada Stage 1 (~2000/LLM)")
    parser.add_argument("--stage1-epochs", type=int, default=2)
    parser.add_argument("--stage2-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Launching Two-Stage MoS Scar-Expert Training on {device} | Model: {args.model_id}")

    df_train = pd.read_csv(args.train_csv)
    df_train["text"] = df_train["text"].fillna("").astype(str)
    df_train["y"] = df_train["model"].map(L2I).values
    prompts_train = df_train["prompt_id"].astype(str).values

    df_ext = load_clean_stage1_corpus(Path(args.train_csv), Path(args.pool_v4), Path(args.passages_csv))
    if len(df_ext) > args.stage1_max_rows:
        per_cls = args.stage1_max_rows // K
        df_ext = (
            df_ext.groupby("y", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), per_cls), random_state=42))
            .reset_index(drop=True)
        )
        print(f"✓ Balanced Stage-1 Subset: {len(df_ext)} rows ({per_cls}/class)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    print("⬇ Carregando e quantizando backbone NF4 MoS base...")
    base_backbone = AutoModel.from_pretrained(args.model_id, quantization_config=bnb_cfg, device_map={"": 0})
    base_backbone.gradient_checkpointing_enable()

    # Salva uma vez o backbone NF4 quantizado para upload ao Kaggle (compatível com extract_mos_nf4_streaming!)
    bb_save_dir = out_dir / "backbone_nf4_mos"
    if not (bb_save_dir / "config.json").exists():
        print(f"💾 Exportando backbone NF4 MoS para {bb_save_dir}...")
        base_backbone.save_pretrained(bb_save_dir, safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(bb_save_dir)

    d_model = int(base_backbone.config.hidden_size)
    injected = inject_mos_scar_experts(base_backbone, rank=args.rank, alpha=2 * args.rank, dropout=0.05)
    model = MultiDepthMoSSCacheClassifier(base_backbone, d_model=d_model).to(device)
    # Mantém cabeças em FP32 para estabilidade numérica sob autocast BF16
    for name, p in model.named_parameters():
        if not name.startswith("backbone."):
            p.data = p.data.float()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"✓ Injected {len(injected)} MoS Residual Scar projections | Trainable params: {sum(p.numel() for p in trainable_params):,}")

    def run_epoch(df_ep: pd.DataFrame, opt: torch.optim.Optimizer, lr_sched, seed: int) -> float:
        model.train()
        batches = build_prompt_grouped_batches(df_ep, batch_size=args.batch_size, seed=seed)
        tot_loss, n_steps = 0.0, 0
        pids_all = df_ep["prompt_id"].astype(str).values
        texts_all = df_ep["text"].tolist()
        y_all = df_ep["y"].values

        for b_idx in batches:
            b_texts = [texts_all[i] for i in b_idx]
            _, u_inv = np.unique(pids_all[b_idx], return_inverse=True)
            tok = tokenizer(b_texts, truncation=True, max_length=512, padding=True, return_tensors="pt").to(device)
            yt = torch.from_numpy(y_all[b_idx]).long().to(device)
            gt = torch.from_numpy(u_inv).long().to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = model.encode_passages(tok["input_ids"], tok["attention_mask"])
                lp, lc = model.apply_scache_and_heads(z.float(), gt)
                loss = 0.45 * F.cross_entropy(lp, yt, label_smoothing=0.05) + 0.55 * F.cross_entropy(lc, yt, label_smoothing=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            if lr_sched is not None:
                lr_sched.step()
            tot_loss += float(loss.item())
            n_steps += 1
        return tot_loss / max(n_steps, 1)

    # ==========================================================================
    # ESTÁGIO 1: Adaptação Geral no Open Pool Limpo (Salva Checkpoint Base Stage-1)
    # ==========================================================================
    print("\n=== [STAGE 1] Pre-Adapting MoS Residual Scar Experts on Clean Open Pool ===")
    # Filtra `train_regen` do Estágio 1 global para preservar 100% de independência para todos os 5 folds!
    df_ext_stage1 = df_ext[df_ext["src_pid"] == ""].reset_index(drop=True)
    total_s1_steps = max(len(build_prompt_grouped_batches(df_ext_stage1, args.batch_size)) * args.stage1_epochs, 1)
    opt_s1 = torch.optim.AdamW(trainable_params, lr=2e-4, weight_decay=0.02)
    sched_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s1, T_max=total_s1_steps, eta_min=2e-5)

    for ep in range(args.stage1_epochs):
        t0 = time.time()
        l_ep = run_epoch(df_ext_stage1, opt_s1, sched_s1, seed=100 + ep)
        print(f"  Stage-1 Epoch {ep+1}/{args.stage1_epochs} | Loss={l_ep:.4f} | Time={time.time()-t0:.1f}s", flush=True)

    # Guarda snapshot em RAM dos pesos treináveis pós-Stage-1 para ramificar cada um dos 5 Folds Stage-2!
    stage1_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if any(t in k for t in ("scar_A", "scar_B")) or not k.startswith("backbone.")}

    # ==========================================================================
    # ESTÁGIO 2: Calibração Fina de Registro do Host em 5-Fold StratifiedGroupKFold
    # ==========================================================================
    print("\n=== [STAGE 2] 5-Fold Host-Register Calibration on Official `train.csv` ===")
    sgkf = StratifiedGroupKFold(n_splits=5)
    oof_probs = np.zeros((len(df_train), K), dtype=np.float32)

    for fold_i, (tr_idx, va_idx) in enumerate(sgkf.split(df_train, df_train["y"].values, groups=prompts_train)):
        tf0 = time.time()
        # Restaura estado Stage-1 limpo
        cur_sd = model.state_dict()
        for k, v in stage1_snapshot.items():
            cur_sd[k].copy_(v.to(cur_sd[k].device))

        df_fold_tr = df_train.iloc[tr_idx][["prompt_id", "text", "y"]].reset_index(drop=True)
        total_s2_steps = max(len(build_prompt_grouped_batches(df_fold_tr, args.batch_size)) * args.stage2_epochs, 1)
        opt_s2 = torch.optim.AdamW(trainable_params, lr=8e-5, weight_decay=0.03)
        sched_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s2, T_max=total_s2_steps, eta_min=5e-6)

        for ep in range(args.stage2_epochs):
            run_epoch(df_fold_tr, opt_s2, sched_s2, seed=500 + fold_i * 10 + ep)

        va_texts = df_train.iloc[va_idx]["text"].tolist()
        va_groups = prompts_train[va_idx]
        p_va = evaluate_covering_windows(model, tokenizer, va_texts, va_groups, device=device)
        oof_probs[va_idx] = p_va

        fold_acc = balanced_accuracy_score(df_train["y"].values[va_idx], p_va.argmax(axis=1))
        fold_path = out_dir / f"titan4_qwen3_emb_8b_fold{fold_i}_scars.safetensors"
        export_fold_safetensors(model, injected, fold_path)
        print(f"  ✓ Fold {fold_i+1}/5 | Host Val BalAcc = {fold_acc*100:.2f}% | Saved {fold_path.name} ({time.time()-tf0:.1f}s)", flush=True)
        gc.collect()
        torch.cuda.empty_cache()

    oof_bal = balanced_accuracy_score(df_train["y"].values, oof_probs.argmax(axis=1))
    oof_ll = log_loss(df_train["y"].values, np.clip(oof_probs, 1e-9, 1.0))
    np.save(out_dir / "oof_titan4_qwen3_emb_8b_8depth.npy", oof_probs)
    meta = {
        "model_id": args.model_id,
        "capture_layers_1b": CAPTURE_LAYERS_QWEN3_8B,
        "rank": args.rank,
        "oof_balanced_accuracy": float(oof_bal),
        "oof_log_loss": float(oof_ll),
        "labels": LABELS,
    }
    (out_dir / "meta_titan4_qwen3_emb_8b.json").write_text(json.dumps(meta, indent=2))
    print("=" * 80)
    print(f"🏆 TITAN 4 (Qwen3-Embedding-8B 8-Depth MoS Scar Experts) 5-FOLD OOF BALANCED ACC: {oof_bal*100:.2f}% | LL: {oof_ll:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
