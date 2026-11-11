"""
System One — Unified Zero-Leak Pure-PyTorch FP16 MoS Layer-Streaming Engine
===========================================================================
Suporta:
1. Qualquer backbone NF4 MoS (Mistral Large 123B, Yi-34B, Qwen3-Embedding-8B, Qwen-72B)
   com detecção automática de prefixo ('model.' vs ''), GQA, RoPE theta e left/right padding.
2. Isolamento estrito de escopo em _step_attn() e _step_mlp(): zero vazamento de tensores
   entre subcamadas (pico de VRAM <= 4.2 GiB por GPU T4 mesmo para N = 10.000 passagens).
3. Extração Multi-Depth simultânea opcional (ex: 8 fatias [4,8,12,16,20,24,30,36]) com
   custo zero de I/O extra + fusão in-place opcional de MoS Residual Scar Experts (dW = B @ A).
4. Janelas deslizantes de cobertura completa (covering windows sem truncamento) com agregação
   ponderada por contagem de tokens válidos.
"""

from __future__ import annotations

import gc
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def build_covering_windows(
    ids_list: Sequence[Sequence[int]],
    max_len: int = 512,
    stride: int = 384,
) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
    """
    Decompõe sequências longas em janelas deslizantes de até `max_len` tokens sem
    descartar parágrafos finais. Para sequências <= max_len, gera exatamente 1 janela.
    Retorna:
      win_ids: lista de sequências de tokens por janela
      doc_idx: índice do documento original de cada janela (M,)
      win_weights: peso proporcional ao comprimento útil de cada janela (M,)
    """
    win_ids: List[List[int]] = []
    doc_idx: List[int] = []
    win_weights: List[float] = []

    for i, seq in enumerate(ids_list):
        s = list(seq)
        L = len(s)
        if L <= max_len:
            win_ids.append(s)
            doc_idx.append(i)
            win_weights.append(float(max(L, 1)))
        else:
            starts = list(range(0, L - max_len + 1, stride))
            if starts[-1] + max_len < L:
                starts.append(L - max_len)
            for st in starts:
                chunk = s[st : st + max_len]
                win_ids.append(chunk)
                doc_idx.append(i)
                win_weights.append(float(len(chunk)))

    return win_ids, np.asarray(doc_idx, dtype=np.int64), np.asarray(win_weights, dtype=np.float32)


def aggregate_window_embeddings(
    Z_win: np.ndarray,
    doc_idx: np.ndarray,
    win_weights: np.ndarray,
    n_docs: int,
    unit_rms: bool = True,
) -> np.ndarray:
    """Agrega embeddings de janelas deslizantes de volta para (n_docs, D)."""
    D = Z_win.shape[1]
    out = np.zeros((n_docs, D), dtype=np.float32)
    w_sum = np.zeros((n_docs, 1), dtype=np.float32)
    np.add.at(out, doc_idx, Z_win * win_weights[:, None])
    np.add.at(w_sum[:, 0], doc_idx, win_weights)
    out /= np.maximum(w_sum, 1e-6)
    if unit_rms:
        rms = 1.0 / np.sqrt(np.mean(out * out, axis=-1, keepdims=True) + 1e-6)
        out = out * rms
    return out


def extract_mos_nf4_streaming(
    backbone_dir: str,
    ids_list: Sequence[Sequence[int]],
    pad_id: int,
    devices: Sequence[str],
    padding_side: str = "left",
    tag: str = "MoS",
    micro_bsz: int = 8,
    max_tokens_per_dev: int = 60000,
    capture_layers: Optional[Sequence[int]] = None,
    scar_tensors: Optional[Dict[str, torch.Tensor]] = None,
    scar_scale: float = 1.0,
    log_every: int = 10,
) -> np.ndarray:
    """
    Motor canônico puro-PyTorch FP16 de Layer-Streaming MoS.
    - Se `capture_layers` for None: retorna (N, d_model) da última camada do backbone.
    - Se `capture_layers` for uma lista de índices 1-based (ex: [4,8,12,16,20,24,30,36]):
      retorna (N, len(capture_layers) * d_model) concatenando os pools normalizados!
    """
    devices = list(dict.fromkeys(devices))
    with open(os.path.join(backbone_dir, "config.json"), "r") as f:
        cfg_raw = json.load(f)

    n_layers = int(cfg_raw["num_hidden_layers"])
    d_model = int(cfg_raw["hidden_size"])
    n_heads = int(cfg_raw["num_attention_heads"])
    n_kv = int(cfg_raw["num_key_value_heads"])
    head_dim = int(cfg_raw.get("head_dim") or (d_model // n_heads))
    eps = float(cfg_raw.get("rms_norm_eps", 1e-5))
    rope_params = cfg_raw.get("rope_parameters") or {}
    theta = float(cfg_raw.get("rope_theta") or rope_params.get("rope_theta") or 1000000.0)
    gqa_rep = n_heads // n_kv
    has_qkv_bias = bool(cfg_raw.get("attention_bias", False))

    with open(os.path.join(backbone_dir, "model.safetensors.index.json"), "r") as f:
        idx_map = json.load(f)["weight_map"]

    base_pfx = "model." if "model.embed_tokens.weight" in idx_map else ""
    layers_pfx = f"{base_pfx}layers."
    norm_key = f"{base_pfx}norm.weight"
    open_shards: Dict[str, object] = {}

    target_layers_1b = list(capture_layers) if capture_layers is not None else [n_layers]
    max_target_layer = max(target_layers_1b)
    if max_target_layer > n_layers:
        raise ValueError(f"Requested layer {max_target_layer} exceeds backbone num_hidden_layers={n_layers}")
    capture_map = {l_1b - 1: slot_i for slot_i, l_1b in enumerate(target_layers_1b)}

    def get_tensor_cpu(key: str) -> torch.Tensor:
        shard_name = idx_map[key]
        if shard_name not in open_shards:
            open_shards[shard_name] = safe_open(
                os.path.join(backbone_dir, shard_name), framework="pt", device="cpu"
            )
        return open_shards[shard_name].get_tensor(key)

    def maybe_get_bias(prefix: str, dev: str, dtype=torch.float16) -> Optional[torch.Tensor]:
        b_key = f"{prefix}.bias"
        if b_key in idx_map:
            return get_tensor_cpu(b_key).to(dev, dtype=dtype)
        return None

    def dequant_nf4_mos(prefix: str, dev: str, dtype=torch.float16) -> torch.Tensor:
        w_key = f"{prefix}.weight"
        q_key = f"{prefix}.weight.quant_state.bitsandbytes__nf4"
        if q_key not in idx_map:
            w_out = get_tensor_cpu(w_key).to(dev, dtype=dtype)
        else:
            w_u8 = get_tensor_cpu(w_key)
            abs_u8 = get_tensor_cpu(f"{prefix}.weight.absmax")
            nabs = get_tensor_cpu(f"{prefix}.weight.nested_absmax")
            qmap = get_tensor_cpu(f"{prefix}.weight.quant_map")
            nqmap = get_tensor_cpu(f"{prefix}.weight.nested_quant_map")
            qs_b = get_tensor_cpu(q_key)
            qs = json.loads(bytes(qs_b.numpy()).decode("utf-8"))
            shape = tuple(qs["shape"])
            bs = int(qs["blocksize"])
            nbs = int(qs["nested_blocksize"])
            noff = float(qs["nested_offset"])

            qmap_d = qmap.to(dev, dtype=torch.float32)
            nqmap_d = nqmap.to(dev, dtype=torch.float32)
            real_abs = (
                nqmap_d[abs_u8.to(dev).long()].view(-1, nbs)
                * nabs.to(dev, dtype=torch.float32).view(-1, 1)
            ).reshape(-1) + noff
            idx256 = torch.arange(256, device=dev, dtype=torch.long)
            pair_lut = torch.stack([qmap_d[idx256 >> 4], qmap_d[idx256 & 0x0F]], dim=1)

            w_flat = w_u8.reshape(-1)
            out_flat = torch.empty(shape[0] * shape[1], device=dev, dtype=dtype)
            chunk_bytes = 67108864  # 64 MB packed = 128M weights (keeps LUT temp buffer <= 512 MB!)
            for b_s in range(0, w_flat.numel(), chunk_bytes):
                b_e = min(b_s + chunk_bytes, w_flat.numel())
                c_u8 = w_flat[b_s:b_e].to(dev, non_blocking=True)
                vals = pair_lut[c_u8.to(torch.int32)].view(-1, bs)
                vals.mul_(real_abs[(b_s * 2) // bs : (b_e * 2) // bs, None])
                out_flat[b_s * 2 : b_e * 2] = vals.view(-1).to(dtype)
                del c_u8, vals
            del qmap_d, nqmap_d, real_abs, idx256, pair_lut
            w_out = out_flat.view(shape)

        # In-place fusion of proprietary MoS Residual Scar Experts (dW = scar_scale * B @ A)
        if scar_tensors is not None:
            a_key = f"{prefix}.scar_A"
            b_key = f"{prefix}.scar_B"
            if a_key in scar_tensors and b_key in scar_tensors:
                sA = scar_tensors[a_key].to(dev, dtype=torch.float32)
                sB = scar_tensors[b_key].to(dev, dtype=torch.float32)
                w_out.add_((torch.matmul(sB, sA) * scar_scale).to(dtype))
                del sA, sB
        return w_out

    clean_ids = [list(seq) if len(seq) > 0 else [pad_id] for seq in ids_list]
    n_samples = len(clean_ids)
    order = np.argsort([len(x) for x in clean_ids], kind="stable")

    micro_specs = []
    for i in range(0, n_samples, micro_bsz):
        b_idx = order[i : i + micro_bsz]
        max_s = max(len(clean_ids[j]) for j in b_idx)
        inp_ids = torch.full((len(b_idx), max_s), pad_id, dtype=torch.long)
        att_msk = torch.zeros((len(b_idx), max_s), dtype=torch.bool)
        for r, j in enumerate(b_idx):
            seq = clean_ids[j]
            L = len(seq)
            if padding_side == "left":
                inp_ids[r, max_s - L :] = torch.tensor(seq, dtype=torch.long)
                att_msk[r, max_s - L :] = True
            else:
                inp_ids[r, :L] = torch.tensor(seq, dtype=torch.long)
                att_msk[r, :L] = True
        micro_specs.append((b_idx, inp_ids, att_msk))

    max_tokens_macro = max(int(max_tokens_per_dev) * max(len(devices), 1), 1)
    macro_chunks = []
    cur_chunk, cur_toks = [], 0
    for spec in micro_specs:
        t_cnt = int(spec[1].numel())
        if cur_chunk and cur_toks + t_cnt > max_tokens_macro:
            macro_chunks.append(cur_chunk)
            cur_chunk, cur_toks = [], 0
        cur_chunk.append(spec)
        cur_toks += t_cnt
    if cur_chunk:
        macro_chunks.append(cur_chunk)

    n_slots = len(target_layers_1b)
    Z_out = np.zeros((n_samples, n_slots * d_model), dtype=np.float32)

    def rot_half(z: torch.Tensor) -> torch.Tensor:
        z1, z2 = z[..., : head_dim // 2], z[..., head_dim // 2 :]
        return torch.cat([-z2, z1], dim=-1)

    def build_rope_and_bias(att_msk_d: torch.Tensor, dev: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, S = att_msk_d.shape
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=dev) / head_dim))
        t = torch.arange(S, dtype=torch.float32, device=dev)
        freqs = torch.outer(t, inv_freq)
        emb_pos = torch.cat([freqs, freqs], dim=-1)
        cos = emb_pos.cos().to(torch.float16)[None, None, :, :]
        sin = emb_pos.sin().to(torch.float16)[None, None, :, :]

        causal = torch.tril(torch.ones((S, S), dtype=torch.bool, device=dev))
        allow = causal[None, None, :, :] & att_msk_d[:, None, None, :]
        all_masked = ~allow.any(dim=-1, keepdim=True)
        allow = allow | (all_masked & causal[None, None, :, :])
        attn_bias = torch.zeros((bsz, 1, S, S), dtype=torch.float16, device=dev).masked_fill(
            ~allow, torch.finfo(torch.float16).min
        )
        return cos, sin, attn_bias

    def _step_attn(lp: str, active_devs: List[str], dev_states: Dict[str, list]) -> None:
        d0 = active_devs[0]
        in_ln_0 = get_tensor_cpu(f"{lp}.input_layernorm.weight").to(d0, dtype=torch.float16)
        q_0 = dequant_nf4_mos(f"{lp}.self_attn.q_proj", d0)
        k_0 = dequant_nf4_mos(f"{lp}.self_attn.k_proj", d0)
        v_0 = dequant_nf4_mos(f"{lp}.self_attn.v_proj", d0)
        o_0 = dequant_nf4_mos(f"{lp}.self_attn.o_proj", d0)
        qb_0 = maybe_get_bias(f"{lp}.self_attn.q_proj", d0) if has_qkv_bias else None
        kb_0 = maybe_get_bias(f"{lp}.self_attn.k_proj", d0) if has_qkv_bias else None
        vb_0 = maybe_get_bias(f"{lp}.self_attn.v_proj", d0) if has_qkv_bias else None
        qn_0 = get_tensor_cpu(f"{lp}.self_attn.q_norm.weight").to(d0, dtype=torch.float16) if f"{lp}.self_attn.q_norm.weight" in idx_map else None
        kn_0 = get_tensor_cpu(f"{lp}.self_attn.k_norm.weight").to(d0, dtype=torch.float16) if f"{lp}.self_attn.k_norm.weight" in idx_map else None

        attn_w = {d0: (in_ln_0, q_0, k_0, v_0, o_0, qb_0, kb_0, vb_0, qn_0, kn_0)}
        for d in active_devs[1:]:
            attn_w[d] = tuple(
                t.to(d, non_blocking=True) if t is not None else None for t in attn_w[d0]
            )
        del in_ln_0, q_0, k_0, v_0, o_0, qb_0, kb_0, vb_0, qn_0, kn_0

        max_b = max(len(dev_states[d]) for d in active_devs)
        for bi in range(max_b):
            for d in active_devs:
                if bi < len(dev_states[d]):
                    _, x, att_msk_d = dev_states[d][bi]
                    in_ln, qw, kw, vw, ow, qb, kb, vb, qn, kn = attn_w[d]
                    bsz, S, _ = x.shape
                    cos, sin, attn_bias = build_rope_and_bias(att_msk_d, d)
                    xf = x.float()
                    normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(torch.float16) * in_ln
                    del xf
                    q = F.linear(normed, qw, qb).view(bsz, S, n_heads, head_dim)
                    k = F.linear(normed, kw, kb).view(bsz, S, n_kv, head_dim)
                    v = F.linear(normed, vw, vb).view(bsz, S, n_kv, head_dim).transpose(1, 2)
                    del normed
                    if qn is not None:
                        qf = q.float()
                        q = (qf * torch.rsqrt(qf.pow(2).mean(-1, keepdim=True) + eps)).to(torch.float16) * qn
                    if kn is not None:
                        kf = k.float()
                        k = (kf * torch.rsqrt(kf.pow(2).mean(-1, keepdim=True) + eps)).to(torch.float16) * kn
                    q = q.transpose(1, 2)
                    k = k.transpose(1, 2)
                    q = (q * cos) + (rot_half(q) * sin)
                    k = (k * cos) + (rot_half(k) * sin)
                    k_rep = k.repeat_interleave(gqa_rep, dim=1) if gqa_rep > 1 else k
                    v_rep = v.repeat_interleave(gqa_rep, dim=1) if gqa_rep > 1 else v
                    ao = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=attn_bias)
                    dev_states[d][bi][1] = x + F.linear(
                        ao.transpose(1, 2).contiguous().view(bsz, S, n_heads * head_dim), ow
                    )
                    del cos, sin, attn_bias, q, k, v, k_rep, v_rep, ao
        del attn_w

    def _step_mlp(lp: str, active_devs: List[str], dev_states: Dict[str, list]) -> None:
        d0 = active_devs[0]
        post_ln_0 = get_tensor_cpu(f"{lp}.post_attention_layernorm.weight").to(d0, dtype=torch.float16)
        gate_0 = dequant_nf4_mos(f"{lp}.mlp.gate_proj", d0)
        up_0 = dequant_nf4_mos(f"{lp}.mlp.up_proj", d0)
        down_0 = dequant_nf4_mos(f"{lp}.mlp.down_proj", d0)

        mlp_w = {d0: (post_ln_0, gate_0, up_0, down_0)}
        for d in active_devs[1:]:
            mlp_w[d] = tuple(t.to(d, non_blocking=True) for t in mlp_w[d0])
        del post_ln_0, gate_0, up_0, down_0

        max_b = max(len(dev_states[d]) for d in active_devs)
        for bi in range(max_b):
            for d in active_devs:
                if bi < len(dev_states[d]):
                    x = dev_states[d][bi][1]
                    pln, gw, uw, dw = mlp_w[d]
                    xf = x.float()
                    normed = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(torch.float16) * pln
                    del xf
                    act = F.silu(F.linear(normed, gw))
                    act.mul_(F.linear(normed, uw))
                    del normed
                    dev_states[d][bi][1] = x + F.linear(act, dw)
                    del act
        del mlp_w

    def _pool_slot(slot_i: int, apply_final_norm: bool, final_ln_cpu: torch.Tensor, active_devs: List[str], dev_states: Dict[str, list]) -> None:
        col_s = slot_i * d_model
        col_e = col_s + d_model
        for d in active_devs:
            fln = final_ln_cpu.to(d, dtype=torch.float16) if apply_final_norm else None
            for b_idx, x, att_msk_d in dev_states[d]:
                xf = x.float()
                if fln is not None:
                    x_use = ((xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(torch.float16) * fln).float()
                else:
                    x_use = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
                m = att_msk_d.unsqueeze(-1).float()
                pooled = (x_use * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
                rms = torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
                Z_out[b_idx, col_s:col_e] = (pooled * rms).cpu().numpy()
                del xf, x_use, m, pooled, rms
            del fln

    t_mos0 = time.time()
    with torch.no_grad():
        for mc_idx, m_specs in enumerate(macro_chunks):
            open_shards.clear()
            emb_w = get_tensor_cpu(f"{base_pfx}embed_tokens.weight")
            dev_states = {d: [] for d in devices}
            for bi, (b_idx, inp_ids, att_msk) in enumerate(m_specs):
                d = devices[bi % len(devices)]
                x = F.embedding(inp_ids, emb_w).to(d, dtype=torch.float16)
                dev_states[d].append([b_idx, x, att_msk.to(d)])
            del emb_w
            active_devs = [d for d in devices if len(dev_states[d]) > 0]
            final_ln_cpu = get_tensor_cpu(norm_key)

            for l in range(max_target_layer):
                lp = f"{base_pfx}layers.{l}"
                needed_from_l = {
                    idx_map[k]
                    for k in idx_map
                    if k == norm_key
                    or (k.startswith(layers_pfx) and int(k[len(layers_pfx) :].split(".")[0]) >= l)
                }
                for s_name in list(open_shards.keys()):
                    if s_name not in needed_from_l:
                        del open_shards[s_name]

                _step_attn(lp, active_devs, dev_states)
                _step_mlp(lp, active_devs, dev_states)

                if l in capture_map:
                    _pool_slot(
                        capture_map[l],
                        apply_final_norm=(l == n_layers - 1),
                        final_ln_cpu=final_ln_cpu,
                        active_devs=active_devs,
                        dev_states=dev_states,
                    )

                if (l + 1) % log_every == 0 or (l + 1) == max_target_layer:
                    gc.collect()
                    vram_gb = (
                        torch.cuda.memory_allocated(0) / (1024 ** 3)
                        if torch.cuda.is_available()
                        else 0.0
                    )
                    print(
                        f"  [MoS Stream - {tag}] Macro {mc_idx+1}/{len(macro_chunks)} | "
                        f"Layer {l+1}/{max_target_layer} | Elapsed {time.time()-t_mos0:.1f}s | "
                        f"GPU0 Post-Layer Alloc {vram_gb:.2f} GiB",
                        flush=True,
                    )

            del dev_states, final_ln_cpu
            open_shards.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return Z_out
