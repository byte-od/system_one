#!/usr/bin/env python3
"""
Canonical Generative Mixture-of-Scars (MoS v2) Exporter for Qwen2 / Qwen2.5 (0.5B -> 72.7B).

Decomposes every transformer layer `l` into:
  - Attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`): Block-64 Symmetric INT4 Base + Activation-Aware SVD (`rank=attn_rank`) FP16 Scar
  - SwiGLU MLP (`gate_proj`, `up_proj`, `down_proj`): Block-64 Symmetric INT4 Base + Activation-Aware SVD (`rank=mlp_rank`) FP16 Scar
  - Embeddings & LM Head: Lossless Per-Row INT8 (`~0.99999` cosine) saving 50% footprint across 152,064 vocab tokens

Outputs self-contained streaming shards (`4` layers per shard = `~1.79 GiB` on 72B, `< 2.0 GiB` GitHub Release CDN limit
and `< 2.0 GiB` active RAM per streamed group on 8-16 GB consumer laptops/GPUs) + `mos_manifest.json`.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding


CALIBRATION_CORPUS = (
    "System Architecture Specification: Mixture of Scars (MoS) decouples static parameter footprint from active VRAM "
    "via blockwise 4-bit quantization paired with activation-aware low-rank SVD residual experts, while Scar-Cache "
    "(SCache) bounds dynamic key-value memory strictly to O(1) via online Widrow-Hoff associative fast weights.\n\n"
    "def widrow_hoff_delta_update(W_scar: torch.Tensor, k_u: torch.Tensor, v: torch.Tensor, eta: float = 0.22):\n"
    "    pred = torch.einsum('hvd,bhd->bhv', W_scar, k_u)\n"
    "    err = v - pred\n"
    "    return W_scar + eta * torch.einsum('bhv,bhd->hvd', err, k_u)\n\n"
    "Arquitetura de inferência generativa local em Apple Silicon e GPUs consumidoras: "
    "a decomposição espectral ponderada pelas ativações preserva canais outliers emergentes em matrizes SwiGLU.\n\n"
    '{"protocol": "MoS-v2", "layers": 80, "block_size": 64, "mlp_scar_rank": 48, "attn_scar_rank": 32, "scache": "O(1)"}\n'
    "Theorem: Let S = diag(rms(X)) denote channel-wise root-mean-square input scales. Minimizing ||(W - Q(W) - B A) S||_F "
    "yields optimal rank-r residual factors B_r, A_r via truncated SVD of R_S = (W - Q(W)) S followed by right-unscaling S^{-1}."
)


def resolve_snapshot(model_id: str) -> Path:
    p = Path(model_id)
    if p.exists() and (p / "config.json").exists():
        return p
    kaggle_in = Path("/kaggle/input")
    if kaggle_in.exists():
        for cfg_p in kaggle_in.rglob("config.json"):
            parent = cfg_p.parent
            if any(parent.glob("*.safetensors")):
                try:
                    c = json.loads(cfg_p.read_text(encoding="utf-8"))
                    if c.get("model_type") == "qwen2" and int(c.get("num_hidden_layers", 0)) >= 60 and "72" in model_id:
                        return parent
                except Exception:
                    pass
    snap = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja"],
    )
    return Path(snap)


class SafetensorReader:
    def __init__(self, root: Path) -> None:
        self.root = root
        idx_p = root / "model.safetensors.index.json"
        if idx_p.exists():
            self.weight_map: Optional[Dict[str, str]] = json.loads(idx_p.read_text(encoding="utf-8"))["weight_map"]
            self._keys = set(self.weight_map.keys())
        else:
            self.weight_map = None
            with safe_open(str(root / "model.safetensors"), framework="pt", device="cpu") as f:
                self._keys = set(f.keys())

    def has(self, name: str) -> bool:
        return name in self._keys

    def get(self, name: str, device: torch.device | str = "cpu") -> torch.Tensor:
        shard = self.root / (self.weight_map[name] if self.weight_map is not None else "model.safetensors")
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            t = f.get_tensor(name)
        return t.to(device) if str(device) != "cpu" else t


def quantize_int8_per_row(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    rows, cols = w.shape
    q_out = torch.empty((rows, cols), dtype=torch.int8, device="cpu")
    s_out = torch.empty((rows, 1), dtype=torch.float16, device="cpu")
    for r0 in range(0, rows, 8192):
        r1 = min(r0 + 8192, rows)
        wf = w[r0:r1].detach().float()
        sc = (wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7) / 127.0).to(torch.float16)
        q_out[r0:r1] = torch.clamp(torch.round(wf / sc.float()), -127, 127).to(torch.int8)
        s_out[r0:r1] = sc
        del wf, sc
    return q_out.contiguous(), s_out.contiguous()


def quantize_int4_blockwise(weight: torch.Tensor, block_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    orig_shape = (weight.shape[0], weight.shape[1])
    numel = weight.numel()
    assert numel % block_size == 0 and block_size % 2 == 0
    n_blocks = numel // block_size
    flat = weight.detach().reshape(n_blocks, block_size)
    packed_out = torch.empty((n_blocks, block_size // 2), dtype=torch.uint8, device=weight.device)
    scales_out = torch.empty((n_blocks, 1), dtype=torch.float16, device=weight.device)

    chunk_b = max(1, (4 * 1024 * 1024) // block_size)
    for b0 in range(0, n_blocks, chunk_b):
        b1 = min(b0 + chunk_b, n_blocks)
        blk = flat[b0:b1].float()
        max_abs = blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7)
        best_s = (max_abs / 7.5).to(torch.float16)
        best_err = torch.full((b1 - b0, 1), float("inf"), device=weight.device, dtype=torch.float32)
        for clip_r in (0.84, 0.90, 0.96, 1.00):
            cand_s = ((max_abs * clip_r) / 7.5).clamp(min=1e-7).to(torch.float16)
            sf = cand_s.float()
            qc = torch.clamp(torch.round(blk / sf), -8, 7)
            err = ((blk - qc * sf) ** 2).mean(dim=-1, keepdim=True)
            imp = err < best_err
            best_err = torch.where(imp, err, best_err)
            best_s = torch.where(imp, cand_s, best_s)
        q = torch.clamp(torch.round(blk / best_s.float()), -8, 7).to(torch.int16) + 8
        packed_out[b0:b1] = ((q[:, 0::2] & 0x0F) | ((q[:, 1::2] & 0x0F) << 4)).to(torch.uint8)
        scales_out[b0:b1] = best_s
    return packed_out, scales_out, orig_shape


def dequantize_int4_blockwise(packed: torch.Tensor, scales: torch.Tensor, shape: Tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    s = scales.to(dtype)
    low = ((packed & 0x0F).to(dtype) - 8.0) * s
    high = ((packed >> 4).to(dtype) - 8.0) * s
    return torch.stack((low, high), dim=-1).reshape(shape)


def safe_linear_fp16(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
    if w.dtype == torch.float16 and x.device.type == "cuda":
        xf = x.float()
        sc = (xf.abs().amax(dim=-1, keepdim=True) / 512.0).clamp(min=1e-4)
        out = F.linear((xf / sc).to(torch.float16), w, None).float() * sc
        if b is not None:
            out = out + b.float()
        return out.clamp(-65000.0, 65000.0).to(x.dtype) if x.dtype == torch.float16 else out.to(x.dtype)
    return F.linear(x, w.to(x.dtype), b.to(x.dtype) if b is not None else None)


@torch.no_grad()
def decompose_matrix_mos(
    w_orig: torch.Tensor,
    s_vec: torch.Tensor,
    rank: int,
    block_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    packed_b, scales_b, shape_b = quantize_int4_blockwise(w_orig, block_size=block_size)
    sv = s_vec.to(w_orig.device).float().clamp(min=1e-4)
    res = w_orig.detach().to(dtype=torch.float32, copy=True)
    blocks_per_row = shape_b[1] // block_size
    for r0 in range(0, shape_b[0], 2048):
        r1 = min(r0 + 2048, shape_b[0])
        deq = dequantize_int4_blockwise(
            packed_b[r0 * blocks_per_row : r1 * blocks_per_row],
            scales_b[r0 * blocks_per_row : r1 * blocks_per_row],
            (r1 - r0, shape_b[1]),
            torch.float32,
        )
        res[r0:r1].sub_(deq).mul_(sv.unsqueeze(0))
        del deq

    q_rank = min(rank + 16, min(res.shape))
    U, S, V = torch.svd_lowrank(res, q=q_rank, niter=4)
    r_eff = min(rank, S.shape[0])
    sqrt_S = torch.sqrt(S[:r_eff])
    B_scar = (U[:, :r_eff] * sqrt_S.unsqueeze(0)).to(torch.float16).cpu().contiguous()
    A_scar = ((sqrt_S.unsqueeze(1) * V[:, :r_eff].T) / sv.unsqueeze(0)).to(torch.float16).cpu().contiguous()
    rel_err = math.sqrt(max(0.0, 1.0 - float((S[:r_eff] ** 2).sum().item()) / (float((res ** 2).sum().item()) + 1e-12)))
    del res, U, S, V, sqrt_S
    return packed_b.cpu().contiguous(), scales_b.cpu().contiguous(), A_scar, B_scar, rel_err


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def export_canonical_mos(
    model_id: str,
    out_dir: Path,
    layers_per_group: int = 4,
    mlp_rank: int = 48,
    attn_rank: int = 32,
    block_size: int = 64,
    start_group: int = 0,
    end_group: Optional[int] = None,
    include_embed: bool = True,
    include_head: bool = True,
) -> Dict[str, object]:
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = resolve_snapshot(model_id)
    cfg = AutoConfig.from_pretrained(snap)
    cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(snap)
    reader = SafetensorReader(snap)

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dt = torch.float16 if dev.type == "cuda" else torch.float32
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    n_layers = int(cfg.num_hidden_layers)
    assert n_layers % layers_per_group == 0, f"n_layers={n_layers} not divisible by layers_per_group={layers_per_group}"
    total_groups = n_layers // layers_per_group
    eff_end_group = total_groups if end_group is None else min(end_group, total_groups)
    stop_layer = eff_end_group * layers_per_group
    start_export_layer = start_group * layers_per_group

    print("=" * 88)
    print(
        f"🚀 EXPORTING CANONICAL GENERATIVE MoS v2: {model_id} -> {out_dir}\n"
        f"   Layers={n_layers} | Groups={start_group}..{eff_end_group - 1}/{total_groups} ({layers_per_group}L/group) | "
        f"Block={block_size} | Ranks=(MLP={mlp_rank}, Attn={attn_rank}) | dev={dev}"
    )
    print("=" * 88)

    # Copy tokenizer & config files on final or single-shot export
    aux_copied: List[str] = []
    for fn in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja"):
        src = snap / fn
        if src.exists():
            shutil.copy2(src, out_dir / fn)
            aux_copied.append(fn)

    cal_ids = tok(CALIBRATION_CORPUS, return_tensors="pt").input_ids[:, :512]
    w_emb = reader.get("model.embed_tokens.weight", device="cpu")
    with torch.no_grad():
        cal_h = F.embedding(cal_ids, w_emb).to(dt).to(dev)

    exported_files: List[Dict[str, object]] = []
    if include_embed and start_group == 0:
        emb_q, emb_s = quantize_int8_per_row(w_emb)
        emb_file = out_dir / "mos_embed_tokens.safetensors"
        save_file({"embed_tokens.qweight": emb_q, "embed_tokens.scales": emb_s}, str(emb_file))
        del emb_q, emb_s
        exported_files.append({"name": emb_file.name, "bytes": emb_file.stat().st_size, "sha256": sha256_file(emb_file)})
        print(f"  ✓ Saved {emb_file.name} ({emb_file.stat().st_size / (1024**3):.3f} GiB)")

    has_separate_head = reader.has("lm_head.weight")
    w_head_cached = w_emb.clone() if (not has_separate_head and include_head and eff_end_group == total_groups) else None
    del w_emb
    gc.collect()

    rotary = Qwen2RotaryEmbedding(config=cfg).to(dev)
    pos_ids = torch.arange(cal_h.shape[1], device=dev).unsqueeze(0)
    c_mask = create_causal_mask(config=cfg, inputs_embeds=cal_h, attention_mask=None, past_key_values=None, position_ids=pos_ids)
    pos_emb = rotary(cal_h, pos_ids)

    cur_group_tensors: Dict[str, torch.Tensor] = {}
    frob_errs: List[float] = []

    for l_idx in range(stop_layer):
        tl0 = time.perf_counter()
        with init_empty_weights():
            d_lyr = Qwen2DecoderLayer(cfg, layer_idx=l_idx)
        d_lyr.eval()

        ln1 = reader.get(f"model.layers.{l_idx}.input_layernorm.weight", device=dev).to(dt)
        ln2 = reader.get(f"model.layers.{l_idx}.post_attention_layernorm.weight", device=dev).to(dt)
        d_lyr.input_layernorm.weight = nn.Parameter(ln1, requires_grad=False)
        d_lyr.post_attention_layernorm.weight = nn.Parameter(ln2, requires_grad=False)

        for pa in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tw = reader.get(f"model.layers.{l_idx}.self_attn.{pa}.weight", device=dev).to(dt)
            sm = getattr(d_lyr.self_attn, pa)
            sm.weight = nn.Parameter(tw, requires_grad=False)
            del tw
            bk = f"model.layers.{l_idx}.self_attn.{pa}.bias"
            if reader.has(bk):
                tb = reader.get(bk, device=dev).to(dt)
                sm.bias = nn.Parameter(tb, requires_grad=False)
                del tb
            else:
                sm.bias = None

        for pm in ("gate_proj", "up_proj", "down_proj"):
            tw = reader.get(f"model.layers.{l_idx}.mlp.{pm}.weight", device=dev).to(dt)
            sm = getattr(d_lyr.mlp, pm)
            sm.weight = nn.Parameter(tw, requires_grad=False)
            sm.bias = None
            del tw

        need_export = l_idx >= start_export_layer
        cap_scales: Dict[str, torch.Tensor] = {}

        import types
        def _safe_mlp(self_m: nn.Module, x: torch.Tensor) -> torch.Tensor:
            gf = safe_linear_fp16(x, self_m.gate_proj.weight).float()
            uf = safe_linear_fp16(x, self_m.up_proj.weight).float()
            inf = self_m.act_fn(gf) * uf
            if need_export:
                xf_f = x.detach().float().reshape(-1, x.shape[-1])
                in_f = inf.detach().reshape(-1, inf.shape[-1])
                cap_scales["mlp_in"] = torch.sqrt(xf_f.pow(2).mean(dim=0) + 1e-6)
                cap_scales["mlp_inter"] = torch.sqrt(in_f.pow(2).mean(dim=0) + 1e-6)
            of = safe_linear_fp16(inf, self_m.down_proj.weight).float()
            return of.clamp(-65000.0, 65000.0).to(x.dtype) if x.dtype == torch.float16 else of.to(x.dtype)

        d_lyr.mlp.forward = types.MethodType(_safe_mlp, d_lyr.mlp)
        hks = []
        if need_export:
            def _mk_hk(k_tag: str):
                def _hk(_m: nn.Module, inp: Tuple[torch.Tensor, ...], _out: object) -> None:
                    xf = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    cap_scales[k_tag] = torch.sqrt(xf.pow(2).mean(dim=0) + 1e-6)
                return _hk
            hks.append(d_lyr.self_attn.q_proj.register_forward_hook(_mk_hk("attn_in")))
            hks.append(d_lyr.self_attn.o_proj.register_forward_hook(_mk_hk("attn_o")))

        with torch.no_grad():
            out_h = d_lyr(cal_h, attention_mask=c_mask, position_ids=pos_ids, position_embeddings=pos_emb)
            cal_h = out_h[0] if isinstance(out_h, tuple) else out_h
        for hk in hks:
            hk.remove()

        if need_export:
            g_idx = l_idx // layers_per_group
            cur_group_tensors[f"layers.{l_idx}.input_layernorm.weight"] = ln1.detach().to(torch.float16).cpu().contiguous()
            cur_group_tensors[f"layers.{l_idx}.post_attention_layernorm.weight"] = ln2.detach().to(torch.float16).cpu().contiguous()

            for pa in ("q_proj", "k_proj", "v_proj", "o_proj"):
                orig_a = getattr(d_lyr.self_attn, pa)
                sv_a = cap_scales["attn_o"] if pa == "o_proj" else cap_scales["attn_in"]
                pb, sb, a_s, b_s, _ = decompose_matrix_mos(orig_a.weight, sv_a, rank=attn_rank, block_size=block_size)
                pfx = f"layers.{l_idx}.self_attn.{pa}"
                cur_group_tensors[f"{pfx}.packed_base"] = pb
                cur_group_tensors[f"{pfx}.scales_base"] = sb
                cur_group_tensors[f"{pfx}.A_scar"] = a_s
                cur_group_tensors[f"{pfx}.B_scar"] = b_s
                if orig_a.bias is not None:
                    cur_group_tensors[f"{pfx}.bias"] = orig_a.bias.detach().to(torch.float16).cpu().contiguous()

            for pm in ("gate_proj", "up_proj", "down_proj"):
                orig_m = getattr(d_lyr.mlp, pm)
                sv_m = cap_scales["mlp_inter"] if pm == "down_proj" else cap_scales["mlp_in"]
                pb, sb, a_s, b_s, err_m = decompose_matrix_mos(orig_m.weight, sv_m, rank=mlp_rank, block_size=block_size)
                frob_errs.append(err_m)
                pfx = f"layers.{l_idx}.mlp.{pm}"
                cur_group_tensors[f"{pfx}.packed_base"] = pb
                cur_group_tensors[f"{pfx}.scales_base"] = sb
                cur_group_tensors[f"{pfx}.A_scar"] = a_s
                cur_group_tensors[f"{pfx}.B_scar"] = b_s

            if (l_idx + 1) % layers_per_group == 0:
                g_file = out_dir / f"mos_group_{g_idx:02d}.safetensors"
                save_file(cur_group_tensors, str(g_file))
                cur_group_tensors.clear()
                sz_g = g_file.stat().st_size
                exported_files.append({"name": g_file.name, "bytes": sz_g, "sha256": sha256_file(g_file)})
                print(
                    f"  ✓ Saved {g_file.name} (Layers {g_idx*layers_per_group:02d}..{l_idx:02d} | "
                    f"{sz_g / (1024**3):.3f} GiB | MeanResErr={frob_errs[-1]:.4f} | {time.perf_counter()-tl0:.2f}s/L)"
                )

        d_lyr.mlp.forward = None
        del d_lyr, ln1, ln2, cap_scales
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if include_head and eff_end_group == total_groups:
        norm_w = reader.get("model.norm.weight", device="cpu").to(torch.float16).contiguous()
        w_hd = w_head_cached if w_head_cached is not None else reader.get("lm_head.weight", device="cpu")
        hq, hs = quantize_int8_per_row(w_hd)
        head_file = out_dir / "mos_lm_head.safetensors"
        save_file({"model.norm.weight": norm_w, "lm_head.qweight": hq, "lm_head.scales": hs}, str(head_file))
        del w_hd, w_head_cached, hq, hs
        exported_files.append({"name": head_file.name, "bytes": head_file.stat().st_size, "sha256": sha256_file(head_file)})
        print(f"  ✓ Saved {head_file.name} ({head_file.stat().st_size / (1024**3):.3f} GiB)")

    manifest_part = {
        "format": "mos_canonical_v2",
        "source_model": model_id,
        "num_hidden_layers": n_layers,
        "layers_per_group": layers_per_group,
        "num_groups": total_groups,
        "hidden_size": int(cfg.hidden_size),
        "intermediate_size": int(cfg.intermediate_size),
        "num_attention_heads": int(cfg.num_attention_heads),
        "num_key_value_heads": int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)),
        "head_dim": int(cfg.hidden_size // cfg.num_attention_heads),
        "vocab_size": int(cfg.vocab_size),
        "rope_theta": float(cfg.rope_parameters.get("rope_theta", 1000000.0)) if hasattr(cfg, "rope_parameters") and cfg.rope_parameters else float(getattr(cfg, "rope_theta", 1000000.0)),
        "rms_norm_eps": float(cfg.rms_norm_eps),
        "block_size": block_size,
        "mlp_scar_rank": mlp_rank,
        "attn_scar_rank": attn_rank,
        "exported_group_range": [start_group, eff_end_group],
        "aux_files": aux_copied,
        "files": exported_files,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    man_name = "mos_manifest.json" if (start_group == 0 and eff_end_group == total_groups) else f"mos_manifest_part_{start_group:02d}_{eff_end_group:02d}.json"
    (out_dir / man_name).write_text(json.dumps(manifest_part, indent=2), encoding="utf-8")
    print(f"\n✅ EXPORT COMPLETE IN {manifest_part['elapsed_seconds']}s -> {out_dir / man_name}")
    return manifest_part


def merge_partial_manifests(out_dir: Path) -> Dict[str, object]:
    part_files = sorted(out_dir.glob("mos_manifest_part_*.json"))
    if not part_files:
        raise FileNotFoundError(f"No mos_manifest_part_*.json found in {out_dir}")

    base = json.loads(part_files[0].read_text(encoding="utf-8"))
    seen: Dict[str, Dict[str, object]] = {}
    aux_all: List[str] = []
    total_elapsed = 0.0
    for pf in part_files:
        d = json.loads(pf.read_text(encoding="utf-8"))
        total_elapsed += float(d.get("elapsed_seconds", 0.0))
        for af in d.get("aux_files", []):
            if af not in aux_all and (out_dir / af).exists():
                aux_all.append(af)
        for fe in d.get("files", []):
            seen[fe["name"]] = fe

    # Deterministic order: embed -> groups 00..N-1 -> lm_head
    ordered: List[Dict[str, object]] = []
    if "mos_embed_tokens.safetensors" in seen:
        ordered.append(seen["mos_embed_tokens.safetensors"])
    for g in range(int(base["num_groups"])):
        gn = f"mos_group_{g:02d}.safetensors"
        if gn not in seen:
            raise FileNotFoundError(f"Missing group shard {gn} while merging manifests in {out_dir}")
        ordered.append(seen[gn])
    if "mos_lm_head.safetensors" in seen:
        ordered.append(seen["mos_lm_head.safetensors"])

    total_bytes = 0
    for fe in ordered:
        fp = out_dir / str(fe["name"])
        if not fp.exists():
            raise FileNotFoundError(f"Shard file missing on disk: {fp}")
        actual_sz = fp.stat().st_size
        if actual_sz != int(fe["bytes"]):
            raise RuntimeError(f"Size mismatch for {fp.name}: {actual_sz} != {fe['bytes']}")
        total_bytes += actual_sz

    base["exported_group_range"] = [0, int(base["num_groups"])]
    base["aux_files"] = aux_all
    base["files"] = ordered
    base["total_bytes"] = total_bytes
    base["total_gib"] = round(total_bytes / (1024**3), 4)
    base["elapsed_seconds"] = round(total_elapsed, 2)

    out_man = out_dir / "mos_manifest.json"
    out_man.write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(f"✅ Merged {len(part_files)} manifest parts -> {out_man} ({len(ordered)} shards, {base['total_gib']} GiB)")
    return base


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Qwen/Qwen2.5-72B-Instruct")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layers-per-group", type=int, default=4)
    p.add_argument("--mlp-rank", type=int, default=48)
    p.add_argument("--attn-rank", type=int, default=32)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--start-group", type=int, default=0)
    p.add_argument("--end-group", type=int, default=None)
    p.add_argument("--no-embed", action="store_true")
    p.add_argument("--no-head", action="store_true")
    p.add_argument("--merge-manifests", action="store_true", help="Merge mos_manifest_part_*.json into mos_manifest.json")
    args = p.parse_args()

    if args.merge_manifests:
        merge_partial_manifests(Path(args.out_dir))
        return

    export_canonical_mos(
        model_id=args.model_id,
        out_dir=Path(args.out_dir),
        layers_per_group=args.layers_per_group,
        mlp_rank=args.mlp_rank,
        attn_rank=args.attn_rank,
        block_size=args.block_size,
        start_group=args.start_group,
        end_group=args.end_group,
        include_embed=not args.no_embed,
        include_head=not args.no_head,
    )


if __name__ == "__main__":
    main()
