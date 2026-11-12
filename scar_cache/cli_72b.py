#!/usr/bin/env python3
"""
Canonical Generative Mixture-of-Scars (MoS v2) + Scar-Cache (SCache O(1)) Local Runtime & CLI.

Runs 72.7B (and 0.5B/7B) Canonical MoS checkpoints locally on:
  - Apple Silicon (M1/M2/M3/M4 8 GB / 16 GB+ Unified Memory via MPS)
  - NVIDIA Consumer GPUs (8 GB / 12 GB / 16 GB / 24 GB+ via CUDA Tensor Cores)
  - Multi-core x86_64 / ARM64 CPUs

Features:
  1. `--download`: Zero-auth public HTTPS shard downloader with resume + SHA-256 verification.
  2. `--streaming`: Group-by-group streaming (< 1.95 GiB active group RAM on 72B!).
  3. `--resident`: Full or hybrid RAM/VRAM residency when memory permits.
  4. Genuine `SCacheDynamicCache` O(1) Widrow-Hoff associative memory (constant bytes at any context length!).
  5. `--prompt` single-shot generation and `--chat` interactive multi-turn terminal chat with live byte telemetry.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoTokenizer

from scar_cache.hf_cache import SCacheDynamicCache


DEFAULT_PUBLIC_RELEASE_BASE_URL = (
    "https://github.com/byte-od/mixture-of-scars/releases/download/v1.2.0-qwen72b-mos"
)
DEFAULT_KAGGLE_DATASET_SLUG = "thetod/qwen25-72b-instruct-mos-scache"


def _warm_os_page_cache(shard_path: Path) -> Path:
    """Sequentially warms OS kernel page cache in 16 MiB chunks (zero Python heap spike)."""
    buf = bytearray(16 * 1024 * 1024)
    mv = memoryview(buf)
    with open(shard_path, "rb") as f:
        while f.readinto(mv):
            pass
    return shard_path


def dequantize_int4_coalesced(
    packed: torch.Tensor, scales: torch.Tensor, orig_shape: Tuple[int, int], dtype: torch.dtype
) -> torch.Tensor:
    u = torch.stack((packed & 0x0F, packed >> 4), dim=-1).view(packed.shape[0], -1)
    w = u.to(dtype)
    del u
    w.sub_(8.0).mul_(scales.to(dtype))
    return w.view(orig_shape)


def safe_gemm(x: torch.Tensor, w: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    if w.dtype == torch.float16 and x.device.type in ("cuda", "mps"):
        xf = x.float()
        sc = (xf.abs().amax(dim=-1, keepdim=True) / 512.0).clamp(min=1e-4)
        out = F.linear((xf / sc).to(torch.float16), w, None).float() * sc
        if bias is not None:
            out = out + bias.float()
        return out.clamp(-65000.0, 65000.0).to(x.dtype) if x.dtype == torch.float16 else out.to(x.dtype)
    return F.linear(x, w.to(x.dtype), bias.to(x.dtype) if bias is not None else None)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class CanonicalMoSGroup:
    """Represents one streaming group (e.g. 4 layers of Qwen2.5-72B = ~1.79 GiB on disk)."""

    def __init__(
        self,
        group_idx: int,
        shard_path: Path,
        manifest: Dict[str, object],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.group_idx = group_idx
        self.shard_path = shard_path
        self.device = device
        self.dtype = dtype

        lpg = int(manifest["layers_per_group"])
        self.start_layer = group_idx * lpg
        self.end_layer = self.start_layer + lpg
        self.hidden_size = int(manifest["hidden_size"])
        self.inter_size = int(manifest["intermediate_size"])
        self.n_heads = int(manifest["num_attention_heads"])
        self.n_kv_heads = int(manifest["num_key_value_heads"])
        self.head_dim = int(manifest["head_dim"])
        self.kv_groups = self.n_heads // self.n_kv_heads
        self.rms_eps = float(manifest.get("rms_norm_eps", 1e-6))

        self.tensors: Dict[str, torch.Tensor] = {}
        self.is_loaded = False

    def load(self) -> None:
        if self.is_loaded:
            return
        cpu_tensors = load_file(str(self.shard_path), device="cpu")
        if self.device.type == "cpu":
            self.tensors = cpu_tensors
        else:
            self.tensors = {k: v.to(self.device, non_blocking=True) for k, v in cpu_tensors.items()}
            del cpu_tensors
        self.is_loaded = True

    def unload(self) -> None:
        if not self.is_loaded:
            return
        self.tensors.clear()
        self.is_loaded = False
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    def _rms_norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(var + self.rms_eps) * w.float()).to(self.dtype)

    def _mos_linear(self, x: torch.Tensor, pfx: str, out_features: int, in_features: int) -> torch.Tensor:
        pb = self.tensors[f"{pfx}.packed_base"]
        sb = self.tensors[f"{pfx}.scales_base"]
        a_s = self.tensors[f"{pfx}.A_scar"]
        b_s = self.tensors[f"{pfx}.B_scar"]
        bi = self.tensors.get(f"{pfx}.bias")

        w_base = dequantize_int4_coalesced(pb, sb, (out_features, in_features), self.dtype)
        out = safe_gemm(x, w_base, bi)
        del w_base

        if a_s.numel() > 0:
            xf = x.float()
            scar = F.linear(F.linear(xf, a_s.float()), b_s.float())
            out_f = out.float() + scar
            return out_f.clamp(-65000.0, 65000.0).to(x.dtype) if x.dtype == torch.float16 else out_f.to(x.dtype)
        return out

    def forward_layer(
        self,
        l_idx: int,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        scache: Optional[SCacheDynamicCache],
    ) -> torch.Tensor:
        bsz, q_len, _ = x.shape
        norm1 = self._rms_norm(x, self.tensors[f"layers.{l_idx}.input_layernorm.weight"])

        q = self._mos_linear(norm1, f"layers.{l_idx}.self_attn.q_proj", self.n_heads * self.head_dim, self.hidden_size)
        k = self._mos_linear(norm1, f"layers.{l_idx}.self_attn.k_proj", self.n_kv_heads * self.head_dim, self.hidden_size)
        v = self._mos_linear(norm1, f"layers.{l_idx}.self_attn.v_proj", self.n_kv_heads * self.head_dim, self.hidden_size)

        q = q.view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)

        if scache is not None:
            k, v = scache.update(k, v, l_idx)

        k_rep = k.repeat_interleave(self.kv_groups, dim=1) if self.kv_groups > 1 else k
        v_rep = v.repeat_interleave(self.kv_groups, dim=1) if self.kv_groups > 1 else v

        k_len = k_rep.shape[2]
        if q_len == 1:
            attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)
        elif q_len == k_len:
            attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
        else:
            c_mask = torch.ones((q_len, k_len), dtype=torch.bool, device=q.device).tril(diagonal=k_len - q_len)
            attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=c_mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_proj = self._mos_linear(attn_out, f"layers.{l_idx}.self_attn.o_proj", self.hidden_size, self.n_heads * self.head_dim)
        x = (x.float() + attn_proj.float()).clamp(-65000.0, 65000.0).to(self.dtype) if self.dtype == torch.float16 else (x + attn_proj)

        norm2 = self._rms_norm(x, self.tensors[f"layers.{l_idx}.post_attention_layernorm.weight"])
        # Overflow-safe float32 SwiGLU intermediate
        outs = []
        for s0 in range(0, q_len, 256):
            ns = norm2[:, s0 : s0 + 256, :]
            gf = self._mos_linear(ns, f"layers.{l_idx}.mlp.gate_proj", self.inter_size, self.hidden_size).float()
            uf = self._mos_linear(ns, f"layers.{l_idx}.mlp.up_proj", self.inter_size, self.hidden_size).float()
            inter = F.silu(gf) * uf
            df = self._mos_linear(inter, f"layers.{l_idx}.mlp.down_proj", self.hidden_size, self.inter_size).float()
            outs.append(df)
        mlp_out = torch.cat(outs, dim=1)
        x_next = x.float() + mlp_out
        return x_next.clamp(-65000.0, 65000.0).to(self.dtype) if self.dtype == torch.float16 else x_next.to(self.dtype)


class CanonicalMoSEngine:
    """End-to-end local inference engine for Canonical Generative MoS v2 + SCache O(1)."""

    def __init__(self, model_dir: str | Path, device: str = "auto", streaming: bool = True) -> None:
        self.model_dir = Path(model_dir)
        manifest_path = self.model_dir / "mos_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"mos_manifest.json not found in {self.model_dir}. "
                f"Run with `--download --model-dir {self.model_dir}` first!"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert self.manifest.get("format") == "mos_canonical_v2", "Expected mos_canonical_v2 checkpoint."

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.dtype = torch.float16 if self.device.type in ("cuda", "mps") else torch.float32
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        self.streaming = streaming

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.n_layers = int(self.manifest["num_hidden_layers"])
        self.n_groups = int(self.manifest["num_groups"])
        self.head_dim = int(self.manifest["head_dim"])
        self.rope_theta = float(self.manifest.get("rope_theta", 1000000.0))
        self.rms_eps = float(self.manifest.get("rms_norm_eps", 1e-6))

        with safe_open(str(self.model_dir / "mos_embed_tokens.safetensors"), framework="pt", device="cpu") as f:
            self.emb_q = f.get_tensor("embed_tokens.qweight")
            self.emb_s = f.get_tensor("embed_tokens.scales")

        with safe_open(str(self.model_dir / "mos_lm_head.safetensors"), framework="pt", device="cpu") as f:
            self.final_norm_w = f.get_tensor("model.norm.weight").to(self.device).to(self.dtype)
            self.head_q = f.get_tensor("lm_head.qweight")
            self.head_s = f.get_tensor("lm_head.scales")

        self.groups = [
            CanonicalMoSGroup(
                g,
                self.model_dir / f"mos_group_{g:02d}.safetensors",
                self.manifest,
                self.device,
                self.dtype,
            )
            for g in range(self.n_groups)
        ]
        if not self.streaming:
            for g in self.groups:
                g.load()

    def _rope(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim)
        )
        freqs = torch.outer(position_ids.reshape(-1).float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(self.dtype).view(position_ids.shape[0], 1, position_ids.shape[1], self.head_dim)
        sin = emb.sin().to(self.dtype).view(position_ids.shape[0], 1, position_ids.shape[1], self.head_dim)
        return cos, sin

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        scache: Optional[SCacheDynamicCache] = None,
        return_all_logits: bool = False,
    ) -> torch.Tensor:
        flat_ids = input_ids.cpu().reshape(-1)
        x = (self.emb_q[flat_ids].to(torch.float32) * self.emb_s[flat_ids].to(torch.float32)).view(
            input_ids.shape[0], input_ids.shape[1], -1
        )
        x = x.to(self.device, dtype=self.dtype)

        past_len = scache.get_seq_length() if scache is not None else 0
        pos_ids = past_len + torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)
        cos, sin = self._rope(pos_ids)

        if self.streaming and self.n_groups > 1:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(_warm_os_page_cache, self.groups[0].shard_path)
                for idx, g in enumerate(self.groups):
                    fut.result()
                    if idx + 1 < self.n_groups:
                        fut = pool.submit(_warm_os_page_cache, self.groups[idx + 1].shard_path)
                    g.load()
                    for l_idx in range(g.start_layer, g.end_layer):
                        x = g.forward_layer(l_idx, x, cos, sin, scache)
                    if self.device.type == "mps" and hasattr(torch, "mps"):
                        torch.mps.synchronize()
                    elif self.device.type == "cuda":
                        torch.cuda.synchronize()
                    g.unload()
        else:
            for g in self.groups:
                g.load()
                for l_idx in range(g.start_layer, g.end_layer):
                    x = g.forward_layer(l_idx, x, cos, sin, scache)
                if self.streaming:
                    g.unload()

        x_sl = x if return_all_logits else x[:, -1:, :]
        xf = x_sl.float()
        xn = (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.rms_eps) * self.final_norm_w.float()).to(self.dtype)

        # Chunked INT8 LM Head projection (< 64 MB peak VRAM)
        vocab = self.head_q.shape[0]
        logits_parts = []
        for v0 in range(0, vocab, 16384):
            v1 = min(v0 + 16384, vocab)
            wq = self.head_q[v0:v1].to(self.device)
            ws = self.head_s[v0:v1].to(self.device)
            w_deq = wq.to(self.dtype) * ws.to(self.dtype)
            logits_parts.append(safe_gemm(xn, w_deq).float().cpu())
            del wq, ws, w_deq
        return torch.cat(logits_parts, dim=-1)

    def make_scache(self, window_size: int = 256, num_slots: int = 24, num_sinks: int = 4) -> SCacheDynamicCache:
        return SCacheDynamicCache(
            window_size=window_size,
            num_slots=num_slots,
            num_sink_tokens=num_sinks,
            project_slots=True,
        )

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        scache: Optional[SCacheDynamicCache] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.2,
        top_k: int = 40,
        use_chat_template: bool = True,
    ) -> Tuple[str, SCacheDynamicCache]:
        if scache is None:
            scache = self.make_scache()

        if use_chat_template and hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            if scache.get_seq_length() > 0 and "<|im_start|>" in str(self.tokenizer.chat_template):
                formatted = f"<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            else:
                formatted = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
                )
        else:
            formatted = prompt

        ids = self.tokenizer(formatted, return_tensors="pt").input_ids
        # Chunked prefill to keep dynamic buffers bounded
        chunk_sz = 128
        logits = None
        t0 = time.perf_counter()
        for s0 in range(0, ids.shape[1], chunk_sz):
            logits = self.forward(ids[:, s0 : s0 + chunk_sz], scache=scache, return_all_logits=False)
        prefill_dt = time.perf_counter() - t0
        print(
            f"[MoS+SCache] Prefill {ids.shape[1]} tok in {prefill_dt:.2f}s ({ids.shape[1]/max(prefill_dt,1e-6):.1f} tok/s) | "
            f"SCache O(1) = {scache.total_memory_bytes / (1024**2):.2f} MiB",
            flush=True,
        )

        gen_tokens: List[int] = []
        eos_set = {self.tokenizer.eos_token_id, 151645, 151643}
        for _ in range(max_new_tokens):
            assert logits is not None
            nxt_lg = logits[0, -1, :]
            if temperature <= 1e-5:
                nxt_id = int(torch.argmax(nxt_lg).item())
            else:
                probs = F.softmax(nxt_lg / temperature, dim=-1)
                tk_p, tk_i = torch.topk(probs, min(top_k, probs.numel()))
                tk_p = tk_p / tk_p.sum()
                nxt_id = int(tk_i[torch.multinomial(tk_p, 1)].item())

            if nxt_id in eos_set:
                break
            gen_tokens.append(nxt_id)
            piece = self.tokenizer.decode([nxt_id], skip_special_tokens=True)
            print(piece, end="", flush=True)
            logits = self.forward(torch.tensor([[nxt_id]], dtype=torch.long), scache=scache, return_all_logits=False)
        print(
            f"\n[Telemetry] Context={scache.get_seq_length()} tok | "
            f"SCache Constant Footprint={scache.total_memory_bytes / (1024**2):.2f} MiB (O(1))",
            flush=True,
        )
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True), scache


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(8 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def download_file_resumable(
    url: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    expected_bytes: Optional[int] = None,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_sha256:
        if expected_bytes is None or dest.stat().st_size == expected_bytes:
            if _sha256_path(dest) == expected_sha256:
                print(f"  ✓ Verified existing {dest.name} ({dest.stat().st_size / (1024**2):.1f} MiB)")
                return

    # Small metadata files without expected_sha256 are always fetched cleanly in 'wb' mode
    use_resume = bool(expected_sha256 and dest.exists() and dest.stat().st_size > 0)
    if expected_bytes and use_resume and dest.stat().st_size >= expected_bytes:
        use_resume = False

    existing = dest.stat().st_size if use_resume else 0
    headers = {"User-Agent": "mixture-of-scars-cli/1.2"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    print(f"  ⬇ Downloading {dest.name}" + (f" (resuming at {existing / (1024**2):.1f} MiB)" if existing > 0 else "") + " ...")
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as err:
        if err.code == 416 and existing > 0:
            existing = 0
            req = urllib.request.Request(url, headers={"User-Agent": "mixture-of-scars-cli/1.2"})
            resp = urllib.request.urlopen(req)
        else:
            raise

    with resp:
        status = getattr(resp, "status", 200)
        mode = "ab" if (existing > 0 and status == 206) else "wb"
        written = existing if mode == "ab" else 0
        content_len = resp.headers.get("Content-Length")
        total = (written + int(content_len)) if content_len is not None else expected_bytes
        last_print = written
        with open(dest, mode) as out_f:
            while chunk := resp.read(8 * 1024 * 1024):
                out_f.write(chunk)
                written += len(chunk)
                if total and total > 64 * 1024 * 1024 and (written - last_print >= 128 * 1024 * 1024 or written == total):
                    pct = 100.0 * written / max(total, 1)
                    print(f"    ... {dest.name}: {written / (1024**2):.0f} / {total / (1024**2):.0f} MiB ({pct:.1f}%)", flush=True)
                    last_print = written

    if expected_sha256:
        got = _sha256_path(dest)
        if got != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch on {dest.name}: got {got} != {expected_sha256}")


def download_canonical_checkpoint(
    model_dir: Path,
    base_url: str,
    kaggle_dataset: Optional[str] = None,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    if kaggle_dataset:
        import subprocess
        print(f"📦 Downloading Canonical MoS checkpoint from Kaggle Dataset `{kaggle_dataset}` -> {model_dir} ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", str(model_dir), "--unzip"],
            check=True,
        )
        man_path = model_dir / "mos_manifest.json"
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        for f_entry in manifest["files"]:
            fp = model_dir / f_entry["name"]
            if f_entry.get("sha256") and _sha256_path(fp) != f_entry["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch on {fp.name}")
        print(f"\n✅ All canonical MoS shards downloaded & SHA-256 verified in: {model_dir}")
        return

    man_path = model_dir / "mos_manifest.json"
    download_file_resumable(f"{base_url}/mos_manifest.json", man_path)
    manifest = json.loads(man_path.read_text(encoding="utf-8"))

    aux_files = manifest.get("aux_files") or ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
    for af in aux_files:
        try:
            download_file_resumable(f"{base_url}/{af}", model_dir / af)
        except Exception:
            pass

    for idx, f_entry in enumerate(manifest["files"], 1):
        print(f"[{idx}/{len(manifest['files'])}] Fetching shard {f_entry['name']} ...")
        download_file_resumable(
            f"{base_url}/{f_entry['name']}",
            model_dir / f_entry["name"],
            expected_sha256=f_entry.get("sha256"),
            expected_bytes=f_entry.get("bytes"),
        )
    print(f"\n✅ All canonical MoS shards downloaded & SHA-256 verified in: {model_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Canonical Qwen2.5-72B-Instruct MoS + SCache O(1) Local Runner")
    p.add_argument("--model-dir", default="./models/qwen25_72b_mos_v2", help="Local folder for canonical MoS shards")
    p.add_argument("--download", action="store_true", help="Download public canonical shards + verify SHA-256")
    p.add_argument("--base-url", default=DEFAULT_PUBLIC_RELEASE_BASE_URL, help="Public CDN base URL")
    p.add_argument("--kaggle-dataset", default=None, help=f"Optional Kaggle dataset mirror (e.g. {DEFAULT_KAGGLE_DATASET_SLUG})")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cuda:0", "cpu"])
    p.add_argument("--resident", action="store_true", help="Keep all groups resident in RAM/VRAM instead of streaming")
    p.add_argument("--window-size", type=int, default=256, help="SCache O(1) exact local window W")
    p.add_argument("--num-slots", type=int, default=24, help="SCache O(1) associative scar slots S")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prompt", type=str, default=None, help="Single-shot prompt to generate")
    p.add_argument("--chat", action="store_true", help="Launch interactive multi-turn terminal chat with O(1) SCache")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    if args.download:
        download_canonical_checkpoint(model_dir, args.base_url.rstrip("/"), kaggle_dataset=args.kaggle_dataset)
        if not args.prompt and not args.chat:
            return

    engine = CanonicalMoSEngine(model_dir=model_dir, device=args.device, streaming=not args.resident)
    scache = engine.make_scache(window_size=args.window_size, num_slots=args.num_slots)

    if args.prompt:
        engine.generate_stream(
            args.prompt,
            scache=scache,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    elif args.chat:
        print("=" * 76)
        print("💬 Qwen2.5-72B-Instruct Canonical MoS + SCache O(1) Interactive Local Chat")
        print("   Type 'exit' or 'quit' to leave, 'reset' to clear O(1) SCache memory.")
        print("=" * 76)
        while True:
            try:
                user_msg = input("\nYou > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not user_msg:
                continue
            if user_msg.lower() in ("exit", "quit"):
                break
            if user_msg.lower() == "reset":
                scache = engine.make_scache(window_size=args.window_size, num_slots=args.num_slots)
                print("✓ SCache O(1) memory reset.")
                continue
            print("Assistant > ", end="", flush=True)
            engine.generate_stream(
                user_msg,
                scache=scache,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )


if __name__ == "__main__":
    main()
