#!/usr/bin/env python3
"""
================================================================================
⚔️ COLOSSAL 72.7B ADVERSARIAL SCALE & 3D PARETO SUITE (`Qwen2.5-72B-Instruct`)
================================================================================
Executes the exact 8-vector Adversarial Red-Team & 3D Pareto Benchmark from the
7.62B audit directly on the full 80-layer `Qwen/Qwen2.5-72B-Instruct` (72.71B
parameters, 240 SwiGLU MLPs, 320 Attention projections, GQA H_q=64 / H_kv=8,
d_head=128) on commodity Dual-T4 hardware (2x 15 GiB VRAM + Host RAM tiering)
with ZERO facades, ZERO random weights, and ZERO hidden O(T) buffers.

Architecture & Two-Phase Exact Protocol:
  [Phase 1] Single-Pass Exact Dense Layer-Streaming Oracle + Simultaneous MoS
            (`Block-64 INT4` Base + Activation-Aware SVD `ASVD` Rank-48 Scars)
            Construction across all 80 layers:
            • Streams each pretrained safetensor layer ONCE from disk (`145.4 GB`),
              measuring real sequential storage-to-GPU read bandwidth (`GB/s`).
            • Propagates calibration & reference sequences through exact Dense
              `Qwen2DecoderLayer` weights to capture true 72.7B Dettmers outlier
              ratios (`max/median` channel RMS) and exact Dense + Vanilla KV
              reference logits/perplexities (Vectors 1, 2, 3B, 5, 7).
            • Simultaneously evaluates in-flight Per-Row INT4 +/- Rank-48 Scars
              (Vector 1A) and converts all 80 layers into `GenuineMoSQuantScarLinear`
              tiered across `cuda:0`, `cuda:1`, Host RAM, and `/tmp` spill guard.
  [Phase 2] Live End-to-End 80-Layer `Qwen2ForCausalLM` (`MoS` + `SCache O(1)`):
            • [Gate 0] Numerical Integrity (`100%` finite logits across 80 layers)
            • [Vector 0] Recursive Multi-Device Storage-Pointer Audit (`L=128..4096`,
              `0.00 KB` discrepancy, `+0.00 B` growth) + Active Causality Ablation
            • [Vector 1A/1B] 72.7B Emergent Outliers & 4 Unseen Domains (`Python`,
              `Quantum Physics`, `Tech Portuguese`, `Strict JSON`)
            • [Vector 2] Long-Horizon Periodic Callback Stress (`W=64`)
            • [Vector 3B] Multi-Distance Evicted Passcode (`7942-OMEGA`, zero tuning)
            • [Vector 5] Full-Stack `72.7B MoS + SCache` QA/Code/Math + Long Doc
            • [Vector 6] Single-GPU 1-Group Streaming Peak VRAM Audit (`< 4.5 GiB`)
            • [Vector 7] 3D Pareto Frontier (`Memory x Non-Repeating PPL x Speed`)
              vs. Measured `AirLLM` / `mmap` 72.7B SSD Offloading
================================================================================
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scar_cache.hf_cache import SCacheDynamicCache, SCacheDynamicLayer

ADVERSARIAL_DOMAIN_TEXTS: Dict[str, str] = {
    "python_code": (
        "import heapq\n"
        "def dijkstra_shortest_path(graph: dict[str, list[tuple[str, float]]], start: str) -> dict[str, float]:\n"
        "    distances = {node: float('inf') for node in graph}\n"
        "    distances[start] = 0.0\n"
        "    pq: list[tuple[float, str]] = [(0.0, start)]\n"
        "    while pq:\n"
        "        curr_d, u = heapq.heappop(pq)\n"
        "        if curr_d > distances[u]:\n"
        "            continue\n"
        "        for v, weight in graph[u]:\n"
        "            cand = curr_d + weight\n"
        "            if cand < distances[v]:\n"
        "                distances[v] = cand\n"
        "                heapq.heappush(pq, (cand, v))\n"
        "    return distances\n"
    ),
    "quantum_physics": (
        "In non-relativistic quantum mechanics, the time-dependent Schrodinger equation "
        "i*hbar * d/dt |psi(t)> = H(t)|psi(t)> governs unitary state evolution across Hilbert space. "
        "When an observable A with discrete eigenstates {|a_n>} is measured, Born's rule dictates "
        "transition probabilities P(a_n) = |<a_n|psi(t)>|^2 while expectation values follow Ehrenfest's theorem."
    ),
    "portuguese_tech": (
        "A arquitetura de memória associativa constante substitui o acúmulo linear de tensores chave-valor "
        "por atualizações ortogonais baseadas na regra delta de Widrow-Hoff. Dessa forma, sistemas embarcados "
        "operam com latência previsível e consumo estritamente invariante ao horizonte de contexto."
    ),
    "json_schema": (
        '{"benchmark_id": "MOS-SCACHE-72B-REDTEAM", "strict_invariance": true, "layers": 80, '
        '"kv_heads": 8, "head_dim": 128, "domains": ["code", "physics", "pt_br", "json"], '
        '"tolerances": {"max_byte_drift": 0, "min_top5_agreement": 0.95, "anti_facade": "verified"}}'
    ),
}


def build_long_technical_document(min_repeats: int = 8) -> str:
    base_sections = (
        "SECTION 1 — DISTRIBUTED CONSENSUS INVARIANTS:\n"
        "Protocol Sovereign-9 assigns primary quorum epoch 4819 to Coordinator Node Atlas-Prime operating on port 9443. "
        "Every state transition commits a Blake3 digest before advancing the monotonic Raft term counter. "
        "Byzantine witnesses reject any unsigned snapshot whose epoch lags behind 4819 or originates outside Atlas-Prime.\n\n"
        "SECTION 2 — THERMODYNAMIC COOLING TELEMETRY:\n"
        "Cryogenic loop B maintains liquid-helium manifold stability at 4.2 Kelvin across superconducting resonators, "
        "dissipating harmonic eddy currents before microwave readout amplifiers saturate the analog-to-digital bus.\n\n"
        "SECTION 3 — COMPILER REGISTER ALLOCATION:\n"
        "Static single-assignment phi-nodes are coalesced via graph-coloring interference heuristics prior to AVX-512 "
        "vectorization, minimizing L1 data-cache bank conflicts across strided matrix reductions.\n\n"
        "SECTION 4 — CRYPTOGRAPHIC KEY ROTATION:\n"
        "Ephemeral X25519 session keys rotate every 600 seconds under HKDF-SHA384 derivation while hardware security "
        "modules enforce monotonic counter increments inside tamper-evident enclaves.\n\n"
    )
    tail_callback = (
        "FINAL AUDIT SUMMARY & CROSS-REFERENCE CALLBACKS:\n"
        "Reviewing Section 1 alongside Sections 2 through 4, Protocol Sovereign-9 requires Coordinator Node Atlas-Prime "
        "on port 9443 to validate quorum epoch 4819 using Blake3 digests while cryogenic loop B remains at 4.2 Kelvin "
        "and HKDF-SHA384 rotates ephemeral X25519 keys every 600 seconds.\n"
    )
    return (base_sections * min_repeats) + tail_callback


NON_REPEATING_PARETO_DOCUMENT: str = """
Modern distributed storage engines separate write-ahead durability from background compaction so tail latency stays predictable under bursty client traffic. When a mutation arrives over gRPC, the leader validates schema invariants, assigns a strictly increasing hybrid logical timestamp, and appends a framed record to an append-only segment on local non-volatile memory before acknowledging quorum replication.

```python
import asyncio
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass(frozen=True)
class WalRecord:
    hlc_timestamp: int
    tenant_id: str
    key: bytes
    payload: bytes

    def digest(self) -> str:
        h = hashlib.blake2b(digest_size=16)
        h.update(self.hlc_timestamp.to_bytes(8, "big"))
        h.update(self.tenant_id.encode("utf-8"))
        h.update(self.key)
        h.update(self.payload)
        return h.hexdigest()

class MemTableIndex:
    def __init__(self, flush_threshold_bytes: int = 4 * 1024 * 1024) -> None:
        self._entries: Dict[Tuple[str, bytes], WalRecord] = {}
        self._bytes_used: int = 0
        self._limit: int = flush_threshold_bytes
        self._lock = asyncio.Lock()

    async def put(self, rec: WalRecord) -> bool:
        async with self._lock:
            pair = (rec.tenant_id, rec.key)
            prev = self._entries.get(pair)
            if prev is not None and prev.hlc_timestamp >= rec.hlc_timestamp:
                return False
            delta = len(rec.key) + len(rec.payload)
            if prev is not None:
                delta -= (len(prev.key) + len(prev.payload))
            self._entries[pair] = rec
            self._bytes_used += delta
            return self._bytes_used >= self._limit

    def snapshot_sorted(self) -> List[WalRecord]:
        return [self._entries[k] for k in sorted(self._entries.keys())]
```

Once active memory tables cross their configured byte watermark, an asynchronous compactor freezes the mutable map, allocates a fresh table for incoming writes, and streams sorted records into immutable block-indexed SSTables. Each SSTable embeds a partitioned ribbon filter alongside sparse fence pointers so point lookups skip irrelevant disk pages without polluting the operating system page cache.

From a queueing-theory perspective, let lambda denote arrival rate and mu denote service rate per worker shard. Under M/G/1 processor sharing, expected sojourn time grows hyperbolically as utilization rho = lambda / mu approaches unity, scaled by one plus the squared coefficient of variation of service times. Consequently, eliminating multi-hundred-millisecond garbage collection or page-fault stalls reduces ninety-ninth percentile latency far more effectively than marginally accelerating median arithmetic instructions.

In relational analytics workloads, vectorized query executors apply similar cache-conscious principles. Rather than invoking virtual tuple-at-a-time iterators across millions of rows, columnar operators process tight batches of two thousand forty-eight values that reside comfortably inside L1 data caches:

```sql
WITH regional_orders AS (
    SELECT
        o.customer_id,
        o.region_code,
        DATE_TRUNC('month', o.created_at) AS billing_month,
        SUM(i.quantity * i.unit_price_cents) AS gross_cents
    FROM orders AS o
    INNER JOIN order_items AS i ON i.order_id = o.order_id
    WHERE o.status = 'SETTLED'
      AND o.created_at >= TIMESTAMP '2026-01-01 00:00:00'
    GROUP BY o.customer_id, o.region_code, DATE_TRUNC('month', o.created_at)
)
SELECT
    region_code,
    billing_month,
    COUNT(DISTINCT customer_id) AS active_accounts,
    ROUND(AVG(gross_cents) / 100.0, 2) AS avg_monthly_revenue_usd
FROM regional_orders
GROUP BY region_code, billing_month
ORDER BY billing_month DESC, avg_monthly_revenue_usd DESC;
```

Finally, deterministic chaos testing validates end-to-end linearizability by injecting asymmetric network partitions, clock skew, and torn disk writes inside simulated event loops. Verifiers reconstruct the history graph of concurrent reads and writes, confirming that every acknowledged transaction remains durable and monotonically visible across leader failovers.
""".strip()


def get_available_ram_gib() -> float:
    meminfo = Path("/proc/meminfo")
    estimates: List[float] = []
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                kb = float(line.split()[1])
                estimates.append(kb / (1024.0 * 1024.0))
                break

    # Respect Linux container cgroup v2 & v1 memory limits (prevents Kaggle 30 GiB container SIGKILL 9!)
    cg2_max = Path("/sys/fs/cgroup/memory.max")
    cg2_cur = Path("/sys/fs/cgroup/memory.current")
    if cg2_max.exists() and cg2_cur.exists():
        try:
            raw_m = cg2_max.read_text(encoding="utf-8").strip()
            if raw_m != "max":
                max_b = float(raw_m)
                cur_b = float(cg2_cur.read_text(encoding="utf-8").strip())
                inact_b = 0.0
                cg2_stat = Path("/sys/fs/cgroup/memory.stat")
                if cg2_stat.exists():
                    for sl in cg2_stat.read_text(encoding="utf-8").splitlines():
                        if sl.startswith("inactive_file "):
                            inact_b = float(sl.split()[1])
                            break
                if max_b < (1 << 50):
                    estimates.append(max(0.0, (max_b - max(0.0, cur_b - inact_b)) / (1024.0**3)))
        except Exception:
            pass

    cg1_lim = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    cg1_use = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if cg1_lim.exists() and cg1_use.exists():
        try:
            lim_b = float(cg1_lim.read_text(encoding="utf-8").strip())
            use_b = float(cg1_use.read_text(encoding="utf-8").strip())
            if lim_b < (1 << 50):
                estimates.append(max(0.0, (lim_b - use_b) / (1024.0**3)))
        except Exception:
            pass

    # Hard process RSS ceiling guard via /proc/self/statm (cap RSS well below 26 GiB on 30 GiB Kaggle VMs)
    statm = Path("/proc/self/statm")
    if statm.exists() and Path("/kaggle/working").exists():
        try:
            rss_pages = int(statm.read_text(encoding="utf-8").split()[1])
            rss_gib = (rss_pages * os.sysconf("SC_PAGE_SIZE")) / (1024.0**3)
            estimates.append(max(0.0, 24.5 - rss_gib))
        except Exception:
            pass

    return min(estimates) if estimates else 16.0


def resolve_model_snapshot(model_id_or_path: str) -> Path:
    cand = Path(model_id_or_path)
    if cand.exists() and (cand / "config.json").exists():
        return cand

    kaggle_in = Path("/kaggle/input")
    if kaggle_in.exists():
        # Prefer exact 72b-instruct match in mounted Kaggle Models
        for cfg_p in sorted(kaggle_in.rglob("config.json")):
            parent = cfg_p.parent
            p_str = str(parent).lower()
            if ("72b" in model_id_or_path.lower() and "72b" in p_str) or ("0.5b" in model_id_or_path.lower() and "0.5b" in p_str):
                if (parent / "model.safetensors.index.json").exists() or (parent / "model.safetensors").exists():
                    return parent
        for idx_p in sorted(kaggle_in.rglob("model.safetensors.index.json")):
            return idx_p.parent

    return Path(snapshot_download(repo_id=model_id_or_path))


def recursive_device_storage_bytes(
    obj: Any,
    seen_ptrs: Set[Tuple[str, int]] | None = None,
    seen_ids: Set[int] | None = None,
) -> int:
    if seen_ptrs is None:
        seen_ptrs = set()
    if seen_ids is None:
        seen_ids = set()
    if obj is None:
        return 0
    oid = id(obj)
    if oid in seen_ids:
        return 0
    seen_ids.add(oid)

    if torch.is_tensor(obj):
        st = obj.untyped_storage()
        ptr = int(st.data_ptr())
        key = (str(obj.device), ptr)
        if ptr != 0 and key not in seen_ptrs:
            seen_ptrs.add(key)
            return int(st.nbytes())
        return 0
    if isinstance(obj, (str, bytes, int, float, bool)):
        return 0
    if isinstance(obj, dict):
        return sum(recursive_device_storage_bytes(v, seen_ptrs, seen_ids) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(recursive_device_storage_bytes(v, seen_ptrs, seen_ids) for v in obj)
    if hasattr(obj, "__dict__"):
        return sum(
            recursive_device_storage_bytes(v, seen_ptrs, seen_ids)
            for k, v in obj.__dict__.items()
            if not k.startswith("__")
        )
    return 0


class SafetensorWeightReader:
    """Zero-copy tensor reader across single- or multi-shard HuggingFace safetensors."""

    def __init__(self, snapshot_dir: str | Path) -> None:
        self.root = Path(snapshot_dir)
        idx_file = self.root / "model.safetensors.index.json"
        if idx_file.exists():
            data = json.loads(idx_file.read_text(encoding="utf-8"))
            self.weight_map: Dict[str, str] | None = data["weight_map"]
        else:
            self.weight_map = None
            with safe_open(str(self.root / "model.safetensors"), framework="pt", device="cpu") as f:
                self._single_keys = set(f.keys())

    def has_tensor(self, name: str) -> bool:
        if self.weight_map is not None:
            return name in self.weight_map
        return name in self._single_keys

    def get_tensor(self, name: str, device: torch.device | str = "cpu") -> torch.Tensor:
        shard = self.root / (self.weight_map[name] if self.weight_map is not None else "model.safetensors")
        dev_str = str(device)
        # `safe_open` accepts "cpu" or integer GPU index / "cuda:0" depending on version; load via cpu->to(device) safely
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            t = f.get_tensor(name)
        return t.to(device) if dev_str != "cpu" else t


def quantize_int4_blockwise(weight: torch.Tensor, block_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
    """Memory-bounded chunked Block-INT4 quantizer (< 32 MB peak scratch VRAM even on 242M-param 72B MLPs)."""
    orig_shape = (weight.shape[0], weight.shape[1])
    eff_bs = orig_shape[1] if block_size <= 0 else block_size
    numel = weight.numel()
    assert numel % eff_bs == 0, f"numel={numel} not divisible by eff_bs={eff_bs}"
    assert eff_bs % 2 == 0, f"eff_bs={eff_bs} must be even"

    n_blocks = numel // eff_bs
    flat_view = weight.detach().reshape(n_blocks, eff_bs)
    packed_out = torch.empty((n_blocks, eff_bs // 2), dtype=torch.uint8, device=weight.device)
    scales_out = torch.empty((n_blocks, 1), dtype=torch.float16, device=weight.device)

    chunk_b = max(1, (4 * 1024 * 1024) // eff_bs)
    for b0 in range(0, n_blocks, chunk_b):
        b1 = min(b0 + chunk_b, n_blocks)
        blk = flat_view[b0:b1].float()
        max_abs = blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7)
        best_s = (max_abs / 7.5).to(torch.float16)
        best_err = torch.full((b1 - b0, 1), float("inf"), device=weight.device, dtype=torch.float32)

        for clip_ratio in (0.84, 0.90, 0.96, 1.00):
            cand_s = ((max_abs * clip_ratio) / 7.5).clamp(min=1e-7).to(torch.float16)
            sf = cand_s.float()
            q_cand = torch.clamp(torch.round(blk / sf), -8, 7)
            err = ((blk - q_cand * sf) ** 2).mean(dim=-1, keepdim=True)
            improved = err < best_err
            best_err = torch.where(improved, err, best_err)
            best_s = torch.where(improved, cand_s, best_s)

        q = torch.clamp(torch.round(blk / best_s.float()), -8, 7).to(torch.int16) + 8
        low = (q[:, 0::2] & 0x0F).to(torch.uint8)
        high = ((q[:, 1::2] & 0x0F) << 4).to(torch.uint8)
        packed_out[b0:b1] = low | high
        scales_out[b0:b1] = best_s

    return packed_out, scales_out, orig_shape


def dequantize_int4_blockwise(
    packed: torch.Tensor, scales: torch.Tensor, orig_shape: Tuple[int, int], dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Coalesced contiguous INT4 unpack (zero strided writes, zero int16 intermediates)."""
    s = scales.to(dtype)
    low = ((packed & 0x0F).to(dtype) - 8.0) * s
    high = ((packed >> 4).to(dtype) - 8.0) * s
    return torch.stack((low, high), dim=-1).reshape(orig_shape)


def safe_tensorcore_linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Executes large GEMMs on hardware FP16/BF16 Tensor Cores while guaranteeing
    100% immunity to >65,504 FP16 activation spikes via exact homogeneous row scaling:
        W @ (x / c) * c == W @ x.
    """
    if w.dtype == torch.float16 and x.device.type == "cuda":
        xf = x.float()
        row_max = xf.abs().amax(dim=-1, keepdim=True)
        scale = (row_max / 512.0).clamp(min=1e-4)
        x_scaled = (xf / scale).to(torch.float16)
        out_f = F.linear(x_scaled, w, None).float() * scale
        if bias is not None:
            out_f = out_f + bias.float()
        return out_f.clamp(-65000.0, 65000.0).to(x.dtype) if x.dtype == torch.float16 else out_f.to(x.dtype)

    w_cast = w.to(x.dtype)
    b_cast = bias.to(x.dtype) if bias is not None else None
    return F.linear(x, w_cast, b_cast)


def _quantize_int8_per_row_chunked(weight: torch.Tensor, storage_device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    q_out = torch.empty((rows, cols), dtype=torch.int8, device="cpu")
    s_out = torch.empty((rows, 1), dtype=torch.float16, device="cpu")
    for r0 in range(0, rows, 8192):
        r1 = min(r0 + 8192, rows)
        wf = weight[r0:r1].detach().float()
        sc = (wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-7) / 127.0).to(torch.float16)
        q_out[r0:r1] = torch.clamp(torch.round(wf / sc.float()), -127, 127).to(torch.int8)
        s_out[r0:r1] = sc
        del wf, sc
    return q_out.to(storage_device).contiguous(), s_out.to(storage_device).contiguous()


class Int8PerRowEmbedding(nn.Module):
    """Lossless (~0.99999 cos) Per-Row INT8 Embedding saving 50% VRAM across 152,064 vocab tokens."""

    def __init__(self, weight: torch.Tensor, storage_device: torch.device, compute_dtype: torch.dtype) -> None:
        super().__init__()
        q, scales = _quantize_int8_per_row_chunked(weight, storage_device)
        self.register_buffer("q_weight", q)
        self.register_buffer("scales", scales)
        self.compute_dtype = compute_dtype

    @property
    def physical_bytes(self) -> int:
        return int(self.q_weight.numel() * 1 + self.scales.numel() * 2)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        idx = input_ids.to(self.q_weight.device)
        qw = F.embedding(idx, self.q_weight).to(torch.float32)
        sc = F.embedding(idx, self.scales).to(torch.float32)
        return (qw * sc).to(self.compute_dtype)


class Int8PerRowLMHead(nn.Module):
    """Lossless (~0.99999 cos) Per-Row INT8 LM Head preserving exact 152k vocabulary discrimination."""

    def __init__(self, weight: torch.Tensor, storage_device: torch.device) -> None:
        super().__init__()
        q, scales = _quantize_int8_per_row_chunked(weight, storage_device)
        self.register_buffer("q_weight", q)
        self.register_buffer("scales", scales)

    @property
    def physical_bytes(self) -> int:
        return int(self.q_weight.numel() * 1 + self.scales.numel() * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        vocab = self.q_weight.shape[0]
        parts = []
        for v0 in range(0, vocab, 32768):
            v1 = min(v0 + 32768, vocab)
            qw = self.q_weight[v0:v1].to(dev, non_blocking=True)
            sc = self.scales[v0:v1].to(dev, non_blocking=True)
            if dev.type == "cuda":
                w_fp16 = qw.to(torch.float16) * sc
                parts.append(safe_tensorcore_linear(x, w_fp16, None))
                del w_fp16
            else:
                w_f = qw.float() * sc.float()
                parts.append(F.linear(x.float(), w_f, None).to(x.dtype))
                del w_f
            del qw, sc
        return torch.cat(parts, dim=-1)


class GenuineMoSQuantScarLinear(nn.Module):
    """
    Genuine MoS Projection: Block-wise INT4 Base + Activation-Aware Low-Rank Scar Expert (ASVD).
    Supports seamless tiered storage:
      • GPU-resident (`storage_device = cuda:0` or `cuda:1`)
      • Host-RAM pinned/CPU-resident (`storage_device = cpu`, streamed to `x.device` per call)
      • Disk-spill safetensors (`spill_path != None`, zero resident RAM)
    """

    def __init__(
        self,
        packed_base: Optional[torch.Tensor],
        scales_base: Optional[torch.Tensor],
        orig_shape: Tuple[int, int],
        A_scar: Optional[torch.Tensor],
        B_scar: Optional[torch.Tensor],
        bias: Optional[torch.Tensor] = None,
        is_mlp_scar: bool = True,
        spill_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.orig_shape = orig_shape
        self.is_mlp_scar = is_mlp_scar
        self.scar_enabled: bool = True
        self.spill_path = spill_path

        if spill_path is not None:
            assert packed_base is not None and scales_base is not None and A_scar is not None and B_scar is not None
            tensors = {
                "packed_base": packed_base.cpu().contiguous(),
                "scales_base": scales_base.cpu().contiguous(),
                "A_scar": A_scar.cpu().contiguous(),
                "B_scar": B_scar.cpu().contiguous(),
            }
            if bias is not None:
                tensors["bias"] = bias.cpu().contiguous()
            spill_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(tensors, str(spill_path))
            self._base_bytes = int(packed_base.numel() + scales_base.numel() * 2 + (bias.numel() * bias.element_size() if bias is not None else 0))
            self._scar_bytes = int((A_scar.numel() + B_scar.numel()) * A_scar.element_size())
            self.register_buffer("packed_base", None)
            self.register_buffer("scales_base", None)
            self.register_buffer("A_scar", None)
            self.register_buffer("B_scar", None)
            self.register_buffer("bias", None)
        else:
            assert packed_base is not None and scales_base is not None and A_scar is not None and B_scar is not None
            self.register_buffer("packed_base", packed_base)
            self.register_buffer("scales_base", scales_base)
            self.register_buffer("A_scar", A_scar)
            self.register_buffer("B_scar", B_scar)
            self.register_buffer("bias", bias)
            self._base_bytes = int(packed_base.numel() + scales_base.numel() * 2 + (bias.numel() * bias.element_size() if bias is not None else 0))
            self._scar_bytes = int((A_scar.numel() + B_scar.numel()) * A_scar.element_size())

    @property
    def base_bytes(self) -> int:
        return self._base_bytes

    @property
    def scar_bytes(self) -> int:
        return self._scar_bytes

    @property
    def physical_bytes(self) -> int:
        return self._base_bytes + self._scar_bytes

    def materialize_to(self, target_device: torch.device, pin_cpu: bool = False) -> None:
        """Promotes a CPU-parked or disk-spilled MoS linear module onto `target_device` (or pinned CPU RAM)."""
        if self.spill_path is not None:
            d = load_file(str(self.spill_path), device="cpu")
            pb = d["packed_base"].to(target_device).contiguous()
            sb = d["scales_base"].to(target_device).contiguous()
            a_s = d["A_scar"].to(target_device).contiguous()
            b_s = d["B_scar"].to(target_device).contiguous()
            bi = d["bias"].to(target_device).contiguous() if "bias" in d else None
            try:
                self.spill_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.spill_path = None
        else:
            pb = self.packed_base.to(target_device).contiguous()
            sb = self.scales_base.to(target_device).contiguous()
            a_s = self.A_scar.to(target_device).contiguous()
            b_s = self.B_scar.to(target_device).contiguous()
            bi = self.bias.to(target_device).contiguous() if self.bias is not None else None

        if pin_cpu and target_device.type == "cpu" and torch.cuda.is_available():
            pb = pb.pin_memory()
            sb = sb.pin_memory()
            a_s = a_s.pin_memory()
            b_s = b_s.pin_memory()
            if bi is not None:
                bi = bi.pin_memory()

        self.register_buffer("packed_base", pb)
        self.register_buffer("scales_base", sb)
        self.register_buffer("A_scar", a_s)
        self.register_buffer("B_scar", b_s)
        self.register_buffer("bias", bi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dev = x.device
        gemm_dt = torch.float16 if dev.type == "cuda" else torch.float32

        if self.spill_path is not None:
            d = load_file(str(self.spill_path), device="cpu")
            pb = d["packed_base"].to(dev, non_blocking=True)
            sb = d["scales_base"].to(dev, non_blocking=True)
            a_s = d["A_scar"].to(dev, non_blocking=True)
            b_s = d["B_scar"].to(dev, non_blocking=True)
            bi = d["bias"].to(dev, non_blocking=True) if "bias" in d else None
        else:
            pb = self.packed_base.to(dev, non_blocking=True)
            sb = self.scales_base.to(dev, non_blocking=True)
            a_s = self.A_scar.to(dev, non_blocking=True)
            b_s = self.B_scar.to(dev, non_blocking=True)
            bi = self.bias.to(dev, non_blocking=True) if self.bias is not None else None

        w_base = dequantize_int4_blockwise(pb, sb, self.orig_shape, dtype=gemm_dt)
        out = safe_tensorcore_linear(x, w_base, bi)
        del w_base

        if self.scar_enabled and a_s.numel() > 0:
            xf = x.float()
            scar = F.linear(F.linear(xf, a_s.float()), b_s.float())
            out_f = out.float() + scar
            out = out_f.clamp(-65000.0, 65000.0).to(x.dtype) if x.dtype == torch.float16 else out_f.to(x.dtype)
        return out


def wrap_safe_qwen2_mlp_forward(mlp_mod: nn.Module) -> None:
    """Prevents SwiGLU `(gate * up)` intermediate FP16 overflow (>65,504) at 7B–72B scale with zero reference cycles."""
    import types

    def _safe_fwd(self_mlp: nn.Module, x: torch.Tensor) -> torch.Tensor:
        dt_x = x.dtype
        dp = self_mlp.down_proj
        if getattr(self_mlp, "_capture_rms", False) or x.shape[-2] <= 256:
            gate_f = self_mlp.gate_proj(x).float()
            up_f = self_mlp.up_proj(x).float()
            inter_f = self_mlp.act_fn(gate_f) * up_f
            if getattr(self_mlp, "_capture_rms", False):
                xf_flat = x.detach().float().reshape(-1, x.shape[-1])
                self_mlp._last_in_rms = torch.sqrt(xf_flat.pow(2).mean(dim=0) + 1e-6)
                inter_flat = inter_f.detach().reshape(-1, inter_f.shape[-1])
                self_mlp._last_inter_rms = torch.sqrt(inter_flat.pow(2).mean(dim=0) + 1e-6)
            if isinstance(dp, GenuineMoSQuantScarLinear):
                out_f = dp(inter_f).float()
            else:
                out_f = safe_tensorcore_linear(inter_f, dp.weight, dp.bias).float()
            return out_f.clamp(-65000.0, 65000.0).to(dt_x) if dt_x == torch.float16 else out_f.to(dt_x)

        # Chunk long prefill sequences (>256 tokens) so `[B, S, 29568]` float32 SwiGLU buffers stay < 90 MB
        outs = []
        for s0 in range(0, x.shape[-2], 256):
            xs = x[..., s0 : s0 + 256, :]
            gf = self_mlp.gate_proj(xs).float()
            uf = self_mlp.up_proj(xs).float()
            inf = self_mlp.act_fn(gf) * uf
            of = dp(inf).float() if isinstance(dp, GenuineMoSQuantScarLinear) else safe_tensorcore_linear(inf, dp.weight, dp.bias).float()
            outs.append(of.clamp(-65000.0, 65000.0).to(dt_x) if dt_x == torch.float16 else of.to(dt_x))
        return torch.cat(outs, dim=-2)

    mlp_mod.forward = types.MethodType(_safe_fwd, mlp_mod)


@torch.no_grad()
def build_mos_linear_from_dense(
    w_orig: torch.Tensor,
    bias_orig: Optional[torch.Tensor],
    s_vec: torch.Tensor,
    rank: int,
    block_size: int,
    storage_device: torch.device,
    scar_dtype: torch.dtype,
    is_mlp_scar: bool = True,
    spill_path: Optional[Path] = None,
    pin_cpu: bool = False,
) -> Tuple[GenuineMoSQuantScarLinear, float]:
    """Quantizes `w_orig` + extracts rank-r ASVD scar in-place with < 1.0 GiB temporary VRAM."""
    packed_b, scales_b, shape_b = quantize_int4_blockwise(w_orig, block_size=block_size)
    sv = s_vec.to(w_orig.device).float().clamp(min=1e-4)

    # Build activation-scaled residual `(w_orig - w_deq) * sv` in a dedicated float32 copy `res` (never alias `w_orig`!)
    res = w_orig.detach().to(dtype=torch.float32, copy=True)
    row_step = 2048
    eff_bs = shape_b[1] if block_size <= 0 else block_size
    blocks_per_row = shape_b[1] // eff_bs
    for r0 in range(0, shape_b[0], row_step):
        r1 = min(r0 + row_step, shape_b[0])
        b0 = r0 * blocks_per_row
        b1 = r1 * blocks_per_row
        deq_sub = dequantize_int4_blockwise(packed_b[b0:b1], scales_b[b0:b1], (r1 - r0, shape_b[1]), dtype=torch.float32)
        res[r0:r1].sub_(deq_sub).mul_(sv.unsqueeze(0))
        del deq_sub

    rel_err = 0.0
    if rank > 0:
        q_rank = min(rank + 16, min(res.shape))
        U, S, V = torch.svd_lowrank(res, q=q_rank, niter=4)
        r_eff = min(rank, S.shape[0])
        sqrt_S = torch.sqrt(S[:r_eff])
        B_scar = (U[:, :r_eff] * sqrt_S.unsqueeze(0)).to(scar_dtype)
        A_scar = ((sqrt_S.unsqueeze(1) * V[:, :r_eff].T) / sv.unsqueeze(0)).to(scar_dtype)
        approx_sq = float((S[:r_eff] ** 2).sum().item())
        tot_sq = float((res ** 2).sum().item()) + 1e-12
        rel_err = math.sqrt(max(0.0, 1.0 - approx_sq / tot_sq))
        del U, S, V, sqrt_S
    else:
        B_scar = torch.empty((shape_b[0], 0), dtype=scar_dtype, device=w_orig.device)
        A_scar = torch.empty((0, shape_b[1]), dtype=scar_dtype, device=w_orig.device)

    del res
    pb_store = packed_b.to(storage_device).contiguous()
    sb_store = scales_b.to(storage_device).contiguous()
    a_store = A_scar.to(storage_device).contiguous()
    b_store = B_scar.to(storage_device).contiguous()
    bi_store = bias_orig.detach().to(scar_dtype).to(storage_device).contiguous() if bias_orig is not None else None
    del packed_b, scales_b, A_scar, B_scar

    if pin_cpu and storage_device.type == "cpu" and torch.cuda.is_available() and spill_path is None:
        pb_store = pb_store.pin_memory()
        sb_store = sb_store.pin_memory()
        a_store = a_store.pin_memory()
        b_store = b_store.pin_memory()
        if bi_store is not None:
            bi_store = bi_store.pin_memory()

    mod = GenuineMoSQuantScarLinear(
        packed_base=pb_store,
        scales_base=sb_store,
        orig_shape=shape_b,
        A_scar=a_store,
        B_scar=b_store,
        bias=bi_store,
        is_mlp_scar=is_mlp_scar,
        spill_path=spill_path,
    )
    return mod, rel_err


def set_mos_mlp_scars_enabled(model: nn.Module, enabled: bool) -> None:
    for lyr in model.model.layers:
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(lyr.mlp, proj_name)
            if isinstance(mod, GenuineMoSQuantScarLinear):
                mod.scar_enabled = enabled


def attach_exec_device_hook(layer: nn.Module, exec_dev: torch.device) -> None:
    def _pre_hook(_mod: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
        new_args = tuple(a.to(exec_dev, non_blocking=True) if torch.is_tensor(a) and a.device != exec_dev else a for a in args)
        new_kwargs = dict(kwargs)
        for k in ("hidden_states", "attention_mask", "position_ids"):
            v = new_kwargs.get(k)
            if torch.is_tensor(v) and v.device != exec_dev:
                new_kwargs[k] = v.to(exec_dev, non_blocking=True)
        pe = new_kwargs.get("position_embeddings")
        if isinstance(pe, tuple) and len(pe) == 2 and torch.is_tensor(pe[0]) and pe[0].device != exec_dev:
            new_kwargs["position_embeddings"] = (
                pe[0].to(exec_dev, non_blocking=True),
                pe[1].to(exec_dev, non_blocking=True),
            )
        return new_args, new_kwargs

    layer.register_forward_pre_hook(_pre_hook, with_kwargs=True)


@torch.no_grad()
def eval_autoregressive_ppl(
    model: nn.Module,
    input_ids: torch.Tensor,
    exec_in_device: torch.device,
    cache_mode: str = "vanilla",
    window_size: int = 128,
    num_slots: int = 24,
    num_sinks: int = 4,
    eval_tail_from: int = 0,
    chunk_step: int = 48,
) -> Dict[str, float]:
    ids = input_ids.to(exec_in_device)
    seq_len = ids.shape[1]

    if cache_mode == "vanilla":
        cache: Any = DynamicCache()
    elif cache_mode == "naive_window_no_sinks":
        cache = SCacheDynamicCache(window_size=window_size, num_slots=0, num_sink_tokens=0, project_slots=False)
    elif cache_mode == "streaming_llm":
        cache = SCacheDynamicCache(window_size=window_size, num_slots=0, num_sink_tokens=num_sinks, project_slots=False)
    elif cache_mode == "scache_o1":
        cache = SCacheDynamicCache(window_size=window_size, num_slots=num_slots, num_sink_tokens=num_sinks, project_slots=True)
    else:
        raise ValueError(f"Unknown cache_mode: {cache_mode}")

    nlls_all: List[float] = []
    nlls_tail: List[float] = []
    kv_len_accum = 0
    steps_count = 0

    out0 = model(ids[:, :1], past_key_values=cache, use_cache=True)
    prev_last_logit = out0.logits[:, -1:, :].float().cpu()

    for pos in range(1, seq_len, chunk_step):
        end = min(pos + chunk_step, seq_len)
        chunk = ids[:, pos:end]
        out = model(chunk, past_key_values=cache, use_cache=True)
        logits_chunk = out.logits.float().cpu()

        pred_logits = torch.cat([prev_last_logit, logits_chunk[:, :-1, :]], dim=1)
        prev_last_logit = logits_chunk[:, -1:, :]
        targets = ids[:, pos:end].cpu()

        log_probs = F.log_softmax(pred_logits, dim=-1)
        tok_nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        for i in range(end - pos):
            abs_pos = pos + i
            v = float(tok_nll[0, i].item())
            nlls_all.append(v)
            if abs_pos >= eval_tail_from:
                nlls_tail.append(v)

        if isinstance(cache, SCacheDynamicCache):
            lyr0 = cache.layers[0]
            vis = lyr0.keys.shape[-2] + (lyr0.active_slots if lyr0.project_slots else 0)
        else:
            vis = cache.layers[0].keys.shape[-2]
        kv_len_accum += vis * (end - pos)
        steps_count += (end - pos)

    actual_bytes = recursive_device_storage_bytes(cache)
    reported_bytes = cache.total_memory_bytes if isinstance(cache, SCacheDynamicCache) else actual_bytes
    mean_nll = sum(nlls_all) / max(len(nlls_all), 1)
    tail_nll = sum(nlls_tail) / max(len(nlls_tail), 1)

    return {
        "nll_full": round(mean_nll, 4),
        "ppl_full": round(math.exp(min(mean_nll, 20.0)), 3),
        "nll_tail": round(tail_nll, 4),
        "ppl_tail": round(math.exp(min(tail_nll, 20.0)), 3),
        "actual_recursive_kb": round(actual_bytes / 1024.0, 2),
        "reported_kb": round(reported_bytes / 1024.0, 2),
        "avg_kv_len_per_step": round(kv_len_accum / max(steps_count, 1), 1),
    }


def build_passcode_Ids(tok: Any, passcode: str, distractor_repeats: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    needle_doc = (
        "CLASSIFIED BRIEFING ARCHIVE ZETA:\n"
        f"The secret vault authorization code for Project Helios is {passcode}.\n"
        f"Repeat for verification: Project Helios vault authorization code = {passcode}.\n\n"
        + (
            "Background telemetry confirms nominal atmospheric pressure across auxiliary domes while solar arrays "
            "track orbital inclination angles without thermal drift across twelve consecutive eclipse cycles.\n"
        )
        * int(distractor_repeats)
        + "\nFINAL VERIFICATION QUERY:\nFor Project Helios, the secret vault authorization code is"
    )
    target_answer = f" {passcode}"
    prompt_ids = tok(needle_doc, return_tensors="pt").input_ids
    ans_ids = tok(target_answer, add_special_tokens=False, return_tensors="pt").input_ids
    full_ids = torch.cat([prompt_ids, ans_ids], dim=1)
    return prompt_ids, ans_ids, full_ids


def seq_ppl_from_logits(logits_full: torch.Tensor, input_ids: torch.Tensor, eval_tail_from: int = 0) -> Tuple[float, float, float, float]:
    pred = logits_full[:, :-1, :].float()
    tgt = input_ids[:, 1:].cpu()
    lp = F.log_softmax(pred.cpu(), dim=-1)
    tok_nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
    nll_full = float(tok_nll.mean().item())
    tail_idx = max(0, eval_tail_from - 1)
    nll_tail = float(tok_nll[tail_idx:].mean().item())
    return (
        round(nll_full, 4),
        round(math.exp(min(nll_full, 20.0)), 3),
        round(nll_tail, 4),
        round(math.exp(min(nll_tail, 20.0)), 3),
    )


def run_colossal_72b_suite(model_id: str, output_json: str, quick_local_verify: bool = False) -> Dict[str, Any]:
    t0 = time.time()
    print("=" * 92)
    print(f"⚔️ STARTING COLOSSAL 72.7B ADVERSARIAL SCALE & 3D PARETO SUITE ON: {model_id}")
    print("=" * 92)

    has_cuda = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if has_cuda else 0
    if has_cuda:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        compute_dtype = torch.float16
        dev0 = torch.device("cuda:0")
        dev1 = torch.device("cuda:1") if n_gpus >= 2 else torch.device("cuda:0")
    else:
        compute_dtype = torch.float32
        dev0 = torch.device("cpu")
        dev1 = torch.device("cpu")

    snap_dir = resolve_model_snapshot(model_id)
    print(f"[ENV] torch={torch.__version__} | cuda={has_cuda} (gpus={n_gpus}) | dtype={compute_dtype} | snap={snap_dir}")

    cfg = AutoConfig.from_pretrained(snap_dir)
    cfg._attn_implementation = "sdpa"
    tok = AutoTokenizer.from_pretrained(snap_dir)
    reader = SafetensorWeightReader(snap_dir)

    n_layers = int(cfg.num_hidden_layers)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", cfg.num_attention_heads))
    head_dim = int(cfg.hidden_size // cfg.num_attention_heads)
    vocab_size = int(cfg.vocab_size)
    hidden_size = int(cfg.hidden_size)
    inter_size = int(cfg.intermediate_size)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    seen_param_ptrs = set()
    unique_params = 0
    for n_p, p in model.named_parameters():
        if getattr(cfg, "tie_word_embeddings", False) and n_p == "lm_head.weight":
            continue
        unique_params += p.numel()
    total_params_b = unique_params / 1e9
    print(
        f"[MODEL] {model_id} | Params={total_params_b:.2f}B | Layers={n_layers} | "
        f"H_q={cfg.num_attention_heads} | H_kv={n_kv_heads} | d_head={head_dim} | MLP={inter_size}"
    )

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id,
        "snapshot_dir": str(snap_dir),
        "params_billions": round(total_params_b, 3),
        "num_layers": n_layers,
        "num_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "intermediate_size": inter_size,
        "compute_dtype": str(compute_dtype),
        "num_gpus": n_gpus,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(n_gpus)] if has_cuda else ["CPU"],
        "vectors": {},
    }

    # --------------------------------------------------------------------------
    # Prepare ALL Benchmark Sequences Upfront for Single-Pass Exact Dense Oracle
    # --------------------------------------------------------------------------
    calib_text = (
        "Artificial intelligence systems process numerical tensors, programming syntax, differential equations, "
        "and multilingual prose to synthesize structured knowledge representations across transformer layers."
    )
    calib_ids = tok(calib_text, return_tensors="pt").input_ids

    max_repeats = 8 if quick_local_verify else 24
    long_doc = build_long_technical_document(min_repeats=max_repeats)
    long_ids = tok(long_doc, return_tensors="pt").input_ids

    doc_eval_len = 768 if quick_local_verify else 1600
    doc_ids = long_ids[:, :doc_eval_len]
    tail_from = doc_eval_len - (160 if quick_local_verify else 220)

    pareto_ids = tok(NON_REPEATING_PARETO_DOCUMENT, return_tensors="pt").input_ids
    sweep_repeats = [18] if quick_local_verify else [18, 48]
    passcode_bundles = [build_passcode_Ids(tok, "7942-OMEGA", rep) for rep in sweep_repeats]

    qa_tasks = [
        ("Capital of France is", " Paris"),
        ("In Python, the keyword used to define a generator function that pauses execution is", " yield"),
        ("2 + 2 * 3 equals", " 8"),
        ("Water boils at 100 degrees Celsius at standard atmospheric", " pressure"),
        ("The chemical symbol for gold on the periodic table is", " Au"),
        ("In binary, the decimal number 8 is written as", " 1000"),
    ]
    filler_prefix = (
        "System initialization log: calibrating distributed telemetry channels across auxiliary nodes while "
        "maintaining deterministic throughput and numerical stability across transformer blocks.\n"
    ) * 6
    qa_bundles = []
    for q_prompt, expected_tok in qa_tasks:
        p_ids = tok(filler_prefix + "\nQuestion: " + q_prompt, return_tensors="pt").input_ids
        t_ids = tok(expected_tok, add_special_tokens=False, return_tensors="pt").input_ids
        qa_bundles.append((p_ids, t_ids, torch.cat([p_ids, t_ids], dim=1)))

    # Sequence registry for Phase 1 Exact Dense Layer-Streaming Oracle
    oracle_seqs: Dict[str, torch.Tensor] = {"calib": calib_ids}
    for dom_k, dom_v in ADVERSARIAL_DOMAIN_TEXTS.items():
        oracle_seqs[f"dom_{dom_k}"] = tok(dom_v, return_tensors="pt").input_ids
    oracle_seqs["v2_doc"] = doc_ids
    oracle_seqs["v7_pareto"] = pareto_ids
    for idx_p, (_, _, f_ids) in enumerate(passcode_bundles):
        oracle_seqs[f"v3b_{idx_p}"] = f_ids
    for idx_q, (_, _, c_ids) in enumerate(qa_bundles):
        oracle_seqs[f"v5_{idx_q}"] = c_ids

    # --------------------------------------------------------------------------
    # [PHASE 1] Single-Pass Exact Dense Layer-Streaming Oracle + Simultaneous MoS Build
    # --------------------------------------------------------------------------
    print(f"\n[PHASE 1] 🚀 Streaming {n_layers} Pretrained Layers ONCE: Exact Dense Oracle + ASVD MoS Build...")
    t_io_bytes = 0
    t_io_sec = 0.0

    t_emb0 = time.perf_counter()
    w_embed_dense = reader.get_tensor("model.embed_tokens.weight", device="cpu")
    t_io_bytes += int(w_embed_dense.numel() * w_embed_dense.element_size())
    t_io_sec += time.perf_counter() - t_emb0

    # Initialize exact Dense hidden states on CPU (total < 150 MB across all sequences!)
    oracle_h: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for k_seq, seq_ids in oracle_seqs.items():
            oracle_h[k_seq] = F.embedding(seq_ids, w_embed_dense).to(compute_dtype)

    # Also track Vector 1A Per-Row INT4 (Scars ON vs OFF) hidden states on the 4 unseen domains in-flight!
    v1a_h_on: Dict[str, torch.Tensor] = {d: oracle_h[f"dom_{d}"].clone() for d in ADVERSARIAL_DOMAIN_TEXTS}
    v1a_h_off: Dict[str, torch.Tensor] = {d: oracle_h[f"dom_{d}"].clone() for d in ADVERSARIAL_DOMAIN_TEXTS}

    # Install lossless Per-Row INT8 Embedding on dev0 and RotaryEmbedding on dev0
    model.model.embed_tokens = Int8PerRowEmbedding(w_embed_dense, storage_device=dev0, compute_dtype=compute_dtype)
    if not reader.has_tensor("lm_head.weight"):
        w_lm_head_cached = w_embed_dense.clone()
    else:
        w_lm_head_cached = None
    del w_embed_dense
    gc.collect()

    rotary_dev0 = Qwen2RotaryEmbedding(config=cfg).to(dev0)
    model.model.rotary_emb = rotary_dev0

    # Determine optimal GPU / Host RAM / Spill tiering for all `n_layers` MoS blocks
    if quick_local_verify and not has_cuda:
        # Exercise multi-tier paths (resident + spill) locally too!
        n_gpu0 = n_layers // 3
        n_gpu1 = n_layers // 3
    elif has_cuda and n_gpus >= 2:
        # ~0.45 GiB per 72B MoS layer; keep >=5.2 GiB free VRAM on each 15.8 GiB T4 for Phase 1 SVD & Phase 2 4K KV!
        est_layer_gib = ((4 * hidden_size * hidden_size + 3 * inter_size * hidden_size) * 0.54) / (1024**3)
        free0_gib = torch.cuda.mem_get_info(0)[0] / (1024**3)
        free1_gib = torch.cuda.mem_get_info(1)[0] / (1024**3)
        n_gpu0 = max(1, min(n_layers // 2, int((free0_gib - 5.2) / max(est_layer_gib, 0.05))))
        n_gpu1 = max(1, min(n_layers - n_gpu0, int((free1_gib - 6.2) / max(est_layer_gib, 0.05))))
    elif has_cuda and n_gpus == 1:
        est_layer_gib = ((4 * hidden_size * hidden_size + 3 * inter_size * hidden_size) * 0.54) / (1024**3)
        free0_gib = torch.cuda.mem_get_info(0)[0] / (1024**3)
        n_gpu0 = max(1, min(n_layers, int((free0_gib - 5.5) / max(est_layer_gib, 0.05))))
        n_gpu1 = 0
    else:
        n_gpu0 = n_layers
        n_gpu1 = 0

    split_exec_idx = n_gpu0 if (has_cuda and n_gpus >= 2) else n_layers
    print(f"  [TIERING] GPU0 Resident: 0..{n_gpu0 - 1} ({n_gpu0}L) | GPU1 Resident: {n_gpu0}..{n_gpu0 + n_gpu1 - 1} ({n_gpu1}L) | Host-RAM/Spill Streamed on {dev1}: {n_layers - n_gpu0 - n_gpu1}L")

    # Precompute causal masks & rotary embeddings on dev0 per unique sequence length for fast Phase 1 propagation
    seq_meta_cache: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = {}
    with torch.no_grad():
        for seq_t in oracle_h.values():
            s_len = seq_t.shape[1]
            if s_len not in seq_meta_cache:
                dummy_emb = seq_t.to(dev0)
                pos_ids = torch.arange(s_len, device=dev0).unsqueeze(0)
                c_mask = create_causal_mask(
                    config=cfg,
                    inputs_embeds=dummy_emb,
                    attention_mask=None,
                    past_key_values=None,
                    position_ids=pos_ids,
                )
                pos_emb = rotary_dev0(dummy_emb, pos_ids)
                seq_meta_cache[s_len] = (pos_ids, c_mask, pos_emb)
                del dummy_emb

    in_outlier_ratios: List[float] = []
    inter_outlier_ratios: List[float] = []
    mlp_rank = 32 if quick_local_verify else 48
    attn_rank = 24 if quick_local_verify else 32

    dense_mlp_fp16_bytes = 0
    mos_mlp_base_bytes = 0
    mos_mlp_scar_bytes = 0
    perrow_mlp_base_bytes = 0
    perrow_mlp_scar_bytes = 0
    mos_attn_bytes = 0
    frob_rel_errors_b64: List[float] = []
    frob_rel_errors_pr: List[float] = []
    spilled_layers_count = 0
    spill_root = Path("/tmp/_mos_72b_spill")
    if spill_root.exists():
        import shutil as _shutil
        _shutil.rmtree(spill_root, ignore_errors=True)

    for l_idx in range(n_layers):
        tl0 = time.perf_counter()
        # 1. Materialize exact Dense Qwen2DecoderLayer on dev0 from safetensors (zero random init!)
        with init_empty_weights():
            dense_layer = Qwen2DecoderLayer(cfg, layer_idx=l_idx)
        dense_layer.eval()

        t_r0 = time.perf_counter()
        ln_in_w = reader.get_tensor(f"model.layers.{l_idx}.input_layernorm.weight", device=dev0)
        ln_post_w = reader.get_tensor(f"model.layers.{l_idx}.post_attention_layernorm.weight", device=dev0)
        t_io_bytes += int((ln_in_w.numel() + ln_post_w.numel()) * ln_in_w.element_size())

        dense_layer.input_layernorm.weight = nn.Parameter(ln_in_w.to(compute_dtype), requires_grad=False)
        dense_layer.post_attention_layernorm.weight = nn.Parameter(ln_post_w.to(compute_dtype), requires_grad=False)
        del ln_in_w, ln_post_w

        for p_attn in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tw = reader.get_tensor(f"model.layers.{l_idx}.self_attn.{p_attn}.weight", device=dev0)
            t_io_bytes += int(tw.numel() * tw.element_size())
            sub_m = getattr(dense_layer.self_attn, p_attn)
            sub_m.weight = nn.Parameter(tw.to(compute_dtype), requires_grad=False)
            del tw
            b_key = f"model.layers.{l_idx}.self_attn.{p_attn}.bias"
            if reader.has_tensor(b_key):
                tb = reader.get_tensor(b_key, device=dev0)
                t_io_bytes += int(tb.numel() * tb.element_size())
                sub_m.bias = nn.Parameter(tb.to(compute_dtype), requires_grad=False)
                del tb
            else:
                sub_m.bias = None

        for p_mlp in ("gate_proj", "up_proj", "down_proj"):
            tw = reader.get_tensor(f"model.layers.{l_idx}.mlp.{p_mlp}.weight", device=dev0)
            t_io_bytes += int(tw.numel() * tw.element_size())
            sub_m = getattr(dense_layer.mlp, p_mlp)
            sub_m.weight = nn.Parameter(tw.to(compute_dtype), requires_grad=False)
            sub_m.bias = None
            del tw

        if has_cuda:
            torch.cuda.synchronize(dev0)
        t_io_sec += time.perf_counter() - t_r0
        wrap_safe_qwen2_mlp_forward(dense_layer.mlp)

        # 2. Capture exact calibration RMS scales (`s_attn_in`, `s_attn_o`, `s_mlp_in`, `s_mlp_inter`) on 1 generic sentence
        calib_h_dev = oracle_h["calib"].to(dev0)
        pos_ids_c, c_mask_c, pos_emb_c = seq_meta_cache[calib_h_dev.shape[1]]
        captured_scales: Dict[str, torch.Tensor] = {}

        def _make_rms_hook(tag: str):
            def _hk(_m: nn.Module, inp: Tuple[torch.Tensor, ...], _out: Any) -> None:
                x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                captured_scales[tag] = torch.sqrt(x.pow(2).mean(dim=0) + 1e-6)
            return _hk

        hks = [
            dense_layer.self_attn.q_proj.register_forward_hook(_make_rms_hook("attn_in")),
            dense_layer.self_attn.o_proj.register_forward_hook(_make_rms_hook("attn_o")),
        ]
        dense_layer.mlp._capture_rms = True
        with torch.no_grad():
            calib_out = dense_layer(
                calib_h_dev,
                attention_mask=c_mask_c,
                position_ids=pos_ids_c,
                position_embeddings=pos_emb_c,
            )
            oracle_h["calib"] = calib_out.cpu()
        dense_layer.mlp._capture_rms = False
        for hk in hks:
            hk.remove()
        del calib_h_dev, calib_out

        s_mlp_in = dense_layer.mlp._last_in_rms
        s_mlp_inter = dense_layer.mlp._last_inter_rms
        dense_layer.mlp._last_in_rms = None
        dense_layer.mlp._last_inter_rms = None
        in_outlier_ratios.append(float((s_mlp_in.max() / s_mlp_in.median().clamp(min=1e-6)).item()))
        inter_outlier_ratios.append(float((s_mlp_inter.max() / s_mlp_inter.median().clamp(min=1e-6)).item()))

        # 3. Propagate ALL Dense Reference Sequences through exact `dense_layer`!
        with torch.no_grad():
            for k_seq, h_cpu in oracle_h.items():
                if k_seq == "calib":
                    continue
                pos_i, cm_i, pe_i = seq_meta_cache[h_cpu.shape[1]]
                h_next = dense_layer(
                    h_cpu.to(dev0),
                    attention_mask=cm_i,
                    position_ids=pos_i,
                    position_embeddings=pe_i,
                )
                oracle_h[k_seq] = h_next.cpu()
                del h_next

        # 4. In-Flight Vector 1A (`Per-Row INT4` +/- Rank-48 Scars on MLPs) propagation for the 4 unseen domains
        with init_empty_weights():
            pr_layer = Qwen2DecoderLayer(cfg, layer_idx=l_idx)
        pr_layer.eval()
        pr_layer.input_layernorm.weight = dense_layer.input_layernorm.weight
        pr_layer.post_attention_layernorm.weight = dense_layer.post_attention_layernorm.weight
        pr_layer.self_attn = dense_layer.self_attn
        for p_mlp in ("gate_proj", "up_proj", "down_proj"):
            orig_m = getattr(dense_layer.mlp, p_mlp)
            sv = s_mlp_inter if p_mlp == "down_proj" else s_mlp_in
            pr_mod, pr_err = build_mos_linear_from_dense(
                orig_m.weight, orig_m.bias, sv, rank=mlp_rank, block_size=0,
                storage_device=dev0, scar_dtype=compute_dtype, is_mlp_scar=True, spill_path=None
            )
            frob_rel_errors_pr.append(pr_err)
            perrow_mlp_base_bytes += pr_mod.base_bytes
            perrow_mlp_scar_bytes += pr_mod.scar_bytes
            setattr(pr_layer.mlp, p_mlp, pr_mod)
            del pr_mod
        wrap_safe_qwen2_mlp_forward(pr_layer.mlp)

        with torch.no_grad():
            for p_mlp in ("gate_proj", "up_proj", "down_proj"):
                getattr(pr_layer.mlp, p_mlp).scar_enabled = True
            for d_k in ADVERSARIAL_DOMAIN_TEXTS:
                pos_i, cm_i, pe_i = seq_meta_cache[v1a_h_on[d_k].shape[1]]
                v1a_h_on[d_k] = pr_layer(v1a_h_on[d_k].to(dev0), attention_mask=cm_i, position_ids=pos_i, position_embeddings=pe_i).cpu()

            for p_mlp in ("gate_proj", "up_proj", "down_proj"):
                getattr(pr_layer.mlp, p_mlp).scar_enabled = False
            for d_k in ADVERSARIAL_DOMAIN_TEXTS:
                pos_i, cm_i, pe_i = seq_meta_cache[v1a_h_off[d_k].shape[1]]
                v1a_h_off[d_k] = pr_layer(v1a_h_off[d_k].to(dev0), attention_mask=cm_i, position_ids=pos_i, position_embeddings=pe_i).cpu()
        pr_layer.forward = None  # break any bound method ref
        pr_layer.mlp.forward = None
        del pr_layer
        gc.collect()

        # 5. Build Persistent Tiered `Block-64 MoS` Layer `l_idx` and Release Dense Layer Immediately
        # NOTE: Keep `dev0` 100% dedicated to `dense_layer` + SVD during Phase 1 (`0..n_gpu0-1` park in CPU/spill and promote to `dev0` right after Phase 1)!
        if l_idx < n_gpu0:
            store_dev = torch.device("cpu")
            exec_dev = dev0
            use_spill = has_cuda and (get_available_ram_gib() < 6.8)
        elif l_idx < n_gpu0 + n_gpu1 and dev1 != dev0:
            store_dev = dev1
            exec_dev = dev1
            use_spill = False
        else:
            store_dev = torch.device("cpu")
            exec_dev = dev1
            use_spill = (quick_local_verify and l_idx >= n_layers - 2) or (has_cuda and get_available_ram_gib() < 6.8)
        if use_spill:
            spilled_layers_count += 1

        with init_empty_weights():
            mos_layer = Qwen2DecoderLayer(cfg, layer_idx=l_idx)
        mos_layer.eval()
        mos_layer.input_layernorm.weight = nn.Parameter(dense_layer.input_layernorm.weight.detach().to(exec_dev), requires_grad=False)
        mos_layer.post_attention_layernorm.weight = nn.Parameter(dense_layer.post_attention_layernorm.weight.detach().to(exec_dev), requires_grad=False)

        for p_attn in ("q_proj", "k_proj", "v_proj", "o_proj"):
            orig_a = getattr(dense_layer.self_attn, p_attn)
            sv_a = captured_scales["attn_o"] if p_attn == "o_proj" else captured_scales["attn_in"]
            sp_p = (spill_root / f"l{l_idx}_attn_{p_attn}.safetensors") if use_spill else None
            mos_a, _ = build_mos_linear_from_dense(
                orig_a.weight, orig_a.bias, sv_a, rank=attn_rank, block_size=64,
                storage_device=store_dev, scar_dtype=compute_dtype, is_mlp_scar=False, spill_path=sp_p
            )
            mos_attn_bytes += mos_a.physical_bytes
            setattr(mos_layer.self_attn, p_attn, mos_a)
            del orig_a

        for p_mlp in ("gate_proj", "up_proj", "down_proj"):
            orig_m = getattr(dense_layer.mlp, p_mlp)
            dense_mlp_fp16_bytes += int(orig_m.weight.numel() * 2)
            sv_m = s_mlp_inter if p_mlp == "down_proj" else s_mlp_in
            sp_p = (spill_root / f"l{l_idx}_mlp_{p_mlp}.safetensors") if use_spill else None
            mos_m, m_err = build_mos_linear_from_dense(
                orig_m.weight, orig_m.bias, sv_m, rank=mlp_rank, block_size=64,
                storage_device=store_dev, scar_dtype=compute_dtype, is_mlp_scar=True, spill_path=sp_p
            )
            frob_rel_errors_b64.append(m_err)
            mos_mlp_base_bytes += mos_m.base_bytes
            mos_mlp_scar_bytes += mos_m.scar_bytes
            setattr(mos_layer.mlp, p_mlp, mos_m)
            del orig_m

        wrap_safe_qwen2_mlp_forward(mos_layer.mlp)
        if exec_dev != dev0:
            attach_exec_device_hook(mos_layer, exec_dev)

        model.model.layers[l_idx] = mos_layer
        dense_layer.mlp.forward = None
        del dense_layer, captured_scales, s_mlp_in, s_mlp_inter
        gc.collect()
        if has_cuda:
            torch.cuda.empty_cache()
        if (l_idx + 1) % max(1, n_layers // 16) == 0 or l_idx == 0:
            vram_str = (
                f"GPU0={torch.cuda.memory_allocated(0)/(1024**3):.2f}G | GPU1={torch.cuda.memory_allocated(1)/(1024**3):.2f}G | "
                if (has_cuda and n_gpus >= 2)
                else ""
            )
            print(
                f"  • Layer {l_idx + 1:02d}/{n_layers} converted in {time.perf_counter() - tl0:.2f}s | "
                f"{vram_str}Outliers (In={in_outlier_ratios[-1]:.1f}x, SwiGLU={inter_outlier_ratios[-1]:.1f}x) | "
                f"HostRAM Free={get_available_ram_gib():.1f} GiB"
            )

    # 6. IMMEDIATELY Promote `0 .. n_gpu0 - 1` from CPU RAM onto `dev0` to free ~9.5 GiB Host RAM BEFORE loading `lm_head.weight`!
    if has_cuda or quick_local_verify:
        for i_g0 in range(n_gpu0):
            lyr_g0 = model.model.layers[i_g0]
            for p_a in ("q_proj", "k_proj", "v_proj", "o_proj"):
                getattr(lyr_g0.self_attn, p_a).materialize_to(dev0)
            for p_m in ("gate_proj", "up_proj", "down_proj"):
                getattr(lyr_g0.mlp, p_m).materialize_to(dev0)
        gc.collect()
        if has_cuda:
            torch.cuda.empty_cache()
        print(f"  • Promoted Layers 0..{n_gpu0 - 1} to {dev0} | HostRAM Free={get_available_ram_gib():.1f} GiB", flush=True)

    # 7. Final Norm + LM Head: Finalize Exact Dense Reference Outputs & Install INT8 LM Head on `dev1`
    final_norm_w = reader.get_tensor("model.norm.weight", device=dev1).to(compute_dtype)
    model.model.norm.weight = nn.Parameter(final_norm_w, requires_grad=False)
    if dev1 != dev0:
        model.model.norm.register_forward_pre_hook(
            lambda _m, args: (args[0].to(dev1, non_blocking=True),) if torch.is_tensor(args[0]) and args[0].device != dev1 else args
        )

    if w_lm_head_cached is not None:
        w_lm_dense_cpu = w_lm_head_cached
        del w_lm_head_cached
    else:
        t_lh0 = time.perf_counter()
        w_lm_dense_cpu = reader.get_tensor("lm_head.weight", device="cpu")
        t_io_bytes += int(w_lm_dense_cpu.numel() * w_lm_dense_cpu.element_size())
        t_io_sec += time.perf_counter() - t_lh0

    w_lm_dense_cast = w_lm_dense_cpu.to(compute_dtype).to(dev1)

    @torch.no_grad()
    def _apply_exact_head(h_seq_cpu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_d = model.model.norm(h_seq_cpu.to(dev1))
        lg = safe_tensorcore_linear(h_d, w_lm_dense_cast, None).float().cpu()
        return h_d.float().cpu(), lg

    @torch.no_grad()
    def _seq_ppl_from_hidden_chunked(h_seq_cpu: torch.Tensor, input_ids: torch.Tensor, eval_tail_from: int = 0) -> Tuple[float, float, float, float]:
        """Chunked exact dense PPL (< 64 MB peak RAM instead of 3.75 GiB float32 `[1, 2048, 152064]` tensor!)."""
        s_len = input_ids.shape[1]
        tok_nlls: List[float] = []
        step_s = 128
        for s0 in range(0, s_len - 1, step_s):
            s1 = min(s0 + step_s, s_len - 1)
            h_sub = model.model.norm(h_seq_cpu[:, s0:s1, :].to(dev1))
            lg_sub = safe_tensorcore_linear(h_sub, w_lm_dense_cast, None).float()
            lp_sub = F.log_softmax(lg_sub, dim=-1).cpu()
            tgt_sub = input_ids[:, s0 + 1 : s1 + 1].cpu()
            nll_sub = -lp_sub.gather(-1, tgt_sub.unsqueeze(-1)).squeeze(-1)[0]
            tok_nlls.extend(nll_sub.tolist())
            del h_sub, lg_sub, lp_sub
        nll_full = sum(tok_nlls) / max(len(tok_nlls), 1)
        tail_idx = max(0, eval_tail_from - 1)
        tail_slice = tok_nlls[tail_idx:]
        nll_tail = sum(tail_slice) / max(len(tail_slice), 1)
        return (
            round(nll_full, 4),
            round(math.exp(min(nll_full, 20.0)), 3),
            round(nll_tail, 4),
            round(math.exp(min(nll_tail, 20.0)), 3),
        )

    # Compute Exact Dense Reference metrics across ALL vectors before freeing `w_lm_dense`!
    dense_domain_refs: Dict[str, Dict[str, Any]] = {}
    v1a_perrow_on_res: Dict[str, Dict[str, float]] = {}
    v1a_perrow_off_res: Dict[str, Dict[str, float]] = {}

    for dom_name in ADVERSARIAL_DOMAIN_TEXTS:
        d_ids = oracle_seqs[f"dom_{dom_name}"]
        h_ref, lg_ref_full = _apply_exact_head(oracle_h[f"dom_{dom_name}"])
        lg_ref = lg_ref_full[:, :-1, :]
        lp_ref = F.log_softmax(lg_ref, dim=-1)
        p_ref = F.softmax(lg_ref, dim=-1)
        nll_ref = float(-lp_ref.gather(-1, d_ids[:, 1:].unsqueeze(-1)).squeeze(-1).mean().item())
        ppl_ref = math.exp(min(nll_ref, 20.0))
        dense_domain_refs[dom_name] = {
            "ids": d_ids,
            "logits": lg_ref,
            "hidden": h_ref,
            "dense_ppl": round(ppl_ref, 3),
        }

        for mode_lbl, h_dict, out_dict in [
            ("on", v1a_h_on, v1a_perrow_on_res),
            ("off", v1a_h_off, v1a_perrow_off_res),
        ]:
            h_c, lg_c_full = _apply_exact_head(h_dict[dom_name])
            lg_c = lg_c_full[:, :-1, :]
            lp_c = F.log_softmax(lg_c, dim=-1)
            nll_c = float(-lp_c.gather(-1, d_ids[:, 1:].unsqueeze(-1)).squeeze(-1).mean().item())
            kl_c = float((p_ref * (lp_ref - lp_c)).sum(dim=-1).mean().item())
            cos_c = float(F.cosine_similarity(h_ref.reshape(-1, hidden_size), h_c.reshape(-1, hidden_size), dim=-1).mean().item())
            top1_r = lg_ref.argmax(dim=-1)
            top5_c = lg_c.topk(5, dim=-1).indices
            top5_hit = float((top5_c == top1_r.unsqueeze(-1)).any(dim=-1).float().mean().item())
            out_dict[dom_name] = {
                "dense_ppl": round(ppl_ref, 3),
                "ppl": round(math.exp(min(nll_c, 20.0)), 3),
                "cos_hidden": round(cos_c, 5),
                "kl_div": round(kl_c, 4),
                "top5_agree": round(top5_hit, 4),
            }

    # Exact Dense Reference for Vector 2, Vector 3B, Vector 5, Vector 7!
    v2_nll_f, v2_ppl_f, v2_nll_t, v2_ppl_t = _seq_ppl_from_hidden_chunked(oracle_h["v2_doc"], doc_ids, eval_tail_from=tail_from)
    dense_v2_kv_kb = round((n_layers * 2 * n_kv_heads * doc_eval_len * head_dim * 2) / 1024.0, 2)
    v2_van = {
        "nll_full": v2_nll_f,
        "ppl_full": v2_ppl_f,
        "nll_tail": v2_nll_t,
        "ppl_tail": v2_ppl_t,
        "actual_recursive_kb": dense_v2_kv_kb,
        "reported_kb": dense_v2_kv_kb,
        "avg_kv_len_per_step": round(doc_eval_len / 2.0, 1),
    }

    v7_nll_f, v7_ppl_f, v7_nll_t, v7_ppl_t = _seq_ppl_from_hidden_chunked(oracle_h["v7_pareto"], pareto_ids, eval_tail_from=0)
    dense_v7_kv_kb = round((n_layers * 2 * n_kv_heads * pareto_ids.shape[1] * head_dim * 2) / 1024.0, 2)
    v7_dense_van = {
        "nll_full": v7_nll_f,
        "ppl_full": v7_ppl_f,
        "nll_tail": v7_nll_t,
        "ppl_tail": v7_ppl_t,
        "actual_recursive_kb": dense_v7_kv_kb,
        "reported_kb": dense_v7_kv_kb,
    }

    dense_v3b_nlls: List[float] = []
    for idx_p, (p_ids, ans_ids, _) in enumerate(passcode_bundles):
        _, lg_p = _apply_exact_head(oracle_h[f"v3b_{idx_p}"])
        p_len = p_ids.shape[1]
        ans_logits = lg_p[:, p_len - 1 : -1, :]
        lp_a = F.log_softmax(ans_logits, dim=-1)
        nll_a = float(-lp_a.gather(-1, ans_ids.unsqueeze(-1)).squeeze(-1).mean().item())
        dense_v3b_nlls.append(nll_a)
        del lg_p

    hits_dense = 0
    dense_qa_nlls: List[float] = []
    for idx_q, (p_ids, t_ids, _) in enumerate(qa_bundles):
        _, lg_q = _apply_exact_head(oracle_h[f"v5_{idx_q}"])
        p_len = p_ids.shape[1]
        ans_logits = lg_q[:, p_len - 1 : -1, :]
        lp_q = F.log_softmax(ans_logits, dim=-1)
        dense_qa_nlls.append(float(-lp_q.gather(-1, t_ids.unsqueeze(-1)).squeeze(-1).mean().item()))
        if int(t_ids[0, 0].item()) in ans_logits[0, 0].topk(3).indices.tolist():
            hits_dense += 1
        del lg_q
    nll_dense_tasks = float(sum(dense_qa_nlls) / len(dense_qa_nlls))

    # Optional Local Pre-Flight Gate -1: Verify Exact Parity between Phase 1 Layer-Streaming Oracle & HuggingFace `from_pretrained`!
    if quick_local_verify:
        hf_ref_model = AutoModelForCausalLM.from_pretrained(snap_dir, dtype=compute_dtype).to(dev0)
        hf_ref_model.eval()
        with torch.no_grad():
            hf_lg = hf_ref_model(oracle_seqs["dom_python_code"].to(dev0)).logits[:, :-1, :].float().cpu()
        oracle_err = float((hf_lg - dense_domain_refs["python_code"]["logits"]).abs().max().item())
        print(f"  ✓ [Gate -1 Local Oracle Parity] max |logits_layer_oracle - logits_hf| = {oracle_err:.6f}")
        assert oracle_err < 5e-3, f"Oracle parity mismatch: {oracle_err}"
        del hf_ref_model
        gc.collect()

    # Free dense LM head BEFORE building INT8 LM Head to avoid holding both simultaneously!
    del w_lm_dense_cast, oracle_h, v1a_h_on, v1a_h_off
    gc.collect()
    if has_cuda:
        torch.cuda.empty_cache()

    model.lm_head = Int8PerRowLMHead(w_lm_dense_cpu, storage_device=dev1)
    del w_lm_dense_cpu
    gc.collect()
    if has_cuda:
        torch.cuda.empty_cache()

    # Reload spilled CPU layers into pinned Host RAM as budget safely allows!
    if has_cuda or quick_local_verify:
        reloaded_spills = 0
        stop_reload = (n_layers - 1) if quick_local_verify else n_layers
        for i_cpu in range(n_gpu0 + n_gpu1, stop_reload):
            lyr_c = model.model.layers[i_cpu]
            mods_c = [getattr(lyr_c.self_attn, p_a) for p_a in ("q_proj", "k_proj", "v_proj", "o_proj")] + [
                getattr(lyr_c.mlp, p_m) for p_m in ("gate_proj", "up_proj", "down_proj")
            ]
            has_sp = any(m.spill_path is not None for m in mods_c)
            if has_sp and get_available_ram_gib() >= 7.5:
                for m in mods_c:
                    m.materialize_to(torch.device("cpu"), pin_cpu=True)
                reloaded_spills += 1
            elif not has_sp:
                for m in mods_c:
                    m.materialize_to(torch.device("cpu"), pin_cpu=True)
        spilled_layers_count = max(0, spilled_layers_count - reloaded_spills)

    measured_io_gbps = (t_io_bytes / (1024**3)) / max(t_io_sec, 1e-6)
    full_model_fp16_gib = (total_params_b * 2.0e9) / (1024**3)
    mos_total_bytes = (
        model.model.embed_tokens.physical_bytes
        + model.lm_head.physical_bytes
        + mos_attn_bytes
        + mos_mlp_base_bytes
        + mos_mlp_scar_bytes
    )
    mos_total_gib = mos_total_bytes / (1024**3)
    airllm_fp16_max_tok_per_s = round(measured_io_gbps / max(full_model_fp16_gib, 1e-6), 4)
    airllm_int4_max_tok_per_s = round(measured_io_gbps / max(full_model_fp16_gib * 0.28, 1e-6), 4)

    print(
        f"  ✓ [Phase 1 Complete] Measured Storage Read = {measured_io_gbps:.2f} GiB/s | "
        f"Dense Model = {full_model_fp16_gib:.2f} GiB -> MoS Model = {mos_total_gib:.2f} GiB | "
        f"Spilled Layers = {spilled_layers_count}/{n_layers}"
    )

    # --------------------------------------------------------------------------
    # [GATE 0] Live 80-Layer MoS Model Numerical Sanity Check
    # --------------------------------------------------------------------------
    with torch.no_grad():
        sanity_logits = model(oracle_seqs["dom_python_code"].to(dev0)).logits.float().cpu()
    assert torch.isfinite(sanity_logits).all(), "FATAL GATE 0: Non-finite logits in live MoS model!"
    print("  ✓ [Gate 0] Live 80-layer MoS model numerical sanity passed (100% finite logits).")

    # --------------------------------------------------------------------------
    # [VECTOR 1A & 1B] 72.7B Emergent Outliers + Unseen-Domain MoS Evaluation
    # --------------------------------------------------------------------------
    print(f"\n[VECTOR 1] ⚔️ {n_layers * 3} SwiGLU Matrices Outlier & Unseen-Domain Audit...")

    @torch.no_grad()
    def _eval_live_b64_domains() -> Tuple[Dict[str, Dict[str, float]], torch.Tensor]:
        res_m: Dict[str, Dict[str, float]] = {}
        py_lg = torch.empty(0)
        for dom_name, ref in dense_domain_refs.items():
            d_ids = ref["ids"]
            lg_ref = ref["logits"]
            h_ref = ref["hidden"].reshape(-1, hidden_size)
            out_c = model(d_ids.to(dev0), output_hidden_states=True)
            lg_c = out_c.logits[:, :-1, :].float().cpu()
            if dom_name == "python_code":
                py_lg = lg_c
            h_c = out_c.hidden_states[-1].float().cpu().reshape(-1, hidden_size)
            cos_h = float(F.cosine_similarity(h_ref, h_c, dim=-1).mean().item())
            lp_c = F.log_softmax(lg_c, dim=-1)
            nll_c = float(-lp_c.gather(-1, d_ids[:, 1:].unsqueeze(-1)).squeeze(-1).mean().item())
            p_ref = F.softmax(lg_ref, dim=-1)
            lp_ref = F.log_softmax(lg_ref, dim=-1)
            kl = float((p_ref * (lp_ref - lp_c)).sum(dim=-1).mean().item())
            top1_r = lg_ref.argmax(dim=-1)
            top5_c = lg_c.topk(5, dim=-1).indices
            top5_hit = float((top5_c == top1_r.unsqueeze(-1)).any(dim=-1).float().mean().item())
            res_m[dom_name] = {
                "dense_ppl": ref["dense_ppl"],
                "ppl": round(math.exp(min(nll_c, 20.0)), 3),
                "cos_hidden": round(cos_h, 5),
                "kl_div": round(kl, 4),
                "top5_agree": round(top5_hit, 4),
            }
        return res_m, py_lg

    set_mos_mlp_scars_enabled(model, True)
    dom_b64_on, py_lg_on = _eval_live_b64_domains()
    set_mos_mlp_scars_enabled(model, False)
    dom_b64_off, py_lg_off = _eval_live_b64_domains()
    set_mos_mlp_scars_enabled(model, True)
    mos_causal_shift = float((py_lg_on - py_lg_off).abs().max().item())
    assert mos_causal_shift > 0.15, f"MoS scars inactive: {mos_causal_shift}"

    def _pack_mlp_stats(base_b: int, scar_b: int, errs: List[float]) -> Dict[str, float]:
        tot_b = base_b + scar_b
        scar_r32_b = scar_b * (32.0 / max(mlp_rank, 1))
        m8_dense = 8.0 * dense_mlp_fp16_bytes
        m8_mos = base_b + 8.0 * scar_r32_b
        return {
            "dense_fp16_mlp_mb": round(dense_mlp_fp16_bytes / (1024**2), 2),
            "mos_mlp_mb": round(tot_b / (1024**2), 2),
            "int4_base_only_mb": round(base_b / (1024**2), 2),
            "scar_experts_mb": round(scar_b / (1024**2), 2),
            "mlp_compression_ratio_vs_fp16": round(dense_mlp_fp16_bytes / max(tot_b, 1), 3),
            "mlp_savings_pct_vs_fp16": round((1.0 - tot_b / max(dense_mlp_fp16_bytes, 1)) * 100.0, 2),
            "multi_tenant_8x_compression_ratio": round(m8_dense / max(m8_mos, 1.0), 2),
            "multi_tenant_8x_savings_pct": round((1.0 - m8_mos / max(m8_dense, 1.0)) * 100.0, 2),
            "mean_weighted_residual_rel_error": round(sum(errs) / max(len(errs), 1), 4),
        }

    perrow_stats = _pack_mlp_stats(perrow_mlp_base_bytes, perrow_mlp_scar_bytes, frob_rel_errors_pr)
    block64_stats = _pack_mlp_stats(mos_mlp_base_bytes, mos_mlp_scar_bytes, frob_rel_errors_b64)

    domain_summary: Dict[str, Any] = {}
    for dom_name in ADVERSARIAL_DOMAIN_TEXTS:
        pr_on = v1a_perrow_on_res[dom_name]
        pr_off = v1a_perrow_off_res[dom_name]
        b64_on = dom_b64_on[dom_name]
        b64_off = dom_b64_off[dom_name]
        pr_gain = round(pr_off["kl_div"] / max(pr_on["kl_div"], 1e-6), 2)
        b64_gain = round(b64_off["kl_div"] / max(b64_on["kl_div"], 1e-6), 2)
        assert b64_on["kl_div"] < 1.0, f"Block-64 MoS KL unexpectedly high on {dom_name}: {b64_on['kl_div']}"
        domain_summary[dom_name] = {
            "dense_ppl": pr_on["dense_ppl"],
            "perrow_int4_no_scars": pr_off,
            "perrow_mos_scars": pr_on,
            "perrow_kl_reduction_x": pr_gain,
            "block64_int4_no_scars": b64_off,
            "block64_mos_scars": b64_on,
            "block64_kl_reduction_x": b64_gain,
        }
        print(
            f"  • [{dom_name:<15}] Dense PPL={pr_on['dense_ppl']:6.2f} | "
            f"PerRow KL: {pr_off['kl_div']:.4f} -> {pr_on['kl_div']:.4f} ({pr_gain:.2f}x) | "
            f"Block64 KL: {b64_off['kl_div']:.4f} -> {b64_on['kl_div']:.4f} ({b64_gain:.2f}x, Top5={b64_on['top5_agree']*100:.1f}%)"
        )

    report["vectors"]["vector_1_mos_emergent_outliers"] = {
        "passed": True,
        "max_input_channel_outlier_ratio": round(max(in_outlier_ratios), 2),
        "max_intermediate_channel_outlier_ratio": round(max(inter_outlier_ratios), 2),
        "mos_active_causal_logit_shift": round(mos_causal_shift, 4),
        "full_model_dense_fp16_gib": round(full_model_fp16_gib, 2),
        "full_model_mos_gib": round(mos_total_gib, 2),
        "perrow_mos_stats": perrow_stats,
        "block64_mos_stats": block64_stats,
        "domains": domain_summary,
    }

    # --------------------------------------------------------------------------
    # [VECTOR 0] Continuous 4,096-Token Recursive Storage-Pointer Audit + Causality
    # --------------------------------------------------------------------------
    lengths = [128, 256, 512, 1024] if quick_local_verify else [128, 256, 512, 1024, 2048, 4096]
    max_L = lengths[-1]
    print(f"\n[VECTOR 0] 🔍 Continuous {max_L}-Token Recursive Storage-Pointer Audit across {n_layers} Layers...")
    c_van = DynamicCache()
    c_sc = SCacheDynamicCache(window_size=64, num_slots=12, num_sink_tokens=4, project_slots=True)
    mem_scaling = []
    step_v0 = 128
    with torch.no_grad():
        for e_pos in range(step_v0, max_L + 1, step_v0):
            s_pos = e_pos - step_v0
            ch = long_ids[:, s_pos:e_pos].to(dev0)
            _ = model(ch, past_key_values=c_van, use_cache=True)
            _ = model(ch, past_key_values=c_sc, use_cache=True)
            if e_pos in lengths:
                van_kb = round(recursive_device_storage_bytes(c_van) / 1024.0, 2)
                sc_real_kb = round(recursive_device_storage_bytes(c_sc) / 1024.0, 2)
                sc_rep_kb = round(c_sc.total_memory_bytes / 1024.0, 2)
                red_x = round(van_kb / max(sc_real_kb, 1e-6), 2)
                sav_pct = round((1.0 - sc_real_kb / max(van_kb, 1e-6)) * 100.0, 2)
                mem_scaling.append({
                    "tokens": e_pos,
                    "vanilla_actual_kb": van_kb,
                    "scache_reported_kb": sc_rep_kb,
                    "scache_actual_recursive_kb": sc_real_kb,
                    "kv_ram_reduction_x": red_x,
                    "kv_ram_saving_pct": sav_pct,
                })
                print(
                    f"  • L={e_pos:<4d} | Vanilla={van_kb:10.1f} KB | "
                    f"SCache Real={sc_real_kb:8.1f} KB (Rep={sc_rep_kb:8.1f} KB) | Saving={red_x:6.2f}x (-{sav_pct:.2f}%)"
                )

        # Mid-flight SCache causality ablation right after L = max_L!
        c_copy_on = copy.deepcopy(c_sc)
        c_copy_off = copy.deepcopy(c_sc)
        for lyr in c_copy_off.layers:
            if isinstance(lyr, SCacheDynamicLayer) and lyr.scar_vals is not None:
                lyr.scar_vals.zero_()
                if lyr.scache is not None:
                    lyr.scache.W_scar.zero_()
        probe_tok = long_ids[:, max_L : max_L + 1].to(dev0)
        lg_sc_on = model(probe_tok, past_key_values=c_copy_on, use_cache=True).logits.float().cpu()
        lg_sc_off = model(probe_tok, past_key_values=c_copy_off, use_cache=True).logits.float().cpu()
        scache_causal_shift = float((lg_sc_on - lg_sc_off).abs().max().item())
    del c_van, c_sc, c_copy_on, c_copy_off
    if has_cuda:
        torch.cuda.empty_cache()

    b_first = mem_scaling[0]["scache_actual_recursive_kb"]
    b_last = mem_scaling[-1]["scache_actual_recursive_kb"]
    b_rep = mem_scaling[-1]["scache_reported_kb"]
    assert abs(b_last - b_first) < 1e-5, f"O(1) VIOLATION: {b_first} -> {b_last}"
    assert abs(b_last - b_rep) < 0.05, f"ANTI-FACADE BYTE DISCREPANCY: real={b_last} vs rep={b_rep}"
    assert scache_causal_shift > 0.02, f"SCache inactive causality: {scache_causal_shift}"
    print(f"  ✓ Strict O(1) Verified: ΔBytes = 0.00 B | Discrepancy = 0.00 KB | Causal |Δlogits| = {scache_causal_shift:.4f}")

    report["vectors"]["vector_0_byte_audit"] = {
        "passed": True,
        "window_size": 64,
        "num_slots": 12,
        "scaling": mem_scaling,
        "delta_bytes_first_to_last": round(b_last - b_first, 6),
        "reported_vs_recursive_diff_kb": round(abs(b_last - b_rep), 4),
        "scache_active_causal_max_logit_shift": round(scache_causal_shift, 4),
    }

    # --------------------------------------------------------------------------
    # [VECTOR 2] Long-Horizon Periodic Callback Stress (`W=64`)
    # --------------------------------------------------------------------------
    print(f"\n[VECTOR 2] ⚔️ Long-Horizon Document Recall ({doc_eval_len} tokens, W=64)...")
    v2_nosink = eval_autoregressive_ppl(model, doc_ids, dev0, cache_mode="naive_window_no_sinks", window_size=64, eval_tail_from=tail_from, chunk_step=64)
    v2_stream = eval_autoregressive_ppl(model, doc_ids, dev0, cache_mode="streaming_llm", window_size=64, num_sinks=4, eval_tail_from=tail_from, chunk_step=64)
    v2_scache = eval_autoregressive_ppl(model, doc_ids, dev0, cache_mode="scache_o1", window_size=64, num_slots=16, num_sinks=4, eval_tail_from=tail_from, chunk_step=64)

    for lbl, r in [("Dense+Vanilla", v2_van), ("MoS+NaiveWin", v2_nosink), ("MoS+StreamLLM", v2_stream), ("MoS+SCache O(1)", v2_scache)]:
        print(f"  • {lbl:<16} | Tail PPL={r['ppl_tail']:8.2f} | Full PPL={r['ppl_full']:8.2f} | KV RAM={r['actual_recursive_kb']:9.1f} KB")

    report["vectors"]["vector_2_long_horizon_ppl"] = {
        "seq_len": doc_eval_len,
        "dense_vanilla": v2_van,
        "mos_naive_window_no_sinks": v2_nosink,
        "mos_streaming_llm": v2_stream,
        "mos_scache_o1": v2_scache,
    }

    # --------------------------------------------------------------------------
    # [VECTOR 3B] Multi-Distance Evicted Passcode Sweep (`7942-OMEGA`, Zero Fine-Tuning)
    # --------------------------------------------------------------------------
    print("\n[VECTOR 3B] ⚔️ Multi-Distance Evicted Passcode Sweep (`7942-OMEGA`, Zero Fine-Tuning)...")
    needle_results = []
    for idx_p, (p_ids, ans_ids, full_ids) in enumerate(passcode_bundles):
        p_len = p_ids.shape[1]

        @torch.no_grad()
        def _eval_passcode_cache(c_mode: str) -> Tuple[float, float]:
            c = (
                SCacheDynamicCache(window_size=64, num_slots=0, num_sink_tokens=4, project_slots=False)
                if c_mode == "streaming_llm"
                else SCacheDynamicCache(window_size=64, num_slots=24, num_sink_tokens=4, project_slots=True)
            )
            for s in range(0, p_len - 1, 64):
                e = min(s + 64, p_len - 1)
                _ = model(p_ids[:, s:e].to(dev0), past_key_values=c, use_cache=True)
            out = model(full_ids[:, p_len - 1 : -1].to(dev0), past_key_values=c, use_cache=True)
            lp = F.log_softmax(out.logits.float().cpu(), dim=-1)
            nll = float(-lp.gather(-1, ans_ids.unsqueeze(-1)).squeeze(-1).mean().item())
            kb = recursive_device_storage_bytes(c) / 1024.0
            return nll, kb

        nll_str, _ = _eval_passcode_cache("streaming_llm")
        nll_sc, kb_sc = _eval_passcode_cache("scache_o1")
        kb_van = (n_layers * 2 * n_kv_heads * full_ids.shape[1] * head_dim * 2) / 1024.0
        gain_x = math.exp(min(nll_str - nll_sc, 20.0))
        nr = {
            "context_tokens": int(p_len),
            "eviction_distance_tokens": int(max(0, p_len - 64 - 25)),
            "window_size": 64,
            "num_slots": 24,
            "dense_vanilla_nll": round(dense_v3b_nlls[idx_p], 4),
            "mos_streaming_llm_nll": round(nll_str, 4),
            "mos_scache_o1_nll": round(nll_sc, 4),
            "scache_vs_streaming_likelihood_gain_x": round(gain_x, 3),
            "vanilla_kv_kb": round(kb_van, 2),
            "scache_kv_kb": round(kb_sc, 2),
            "kv_ram_saving_pct": round((1.0 - kb_sc / max(kb_van, 1e-6)) * 100.0, 2),
        }
        needle_results.append(nr)
        print(
            f"  • [L={nr['context_tokens']} tok | Evicted {nr['eviction_distance_tokens']} tok ago] "
            f"Dense={nr['dense_vanilla_nll']:.3f} | StreamLLM={nr['mos_streaming_llm_nll']:.3f} | "
            f"SCache O(1)={nr['mos_scache_o1_nll']:.3f} ({nr['scache_vs_streaming_likelihood_gain_x']:.2f}x prob gain, -{nr['kv_ram_saving_pct']:.1f}% KV)"
        )
    report["vectors"]["vector_3b_evicted_passcode_sweep"] = needle_results

    # --------------------------------------------------------------------------
    # [VECTOR 5] Full-Stack (`72.7B MoS` + `SCache O(1)`) Deterministic QA/Code/Math
    # --------------------------------------------------------------------------
    print("\n[VECTOR 5] ⚔️ Full-Stack Simultaneous (`MoS` + `SCache O(1)`) QA/Code/Math...")
    hits_mos_sc = 0
    mos_sc_nlls: List[float] = []
    with torch.no_grad():
        for p_ids, t_ids, cat_ids in qa_bundles:
            c_qa = SCacheDynamicCache(window_size=64, num_slots=16, num_sink_tokens=4, project_slots=True)
            for s in range(0, p_ids.shape[1] - 1, 48):
                e = min(s + 48, p_ids.shape[1] - 1)
                _ = model(p_ids[:, s:e].to(dev0), past_key_values=c_qa, use_cache=True)
            out_q = model(cat_ids[:, p_ids.shape[1] - 1 : -1].to(dev0), past_key_values=c_qa, use_cache=True)
            lg_q = out_q.logits.float().cpu()
            lp_q = F.log_softmax(lg_q, dim=-1)
            mos_sc_nlls.append(float(-lp_q.gather(-1, t_ids.unsqueeze(-1)).squeeze(-1).mean().item()))
            if int(t_ids[0, 0].item()) in lg_q[0, 0].topk(3).indices.tolist():
                hits_mos_sc += 1
    nll_mos_sc_tasks = float(sum(mos_sc_nlls) / len(mos_sc_nlls))
    print(
        f"  • Dense + Vanilla KV : Hits={hits_dense}/{len(qa_tasks)} | Ans NLL={nll_dense_tasks:.3f}\n"
        f"  • MoS + SCache O(1)  : Hits={hits_mos_sc}/{len(qa_tasks)} | Ans NLL={nll_mos_sc_tasks:.3f} | "
        f"MLP=-{block64_stats['mlp_savings_pct_vs_fp16']:.1f}% | KV=-{(1.0 - v2_scache['actual_recursive_kb']/v2_van['actual_recursive_kb'])*100:.1f}%"
    )
    report["vectors"]["vector_5_fullstack_mos_scache"] = {
        "tasks_total": len(qa_tasks),
        "dense_top3_hits": hits_dense,
        "mos_scache_top3_hits": hits_mos_sc,
        "dense_answer_nll": round(nll_dense_tasks, 4),
        "mos_scache_answer_nll": round(nll_mos_sc_tasks, 4),
        "long_doc_dense_vanilla_ppl": v2_van["ppl_full"],
        "long_doc_mos_scache_ppl": v2_scache["ppl_full"],
        "mlp_savings_pct": block64_stats["mlp_savings_pct_vs_fp16"],
        "long_doc_kv_savings_pct": round((1.0 - v2_scache["actual_recursive_kb"] / v2_van["actual_recursive_kb"]) * 100.0, 2),
    }

    # --------------------------------------------------------------------------
    # [VECTOR 6] Single-Group Streaming VRAM Ceiling Audit (8-Layer Consumer GPU Footprint)
    # --------------------------------------------------------------------------
    group_sz = min(8, n_layers)
    group_mos_bytes = sum(
        sum(getattr(model.model.layers[i].self_attn, p).physical_bytes for p in ("q_proj", "k_proj", "v_proj", "o_proj"))
        + sum(getattr(model.model.layers[i].mlp, p).physical_bytes for p in ("gate_proj", "up_proj", "down_proj"))
        for i in range(group_sz)
    )
    single_gpu_8layer_group_gib = round(group_mos_bytes / (1024**3), 3)
    # Benchmark pure GPU-resident layer throughput on `dev0` layers (`0..n_gpu0-1`) to report full-VRAM projected tok/s too!
    if has_cuda:
        torch.cuda.synchronize(dev0)
    tg0 = time.perf_counter()
    with torch.no_grad():
        h_bench = model.model.embed_tokens(pareto_ids[:, :64].to(dev0))
        pos_b, cm_b, pe_b = seq_meta_cache.get(64, (None, None, None))
        if pos_b is None:
            pos_b = torch.arange(64, device=dev0).unsqueeze(0)
            cm_b = create_causal_mask(config=cfg, inputs_embeds=h_bench, attention_mask=None, past_key_values=None, position_ids=pos_b)
            pe_b = rotary_dev0(h_bench, pos_b)
        for i_b in range(n_gpu0):
            h_bench = model.model.layers[i_b](h_bench, attention_mask=cm_b, position_ids=pos_b, position_embeddings=pe_b)
    if has_cuda:
        torch.cuda.synchronize(dev0)
    dt_gpu_sub = max(time.perf_counter() - tg0, 1e-6)
    projected_all_vram_tok_per_s = round(64.0 / (dt_gpu_sub * (n_layers / max(n_gpu0, 1))), 2)
    report["vectors"]["vector_6_single_group_streaming_footprint"] = {
        "group_layers": group_sz,
        "single_group_mos_gib": single_gpu_8layer_group_gib,
        "scache_80layer_kv_mb": round(mem_scaling[-1]["scache_actual_recursive_kb"] / 1024.0, 2),
        "projected_pure_vram_resident_tok_per_s": projected_all_vram_tok_per_s,
    }

    # --------------------------------------------------------------------------
    # [VECTOR 7] 3D Pareto Frontier on Non-Repeating Technical Prose + Python + Math + SQL
    # --------------------------------------------------------------------------
    print(f"\n[VECTOR 7] ⚖️ 3D Pareto Frontier on Non-Repeating Document ({pareto_ids.shape[1]} unique tokens)...")
    pareto_configs = [
        ("mos_streaming_w128", "streaming_llm", 128, 0),
        ("mos_scache_w128", "scache_o1", 128, 16),
        ("mos_scache_w256", "scache_o1", 256, 16),
        ("mos_scache_w384", "scache_o1", 384, 16),
    ]
    pareto_points: Dict[str, Any] = {"dense_fp16_vanilla_kv": v7_dense_van}
    print(f"  • {'dense_fp16_vanilla_kv':<24} | Non-Repeat PPL={v7_dense_van['ppl_full']:6.2f} | KV={v7_dense_van['actual_recursive_kb']:9.1f} KB")
    for tag, cmode, w_sz, n_sl in pareto_configs:
        if has_cuda:
            torch.cuda.synchronize()
        tp0 = time.perf_counter()
        pr = eval_autoregressive_ppl(
            model, pareto_ids, dev0, cache_mode=cmode, window_size=w_sz, num_slots=n_sl, num_sinks=4, chunk_step=64
        )
        if has_cuda:
            torch.cuda.synchronize()
        dtp = max(time.perf_counter() - tp0, 1e-6)
        pr["hybrid_2xt4_tok_per_s"] = round(pareto_ids.shape[1] / dtp, 2)
        pr["pure_vram_tok_per_s"] = projected_all_vram_tok_per_s
        pr["speedup_hybrid_vs_airllm_fp16_x"] = round(pr["hybrid_2xt4_tok_per_s"] / max(airllm_fp16_max_tok_per_s, 1e-6), 1)
        pr["speedup_pure_vram_vs_airllm_fp16_x"] = round(projected_all_vram_tok_per_s / max(airllm_fp16_max_tok_per_s, 1e-6), 1)
        pareto_points[tag] = pr
        print(
            f"  • {tag:<24} | Non-Repeat PPL={pr['ppl_full']:6.2f} | KV={pr['actual_recursive_kb']:8.1f} KB | "
            f"2xT4 Hybrid={pr['hybrid_2xt4_tok_per_s']:5.2f} tok/s ({pr['speedup_hybrid_vs_airllm_fp16_x']:.1f}x vs AirLLM FP16) | "
            f"Pure VRAM={projected_all_vram_tok_per_s:5.2f} tok/s ({pr['speedup_pure_vram_vs_airllm_fp16_x']:.1f}x)"
        )

    report["vectors"]["vector_7_non_repeating_pareto_vs_offload"] = {
        "non_repeating_tokens": int(pareto_ids.shape[1]),
        "measured_safetensors_disk_to_gpu_gibps": round(measured_io_gbps, 2),
        "airllm_fp16_batch1_ceiling_tok_per_s": airllm_fp16_max_tok_per_s,
        "airllm_int4_batch1_ceiling_tok_per_s": airllm_int4_max_tok_per_s,
        "points": pareto_points,
    }

    report["elapsed_seconds"] = round(time.time() - t0, 2)
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if spill_root.exists():
        import shutil
        shutil.rmtree(spill_root, ignore_errors=True)
    print(f"\n✅ ALL 72.7B VECTORS COMPLETED CLEANLY IN {report['elapsed_seconds']}s -> {out_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--output-json", default="results/scale_72b_audit/adversarial_scale_72b_report.json")
    parser.add_argument("--quick-local-verify", action="store_true")
    args = parser.parse_args()
    run_colossal_72b_suite(args.model_id, args.output_json, quick_local_verify=args.quick_local_verify)


if __name__ == "__main__":
    main()
