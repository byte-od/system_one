"""HuggingFace SCache Integration: Drop-in Cache O(1) para Modelos Reais (Llama, SmolLM, Qwen).

Substitui o DynamicCache linear O(T) do HuggingFace por uma arquitetura híbrida O(1):
  1. Attention Sinks (primeiros `num_sink_tokens` fixos para ancoragem de fase RoPE/Softmax).
  2. Janela Deslizante Local (últimos `window_size - num_sink_tokens` tokens recentes).
  3. Banco Associativo de Cicatrizes Por-Cabeça (`scar_keys`, `scar_vals`, `W_delta`) atualizado
     via Roteamento Competitivo por Salience/Novidade + Regra Delta de Widrow-Hoff (Anti-Ghosting),
     com decaimento multi-escala heterogêneo e footprint físico 100% constante O(1).
  4. Compatibilidade nativa com `transformers` v5 (`cumulative_length`, `get_seq_length`,
     `get_mask_sizes`) suportando tanto decodificação token-a-token quanto prefill em blocos.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, DynamicLayer

from scar_cache.multi_slot import MultiSlotSCache


class SCacheDynamicLayer(DynamicLayer):
    """Camada de Cache do HuggingFace com tamanho físico estritamente constante O(1)."""

    is_sliding = True

    def __init__(
        self,
        window_size: int = 64,
        num_slots: int = 8,
        num_sink_tokens: int = 4,
        project_slots: bool = False,
        merge_cosine_threshold: float = 0.72,
        delta_blend: float = 0.15,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.num_slots = int(num_slots)
        self.num_sink_tokens = min(int(num_sink_tokens), self.window_size // 2) if num_sink_tokens > 0 else 0
        self.project_slots = bool(project_slots)
        self.merge_cosine_threshold = float(merge_cosine_threshold)
        self.delta_blend = float(delta_blend)

        self.scache: Optional[MultiSlotSCache] = None
        self.scar_keys: Optional[torch.Tensor] = None
        self.scar_vals: Optional[torch.Tensor] = None
        self.scar_salience: Optional[torch.Tensor] = None
        self.scar_decay: Optional[torch.Tensor] = None
        self.active_slots: int = 0
        self.total_tokens_absorbed: int = 0
        self.cumulative_length: int = 0

    def _init_scar_bank(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        bsz, n_heads, _, head_dim = key_states.shape
        dev = key_states.device
        dt = key_states.dtype

        if self.num_slots > 0:
            # Uma matriz associativa Widrow-Hoff exata `W_scar[h]` [head_dim, head_dim] por cabeça KV!
            self.scache = MultiSlotSCache(
                slot_dim=head_dim,
                num_slots=n_heads,
                use_delta_rule=True,
                multi_scale_decay=False,
            ).to(dev)
            self.scar_keys = torch.zeros((bsz, n_heads, self.num_slots, head_dim), dtype=dt, device=dev)
            self.scar_vals = torch.zeros((bsz, n_heads, self.num_slots, head_dim), dtype=value_states.dtype, device=dev)
            self.scar_salience = torch.zeros((bsz, n_heads, self.num_slots), dtype=torch.float32, device=dev)
            if self.num_slots == 1:
                decays = torch.tensor([0.999], dtype=torch.float32, device=dev)
            else:
                decays = torch.exp(
                    torch.linspace(math.log(0.93), math.log(0.9998), self.num_slots, device=dev, dtype=torch.float32)
                )
            self.scar_decay = decays

    def _predict_returned_kv_length(self, query_length: int) -> int:
        cached_len = self.keys.shape[-2] if (self.is_initialized and self.keys is not None) else 0
        total_raw = cached_len + query_length
        if total_raw <= self.window_size:
            return total_raw

        slide_len = self.window_size - self.num_sink_tokens
        num_excess = total_raw - self.window_size
        n_cand = min(num_excess, max(self.num_slots * 4, 32))
        pred_active = min(self.num_slots, self.active_slots + n_cand) if (self.project_slots and self.num_slots > 0) else 0

        # Se query_length > slide_len (ex: prefill longo de uma vez), mantemos os `query_length`
        # tokens atuais visíveis na chamada corrente para causalidade intra-chunk exata,
        # mas salvamos apenas `window_size` em `self.keys`!
        tail_vis = max(slide_len, query_length) if self.project_slots else slide_len
        sink_vis = (
            min(self.num_sink_tokens, max(0, total_raw - tail_vis))
            if (self.project_slots and query_length > slide_len)
            else self.num_sink_tokens
        )
        return sink_vis + pred_active + tail_vis

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_length(self) -> int:
        return self.window_size + (self.num_slots if self.project_slots else 0)

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        kv_length = self._predict_returned_kv_length(query_length)
        prefix_len = kv_length - query_length
        kv_offset = self.cumulative_length - prefix_len
        return kv_length, kv_offset

    @torch.no_grad()
    def _ingest_evicted_tokens(self, excess_k: torch.Tensor, excess_v: torch.Tensor) -> None:
        """Comprime tokens desalojados no banco O(1) via Roteamento Competitivo + Regra Delta."""
        num_excess = excess_k.shape[-2]
        if num_excess == 0 or self.num_slots <= 0:
            return

        if self.scar_keys is None or self.scar_vals is None or self.scar_salience is None or self.scache is None:
            return

        bsz, n_heads, _, head_dim = excess_k.shape
        kf = excess_k.float()
        vf = excess_v.float()

        k_norm = torch.norm(kf, dim=-1, keepdim=True).clamp(min=1e-6)
        v_norm = torch.norm(vf, dim=-1)
        k_unit = kf / k_norm
        sal_all = v_norm * torch.sqrt(k_norm.squeeze(-1))

        # Atualização associativa Widrow-Hoff Delta-Rule por cabeça diretamente sobre `self.scache.W_scar` [H, D, D]
        W_cur = self.scache.W_scar
        pred_v = torch.einsum("bhtd,hvd->bhtv", k_unit, W_cur)
        err_v = vf - pred_v
        dW = torch.einsum("bhtv,bhtd->hvd", err_v, k_unit) / float(max(bsz * num_excess, 1))
        W_cur.mul_(0.985).add_(dW, alpha=0.35)
        if hasattr(self.scache, "scars_count"):
            self.scache.scars_count.add_(num_excess)

        max_candidates = min(num_excess, max(self.num_slots * 4, 32))
        if num_excess > max_candidates:
            top_idx = torch.topk(sal_all, k=max_candidates, dim=-1).indices
            top_idx, _ = torch.sort(top_idx, dim=-1)
            idx_exp = top_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            kf_sel = torch.gather(kf, 2, idx_exp)
            vf_sel = torch.gather(vf, 2, idx_exp)
            ku_sel = torch.gather(k_unit, 2, idx_exp)
            sal_sel = torch.gather(sal_all, 2, top_idx)
        else:
            kf_sel, vf_sel, ku_sel, sal_sel = kf, vf, k_unit, sal_all

        n_cand = kf_sel.shape[2]
        sk_f = self.scar_keys.float()
        sv_f = self.scar_vals.float()
        dec = self.scar_decay.view(1, 1, self.num_slots)

        for t in range(n_cand):
            kt = kf_sel[:, :, t, :]
            vt = vf_sel[:, :, t, :]
            kut = ku_sel[:, :, t, :]
            st = sal_sel[:, :, t]

            if self.active_slots < self.num_slots:
                s_idx = self.active_slots
                sk_f[:, :, s_idx, :] = kt
                sv_f[:, :, s_idx, :] = vt
                self.scar_salience[:, :, s_idx] = st
                self.active_slots += 1
                continue

            self.scar_salience.mul_(dec)

            sk_unit = sk_f / torch.norm(sk_f, dim=-1, keepdim=True).clamp(min=1e-6)
            cos_sim = torch.einsum("bhd,bhsd->bhs", kut, sk_unit)
            max_cos, best_s = cos_sim.max(dim=-1)

            is_merge = max_cos >= self.merge_cosine_threshold
            min_sal, min_s = self.scar_salience.min(dim=-1)
            is_replace = (~is_merge) & (st >= 0.85 * min_sal)

            target_s = torch.where(is_merge, best_s, min_s)
            s_oh = F.one_hot(target_s, num_classes=self.num_slots).float()

            alpha_k = torch.where(is_merge, torch.full_like(st, 0.35), torch.where(is_replace, torch.ones_like(st), torch.zeros_like(st)))
            alpha_v = torch.where(is_merge, torch.full_like(st, 0.78), torch.where(is_replace, torch.ones_like(st), torch.zeros_like(st)))

            wk = (alpha_k.unsqueeze(-1) * s_oh).unsqueeze(-1)
            wv = (alpha_v.unsqueeze(-1) * s_oh).unsqueeze(-1)

            sk_f = (1.0 - wk) * sk_f + wk * kt.unsqueeze(2)
            sv_f = (1.0 - wv) * sv_f + wv * vt.unsqueeze(2)

            new_sal = torch.where(
                is_merge,
                torch.maximum(torch.gather(self.scar_salience, -1, target_s.unsqueeze(-1)).squeeze(-1), st) * 1.04,
                st,
            )
            upd_mask = (is_merge | is_replace).unsqueeze(-1).float() * s_oh
            self.scar_salience = (1.0 - upd_mask) * self.scar_salience + upd_mask * new_sal.unsqueeze(-1)

            if (t + 1) % 8 == 0 and self.num_slots >= 2:
                top_s = self.scar_salience.argmax(dim=-1)
                last_s = self.num_slots - 1
                if (top_s != last_s).any():
                    for b_i in range(bsz):
                        for h_i in range(n_heads):
                            ts = int(top_s[b_i, h_i].item())
                            if ts != last_s:
                                sk_f[b_i, h_i, [ts, last_s]] = sk_f[b_i, h_i, [last_s, ts]]
                                sv_f[b_i, h_i, [ts, last_s]] = sv_f[b_i, h_i, [last_s, ts]]
                                self.scar_salience[b_i, h_i, [ts, last_s]] = self.scar_salience[b_i, h_i, [last_s, ts]]

        self.scar_keys.copy_(sk_f.to(self.scar_keys.dtype))
        self.scar_vals.copy_(sv_f.to(self.scar_vals.dtype))

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self._init_scar_bank(key_states, value_states)

        q_len = key_states.shape[-2]
        self.cumulative_length += q_len

        full_k = torch.cat([self.keys, key_states], dim=-2)
        full_v = torch.cat([self.values, value_states], dim=-2)
        total_raw = full_k.shape[-2]

        if total_raw <= self.window_size:
            self.keys = full_k
            self.values = full_v
            return self.keys, self.values

        slide_len = self.window_size - self.num_sink_tokens
        excess_k = full_k[..., self.num_sink_tokens : -slide_len, :]
        excess_v = full_v[..., self.num_sink_tokens : -slide_len, :]
        num_excess = excess_k.shape[-2]

        self._ingest_evicted_tokens(excess_k, excess_v)
        self.total_tokens_absorbed += num_excess

        # Armazenamento físico estritamente congelado em `window_size` O(1)!
        if self.num_sink_tokens > 0:
            sink_k = full_k[..., : self.num_sink_tokens, :]
            sink_v = full_v[..., : self.num_sink_tokens, :]
            slide_k = full_k[..., -slide_len:, :]
            slide_v = full_v[..., -slide_len:, :]
            self.keys = torch.cat([sink_k, slide_k], dim=-2)
            self.values = torch.cat([sink_v, slide_v], dim=-2)
        else:
            self.keys = full_k[..., -self.window_size :, :].clone()
            self.values = full_v[..., -self.window_size :, :].clone()

        tail_vis = max(slide_len, q_len) if self.project_slots else slide_len
        tail_k = full_k[..., -tail_vis:, :]
        tail_v = full_v[..., -tail_vis:, :]
        sink_vis = (
            min(self.num_sink_tokens, max(0, total_raw - tail_vis))
            if (self.project_slots and q_len > slide_len)
            else self.num_sink_tokens
        )
        vis_sink_k = full_k[..., :sink_vis, :]
        vis_sink_v = full_v[..., :sink_vis, :]

        if (
            self.project_slots
            and self.num_slots > 0
            and self.active_slots > 0
            and self.scar_keys is not None
            and self.scar_vals is not None
            and self.scache is not None
        ):
            act = self.active_slots
            k_slots = self.scar_keys[:, :, :act, :]
            v_proto = self.scar_vals[:, :, :act, :].float()
            k_unit = k_slots.float() / torch.norm(k_slots.float(), dim=-1, keepdim=True).clamp(min=1e-6)
            v_delta = torch.einsum("bhsd,hvd->bhsv", k_unit, self.scache.W_scar)
            v_slots = ((1.0 - self.delta_blend) * v_proto + self.delta_blend * v_delta).to(self.values.dtype)

            if sink_vis > 0:
                return (
                    torch.cat([vis_sink_k, k_slots, tail_k], dim=-2),
                    torch.cat([vis_sink_v, v_slots, tail_v], dim=-2),
                )
            return torch.cat([k_slots, tail_k], dim=-2), torch.cat([v_slots, tail_v], dim=-2)

        if sink_vis > 0:
            return torch.cat([vis_sink_k, tail_k], dim=-2), torch.cat([vis_sink_v, tail_v], dim=-2)
        return tail_k, tail_v

    @property
    def physical_bytes(self) -> int:
        if not self.is_initialized or self.keys is None:
            return 0
        return int((self.keys.numel() + self.values.numel()) * self.keys.element_size())

    @property
    def scache_bytes(self) -> int:
        total = 0
        if self.scache is not None:
            for p in self.scache.parameters():
                total += int(p.numel() * p.element_size())
            for b in self.scache.buffers():
                total += int(b.numel() * b.element_size())
        for buf in (self.scar_keys, self.scar_vals, self.scar_salience, self.scar_decay):
            if buf is not None:
                total += int(buf.numel() * buf.element_size())
        return total


class SCacheDynamicCache(DynamicCache):
    """Drop-in Replacement O(1) para DynamicCache do HuggingFace compatível com qualquer LLM."""

    def __init__(
        self,
        window_size: int = 64,
        num_slots: int = 8,
        num_sink_tokens: int = 4,
        project_slots: bool = False,
        merge_cosine_threshold: float = 0.72,
        delta_blend: float = 0.15,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.num_slots = int(num_slots)
        self.num_sink_tokens = int(num_sink_tokens)
        self.project_slots = bool(project_slots)
        self.merge_cosine_threshold = float(merge_cosine_threshold)
        self.delta_blend = float(delta_blend)
        self.layer_class_to_replicate = lambda: SCacheDynamicLayer(
            window_size=self.window_size,
            num_slots=self.num_slots,
            num_sink_tokens=self.num_sink_tokens,
            project_slots=self.project_slots,
            merge_cosine_threshold=self.merge_cosine_threshold,
            delta_blend=self.delta_blend,
        )

    @property
    def total_physical_bytes(self) -> int:
        return sum(
            layer.physical_bytes for layer in self.layers if isinstance(layer, SCacheDynamicLayer)
        )

    @property
    def total_scache_bytes(self) -> int:
        return sum(
            layer.scache_bytes for layer in self.layers if isinstance(layer, SCacheDynamicLayer)
        )

    @property
    def total_memory_bytes(self) -> int:
        return self.total_physical_bytes + self.total_scache_bytes

    def get_max_physical_length(self) -> int:
        lengths = [
            layer.keys.shape[-2]
            for layer in self.layers
            if isinstance(layer, SCacheDynamicLayer) and layer.keys is not None
        ]
        return max(lengths) if lengths else 0
