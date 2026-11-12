"""HuggingFace SCache Integration: Drop-in Cache O(1) para Modelos Reais (Llama, SmolLM, Qwen).

Substitui o DynamicCache linear O(T) do HuggingFace por uma arquitetura híbrida O(1):
  1. Attention Sinks (primeiros `num_sink_tokens` fixos para ancoragem de fase RoPE/Softmax).
  2. Janela Deslizante Local (últimos `window_size - num_sink_tokens` tokens recentes).
  3. Banco Associativo de Cicatrizes Por-Cabeça (`scar_keys`, `scar_vals`, `W_scar`) atualizado
     via Micro-Páginas Contíguas Sincronizadas + Roteamento Competitivo por Surpresa Widrow-Hoff
     (Anti-Ghosting), com decaimento lento anti-flood e footprint físico 100% constante O(1).
  4. Compatibilidade nativa com `transformers` v5 (`cumulative_length`, `get_seq_length`,
     `get_mask_sizes`) suportando decodificação token-a-token, prefill em blocos e prefill único.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, DynamicLayer

from scar_cache.multi_slot import MultiSlotSCache


def _choose_page_size(num_slots: int) -> int:
    """Seleciona tamanho de micro-página contígua (8 ou 4 tokens) preservando n-grams completos."""
    for p in (8, 4):
        if num_slots >= 2 * p and num_slots % p == 0:
            return p
    return 1


def _extract_top_pages(sync_1d: torch.Tensor, page_sz: int, max_pages: int) -> List[List[Tuple[int, float]]]:
    """Extrai exatamente `min(max_pages, T // page_sz)` janelas disjuntas ordenadas cronologicamente."""
    bsz, T = sync_1d.shape
    k_pages = min(max_pages, T // page_sz)
    if k_pages <= 0:
        return [[] for _ in range(bsz)]

    windows_all = sync_1d.unfold(1, page_sz, 1)  # [B, M, page_sz]
    win_scores_all = windows_all.mean(dim=-1).cpu()  # [B, M]
    M = win_scores_all.shape[1]

    pages_per_b: List[List[Tuple[int, float]]] = []
    for b in range(bsz):
        w = win_scores_all[b]  # [M]
        used = torch.zeros(T, dtype=torch.bool)
        picked: List[Tuple[int, float]] = []
        for _ in range(k_pages):
            overlap = F.max_pool1d(used.float().view(1, 1, T), kernel_size=page_sz, stride=1).view(-1) > 0
            if bool(overlap.all()):
                break
            best_s = int(w.masked_fill(overlap, -1e9).argmax().item())
            if bool(used[best_s : best_s + page_sz].any()):
                break
            used[best_s : best_s + page_sz] = True
            picked.append((best_s, float(w[best_s].item())))

        if len(picked) == k_pages:
            picked.sort(key=lambda x: x[0])
            pages_per_b.append(picked)
            continue

        # Fallback exato via Programação Dinâmica 1D quando `T` é apertado (garante `len == k_pages` sempre!)
        backptrs: List[torch.Tensor] = []
        dp_prev = w.clone()
        for r in range(1, k_pages):
            min_s = r * page_sz
            pref_vals, pref_idx = torch.cummax(dp_prev[: M - page_sz], dim=0)
            dp_cur = torch.full((M,), -1e18, dtype=torch.float32)
            bp_cur = torch.zeros((M,), dtype=torch.long)
            dp_cur[min_s:] = w[min_s:] + pref_vals[min_s - page_sz : M - page_sz]
            bp_cur[min_s:] = pref_idx[min_s - page_sz : M - page_sz]
            backptrs.append(bp_cur)
            dp_prev = dp_cur

        last_min_s = (k_pages - 1) * page_sz
        rel_best = int(dp_prev[last_min_s:].argmax().item())
        cur_s = last_min_s + rel_best
        starts = [cur_s]
        for r in range(k_pages - 2, -1, -1):
            cur_s = int(backptrs[r][cur_s].item())
            starts.append(cur_s)
        starts.reverse()
        pages_per_b.append([(s, float(w[s].item())) for s in starts])

    return pages_per_b


class SCacheDynamicLayer(DynamicLayer):
    """Camada de Cache do HuggingFace com tamanho físico estritamente constante O(1)."""

    is_sliding = True

    def __init__(
        self,
        window_size: int = 64,
        num_slots: int = 8,
        num_sink_tokens: int = 4,
        project_slots: bool = False,
        merge_cosine_threshold: float = 0.995,
        delta_blend: float = 0.05,
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
        tail_vis = max(slide_len, query_length) if self.project_slots else slide_len
        tail_start = total_raw - tail_vis
        sink_vis = min(self.num_sink_tokens, tail_start) if self.project_slots else self.num_sink_tokens
        split_idx = max(self.num_sink_tokens, tail_start)
        pre_excess = split_idx - self.num_sink_tokens
        page_sz = _choose_page_size(self.num_slots)

        if page_sz > 1:
            if pre_excess >= page_sz:
                add_slots = (pre_excess // page_sz) * page_sz
            elif self.active_slots == self.num_slots:
                add_slots = 0
            else:
                add_slots = pre_excess
        else:
            add_slots = pre_excess

        pred_active = (
            min(self.num_slots, self.active_slots + add_slots)
            if (
                self.project_slots
                and self.num_slots > 0
                and tail_start > self.num_sink_tokens
                and (self.active_slots + add_slots) > 0
            )
            else 0
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

    def _compute_token_salience(
        self,
        excess_k: torch.Tensor,
        excess_v: torch.Tensor,
        obs_k: Optional[torch.Tensor] = None,
        obs_v: Optional[torch.Tensor] = None,
        sink_v: Optional[torch.Tensor] = None,
        prev_sync_sal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, n_heads, T, _ = excess_k.shape
        kf = excess_k.float()
        vf = excess_v.float()
        k_norm = torch.norm(kf, dim=-1, keepdim=True).clamp(min=1e-6)
        v_norm = torch.norm(vf, dim=-1).clamp(min=1e-6)
        k_unit = kf / k_norm
        v_unit = vf / v_norm.unsqueeze(-1)

        assert self.scache is not None
        W_cur = self.scache.W_scar
        pred_v = torch.einsum("bhtd,hvd->bhtv", k_unit, W_cur)
        err_norm = torch.norm(vf - pred_v, dim=-1)
        raw_intr = 0.50 * v_norm + 0.50 * err_norm
        med_intr = raw_intr.median(dim=-1, keepdim=True).values.clamp(min=1e-5)
        intr_bounded = torch.log1p((raw_intr / med_intr).clamp(max=6.0))

        if sink_v is not None and sink_v.shape[-2] > 0:
            sv = sink_v.float()
            sv_u = sv / torch.norm(sv, dim=-1, keepdim=True).clamp(min=1e-6)
            sv_sim = torch.einsum("bhtd,bhsd->bhts", v_unit, sv_u).abs().amax(dim=-1)
            intr_bounded = intr_bounded * (1.0 - 0.70 * (sv_sim > 0.75).float() - 0.35 * sv_sim).clamp(min=0.0)

        kn_s = k_norm.squeeze(-1)
        med_kn = kn_s.median(dim=-1, keepdim=True).values.clamp(min=1e-5)
        delim_mask = (kn_s > 2.2 * med_kn).float()
        intr_bounded = intr_bounded * (1.0 - 0.65 * delim_mask)

        intr_1d = intr_bounded.mean(dim=1)  # [B, T]

        if T >= 6 and obs_k is not None and obs_v is not None and obs_k.shape[-2] > 0:
            ok = obs_k[..., -min(obs_k.shape[-2], 44) :, :].float()
            ov = obs_v[..., -min(obs_v.shape[-2], 44) :, :].float()
            ok_u = ok / torch.norm(ok, dim=-1, keepdim=True).clamp(min=1e-6)
            ov_u = ov / torch.norm(ov, dim=-1, keepdim=True).clamp(min=1e-6)
            kk_max = torch.einsum("bhtd,bhsd->bhts", k_unit, ok_u).amax(dim=-1)
            vv_max = torch.einsum("bhtd,bhsd->bhts", v_unit, ov_u).amax(dim=-1)
            k_bg = min(21, T if T % 2 == 1 else T - 1)
            if k_bg >= 3:
                pad_bg = k_bg // 2
                kk3 = kk_max.reshape(bsz * n_heads, 1, T)
                kk_bg = F.avg_pool1d(F.pad(kk3, (pad_bg, pad_bg), mode="replicate"), kernel_size=k_bg, stride=1).view(
                    bsz, n_heads, T
                )
                kk_det = (kk_max - kk_bg).clamp(min=0.0)
                vv3 = vv_max.reshape(bsz * n_heads, 1, T)
                vv_bg = F.avg_pool1d(F.pad(vv3, (pad_bg, pad_bg), mode="replicate"), kernel_size=k_bg, stride=1).view(
                    bsz, n_heads, T
                )
                vv_det = (vv_max - vv_bg).clamp(min=0.0)
            else:
                kk_det = kk_max.clamp(min=0.0)
                vv_det = vv_max.clamp(min=0.0)

            raw_det = (0.30 * kk_det + 0.70 * vv_det) * (1.0 - 0.80 * delim_mask)
            anc_1d = raw_det.mean(dim=1)
            anc_3d = anc_1d.reshape(bsz, 1, T)
            anc_span = F.max_pool1d(F.pad(anc_3d, (11, 0), mode="replicate"), kernel_size=12, stride=1).squeeze(1)
            cur_1d = intr_1d + 2.4 * anc_span
        else:
            cur_1d = intr_1d

        if prev_sync_sal is not None and prev_sync_sal.shape == cur_1d.shape:
            sync_1d = 0.75 * prev_sync_sal.to(cur_1d.device) + 0.25 * cur_1d
        else:
            sync_1d = cur_1d

        return k_unit, sync_1d

    @torch.no_grad()
    def _ingest_evicted_tokens(
        self,
        excess_k: torch.Tensor,
        excess_v: torch.Tensor,
        obs_k: Optional[torch.Tensor] = None,
        obs_v: Optional[torch.Tensor] = None,
        sink_v: Optional[torch.Tensor] = None,
        prev_sync_sal: Optional[torch.Tensor] = None,
        use_obs: bool = True,
    ) -> Optional[torch.Tensor]:
        """Comprime tokens desalojados no banco O(1) via Micro-Páginas Contíguas + Regra Delta."""
        num_excess = excess_k.shape[-2]
        if num_excess == 0 or self.num_slots <= 0:
            return None

        if self.scar_keys is None or self.scar_vals is None or self.scar_salience is None or self.scache is None:
            return None

        bsz, n_heads, _, _ = excess_k.shape
        kf = excess_k.float()
        vf = excess_v.float()

        k_unit, sync_1d = self._compute_token_salience(
            excess_k,
            excess_v,
            obs_k=(obs_k if use_obs else None),
            obs_v=(obs_v if use_obs else None),
            sink_v=sink_v,
            prev_sync_sal=prev_sync_sal,
        )

        # Atualização associativa Widrow-Hoff Delta-Rule por cabeça sobre `self.scache.W_scar` [H, D, D]
        W_cur = self.scache.W_scar
        pred_v = torch.einsum("bhtd,hvd->bhtv", k_unit, W_cur)
        err_v = vf - pred_v
        dW = torch.einsum("bhtv,bhtd->hvd", err_v, k_unit) / float(max(bsz * num_excess, 1))
        W_cur.mul_(0.985).add_(dW, alpha=0.35)
        if hasattr(self.scache, "scars_count"):
            self.scache.scars_count.add_(num_excess)

        page_sz = _choose_page_size(self.num_slots)

        if page_sz > 1 and num_excess >= page_sz:
            n_total_pages = self.num_slots // page_sz
            cand_pages = _extract_top_pages(sync_1d, page_sz, n_total_pages)
            sk_f = self.scar_keys.float()
            sv_f = self.scar_vals.float()

            if self.active_slots > 0:
                self.scar_salience.mul_(0.996)

            max_cands = max((len(c) for c in cand_pages), default=0)
            for c_idx in range(max_cands):
                if self.active_slots + page_sz <= self.num_slots:
                    s0 = self.active_slots
                    for b in range(bsz):
                        if c_idx < len(cand_pages[b]):
                            st_idx, p_score = cand_pages[b][c_idx]
                        else:
                            st_idx, p_score = 0, float(sync_1d[b, :page_sz].mean().item())
                        sk_f[b, :, s0 : s0 + page_sz, :] = kf[b, :, st_idx : st_idx + page_sz, :]
                        sv_f[b, :, s0 : s0 + page_sz, :] = vf[b, :, st_idx : st_idx + page_sz, :]
                        self.scar_salience[b, :, s0 : s0 + page_sz] = p_score
                    self.active_slots += page_sz
                elif self.active_slots >= page_sz:
                    act_pages = self.active_slots // page_sz
                    for b in range(bsz):
                        if c_idx >= len(cand_pages[b]):
                            continue
                        st_idx, p_score = cand_pages[b][c_idx]
                        pk = kf[b, :, st_idx : st_idx + page_sz, :]
                        pv = vf[b, :, st_idx : st_idx + page_sz, :]
                        page_sals = [
                            float(self.scar_salience[b, :, p * page_sz : (p + 1) * page_sz].mean().item())
                            for p in range(act_pages)
                        ]
                        min_p = int(min(range(act_pages), key=lambda p: page_sals[p]))
                        if p_score > 1.05 * page_sals[min_p]:
                            keep_ranges = [(p * page_sz, (p + 1) * page_sz) for p in range(act_pages) if p != min_p]
                            kept_k = [sk_f[b, :, r0:r1, :].clone() for r0, r1 in keep_ranges] + [pk]
                            kept_v = [sv_f[b, :, r0:r1, :].clone() for r0, r1 in keep_ranges] + [pv]
                            kept_s = [self.scar_salience[b, :, r0:r1].clone() for r0, r1 in keep_ranges] + [
                                torch.full((n_heads, page_sz), p_score, device=sk_f.device, dtype=torch.float32)
                            ]
                            tot_len = act_pages * page_sz
                            sk_f[b, :, :tot_len, :] = torch.cat(kept_k, dim=1)
                            sv_f[b, :, :tot_len, :] = torch.cat(kept_v, dim=1)
                            self.scar_salience[b, :, :tot_len] = torch.cat(kept_s, dim=1)

            self.scar_keys.copy_(sk_f.to(self.scar_keys.dtype))
            self.scar_vals.copy_(sv_f.to(self.scar_vals.dtype))
            return sync_1d

        # Durante decode token-a-token com páginas atômicas cheias, protege páginas contra fragmentação
        if page_sz > 1 and self.active_slots == self.num_slots and num_excess < page_sz:
            return sync_1d

        sal_all = sync_1d.unsqueeze(1).expand(-1, n_heads, -1)
        sk_f = self.scar_keys.float()
        sv_f = self.scar_vals.float()
        assert self.scar_decay is not None
        raw_dec = self.scar_decay.view(1, 1, self.num_slots)

        for t in range(min(num_excess, self.num_slots)):
            kt = kf[:, :, t, :]
            vt = vf[:, :, t, :]
            kut = k_unit[:, :, t, :]
            st = sal_all[:, :, t]

            if self.active_slots < self.num_slots:
                s_idx = self.active_slots
                sk_f[:, :, s_idx, :] = kt
                sv_f[:, :, s_idx, :] = vt
                self.scar_salience[:, :, s_idx] = st
                self.active_slots += 1
                continue

            self.scar_salience.mul_(raw_dec.pow(0.15))
            sk_unit = sk_f / torch.norm(sk_f, dim=-1, keepdim=True).clamp(min=1e-6)
            cos_sim = torch.einsum("bhd,bhsd->bhs", kut, sk_unit)
            max_cos, best_s = cos_sim.max(dim=-1)

            is_merge = max_cos >= self.merge_cosine_threshold
            min_sal, min_s = self.scar_salience.min(dim=-1)
            is_replace = (~is_merge) & (st > 1.12 * min_sal)

            target_s = torch.where(is_merge, best_s, min_s)
            s_oh = F.one_hot(target_s, num_classes=self.num_slots).float()

            alpha_k = torch.where(
                is_merge, torch.full_like(st, 0.35), torch.where(is_replace, torch.ones_like(st), torch.zeros_like(st))
            )
            alpha_v = torch.where(
                is_merge, torch.full_like(st, 0.78), torch.where(is_replace, torch.ones_like(st), torch.zeros_like(st))
            )

            wk = (alpha_k.unsqueeze(-1) * s_oh).unsqueeze(-1)
            wv = (alpha_v.unsqueeze(-1) * s_oh).unsqueeze(-1)

            sk_f = (1.0 - wk) * sk_f + wk * kt.unsqueeze(2)
            sv_f = (1.0 - wv) * sv_f + wv * vt.unsqueeze(2)

            upd_mask = (is_merge | is_replace).unsqueeze(-1).float() * s_oh
            self.scar_salience = (1.0 - upd_mask) * self.scar_salience + upd_mask * st.unsqueeze(-1)

        self.scar_keys.copy_(sk_f.to(self.scar_keys.dtype))
        self.scar_vals.copy_(sv_f.to(self.scar_vals.dtype))
        return sync_1d

    def _update_internal(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        prev_pre_sal: Optional[torch.Tensor] = None,
        prev_in_sal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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
            return self.keys, self.values, None, None

        slide_len = self.window_size - self.num_sink_tokens
        evict_end = total_raw - slide_len
        tail_vis = max(slide_len, q_len) if self.project_slots else slide_len
        tail_start = total_raw - tail_vis
        sink_vis = min(self.num_sink_tokens, tail_start) if self.project_slots else self.num_sink_tokens
        split_idx = max(self.num_sink_tokens, tail_start)

        obs_k = full_k[..., -slide_len:, :]
        obs_v = full_v[..., -slide_len:, :]
        s_v = full_v[..., : self.num_sink_tokens, :] if self.num_sink_tokens > 0 else None
        single_shot_prefill = self.active_slots == 0 and split_idx == self.num_sink_tokens

        # 1. Tokens desalojados estritamente ANTES de `tail_k` entram agora (100% causal).
        out_pre_sal: Optional[torch.Tensor] = None
        pre_tail_k = full_k[..., self.num_sink_tokens : split_idx, :]
        pre_tail_v = full_v[..., self.num_sink_tokens : split_idx, :]
        if pre_tail_k.shape[-2] > 0:
            out_pre_sal = self._ingest_evicted_tokens(
                pre_tail_k,
                pre_tail_v,
                obs_k=obs_k,
                obs_v=obs_v,
                sink_v=s_v,
                prev_sync_sal=prev_pre_sal,
                use_obs=False,
            )

        tail_k = full_k[..., -tail_vis:, :]
        tail_v = full_v[..., -tail_vis:, :]
        vis_sink_k = full_k[..., :sink_vis, :]
        vis_sink_v = full_v[..., :sink_vis, :]

        if (
            self.project_slots
            and self.num_slots > 0
            and tail_start > self.num_sink_tokens
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
                ret_k = torch.cat([vis_sink_k, k_slots, tail_k], dim=-2)
                ret_v = torch.cat([vis_sink_v, v_slots, tail_v], dim=-2)
            else:
                ret_k = torch.cat([k_slots, tail_k], dim=-2)
                ret_v = torch.cat([v_slots, tail_v], dim=-2)
        elif sink_vis > 0:
            ret_k = torch.cat([vis_sink_k, tail_k], dim=-2)
            ret_v = torch.cat([vis_sink_v, tail_v], dim=-2)
        else:
            ret_k, ret_v = tail_k, tail_v

        # 2. Tokens já presentes dentro de `tail_k` (`[split_idx : evict_end]`) são absorvidos APÓS
        # construir `(ret_k, ret_v)` (zero vazamento causal, zero duplicação no prefill).
        out_in_sal: Optional[torch.Tensor] = None
        in_tail_k = full_k[..., split_idx:evict_end, :]
        in_tail_v = full_v[..., split_idx:evict_end, :]
        if in_tail_k.shape[-2] > 0:
            out_in_sal = self._ingest_evicted_tokens(
                in_tail_k,
                in_tail_v,
                obs_k=obs_k,
                obs_v=obs_v,
                sink_v=s_v,
                prev_sync_sal=prev_in_sal,
                use_obs=single_shot_prefill,
            )

        self.total_tokens_absorbed += evict_end - self.num_sink_tokens

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

        return ret_k, ret_v, out_pre_sal, out_in_sal

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ret_k, ret_v, _, _ = self._update_internal(key_states, value_states)
        return ret_k, ret_v

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
        merge_cosine_threshold: float = 0.995,
        delta_blend: float = 0.05,
    ):
        super().__init__()
        self.window_size = int(window_size)
        self.num_slots = int(num_slots)
        self.num_sink_tokens = int(num_sink_tokens)
        self.project_slots = bool(project_slots)
        self.merge_cosine_threshold = float(merge_cosine_threshold)
        self.delta_blend = float(delta_blend)
        self._step_pre_sal: Optional[List[List[float]]] = None
        self._step_in_sal: Optional[List[List[float]]] = None
        self.layer_class_to_replicate = lambda: SCacheDynamicLayer(
            window_size=self.window_size,
            num_slots=self.num_slots,
            num_sink_tokens=self.num_sink_tokens,
            project_slots=self.project_slots,
            merge_cosine_threshold=self.merge_cosine_threshold,
            delta_blend=self.delta_blend,
        )

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            self.layers.append(self.layer_class_to_replicate())

        if layer_idx == 0:
            self._step_pre_sal = None
            self._step_in_sal = None

        layer = self.layers[layer_idx]
        if isinstance(layer, SCacheDynamicLayer):
            prev_p = (
                torch.tensor(self._step_pre_sal, dtype=torch.float32) if self._step_pre_sal is not None else None
            )
            prev_i = (
                torch.tensor(self._step_in_sal, dtype=torch.float32) if self._step_in_sal is not None else None
            )
            ret_k, ret_v, out_p, out_i = layer._update_internal(
                key_states, value_states, prev_pre_sal=prev_p, prev_in_sal=prev_i
            )
            if out_p is not None:
                self._step_pre_sal = out_p.detach().cpu().tolist()
            if out_i is not None:
                self._step_in_sal = out_i.detach().cpu().tolist()
            return ret_k, ret_v

        return super().update(key_states, value_states, layer_idx, cache_kwargs)

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
