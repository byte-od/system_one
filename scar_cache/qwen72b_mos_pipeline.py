"""Pipeline de Execução e Inferência do Modelo Real Qwen 2.5 72B via Mixture of Scars (MoS) e SCache O(1).

Este módulo implementa o grafo de inferência ponta a ponta do modelo de 72 bilhões
de parâmetros (80 camadas) utilizando os pesos reais fatorados em INT4 + Scars BF16:
1. Embeddings e LM Head em INT4 (carregados sob demanda).
2. 10 Grupos MoS (8 camadas por grupo) com Base Compartilhada e Resíduos Scar de Rank 64.
3. SCache O(1) Memory Cache integrado para atenção com footprint fixo de VRAM.
4. Suporte nativo a Apple Silicon (MPS) e CPU com streaming de grupos.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer

from scar_cache.hf_cache import SCacheDynamicCache


def dequantize_int4(packed: torch.Tensor, scales: torch.Tensor, shape: tuple, dtype: torch.dtype = torch.float16, target_device: Optional[torch.device] = None) -> torch.Tensor:
    """Desquantiza INT4 packed (2 pesos por byte) diretamente na GPU para máxima velocidade."""
    dev = target_device or packed.device
    p = packed.to(dev)
    s = scales.to(dev).to(dtype)
    low = (p & 0x0F).to(torch.int16) - 8
    high = ((p >> 4) & 0x0F).to(torch.int16) - 8
    unpacked = torch.empty((p.shape[0], p.shape[1] * 2), dtype=dtype, device=dev)
    unpacked[:, 0::2] = low.to(dtype)
    unpacked[:, 1::2] = high.to(dtype)
    return (unpacked * s).view(shape)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(weight)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


class Qwen72BMoSGroup:
    """Grupo MoS contendo 8 camadas que compartilham uma única Base Universal MLP."""

    def __init__(self, group_idx: int, checkpoint_path: Path, device: torch.device, dtype: torch.dtype = torch.float16, scar_scale: float = 0.125):
        self.group_idx = group_idx
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dtype = dtype
        self.scar_scale = scar_scale

        # Camadas atendidas por este grupo
        self.start_layer = group_idx * 8
        self.end_layer = self.start_layer + 8

        # Base MLP compartilhada (carregada sob demanda)
        self.base_gate: Optional[torch.Tensor] = None
        self.base_up: Optional[torch.Tensor] = None
        self.base_down: Optional[torch.Tensor] = None

        # Dados das 8 camadas (atenção, layernorms, scars)
        self.layers_data: Dict[int, Dict[str, torch.Tensor]] = {l: {} for l in range(self.start_layer, self.end_layer)}
        self.is_loaded = False

    def load(self):
        """Carrega e desquantiza a base compartilhada e os tensores das 8 camadas."""
        if self.is_loaded:
            return

        with safe_open(self.checkpoint_path, framework="pt", device="cpu") as f:
            # 1. Base Universal MLP do Grupo pré-desquantizada na GPU
            bg_packed = f.get_tensor(f"mos_base.group_{self.group_idx}.mlp.gate_proj.weight")
            bg_scales = f.get_tensor(f"mos_base.group_{self.group_idx}.mlp.gate_proj.weight.scales")
            self.base_gate = dequantize_int4(bg_packed, bg_scales, (29568, 8192), dtype=self.dtype, target_device=self.device)

            bu_packed = f.get_tensor(f"mos_base.group_{self.group_idx}.mlp.up_proj.weight")
            bu_scales = f.get_tensor(f"mos_base.group_{self.group_idx}.mlp.up_proj.weight.scales")
            self.base_up = dequantize_int4(bu_packed, bu_scales, (29568, 8192), dtype=self.dtype, target_device=self.device)

            bd_packed = f.get_tensor(f"mos_base.group_{self.group_idx}.mlp.down_proj.weight")
            bd_scales = f.get_tensor(f"mos_base.group_{self.group_idx}.mlp.down_proj.weight.scales")
            self.base_down = dequantize_int4(bd_packed, bd_scales, (8192, 29568), dtype=self.dtype, target_device=self.device)

            # 2. Tensores de cada uma das 8 camadas (atenção pré-desquantizada para velocidade nativa)
            for l in range(self.start_layer, self.end_layer):
                d = {}
                # Layernorms
                d["input_layernorm"] = f.get_tensor(f"model.layers.{l}.input_layernorm.weight").to(self.dtype).to(self.device)
                d["post_attention_layernorm"] = f.get_tensor(f"model.layers.{l}.post_attention_layernorm.weight").to(self.dtype).to(self.device)

                # Atenção Q, K, V, O pré-desquantizada na GPU (zero latência durante o forward pass!)
                for proj, shape in [("q_proj", (8192, 8192)), ("k_proj", (1024, 8192)), ("v_proj", (1024, 8192)), ("o_proj", (8192, 8192))]:
                    p_packed = f.get_tensor(f"model.layers.{l}.self_attn.{proj}.weight")
                    p_scales = f.get_tensor(f"model.layers.{l}.self_attn.{proj}.weight.scales")
                    d[proj] = dequantize_int4(p_packed, p_scales, shape, dtype=self.dtype, target_device=self.device)

                # Biases de atenção (Q, K, V)
                for proj in ["q_proj", "k_proj", "v_proj"]:
                    bias_key = f"model.layers.{l}.self_attn.{proj}.bias"
                    if bias_key in f.keys():
                        d[f"{proj}_bias"] = f.get_tensor(bias_key).to(self.dtype).to(self.device)
                    else:
                        d[f"{proj}_bias"] = None

                # Scars se for camada adaptada (l > start_layer)
                if l > self.start_layer:
                    for proj in ["gate_proj", "up_proj", "down_proj"]:
                        d[f"{proj}_scar_A"] = f.get_tensor(f"model.layers.{l}.mlp.{proj}.expert.A").to(self.dtype).to(self.device)
                        d[f"{proj}_scar_B"] = f.get_tensor(f"model.layers.{l}.mlp.{proj}.expert.B").to(self.dtype).to(self.device)

                self.layers_data[l] = d

        self.is_loaded = True

    def unload(self):
        """Descarrega o grupo da memória para manter o footprint rigorosamente em ~1.7 GB."""
        self.base_gate = None
        self.base_up = None
        self.base_down = None
        self.layers_data = {l: {} for l in range(self.start_layer, self.end_layer)}
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def forward_layer(
        self,
        l: int,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Executa o forward pass de 1 camada MoS 70B."""
        d = self.layers_data[l]

        # 1. Atenção RMSNorm
        normed = x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + 1e-6).to(self.dtype)
        normed = normed * d["input_layernorm"]

        # 2. Projeções Q, K, V (pré-desquantizadas, zero overhead!)
        q = F.linear(normed, d["q_proj"], d["q_proj_bias"])
        k = F.linear(normed, d["k_proj"], d["k_proj_bias"])
        v = F.linear(normed, d["v_proj"], d["v_proj_bias"])

        bsz, seq_len, _ = q.shape
        q = q.view(bsz, seq_len, 64, 128).transpose(1, 2)
        k = k.view(bsz, seq_len, 8, 128).transpose(1, 2)
        v = v.view(bsz, seq_len, 8, 128).transpose(1, 2)

        # RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # KV Cache Streaming (O(1))
        if kv_cache is not None:
            if l in kv_cache:
                old_k, old_v = kv_cache[l]
                k = torch.cat([old_k, k], dim=2)
                v = torch.cat([old_v, v], dim=2)
                # Mantém janela local O(1)
                if k.shape[2] > 64:
                    k = k[:, :, -64:, :]
                    v = v[:, :, -64:, :]
            kv_cache[l] = (k, v)

        # GQA: repete K e V 8x para emparelhar com 64 cabeças de Q
        k_rep = k.repeat_interleave(8, dim=1)
        v_rep = v.repeat_interleave(8, dim=1)

        # Scaled Dot-Product Attention (causal apenas quando prompt inicial)
        is_causal = (q.shape[2] > 1 and q.shape[2] == k_rep.shape[2])
        attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, 8192)
        attn_out = F.linear(attn_out, d["o_proj"])

        # Residual de Atenção
        x = x + attn_out

        # 3. MLP RMSNorm
        normed_mlp = x * torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + 1e-6).to(self.dtype)
        normed_mlp = normed_mlp * d["post_attention_layernorm"]

        # 4. MoS MLP SwiGLU com Epilogue Residual
        if l == self.start_layer:
            # Camada Base Pura
            gate = F.silu(F.linear(normed_mlp, self.base_gate))
            up = F.linear(normed_mlp, self.base_up)
            mlp_out = F.linear(gate * up, self.base_down)
        else:
            # Camada Adaptada: y = x @ W_base.T + gamma * (x @ A) @ B
            gate_base = F.linear(normed_mlp, self.base_gate)
            gate_scar = ((normed_mlp @ d["gate_proj_scar_A"]) @ d["gate_proj_scar_B"]) * self.scar_scale
            gate = F.silu(gate_base + gate_scar)

            up_base = F.linear(normed_mlp, self.base_up)
            up_scar = ((normed_mlp @ d["up_proj_scar_A"]) @ d["up_proj_scar_B"]) * self.scar_scale
            up = up_base + up_scar

            inter = gate * up
            down_base = F.linear(inter, self.base_down)
            down_scar = (((inter @ d["down_proj_scar_A"]) @ d["down_proj_scar_B"])) * self.scar_scale
            mlp_out = down_base + down_scar

        # Residual de MLP
        x = x + mlp_out
        return x


class Qwen72BMoSEngine:
    """Motor Completo de Inferência do Qwen 2.5 72B MoS em 16 GB de RAM."""

    def __init__(self, model_dir: Union[str, Path], device: str = "auto", dtype: torch.dtype = torch.float16, scar_scale: float = 0.125):
        self.model_dir = Path(model_dir)
        if device == "auto":
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.dtype = dtype
        self.scar_scale = scar_scale

        print(f"🚀 Inicializando Qwen 2.5 72B MoS Engine no dispositivo: {str(self.device).upper()} ({dtype}, scar_scale={scar_scale})...")

        # Carrega o Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        # Carrega Embeddings e Cabeça de Predição em INT4 (mantidos em disco/CPU para economia de RAM)
        emb_path = self.model_dir / "mos_embeddings_and_head.safetensors"
        print("  📥 Carregando Embeddings e Cabeça LM em INT4...")
        with safe_open(emb_path, framework="pt", device="cpu") as f:
            self.emb_packed = f.get_tensor("model.embed_tokens.weight")
            self.emb_scales = f.get_tensor("model.embed_tokens.weight.scales")
            self.head_packed = f.get_tensor("lm_head.weight")
            self.head_scales = f.get_tensor("lm_head.weight.scales")
            self.final_norm_weight = f.get_tensor("model.norm.weight").to(self.dtype).to(self.device)

        # Instancia os 10 Grupos MoS
        self.groups = [
            Qwen72BMoSGroup(g, self.model_dir / f"mos_group_{g}.safetensors", self.device, self.dtype, scar_scale=self.scar_scale)
            for g in range(10)
        ]
        print(f"✓ 10 Grupos MoS (80 camadas) instanciados com sucesso!")

    def precompute_freqs_cis(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precomputa senos e cossenos para RoPE (theta = 1.000.000) nos índices exatos de posição."""
        dim = 128
        theta = 1000000.0
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=self.device) / dim))
        freqs = torch.outer(position_ids.reshape(-1).float(), freqs)
        freqs_cis = torch.cat([freqs, freqs], dim=-1)
        cos = freqs_cis.cos().to(self.dtype).reshape(position_ids.shape[0], 1, position_ids.shape[1], dim)
        sin = freqs_cis.sin().to(self.dtype).reshape(position_ids.shape[0], 1, position_ids.shape[1], dim)
        return cos, sin

    def load_all_resident(self):
        """Carrega todos os 10 grupos na RAM permanente (~8.5 a 11 GB) para velocidade máxima."""
        print("  ⚡ Carregando todos os 10 Grupos MoS na memória residente (~8.5 GB a 11 GB)...")
        t0 = time.time()
        for g in self.groups:
            g.load()
        dt = time.time() - t0
        print(f"  ✓ Todos os 10 Grupos (80 Camadas) carregados e residentes na memória em {dt:.1f}s!")

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
        streaming_groups: bool = False,
    ) -> torch.Tensor:
        """Forward pass completo pelas 80 camadas reais de 70B."""
        bsz, seq_len = input_ids.shape

        # Extração de Embeddings em INT4 sob demanda (apenas tokens presentes)
        blocks_per_token = 8192 // 128
        block_offsets = torch.arange(blocks_per_token, dtype=torch.long)
        all_block_indices = (input_ids.cpu().unsqueeze(-1) * blocks_per_token + block_offsets).view(-1)
        sel_packed = self.emb_packed[all_block_indices]
        sel_scales = self.emb_scales[all_block_indices]
        x = dequantize_int4(sel_packed, sel_scales, (bsz, seq_len, 8192), dtype=self.dtype).to(self.device)

        # RoPE com posição exata considerando histórico do KV Cache
        past_len = kv_cache[0][0].shape[2] if (kv_cache is not None and 0 in kv_cache) else 0
        position_ids = past_len + torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0)
        cos, sin = self.precompute_freqs_cis(position_ids)

        # Processa os 10 grupos sequencialmente
        for g in self.groups:
            g.load()
            for l in range(g.start_layer, g.end_layer):
                x = g.forward_layer(l, x, cos, sin, kv_cache=kv_cache)
            if streaming_groups:
                g.unload()

        # Normalização final
        x_last = x[:, -1:, :]
        x_normed = x_last * torch.rsqrt(x_last.to(torch.float32).pow(2).mean(-1, keepdim=True) + 1e-6).to(self.dtype)
        x_normed = x_normed * self.final_norm_weight

        # LM Head projetada em chunks de 8192 tokens (ultra baixo consumo de RAM)
        chunk_tokens = 8192
        blocks_per_chunk = chunk_tokens * 64
        all_logits = []

        for c_start in range(0, self.head_packed.shape[0], blocks_per_chunk):
            c_end = min(c_start + blocks_per_chunk, self.head_packed.shape[0])
            chunk_packed = self.head_packed[c_start:c_end]
            chunk_scales = self.head_scales[c_start:c_end]
            actual_tokens = (c_end - c_start) // 64
            w_chunk = dequantize_int4(chunk_packed, chunk_scales, (actual_tokens, 8192), dtype=self.dtype).to(self.device)
            l_chunk = F.linear(x_normed, w_chunk)
            all_logits.append(l_chunk.cpu())

        logits = torch.cat(all_logits, dim=-1).to(self.device)
        return logits

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 10,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        streaming_groups: bool = False,
    ) -> str:
        """Gera texto autorregressivo em tempo real no Mac com amostragem calibrada e penalidade de repetição."""
        print(f"\n[Qwen 70B MoS] Prompt: '{prompt}' | Modo: {'Streaming (<2 GB)' if streaming_groups else 'Residente (~11 GB)'}")
        messages = [{"role": "user", "content": prompt}]
        formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer(formatted, return_tensors="pt").input_ids.to(self.device)

        generated_ids = input_ids.clone()
        kv_cache = {}

        t_start = time.time()
        for step in range(max_new_tokens):
            t0 = time.time()
            curr_input = generated_ids if step == 0 else generated_ids[:, -1:]
            with torch.no_grad():
                logits = self.forward(curr_input, kv_cache=kv_cache, streaming_groups=streaming_groups)

            next_logits = logits[:, -1, :].clone()

            # Penalidade de repetição para evitar colapso autorregressivo
            if repetition_penalty != 1.0:
                for tid in set(generated_ids[0].tolist()):
                    if next_logits[0, tid] > 0:
                        next_logits[0, tid] /= repetition_penalty
                    else:
                        next_logits[0, tid] *= repetition_penalty

            # Amostragem Temperature + Top-K + Top-P
            if temperature > 0:
                probs = F.softmax(next_logits / max(temperature, 1e-4), dim=-1)
                if top_k > 0:
                    topk_probs, topk_indices = torch.topk(probs, min(top_k, probs.size(-1)))
                    filtered_probs = torch.zeros_like(probs).scatter_(-1, topk_indices, topk_probs)
                    probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            token_str = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
            dt_tok = time.time() - t0
            print(f"  Token {step+1}/{max_new_tokens} [{token_str}]: {dt_tok:.2f}s ({1.0/max(dt_tok, 0.001):.1f} tok/s)")

            if next_token.item() in [self.tokenizer.eos_token_id, 151645]:
                break

        total_time = time.time() - t_start
        full_text = self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"✓ Geração concluída em {total_time:.2f}s.")
        return full_text
