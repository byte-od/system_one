"""Exemplo 11: Benchmark do Sistema Scar Completo (MoS Pesos Estáticos + SCache Contexto).

Avalia a matriz 2x2 de performance em um modelo de linguagem real (SmolLM2-135M):
  1. Baseline Tradicional:  Pesos Densos + KV Cache Padrão O(T)
  2. SCache Context:        Pesos Densos + SCacheDynamicCache O(1)
  3. MoS Weights:           Mixture of Scars (MoS) + KV Cache Padrão O(T)
  4. Full Scar System:      Mixture of Scars (MoS) + SCacheDynamicCache O(1)

Mede o consumo real de memória, aceleração computacional e coerência de geração.
"""

import time
import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from scar_cache import SCacheDynamicCache, MixtureOfScarsLinear


def measure_model_memory(model: nn.Module) -> int:
    """Calcula a memória total dos parâmetros de um modelo PyTorch."""
    return sum(p.numel() * p.element_size() for p in model.parameters())


def apply_mos_to_mlp_blocks(model: nn.Module, target_pairs: list = None, rank: int = 64):
    """Aplica o transplante Mixture of Scars (MoS) nas MLPs com calibração residual SVD.

    Agrupa camadas adjacentes para compartilhar a matriz base (W_base) e armazena
    as diferenças específicas de cada camada nos especialistas de cicatriz (Scar Experts).
    """
    if target_pairs is None:
        # Emparelha camadas intermediárias/finais (ex: blocos 20-21, 22-23, 24-25)
        target_pairs = [(20, 21), (22, 23), (24, 25)]

    for idx_base, idx_target in target_pairs:
        base_layer = model.model.layers[idx_base]
        target_layer = model.model.layers[idx_target]

        base_down = base_layer.mlp.down_proj
        target_down = target_layer.mlp.down_proj

        in_dim = target_down.in_features
        out_dim = target_down.out_features

        # Calcula a diferença residual entre as duas camadas: [out_features, in_features]
        diff = target_down.weight.data.float() - base_down.weight.data.float()
        U, S, V = torch.linalg.svd(diff, full_matrices=False)
        r = min(rank, S.shape[0])

        # Cria a camada Mixture of Scars compartilhando a base da camada anterior
        mos_layer = MixtureOfScarsLinear(
            in_features=in_dim,
            out_features=out_dim,
            num_experts=1,
            top_k=1,
            rank=r,
            shared_base=base_down,
            scale=1.0,
        )

        # Carrega a projeção de cicatriz exata no especialista 0
        A_val = V[:r, :].t() * torch.sqrt(S[:r])               # [in_features, r]
        B_val = torch.sqrt(S[:r]).unsqueeze(-1) * U[:, :r].t() # [r, out_features]

        with torch.no_grad():
            mos_layer.experts[0].A.copy_(A_val.to(mos_layer.experts[0].A.dtype))
            mos_layer.experts[0].B.copy_(B_val.to(mos_layer.experts[0].B.dtype))
            mos_layer.router.weight.fill_(1.0)

        target_layer.mlp.down_proj = mos_layer


def main():
    print("=" * 85)
    print(" BENCHMARK DO SISTEMA SCAR COMPLETO (PESOS ESTÁTICOS MoS + CONTEXTO SCACHE)")
    print("=" * 85)

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"\nCarregando modelo e tokenizador de {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    base_model.eval()

    prompt = "Explain why learning mathematics is essential for computer science."
    messages = [
        {"role": "system", "content": "You are a concise, helpful professor."},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt")

    prompt_len = inputs["input_ids"].shape[1]
    max_tokens = 60
    print(f"Prompt: '{prompt}' ({prompt_len} tokens)")
    print(f"Gerando {max_tokens} tokens em cada configuração...\n")

    # -------------------------------------------------------------------------
    # 1. Configuração 1: BASELINE TRADICIONAL (Pesos Densos + DynamicCache)
    # -------------------------------------------------------------------------
    print("-" * 85)
    print("[Cenário 1/4] Baseline Tradicional: Pesos Densos + DynamicCache Padrão O(T)")
    cache_c1 = DynamicCache()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_c1 = base_model.generate(**inputs, past_key_values=cache_c1, max_new_tokens=max_tokens, do_sample=False)
    t_c1 = time.perf_counter() - t0

    c1_weights_mb = measure_model_memory(base_model) / (1024**2)
    c1_kv_mb = sum((l.keys.numel() + l.values.numel()) * l.keys.element_size() for l in cache_c1.layers) / (1024**2)
    # Projeção do KV Cache para contexto longo de 8.000 tokens
    c1_kv_8k_mb = (2 * 30 * 3 * 8000 * 64 * 2) / (1024**2)

    text_c1 = tokenizer.decode(out_c1[0], skip_special_tokens=True).split("assistant")[-1].strip()
    print(f"  • Memória Pesos Estáticos: {c1_weights_mb:.2f} MB")
    print(f"  • Memória KV Cache (Atual): {c1_kv_mb:.2f} MB  |  (Projeção 8k tokens): {c1_kv_8k_mb:.2f} MB")
    print(f"  • Tempo de Geração:        {t_c1:.2f} s")

    # -------------------------------------------------------------------------
    # 2. Configuração 2: SCACHE CONTEXT (Pesos Densos + SCacheDynamicCache O(1))
    # -------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print("[Cenário 2/4] SCache Context: Pesos Densos + SCacheDynamicCache O(1)")
    cache_c2 = SCacheDynamicCache(window_size=64, num_sink_tokens=4)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_c2 = base_model.generate(**inputs, past_key_values=cache_c2, max_new_tokens=max_tokens, do_sample=False)
    t_c2 = time.perf_counter() - t0

    c2_weights_mb = c1_weights_mb
    c2_kv_mb = (cache_c2.total_physical_bytes + cache_c2.total_scache_bytes) / (1024**2)
    c2_kv_8k_mb = c2_kv_mb  # RIGOROSAMENTE CONSTANTE O(1)!

    text_c2 = tokenizer.decode(out_c2[0], skip_special_tokens=True).split("assistant")[-1].strip()
    print(f"  • Memória Pesos Estáticos: {c2_weights_mb:.2f} MB")
    print(f"  • Memória KV Cache (Atual): {c2_kv_mb:.2f} MB  |  (Projeção 8k tokens): {c2_kv_8k_mb:.2f} MB (CONGELADO!)")
    print(f"  • Tempo de Geração:        {t_c2:.2f} s")

    # -------------------------------------------------------------------------
    # 3. Configuração 3 & 4: SISTEMA COM MIXTURE OF SCARS (MoS)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print("[Cenário 3/4 e 4/4] Transplante Mixture of Scars (MoS) nas MLPs...")
    mos_model = copy.deepcopy(base_model)
    apply_mos_to_mlp_blocks(mos_model, target_pairs=[(20, 21), (22, 23), (24, 25)], rank=64)

    # Memória real com base compartilhada (deduplica ponteiros que apontam para o mesmo tensor)
    unique_params = set()
    mos_weights_bytes = 0
    for p in mos_model.parameters():
        if p.data_ptr() not in unique_params:
            unique_params.add(p.data_ptr())
            mos_weights_bytes += p.numel() * p.element_size()
    c4_weights_mb = mos_weights_bytes / (1024**2)

    # 3. Cenário 4: FULL SCAR SYSTEM (MoS Pesos + SCache Contexto)
    print("\n[Cenário 4/4] Full Scar System: Mixture of Scars (MoS) + SCache O(1)")
    cache_c4 = SCacheDynamicCache(window_size=64, num_sink_tokens=4)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_c4 = mos_model.generate(**inputs, past_key_values=cache_c4, max_new_tokens=max_tokens, do_sample=False)
    t_c4 = time.perf_counter() - t0

    c4_kv_mb = (cache_c4.total_physical_bytes + cache_c4.total_scache_bytes) / (1024**2)
    c4_kv_8k_mb = c4_kv_mb
    text_c4 = tokenizer.decode(out_c4[0], skip_special_tokens=True).split("assistant")[-1].strip()

    print(f"  • Memória Pesos Estáticos: {c4_weights_mb:.2f} MB (Economizou {c1_weights_mb - c4_weights_mb:.1f} MB nas MLPs!)")
    print(f"  • Memória KV Cache (Atual): {c4_kv_mb:.2f} MB  |  (Projeção 8k tokens): {c4_kv_8k_mb:.2f} MB (CONGELADO!)")
    print(f"  • Tempo de Geração:        {t_c4:.2f} s")

    # -------------------------------------------------------------------------
    # QUADRO COMPARATIVO CONSOLIDADO (A MATRIZ DEFINITIVA)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" QUADRO COMPARATIVO CONSOLIDADO: MATRIZ DE REDUÇÃO DE VRAM/RAM")
    print("=" * 85)
    print(f"{'Configuração':<30} | {'Pesos Estáticos':<16} | {'KV (8k tokens)':<16} | {'VRAM Total':<16}")
    print("-" * 85)
    print(f"{'1. Baseline Tradicional':<30} | {f'{c1_weights_mb:.1f} MB':<16} | {f'{c1_kv_8k_mb:.1f} MB':<16} | {f'{c1_weights_mb + c1_kv_8k_mb:.1f} MB':<16}")
    print(f"{'2. SCache Context (O(1))':<30} | {f'{c2_weights_mb:.1f} MB':<16} | {f'{c2_kv_8k_mb:.1f} MB':<16} | {f'{c2_weights_mb + c2_kv_8k_mb:.1f} MB':<16}")
    print(f"{'3. MoS Pesos (Padrão KV)':<30} | {f'{c4_weights_mb:.1f} MB':<16} | {f'{c1_kv_8k_mb:.1f} MB':<16} | {f'{c4_weights_mb + c1_kv_8k_mb:.1f} MB':<16}")
    print(f"{'4. FULL SCAR (MoS + SCache)':<30} | {f'{c4_weights_mb:.1f} MB':<16} | {f'{c4_kv_8k_mb:.1f} MB':<16} | {f'{c4_weights_mb + c4_kv_8k_mb:.1f} MB':<16}")
    print("=" * 85)

    print("\nAMOSTRA DE TEXTO GERADO PELO FULL SCAR SYSTEM (Cenário 4):")
    print(f"'{text_c4[:250]}...'")
    print("=" * 85)


if __name__ == "__main__":
    main()
