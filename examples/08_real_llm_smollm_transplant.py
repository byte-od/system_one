"""Exemplo 08: Transplante Real do SCache em um LLM da HuggingFace (SmolLM2-135M).

Substitui o DynamicCache ilimitado da HuggingFace pelo SCacheDynamicCache,
congelando a memória física em uma janela de 64 tokens enquanto comprime
todo o histórico passado em tensores MultiSlotSCache no L1 Cache via Regra Delta.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from scar_cache import SCacheDynamicCache


def main():
    print("=" * 80)
    print(" TRANSPLANTE REAL: SCACHE EM UM LLM PRE-TREINADO (SmolLM2-135M-Instruct)")
    print("=" * 80)

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    print(f"\nCarregando modelo e tokenizador de {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    prompt = "Explain in simple terms how computers store and process memory."
    messages = [
        {"role": "system", "content": "You are a concise, helpful computer science teacher."},
        {"role": "user", "content": prompt},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt")

    prompt_len = inputs["input_ids"].shape[1]
    max_new_tokens = 90
    print(f"Prompt: '{prompt}' ({prompt_len} tokens de entrada)")
    print(f"Gerando {max_new_tokens} tokens adicionais...\n")

    # 1. GERAÇÃO COM KV CACHE PADRÃO DA HUGGINGFACE (DynamicCache)
    print("-" * 80)
    print("[1/2] Geração com KV Cache Tradicional da HuggingFace (DynamicCache):")
    std_cache = DynamicCache()

    t0_std = time.perf_counter()
    with torch.no_grad():
        out_std = model.generate(
            **inputs,
            past_key_values=std_cache,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
        )
    t1_std = time.perf_counter()

    std_text = tokenizer.decode(out_std[0], skip_special_tokens=True)
    std_seq_len = std_cache.layers[0].keys.shape[-2]
    # 30 camadas * (k + v) * num_kv_heads (3) * seq_len * head_dim (64) * 2 bytes (bfloat16)
    std_bytes = sum(
        (layer.keys.numel() + layer.values.numel()) * layer.keys.element_size()
        for layer in std_cache.layers
    )

    print(f"Texto Gerado:\n{std_text.split('assistant')[-1].strip()}\n")
    print(f"Métricas KV Cache Tradicional:")
    print(f"  • Comprimento do tensor K/V na camada 0: {std_seq_len} tokens (Cresceu livremente O(T))")
    print(f"  • Memória total consumida em RAM:        {std_bytes / 1024:.2f} KB ({std_bytes} bytes)")
    print(f"  • Tempo de geração:                      {(t1_std - t0_std):.2f} s")

    # 2. GERAÇÃO COM SCACHE DYNAMIC CACHE (CAPPED AT 64 TOKENS NO L1)
    print("\n" + "-" * 80)
    print("[2/2] Geração com SCacheDynamicCache (Janela Física Congelada em 64 tokens + MultiSlotSCache):")
    window_size = 64
    scar_cache = SCacheDynamicCache(window_size=window_size, num_slots=8)

    t0_scar = time.perf_counter()
    with torch.no_grad():
        out_scar = model.generate(
            **inputs,
            past_key_values=scar_cache,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
        )
    t1_scar = time.perf_counter()

    scar_text = tokenizer.decode(out_scar[0], skip_special_tokens=True)
    scar_seq_len = scar_cache.layers[0].keys.shape[-2]
    total_tokens_generated = out_scar.shape[1]
    absorbed = scar_cache.layers[0].total_tokens_absorbed

    print(f"Texto Gerado:\n{scar_text.split('assistant')[-1].strip()}\n")
    print(f"Métricas SCache Transplante:")
    print(f"  • Comprimento FÍSICO do tensor K/V:     {scar_seq_len} tokens (CONGELADO em {window_size}!)")
    print(f"  • Tokens excedentes absorvidos no SCache: {absorbed} tokens comprimidos no L1 via Regra Delta")
    print(f"  • Memória física da janela:              {scar_cache.total_physical_bytes / 1024:.2f} KB")
    print(f"  • Tempo de geração:                      {(t1_scar - t0_scar):.2f} s")

    print("\n" + "=" * 80)
    print(" CONCLUSÃO DO TRANSPLANTE EM LLM REAL:")
    print(f"  • O modelo gerou {total_tokens_generated} tokens totais.")
    print(f"  • O KV Cache padrão cresceu para {std_seq_len} tokens em RAM.")
    print(f"  • O SCache travou o tensor em exatamente {scar_seq_len} tokens, absorvendo o passado em O(1)!")
    print("=" * 80)


if __name__ == "__main__":
    main()
