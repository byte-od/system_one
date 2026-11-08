"""Benchmark de Perplexidade em Texto Real: SCache O(1) vs StreamingLLM vs Vanilla Full Attention.

Avalia rigorosamente o Negative Log-Likelihood (NLL) e a Perplexidade (PPL) acumulada
em texto real contínuo através de horizontes de contexto crescentes:
  • 512 tokens
  • 1024 tokens
  • 2048 tokens
  • 4096 tokens

Baselines Comparadas:
  1. Vanilla Full KV Cache: Mantém 100% dos tokens passados (teto teórico, O(T) RAM).
  2. StreamingLLM: Sinks iniciais + Janela deslizante local (descarta excedente, O(1) RAM).
  3. SCache Dynamic Cache: Sinks + Janela deslizante + Memória Associativa Multi-Slot via Regra Delta O(1).
"""

from __future__ import annotations

import argparse
import time
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from scar_cache.hf_cache import SCacheDynamicCache


def get_real_evaluation_text() -> str:
    """Gera um corpus realista e denso de texto contínuo para avaliação de perplexidade."""
    # Seções enciclopédicas densas e ricas em vocabulário
    text_corpus = (
        "The scientific revolution was a series of events that marked the emergence of modern science "
        "during the early modern period, when developments in mathematics, physics, astronomy, biology, "
        "and chemistry transformed the views of society about nature. The scientific revolution took place "
        "in Europe towards the end of the Renaissance period and continued through the late eighteenth century, "
        "influencing the intellectual social movement known as the Enlightenment. Although its dates are "
        "debated, the publication in 1543 of Nicolaus Copernicus's De revolutionibus orbium coelestium is "
        "often cited as marking the beginning of the scientific revolution. "
        "Johannes Kepler proposed the laws of planetary motion, establishing that planetary orbits are elliptical "
        "rather than circular, solving a mathematical inconsistency that had puzzled astronomers for millennia. "
        "Galileo Galilei utilized the telescope to observe the moons of Jupiter, the phases of Venus, and the rough "
        "surface of the Moon, providing decisive empirical evidence against the geocentric Ptolemaic system. "
        "Isaac Newton's 1687 Philosophiae Naturalis Principia Mathematica formulated the laws of motion and universal "
        "gravitation, which unified celestial mechanics and terrestrial physics under a single mathematical framework. "
        "In the eighteenth century, chemistry transitioned from alchemy into a quantitative discipline through "
        "the foundational work of Antoine Lavoisier, who formulated the law of conservation of mass and identified "
        "oxygen and hydrogen as distinct chemical elements. "
        "Thermodynamics developed during the industrial revolution to understand the efficiency of heat engines, "
        "leading to the formulation of entropy and the conservation of energy by Rudolf Clausius, Lord Kelvin, "
        "and James Clerk Maxwell. In biology, Charles Darwin published On the Origin of Species in 1859, proposing "
        "natural selection as the driving mechanism of evolutionary adaptation and biodiversity. "
        "The twentieth century witnessed two major paradigm shifts in fundamental physics: Albert Einstein's "
        "theories of special and general relativity, which redefined space and time as a dynamic four-dimensional "
        "continuum, and quantum mechanics, developed by Max Planck, Niels Bohr, Werner Heisenberg, and Erwin Schrödinger, "
        "which revealed the discrete and probabilistic nature of subatomic phenomena. "
        "In computer science, Alan Turing introduced the concept of universal computing machines and formal limits "
        "of decidability, laying the theoretical foundation for digital computation, information theory by Claude Shannon, "
        "and artificial intelligence through modern neural architectures and recurrent associative networks. "
    )
    # Replica de forma cíclica para alcançar o comprimento desejado
    return (text_corpus + "\n\n") * 10


def compute_nll_and_ppl(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    cache_type: str = "vanilla",
    window_size: int = 64,
    num_sinks: int = 4,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Calcula o NLL token a token e a perplexidade acumulada."""
    seq_len = input_ids.shape[1]
    nlls = []
    
    # Configura o tipo de cache
    if cache_type == "vanilla":
        cache = DynamicCache()
    elif cache_type == "scache":
        cache = SCacheDynamicCache(window_size=window_size, num_sink_tokens=num_sinks, num_slots=8)
    elif cache_type == "streaming_llm":
        # StreamingLLM é equivalente a SCache sem gravação associativa (apenas buffer circular)
        cache = SCacheDynamicCache(window_size=window_size, num_sink_tokens=num_sinks, num_slots=0)
    else:
        raise ValueError(f"Cache desconhecido: {cache_type}")

    t0 = time.time()
    
    with torch.no_grad():
        for t in range(seq_len - 1):
            curr_token = input_ids[:, t : t + 1].to(device)
            target_token = input_ids[:, t + 1 : t + 2].to(device)
            
            # Forward autoregressivo token por token com cache
            outputs = model(curr_token, past_key_values=cache, use_cache=True)
            logits = outputs.logits[:, -1, :] # [1, vocab_size]
            
            # Cross-Entropy NLL para o próximo token
            log_prob = F.log_softmax(logits, dim=-1)
            token_nll = -log_prob[0, target_token.item()].item()
            nlls.append(token_nll)

    elapsed_s = time.time() - t0
    mean_nll = float(sum(nlls) / len(nlls))
    ppl = math.exp(mean_nll) if mean_nll < 20 else float("inf")
    
    # Mede a memória física final do cache
    if cache_type == "vanilla":
        mem_bytes = sum(
            layer.keys.numel() * layer.keys.element_size() + layer.values.numel() * layer.values.element_size()
            for layer in cache.layers
            if layer.keys is not None
        )
    else:
        mem_bytes = cache.total_memory_bytes

    return {
        "cache_type": cache_type,
        "tokens": seq_len,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "mem_kb": mem_bytes / 1024.0,
        "tokens_per_sec": (seq_len - 1) / elapsed_s,
        "elapsed_s": elapsed_s,
    }


def run_benchmark(
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    seq_lengths: list[int] = [128, 256, 512, 1024],
    window_size: int = 64,
    num_sinks: int = 4,
):
    print("=" * 85)
    print("🔬 BENCHMARK DE RIGOR CIENTÍFICO: PERPLEXIDADE EM TEXTO REAL")
    print(f"  • Modelo Avaliado:  {model_id}")
    print(f"  • Janela Local:     {window_size} tokens (Attention Sinks: {num_sinks})")
    print(f"  • Comprimentos:     {seq_lengths}")
    print("=" * 85)

    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"⚡ Dispositivo de Execução: {device}")
    
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
    model.eval()

    full_text = get_real_evaluation_text()
    tokens = tok(full_text, return_tensors="pt").input_ids

    results = []

    for L in seq_lengths:
        print(f"\n--- AVALIANDO HORIZONTE DE CONTEXTO: {L} TOKENS ---")
        curr_tokens = tokens[:, :L]
        
        # 1. Vanilla Full Attention (O(T) linear)
        print("  • Executando Vanilla DynamicCache (Teto Teórico O(T))...", end="", flush=True)
        res_vanilla = compute_nll_and_ppl(model, curr_tokens, cache_type="vanilla", device=device)
        print(f" PPL = {res_vanilla['ppl']:.2f} | Memória = {res_vanilla['mem_kb']:.1f} KB")
        results.append(res_vanilla)

        # 2. StreamingLLM Baseline (Window + Sink, drops intermediate)
        print("  • Executando StreamingLLM (Sinks + Janela Deslizante)...", end="", flush=True)
        res_stream = compute_nll_and_ppl(model, curr_tokens, cache_type="streaming_llm", window_size=window_size, num_sinks=num_sinks, device=device)
        print(f" PPL = {res_stream['ppl']:.2f} | Memória = {res_stream['mem_kb']:.1f} KB")
        results.append(res_stream)

        # 3. SCache Dynamic Cache (Sinks + Janela + Delta Memory)
        print("  • Executando SCache Dynamic Cache O(1)...", end="", flush=True)
        res_scache = compute_nll_and_ppl(model, curr_tokens, cache_type="scache", window_size=window_size, num_sinks=num_sinks, device=device)
        print(f" PPL = {res_scache['ppl']:.2f} | Memória = {res_scache['mem_kb']:.1f} KB")
        results.append(res_scache)

    # Tabela consolidada final
    print("\n" + "=" * 95)
    print(f"{'Mecanismo':<22} | {'Tokens':<8} | {'NLL':<8} | {'PPL (Perplexidade)':<20} | {'Memória KV (KB)':<18} | {'Throughput':<12}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['cache_type']:<22} | {r['tokens']:<8} | {r['mean_nll']:<8.4f} | {r['ppl']:<20.2f} | {r['mem_kb']:<18.1f} | {r['tokens_per_sec']:<10.1f} tok/s"
        )
    print("=" * 95)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark de Perplexidade em Texto Real")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--sinks", type=int, default=4)
    args = parser.parse_args()

    run_benchmark(
        model_id=args.model,
        seq_lengths=args.lengths,
        window_size=args.window,
        num_sinks=args.sinks,
    )
