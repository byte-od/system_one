import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from scar_cache import NativeMoS70B, MixtureOfScarsLinear, SCacheDynamicCache


def test_native_mos_70b_initialization_and_forward():
    """Valida a instanciação da arquitetura profunda de 72 camadas MoS com SmolLM2."""
    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

    mos_70b = NativeMoS70B.from_pretrained(
        model_id=model_id,
        target_layers=72,
        rank=16,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    assert mos_70b.num_layers == 72
    assert len(mos_70b.model.model.layers) == 72
    # Verifica que a projeção down_proj é MixtureOfScarsLinear
    assert isinstance(mos_70b.model.model.layers[0].mlp.down_proj, MixtureOfScarsLinear)

    # Verifica que camadas compartilham a base universal (zero duplicação de tensores)
    base_l0 = mos_70b.model.model.layers[0].mlp.down_proj.base
    base_l30 = mos_70b.model.model.layers[30].mlp.down_proj.base
    assert base_l0.weight.data_ptr() == base_l30.weight.data_ptr()

    # Gera texto através de todas as 72 camadas
    prompt = "A teoria da relatividade geral formulada por Albert Einstein estabelece que"
    output_text, metrics = mos_70b.generate(
        prompt=prompt,
        max_new_tokens=15,
        window_size=32,
    )

    assert len(output_text) > 0
    assert metrics["num_layers"] == 72
    assert metrics["generated_tokens"] == 15
    assert metrics["kv_cache_mb"] < 10.0  # Capped at O(1)
