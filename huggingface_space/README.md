---
title: Mixture of Scars & SCache O(1) Red-Team Lab
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: true
license: mit
short_description: Live O(1) Widrow-Hoff KV Cache + MoS ASVD INT4 Red-Team Lab
tags:
  - kv-cache
  - fast-weights
  - widrow-hoff
  - quantization
  - transformers
  - pytorch
---

# Mixture of Scars (`MoS`) & `SCache` $\mathcal{O}(1)$ — Live Adversarial Red-Team Laboratory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24143839.svg)](https://doi.org/10.5281/zenodo.24143839)
[![GitHub Release](https://img.shields.io/badge/GitHub-v1.0.0-181717.svg)](https://github.com/byte-od/mixture-of-scars/releases/tag/v1.0.0)

Interactive verification laboratory for **[Mixture of Scars (MoS) & SCache $\mathcal{O}(1)$](https://doi.org/10.5281/zenodo.24143839)** by **Julio Cesar Reis Filho (*Byte-OD Research*)**.

## What You Can Test Live in This Space
1. **Live Pretrained LLM Evicted-Passcode & Byte-Pointer Audit (`Qwen/Qwen2.5-0.5B-Instruct`):**
   - Buries your custom passcode near token ~20, floods the context past `window_size=64` so the passcode is **physically evicted ~470 tokens prior to query**, walks `tensor.untyped_storage().data_ptr()` recursively to verify **`0.00 KB` hidden byte discrepancy**, and demonstrates **`22.58x` higher exact target probability** than **StreamingLLM** with zero fine-tuning.
2. **Widrow-Hoff Delta vs. Additive Hebbian Anti-Ghosting Arena (`34.8 KB` Canonical Buffer):**
   - Mutates state variables repeatedly on the same key alongside orthogonal distractors and `500x` antipodal DoS noise to demonstrate exact orthogonal state overwrite (`cos = 1.0000`) vs. additive Hebbian ghosting.
3. **Live 24-Layer `MoS` (`INT4` + `ASVD` Scar Experts) Outlier & Causality Audit:**
   - Converts all `72` SwiGLU MLP projections of `Qwen/Qwen2.5-0.5B-Instruct` into `INT4 + ASVD Scar Experts` calibrated on a single 24-token generic English sentence and tests unseen Python, JSON, Quantum Physics, or your own custom prompt (`>3x` lower logit KL error + `-92%+` 8-specialist multi-tenant RAM).

## Quick Drop-in Usage (`transformers`)
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from scar_cache.hf_cache import SCacheDynamicCache

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

cache = SCacheDynamicCache(
    window_size=64,
    num_slots=24,
    num_sink_tokens=4,
    project_slots=True,
)
```

## Official Citation
```bibtex
@techreport{reisfilho2026mos_scache,
  author      = {Reis Filho, Julio Cesar},
  title       = {Mixture of Scars (MoS) and SCache O(1): Breaking the Dual Memory Wall of Modern Transformers via Activation-Aware Low-Rank Scar Experts and Widrow-Hoff Associative Memory},
  institution = {Byte-OD Research},
  year        = {2026},
  month       = nov,
  doi         = {10.5281/zenodo.24143839},
  url         = {https://doi.org/10.5281/zenodo.24143839}
}
```
