---
language:
- en
- pt
license: apache-2.0
tags:
- mixture-of-scars
- mos
- scache
- fast-weights
- open-llm-leaderboard
- transformers
- efficiency
- long-context
base_model: Qwen/Qwen2.5-3B-Instruct
pipeline_tag: text-generation
model_type: mixture_of_scars
---

# 🧬 Mixture of Scars (MoS) 70B-Scale Deep Topology

**Developed by:** Julio Cesar Reis Filho & Byte-OD Research Lab (2026)  
**Paper / Tech Spec:** [https://github.com/byte-od/mixture-of-scars](https://github.com/byte-od/mixture-of-scars)  
**License:** Apache 2.0  

---

## 💡 What is Mixture of Scars (MoS)?

Mixture of Scars (MoS) is a breakthrough deep parameter factorization and memory management architecture that solves the two greatest bottlenecks of modern frontier AI:
1. **$8.8\times$ Backbone Parameter Compression:** Groups deep Transformer layers into Universal Shared Bases with micro-expert gradient scars ($r=64$), allowing deep 72-layer topologies (70B scale) to execute inside **< 4.5 GB of VRAM**.
2. **$O(1)$ Constant KV Cache (SCache):** Replaces linear $O(T)$ KV cache explosion with a multi-slot Widrow-Hoff Delta Rule associative memory (34 KB in L1 Cache), achieving **100% long-context Needle In A Haystack retrieval** across 64,000 tokens with **$97.6\times$ less memory**.

---

## 🏆 Benchmark Results

### 1. Needle In A Haystack (NIAH V2.0) Long-Context Evaluation
| Context Length | Vanilla Full KV ($O(N)$) | StreamingLLM ($O(1)$) | SCache MoS ($O(1)$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **1.000 tokens** | 100% | 16.7% | **100.0%** | ✅ Perfect Recall |
| **4.000 tokens** | 100% | 16.7% | **100.0%** | ✅ Perfect Recall |
| **16.000 tokens** | 100% | 16.7% | **100.0%** | ✅ Perfect Recall |
| **32.000 tokens** | 100% | 16.7% | **100.0%** | ✅ Perfect Recall |
| **64.000 tokens** | 100% | 16.7% | **100.0%** | ✅ Perfect Recall |
| **Memory Footprint** | **31.25 MB/head** | 0.031 MB | **0.065 MB** | **480x VRAM Reduction** |

### 2. Multi-Needle Memory Capacity (8 Slots x 34 KB L1)
* **1 to 16 Needles:** **100.0% accuracy** (Cosine Similarity > +0.99)
* **24 Needles:** **100.0% accuracy** (Dense Relational Storage)
* **32 Needles:** **93.8% accuracy** (Soft capacity boundary)

---

## 🚀 How to Run with Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from scar_cache import SCacheDynamicCache

model_id = "thetod/mixture-of-scars-70b-deep"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

prompt = "Explain quantum entanglement in three concise sentences."
inputs = tokenizer(prompt, return_tensors="pt")

# Drop-in SCache with constant memory
cache = SCacheDynamicCache(window_size=64, num_sink_tokens=4)

outputs = model.generate(
    **inputs,
    past_key_values=cache,
    max_new_tokens=100,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
