"""SCache (Gradient Scar Memory): Universal O(1) Fast-Weights Associative Memory.

Substrato de memória de tamanho constante O(1) guiada por gradientes,
imunidade de gating por manifold, projeção de conflito ortogonal (PCGrad)
e arquitetura tensorial Multi-Slot com Regra Delta para substituição do KV Cache.
"""

from scar_cache.attention import SlotScarAttention
from scar_cache.core import GradientScarMemory, SCache
from scar_cache.hf_cache import SCacheDynamicCache, SCacheDynamicLayer
from scar_cache.modules import ScarGatedLayer, ScarLinear, SlotScarLinear
from scar_cache.mos import MixtureOfScarsLinear, ScarExpert
from scar_cache.multi_slot import MultiSlotSCache
from scar_cache.projections import (
    delta_rule_update,
    hebbian_outer_product,
    project_conflicting_gradient,
    project_conflicting_gradients_per_slot,
)

__version__ = "0.3.0"

__all__ = [
    "SCache",
    "GradientScarMemory",
    "MultiSlotSCache",
    "SlotScarAttention",
    "SCacheDynamicCache",
    "SCacheDynamicLayer",
    "ScarLinear",
    "ScarGatedLayer",
    "SlotScarLinear",
    "ScarExpert",
    "MixtureOfScarsLinear",
    "project_conflicting_gradient",
    "project_conflicting_gradients_per_slot",
    "delta_rule_update",
    "hebbian_outer_product",
    "__version__",
]
