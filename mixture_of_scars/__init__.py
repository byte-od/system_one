"""Mixture of Scars (MoS) & SCache: Radical Static Weight Compression & O(1) Constant Memory KV-Cache Replacement for LLMs."""

from scar_cache import (
    GradientScarMemory,
    MixtureOfScarsLinear,
    MultiSlotSCache,
    SCache,
    SCacheDynamicCache,
    SCacheDynamicLayer,
    ScarExpert,
    ScarGatedLayer,
    ScarLinear,
    SlotScarAttention,
    SlotScarLinear,
    __version__,
    delta_rule_update,
    hebbian_outer_product,
    project_conflicting_gradient,
    project_conflicting_gradients_per_slot,
)

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
