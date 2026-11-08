"""SCache (Gradient Scar Memory): Universal O(1) Fast-Weights Associative Memory.

Substrato de memória de tamanho constante O(1) guiada por gradientes,
imunidade de gating por manifold e projeção de conflito ortogonal (PCGrad).
"""

from scar_cache.core import GradientScarMemory, SCache
from scar_cache.modules import ScarGatedLayer, ScarLinear
from scar_cache.projections import hebbian_outer_product, project_conflicting_gradient

__version__ = "0.1.0"

__all__ = [
    "SCache",
    "GradientScarMemory",
    "ScarLinear",
    "ScarGatedLayer",
    "project_conflicting_gradient",
    "hebbian_outer_product",
    "__version__",
]
