"""
Task6: SPMD + RVV backend helpers (tiling search + cost model + codegen).
"""

from .analysis.tiling_search import SPMDProfile, TileChoice, choose_tiles  # noqa: F401
from .analysis.hardware_profile import RVVHardwareProfile, load_profile_from_json  # noqa: F401
from .analysis.cost_model import GEMMCostModel, CostEstimate  # noqa: F401
from .codegen.matmul_c import generate_c  # noqa: F401
from .codegen.intentir_to_c import lower_intent_to_c_with_files  # noqa: F401

__all__ = [
    "SPMDProfile",
    "TileChoice",
    "choose_tiles",
    "RVVHardwareProfile",
    "load_profile_from_json",
    "GEMMCostModel",
    "CostEstimate",
    "generate_c",
    "lower_intent_to_c_with_files",
]
