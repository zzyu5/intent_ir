"""
Task6: SPMD + RVV backend helpers (tiling search + cost model + codegen).
"""

from .tiling_search import SPMDProfile, TileChoice, choose_tiles  # noqa: F401
from .hardware_profile import RVVHardwareProfile, load_profile_from_json  # noqa: F401
from .cost_model import GEMMCostModel, CostEstimate  # noqa: F401
from .codegen_c import generate_c  # noqa: F401
from .codegen_rvv import generate_rvv  # noqa: F401

__all__ = [
    "SPMDProfile",
    "TileChoice",
    "choose_tiles",
    "RVVHardwareProfile",
    "load_profile_from_json",
    "GEMMCostModel",
    "CostEstimate",
    "generate_c",
    "generate_rvv",
]
