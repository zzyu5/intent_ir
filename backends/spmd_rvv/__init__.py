"""
Task6: SPMD + RVV backend helpers (analysis + strict MLIR pipeline).
"""

from .analysis.tiling_search import SPMDProfile, TileChoice, choose_tiles  # noqa: F401
from .analysis.hardware_profile import RVVHardwareProfile, load_profile_from_json  # noqa: F401
from .analysis.cost_model import GEMMCostModel, CostEstimate  # noqa: F401
from .opset import SPMD_RVV_SUPPORTED_OPS  # noqa: F401
from .pipeline import RvvPipelineResult, RvvPipelineStage, run_rvv_pipeline  # noqa: F401

__all__ = [
    "SPMDProfile",
    "TileChoice",
    "choose_tiles",
    "RVVHardwareProfile",
    "load_profile_from_json",
    "GEMMCostModel",
    "CostEstimate",
    "RvvPipelineStage",
    "RvvPipelineResult",
    "run_rvv_pipeline",
    "SPMD_RVV_SUPPORTED_OPS",
]
