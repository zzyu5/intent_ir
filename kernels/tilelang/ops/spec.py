from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

from intent_ir.ir import IntentFunction
from verify.gen_cases import TestCase


@dataclass
class TileLangKernelSpec:
    """
    A TileLang kernel test spec:
      - `prim_func`: real TileLang/TVM PrimFunc (for evidence extraction)
      - `runner`: numpy reference implementation (Task5 ref side)
      - `intent_builder`: deterministic IntentIR builder (Task5 pred side)
    """

    name: str
    prim_func: Any
    arg_names: List[str]
    canonical_shapes: Dict[str, int]
    vary_axes: List[str]
    runner: Callable[[TestCase], Dict[str, np.ndarray]]
    intent_builder: Callable[[], IntentFunction]
    exclude_axes: List[str] | None = None
    constexpr_names: List[str] | None = None


__all__ = ["TileLangKernelSpec"]

