from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType
from verify.gen_cases import TestCase


GEMM_TILELANG_DSL = json.dumps(
    {
        "schema_version": "tilelang_dsl_v0.1",
        "kernel": "gemm",
        "anchors": {"kernel_kind_hint": "matmul", "has_dot": True, "has_reduce": False, "has_atomic": False},
        "accesses": [
            {
                "kind": "load",
                "tensor": "A",
                "dtype": "f32",
                "rank": 2,
                "index_exprs": [{"terms": {"pid0": 1}, "const": 0}, {"terms": {"r0": 1}, "const": 0}],
                "predicate": {"clauses": ["0 <= pid0", "pid0 < M", "0 <= r0", "r0 < K"]},
            },
            {
                "kind": "load",
                "tensor": "B",
                "dtype": "f32",
                "rank": 2,
                "index_exprs": [{"terms": {"r0": 1}, "const": 0}, {"terms": {"pid1": 1}, "const": 0}],
                "predicate": {"clauses": ["0 <= pid1", "pid1 < N", "0 <= r0", "r0 < K"]},
            },
            {
                "kind": "store",
                "tensor": "C",
                "dtype": "f32",
                "rank": 2,
                "index_exprs": [{"terms": {"pid0": 1}, "const": 0}, {"terms": {"pid1": 1}, "const": 0}],
                "predicate": {"clauses": ["0 <= pid0", "pid0 < M", "0 <= pid1", "pid1 < N"]},
            },
        ],
    },
    indent=2,
    sort_keys=True,
)


def gemm_reference(case: TestCase) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(int(case.seed))
    M = int(case.shapes["M"])
    N = int(case.shapes["N"])
    K = int(case.shapes["K"])
    A = rng.standard_normal((M, K), dtype=np.float32)
    B = rng.standard_normal((K, N), dtype=np.float32)
    C = A @ B
    return {"A": A, "B": B, "C": C}


def gemm_intent() -> IntentFunction:
    rm = TensorLayout(kind="row_major", params={})
    tensors = {
        "A": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "K")], layout=rm),
        "B": TensorType(dtype="f32", shape=[Dim("sym", "K"), Dim("sym", "N")], layout=rm),
        "C": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
    }
    ops = [Op(op="matmul", inputs=["A", "B"], output="C", attrs={})]
    schedule = ScheduleSketch(tile_m=8, tile_n=8, tile_k=16, vec_width=1, pipeline_depth=1)
    return IntentFunction(
        name="tilelang_gemm",
        tensors=tensors,
        ops=ops,
        outputs=["C"],
        schedule=schedule,
        axis_roles={"M": "spatial", "N": "spatial", "K": "reduction"},
    )


@dataclass
class TileLangKernelSpec:
    name: str
    source_text: str
    arg_names: List[str]
    canonical_shapes: Dict[str, int]
    vary_axes: List[str]
    runner: Callable[[TestCase], Dict[str, np.ndarray]]
    intent_builder: Callable[[], IntentFunction]
    exclude_axes: List[str] | None = None
    constexpr_names: List[str] | None = None


def gemm_spec() -> TileLangKernelSpec:
    return TileLangKernelSpec(
        name="tilelang_gemm",
        source_text=GEMM_TILELANG_DSL,
        arg_names=["A", "B", "C", "M", "N", "K"],
        canonical_shapes={"M": 4, "N": 8, "K": 16},
        vary_axes=["M", "N", "K"],
        runner=gemm_reference,
        intent_builder=gemm_intent,
        exclude_axes=[],
        constexpr_names=[],
    )


__all__ = ["TileLangKernelSpec", "GEMM_TILELANG_DSL", "gemm_reference", "gemm_intent", "gemm_spec"]

