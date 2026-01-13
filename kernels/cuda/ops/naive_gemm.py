from __future__ import annotations

from pathlib import Path


NAIVE_GEMM_CU_PATH = Path(__file__).with_name("naive_gemm.cu")


naive_gemm_io = {
    "arg_names": ["A", "B", "C", "M", "N", "K"],
    "tensors": {
        "A": {"dtype": "f32", "rank": 2, "shape": ["M", "K"]},
        "B": {"dtype": "f32", "rank": 2, "shape": ["K", "N"]},
        "C": {"dtype": "f32", "rank": 2, "shape": ["M", "N"]},
    },
    "scalars": {"M": "i32", "N": "i32", "K": "i32"},
    "kernel_name": "naive_gemm",
}


__all__ = ["NAIVE_GEMM_CU_PATH", "naive_gemm_io"]

