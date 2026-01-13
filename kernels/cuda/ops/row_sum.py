from __future__ import annotations

from pathlib import Path


ROW_SUM_CU_PATH = Path(__file__).with_name("row_sum.cu")


row_sum_io = {
    "arg_names": ["inp", "out", "M", "N"],
    "tensors": {
        "inp": {"dtype": "f32", "rank": 2, "shape": ["M", "N"]},
        "out": {"dtype": "f32", "rank": 1, "shape": ["M"]},
    },
    "scalars": {"M": "i32", "N": "i32"},
    "kernel_name": "row_sum",
}


__all__ = ["ROW_SUM_CU_PATH", "row_sum_io"]

