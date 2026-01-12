from __future__ import annotations

from pathlib import Path


TRANSPOSE2D_CU_PATH = Path(__file__).with_name("transpose2d.cu")


transpose2d_io = {
    "arg_names": ["inp", "out", "M", "N"],
    "tensors": {
        "inp": {"dtype": "f32", "rank": 2, "shape": ["M", "N"]},
        "out": {"dtype": "f32", "rank": 2, "shape": ["N", "M"]},
    },
    "scalars": {"M": "i32", "N": "i32"},
    "kernel_name": "transpose2d",
}


__all__ = ["TRANSPOSE2D_CU_PATH", "transpose2d_io"]
