from __future__ import annotations

from pathlib import Path


VEC_ADD_CU_PATH = Path(__file__).with_name("vec_add.cu")


vec_add_io = {
    # Matches kernel parameter order.
    "arg_names": ["A", "B", "C", "N"],
    # Pointer args (tensor name -> dtype, logical rank)
    "tensors": {
        "A": {"dtype": "f32", "rank": 1, "shape": ["N"]},
        "B": {"dtype": "f32", "rank": 1, "shape": ["N"]},
        "C": {"dtype": "f32", "rank": 1, "shape": ["N"]},
    },
    # Scalar args (name -> dtype)
    "scalars": {"N": "i32"},
    "kernel_name": "vec_add",
}


__all__ = ["VEC_ADD_CU_PATH", "vec_add_io"]
