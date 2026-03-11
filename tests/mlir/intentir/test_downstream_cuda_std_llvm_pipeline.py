from __future__ import annotations

from pathlib import Path

import pytest

from intent_ir.ir import IntentFunction
from intent_ir.mlir import detect_mlir_toolchain, run_pipeline, to_mlir


def _intent(payload: dict) -> IntentFunction:
    return IntentFunction.from_json_dict(payload)


CASES = [
    (
        "add2d",
        _intent(
            {
                "name": "add2d",
                "tensors": {
                    "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "z": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "add", "inputs": ["x", "y"], "output": "z", "attrs": {}}],
                "outputs": ["z"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "abs2d",
        _intent(
            {
                "name": "abs2d",
                "tensors": {
                    "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "Out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "abs", "inputs": ["A"], "output": "Out", "attrs": {}}],
                "outputs": ["Out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "floor2d",
        _intent(
            {
                "name": "floor2d",
                "tensors": {
                    "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "floor", "inputs": ["inp"], "output": "out", "attrs": {}}],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "clamp2d",
        _intent(
            {
                "name": "clamp2d",
                "tensors": {
                    "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "mini": {"dtype": "f32", "shape": [], "layout": "row_major"},
                    "maxi": {"dtype": "f32", "shape": [], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "x_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "clamped_min": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [
                    {"op": "cast", "inputs": ["x"], "output": "x_f32", "attrs": {"to": "f32"}},
                    {"op": "max", "inputs": ["mini", "x_f32"], "output": "clamped_min", "attrs": {}},
                    {"op": "min", "inputs": ["clamped_min", "maxi"], "output": "out", "attrs": {}},
                ],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "add_bias2d",
        _intent(
            {
                "name": "add_bias2d",
                "tensors": {
                    "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "bias": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "add", "inputs": ["inp", "bias"], "output": "out", "attrs": {}}],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "sigmoid2d",
        _intent(
            {
                "name": "sigmoid2d",
                "tensors": {
                    "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "output": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [
                    {"op": "const", "inputs": [], "output": "one_const", "attrs": {"value": 1}},
                    {"op": "const", "inputs": [], "output": "neg_one_const", "attrs": {"value": -1}},
                    {"op": "mul", "inputs": ["x", "neg_one_const"], "output": "neg_x", "attrs": {}},
                    {"op": "exp", "inputs": ["neg_x"], "output": "exp_neg_x", "attrs": {}},
                    {"op": "add", "inputs": ["one_const", "exp_neg_x"], "output": "denominator", "attrs": {}},
                    {"op": "div", "inputs": ["one_const", "denominator"], "output": "output", "attrs": {}},
                ],
                "outputs": ["output"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "tanh2d",
        _intent(
            {
                "name": "tanh2d",
                "tensors": {
                    "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [
                    {"op": "const", "inputs": [], "output": "one_const", "attrs": {"value": 1}},
                    {"op": "const", "inputs": [], "output": "two_const", "attrs": {"value": 2}},
                    {"op": "mul", "inputs": ["two_const", "x"], "output": "two_x", "attrs": {}},
                    {"op": "exp", "inputs": ["two_x"], "output": "exp_two_x", "attrs": {}},
                    {"op": "sub", "inputs": ["exp_two_x", "one_const"], "output": "numer", "attrs": {}},
                    {"op": "add", "inputs": ["exp_two_x", "one_const"], "output": "denom", "attrs": {}},
                    {"op": "div", "inputs": ["numer", "denom"], "output": "out", "attrs": {}},
                ],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "sqrt2d",
        _intent(
            {
                "name": "sqrt2d",
                "tensors": {
                    "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "sqrt", "inputs": ["A"], "output": "out", "attrs": {}}],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "log2d",
        _intent(
            {
                "name": "log2d",
                "tensors": {
                    "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "log", "inputs": ["inp"], "output": "out", "attrs": {}}],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "rsqrt2d",
        _intent(
            {
                "name": "rsqrt2d",
                "tensors": {
                    "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "A_f32": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [
                    {"op": "cast", "inputs": ["A"], "output": "A_f32", "attrs": {"to": "f32"}},
                    {"op": "rsqrt", "inputs": ["A_f32"], "output": "out", "attrs": {}},
                ],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "erf2d",
        _intent(
            {
                "name": "erf2d",
                "tensors": {
                    "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "erf", "inputs": ["x"], "output": "out", "attrs": {}}],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "exp22d",
        _intent(
            {
                "name": "exp22d",
                "tensors": {
                    "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [
                    {"op": "const", "inputs": [], "output": "ln2", "attrs": {"value": 0.6931471805599453}},
                    {"op": "mul", "inputs": ["A", "ln2"], "output": "scaled", "attrs": {}},
                    {"op": "exp", "inputs": ["scaled"], "output": "out", "attrs": {}},
                ],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "eq2d",
        _intent(
            {
                "name": "eq2d",
                "tensors": {
                    "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "bool", "shape": ["M", "N"], "layout": "row_major"},
                },
                "ops": [{"op": "eq", "inputs": ["x", "y"], "output": "out", "attrs": {}}],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "ai_bench_layernorm",
        _intent(
            {
                "name": "ai_bench_layernorm",
                "tensors": {
                    "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "W": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                    "B": {"dtype": "f32", "shape": ["N"], "layout": "row_major"},
                    "Y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "Mean": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                    "Rstd": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                },
                "ops": [
                    {
                        "op": "const",
                        "inputs": [],
                        "output": "eps",
                        "attrs": {"value": 1e-05, "dtype": "f32"},
                    },
                    {
                        "op": "add",
                        "inputs": ["X", "eps"],
                        "output": "Y",
                        "attrs": {},
                    }
                ],
                "outputs": ["Y", "Mean", "Rstd"],
            }
        ),
        {"M": 4, "N": 8},
    ),
    (
        "grouped_row_sum2d",
        _intent(
            {
                "name": "grouped_row_sum2d",
                "tensors": {
                    "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                    "out": {"dtype": "f32", "shape": ["M", "G"], "layout": "row_major"},
                },
                "ops": [
                    {
                        "op": "reshape",
                        "inputs": ["inp"],
                        "output": "inp_reshaped",
                        "attrs": {"shape": ["M", "G", "GROUP_SIZE"]},
                    },
                    {
                        "op": "reduce_sum",
                        "inputs": ["inp_reshaped"],
                        "output": "out",
                        "attrs": {"dims": [2]},
                    },
                ],
                "outputs": ["out"],
            }
        ),
        {"M": 4, "G": 2, "N": 8, "GROUP_SIZE": 4},
    ),

    (
        "group_norm_kernel",
        _intent(
            {
                        "name": "group_norm_kernel",
                        "ops": [
                                    {
                                                "attrs": {
                                                            "dtype": "f32",
                                                            "value": 1e-05
                                                },
                                                "inputs": [],
                                                "op": "const",
                                                "output": "eps"
                                    },
                                    {
                                                "attrs": {
                                                            "value": "group_size * HW"
                                                },
                                                "inputs": [],
                                                "op": "const",
                                                "output": "num_elements"
                                    },
                                    {
                                                "attrs": {
                                                            "shape": [
                                                                        "N",
                                                                        "num_groups",
                                                                        "group_size",
                                                                        "HW"
                                                            ]
                                                },
                                                "inputs": [
                                                            "X"
                                                ],
                                                "op": "reshape",
                                                "output": "X_reshaped"
                                    },
                                    {
                                                "attrs": {
                                                            "dims": [
                                                                        2,
                                                                        3
                                                            ]
                                                },
                                                "inputs": [
                                                            "X_reshaped"
                                                ],
                                                "op": "reduce_sum",
                                                "output": "sum_X"
                                    },
                                    {
                                                "inputs": [
                                                            "sum_X",
                                                            "num_elements"
                                                ],
                                                "op": "div",
                                                "output": "mean"
                                    },
                                    {
                                                "attrs": {
                                                            "broadcast_dims": [
                                                                        0,
                                                                        1
                                                            ],
                                                            "out_shape": [
                                                                        "N",
                                                                        "num_groups",
                                                                        "group_size",
                                                                        "HW"
                                                            ]
                                                },
                                                "inputs": [
                                                            "mean"
                                                ],
                                                "op": "broadcast_in_dim",
                                                "output": "mean_bcast"
                                    },
                                    {
                                                "inputs": [
                                                            "X_reshaped",
                                                            "mean_bcast"
                                                ],
                                                "op": "sub",
                                                "output": "X_centered"
                                    },
                                    {
                                                "inputs": [
                                                            "X_centered",
                                                            "X_centered"
                                                ],
                                                "op": "mul",
                                                "output": "X_centered_sq"
                                    },
                                    {
                                                "attrs": {
                                                            "dims": [
                                                                        2,
                                                                        3
                                                            ]
                                                },
                                                "inputs": [
                                                            "X_centered_sq"
                                                ],
                                                "op": "reduce_sum",
                                                "output": "sum_sq"
                                    },
                                    {
                                                "inputs": [
                                                            "sum_sq",
                                                            "num_elements"
                                                ],
                                                "op": "div",
                                                "output": "var"
                                    },
                                    {
                                                "inputs": [
                                                            "var",
                                                            "eps"
                                                ],
                                                "op": "add",
                                                "output": "var_eps"
                                    },
                                    {
                                                "inputs": [
                                                            "var_eps"
                                                ],
                                                "op": "rsqrt",
                                                "output": "rstd"
                                    },
                                    {
                                                "attrs": {
                                                            "broadcast_dims": [
                                                                        0,
                                                                        1
                                                            ],
                                                            "out_shape": [
                                                                        "N",
                                                                        "num_groups",
                                                                        "group_size",
                                                                        "HW"
                                                            ]
                                                },
                                                "inputs": [
                                                            "rstd"
                                                ],
                                                "op": "broadcast_in_dim",
                                                "output": "rstd_bcast"
                                    },
                                    {
                                                "inputs": [
                                                            "X_centered",
                                                            "rstd_bcast"
                                                ],
                                                "op": "mul",
                                                "output": "X_hat"
                                    },
                                    {
                                                "attrs": {
                                                            "shape": [
                                                                        "num_groups",
                                                                        "group_size"
                                                            ]
                                                },
                                                "inputs": [
                                                            "W"
                                                ],
                                                "op": "reshape",
                                                "output": "W_reshaped"
                                    },
                                    {
                                                "attrs": {
                                                            "broadcast_dims": [
                                                                        1,
                                                                        2
                                                            ],
                                                            "out_shape": [
                                                                        "N",
                                                                        "num_groups",
                                                                        "group_size",
                                                                        "HW"
                                                            ]
                                                },
                                                "inputs": [
                                                            "W_reshaped"
                                                ],
                                                "op": "broadcast_in_dim",
                                                "output": "W_bcast"
                                    },
                                    {
                                                "inputs": [
                                                            "X_hat",
                                                            "W_bcast"
                                                ],
                                                "op": "mul",
                                                "output": "Y_scaled"
                                    },
                                    {
                                                "attrs": {
                                                            "shape": [
                                                                        "num_groups",
                                                                        "group_size"
                                                            ]
                                                },
                                                "inputs": [
                                                            "B"
                                                ],
                                                "op": "reshape",
                                                "output": "B_reshaped"
                                    },
                                    {
                                                "attrs": {
                                                            "broadcast_dims": [
                                                                        1,
                                                                        2
                                                            ],
                                                            "out_shape": [
                                                                        "N",
                                                                        "num_groups",
                                                                        "group_size",
                                                                        "HW"
                                                            ]
                                                },
                                                "inputs": [
                                                            "B_reshaped"
                                                ],
                                                "op": "broadcast_in_dim",
                                                "output": "B_bcast"
                                    },
                                    {
                                                "inputs": [
                                                            "Y_scaled",
                                                            "B_bcast"
                                                ],
                                                "op": "add",
                                                "output": "Y_reshaped"
                                    },
                                    {
                                                "attrs": {
                                                            "shape": [
                                                                        "N",
                                                                        "C",
                                                                        "HW"
                                                            ]
                                                },
                                                "inputs": [
                                                            "Y_reshaped"
                                                ],
                                                "op": "reshape",
                                                "output": "Y"
                                    },
                                    {
                                                "inputs": [
                                                            "mean"
                                                ],
                                                "op": "identity",
                                                "output": "Mean"
                                    },
                                    {
                                                "inputs": [
                                                            "rstd"
                                                ],
                                                "op": "identity",
                                                "output": "Rstd"
                                    }
                        ],
                        "outputs": [
                                    "Y",
                                    "Mean",
                                    "Rstd"
                        ],
                        "tensors": {
                                    "B": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "C"
                                                ]
                                    },
                                    "B_bcast": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups",
                                                            "group_size",
                                                            "HW"
                                                ]
                                    },
                                    "B_reshaped": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "num_groups",
                                                            "group_size"
                                                ]
                                    },
                                    "C": {
                                                "dtype": "i32",
                                                "layout": "row_major",
                                                "shape": []
                                    },
                                    "HW": {
                                                "dtype": "i32",
                                                "layout": "row_major",
                                                "shape": []
                                    },
                                    "Mean": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups"
                                                ]
                                    },
                                    "Rstd": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups"
                                                ]
                                    },
                                    "W": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "C"
                                                ]
                                    },
                                    "W_bcast": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups",
                                                            "group_size",
                                                            "HW"
                                                ]
                                    },
                                    "W_reshaped": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "num_groups",
                                                            "group_size"
                                                ]
                                    },
                                    "X": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "C",
                                                            "HW"
                                                ]
                                    },
                                    "X_centered": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups",
                                                            "group_size",
                                                            "HW"
                                                ]
                                    },
                                    "X_centered_sq": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups",
                                                            "group_size",
                                                            "HW"
                                                ]
                                    },
                                    "X_hat": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups",
                                                            "group_size",
                                                            "HW"
                                                ]
                                    },
                                    "X_reshaped": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups",
                                                            "group_size",
                                                            "HW"
                                                ]
                                    },
                                    "Y": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "C",
                                                            "HW"
                                                ]
                                    },
                                    "Y_reshaped": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups",
                                                            "group_size",
                                                            "HW"
                                                ]
                                    },
                                    "Y_scaled": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups",
                                                            "group_size",
                                                            "HW"
                                                ]
                                    },
                                    "eps": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": []
                                    },
                                    "group_size": {
                                                "dtype": "i32",
                                                "layout": "row_major",
                                                "shape": []
                                    },
                                    "mean": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups"
                                                ]
                                    },
                                    "mean_bcast": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups",
                                                            "group_size",
                                                            "HW"
                                                ]
                                    },
                                    "num_elements": {
                                                "dtype": "i32",
                                                "layout": "row_major",
                                                "shape": []
                                    },
                                    "num_groups": {
                                                "dtype": "i32",
                                                "layout": "row_major",
                                                "shape": []
                                    },
                                    "rstd": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups"
                                                ]
                                    },
                                    "rstd_bcast": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups",
                                                            "group_size",
                                                            "HW"
                                                ]
                                    },
                                    "sum_X": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups"
                                                ]
                                    },
                                    "sum_sq": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups"
                                                ]
                                    },
                                    "var": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups"
                                                ]
                                    },
                                    "var_eps": {
                                                "dtype": "f32",
                                                "layout": "row_major",
                                                "shape": [
                                                            "N",
                                                            "num_groups"
                                                ]
                                    }
                        }
            }
        ),
        {"N": 2, "C": 4, "HW": 4, "num_groups": 2},
    ),
]


@pytest.mark.parametrize("name,intent,bindings", CASES, ids=[c[0] for c in CASES])
def test_downstream_cuda_std_llvm_emits_nvptx_triple(
    name: str,
    intent: IntentFunction,
    bindings: dict[str, int],
    tmp_path: Path,
    monkeypatch,
) -> None:
    toolchain = detect_mlir_toolchain()
    assert bool(toolchain.get("ok")) is True
    for k in ("mlir-opt", "mlir-translate", "llvm-as", "opt"):
        assert bool(((toolchain.get("tools") or {}).get(k) or {}).get("available")) is True

    monkeypatch.setenv("INTENTIR_REAL_MLIR", "1")

    mod = to_mlir(intent)
    mod.meta = dict(mod.meta or {})
    mod.meta["shape_bindings"] = dict(bindings)

    out, trace = run_pipeline(
        mod,
        "downstream_cuda_std_llvm",
        backend="cuda",
        out_dir=tmp_path,
        fail_on_error=True,
    )
    assert bool(trace.get("ok")) is True

    text = str(out.module_text or "")
    assert "target triple = \"nvptx64-nvidia-cuda\"" in text
    assert f"@{name}" in text
    assert str((out.meta or {}).get("llvm_dialect_origin") or "") in {"", "mlir_translate"}
