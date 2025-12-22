import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from intent_ir.ir_types import IntentFunction
from intent_ir.printer_mlir_like import print_mlir_like


def _base_intent_json():
    return {
        "name": "gemm",
        "tensors": {
            "B": {"dtype": "f16", "shape": ["K", "N"], "layout": "row_major"},
            "A": {"dtype": "f16", "shape": ["M", "K"], "layout": "row_major"},
            "bias": {"dtype": "f16", "shape": ["N"], "layout": "row_major"},
            "C": {"dtype": "f16", "shape": ["M", "N"], "layout": "row_major"},
        },
        "ops": [
            {"op": "matmul", "inputs": ["A", "B"], "output": "Y", "attrs": {"accum_dtype": "f32"}},
            {
                "op": "broadcast_in_dim",
                "inputs": ["bias"],
                "output": "bias2d",
                "attrs": {"out_shape": ["M", "N"], "broadcast_dims": [1]},
            },
            {"op": "add", "inputs": ["Y", "bias2d"], "output": "Z"},
            {"op": "relu", "inputs": ["Z"], "output": "C"},
        ],
        "outputs": ["C"],
        "parallel_axes": ["M", "N"],
        "axis_roles": {"M": "spatial", "N": "spatial", "K": "reduction"},
        "schedule": {
            "tile_m": "BLOCK_M",
            "tile_n": "BLOCK_N",
            "tile_k": "BLOCK_K",
            "vec_width": "VEC",
            "axis_bindings": {"tile_m": "M", "tile_n": "N", "tile_k": "K"},
            "vec_axis": "N",
            "parallel_axes": ["M", "N"],
            "memory_hint": {"residency": {"A": "local", "B": "local"}},
        },
    }


def test_print_gemm_matches_key_tokens():
    intent = IntentFunction.from_json_dict(_base_intent_json())
    txt = print_mlir_like(intent)
    assert "func @gemm" in txt
    assert "intent.matmul" in txt
    assert "attach_schedule %C" in txt
    assert "return %C" in txt


def test_deterministic_ordering_inputs_sorted():
    intent = IntentFunction.from_json_dict(_base_intent_json())
    txt = print_mlir_like(intent)
    pos_a = txt.index("A: tensor")
    pos_b = txt.index("B: tensor")
    pos_bias = txt.index("bias: tensor")
    assert pos_a < pos_b < pos_bias


def test_elemwise_prints_as_intent_elemwise():
    intent = IntentFunction.from_json_dict(_base_intent_json())
    txt = print_mlir_like(intent)
    assert 'intent.elemwise("add"' in txt
    assert 'intent.elemwise("relu"' in txt


def test_symbolic_schedule_prints_sym():
    intent = IntentFunction.from_json_dict(_base_intent_json())
    txt = print_mlir_like(intent)
    assert 'tile_m = sym("BLOCK_M")' in txt


def test_int_schedule_prints_int():
    js = _base_intent_json()
    js["schedule"]["tile_m"] = 16
    intent = IntentFunction.from_json_dict(js)
    txt = print_mlir_like(intent)
    assert "tile_m = 16" in txt


def test_memory_hint_printing():
    intent = IntentFunction.from_json_dict(_base_intent_json())
    txt = print_mlir_like(intent)
    assert 'memory_hint = {residency={A="local", B="local"}}' in txt
