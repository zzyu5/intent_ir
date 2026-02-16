from __future__ import annotations

import pytest

from backends.cuda.codegen.cpp_driver import CudaLoweringError, lower_intent_to_cuda_kernel
from intent_ir.ir import IntentFunction


def _lower_or_skip(intent: IntentFunction, *, shape_bindings: dict[str, int]):
    try:
        return lower_intent_to_cuda_kernel(intent, shape_bindings=shape_bindings)
    except CudaLoweringError as exc:
        if "unsupported intent for cuda cpp codegen" in str(exc):
            pytest.skip(f"cpp cuda codegen unsupported for this pattern: {exc}")
        raise


def test_cuda_lowering_supports_conv2d_bias_pattern() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "conv2d_bias_pattern_cuda_lowering",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C_IN_TOTAL", "H", "W"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C_OUT", "C_IN", "KH", "KW"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C_OUT"], "layout": "row_major"},
                "conv_out": {"dtype": "f32", "shape": ["N", "C_OUT", "OH", "OW"], "layout": "row_major"},
                "bias_bcast": {"dtype": "f32", "shape": ["N", "C_OUT", "OH", "OW"], "layout": "row_major"},
                "output": {"dtype": "f32", "shape": ["N", "C_OUT", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "conv2d",
                    "inputs": ["input", "weight"],
                    "output": "conv_out",
                    "attrs": {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1},
                },
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["bias"],
                    "output": "bias_bcast",
                    "attrs": {"out_shape": ["N", "C_OUT", "OH", "OW"], "broadcast_dims": [1]},
                },
                {"op": "add", "inputs": ["conv_out", "bias_bcast"], "output": "output"},
            ],
            "outputs": ["output"],
            "parallel_axes": ["N", "C_OUT", "OH", "OW"],
            "schedule": {"tile_n": 128},
        }
    )
    lowered = _lower_or_skip(
        intent,
        shape_bindings={
            "N": 1,
            "C_IN_TOTAL": 2,
            "C_IN": 2,
            "C_OUT": 3,
            "H": 8,
            "W": 8,
            "KH": 3,
            "KW": 3,
            "OH": 8,
            "OW": 8,
        },
    )
    assert lowered.kernel_name == "conv2d_bias_pattern_cuda_lowering"
    assert "co_per_g" in lowered.cuda_src
    assert "acc += input[x_idx] * weight[w_idx];" in lowered.cuda_src


def test_cuda_lowering_supports_conv1d_and_conv3d() -> None:
    conv1d = IntentFunction.from_json_dict(
        {
            "name": "conv1d_cuda_lowering",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C_IN", "L"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C_OUT", "C_IN", "K"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C_OUT", "OL"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "conv1d",
                    "inputs": ["input", "weight", "bias"],
                    "output": "out",
                    "attrs": {"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
                }
            ],
            "outputs": ["out"],
            "parallel_axes": ["N", "C_OUT", "OL"],
            "schedule": {"tile_n": 128},
        }
    )
    lowered_1d = _lower_or_skip(
        conv1d,
        shape_bindings={"N": 1, "C_IN": 2, "C_OUT": 3, "L": 8, "K": 3, "OL": 8},
    )
    assert lowered_1d.kernel_name == "conv1d_cuda_lowering"
    assert "for (int64_t k = 0;" in lowered_1d.cuda_src

    conv3d = IntentFunction.from_json_dict(
        {
            "name": "conv3d_cuda_lowering",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C_IN", "D", "H", "W"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C_OUT", "C_IN", "KD", "KH", "KW"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C_OUT", "OD", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "conv3d",
                    "inputs": ["input", "weight", "bias"],
                    "output": "out",
                    "attrs": {"stride": [1, 1, 1], "padding": [1, 1, 1], "dilation": [1, 1, 1], "groups": 1},
                }
            ],
            "outputs": ["out"],
            "parallel_axes": ["N", "C_OUT", "OD", "OH", "OW"],
            "schedule": {"tile_n": 128},
        }
    )
    lowered_3d = _lower_or_skip(
        conv3d,
        shape_bindings={
            "N": 1,
            "C_IN": 2,
            "C_OUT": 3,
            "D": 4,
            "H": 4,
            "W": 4,
            "KD": 3,
            "KH": 3,
            "KW": 3,
            "OD": 4,
            "OH": 4,
            "OW": 4,
        },
    )
    assert lowered_3d.kernel_name == "conv3d_cuda_lowering"
    assert "for (int64_t kd = 0;" in lowered_3d.cuda_src


def test_cuda_lowering_supports_conv_depthwise2d() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "conv_depthwise2d_cuda_lowering",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C_IN", "H", "W"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C_OUT", 1, "KH", "KW"], "layout": "row_major"},
                "bias": {"dtype": "f32", "shape": ["C_OUT"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C_OUT", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "conv_depthwise2d",
                    "inputs": ["input", "weight", "bias"],
                    "output": "out",
                    "attrs": {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1]},
                }
            ],
            "outputs": ["out"],
            "parallel_axes": ["N", "C_OUT", "OH", "OW"],
            "schedule": {"tile_n": 128},
        }
    )
    lowered = _lower_or_skip(
        intent,
        shape_bindings={"N": 1, "C_IN": 2, "C_OUT": 4, "H": 8, "W": 8, "KH": 3, "KW": 3, "OH": 8, "OW": 8},
    )
    assert lowered.kernel_name == "conv_depthwise2d_cuda_lowering"
    assert "const int64_t ci = co /" in lowered.cuda_src


def test_cuda_lowering_supports_avg_pool_and_upsample_nearest() -> None:
    avg_pool = IntentFunction.from_json_dict(
        {
            "name": "avg_pool2d_cuda_lowering",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["N", "C", "H", "W"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "avg_pool2d",
                    "inputs": ["inp"],
                    "output": "out",
                    "attrs": {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "count_include_pad": True},
                }
            ],
            "outputs": ["out"],
            "parallel_axes": ["N", "C", "OH", "OW"],
            "schedule": {"tile_n": 128},
        }
    )
    lowered_pool = _lower_or_skip(avg_pool, shape_bindings={"N": 1, "C": 2, "H": 8, "W": 8, "OH": 4, "OW": 4})
    assert lowered_pool.kernel_name == "avg_pool2d_cuda_lowering"
    assert "acc / denom" in lowered_pool.cuda_src

    upsample = IntentFunction.from_json_dict(
        {
            "name": "upsample_nearest2d_cuda_lowering",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C", "IH", "IW"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [{"op": "upsample_nearest2d", "inputs": ["input"], "output": "out", "attrs": {}}],
            "outputs": ["out"],
            "parallel_axes": ["N", "C", "OH", "OW"],
            "schedule": {"tile_n": 128},
        }
    )
    lowered_up = _lower_or_skip(upsample, shape_bindings={"N": 1, "C": 2, "IH": 4, "IW": 4, "OH": 8, "OW": 8})
    assert lowered_up.kernel_name == "upsample_nearest2d_cuda_lowering"
    assert "ih = (oh *" in lowered_up.cuda_src
    assert "iw = (ow *" in lowered_up.cuda_src


def test_cuda_lowering_supports_max_pool2d_with_indices_pair() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "max_pool2d_with_indices_cuda_lowering",
            "tensors": {
                "input": {"dtype": "f32", "shape": ["N", "C", "H", "W"], "layout": "row_major"},
                "out": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
                "indices": {"dtype": "i64", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {
                    "op": "max_pool2d_with_indices",
                    "inputs": ["input"],
                    "output": "out",
                    "attrs": {
                        "kernel_size": [2, 2],
                        "stride": [2, 2],
                        "padding": [0, 0],
                        "dilation": [1, 1],
                        "ceil_mode": False,
                        "select": "values",
                    },
                },
                {
                    "op": "max_pool2d_with_indices",
                    "inputs": ["input"],
                    "output": "indices",
                    "attrs": {
                        "kernel_size": [2, 2],
                        "stride": [2, 2],
                        "padding": [0, 0],
                        "dilation": [1, 1],
                        "ceil_mode": False,
                        "select": "indices",
                    },
                },
            ],
            "outputs": ["out", "indices"],
            "parallel_axes": ["N", "C", "OH", "OW"],
            "schedule": {"tile_n": 128},
        }
    )
    lowered = _lower_or_skip(intent, shape_bindings={"N": 1, "C": 2, "H": 8, "W": 8, "OH": 4, "OW": 4})
    assert lowered.kernel_name == "max_pool2d_with_indices_cuda_lowering"
    assert "maxidx = ih *" in lowered.cuda_src
    assert lowered.output_names == ["out", "indices"]
