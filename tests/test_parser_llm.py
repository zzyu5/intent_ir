import pytest

from intent_ir.parser import (
    CandidateIntent,
    LLMJsonParseError,
    merge_tensor_and_symbol_json,
    normalize_candidate_json,
    parse_candidate_json,
)


def _tensor_json():
    return {
        "kernel_type": "matmul_bias_relu",
        "tensors": {
            "A": {"dtype": "f16", "shape": ["M", "K"], "layout": "row_major"},
            "B": {"dtype": "f16", "shape": ["K", "N"], "layout": "row_major"},
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
        "parallel_axes": ["M", "N"],
    }


def _symbol_json():
    return {
        "problem_params": {"M": 128, "N": 256, "K": 64},
        "schedule_params": {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
    }


def test_parse_merged_json_ok():
    d = _tensor_json()
    d["outputs"] = ["C"]
    cand = parse_candidate_json(d)
    assert isinstance(cand, CandidateIntent)
    assert cand.intent.outputs == ["C"]


def test_parse_two_json_merge_ok():
    merged = merge_tensor_and_symbol_json(_tensor_json(), _symbol_json())
    merged["outputs"] = ["C"]
    cand = parse_candidate_json(merged)
    assert cand.problem_params["M"] == 128
    assert cand.schedule_params["BLOCK_M"] == 128


def test_missing_outputs_uses_last_op_output():
    d = _tensor_json()
    d["tensors"]["C"] = d["tensors"]["C"]
    cand = parse_candidate_json(d)
    assert cand.intent.outputs == ["C"]


def test_missing_outputs_but_last_not_in_tensors_raises():
    d = _tensor_json()
    d["ops"][-1]["output"] = "OOPS"
    with pytest.raises(LLMJsonParseError):
        parse_candidate_json(d)


def test_unknown_dtype_raises():
    d = _tensor_json()
    d["tensors"]["A"]["dtype"] = "f8"
    with pytest.raises(LLMJsonParseError):
        parse_candidate_json(d)


def test_op_inputs_reference_unknown_raises():
    d = _tensor_json()
    d["ops"][0]["inputs"] = ["X", "B"]
    with pytest.raises(LLMJsonParseError):
        parse_candidate_json(d)


def test_schedule_params_kept_in_candidateintent():
    merged = merge_tensor_and_symbol_json(_tensor_json(), _symbol_json())
    merged["outputs"] = ["C"]
    cand = parse_candidate_json(merged)
    assert cand.schedule_params["BLOCK_N"] == 128
