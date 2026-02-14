import pytest

from intent_ir.ir import (
    IntentFunction,
    IntentIRValidationError,
)


def _minimal_intent_json():
    return {
        "name": "gemm_bias_relu",
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
            "memory_hint": {"residency": {"A": "local", "B": "local", "C": "local"}},
        },
    }


def test_roundtrip_minimal_gemm():
    src = _minimal_intent_json()
    intent = IntentFunction.from_json_dict(src)
    out = intent.to_json_dict()
    # Key structural fields should survive round-trip
    assert out["tensors"] == src["tensors"]
    assert out["outputs"] == src["outputs"]
    assert out["parallel_axes"] == src["parallel_axes"]
    assert out["axis_roles"] == src["axis_roles"]
    assert out["schedule"]["axis_bindings"] == src["schedule"]["axis_bindings"]


def test_invalid_dtype_rejected():
    bad = _minimal_intent_json()
    bad["tensors"]["A"]["dtype"] = "f8"
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)


def test_op_input_undefined_rejected():
    bad = _minimal_intent_json()
    # Undefined input with a close match and a common broadcast-intent suffix.
    bad["ops"][2]["inputs"] = ["Y", "bias_2d"]
    with pytest.raises(IntentIRValidationError) as exc:
        IntentFunction.from_json_dict(bad)
    msg = str(exc.value)
    assert "references undefined tensor" in msg
    assert "Did you mean" in msg
    assert "Available tensors at this point" in msg
    assert "broadcast_in_dim" in msg


def test_output_not_in_tensors_rejected():
    bad = _minimal_intent_json()
    bad["outputs"] = ["Y"]
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)


def test_schedule_field_types():
    bad = _minimal_intent_json()
    bad["schedule"]["tile_m"] = [16]
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)


def test_layout_blocked_params():
    bad = _minimal_intent_json()
    bad["tensors"]["A"]["layout"] = {"kind": "blocked"}
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)


def test_op_meta_roundtrip_and_validation():
    src = _minimal_intent_json()
    src["ops"][0]["meta"] = {
        "provider": "flaggems",
        "source_op": "add",
        "capability_state": "dual_pass",
    }
    intent = IntentFunction.from_json_dict(src)
    out = intent.to_json_dict()
    assert out["ops"][0]["meta"]["provider"] == "flaggems"
    assert out["ops"][0]["meta"]["source_op"] == "add"


def test_function_meta_provider_fields_optional_but_typed():
    bad = _minimal_intent_json()
    bad["meta"] = {"provider": "flaggems"}
    intent = IntentFunction.from_json_dict(bad)
    assert intent.meta.get("provider") == "flaggems"

    bad2 = _minimal_intent_json()
    bad2["meta"] = {"provider": "flaggems", "source_op": 1}
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad2)


def test_new_structure_ops_validate_success() -> None:
    src = {
        "name": "structure_ok",
        "tensors": {
            "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "Y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "Out": {"dtype": "f32", "shape": ["M2", "N"], "layout": "row_major"},
        },
        "ops": [
            {"op": "concat", "inputs": ["X", "Y"], "output": "Out", "attrs": {"axis": 0}},
            {"op": "tile", "inputs": ["X"], "output": "T", "attrs": {"repeats": [2, 1]}},
            {"op": "repeat_interleave", "inputs": ["X"], "output": "R", "attrs": {"repeats": 2, "axis": 1}},
            {"op": "pad", "inputs": ["X"], "output": "P", "attrs": {"pad_width": {"pairs": [[1, 0], [0, 1]]}}},
            {"op": "sort", "inputs": ["X"], "output": "S", "attrs": {"axis": 1, "descending": True, "stable": True}},
            {"op": "topk", "inputs": ["X"], "output": "K", "attrs": {"k": 1, "axis": 1}},
            {"op": "unique", "inputs": ["K"], "output": "U", "attrs": {"sorted": False}},
            {"op": "nonzero", "inputs": ["X"], "output": "NZ"},
            {"op": "count_nonzero", "inputs": ["X"], "output": "CNZ", "attrs": {"dims": [1]}},
            {"op": "diag", "inputs": ["X"], "output": "D", "attrs": {"diagonal": 0}},
            {"op": "diag_embed", "inputs": ["K"], "output": "DE", "attrs": {"offset": 0, "dim1": -2, "dim2": -1}},
            {"op": "angle", "inputs": ["X"], "output": "ANG"},
            {"op": "bitwise_not", "inputs": ["K"], "output": "BN"},
            {"op": "bitwise_and", "inputs": ["K", "K"], "output": "BA"},
            {"op": "bitwise_or", "inputs": ["K", "K"], "output": "BO"},
            {"op": "bitwise_left_shift", "inputs": ["K", "K"], "output": "BLS"},
            {"op": "bitwise_right_shift", "inputs": ["K", "K"], "output": "BRS"},
            {"op": "avg_pool2d", "inputs": ["X4"], "output": "P2", "attrs": {"kernel_size": [2, 2], "stride": [2, 2]}},
            {"op": "conv1d", "inputs": ["X1D", "W1D", "B1D"], "output": "Y1D", "attrs": {"stride": 1, "padding": 1, "dilation": 1, "groups": 1}},
            {
                "op": "conv3d",
                "inputs": ["X3D", "W3D", "B3D"],
                "output": "Y3D",
                "attrs": {"stride": [1, 1, 1], "padding": [1, 1, 1], "dilation": [1, 1, 1], "groups": 1},
            },
            {
                "op": "conv_depthwise2d",
                "inputs": ["XDW", "WDW", "BDW"],
                "output": "YDW",
                "attrs": {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1]},
            },
            {"op": "trace", "inputs": ["X"], "output": "TR"},
            {"op": "triu", "inputs": ["X"], "output": "TRIU", "attrs": {"diagonal": 0}},
            {"op": "upsample_nearest1d", "inputs": ["UP1"], "output": "UP1O"},
            {"op": "upsample_nearest2d", "inputs": ["UP2"], "output": "UP2O"},
            {"op": "scatter", "inputs": ["X", "SC_IDX", "SC_SRC"], "output": "SC_OUT", "attrs": {"dim": 1}},
            {"op": "select_scatter", "inputs": ["X", "SEL_SRC"], "output": "SEL_OUT", "attrs": {"dim": 1, "index": 0}},
            {
                "op": "slice_scatter",
                "inputs": ["X", "SLS_SRC"],
                "output": "SLS_OUT",
                "attrs": {"dim": 1, "start": 0, "end": 4, "step": 1},
            },
            {
                "op": "quantile",
                "inputs": ["X", "QV"],
                "output": "QT_OUT",
                "attrs": {"dim": 1, "keepdim": False, "interpolation": "linear"},
            },
            {"op": "polar", "inputs": ["ABS_P", "ANG_P"], "output": "POLAR_OUT"},
            {"op": "kron", "inputs": ["X", "Y"], "output": "KRON"},
            {"op": "masked_select", "inputs": ["X", "MASK"], "output": "SEL"},
            {"op": "masked_scatter", "inputs": ["X", "MASK", "SRC"], "output": "MS"},
            {"op": "mse_loss", "inputs": ["X", "Y"], "output": "LOSS", "attrs": {"reduction": 1}},
            {"op": "nan_to_num", "inputs": ["X"], "output": "N2N", "attrs": {"nan": 0.0, "posinf": 9.0, "neginf": -9.0}},
            {
                "op": "nll_loss2d_forward",
                "inputs": ["LOGITS", "TARGET2", "WEIGHT2"],
                "output": "NLL_OUT",
                "attrs": {"reduction": 1, "ignore_index": -100},
            },
            {
                "op": "nll_loss_forward",
                "inputs": ["LOGITS1", "TARGET1", "WEIGHT1"],
                "output": "NLL1_OUT",
                "attrs": {"reduction": 1, "ignore_index": -100},
            },
            {
                "op": "max_pool2d_with_indices",
                "inputs": ["X4"],
                "output": "P2V",
                "attrs": {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "select": "values"},
            },
            {
                "op": "max_pool2d_with_indices",
                "inputs": ["X4"],
                "output": "P2I",
                "attrs": {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0], "select": "indices"},
            },
        ],
        "outputs": ["Out"],
    }
    src["tensors"]["X4"] = {"dtype": "f32", "shape": ["N", "C", "H", "W"], "layout": "row_major"}
    src["tensors"]["P2"] = {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"}
    src["tensors"]["X1D"] = {"dtype": "f32", "shape": ["N1", "C1", "L1"], "layout": "row_major"}
    src["tensors"]["W1D"] = {"dtype": "f32", "shape": ["CO1", "CI1", "K1"], "layout": "row_major"}
    src["tensors"]["B1D"] = {"dtype": "f32", "shape": ["CO1"], "layout": "row_major"}
    src["tensors"]["Y1D"] = {"dtype": "f32", "shape": ["N1", "CO1", "O1"], "layout": "row_major"}
    src["tensors"]["X3D"] = {"dtype": "f32", "shape": ["N3", "C3", "D3", "H3", "W3"], "layout": "row_major"}
    src["tensors"]["W3D"] = {"dtype": "f32", "shape": ["CO3", "CI3", "KD3", "KH3", "KW3"], "layout": "row_major"}
    src["tensors"]["B3D"] = {"dtype": "f32", "shape": ["CO3"], "layout": "row_major"}
    src["tensors"]["Y3D"] = {"dtype": "f32", "shape": ["N3", "CO3", "OD3", "OH3", "OW3"], "layout": "row_major"}
    src["tensors"]["XDW"] = {"dtype": "f32", "shape": ["NDW", "CDW", "HDW", "WDW"], "layout": "row_major"}
    src["tensors"]["WDW"] = {"dtype": "f32", "shape": ["CODW", 1, "KHDW", "KWDW"], "layout": "row_major"}
    src["tensors"]["BDW"] = {"dtype": "f32", "shape": ["CODW"], "layout": "row_major"}
    src["tensors"]["YDW"] = {"dtype": "f32", "shape": ["NDW", "CODW", "OHDW", "OWDW"], "layout": "row_major"}
    src["tensors"]["TR"] = {"dtype": "f32", "shape": [], "layout": "row_major"}
    src["tensors"]["TRIU"] = {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["UP1"] = {"dtype": "f32", "shape": ["NUP1", "CUP1", "LUP1"], "layout": "row_major"}
    src["tensors"]["UP1O"] = {"dtype": "f32", "shape": ["NUP1", "CUP1", "OUP1"], "layout": "row_major"}
    src["tensors"]["UP2"] = {"dtype": "f32", "shape": ["NUP2", "CUP2", "HUP2", "WUP2"], "layout": "row_major"}
    src["tensors"]["UP2O"] = {"dtype": "f32", "shape": ["NUP2", "CUP2", "OHUP2", "OWUP2"], "layout": "row_major"}
    src["tensors"]["SC_IDX"] = {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["SC_SRC"] = {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["SC_OUT"] = {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["SEL_SRC"] = {"dtype": "f32", "shape": ["M"], "layout": "row_major"}
    src["tensors"]["SEL_OUT"] = {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["SLS_SRC"] = {"dtype": "f32", "shape": ["M", 4], "layout": "row_major"}
    src["tensors"]["SLS_OUT"] = {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["QV"] = {"dtype": "f32", "shape": [], "layout": "row_major"}
    src["tensors"]["QT_OUT"] = {"dtype": "f32", "shape": ["M"], "layout": "row_major"}
    src["tensors"]["ABS_P"] = {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["ANG_P"] = {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["POLAR_OUT"] = {"dtype": "f32", "shape": ["M", "N", 2], "layout": "row_major"}
    src["tensors"]["ANG"] = {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["BN"] = {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["BA"] = {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["BO"] = {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["BLS"] = {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["BRS"] = {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["CNZ"] = {"dtype": "i64", "shape": ["M"], "layout": "row_major"}
    src["tensors"]["D"] = {"dtype": "f32", "shape": ["M"], "layout": "row_major"}
    src["tensors"]["DE"] = {"dtype": "i32", "shape": ["M", "N", "N"], "layout": "row_major"}
    src["tensors"]["KRON"] = {"dtype": "f32", "shape": ["MK", "NK"], "layout": "row_major"}
    src["tensors"]["MASK"] = {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["SRC"] = {"dtype": "f32", "shape": ["L"], "layout": "row_major"}
    src["tensors"]["SEL"] = {"dtype": "f32", "shape": ["L"], "layout": "row_major"}
    src["tensors"]["MS"] = {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["LOSS"] = {"dtype": "f32", "shape": [], "layout": "row_major"}
    src["tensors"]["N2N"] = {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"}
    src["tensors"]["LOGITS"] = {"dtype": "f32", "shape": ["NB", "CB", "HB", "WB"], "layout": "row_major"}
    src["tensors"]["TARGET2"] = {"dtype": "i64", "shape": ["NB", "HB", "WB"], "layout": "row_major"}
    src["tensors"]["WEIGHT2"] = {"dtype": "f32", "shape": ["CB"], "layout": "row_major"}
    src["tensors"]["NLL_OUT"] = {"dtype": "f32", "shape": [], "layout": "row_major"}
    src["tensors"]["LOGITS1"] = {"dtype": "f32", "shape": ["NB1", "CB1"], "layout": "row_major"}
    src["tensors"]["TARGET1"] = {"dtype": "i64", "shape": ["NB1"], "layout": "row_major"}
    src["tensors"]["WEIGHT1"] = {"dtype": "f32", "shape": ["CB1"], "layout": "row_major"}
    src["tensors"]["NLL1_OUT"] = {"dtype": "f32", "shape": [], "layout": "row_major"}
    src["tensors"]["P2V"] = {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"}
    src["tensors"]["P2I"] = {"dtype": "i64", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"}
    intent = IntentFunction.from_json_dict(src)
    assert intent.name == "structure_ok"


def test_topk_requires_non_negative_k() -> None:
    bad = {
        "name": "topk_bad",
        "tensors": {
            "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "Out": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
        },
        "ops": [{"op": "topk", "inputs": ["X"], "output": "Out", "attrs": {"k": -1, "axis": 1}}],
        "outputs": ["Out"],
    }
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)


def test_pad_requires_valid_pad_width_pairs() -> None:
    bad = {
        "name": "pad_bad",
        "tensors": {
            "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "Out": {"dtype": "f32", "shape": ["M2", "N2"], "layout": "row_major"},
        },
        "ops": [{"op": "pad", "inputs": ["X"], "output": "Out", "attrs": {"pad_width": {"pairs": [[1], [0, 1]]}}}],
        "outputs": ["Out"],
    }
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)


def test_avg_pool2d_requires_valid_kernel_size() -> None:
    bad = {
        "name": "pool_bad",
        "tensors": {
            "X": {"dtype": "f32", "shape": ["N", "C", "H", "W"], "layout": "row_major"},
            "Y": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"], "layout": "row_major"},
        },
        "ops": [{"op": "avg_pool2d", "inputs": ["X"], "output": "Y", "attrs": {"kernel_size": "2"}}],
        "outputs": ["Y"],
    }
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)


def test_diag_embed_requires_distinct_dims() -> None:
    bad = {
        "name": "diag_embed_bad",
        "tensors": {
            "X": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "Y": {"dtype": "f32", "shape": ["M", "N", "N"], "layout": "row_major"},
        },
        "ops": [{"op": "diag_embed", "inputs": ["X"], "output": "Y", "attrs": {"dim1": -1, "dim2": -1}}],
        "outputs": ["Y"],
    }
    with pytest.raises(IntentIRValidationError):
        IntentFunction.from_json_dict(bad)


def test_glu_cum_index_update_ops_validate_success() -> None:
    src = {
        "name": "glu_cum_index_ok",
        "tensors": {
            "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            "v": {"dtype": "f32", "shape": ["K"], "layout": "row_major"},
            "base": {"dtype": "f32", "shape": ["M", "P"], "layout": "row_major"},
            "idx": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
            "src": {"dtype": "f32", "shape": ["L", "P"], "layout": "row_major"},
            "row_idx": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
            "col_idx": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
            "vals": {"dtype": "f32", "shape": ["L"], "layout": "row_major"},
            "g": {"dtype": "f32", "shape": ["M", "NH"], "layout": "row_major"},
            "mx": {"dtype": "f32", "shape": ["K"], "layout": "row_major"},
            "mn": {"dtype": "f32", "shape": ["K"], "layout": "row_major"},
            "a": {"dtype": "f32", "shape": ["M", "P"], "layout": "row_major"},
            "p": {"dtype": "f32", "shape": ["M", "P"], "layout": "row_major"},
        },
        "ops": [
            {"op": "glu", "inputs": ["x"], "output": "g", "attrs": {"axis": 1}},
            {"op": "cummax", "inputs": ["v"], "output": "mx", "attrs": {"axis": 0}},
            {"op": "cummin", "inputs": ["v"], "output": "mn", "attrs": {"axis": 0}},
            {"op": "index_add", "inputs": ["base", "idx", "src"], "output": "a", "attrs": {"axis": 0, "alpha": 1.0}},
            {"op": "index_put", "inputs": ["base", "row_idx", "col_idx", "vals"], "output": "p", "attrs": {"accumulate": False}},
        ],
        "outputs": ["g", "mx", "mn", "a", "p"],
    }
    intent = IntentFunction.from_json_dict(src)
    assert intent.name == "glu_cum_index_ok"
