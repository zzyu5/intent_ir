from __future__ import annotations

import math

from intent_ir.ir import IntentFunction
from intent_ir.parser import CandidateIntent
from pipeline.triton.flaggems_intent_normalize import (
    canonical_flaggems_intent_for_spec,
    maybe_normalize_flaggems_candidate,
)


def _dummy_candidate(name: str = "dummy") -> CandidateIntent:
    intent = IntentFunction.from_json_dict(
        {
            "name": name,
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [{"op": "identity", "inputs": ["x"], "output": "y"}],
            "outputs": ["y"],
        }
    )
    return CandidateIntent(intent=intent, problem_params={}, schedule_params={}, raw_json={}, llm_trace={})


def test_canonical_intent_templates_exist_for_blocked_kernels() -> None:
    sigmoid = canonical_flaggems_intent_for_spec("sigmoid2d")
    assert sigmoid is not None
    assert sigmoid.name == "sigmoid2d"
    assert sigmoid.outputs == ["output"]
    assert any(op.op == "exp" for op in sigmoid.ops)
    assert not any(("base" in (op.attrs or {})) for op in sigmoid.ops if op.op == "exp")

    batch_norm = canonical_flaggems_intent_for_spec("batch_norm2d")
    assert batch_norm is not None
    assert batch_norm.name == "batch_norm2d"
    assert batch_norm.outputs == ["output_1", "mean", "inv_std", "running_mean_out", "running_var_out"]
    assert any(op.op == "reduce_sum" for op in batch_norm.ops)
    assert any(op.op == "rsqrt" for op in batch_norm.ops)

    isnan = canonical_flaggems_intent_for_spec("isnan2d")
    assert isnan is not None
    assert [op.op for op in isnan.ops] == ["ne"]

    isinf = canonical_flaggems_intent_for_spec("isinf2d")
    assert isinf is not None
    assert [op.op for op in isinf.ops] == ["abs", "const", "gt"]
    const_val = [op.attrs["value"] for op in isinf.ops if op.op == "const"][0]
    assert math.isfinite(float(const_val))

    isfinite = canonical_flaggems_intent_for_spec("isfinite2d")
    assert isfinite is not None
    assert [op.op for op in isfinite.ops] == ["abs", "const", "le"]
    const_val = [op.attrs["value"] for op in isfinite.ops if op.op == "const"][0]
    assert math.isfinite(float(const_val))

    row_all = canonical_flaggems_intent_for_spec("row_all")
    assert row_all is not None
    assert [op.op for op in row_all.ops] == ["const", "eq", "reduce_any", "not"]
    reduce_attrs = row_all.ops[2].attrs or {}
    assert reduce_attrs.get("dims") == [1]
    assert reduce_attrs.get("keepdims") is True
    row_all_json = row_all.to_json_dict()
    assert row_all_json["tensors"]["out"]["shape"] == ["M", 1]

    masked_fill = canonical_flaggems_intent_for_spec("masked_fill2d")
    assert masked_fill is not None
    assert [op.op for op in masked_fill.ops] == ["where"]
    assert "eq" not in [op.op for op in masked_fill.ops]

    cat2d = canonical_flaggems_intent_for_spec("cat2d")
    assert cat2d is not None
    assert [op.op for op in cat2d.ops] == ["concat"]
    assert (cat2d.ops[0].attrs or {}).get("axis") == 1

    hstack2d = canonical_flaggems_intent_for_spec("hstack2d")
    assert hstack2d is not None
    assert [op.op for op in hstack2d.ops] == ["concat"]
    assert (hstack2d.ops[0].attrs or {}).get("axis") == 1

    clamp2d = canonical_flaggems_intent_for_spec("clamp2d")
    assert clamp2d is not None
    assert [op.op for op in clamp2d.ops] == ["cast", "max", "min"]

    const_pad = canonical_flaggems_intent_for_spec("constant_pad_nd2d")
    assert const_pad is not None
    assert [op.op for op in const_pad.ops] == ["pad"]
    assert (const_pad.ops[0].attrs or {}).get("pad_width") == {"pairs": [[1, 0], [1, 2]]}

    gather = canonical_flaggems_intent_for_spec("gather2d")
    assert gather is not None
    assert [op.op for op in gather.ops] == ["gather"]
    assert gather.ops[0].inputs == ["inp", "row_idx", "col_idx"]

    index_select = canonical_flaggems_intent_for_spec("index_select2d")
    assert index_select is not None
    assert [op.op for op in index_select.ops] == ["gather"]
    assert index_select.ops[0].inputs == ["inp", "row_idx", "col_idx"]

    flip = canonical_flaggems_intent_for_spec("flip2d")
    assert flip is not None
    assert [op.op for op in flip.ops] == ["gather"]
    assert flip.ops[0].inputs == ["inp", "row_idx", "col_idx"]

    embedding = canonical_flaggems_intent_for_spec("embedding2d")
    assert embedding is not None
    assert [op.op for op in embedding.ops] == ["gather"]
    assert embedding.ops[0].inputs == ["inp", "row_idx", "col_idx"]

    isin = canonical_flaggems_intent_for_spec("isin1d")
    assert isin is not None
    assert [op.op for op in isin.ops] == ["broadcast_in_dim", "broadcast_in_dim", "ne", "not", "reduce_any"]

    kron = canonical_flaggems_intent_for_spec("kron2d")
    assert kron is not None
    assert [op.op for op in kron.ops] == ["kron"]

    linspace = canonical_flaggems_intent_for_spec("linspace1d")
    assert linspace is not None
    assert [op.op for op in linspace.ops] == ["iota", "cast", "sub", "div", "mul", "add"]

    logspace = canonical_flaggems_intent_for_spec("logspace1d")
    assert logspace is not None
    assert [op.op for op in logspace.ops] == ["iota", "cast", "sub", "div", "mul", "add", "mul", "exp"]

    le2d = canonical_flaggems_intent_for_spec("le2d")
    assert le2d is not None
    assert [op.op for op in le2d.ops] == ["le"]

    log2d = canonical_flaggems_intent_for_spec("log2d")
    assert log2d is not None
    assert [op.op for op in log2d.ops] == ["log"]

    log_sigmoid2d = canonical_flaggems_intent_for_spec("log_sigmoid2d")
    assert log_sigmoid2d is not None
    assert [op.op for op in log_sigmoid2d.ops] == ["abs", "const", "mul", "exp", "const", "add", "log", "const", "min", "sub"]

    log_softmax2d = canonical_flaggems_intent_for_spec("log_softmax2d")
    assert log_softmax2d is not None
    assert [op.op for op in log_softmax2d.ops] == ["softmax", "log"]
    assert (log_softmax2d.ops[0].attrs or {}).get("axis") == 1

    logical_and2d = canonical_flaggems_intent_for_spec("logical_and2d")
    assert logical_and2d is not None
    assert [op.op for op in logical_and2d.ops] == ["and"]

    logical_not2d = canonical_flaggems_intent_for_spec("logical_not2d")
    assert logical_not2d is not None
    assert [op.op for op in logical_not2d.ops] == ["not"]

    masked_scatter = canonical_flaggems_intent_for_spec("masked_scatter2d")
    assert masked_scatter is not None
    assert [op.op for op in masked_scatter.ops] == ["masked_scatter"]

    masked_select = canonical_flaggems_intent_for_spec("masked_select2d")
    assert masked_select is not None
    assert [op.op for op in masked_select.ops] == ["masked_select"]

    mse_loss = canonical_flaggems_intent_for_spec("mse_loss2d")
    assert mse_loss is not None
    assert [op.op for op in mse_loss.ops] == ["mse_loss"]

    nan_to_num = canonical_flaggems_intent_for_spec("nan_to_num2d")
    assert nan_to_num is not None
    assert [op.op for op in nan_to_num.ops] == ["nan_to_num"]

    nll_loss = canonical_flaggems_intent_for_spec("nll_loss2d_forward")
    assert nll_loss is not None
    assert [op.op for op in nll_loss.ops] == ["nll_loss2d_forward"]

    nll_loss_1d = canonical_flaggems_intent_for_spec("nll_loss_forward")
    assert nll_loss_1d is not None
    assert [op.op for op in nll_loss_1d.ops] == ["nll_loss_forward"]

    one_hot = canonical_flaggems_intent_for_spec("one_hot2d")
    assert one_hot is not None
    assert [op.op for op in one_hot.ops] == ["iota", "broadcast_in_dim", "ne", "not", "cast"]

    pool_idx = canonical_flaggems_intent_for_spec("max_pool2d_with_indices_nchw")
    assert pool_idx is not None
    assert [op.op for op in pool_idx.ops] == ["max_pool2d_with_indices", "max_pool2d_with_indices"]

    conv1d = canonical_flaggems_intent_for_spec("conv1d_ncl")
    assert conv1d is not None
    assert [op.op for op in conv1d.ops] == ["conv1d"]

    conv3d = canonical_flaggems_intent_for_spec("conv3d_ncdhw")
    assert conv3d is not None
    assert [op.op for op in conv3d.ops] == ["conv3d"]

    conv_dw = canonical_flaggems_intent_for_spec("conv_depthwise2d_nchw")
    assert conv_dw is not None
    assert [op.op for op in conv_dw.ops] == ["conv_depthwise2d"]

    scatter = canonical_flaggems_intent_for_spec("scatter2d")
    assert scatter is not None
    assert [op.op for op in scatter.ops] == ["scatter"]

    select_scatter = canonical_flaggems_intent_for_spec("select_scatter2d")
    assert select_scatter is not None
    assert [op.op for op in select_scatter.ops] == ["select_scatter"]

    slice_scatter = canonical_flaggems_intent_for_spec("slice_scatter2d")
    assert slice_scatter is not None
    assert [op.op for op in slice_scatter.ops] == ["slice_scatter"]

    quantile = canonical_flaggems_intent_for_spec("quantile2d")
    assert quantile is not None
    assert [op.op for op in quantile.ops] == ["quantile"]

    polar = canonical_flaggems_intent_for_spec("polar2d")
    assert polar is not None
    assert [op.op for op in polar.ops] == ["polar"]

    unique2 = canonical_flaggems_intent_for_spec("unique2d")
    assert unique2 is not None
    assert [op.op for op in unique2.ops] == ["unique"]

    weight_norm = canonical_flaggems_intent_for_spec("weight_norm2d")
    assert weight_norm is not None
    assert [op.op for op in weight_norm.ops] == ["weight_norm_interface"]

    sdpa = canonical_flaggems_intent_for_spec("scaled_dot_product_attention_bhsd")
    assert sdpa is not None
    assert [op.op for op in sdpa.ops] == ["scaled_dot_product_attention"]

    angle = canonical_flaggems_intent_for_spec("angle2d")
    assert angle is not None
    assert [op.op for op in angle.ops] == ["const", "const", "lt", "where"]

    bitwise_and = canonical_flaggems_intent_for_spec("bitwise_and2d")
    assert bitwise_and is not None
    assert [op.op for op in bitwise_and.ops] == ["bitwise_and"]

    bitwise_or = canonical_flaggems_intent_for_spec("bitwise_or2d")
    assert bitwise_or is not None
    assert [op.op for op in bitwise_or.ops] == ["bitwise_or"]

    bitwise_left_shift = canonical_flaggems_intent_for_spec("bitwise_left_shift2d")
    assert bitwise_left_shift is not None
    assert [op.op for op in bitwise_left_shift.ops] == ["bitwise_left_shift"]

    bitwise_right_shift = canonical_flaggems_intent_for_spec("bitwise_right_shift2d")
    assert bitwise_right_shift is not None
    assert [op.op for op in bitwise_right_shift.ops] == ["bitwise_right_shift"]

    bitwise_not = canonical_flaggems_intent_for_spec("bitwise_not2d")
    assert bitwise_not is not None
    assert [op.op for op in bitwise_not.ops] == ["bitwise_not"]

    row_max = canonical_flaggems_intent_for_spec("row_max")
    assert row_max is not None
    assert [op.op for op in row_max.ops] == ["reduce_max"]
    assert row_max.outputs == ["out"]

    any_dim = canonical_flaggems_intent_for_spec("any_kernel_dim")
    assert any_dim is not None
    assert [op.op for op in any_dim.ops] == ["const", "ne", "reduce_any"]

    argmax = canonical_flaggems_intent_for_spec("argmax2d")
    assert argmax is not None
    assert [op.op for op in argmax.ops] == ["argmax"]

    argmin = canonical_flaggems_intent_for_spec("argmin2d")
    assert argmin is not None
    assert [op.op for op in argmin.ops] == ["argmin"]

    avg_pool2d = canonical_flaggems_intent_for_spec("avg_pool2d_nchw")
    assert avg_pool2d is not None
    assert [op.op for op in avg_pool2d.ops] == ["avg_pool2d"]

    trace = canonical_flaggems_intent_for_spec("trace2d")
    assert trace is not None
    assert [op.op for op in trace.ops] == ["trace"]

    triu = canonical_flaggems_intent_for_spec("triu2d")
    assert triu is not None
    assert [op.op for op in triu.ops] == ["triu"]

    up1d = canonical_flaggems_intent_for_spec("upsample_nearest1d_ncl")
    assert up1d is not None
    assert [op.op for op in up1d.ops] == ["upsample_nearest1d"]

    up2d = canonical_flaggems_intent_for_spec("upsample_nearest2d_nchw")
    assert up2d is not None
    assert [op.op for op in up2d.ops] == ["upsample_nearest2d"]

    glu = canonical_flaggems_intent_for_spec("glu2d")
    assert glu is not None
    assert [op.op for op in glu.ops] == ["glu"]

    cummax = canonical_flaggems_intent_for_spec("cummax1d")
    assert cummax is not None
    assert [op.op for op in cummax.ops] == ["cummax"]

    cummin = canonical_flaggems_intent_for_spec("cummin1d")
    assert cummin is not None
    assert [op.op for op in cummin.ops] == ["cummin"]

    index_add = canonical_flaggems_intent_for_spec("index_add2d")
    assert index_add is not None
    assert [op.op for op in index_add.ops] == ["index_add"]

    index_put = canonical_flaggems_intent_for_spec("index_put2d")
    assert index_put is not None
    assert [op.op for op in index_put.ops] == ["index_put"]

    count_nonzero = canonical_flaggems_intent_for_spec("count_nonzero2d")
    assert count_nonzero is not None
    assert [op.op for op in count_nonzero.ops] == ["const", "ne", "cast", "reduce_sum"]

    diag = canonical_flaggems_intent_for_spec("diag2d")
    assert diag is not None
    assert [op.op for op in diag.ops] == ["iota", "gather"]

    diag_embed = canonical_flaggems_intent_for_spec("diag_embed2d")
    assert diag_embed is not None
    assert [op.op for op in diag_embed.ops] == ["const", "broadcast_in_dim", "iota", "iota", "iota", "ne", "not", "gather", "where"]

    elu = canonical_flaggems_intent_for_spec("elu2d")
    assert elu is not None
    assert [op.op for op in elu.ops] == ["const", "const", "gt", "exp", "sub", "where"]

    celu = canonical_flaggems_intent_for_spec("celu2d")
    assert celu is not None
    assert [op.op for op in celu.ops] == ["const", "const", "gt", "exp", "sub", "where"]

    eye = canonical_flaggems_intent_for_spec("eye2d")
    assert eye is not None
    assert [op.op for op in eye.ops] == ["iota", "iota", "ne", "not", "cast"]

    eye_m = canonical_flaggems_intent_for_spec("eye_m2d")
    assert eye_m is not None
    assert [op.op for op in eye_m.ops] == ["iota", "iota", "ne", "not", "cast"]


def test_maybe_normalize_flaggems_candidate_overrides_known_spec() -> None:
    cand = _dummy_candidate("old")
    out, out_expanded, info = maybe_normalize_flaggems_candidate(
        spec_name="sigmoid2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is not None and info.get("applied") is True
    assert out.intent.name == "sigmoid2d"
    assert out_expanded is not None
    assert out.raw_json.get("normalized_by") == "flaggems_canonical"

    out2, out2_expanded, info2 = maybe_normalize_flaggems_candidate(
        spec_name="masked_fill2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info2 is not None and info2.get("applied") is True
    assert out2.intent.name == "masked_fill2d"
    assert out2_expanded is not None
    assert [op.op for op in out2.intent.ops] == ["where"]

    out3, out3_expanded, info3 = maybe_normalize_flaggems_candidate(
        spec_name="one_hot2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info3 is not None and info3.get("applied") is True
    assert out3.intent.name == "one_hot2d"
    assert out3_expanded is not None

    out4, out4_expanded, info4 = maybe_normalize_flaggems_candidate(
        spec_name="conv1d_ncl",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info4 is not None and info4.get("applied") is True
    assert out4.intent.name == "conv1d_ncl"
    assert out4_expanded is not None

    out5, out5_expanded, info5 = maybe_normalize_flaggems_candidate(
        spec_name="trace2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info5 is not None and info5.get("applied") is True
    assert out5.intent.name == "trace2d"
    assert out5_expanded is not None

    out6, out6_expanded, info6 = maybe_normalize_flaggems_candidate(
        spec_name="scatter2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info6 is not None and info6.get("applied") is True
    assert out6.intent.name == "scatter2d"
    assert out6_expanded is not None


def test_maybe_normalize_flaggems_candidate_noop_for_other_specs() -> None:
    cand = _dummy_candidate("keep")
    out, out_expanded, info = maybe_normalize_flaggems_candidate(
        spec_name="relu2d",
        candidate=cand,
        candidate_expanded=None,
    )
    assert info is None
    assert out.intent.name == "keep"
    assert out_expanded is None
