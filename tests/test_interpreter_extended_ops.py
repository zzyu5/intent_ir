from __future__ import annotations

import math
import numpy as np

from intent_ir.ir import IntentFunction
from verify.interpreter import execute_intent


def test_extended_unary_ops_execute() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "extended_unary",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "a": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "b": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "c": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "d": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "neg", "inputs": ["x"], "output": "x_neg"},
                {"op": "ceil", "inputs": ["x_neg"], "output": "x_ceil"},
                {"op": "sqrt", "inputs": ["x"], "output": "x_sqrt"},
                {"op": "log", "inputs": ["x"], "output": "x_log"},
                {"op": "sin", "inputs": ["x"], "output": "x_sin"},
                {"op": "cos", "inputs": ["x"], "output": "x_cos"},
                {"op": "tan", "inputs": ["x"], "output": "x_tan"},
                {"op": "erf", "inputs": ["x"], "output": "x_erf"},
                {"op": "acos", "inputs": ["x"], "output": "x_acos"},
                {"op": "atan", "inputs": ["x"], "output": "x_atan"},
                {"op": "add", "inputs": ["x_sqrt", "x_log"], "output": "a"},
                {"op": "add", "inputs": ["x_sin", "x_cos"], "output": "b"},
                {"op": "add", "inputs": ["x_tan", "x_erf"], "output": "c"},
                {"op": "add", "inputs": ["x_acos", "x_atan"], "output": "d"},
            ],
            "outputs": ["a", "b", "c", "d"],
        }
    )
    x = np.array([[0.5, 1.0], [0.25, 0.75]], dtype=np.float32)
    out = execute_intent(intent, {"x": x}, shape_bindings={"M": 2, "N": 2})

    assert np.allclose(out["a"], np.sqrt(x) + np.log(x), atol=1e-6)
    assert np.allclose(out["b"], np.sin(x) + np.cos(x), atol=1e-6)
    assert np.allclose(out["c"], np.tan(x) + np.vectorize(math.erf)(x), atol=1e-6)
    assert np.allclose(out["d"], np.arccos(x) + np.arctan(x), atol=1e-6)


def test_extended_reduction_ops_execute() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "extended_reduce",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mn": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "pd": {"dtype": "f32", "shape": ["M"], "layout": "row_major"},
                "am": {"dtype": "i64", "shape": ["M"], "layout": "row_major"},
                "cs": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "reduce_min", "inputs": ["x"], "output": "mn", "attrs": {"dims": [1]}},
                {"op": "reduce_prod", "inputs": ["x"], "output": "pd", "attrs": {"dims": [1]}},
                {"op": "argmax", "inputs": ["x"], "output": "am", "attrs": {"axis": 1}},
                {"op": "cumsum", "inputs": ["x"], "output": "cs", "attrs": {"axis": 1}},
            ],
            "outputs": ["mn", "pd", "am", "cs"],
        }
    )
    x = np.array([[1.0, 3.0, 2.0], [4.0, 2.0, 8.0]], dtype=np.float32)
    out = execute_intent(intent, {"x": x}, shape_bindings={"M": 2, "N": 3})

    assert np.allclose(out["mn"], np.min(x, axis=1), atol=1e-6)
    assert np.allclose(out["pd"], np.prod(x, axis=1), atol=1e-6)
    assert np.array_equal(out["am"], np.argmax(x, axis=1))
    assert np.allclose(out["cs"], np.cumsum(x, axis=1), atol=1e-6)


def test_reduce_any_with_and_combine_and_exp_base2() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "reduce_any_and_and_exp2",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "nz": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "all_out": {"dtype": "i1", "shape": ["M"], "layout": "row_major"},
                "exp2_out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
            },
            "ops": [
                {"op": "const", "inputs": [], "output": "zero", "attrs": {"value": 0.0, "dtype": "f32"}},
                {"op": "ne", "inputs": ["x", "zero"], "output": "nz"},
                {"op": "reduce_any", "inputs": ["nz"], "output": "all_out", "attrs": {"dims": [1], "combine_fn": "and"}},
                {"op": "exp", "inputs": ["x"], "output": "exp2_out", "attrs": {"base": 2.0}},
            ],
            "outputs": ["all_out", "exp2_out"],
        }
    )
    x = np.array([[1.0, 2.0, 3.0], [0.0, 4.0, 5.0]], dtype=np.float32)
    out = execute_intent(intent, {"x": x}, shape_bindings={"M": 2, "N": 3})

    assert np.array_equal(out["all_out"], np.all(x != 0, axis=1))
    assert np.allclose(out["exp2_out"], np.exp2(x), atol=1e-6)


def test_extended_structure_ops_execute() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "extended_structure",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "y": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mask": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "u_in": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
                "cat_out": {"dtype": "f32", "shape": ["MC", "N"], "layout": "row_major"},
                "stack_out": {"dtype": "f32", "shape": ["S", "M", "N"], "layout": "row_major"},
                "tile_out": {"dtype": "f32", "shape": ["MT", "N"], "layout": "row_major"},
                "repeat_out": {"dtype": "f32", "shape": ["M", "NR"], "layout": "row_major"},
                "repeat_interleave_out": {"dtype": "f32", "shape": ["M", "NRI"], "layout": "row_major"},
                "pad_out": {"dtype": "f32", "shape": ["MP", "NP"], "layout": "row_major"},
                "sort_out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "topk_out": {"dtype": "f32", "shape": ["M", "K"], "layout": "row_major"},
                "unique_out": {"dtype": "i32", "shape": ["U"], "layout": "row_major"},
                "nonzero_out": {"dtype": "i64", "shape": ["KZ", "D2"], "layout": "row_major"},
            },
            "ops": [
                {"op": "concat", "inputs": ["x", "y"], "output": "cat_out", "attrs": {"axis": 0}},
                {"op": "stack", "inputs": ["x", "y"], "output": "stack_out", "attrs": {"axis": 0}},
                {"op": "tile", "inputs": ["x"], "output": "tile_out", "attrs": {"repeats": [2, 1]}},
                {"op": "repeat", "inputs": ["x"], "output": "repeat_out", "attrs": {"repeats": 2, "axis": 1}},
                {
                    "op": "repeat_interleave",
                    "inputs": ["x"],
                    "output": "repeat_interleave_out",
                    "attrs": {"repeats": [1, 2, 1], "axis": 1},
                },
                {
                    "op": "pad",
                    "inputs": ["x"],
                    "output": "pad_out",
                    "attrs": {"pad_width": {"pairs": [[1, 0], [0, 2]]}, "mode": "constant", "value": 0.0},
                },
                {"op": "sort", "inputs": ["x"], "output": "sort_out", "attrs": {"axis": 1, "descending": True}},
                {"op": "topk", "inputs": ["x"], "output": "topk_out", "attrs": {"k": 2, "axis": 1, "largest": True, "sorted": True}},
                {"op": "unique", "inputs": ["u_in"], "output": "unique_out", "attrs": {"sorted": False}},
                {"op": "nonzero", "inputs": ["mask"], "output": "nonzero_out"},
            ],
            "outputs": [
                "cat_out",
                "stack_out",
                "tile_out",
                "repeat_out",
                "repeat_interleave_out",
                "pad_out",
                "sort_out",
                "topk_out",
                "unique_out",
                "nonzero_out",
            ],
        }
    )
    x = np.array([[3.0, 1.0, 2.0], [0.0, 4.0, 5.0]], dtype=np.float32)
    y = np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    u_in = np.array([3, 1, 3, 2, 1], dtype=np.int32)
    out = execute_intent(
        intent,
        {"x": x, "y": y, "mask": mask, "u_in": u_in},
        shape_bindings={
            "M": 2,
            "N": 3,
            "MC": 4,
            "S": 2,
            "MT": 4,
            "NR": 6,
            "NRI": 4,
            "MP": 3,
            "NP": 5,
            "K": 2,
            "L": 5,
            "U": 3,
            "KZ": 3,
            "D2": 2,
        },
    )

    assert np.allclose(out["cat_out"], np.concatenate([x, y], axis=0), atol=1e-6)
    assert np.allclose(out["stack_out"], np.stack([x, y], axis=0), atol=1e-6)
    assert np.allclose(out["tile_out"], np.tile(x, (2, 1)), atol=1e-6)
    assert np.allclose(out["repeat_out"], np.repeat(x, 2, axis=1), atol=1e-6)
    assert np.allclose(out["repeat_interleave_out"], np.repeat(x, [1, 2, 1], axis=1), atol=1e-6)
    assert np.allclose(out["pad_out"], np.pad(x, ((1, 0), (0, 2)), mode="constant", constant_values=0.0), atol=1e-6)
    assert np.allclose(out["sort_out"], np.flip(np.sort(x, axis=1), axis=1), atol=1e-6)
    assert np.allclose(out["topk_out"], np.flip(np.sort(x, axis=1)[:, -2:], axis=1), atol=1e-6)
    assert np.array_equal(out["unique_out"], np.array([3, 1, 2], dtype=np.int32))
    assert np.array_equal(out["nonzero_out"], np.stack(np.nonzero(mask), axis=-1))


def test_bitwise_angle_avg_pool_ops_execute() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "bitwise_angle_pool",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "a": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "b": {"dtype": "i32", "shape": ["M", "N"], "layout": "row_major"},
                "img": {"dtype": "f32", "shape": ["B", "C", "H", "W"], "layout": "row_major"},
                "x_ang": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "ab_and": {"dtype": "i64", "shape": ["M", "N"], "layout": "row_major"},
                "ab_or": {"dtype": "i64", "shape": ["M", "N"], "layout": "row_major"},
                "a_not": {"dtype": "i64", "shape": ["M", "N"], "layout": "row_major"},
                "a_lshift": {"dtype": "i64", "shape": ["M", "N"], "layout": "row_major"},
                "a_rshift": {"dtype": "i64", "shape": ["M", "N"], "layout": "row_major"},
                "img_pool": {"dtype": "f32", "shape": ["B", "C", "OH", "OW"], "layout": "row_major"},
            },
            "ops": [
                {"op": "angle", "inputs": ["x"], "output": "x_ang"},
                {"op": "bitwise_and", "inputs": ["a", "b"], "output": "ab_and"},
                {"op": "bitwise_or", "inputs": ["a", "b"], "output": "ab_or"},
                {"op": "bitwise_not", "inputs": ["a"], "output": "a_not"},
                {"op": "bitwise_left_shift", "inputs": ["a", "b"], "output": "a_lshift"},
                {"op": "bitwise_right_shift", "inputs": ["a", "b"], "output": "a_rshift"},
                {
                    "op": "avg_pool2d",
                    "inputs": ["img"],
                    "output": "img_pool",
                    "attrs": {"kernel_size": [2, 2], "stride": [2, 2], "padding": [0, 0]},
                },
            ],
            "outputs": ["x_ang", "ab_and", "ab_or", "a_not", "a_lshift", "a_rshift", "img_pool"],
        }
    )
    x = np.array([[1.0, -2.0], [0.5, -0.25]], dtype=np.float32)
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([[1, 1], [0, 2]], dtype=np.int32)
    img = np.arange(1, 17, dtype=np.float32).reshape(1, 1, 4, 4)
    out = execute_intent(
        intent,
        {"x": x, "a": a, "b": b, "img": img},
        shape_bindings={"M": 2, "N": 2, "B": 1, "C": 1, "H": 4, "W": 4},
    )

    assert np.allclose(out["x_ang"], np.angle(x), atol=1e-6)
    assert np.array_equal(out["ab_and"], np.bitwise_and(a, b))
    assert np.array_equal(out["ab_or"], np.bitwise_or(a, b))
    assert np.array_equal(out["a_not"], np.bitwise_not(a.astype(np.int64)))
    assert np.array_equal(out["a_lshift"], np.left_shift(a.astype(np.int64), b.astype(np.int64)))
    assert np.array_equal(out["a_rshift"], np.right_shift(a.astype(np.int64), b.astype(np.int64)))
    expected_pool = np.array([[[[(1 + 2 + 5 + 6) / 4.0, (3 + 4 + 7 + 8) / 4.0], [(9 + 10 + 13 + 14) / 4.0, (11 + 12 + 15 + 16) / 4.0]]]], dtype=np.float32)
    assert np.allclose(out["img_pool"], expected_pool, atol=1e-6)


def test_count_nonzero_diag_and_diag_embed_execute() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "count_diag_diag_embed",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "v": {"dtype": "f32", "shape": ["B", "N"], "layout": "row_major"},
                "cnz": {"dtype": "i64", "shape": ["M"], "layout": "row_major"},
                "d": {"dtype": "f32", "shape": ["K"], "layout": "row_major"},
                "de": {"dtype": "f32", "shape": ["B", "NP", "NP"], "layout": "row_major"},
            },
            "ops": [
                {"op": "count_nonzero", "inputs": ["x"], "output": "cnz", "attrs": {"dims": [1]}},
                {"op": "diag", "inputs": ["x"], "output": "d", "attrs": {"diagonal": 0}},
                {"op": "diag_embed", "inputs": ["v"], "output": "de", "attrs": {"offset": 1, "dim1": -2, "dim2": -1}},
            ],
            "outputs": ["cnz", "d", "de"],
        }
    )
    x = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 0.0]], dtype=np.float32)
    v = np.array([[1.0, 2.0, 3.0], [0.5, 0.25, 0.125]], dtype=np.float32)
    out = execute_intent(intent, {"x": x, "v": v}, shape_bindings={"M": 2, "N": 3, "B": 2, "K": 2, "NP": 4})

    assert np.array_equal(out["cnz"], np.count_nonzero(x, axis=1).astype(np.int64))
    assert np.array_equal(out["d"], np.diag(x))

    expected_de = np.zeros((2, 4, 4), dtype=np.float32)
    expected_de[:, np.arange(3), np.arange(3) + 1] = v
    assert np.allclose(out["de"], expected_de, atol=1e-6)


def test_glu_cum_and_index_update_ops_execute() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "glu_cum_index",
            "tensors": {
                "x": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "v": {"dtype": "f32", "shape": ["K"], "layout": "row_major"},
                "base": {"dtype": "f32", "shape": ["M", "P"], "layout": "row_major"},
                "idx": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
                "src": {"dtype": "f32", "shape": ["L", "P"], "layout": "row_major"},
                "row_idx": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
                "col_idx": {"dtype": "i32", "shape": ["L"], "layout": "row_major"},
                "vals": {"dtype": "f32", "shape": ["L"], "layout": "row_major"},
                "glu_out": {"dtype": "f32", "shape": ["M", "NH"], "layout": "row_major"},
                "cummax_out": {"dtype": "f32", "shape": ["K"], "layout": "row_major"},
                "cummin_out": {"dtype": "f32", "shape": ["K"], "layout": "row_major"},
                "index_add_out": {"dtype": "f32", "shape": ["M", "P"], "layout": "row_major"},
                "index_put_out": {"dtype": "f32", "shape": ["M", "P"], "layout": "row_major"},
            },
            "ops": [
                {"op": "glu", "inputs": ["x"], "output": "glu_out", "attrs": {"axis": 1}},
                {"op": "cummax", "inputs": ["v"], "output": "cummax_out", "attrs": {"axis": 0}},
                {"op": "cummin", "inputs": ["v"], "output": "cummin_out", "attrs": {"axis": 0}},
                {"op": "index_add", "inputs": ["base", "idx", "src"], "output": "index_add_out", "attrs": {"axis": 0, "alpha": 1.0}},
                {
                    "op": "index_put",
                    "inputs": ["base", "row_idx", "col_idx", "vals"],
                    "output": "index_put_out",
                    "attrs": {"accumulate": False},
                },
            ],
            "outputs": ["glu_out", "cummax_out", "cummin_out", "index_add_out", "index_put_out"],
        }
    )

    x = np.array([[1.0, 2.0, -1.0, 0.5], [0.25, -0.75, 1.5, -2.0]], dtype=np.float32)
    v = np.array([1.0, 0.5, 2.0, -1.0], dtype=np.float32)
    base = np.array([[1.0, 0.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    idx = np.array([0, 2], dtype=np.int32)
    src = np.array([[10.0, 11.0], [12.0, 13.0]], dtype=np.float32)
    row_idx = np.array([1, 0], dtype=np.int32)
    col_idx = np.array([0, 1], dtype=np.int32)
    vals = np.array([-3.0, 7.0], dtype=np.float32)

    out = execute_intent(
        intent,
        {"x": x, "v": v, "base": base, "idx": idx, "src": src, "row_idx": row_idx, "col_idx": col_idx, "vals": vals},
        shape_bindings={"M": 3, "N": 4, "NH": 2, "K": 4, "P": 2, "L": 2},
    )

    lhs, rhs = np.split(x, 2, axis=1)
    expected_glu = lhs * (1.0 / (1.0 + np.exp(-rhs)))
    assert np.allclose(out["glu_out"], expected_glu, atol=1e-6)
    assert np.allclose(out["cummax_out"], np.maximum.accumulate(v), atol=1e-6)
    assert np.allclose(out["cummin_out"], np.minimum.accumulate(v), atol=1e-6)

    expected_index_add = base.copy()
    np.add.at(expected_index_add, idx.astype(np.int64), src)
    assert np.allclose(out["index_add_out"], expected_index_add, atol=1e-6)

    expected_index_put = base.copy()
    expected_index_put[row_idx.astype(np.int64), col_idx.astype(np.int64)] = vals
    assert np.allclose(out["index_put_out"], expected_index_put, atol=1e-6)


def test_kron_masked_scatter_and_range_builders_execute() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "kron_masked_scatter_ranges",
            "tensors": {
                "A": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "B": {"dtype": "f32", "shape": ["P", "Q"], "layout": "row_major"},
                "inp": {"dtype": "f32", "shape": ["X", "Y"], "layout": "row_major"},
                "mask": {"dtype": "i1", "shape": ["X", "Y"], "layout": "row_major"},
                "source": {"dtype": "f32", "shape": ["L"], "layout": "row_major"},
                "start": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "end": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "denom": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "log_base": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "in0": {"dtype": "i32", "shape": ["R"], "layout": "row_major"},
                "in1": {"dtype": "i32", "shape": ["S"], "layout": "row_major"},
                "kron_out": {"dtype": "f32", "shape": ["MP", "NQ"], "layout": "row_major"},
                "masked_out": {"dtype": "f32", "shape": ["X", "Y"], "layout": "row_major"},
                "linspace_out": {"dtype": "f32", "shape": ["T"], "layout": "row_major"},
                "logspace_out": {"dtype": "f32", "shape": ["T"], "layout": "row_major"},
                "isin_out": {"dtype": "i1", "shape": ["R"], "layout": "row_major"},
            },
            "ops": [
                {"op": "kron", "inputs": ["A", "B"], "output": "kron_out"},
                {"op": "masked_scatter", "inputs": ["inp", "mask", "source"], "output": "masked_out"},
                {"op": "iota", "inputs": [], "output": "idx", "attrs": {"axis": 0, "shape": ["T"], "dtype": "i32"}},
                {"op": "cast", "inputs": ["idx"], "output": "idx_f", "attrs": {"to": "f32"}},
                {"op": "sub", "inputs": ["end", "start"], "output": "delta"},
                {"op": "div", "inputs": ["delta", "denom"], "output": "step"},
                {"op": "mul", "inputs": ["idx_f", "step"], "output": "scaled"},
                {"op": "add", "inputs": ["start", "scaled"], "output": "linspace_out"},
                {"op": "mul", "inputs": ["linspace_out", "log_base"], "output": "exp_arg"},
                {"op": "exp", "inputs": ["exp_arg"], "output": "logspace_out"},
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["in0"],
                    "output": "in0_rs",
                    "attrs": {"out_shape": ["R", "S"], "broadcast_dims": [0]},
                },
                {
                    "op": "broadcast_in_dim",
                    "inputs": ["in1"],
                    "output": "in1_rs",
                    "attrs": {"out_shape": ["R", "S"], "broadcast_dims": [1]},
                },
                {"op": "ne", "inputs": ["in0_rs", "in1_rs"], "output": "neq_rs"},
                {"op": "not", "inputs": ["neq_rs"], "output": "eq_rs"},
                {"op": "reduce_any", "inputs": ["eq_rs"], "output": "isin_out", "attrs": {"dims": [1]}},
            ],
            "outputs": ["kron_out", "masked_out", "linspace_out", "logspace_out", "isin_out"],
        }
    )

    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    B = np.array([[0.5, 1.5, -1.0]], dtype=np.float32)
    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mask = np.array([[True, False], [True, False]], dtype=np.bool_)
    source = np.array([9.0, -2.0, 7.0], dtype=np.float32)
    start = np.array(-1.0, dtype=np.float32)
    end = np.array(2.0, dtype=np.float32)
    denom = np.array(3.0, dtype=np.float32)
    log_base = np.array(np.log(10.0), dtype=np.float32)
    in0 = np.array([1, 4, 2], dtype=np.int32)
    in1 = np.array([0, 2, 5], dtype=np.int32)

    out = execute_intent(
        intent,
        {
            "A": A,
            "B": B,
            "inp": inp,
            "mask": mask,
            "source": source,
            "start": start,
            "end": end,
            "denom": denom,
            "log_base": log_base,
            "in0": in0,
            "in1": in1,
        },
        shape_bindings={"M": 2, "N": 2, "P": 1, "Q": 3, "MP": 2, "NQ": 6, "X": 2, "Y": 2, "L": 3, "T": 4, "R": 3, "S": 3},
    )

    assert np.allclose(out["kron_out"], np.kron(A, B), atol=1e-6)
    masked_expected = inp.copy().reshape(-1)
    mask_flat = mask.reshape(-1)
    masked_expected[mask_flat] = source[: int(mask_flat.sum())]
    assert np.allclose(out["masked_out"], masked_expected.reshape(inp.shape), atol=1e-6)
    lin_expected = np.linspace(float(start), float(end), 4, dtype=np.float32)
    assert np.allclose(out["linspace_out"], lin_expected, atol=1e-6)
    log_expected = np.exp(lin_expected * float(log_base)).astype(np.float32)
    assert np.allclose(out["logspace_out"], log_expected, atol=1e-6)
    assert np.array_equal(out["isin_out"], np.isin(in0, in1))


def test_masked_select_mse_nan_to_num_and_nll_loss_execute() -> None:
    intent = IntentFunction.from_json_dict(
        {
            "name": "masked_select_mse_nan_nll",
            "tensors": {
                "inp": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "mask": {"dtype": "i1", "shape": ["M", "N"], "layout": "row_major"},
                "target": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "xnan": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "logits": {"dtype": "f32", "shape": ["B", "C", "H", "W"], "layout": "row_major"},
                "labels": {"dtype": "i64", "shape": ["B", "H", "W"], "layout": "row_major"},
                "weight": {"dtype": "f32", "shape": ["C"], "layout": "row_major"},
                "masked_out": {"dtype": "f32", "shape": ["L"], "layout": "row_major"},
                "mse_out": {"dtype": "f32", "shape": [], "layout": "row_major"},
                "nan_out": {"dtype": "f32", "shape": ["M", "N"], "layout": "row_major"},
                "nll_out": {"dtype": "f32", "shape": [], "layout": "row_major"},
            },
            "ops": [
                {"op": "masked_select", "inputs": ["inp", "mask"], "output": "masked_out"},
                {"op": "mse_loss", "inputs": ["inp", "target"], "output": "mse_out", "attrs": {"reduction": 1}},
                {"op": "nan_to_num", "inputs": ["xnan"], "output": "nan_out", "attrs": {"nan": 0.0, "posinf": 9.0, "neginf": -9.0}},
                {
                    "op": "nll_loss2d_forward",
                    "inputs": ["logits", "labels", "weight"],
                    "output": "nll_out",
                    "attrs": {"reduction": 1, "ignore_index": -100},
                },
            ],
            "outputs": ["masked_out", "mse_out", "nan_out", "nll_out"],
        }
    )

    inp = np.array([[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]], dtype=np.float32)
    mask = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    target = np.array([[0.5, -1.5, 2.0], [1.0, 2.5, -2.0]], dtype=np.float32)
    xnan = np.array([[np.nan, np.inf, -np.inf], [1.0, -2.0, 3.5]], dtype=np.float32)

    logits = np.log(
        np.array(
            [
                [
                    [[0.7, 0.2], [0.1, 0.6]],
                    [[0.2, 0.3], [0.6, 0.2]],
                    [[0.1, 0.5], [0.3, 0.2]],
                ]
            ],
            dtype=np.float32,
        )
    )
    labels = np.array([[[0, 2], [1, 0]]], dtype=np.int64)
    weight = np.array([1.0, 2.0, 0.5], dtype=np.float32)

    out = execute_intent(
        intent,
        {"inp": inp, "mask": mask, "target": target, "xnan": xnan, "logits": logits, "labels": labels, "weight": weight},
        shape_bindings={"M": 2, "N": 3, "L": 3, "B": 1, "C": 3, "H": 2, "W": 2},
    )

    assert np.array_equal(out["masked_out"], inp[mask].reshape(-1))
    assert np.allclose(out["mse_out"], np.mean((inp - target) ** 2), atol=1e-6)
    assert np.allclose(out["nan_out"], np.nan_to_num(xnan, nan=0.0, posinf=9.0, neginf=-9.0), atol=1e-6)

    picked = np.array([logits[0, 0, 0, 0], logits[0, 2, 0, 1], logits[0, 1, 1, 0], logits[0, 0, 1, 1]], dtype=np.float32)
    cls = np.array([0, 2, 1, 0], dtype=np.int64)
    expected_nll = float(np.sum((-picked) * weight[cls]) / np.sum(weight[cls]))
    assert np.allclose(out["nll_out"], np.array(expected_nll, dtype=np.float32), atol=1e-6)
