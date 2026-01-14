"""
CUDA MVP full pipeline runner (Task CUDA).

Pipeline shape mirrors Triton/TileLang:
  CUDA source -> PTX -> CertificateV2 -> obligations -> contract
            -> LLM -> IntentIR -> static_validate -> cases+diff (+ Stage C).
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from frontends.common.contract_v2 import evaluate_contract_v2
from frontends.common.obligations import O3_MASK_IMPLIES_INBOUNDS, evaluate_obligations
from frontends.common.static_validate import static_validate
from frontends.cuda.runtime import CudaLaunch, CudaRuntimeError, run_cuda_kernel_io
from frontends.cuda.signature import infer_runtime_io_spec
from frontends.tilelang.cuda_export import build_io_spec_from_tilelang_prim_func, export_tilelang_cuda, tilelang_include_dirs
from frontends.tilelang.runtime import infer_written_global_buffers, run_tilelang_kernel_io
from intent_ir.llm import LLMIntentHub
from intent_ir.macros import expand_macros, enrich_intent_macros
from intent_ir.parser import CandidateIntent
from intent_ir.ir import ScheduleSketch
from intent_ir.ir.printer_mlir_like import print_mlir_like
from pipeline import registry as pipeline_registry
from pipeline.interfaces import FrontendConstraints
from verify.diff_runner import run_diff
from verify.gen_cases import TestCase, generate_cases_split
from verify.metamorphic import run_bounded_exhaustive, run_metamorphic_suite
from verify.mutation import run_mutation_kill
from verify.tolerances import infer_tolerances

from kernels.cuda.ops.vec_add import VEC_ADD_CU_PATH, vec_add_io
from kernels.cuda.ops.transpose2d import TRANSPOSE2D_CU_PATH, transpose2d_io
from kernels.cuda.ops.row_sum import ROW_SUM_CU_PATH, row_sum_io
from kernels.cuda.ops.naive_gemm import NAIVE_GEMM_CU_PATH, naive_gemm_io
from kernels.tilelang.ops.any_kernel_dim import make_any_kernel_dim_prim_func
from kernels.tilelang.ops.groupnorm import make_group_norm_kernel_prim_func
from kernels.tilelang.ops._attn_fwd import make_attn_fwd_prim_func
from kernels.tilelang.ops.softmax_inner import make_softmax_inner_prim_func
from kernels.tilelang.ops.layernorm import make_layer_norm_persistent_prim_func
from kernels.tilelang.ops.upsample_bicubic2d_aa import make_upsample_bicubic2d_aa_prim_func


ROOT = Path(__file__).resolve().parents[2]
_LLM_HUB = LLMIntentHub()

_CUDA_TILELANG_SNAPSHOT_DIR = ROOT / "kernels" / "cuda" / "ops" / "tilelang_generated"
_PTX_ENTRY_RE = re.compile(r"^\s*\.visible\s+\.entry\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(")
_LAUNCH_THREAD_RE = re.compile(r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*T\.launch_thread\(\"(?P<tag>blockIdx\.(?:x|y|z))\",\s*(?P<extent>.+)\)\s*$')


def _eval_int_expr(expr: str, bindings: Dict[str, int]) -> int:
    """
    Evaluate a small, safe subset of Python integer expressions.

    Used for TileLang/TIR-derived launch extents like:
      - M
      - M * 4
      - (N + 64 - 1) // 64
    """
    s = str(expr).strip()
    if not s:
        raise ValueError("empty int expr")

    def ev(node: ast.AST) -> int:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return int(node.value)
        if isinstance(node, ast.Name):
            name = str(node.id)
            if name not in bindings:
                raise KeyError(name)
            return int(bindings[name])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            a = ev(node.left)
            b = ev(node.right)
            if isinstance(node.op, ast.Add):
                return int(a) + int(b)
            if isinstance(node.op, ast.Sub):
                return int(a) - int(b)
            if isinstance(node.op, ast.Mult):
                return int(a) * int(b)
            if isinstance(node.op, ast.FloorDiv):
                if int(b) == 0:
                    raise ZeroDivisionError("floordiv by 0")
                return int(a) // int(b)
            if isinstance(node.op, ast.Mod):
                if int(b) == 0:
                    raise ZeroDivisionError("mod by 0")
                return int(a) % int(b)
        if isinstance(node, ast.Call):
            # Support ceildiv(a,b) / T.ceildiv(a,b) patterns if they appear.
            fn = node.func
            fn_name = None
            if isinstance(fn, ast.Name):
                fn_name = fn.id
            elif isinstance(fn, ast.Attribute):
                fn_name = fn.attr
            if fn_name in {"ceildiv", "ceil_div"} and len(node.args) == 2:
                a = ev(node.args[0])
                b = ev(node.args[1])
                return _ceil_div(int(a), int(b))
            raise ValueError(f"unsupported call in int expr: {ast.unparse(node) if hasattr(ast, 'unparse') else fn_name}")
        raise ValueError(f"unsupported int expr node: {type(node).__name__}")

    tree = ast.parse(s, mode="eval")
    return int(ev(tree))


def _infer_grid_from_tilelang_prim_func(prim_func: Any, bindings: Dict[str, int]) -> tuple[int, int, int]:
    """
    Infer CUDA grid dims from a TileLang/TVM PrimFunc TIR script.

    TileLang emits TIR with launch-thread lines like:
      bx = T.launch_thread("blockIdx.x", M)
      by = T.launch_thread("blockIdx.y", (N + 64 - 1) // 64)
    """
    txt = str(prim_func.script(show_meta=False))
    ext: Dict[str, str] = {}
    for ln in txt.splitlines():
        m = _LAUNCH_THREAD_RE.match(ln.strip())
        if not m:
            continue
        tag = str(m.group("tag"))
        e = str(m.group("extent")).strip()
        # Strip trailing comments if any.
        if "#" in e:
            e = e.split("#", 1)[0].strip()
        ext[tag] = e
    gx = _eval_int_expr(ext.get("blockIdx.x", "1"), bindings) if "blockIdx.x" in ext else 1
    gy = _eval_int_expr(ext.get("blockIdx.y", "1"), bindings) if "blockIdx.y" in ext else 1
    gz = _eval_int_expr(ext.get("blockIdx.z", "1"), bindings) if "blockIdx.z" in ext else 1
    return (max(1, int(gx)), max(1, int(gy)), max(1, int(gz)))


def _read_cuda_src(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b) if int(b) > 0 else 1


def _persist_tilelang_cuda_snapshot(name: str, exp) -> None:
    """
    Persist TileLang->CUDA exports under `kernels/cuda/ops/` for inspection/reuse.

    This keeps the CUDA frontend "kernel library" consistent with Triton/TileLang:
      - user/kernel code lives under kernels/
      - pipeline artifacts live under artifacts/
    """
    out_dir = _CUDA_TILELANG_SNAPSHOT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cuda_path = out_dir / f"{name}.cu"
    ptx_path = out_dir / f"{name}.ptx"
    meta_path = out_dir / f"{name}.tilelang_export.json"

    cuda_path.write_text(str(getattr(exp, "cuda_src", "")), encoding="utf-8")
    ptx_path.write_text(str(getattr(exp, "ptx_text", "")), encoding="utf-8")
    meta = {
        "name": str(name),
        "origin": "tilelang",
        "entry_name": str(getattr(exp, "entry_name", "")),
        "include_dirs": [str(p) for p in (getattr(exp, "include_dirs", []) or [])],
        "notes": "Generated snapshot. Edit the TileLang kernel (kernels/tilelang/ops/*) instead of this file.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_tilelang_cuda_snapshot(name: str) -> tuple[str, str, str, List[Path]]:
    """
    Load a previously-exported TileLang->CUDA snapshot from `kernels/cuda/ops/`.
    Returns: (cuda_src, ptx_text, entry_name, include_dirs)
    """
    cuda_path = _CUDA_TILELANG_SNAPSHOT_DIR / f"{name}.cu"
    ptx_path = _CUDA_TILELANG_SNAPSHOT_DIR / f"{name}.ptx"
    if not cuda_path.is_file() or not ptx_path.is_file():
        raise FileNotFoundError(f"missing TileLang CUDA snapshot for {name}: {cuda_path} / {ptx_path}")
    cuda_src = cuda_path.read_text(encoding="utf-8")
    ptx_text = ptx_path.read_text(encoding="utf-8")
    entry_name = ""
    for ln in ptx_text.splitlines():
        m = _PTX_ENTRY_RE.match(ln)
        if m:
            entry_name = str(m.group("name"))
            break
    if not entry_name:
        raise RuntimeError(f"cannot find .visible .entry in snapshot PTX for {name}: {ptx_path}")
    return cuda_src, ptx_text, entry_name, list(tilelang_include_dirs())


def _ensure_schedule_cuda(intent, *, spec: "KernelSpec") -> None:
    # Keep schedule visible; use block dims as a lightweight hint.
    if getattr(intent, "schedule", None) is not None:
        return
    bx, by, _bz = (int(x) for x in (spec.block or (1, 1, 1)))
    intent.schedule = ScheduleSketch(tile_m=by if by > 1 else None, tile_n=bx if bx > 1 else None, vec_width=1)


def _vec_add_reference(case: TestCase) -> Dict[str, np.ndarray]:
    n = int(case.shapes.get("N", 0))
    rng = np.random.default_rng(int(case.seed))
    a = rng.standard_normal((n,), dtype=np.float32)
    b = rng.standard_normal((n,), dtype=np.float32)
    block = (256, 1, 1)
    launch = CudaLaunch(grid=(_ceil_div(n, block[0]), 1, 1), block=block, shared_mem=0)
    io = run_cuda_kernel_io(
        kernel_name="vec_add",
        cuda_src=_read_cuda_src(VEC_ADD_CU_PATH),
        io_spec=dict(vec_add_io),
        launch=launch,
        bindings=dict(case.shapes),
        inputs_np={"A": a, "B": b},
        output_names=["C"],
    )
    return {"A": io["A"], "B": io["B"], "C": io["C"]}


def _transpose2d_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 0))
    n = int(case.shapes.get("N", 0))
    rng = np.random.default_rng(int(case.seed))
    inp = rng.standard_normal((m, n), dtype=np.float32)
    block = (16, 16, 1)
    launch = CudaLaunch(grid=(_ceil_div(n, block[0]), _ceil_div(m, block[1]), 1), block=block, shared_mem=0)
    io = run_cuda_kernel_io(
        kernel_name="transpose2d",
        cuda_src=_read_cuda_src(TRANSPOSE2D_CU_PATH),
        io_spec=dict(transpose2d_io),
        launch=launch,
        bindings=dict(case.shapes),
        inputs_np={"inp": inp},
        output_names=["out"],
    )
    return {"inp": io["inp"], "out": io["out"]}


def _row_sum_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 0))
    n = int(case.shapes.get("N", 0))
    rng = np.random.default_rng(int(case.seed))
    inp = rng.standard_normal((m, n), dtype=np.float32)
    block = (256, 1, 1)
    launch = CudaLaunch(grid=(_ceil_div(m, block[0]), 1, 1), block=block, shared_mem=0)
    io = run_cuda_kernel_io(
        kernel_name="row_sum",
        cuda_src=_read_cuda_src(ROW_SUM_CU_PATH),
        io_spec=dict(row_sum_io),
        launch=launch,
        bindings=dict(case.shapes),
        inputs_np={"inp": inp},
        output_names=["out"],
    )
    return {"inp": io["inp"], "out": io["out"]}


def _naive_gemm_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes.get("M", 0))
    n = int(case.shapes.get("N", 0))
    k = int(case.shapes.get("K", 0))
    rng = np.random.default_rng(int(case.seed))
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    block = (16, 16, 1)
    launch = CudaLaunch(grid=(_ceil_div(n, block[0]), _ceil_div(m, block[1]), 1), block=block, shared_mem=0)
    io = run_cuda_kernel_io(
        kernel_name="naive_gemm",
        cuda_src=_read_cuda_src(NAIVE_GEMM_CU_PATH),
        io_spec=dict(naive_gemm_io),
        launch=launch,
        bindings=dict(case.shapes),
        inputs_np={"A": a, "B": b},
        output_names=["C"],
    )
    return {"A": io["A"], "B": io["B"], "C": io["C"]}


def _any_kernel_dim_reference(case: TestCase) -> Dict[str, np.ndarray]:
    if case.inputs and "inp" in case.inputs:
        inp = np.asarray(case.inputs["inp"])
    else:
        rng = np.random.default_rng(int(case.seed))
        m = int(case.shapes["M"])
        n = int(case.shapes["N"])
        inp = rng.integers(0, 2, size=(m, n)).astype(np.float32)
    out = np.any(inp != 0, axis=1)
    return {"inp": inp, "out": out}


def _group_norm_reference(case: TestCase) -> Dict[str, np.ndarray]:
    n = int(case.shapes["N"])
    c = int(case.shapes["C"])
    hw = int(case.shapes["HW"])
    g = int(case.shapes["num_groups"])
    if g <= 0 or c % g != 0:
        raise ValueError(f"invalid group config: C={c} num_groups={g}")
    group_size = c // g
    if case.inputs and "X" in case.inputs:
        x = np.asarray(case.inputs["X"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        x = rng.standard_normal((n, c, hw), dtype=np.float32)
    if case.inputs and "W" in case.inputs:
        w = np.asarray(case.inputs["W"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        w = rng.standard_normal((c,), dtype=np.float32)
    if case.inputs and "B" in case.inputs:
        b = np.asarray(case.inputs["B"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        b = rng.standard_normal((c,), dtype=np.float32)
    eps = np.float32(1e-5)

    x4 = x.reshape(n, g, group_size, hw)
    mean = np.mean(x4, axis=(2, 3), keepdims=True)
    var = np.mean((x4 - mean) ** 2, axis=(2, 3), keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    x_hat = (x4 - mean) * rstd
    x_hat3 = x_hat.reshape(n, c, hw)
    y = x_hat3 * w[None, :, None] + b[None, :, None]
    return {
        "X": x,
        "W": w,
        "B": b,
        "Y": y.astype(np.float32),
        "Mean": mean.reshape(n, g).astype(np.float32),
        "Rstd": rstd.reshape(n, g).astype(np.float32),
    }


def _group_norm_fallback_intent():
    """
    Deterministic, compiler-style IntentIR for groupnorm.

    Used as a safety-net when LLM output is semantically invalid (diff failure).
    Keeps original view inputs (X:[N,C,HW]) and introduces explicit reshape to
    group-view for reductions/broadcast.
    """
    from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType  # noqa: PLC0415

    rm = TensorLayout(kind="row_major", params={})
    tensors: Dict[str, TensorType] = {
        "X": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm),
        "W": TensorType(dtype="f32", shape=[Dim("sym", "C")], layout=rm),
        "B": TensorType(dtype="f32", shape=[Dim("sym", "C")], layout=rm),
        "Y": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm),
        "Mean": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups")], layout=rm),
        "Rstd": TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups")], layout=rm),
    }
    ops: list[Op] = []

    ops.append(Op(op="reshape", inputs=["X"], output="X4", attrs={"shape": ["N", "num_groups", "group_size", "HW"]}))
    tensors["X4"] = TensorType(
        dtype="f32",
        shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("sym", "group_size"), Dim("sym", "HW")],
        layout=rm,
    )

    ops.append(Op(op="reduce_sum", inputs=["X4"], output="sum", attrs={"dims": [2, 3], "keepdims": True}))
    tensors["sum"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)], layout=rm)
    ops.append(Op(op="div", inputs=["sum"], output="mean4", attrs={"divisor": "num_elements"}))
    tensors["mean4"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)], layout=rm)

    ops.append(Op(op="sub", inputs=["X4", "mean4"], output="diff", attrs={}))
    tensors["diff"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("sym", "group_size"), Dim("sym", "HW")], layout=rm)
    ops.append(Op(op="mul", inputs=["diff", "diff"], output="sq", attrs={}))
    tensors["sq"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("sym", "group_size"), Dim("sym", "HW")], layout=rm)
    ops.append(Op(op="reduce_sum", inputs=["sq"], output="var_sum", attrs={"dims": [2, 3], "keepdims": True}))
    tensors["var_sum"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)], layout=rm)
    ops.append(Op(op="div", inputs=["var_sum"], output="var4", attrs={"divisor": "num_elements"}))
    tensors["var4"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)], layout=rm)

    ops.append(Op(op="const", inputs=[], output="eps", attrs={"value": 1e-5, "dtype": "f32"}))
    tensors["eps"] = TensorType(dtype="f32", shape=[], layout=rm)
    ops.append(Op(op="add", inputs=["var4", "eps"], output="var_eps", attrs={}))
    tensors["var_eps"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)], layout=rm)
    ops.append(Op(op="rsqrt", inputs=["var_eps"], output="rstd4", attrs={}))
    tensors["rstd4"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("const", 1), Dim("const", 1)], layout=rm)

    ops.append(Op(op="mul", inputs=["diff", "rstd4"], output="xhat4", attrs={}))
    tensors["xhat4"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "num_groups"), Dim("sym", "group_size"), Dim("sym", "HW")], layout=rm)

    ops.append(Op(op="reshape", inputs=["xhat4"], output="xhat3", attrs={"shape": ["N", "C", "HW"]}))
    tensors["xhat3"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm)

    ops.append(Op(op="broadcast_in_dim", inputs=["W"], output="W3", attrs={"out_shape": ["N", "C", "HW"], "broadcast_dims": [1]}))
    tensors["W3"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm)
    ops.append(Op(op="broadcast_in_dim", inputs=["B"], output="B3", attrs={"out_shape": ["N", "C", "HW"], "broadcast_dims": [1]}))
    tensors["B3"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm)

    ops.append(Op(op="mul", inputs=["xhat3", "W3"], output="scaled", attrs={}))
    tensors["scaled"] = TensorType(dtype="f32", shape=[Dim("sym", "N"), Dim("sym", "C"), Dim("sym", "HW")], layout=rm)
    ops.append(Op(op="add", inputs=["scaled", "B3"], output="Y", attrs={}))

    # Mean/Rstd outputs: reshape [N,G,1,1] -> [N,G]
    ops.append(Op(op="reshape", inputs=["mean4"], output="Mean", attrs={"shape": ["N", "num_groups"]}))
    ops.append(Op(op="reshape", inputs=["rstd4"], output="Rstd", attrs={"shape": ["N", "num_groups"]}))

    schedule = ScheduleSketch(tile_m=None, tile_n=None, tile_k=None, vec_width=1, pipeline_depth=1)
    return IntentFunction(
        name="group_norm_kernel",
        tensors=tensors,
        ops=ops,
        outputs=["Y", "Mean", "Rstd"],
        schedule=schedule,
        axis_roles={"N": "batch", "C": "channel", "HW": "spatial", "num_groups": "channel"},
    )


def _softmax_inner_reference(case: TestCase) -> Dict[str, np.ndarray]:
    if case.inputs and "input_ptr" in case.inputs:
        x = np.asarray(case.inputs["input_ptr"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        m = int(case.shapes["M"])
        n = int(case.shapes["N"])
        x = rng.standard_normal((m, n), dtype=np.float32)
    x_max = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - x_max)
    y = e / np.sum(e, axis=1, keepdims=True)
    row_max = x_max.reshape(-1).astype(np.float32)
    row_sum = np.sum(e, axis=1).reshape(-1).astype(np.float32)
    return {
        "input_ptr": x,
        "output_ptr": y.astype(np.float32),
        "row_max_ptr": row_max,
        "row_sum_ptr": row_sum,
    }


def _layer_norm_persistent_reference(case: TestCase) -> Dict[str, np.ndarray]:
    m = int(case.shapes["M"])
    n = int(case.shapes["N"])
    if case.inputs and "in_ptr" in case.inputs:
        x = np.asarray(case.inputs["in_ptr"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        x = rng.standard_normal((m, n), dtype=np.float32)
    if case.inputs and "weight_ptr" in case.inputs:
        w = np.asarray(case.inputs["weight_ptr"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        w = rng.standard_normal((n,), dtype=np.float32)
    if case.inputs and "bias_ptr" in case.inputs:
        b = np.asarray(case.inputs["bias_ptr"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        b = rng.standard_normal((n,), dtype=np.float32)
    eps = np.float32(1e-5)
    mean = np.mean(x, axis=1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    y = (x - mean) * rstd * w[None, :] + b[None, :]
    return {
        "in_ptr": x,
        "weight_ptr": w,
        "bias_ptr": b,
        "out_ptr": y.astype(np.float32),
        "out_mean_ptr": mean.reshape(-1).astype(np.float32),
        "out_rstd_ptr": rstd.reshape(-1).astype(np.float32),
    }


def _attn_fwd_reference(case: TestCase) -> Dict[str, np.ndarray]:
    q_ctx = int(case.shapes.get("Q_CTX", 16))
    kv_ctx = int(case.shapes.get("KV_CTX", 16))
    head_dim = int(case.shapes.get("HEAD_DIM", 16))
    if case.inputs and "Q" in case.inputs:
        q = np.asarray(case.inputs["Q"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        q = rng.standard_normal((q_ctx, head_dim), dtype=np.float32)
    if case.inputs and "K" in case.inputs:
        k = np.asarray(case.inputs["K"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        k = rng.standard_normal((kv_ctx, head_dim), dtype=np.float32)
    if case.inputs and "V" in case.inputs:
        v = np.asarray(case.inputs["V"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        v = rng.standard_normal((kv_ctx, head_dim), dtype=np.float32)
    if case.inputs and "sm_scale" in case.inputs:
        sm_scale = np.asarray(case.inputs["sm_scale"], dtype=np.float32).reshape(())
    else:
        sm_scale = np.array(1.0 / np.sqrt(float(head_dim)), dtype=np.float32)

    scores = (q @ k.T) * sm_scale
    scores_max = np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores - scores_max)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    out = probs @ v
    return {"Q": q, "K": k, "V": v, "sm_scale": sm_scale, "Out": out.astype(np.float32)}


def _bicubic_reciprocal_scale(src_size: int, dst_size: int, align_corners: bool, scale: float | None) -> float:
    if align_corners:
        if dst_size > 1:
            return float(src_size - 1) / float(dst_size - 1)
        return 0.0
    if scale is not None and scale > 0:
        return 1.0 / float(scale)
    return float(src_size) / float(dst_size)


def _upsample_bicubic2d_aa_reference(case: TestCase) -> Dict[str, np.ndarray]:
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    n = int(case.shapes.get("N", 1))
    c = int(case.shapes.get("C", 1))
    ih = int(case.shapes.get("IH", 4))
    iw = int(case.shapes.get("IW", 4))
    oh = int(case.shapes.get("OH", 8))
    ow = int(case.shapes.get("OW", 8))

    if case.inputs and "I" in case.inputs:
        x = np.asarray(case.inputs["I"], dtype=np.float32)
    else:
        rng = np.random.default_rng(int(case.seed))
        x = rng.standard_normal((n, c, ih, iw), dtype=np.float32)

    align_corners = False
    reciprocal_scale_h = _bicubic_reciprocal_scale(ih, oh, align_corners, scale=None)
    reciprocal_scale_w = _bicubic_reciprocal_scale(iw, ow, align_corners, scale=None)

    xt = torch.from_numpy(x)
    yt = F.interpolate(xt, size=(oh, ow), mode="bicubic", align_corners=align_corners, antialias=True)
    y = yt.numpy().astype(np.float32)

    return {
        "I": x,
        "reciprocal_scale_h": np.array(reciprocal_scale_h, dtype=np.float32),
        "reciprocal_scale_w": np.array(reciprocal_scale_w, dtype=np.float32),
        "O": y,
    }


def _np_dtype(dt: str):
    s = str(dt)
    if s == "f16":
        return np.float16
    if s == "f32":
        return np.float32
    if s == "f64":
        return np.float64
    if s == "i32":
        return np.int32
    if s == "u8":
        return np.uint8
    if s == "bool":
        return np.bool_
    raise ValueError(f"unsupported dtype for numpy generator: {dt}")


def _random_array(rng: np.random.Generator, *, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    dt = _np_dtype(dtype)
    if dt in {np.float16, np.float32, np.float64}:
        return rng.standard_normal(shape, dtype=np.float32).astype(dt)
    if dt is np.int32:
        return rng.integers(-8, 9, size=shape, dtype=np.int32)
    if dt is np.uint8:
        return rng.integers(0, 256, size=shape, dtype=np.int32).astype(np.uint8)
    if dt is np.bool_:
        return rng.integers(0, 2, size=shape, dtype=np.int32).astype(np.bool_)
    raise ValueError(f"unsupported dtype for generator: {dtype}")


def _run_tilelang_exported_cuda_kernel(
    *,
    prim_func: Any,
    io_spec: Dict[str, Any],
    case: TestCase,
) -> Dict[str, np.ndarray]:
    """
    Reference runner: execute the real TileLang kernel (CUDA) for baseline IO.

    Inputs are generated from io_spec (buffer shapes); outputs are those written
    by the PrimFunc (inferred via TIR traversal).
    """
    bindings = dict(case.shapes)
    rng = np.random.default_rng(int(case.seed))
    written = set(infer_written_global_buffers(prim_func))

    tensors = io_spec.get("tensors") if isinstance(io_spec.get("tensors"), dict) else {}
    inputs_np: Dict[str, np.ndarray] = {}
    for name, tspec in tensors.items():
        if name in written:
            continue
        if not isinstance(tspec, dict):
            continue
        dt = str(tspec.get("dtype") or "f32")
        shape_tpl = tspec.get("shape") if isinstance(tspec.get("shape"), list) else None
        if not shape_tpl:
            raise RuntimeError(f"missing shape for TileLang buffer {name} in io_spec")
        shape = tuple(int(bindings[str(d)]) if isinstance(d, str) else int(d) for d in shape_tpl)
        inputs_np[name] = _random_array(rng, shape=shape, dtype=dt)

    return run_tilelang_kernel_io(prim_func, bindings=bindings, inputs_np=inputs_np)


def _run_tilelang_snapshot_cuda_kernel(
    *,
    name: str,
    semantic_io_spec: Dict[str, Any],
    launch: CudaLaunch,
    case: TestCase,
    output_names: List[str],
    inputs_np: Dict[str, np.ndarray],
    extra_bindings: Dict[str, int] | None = None,
) -> Dict[str, np.ndarray]:
    """
    Reference runner: execute the *saved* TileLang->CUDA snapshot via the CUDA runtime.

    This is the "real CUDA path" for our CUDA frontend: compile the `.cu` snapshot,
    launch it with explicit grid/block, and return numpy IO.
    """
    cuda_src_raw, _ptx, _entry, include_dirs = _load_tilelang_cuda_snapshot(name)

    # Some TileLang kernels (e.g. matmul) use dynamic shared memory (`extern __shared__`).
    # TileLang's runtime launches with an appropriate shared_mem size; replicate
    # that behavior here with a conservative default.
    if int(getattr(launch, "shared_mem", 0) or 0) <= 0 and "extern __shared__" in cuda_src_raw:
        try:
            smem = int(os.getenv("INTENTIR_CUDA_TILELANG_DYNAMIC_SMEM", "16384"))
        except Exception:
            smem = 16384
        launch = CudaLaunch(grid=launch.grid, block=launch.block, shared_mem=max(0, int(smem)))
    rt_io_spec = infer_runtime_io_spec(cuda_src=cuda_src_raw, kernel_name="main_kernel", semantic_io_spec=dict(semantic_io_spec))
    # Torch's CUDA extension build defines macros that disable half/bf16
    # conversions by default; TileLang templates rely on those conversions.
    cflags = [f"-I{p}" for p in include_dirs]
    cflags += [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
    ]

    bindings = dict(case.shapes)
    if extra_bindings:
        bindings.update({str(k): int(v) for k, v in extra_bindings.items()})

    io = run_cuda_kernel_io(
        kernel_name="main_kernel",
        cuda_src=cuda_src_raw,
        io_spec=rt_io_spec,
        launch=launch,
        bindings=bindings,
        inputs_np=inputs_np,
        output_names=list(output_names),
        extra_cuda_cflags=cflags,
    )

    # If the CUDA snapshot uses i8 for boolean buffers, convert to bool to match
    # semantic io_spec (and the numpy interpreter).
    sem_tensors = semantic_io_spec.get("tensors") if isinstance(semantic_io_spec.get("tensors"), dict) else {}
    rt_tensors = rt_io_spec.get("tensors") if isinstance(rt_io_spec.get("tensors"), dict) else {}
    for out_name in output_names:
        sem = sem_tensors.get(out_name)
        rt = rt_tensors.get(out_name)
        if not isinstance(sem, dict) or not isinstance(rt, dict):
            continue
        if str(sem.get("dtype")) == "bool" and str(rt.get("dtype")) == "i8" and out_name in io:
            io[out_name] = (np.asarray(io[out_name]).astype(np.int8) != 0)
    return io


@dataclass
class KernelSpec:
    name: str
    io_spec: Dict[str, Any]
    canonical_shapes: Dict[str, int]
    vary_axes: List[str]
    runner: Callable[[TestCase], Dict[str, np.ndarray]]
    block: tuple[int, int, int]
    # Stage C (metamorphic / bounded exhaustive / mutation-kill) often requires
    # varying tiny shapes. Some CUDA baselines are specialized (e.g. TileLang-
    # exported kernels with fixed inner dims), so Stage C may need a separate
    # pure-numpy reference runner.
    stage_c_runner: Callable[[TestCase], Dict[str, np.ndarray]] | None = None
    exclude_axes: Optional[List[str]] = None
    cuda_path: Path | None = None
    cuda_src: str | None = None
    include_dirs: List[Path] | None = None
    # Optional prebuilt PTX snapshot (e.g., from TileLang export).
    ptx_text: str | None = None
    ptx_entry: str | None = None
    ptx_origin: str | None = None
    # Optional TileLang origin (used to export CUDA/PTX and run baseline).
    tilelang_prim_func: Any | None = None


def _native_cuda_kernel_specs() -> List[KernelSpec]:
    return [
        KernelSpec(
            name="vec_add",
            cuda_path=VEC_ADD_CU_PATH,
            io_spec=dict(vec_add_io),
            canonical_shapes={"N": 1024},
            vary_axes=["N"],
            runner=_vec_add_reference,
            block=(256, 1, 1),
            exclude_axes=[],
        ),
        KernelSpec(
            name="transpose2d",
            cuda_path=TRANSPOSE2D_CU_PATH,
            io_spec=dict(transpose2d_io),
            canonical_shapes={"M": 128, "N": 256},
            vary_axes=["M", "N"],
            runner=_transpose2d_reference,
            block=(16, 16, 1),
            exclude_axes=[],
        ),
        KernelSpec(
            name="row_sum",
            cuda_path=ROW_SUM_CU_PATH,
            io_spec=dict(row_sum_io),
            canonical_shapes={"M": 256, "N": 256},
            vary_axes=["M", "N"],
            runner=_row_sum_reference,
            block=(256, 1, 1),
            exclude_axes=[],
        ),
        KernelSpec(
            name="naive_gemm",
            cuda_path=NAIVE_GEMM_CU_PATH,
            io_spec=dict(naive_gemm_io),
            canonical_shapes={"M": 64, "N": 64, "K": 64},
            vary_axes=["M", "N", "K"],
            runner=_naive_gemm_reference,
            block=(16, 16, 1),
            exclude_axes=[],
        ),
    ]


def _tilelang_export_regression_specs() -> List[KernelSpec]:
    """
    6-kernel regression suite, mirroring Triton/TileLang names.

    We treat TileLang as a *kernel generator*:
      PrimFunc --(TileLang compile)--> CUDA C + PTX
    then run the CUDA frontend pipeline on the exported artifacts.
    """
    threads = 128
    out: List[KernelSpec] = []

    # any_kernel_dim
    pf_any = make_any_kernel_dim_prim_func(n=16, threads=threads)
    io_any = {
        "arg_names": ["inp", "out", "M", "N"],
        "tensors": {
            "inp": {"dtype": "f32", "shape": ["M", "N"]},
            "out": {"dtype": "bool", "shape": ["M"]},
        },
        "scalars": {"M": "i32", "N": "i32"},
    }
    out.append(
        KernelSpec(
            name="any_kernel_dim",
            io_spec=io_any,
            canonical_shapes={"M": 16, "N": 16},
            vary_axes=["M"],
            runner=(
                lambda case, _io=io_any: _run_tilelang_snapshot_cuda_kernel(
                    name="any_kernel_dim",
                    semantic_io_spec=_io,
                    launch=CudaLaunch(grid=(int(case.shapes.get("M", 16)), 1, 1), block=(threads, 1, 1), shared_mem=0),
                    case=case,
                    output_names=["out"],
                    inputs_np={
                        "inp": (
                            np.asarray(case.inputs.get("inp"), dtype=np.float32)
                            if case.inputs and "inp" in case.inputs
                            else np.random.default_rng(int(case.seed))
                            .integers(0, 2, size=(int(case.shapes.get("M", 16)), int(case.shapes.get("N", 16))), dtype=np.int32)
                            .astype(np.float32)
                        )
                    },
                )
            ),
            stage_c_runner=_any_kernel_dim_reference,
            block=(threads, 1, 1),
            exclude_axes=[],
            tilelang_prim_func=pf_any,
            ptx_origin="tilelang",
        )
    )

    # group_norm_kernel
    pf_gn = make_group_norm_kernel_prim_func(c=64, hw=16, num_groups=4, threads=threads)
    io_gn = {
        "arg_names": ["X", "Y", "W", "B", "Mean", "Rstd", "N", "C", "HW", "num_groups", "group_size", "num_elements"],
        "tensors": {
            "X": {"dtype": "f32", "shape": ["N", "C", "HW"]},
            "Y": {"dtype": "f32", "shape": ["N", "C", "HW"]},
            "W": {"dtype": "f32", "shape": ["C"]},
            "B": {"dtype": "f32", "shape": ["C"]},
            "Mean": {"dtype": "f32", "shape": ["N", "num_groups"]},
            "Rstd": {"dtype": "f32", "shape": ["N", "num_groups"]},
        },
        "scalars": {"N": "i32", "C": "i32", "HW": "i32", "num_groups": "i32", "group_size": "i32", "num_elements": "i32"},
    }
    out.append(
        KernelSpec(
            name="group_norm_kernel",
            io_spec=io_gn,
            canonical_shapes={"N": 16, "C": 64, "HW": 16, "num_groups": 4},
            vary_axes=["N"],
            runner=(
                lambda case, _io=io_gn: (
                    lambda n, c, hw, g: _run_tilelang_snapshot_cuda_kernel(
                        name="group_norm_kernel",
                        semantic_io_spec=_io,
                        launch=CudaLaunch(grid=(n, g, 1), block=(threads, 1, 1), shared_mem=0),
                        case=case,
                        output_names=["Y", "Mean", "Rstd"],
                        inputs_np={
                            "X": (
                                np.asarray(case.inputs.get("X"), dtype=np.float32)
                                if case.inputs and "X" in case.inputs
                                else np.random.default_rng(int(case.seed)).standard_normal((n, c, hw), dtype=np.float32)
                            ),
                            "W": (
                                np.asarray(case.inputs.get("W"), dtype=np.float32)
                                if case.inputs and "W" in case.inputs
                                else np.random.default_rng(int(case.seed) + 1).standard_normal((c,), dtype=np.float32)
                            ),
                            "B": (
                                np.asarray(case.inputs.get("B"), dtype=np.float32)
                                if case.inputs and "B" in case.inputs
                                else np.random.default_rng(int(case.seed) + 2).standard_normal((c,), dtype=np.float32)
                            ),
                        },
                    )
                )(
                    int(case.shapes.get("N", 16)),
                    int(case.shapes.get("C", 64)),
                    int(case.shapes.get("HW", 16)),
                    int(case.shapes.get("num_groups", 4)),
                )
            ),
            stage_c_runner=_group_norm_reference,
            block=(threads, 1, 1),
            exclude_axes=["group_size", "num_elements"],
            tilelang_prim_func=pf_gn,
            ptx_origin="tilelang",
        )
    )

    # _attn_fwd
    pf_attn = make_attn_fwd_prim_func(q_ctx=16, kv_ctx=16, head_dim=16, threads=threads)
    io_attn = {
        "arg_names": ["Q", "K", "V", "Out", "sm_scale", "Q_CTX", "KV_CTX", "HEAD_DIM"],
        "tensors": {
            "Q": {"dtype": "f32", "shape": ["Q_CTX", "HEAD_DIM"]},
            "K": {"dtype": "f32", "shape": ["KV_CTX", "HEAD_DIM"]},
            "V": {"dtype": "f32", "shape": ["KV_CTX", "HEAD_DIM"]},
            "sm_scale": {"dtype": "f32", "shape": []},
            "Out": {"dtype": "f32", "shape": ["Q_CTX", "HEAD_DIM"]},
        },
        "scalars": {"Q_CTX": "i32", "KV_CTX": "i32", "HEAD_DIM": "i32"},
    }

    def _attn_runner(case: TestCase, _io=io_attn) -> Dict[str, np.ndarray]:
        q_ctx = int(case.shapes.get("Q_CTX", 16))
        kv_ctx = int(case.shapes.get("KV_CTX", 16))
        head_dim = int(case.shapes.get("HEAD_DIM", 16))
        rng = np.random.default_rng(int(case.seed))
        q = rng.standard_normal((q_ctx, head_dim), dtype=np.float32)
        k = rng.standard_normal((kv_ctx, head_dim), dtype=np.float32)
        v = rng.standard_normal((kv_ctx, head_dim), dtype=np.float32)
        io = _run_tilelang_snapshot_cuda_kernel(
            name="_attn_fwd",
            semantic_io_spec=_io,
            launch=CudaLaunch(grid=(q_ctx, 1, 1), block=(threads, 1, 1), shared_mem=0),
            case=case,
            output_names=["Out"],
            inputs_np={"Q": q, "K": k, "V": v},
        )
        io["sm_scale"] = np.array(1.0 / np.sqrt(float(head_dim)), dtype=np.float32)
        return io

    out.append(
        KernelSpec(
            name="_attn_fwd",
            io_spec=io_attn,
            canonical_shapes={"Q_CTX": 16, "KV_CTX": 16, "HEAD_DIM": 16},
            vary_axes=[],
            runner=_attn_runner,
            stage_c_runner=_attn_fwd_reference,
            block=(threads, 1, 1),
            exclude_axes=[],
            tilelang_prim_func=pf_attn,
            ptx_origin="tilelang",
        )
    )

    # softmax_inner
    pf_sm = make_softmax_inner_prim_func(n=16, threads=threads)
    io_sm = {
        "arg_names": ["output_ptr", "input_ptr", "row_max_ptr", "row_sum_ptr", "M", "N"],
        "tensors": {
            "input_ptr": {"dtype": "f32", "shape": ["M", "N"]},
            "output_ptr": {"dtype": "f32", "shape": ["M", "N"]},
            "row_max_ptr": {"dtype": "f32", "shape": ["M"]},
            "row_sum_ptr": {"dtype": "f32", "shape": ["M"]},
        },
        "scalars": {"M": "i32", "N": "i32"},
    }
    out.append(
        KernelSpec(
            name="softmax_inner",
            io_spec=io_sm,
            canonical_shapes={"M": 16, "N": 16},
            vary_axes=["M"],
            runner=(
                lambda case, _io=io_sm: _run_tilelang_snapshot_cuda_kernel(
                    name="softmax_inner",
                    semantic_io_spec=_io,
                    launch=CudaLaunch(grid=(int(case.shapes.get("M", 16)), 1, 1), block=(threads, 1, 1), shared_mem=0),
                    case=case,
                    output_names=["output_ptr", "row_max_ptr", "row_sum_ptr"],
                    inputs_np={
                        "input_ptr": (
                            np.asarray(case.inputs.get("input_ptr"), dtype=np.float32)
                            if case.inputs and "input_ptr" in case.inputs
                            else np.random.default_rng(int(case.seed))
                            .standard_normal((int(case.shapes.get("M", 16)), int(case.shapes.get("N", 16))), dtype=np.float32)
                        )
                    },
                )
            ),
            stage_c_runner=_softmax_inner_reference,
            block=(threads, 1, 1),
            exclude_axes=[],
            tilelang_prim_func=pf_sm,
            ptx_origin="tilelang",
        )
    )

    # layer_norm_persistent
    pf_ln = make_layer_norm_persistent_prim_func(n=16, threads=threads)
    io_ln = {
        "arg_names": ["in_ptr", "out_ptr", "weight_ptr", "bias_ptr", "out_mean_ptr", "out_rstd_ptr", "M", "N"],
        "tensors": {
            "in_ptr": {"dtype": "f32", "shape": ["M", "N"]},
            "out_ptr": {"dtype": "f32", "shape": ["M", "N"]},
            "weight_ptr": {"dtype": "f32", "shape": ["N"]},
            "bias_ptr": {"dtype": "f32", "shape": ["N"]},
            "out_mean_ptr": {"dtype": "f32", "shape": ["M"]},
            "out_rstd_ptr": {"dtype": "f32", "shape": ["M"]},
        },
        "scalars": {"M": "i32", "N": "i32"},
    }
    out.append(
        KernelSpec(
            name="layer_norm_persistent",
            io_spec=io_ln,
            canonical_shapes={"M": 16, "N": 16},
            vary_axes=["M"],
            runner=(
                lambda case, _io=io_ln: (
                    lambda m, n: _run_tilelang_snapshot_cuda_kernel(
                        name="layer_norm_persistent",
                        semantic_io_spec=_io,
                        launch=CudaLaunch(grid=(m, 1, 1), block=(threads, 1, 1), shared_mem=0),
                        case=case,
                        output_names=["out_ptr", "out_mean_ptr", "out_rstd_ptr"],
                        inputs_np={
                            "in_ptr": (
                                np.asarray(case.inputs.get("in_ptr"), dtype=np.float32)
                                if case.inputs and "in_ptr" in case.inputs
                                else np.random.default_rng(int(case.seed)).standard_normal((m, n), dtype=np.float32)
                            ),
                            "weight_ptr": (
                                np.asarray(case.inputs.get("weight_ptr"), dtype=np.float32)
                                if case.inputs and "weight_ptr" in case.inputs
                                else np.random.default_rng(int(case.seed) + 1).standard_normal((n,), dtype=np.float32)
                            ),
                            "bias_ptr": (
                                np.asarray(case.inputs.get("bias_ptr"), dtype=np.float32)
                                if case.inputs and "bias_ptr" in case.inputs
                                else np.random.default_rng(int(case.seed) + 2).standard_normal((n,), dtype=np.float32)
                            ),
                        },
                    )
                )(int(case.shapes.get("M", 16)), int(case.shapes.get("N", 16)))
            ),
            stage_c_runner=_layer_norm_persistent_reference,
            block=(threads, 1, 1),
            exclude_axes=[],
            tilelang_prim_func=pf_ln,
            ptx_origin="tilelang",
        )
    )
    # NOTE: TileLang's upsample PrimFunc is "evidence-only" (nearest-neighbor-like).
    # Keep the semantic diff baseline as the PyTorch bicubic reference, but still
    # export CUDA/PTX from the PrimFunc for anchors/accesses/contract.
    up_pf = make_upsample_bicubic2d_aa_prim_func(threads=threads)
    up_io_spec = {
        "arg_names": ["I", "O", "reciprocal_scale_h", "reciprocal_scale_w", "N", "C", "IH", "IW", "OH", "OW"],
        "tensors": {
            "I": {"dtype": "f32", "shape": ["N", "C", "IH", "IW"]},
            "reciprocal_scale_h": {"dtype": "f32", "shape": []},
            "reciprocal_scale_w": {"dtype": "f32", "shape": []},
            "O": {"dtype": "f32", "shape": ["N", "C", "OH", "OW"]},
        },
        "scalars": {},
    }
    out.append(
        KernelSpec(
            name="upsample_bicubic2d_aa",
            io_spec=up_io_spec,
            canonical_shapes={"N": 1, "C": 1, "IH": 4, "IW": 4, "OH": 8, "OW": 8},
            vary_axes=[],
            runner=_upsample_bicubic2d_aa_reference,
            stage_c_runner=_upsample_bicubic2d_aa_reference,
            block=(threads, 1, 1),
            exclude_axes=[],
            tilelang_prim_func=up_pf,
            ptx_origin="tilelang",
        )
    )
    return out


def _tilelang_export_coverage_specs() -> List[KernelSpec]:
    """
    P3: expanded CUDA coverage suite via TileLang->CUDA export.

    We intentionally reuse the same kernel names as Triton/TileLang coverage so:
      - `scripts/full_pipeline_verify.py --suite coverage` stays comparable
      - `scripts/pipeline_summary.py` can aggregate frontends consistently

    Note: Some kernels need structured input generation (e.g., gather indices);
    those get a custom runner instead of the generic io_spec-based generator.
    """
    threads = 128

    # Import lazily to keep CUDA users working without TileLang installed.
    from kernels.tilelang.ops.add2d import make_add2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.transpose2d import make_transpose2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.relu2d import make_relu2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.add_bias2d import make_add_bias2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.where2d import make_where2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.row_sum import make_row_sum_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.exp2d import make_exp2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.floor2d import make_floor2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.clamp2d import make_clamp2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.row_max import make_row_max_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.copy2d_divmod import make_copy2d_divmod_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.gather2d import make_gather2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.matmul_relu2d import make_matmul_relu2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.rms_norm2d import make_rms_norm2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.matmul_bias_relu2d import make_matmul_bias_relu2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.matmul_fused_epilogue2d import make_matmul_fused_epilogue2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.rowmask_where2d import make_rowmask_where2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.masked_softmax2d import make_masked_softmax2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.masked_attention2d import make_masked_attention2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.grouped_row_sum2d import make_grouped_row_sum2d_prim_func  # noqa: PLC0415
    from kernels.tilelang.ops.mlp2d import make_mlp2d_prim_func  # noqa: PLC0415

    def _mk_spec(
        *,
        name: str,
        prim_func: Any,
        canonical_shapes: Dict[str, int],
        vary_axes: List[str],
        exclude_axes: Optional[List[str]] = None,
        stage_c_runner: Callable[[TestCase], Dict[str, np.ndarray]] | None = None,
    ) -> KernelSpec:
        io_spec = build_io_spec_from_tilelang_prim_func(prim_func)
        written = set(infer_written_global_buffers(prim_func))

        def _run(case: TestCase, _pf=prim_func, _io=io_spec) -> Dict[str, np.ndarray]:
            bindings = dict(case.shapes)
            rng = np.random.default_rng(int(case.seed))
            tensors = _io.get("tensors") if isinstance(_io.get("tensors"), dict) else {}
            inputs_np: Dict[str, np.ndarray] = {}
            for tname, tspec in tensors.items():
                if tname in written:
                    continue
                if not isinstance(tspec, dict):
                    continue
                dt = str(tspec.get("dtype") or "f32")
                shape_tpl = tspec.get("shape") if isinstance(tspec.get("shape"), list) else None
                if not shape_tpl:
                    raise RuntimeError(f"missing shape for TileLang buffer {tname} in io_spec")
                shape = tuple(int(bindings[str(d)]) if isinstance(d, str) else int(d) for d in shape_tpl)
                inputs_np[str(tname)] = _random_array(rng, shape=shape, dtype=dt)

            grid = _infer_grid_from_tilelang_prim_func(_pf, bindings)
            launch = CudaLaunch(grid=grid, block=(threads, 1, 1), shared_mem=0)
            out_names = sorted(written)
            return _run_tilelang_snapshot_cuda_kernel(
                name=str(name),
                semantic_io_spec=_io,
                launch=launch,
                case=case,
                output_names=out_names,
                inputs_np=inputs_np,
            )

        run = _run
        return KernelSpec(
            name=str(name),
            io_spec=io_spec,
            canonical_shapes=dict(canonical_shapes),
            vary_axes=list(vary_axes),
            runner=run,
            stage_c_runner=stage_c_runner,
            block=(threads, 1, 1),
            exclude_axes=list(exclude_axes or []),
            tilelang_prim_func=prim_func,
            ptx_origin="tilelang",
        )

    out: List[KernelSpec] = []

    # Elementwise / simple shape ops.
    out.append(_mk_spec(name="add2d", prim_func=make_add2d_prim_func(n=16, threads=threads), canonical_shapes={"M": 16, "N": 16}, vary_axes=["M"]))
    out.append(
        _mk_spec(
            name="transpose2d",
            prim_func=make_transpose2d_prim_func(n=16, threads=threads),
            canonical_shapes={"M": 16, "N": 16},
            vary_axes=["M"],
        )
    )
    out.append(_mk_spec(name="relu2d", prim_func=make_relu2d_prim_func(n=16, threads=threads), canonical_shapes={"M": 16, "N": 16}, vary_axes=["M"]))
    out.append(
        _mk_spec(
            name="add_bias2d",
            prim_func=make_add_bias2d_prim_func(n=16, threads=threads),
            canonical_shapes={"M": 16, "N": 16},
            vary_axes=["M"],
        )
    )
    out.append(_mk_spec(name="where2d", prim_func=make_where2d_prim_func(n=16, threads=threads), canonical_shapes={"M": 16, "N": 16}, vary_axes=["M"]))

    # Reductions.
    out.append(_mk_spec(name="row_sum", prim_func=make_row_sum_prim_func(n=16, threads=threads), canonical_shapes={"M": 16, "N": 16}, vary_axes=["M"]))
    out.append(_mk_spec(name="row_max", prim_func=make_row_max_prim_func(n=64, threads=threads), canonical_shapes={"M": 16, "N": 64}, vary_axes=["M"]))

    # Unary math.
    out.append(_mk_spec(name="exp2d", prim_func=make_exp2d_prim_func(n=64, threads=threads), canonical_shapes={"M": 16, "N": 64}, vary_axes=["M"]))
    out.append(_mk_spec(name="floor2d", prim_func=make_floor2d_prim_func(n=64, threads=threads), canonical_shapes={"M": 16, "N": 64}, vary_axes=["M"]))
    out.append(_mk_spec(name="clamp2d", prim_func=make_clamp2d_prim_func(n=64, lo=-0.5, hi=0.5, threads=threads), canonical_shapes={"M": 16, "N": 64}, vary_axes=["M"]))

    # Copy/transpose indexing.
    out.append(
        _mk_spec(
            name="copy2d_divmod",
            prim_func=make_copy2d_divmod_prim_func(n=64, block_n=16, threads=threads),
            canonical_shapes={"M": 16, "N": 64},
            vary_axes=["M"],
        )
    )

    # Gather (requires in-bounds indices; use a custom runner).
    pf_gather = make_gather2d_prim_func(n=64, l=256, threads=threads)
    io_gather = build_io_spec_from_tilelang_prim_func(pf_gather)

    def _gather_runner(case: TestCase, _pf=pf_gather) -> Dict[str, np.ndarray]:
        M = int(case.shapes.get("M", 64))
        N = 64
        L = 256
        rng = np.random.default_rng(int(case.seed))
        inp = rng.standard_normal((M, N), dtype=np.float32)
        row_idx = rng.integers(0, max(1, M), size=(L,), dtype=np.int32)
        col_idx = rng.integers(0, N, size=(L,), dtype=np.int32)
        bindings = dict(case.shapes)
        grid = _infer_grid_from_tilelang_prim_func(_pf, bindings)
        launch = CudaLaunch(grid=grid, block=(threads, 1, 1), shared_mem=0)
        return _run_tilelang_snapshot_cuda_kernel(
            name="gather2d",
            semantic_io_spec=io_gather,
            launch=launch,
            case=case,
            output_names=["out"],
            inputs_np={"inp": inp, "row_idx": row_idx, "col_idx": col_idx},
        )

    out.append(
        KernelSpec(
            name="gather2d",
            io_spec=io_gather,
            canonical_shapes={"M": 64, "N": 64, "L": 256},
            vary_axes=["M"],
            runner=_gather_runner,
            stage_c_runner=_gather_runner,
            block=(threads, 1, 1),
            exclude_axes=[],
            tilelang_prim_func=pf_gather,
            ptx_origin="tilelang",
        )
    )

    # Matmul-ish / fused kernels.
    out.append(
        _mk_spec(
            name="matmul_relu2d",
            prim_func=make_matmul_relu2d_prim_func(block_m=32, block_n=32, block_k=16, num_stages=2, threads=threads),
            canonical_shapes={"M": 32, "N": 32, "K": 32},
            vary_axes=["M"],
        )
    )
    out.append(
        _mk_spec(
            name="rms_norm2d",
            prim_func=make_rms_norm2d_prim_func(n=64, eps=1e-5, threads=threads),
            canonical_shapes={"M": 16, "N": 64},
            vary_axes=["M"],
        )
    )
    out.append(
        _mk_spec(
            name="matmul_bias_relu2d",
            prim_func=make_matmul_bias_relu2d_prim_func(block_m=32, block_n=32, block_k=16, num_stages=2, threads=threads),
            canonical_shapes={"M": 32, "N": 32, "K": 32},
            vary_axes=["M"],
        )
    )
    out.append(
        _mk_spec(
            name="matmul_fused_epilogue2d",
            prim_func=make_matmul_fused_epilogue2d_prim_func(block_m=32, block_n=32, block_k=16, num_stages=2, threads=threads),
            canonical_shapes={"M": 32, "N": 32, "K": 32},
            vary_axes=["M"],
        )
    )
    out.append(
        _mk_spec(
            name="mlp2d",
            prim_func=make_mlp2d_prim_func(block_m=32, block_n=32, block_k=16, block_h=16, num_stages=2, threads=threads),
            canonical_shapes={"M": 32, "N": 32, "K": 32, "H": 32},
            vary_axes=["M"],
        )
    )

    # Mask/broadcast-heavy kernels.
    out.append(
        _mk_spec(
            name="rowmask_where2d",
            prim_func=make_rowmask_where2d_prim_func(n=64, threads=threads),
            canonical_shapes={"M": 16, "N": 64},
            vary_axes=["M"],
        )
    )
    out.append(
        _mk_spec(
            name="masked_softmax2d",
            prim_func=make_masked_softmax2d_prim_func(n=64, threads=threads),
            canonical_shapes={"M": 16, "N": 64},
            vary_axes=["M"],
        )
    )
    out.append(
        _mk_spec(
            name="masked_attention2d",
            prim_func=make_masked_attention2d_prim_func(q_ctx=16, kv_ctx=16, head_dim=16, threads=threads),
            canonical_shapes={"Q_CTX": 16, "KV_CTX": 16, "HEAD_DIM": 16},
            vary_axes=[],
        )
    )
    out.append(
        _mk_spec(
            name="grouped_row_sum2d",
            prim_func=make_grouped_row_sum2d_prim_func(n=64, group_size=4, threads=threads),
            canonical_shapes={"M": 16, "N": 64, "group_size": 4},
            vary_axes=["M"],
        )
    )

    return out


def default_kernel_specs() -> List[KernelSpec]:
    """
    CUDA smoke suite (cross-frontend parity):
      - Prefer the 6-kernel regression suite (TileLang-exported CUDA/PTX).
      - Fall back to a tiny native CUDA anchor set when TileLang is unavailable.
    """
    try:
        return regression_kernel_specs()
    except Exception:
        return native_kernel_specs()


def coverage_kernel_specs() -> List[KernelSpec]:
    """
    CUDA coverage suite:
      - Always include native CUDA anchors (stable PTX patterns for golden tests).
      - Include the 6-kernel regression suite when TileLang is available.
    """
    specs: List[KernelSpec] = []
    specs.extend(native_kernel_specs())
    try:
        specs.extend(regression_kernel_specs())
    except Exception:
        pass
    try:
        specs.extend(_tilelang_export_coverage_specs())
    except Exception:
        pass
    # De-dup by name while preserving first occurrence order.
    out: List[KernelSpec] = []
    seen: set[str] = set()
    for s in specs:
        if s.name in seen:
            continue
        seen.add(s.name)
        out.append(s)
    return out


def native_kernel_specs() -> List[KernelSpec]:
    """
    Tiny native CUDA anchors (kernel-only .cu sources under kernels/cuda/ops).

    These are intentionally small and stable, mainly used for:
      - CUDA PTX parsing golden tests
      - smoke-testing baseline CUDA runtime (torch extension + nvcc)
    """
    return list(_native_cuda_kernel_specs())


def regression_kernel_specs() -> List[KernelSpec]:
    """
    Mirror Triton/TileLang's 6-kernel regression suite via TileLang->CUDA export.
    """
    return list(_tilelang_export_regression_specs())


def run_pipeline_for_spec(
    spec: KernelSpec,
    *,
    out_dir: Path,
    cases_limit: int = 8,
    stage_c: bool = True,
    mutation_kill: bool = True,
    use_llm: bool = True,
    llm_model: Optional[str] = None,
) -> Dict[str, object]:
    report: Dict[str, object] = {"kernel": spec.name, "frontend": "cuda"}
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = pipeline_registry.get("cuda")

    # If this spec is TileLang-derived, export CUDA source + PTX once and attach
    # them to the KernelSpec for the CUDA adapter/ptx parser to consume.
    if spec.tilelang_prim_func is not None and (not spec.cuda_src or not spec.ptx_text):
        # Prefer a stable snapshot under `kernels/cuda/ops/` to keep the CUDA
        # prompt/cache deterministic across runs. Fall back to exporting from the
        # TileLang PrimFunc when missing (or when forced to refresh).
        refresh = str(os.getenv("INTENTIR_TILELANG_CUDA_SNAPSHOT_REFRESH", "0")).strip() == "1"
        used_snapshot = False
        cuda_src_raw = ""
        ptx_text = ""
        entry_name = ""
        include_dirs: List[Path] = []
        if not refresh:
            try:
                cuda_src_raw, ptx_text, entry_name, include_dirs = _load_tilelang_cuda_snapshot(spec.name)
                used_snapshot = True
            except Exception:
                used_snapshot = False
        if not used_snapshot:
            exp = export_tilelang_cuda(spec.tilelang_prim_func, out_dir=out_dir, stem=spec.name)
            _persist_tilelang_cuda_snapshot(spec.name, exp)
            cuda_src_raw = exp.cuda_src
            ptx_text = exp.ptx_text
            entry_name = exp.entry_name
            include_dirs = list(exp.include_dirs)
        report["tilelang_cuda_snapshot"] = {"used": bool(used_snapshot), "dir": str(_CUDA_TILELANG_SNAPSHOT_DIR)}
        # Keep the prompt "compiler-like": include a high-level TIR snapshot as a comment
        # before the exported CUDA kernel source. This significantly improves LLM
        # recovery for structured reductions/broadcasting (e.g., softmax/attention),
        # while still treating the kernel as a CUDA-source frontend.
        tir_txt = ""
        try:
            tir_txt = str(spec.tilelang_prim_func.script(show_meta=False))
        except Exception:
            try:
                tir_txt = str(spec.tilelang_prim_func)
            except Exception:
                tir_txt = ""
        if tir_txt.strip():
            spec.cuda_src = "// --- TileLang TIR (semantic guide; comment only) ---\n/*\n" + tir_txt + "\n*/\n\n" + cuda_src_raw
        else:
            spec.cuda_src = cuda_src_raw
        # Do not force `cuda_path` for TileLang-derived kernels: we want the
        # augmented `cuda_src` to reach the LLM prompt via KernelDescriptor.source_text.
        spec.ptx_text = ptx_text
        spec.ptx_entry = entry_name
        spec.include_dirs = list(include_dirs)
        if not spec.ptx_origin:
            spec.ptx_origin = "tilelang"

    print(f"[{spec.name}] stage1: build cuda descriptor", flush=True)
    desc = adapter.build_descriptor(spec)
    desc.meta["artifact_dir"] = str(out_dir)
    if spec.ptx_entry:
        desc.meta["ptx_entry"] = str(spec.ptx_entry)
    (out_dir / f"{spec.name}.cuda_src.cu").write_text(desc.source_text, encoding="utf-8")
    report["descriptor"] = desc.to_json_dict()

    print(f"[{spec.name}] stage2: compile CUDA -> PTX (artifacts)", flush=True)
    desc = adapter.ensure_artifacts(desc, spec)

    print(f"[{spec.name}] stage3: launch CUDA once (baseline)", flush=True)
    baseline_case = TestCase(shapes=dict(spec.canonical_shapes), dtypes={}, seed=0)
    npz_path = out_dir / f"{spec.name}.baseline.npz"
    meta_path = out_dir / f"{spec.name}.baseline.meta.json"
    reuse = str(os.getenv("INTENTIR_CUDA_REUSE_BASELINE_NPZ", "1")).strip().lower() not in {"0", "false", "no", "off"}
    src_hash = hashlib.sha256(str(desc.source_text).encode("utf-8")).hexdigest()[:16]
    baseline_io: Dict[str, np.ndarray] = {}
    reused = False
    if reuse and npz_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                isinstance(meta, dict)
                and meta.get("seed") == int(baseline_case.seed)
                and meta.get("shapes") == dict(baseline_case.shapes)
                and meta.get("src_hash") == src_hash
            ):
                data = np.load(npz_path, allow_pickle=False)
                baseline_io = {k: np.asarray(data[k]) for k in data.files}
                reused = True
        except Exception:
            reused = False

    if not reused:
        try:
            baseline_io = spec.runner(baseline_case)
        except Exception as e:
            # Wrap baseline errors to make user-facing scripts more actionable.
            if isinstance(e, CudaRuntimeError):
                raise RuntimeError(f"CUDA baseline failed: {e}") from e
            raise RuntimeError(f"CUDA baseline failed: {type(e).__name__}: {e}") from e
        np.savez(npz_path, **{k: np.asarray(v) for k, v in baseline_io.items()})
        try:
            meta_path.write_text(
                json.dumps({"seed": int(baseline_case.seed), "shapes": dict(baseline_case.shapes), "src_hash": src_hash}, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    report["baseline"] = {
        "shapes": dict(baseline_case.shapes),
        "seed": int(baseline_case.seed),
        "npz_path": str(npz_path.relative_to(ROOT)),
        "keys": sorted(list(baseline_io.keys())),
        "source": ("npz_cache" if reused else ("tilelang_cuda" if spec.tilelang_prim_func is not None else "cuda_runtime")),
    }

    print(f"[{spec.name}] stage4: Task4 facts/constraints/certificate", flush=True)
    facts = adapter.extract_facts(desc)
    constraints: FrontendConstraints = adapter.extract_constraints(desc, facts)
    cert_v2 = adapter.build_certificate(desc, facts, constraints)
    obligations = evaluate_obligations(desc, cert_v2)
    cert_v2.semantic_facts["obligations"] = [o.to_json_dict() for o in obligations]
    contract = evaluate_contract_v2(desc, cert_v2, obligations, constraints=constraints)
    try:
        cert_v2.meta = dict(getattr(cert_v2, "meta", {}) or {})
        cert_v2.meta["contract"] = {"level": str(contract.level), "reasons": list(contract.reasons), "assumptions": list(contract.assumptions)}
    except Exception:
        pass
    report["certificate_v2"] = cert_v2.to_json_dict()
    (out_dir / f"{spec.name}.certificate_v2.json").write_text(json.dumps(report["certificate_v2"], indent=2), encoding="utf-8")
    report["obligations"] = [o.to_json_dict() for o in obligations]
    report["contract"] = {"level": contract.level, "reasons": list(contract.reasons), "assumptions": list(contract.assumptions), "signals": dict(contract.signals)}
    (out_dir / f"{spec.name}.contract.json").write_text(json.dumps(report["contract"], indent=2), encoding="utf-8")

    print(f"[{spec.name}] stage5: LLM -> IntentIR (may take a while)", flush=True)
    llm_err: Exception | None = None
    cand: CandidateIntent | None = None
    if use_llm:
        try:
            cand = _LLM_HUB.lift(desc, model=llm_model)
        except Exception as e:
            llm_err = e
            cand = None
    if cand is None:
        # Resilience fallback: when CUDA kernels are TileLang-exported (our regression
        # suite), we can fall back to TileLang's deterministic intent builder so
        # downstream stages (diff/remote RVV) remain usable even if providers are
        # temporarily unavailable / quota-limited.
        fb_intent = None
        try:
            if spec.tilelang_prim_func is not None:
                from pipeline.tilelang.core import coverage_kernel_specs  # noqa: PLC0415

                for s in coverage_kernel_specs():
                    if getattr(s, "name", None) == spec.name:
                        fb_intent = s.intent_builder()
                        break
        except Exception:
            fb_intent = None
        if fb_intent is None:
            raise llm_err or RuntimeError("CUDA pipeline requires LLM (no deterministic fallback available)")
        cand = CandidateIntent(intent=fb_intent, problem_params={}, schedule_params={}, raw_json={"fallback": True}, llm_trace={})
        report["llm_fallback"] = {
            "used": True,
            "kind": "tilelang_deterministic",
            "reason": (f"{type(llm_err).__name__}: {llm_err}" if llm_err is not None else "use_llm disabled"),
        }

    enrich_intent_macros(cand.intent)
    _ensure_schedule_cuda(cand.intent, spec=spec)
    if cand.llm_trace:
        report["llm_trace"] = dict(cand.llm_trace or {})

    (out_dir / f"{spec.name}.intentir.mlir").write_text(print_mlir_like(cand.intent), encoding="utf-8")
    report["intent"] = cand.intent.to_json_dict()

    cand_expanded: CandidateIntent | None = None
    try:
        expanded_intent = expand_macros(cand.intent)
        cand_expanded = CandidateIntent(
            intent=expanded_intent,
            problem_params=dict(cand.problem_params),
            schedule_params=dict(cand.schedule_params),
            raw_json=dict(cand.raw_json),
            llm_trace=dict(cand.llm_trace),
        )
        _ensure_schedule_cuda(cand_expanded.intent, spec=spec)
        (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(print_mlir_like(expanded_intent), encoding="utf-8")
        report["intent_expanded"] = expanded_intent.to_json_dict()
    except Exception as e:
        report["intent_expanded"] = None
        report["intent_expand_error"] = f"{type(e).__name__}: {e}"

    print(f"[{spec.name}] stage6: Task4 static validation", flush=True)
    sv = static_validate((cand_expanded.intent if cand_expanded is not None else cand.intent), cert_v2)
    report["static_validation"] = {
        "ok": bool(sv.ok),
        "reasons": list(sv.reasons),
        "obligations": [{"id": o.id, "status": o.status, "detail": o.detail} for o in sv.obligations],
    }

    if use_llm and not sv.ok:
        # If this CUDA kernel was generated by TileLang (our regression suite), use a
        # deterministic intent builder on static-validate failure instead of making
        # an extra LLM call (important for request-limited providers).
        if spec.tilelang_prim_func is not None:
            fb_intent = None
            try:
                from pipeline.tilelang.core import coverage_kernel_specs  # noqa: PLC0415

                for s in coverage_kernel_specs():
                    if getattr(s, "name", None) == spec.name:
                        fb_intent = s.intent_builder()
                        break
            except Exception:
                fb_intent = None
            if fb_intent is not None:
                report.setdefault("intent_llm", report.get("intent"))
                report.setdefault("intent_expanded_llm", report.get("intent_expanded"))
                report["static_validation_llm"] = dict(report.get("static_validation") or {})
                cand = CandidateIntent(intent=fb_intent, problem_params={}, schedule_params={}, raw_json={"fallback": True}, llm_trace={})
                report["intent_fallback"] = {
                    "used": True,
                    "kind": "tilelang_deterministic",
                    "reason": "static_validation failed; use deterministic builder",
                    "llm_reasons": list(sv.reasons),
                }
                enrich_intent_macros(cand.intent)
                _ensure_schedule_cuda(cand.intent, spec=spec)
                (out_dir / f"{spec.name}.intentir.fallback.mlir").write_text(print_mlir_like(cand.intent), encoding="utf-8")
                try:
                    expanded_fix = expand_macros(cand.intent)
                    cand_expanded = CandidateIntent(
                        intent=expanded_fix,
                        problem_params=dict(cand.problem_params),
                        schedule_params=dict(cand.schedule_params),
                        raw_json=dict(cand.raw_json),
                        llm_trace=dict(cand.llm_trace),
                    )
                    _ensure_schedule_cuda(cand_expanded.intent, spec=spec)
                    (out_dir / f"{spec.name}.intentir.fallback.expanded.mlir").write_text(print_mlir_like(expanded_fix), encoding="utf-8")
                    report["intent"] = cand.intent.to_json_dict()
                    report["intent_expanded"] = expanded_fix.to_json_dict()
                    sv = static_validate((cand_expanded.intent if cand_expanded is not None else cand.intent), cert_v2)
                    report["static_validation"] = {
                        "ok": bool(sv.ok),
                        "reasons": list(sv.reasons),
                        "obligations": [{"id": o.id, "status": o.status, "detail": o.detail} for o in sv.obligations],
                    }
                except Exception as e:
                    report["intent_expand_error"] = f"{type(e).__name__}: {e}"
        else:
            # One conservative LLM repair round for native CUDA kernels.
            try:
                cand_fix = _LLM_HUB.lift(desc, feedback=list(sv.reasons), model=llm_model)
                enrich_intent_macros(cand_fix.intent)
                _ensure_schedule_cuda(cand_fix.intent, spec=spec)
                report["llm_trace"] = dict(cand_fix.llm_trace or {})
                cand = cand_fix
                (out_dir / f"{spec.name}.intentir.mlir").write_text(print_mlir_like(cand.intent), encoding="utf-8")
                report["intent"] = cand.intent.to_json_dict()
                expanded_fix = expand_macros(cand.intent)
                cand_expanded = CandidateIntent(
                    intent=expanded_fix,
                    problem_params=dict(cand.problem_params),
                    schedule_params=dict(cand.schedule_params),
                    raw_json=dict(cand.raw_json),
                    llm_trace=dict(cand.llm_trace),
                )
                _ensure_schedule_cuda(cand_expanded.intent, spec=spec)
                (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(print_mlir_like(expanded_fix), encoding="utf-8")
                report["intent_expanded"] = expanded_fix.to_json_dict()
                sv = static_validate((cand_expanded.intent if cand_expanded is not None else cand.intent), cert_v2)
                report["static_validation"] = {
                    "ok": bool(sv.ok),
                    "reasons": list(sv.reasons),
                    "obligations": [{"id": o.id, "status": o.status, "detail": o.detail} for o in sv.obligations],
                }
            except Exception:
                pass

    print(f"[{spec.name}] stage7: Task5 cases + diff", flush=True)
    cand_for_run = cand_expanded or cand
    # Stage B: generate cases from (IntentIR + contract assumptions) and diff.
    tolerances = infer_tolerances(cand_for_run.intent, ref_out=baseline_io).to_dict()
    tile_hints: List[int] = []
    try:
        th = (cert_v2.schedule_hints or {}).get("tile_hints")
        if isinstance(th, list):
            tile_hints = [int(x) for x in th if isinstance(x, (int, float)) and int(x) > 0]
    except Exception:
        tile_hints = []
    predicate_clauses: List[str] = []
    try:
        if isinstance(getattr(constraints, "meta", None), dict):
            pc = constraints.meta.get("predicate_clauses")
            if isinstance(pc, list):
                predicate_clauses = [str(x) for x in pc if isinstance(x, str) and x.strip()]
    except Exception:
        predicate_clauses = []

    extra_sizes: List[int] = []
    try:
        if isinstance(getattr(constraints, "meta", None), dict):
            si = constraints.meta.get("static_ints")
            if isinstance(si, list):
                for x in si:
                    try:
                        v = int(x)
                    except Exception:
                        continue
                    if 0 < v <= 2048:
                        extra_sizes.extend([v, max(1, v - 1), v + 1])
        sr = (cert_v2.schedule_hints or {}).get("symbol_ranges")
        if isinstance(sr, dict):
            for rr in sr.values():
                if not isinstance(rr, dict):
                    continue
                try:
                    end = int(rr.get("end"))
                except Exception:
                    continue
                if 0 < end <= 2048:
                    extra_sizes.extend([end, max(1, end - 1), end + 1])
        extra_sizes = sorted(set(int(v) for v in extra_sizes if int(v) > 0))
    except Exception:
        extra_sizes = []

    counterexample_models: List[Dict[str, int]] = []
    try:
        obs = (cert_v2.semantic_facts or {}).get("obligations")
        if isinstance(obs, list):
            for o in obs:
                if not isinstance(o, dict):
                    continue
                if o.get("id") != O3_MASK_IMPLIES_INBOUNDS:
                    continue
                wit = o.get("witness") if isinstance(o.get("witness"), dict) else {}
                for ac in (wit.get("access_checks") or []):
                    if not isinstance(ac, dict):
                        continue
                    for d in (ac.get("dims") or []):
                        if not isinstance(d, dict):
                            continue
                        cx = d.get("counterexample")
                        if not isinstance(cx, dict):
                            continue
                        assigns = cx.get("assignments")
                        if not isinstance(assigns, dict) or not assigns:
                            continue
                        model: Dict[str, int] = {}
                        for k, v in assigns.items():
                            if isinstance(k, str) and isinstance(v, (int, float)):
                                model[str(k)] = int(v)
                        if model:
                            counterexample_models.append(model)
    except Exception:
        counterexample_models = []

    cases = generate_cases_split(
        cand_for_run.intent,
        constraints=constraints,
        limit=int(cases_limit),
        seed=0,
        tile_hints=tile_hints,
        axes=list(spec.vary_axes),
        exclude_axes=list(spec.exclude_axes or []),
        assumptions=list(contract.assumptions),
        base_shapes=dict(spec.canonical_shapes),
        predicate_clauses=predicate_clauses,
        extra_sizes=extra_sizes,
        counterexample_models=counterexample_models,
    )
    report["cases"] = {
        "in_contract": [c.shapes for c in cases.in_contract],
        "out_of_contract": [c.shapes for c in cases.out_of_contract],
    }

    diffs_in, cex_in = run_diff(cand_for_run.intent, spec.runner, cases.in_contract, constraints=constraints, tolerances=tolerances)
    diffs_out, cex_out = run_diff(cand_for_run.intent, spec.runner, cases.out_of_contract, constraints=constraints, tolerances=tolerances)
    all_diffs = diffs_in + diffs_out
    diff_ok = all(d.ok for d in all_diffs) if all_diffs else True
    worst_abs = max((d.max_abs_err for d in all_diffs), default=0.0)
    worst_rel = max((d.max_rel_err for d in all_diffs), default=0.0)
    report["diff"] = {
        "ok": bool(diff_ok),
        "worst": {"summary": ("ok" if diff_ok else "mismatch"), "max_abs": float(worst_abs), "max_rel": float(worst_rel)},
        "results": [{"case_shapes": dict(cases.in_contract[i].shapes), "ok": bool(diffs_in[i].ok), "summary": diffs_in[i].summary, "max_abs": float(diffs_in[i].max_abs_err), "max_rel": float(diffs_in[i].max_rel_err)} for i in range(len(diffs_in))]
        + [{"case_shapes": dict(cases.out_of_contract[i].shapes), "ok": bool(diffs_out[i].ok), "summary": diffs_out[i].summary, "max_abs": float(diffs_out[i].max_abs_err), "max_rel": float(diffs_out[i].max_rel_err)} for i in range(len(diffs_out))],
    }

    # If dynamic diff fails, do one bounded LLM repair round using concrete feedback.
    # This is deliberately conservative (1 retry) to respect LLM rate limits.
    if use_llm and all_diffs and not diff_ok:
        worst_summary = None
        for d in all_diffs:
            if not d.ok:
                worst_summary = d.summary
                break
        cx0 = (cex_in + cex_out)[0] if (cex_in or cex_out) else None
        feedback3: List[str] = []
        feedback3.append(
            "Dynamic diff failed. Fix the IntentIR ops graph so it can execute in the interpreter and match the reference outputs."
        )
        if spec.name == "group_norm_kernel":
            feedback3 += [
                "For group_norm_kernel: keep original view X:[N,C,HW], W/B:[C].",
                "Compute in group-view with explicit reshape: X4=reshape(X,[N,num_groups,group_size,HW]).",
                "Mean/Rstd must be computed in group-view: mean4 and rstd4 shapes [N,num_groups,1,1].",
                "Do NOT broadcast Mean/Rstd from [N,num_groups] directly to [N,C,HW]. Instead subtract/multiply in group-view then reshape back.",
                "Mean = reshape(mean4,[N,num_groups]); Rstd = reshape(rstd4,[N,num_groups]).",
                "Use eps as const f32=1e-5 and model num_elements=group_size*HW via divisor/scale (do not use string constants).",
            ]
        if spec.name == "any_kernel_dim":
            feedback3 += [
                "For any_kernel_dim: out[M] = reduce_any(inp[M,N] != 0, axis=1).",
                "Use ne(inp, zero_const) then reduce_any over N, or directly reduce_any with a predicate; outputs must be produced by ops.",
            ]
        if worst_summary:
            feedback3.append(f"Observed diff failure: {worst_summary}")
        if cx0 is not None:
            feedback3.append(f"Counterexample shapes: {dict(cx0.case.shapes)}")

        try:
            cand_fix = _LLM_HUB.lift(desc, feedback=feedback3, model=llm_model)
            enrich_intent_macros(cand_fix.intent)
            _ensure_schedule_cuda(cand_fix.intent, spec=spec)
            report["llm_trace"] = dict(cand_fix.llm_trace or {})
            cand = cand_fix
            (out_dir / f"{spec.name}.intentir.mlir").write_text(print_mlir_like(cand.intent), encoding="utf-8")
            report["intent"] = cand.intent.to_json_dict()

            expanded_fix = expand_macros(cand.intent)
            cand_expanded = CandidateIntent(
                intent=expanded_fix,
                problem_params=dict(cand.problem_params),
                schedule_params=dict(cand.schedule_params),
                raw_json=dict(cand.raw_json),
                llm_trace=dict(cand.llm_trace),
            )
            _ensure_schedule_cuda(cand_expanded.intent, spec=spec)
            (out_dir / f"{spec.name}.intentir.expanded.mlir").write_text(print_mlir_like(expanded_fix), encoding="utf-8")
            report["intent_expanded"] = expanded_fix.to_json_dict()

            sv2 = static_validate(cand_expanded.intent, cert_v2)
            report["static_validation"] = {
                "ok": bool(sv2.ok),
                "reasons": list(sv2.reasons),
                "obligations": [{"id": o.id, "status": o.status, "detail": o.detail} for o in sv2.obligations],
            }

            cand_for_run = cand_expanded
            diffs_in, cex_in = run_diff(cand_for_run.intent, spec.runner, cases.in_contract, constraints=constraints, tolerances=tolerances)
            diffs_out, cex_out = run_diff(cand_for_run.intent, spec.runner, cases.out_of_contract, constraints=constraints, tolerances=tolerances)
            all_diffs = diffs_in + diffs_out
            diff_ok = all(d.ok for d in all_diffs) if all_diffs else True
            worst_abs = max((d.max_abs_err for d in all_diffs), default=0.0)
            worst_rel = max((d.max_rel_err for d in all_diffs), default=0.0)
            report["diff"] = {
                "ok": bool(diff_ok),
                "worst": {"summary": ("ok" if diff_ok else "mismatch"), "max_abs": float(worst_abs), "max_rel": float(worst_rel)},
                "results": [
                    {
                        "case_shapes": dict(cases.in_contract[i].shapes),
                        "ok": bool(diffs_in[i].ok),
                        "summary": diffs_in[i].summary,
                        "max_abs": float(diffs_in[i].max_abs_err),
                        "max_rel": float(diffs_in[i].max_rel_err),
                    }
                    for i in range(len(diffs_in))
                ]
                + [
                    {
                        "case_shapes": dict(cases.out_of_contract[i].shapes),
                        "ok": bool(diffs_out[i].ok),
                        "summary": diffs_out[i].summary,
                        "max_abs": float(diffs_out[i].max_abs_err),
                        "max_rel": float(diffs_out[i].max_rel_err),
                    }
                    for i in range(len(diffs_out))
                ],
            }
            report["diff_repair"] = {"attempted": True, "ok": bool(diff_ok)}
        except Exception as e:
            report["diff_repair"] = {"attempted": True, "ok": False, "error": f"{type(e).__name__}: {e}"}

    # Safety-net: if groupnorm still fails diff, fall back to a deterministic
    # compiler-style IntentIR so downstream (remote RVV/codegen) remains usable.
    if (not diff_ok) and spec.name == "group_norm_kernel":
        try:
            report.setdefault("intent_llm", report.get("intent"))
            report.setdefault("intent_expanded_llm", report.get("intent_expanded"))
            fb_intent = _group_norm_fallback_intent()
            _ensure_schedule_cuda(fb_intent, spec=spec)
            fb_exp = expand_macros(fb_intent)
            (out_dir / f"{spec.name}.intentir.fallback.mlir").write_text(print_mlir_like(fb_intent), encoding="utf-8")
            (out_dir / f"{spec.name}.intentir.fallback.expanded.mlir").write_text(print_mlir_like(fb_exp), encoding="utf-8")

            sv_fb = static_validate(fb_exp, cert_v2)
            report["static_validation_fallback"] = {
                "ok": bool(sv_fb.ok),
                "reasons": list(sv_fb.reasons),
                "obligations": [{"id": o.id, "status": o.status, "detail": o.detail} for o in sv_fb.obligations],
            }

            diffs_in, cex_in = run_diff(fb_exp, spec.runner, cases.in_contract, constraints=constraints, tolerances=tolerances)
            diffs_out, cex_out = run_diff(fb_exp, spec.runner, cases.out_of_contract, constraints=constraints, tolerances=tolerances)
            all_diffs = diffs_in + diffs_out
            diff_ok = all(d.ok for d in all_diffs) if all_diffs else True
            worst_abs = max((d.max_abs_err for d in all_diffs), default=0.0)
            worst_rel = max((d.max_rel_err for d in all_diffs), default=0.0)
            report["diff"] = {
                "ok": bool(diff_ok),
                "worst": {"summary": ("ok" if diff_ok else "mismatch"), "max_abs": float(worst_abs), "max_rel": float(worst_rel)},
                "results": [
                    {
                        "case_shapes": dict(cases.in_contract[i].shapes),
                        "ok": bool(diffs_in[i].ok),
                        "summary": diffs_in[i].summary,
                        "max_abs": float(diffs_in[i].max_abs_err),
                        "max_rel": float(diffs_in[i].max_rel_err),
                    }
                    for i in range(len(diffs_in))
                ]
                + [
                    {
                        "case_shapes": dict(cases.out_of_contract[i].shapes),
                        "ok": bool(diffs_out[i].ok),
                        "summary": diffs_out[i].summary,
                        "max_abs": float(diffs_out[i].max_abs_err),
                        "max_rel": float(diffs_out[i].max_rel_err),
                    }
                    for i in range(len(diffs_out))
                ],
            }
            if diff_ok:
                report["intent"] = fb_intent.to_json_dict()
                report["intent_expanded"] = fb_exp.to_json_dict()
                cand_for_run = CandidateIntent(intent=fb_exp, llm_trace={"provider": "fallback_group_norm"})
                report["intent_fallback"] = {"used": True, "kind": "group_norm_deterministic"}
        except Exception as e:
            report["intent_fallback"] = {"used": False, "error": f"{type(e).__name__}: {e}"}

    # Stage C: metamorphic + bounded exhaustive + mutation-kill (optional)
    stage_c_ref_fn = spec.stage_c_runner or spec.runner
    if stage_c and diff_ok and cases.in_contract:
        tol = dict(tolerances)
        base_case = cases.in_contract[0] if cases.in_contract else baseline_case
        meta = run_metamorphic_suite(
            spec.name, cand_for_run.intent, stage_c_ref_fn, base_case=base_case, atol=float(tol["atol"]), rtol=float(tol["rtol"])
        )
        bounded = run_bounded_exhaustive(
            spec.name, cand_for_run.intent, stage_c_ref_fn, atol=float(tol["atol"]), rtol=float(tol["rtol"]), max_cases=64
        )
        report["stage_c"] = {
            "metamorphic": {
                "ok": bool(meta.ok),
                "results": [{"relation": r.relation, "ok": bool(r.ok), "detail": r.detail} for r in meta.results],
            },
            "bounded_exhaustive": {
                "ok": bool(bounded.ok),
                "checked": int(bounded.checked),
                "total": int(bounded.total),
                "detail": bounded.detail,
                "first_failure": (dict(bounded.first_failure_case.shapes) if bounded.first_failure_case else None),
                "first_failure_summary": bounded.first_failure_summary,
            },
        }
    else:
        report["stage_c"] = {"skipped": True, "reason": ("diff_failed" if stage_c and not diff_ok else "disabled_or_no_cases")}

    if mutation_kill and diff_ok and cases.in_contract:
        tol = dict(tolerances)
        diff_cases = list(cases.in_contract[:2])
        metamorphic_base = cases.in_contract[0]
        mut = run_mutation_kill(
            spec.name,
            intent=cand_for_run.intent,
            run_ref_fn=stage_c_ref_fn,
            diff_cases=diff_cases,
            metamorphic_base_case=metamorphic_base,
            static_validate_fn=(lambda m, _cert=cert_v2: static_validate(m, _cert)),
            n_mutants=8,
            seed=0,
            atol=float(tol["atol"]),
            rtol=float(tol["rtol"]),
        )
        report["mutation_kill"] = {
            "kill_rate": float(mut.kill_rate),
            "total": int(mut.total),
            "killed": int(mut.killed),
            "survived": int(mut.survived),
            "killed_by_stage": dict(mut.killed_by_stage),
            "mutation_breakdown": dict(mut.mutation_breakdown),
            "outcomes": [
                {
                    "mutant_id": o.mutant_id,
                    "mutation_type": o.mutation_type,
                    "killed_by": o.killed_by,
                    "detail": o.detail,
                    "diff_summary": o.diff_summary,
                }
                for o in mut.outcomes
            ],
        }
    else:
        report["mutation_kill"] = {"skipped": True, "reason": ("diff_failed" if mutation_kill and not diff_ok else "disabled_or_no_cases")}

    return report


__all__ = [
    "KernelSpec",
    "native_kernel_specs",
    "regression_kernel_specs",
    "default_kernel_specs",
    "coverage_kernel_specs",
    "run_pipeline_for_spec",
]
