from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts" / "liger_true_migration"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends.cuda.pipeline.driver import lower_cuda_contract_to_kernel  # noqa: E402
from backends.cuda.runtime import CudaLaunch, load_cuda_ptx_module  # noqa: E402
from pipeline.triton.core import run_pipeline_for_spec  # noqa: E402
from pipeline.triton.providers.liger.specs import liger_kernel_specs  # noqa: E402
from scripts.cuda_backend_smoke import _with_io_aliases_for_names  # noqa: E402
from verify.gen_cases import TestCase  # noqa: E402


DEFAULT_KERNELS = [
    "liger_swiglu",
    "liger_rms_norm",
    "liger_fused_add_rms_norm",
    "liger_rope",
    "liger_cross_entropy",
]


def _env_flag(name: str) -> bool:
    raw = str(os.environ.get(str(name)) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _torch_dtype(dt: str) -> torch.dtype:
    raw = str(dt).strip().lower()
    if raw == "f16":
        return torch.float16
    if raw == "bf16":
        return torch.bfloat16
    if raw == "f32":
        return torch.float32
    if raw == "i64":
        return torch.int64
    if raw == "i32":
        return torch.int32
    if raw in {"bool", "i1"}:
        return torch.bool
    raise KeyError(f"unsupported dtype: {dt}")


def _parse_launch_dict(launch_raw: Any) -> CudaLaunch:
    launch = dict(launch_raw or {}) if isinstance(launch_raw, dict) else {}
    grid = launch.get("grid")
    block = launch.get("block")
    if not (isinstance(grid, list) and len(grid) == 3 and isinstance(block, list) and len(block) == 3):
        raise RuntimeError("invalid launch metadata")
    return CudaLaunch(
        grid=(int(grid[0]), int(grid[1]), int(grid[2])),
        block=(int(block[0]), int(block[1]), int(block[2])),
        shared_mem=int(launch.get("shared_mem", 0)),
    )


def _resolve_tensor_shape(shape_spec: list[Any], bindings: dict[str, Any]) -> tuple[int, ...]:
    dims: list[int] = []
    for dim in list(shape_spec or []):
        if isinstance(dim, int):
            dims.append(int(dim))
            continue
        key = str(dim).strip()
        if key in bindings:
            dims.append(int(bindings[key]))
            continue
        dims.append(int(key))
    return tuple(dims)


def _build_guided_tensors(*, io_spec: dict[str, Any], baseline: dict[str, np.ndarray], bindings: dict[str, Any], outputs: list[str]) -> tuple[list[Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    tensors = io_spec.get("tensors") if isinstance(io_spec.get("tensors"), dict) else {}
    scalars = io_spec.get("scalars") if isinstance(io_spec.get("scalars"), dict) else {}
    output_init = io_spec.get("output_init") if isinstance(io_spec.get("output_init"), dict) else {}
    arg_names = [str(x) for x in list(io_spec.get("arg_names") or [])]
    output_set = {str(x) for x in outputs}
    args: list[Any] = []
    inputs_torch: dict[str, torch.Tensor] = {}
    outputs_torch: dict[str, torch.Tensor] = {}

    def _make_output_tensor(shape: tuple[int, ...], dt: torch.dtype, init_spec: Any) -> torch.Tensor:
        if isinstance(init_spec, dict):
            op = str(init_spec.get("op") or "").strip().lower()
            if op == "fill" and "value" in init_spec:
                return torch.full(shape, float(init_spec.get("value") or 0.0), device="cuda", dtype=dt)
        elif isinstance(init_spec, (int, float)):
            return torch.full(shape, float(init_spec), device="cuda", dtype=dt)
        return torch.empty(shape, device="cuda", dtype=dt)

    def _tensor_alias_base(name: str) -> str:
        raw = str(name).strip()
        head, _sep, _tail = raw.partition("__")
        return head if head else raw

    def _descriptor_suffix(name: str, base_name: str) -> str:
        raw = str(name).strip()
        prefix = f"{base_name}__"
        return raw[len(prefix) :] if raw.startswith(prefix) else ""

    def _size_slot_count(base_name: str) -> int:
        prefix = f"{base_name}__size"
        return sum(1 for x in arg_names if str(x).startswith(prefix))

    def _stride_slot_count(base_name: str) -> int:
        prefix = f"{base_name}__stride"
        return sum(1 for x in arg_names if str(x).startswith(prefix))

    def _resolve_tensor(base_name: str, spec: dict[str, Any]) -> torch.Tensor:
        dt = _torch_dtype(str(spec.get("dtype") or "f32"))
        shape = _resolve_tensor_shape(list(spec.get("shape") or []), bindings)
        if base_name in outputs_torch:
            return outputs_torch[base_name]
        if base_name in inputs_torch:
            return inputs_torch[base_name]
        if base_name in output_set:
            t = _make_output_tensor(tuple(shape), dt, output_init.get(base_name))
            outputs_torch[base_name] = t
            return t
        if len(shape) == 0 and base_name in bindings:
            scalar_value = bindings[base_name]
            if dt in {torch.float16, torch.bfloat16, torch.float32}:
                t = torch.tensor(float(scalar_value), device="cuda", dtype=dt)
            else:
                t = torch.tensor(int(scalar_value), device="cuda", dtype=dt)
            inputs_torch[base_name] = t
            return t
        if base_name not in baseline:
            raise KeyError(f"missing baseline input for {base_name}")
        t = torch.as_tensor(np.asarray(baseline[base_name]), device="cuda", dtype=dt).contiguous()
        inputs_torch[base_name] = t
        return t

    def _descriptor_value(name: str, base_name: str, tensor: torch.Tensor) -> int:
        suffix = _descriptor_suffix(name, base_name)
        if suffix == "offset":
            return int(bindings.get(name, 0))
        if suffix.startswith("size"):
            idx = int(suffix[len("size") :])
            if name in bindings:
                return int(bindings[name])
            size_slots = _size_slot_count(base_name)
            if size_slots == 1:
                return int(tensor.numel()) if int(tensor.dim()) > 0 else 1
            shape = list(tensor.shape)
            if 0 <= idx < len(shape):
                return int(shape[idx])
            if not shape and idx == 0:
                return 1
            raise KeyError(f"cannot resolve {name} for tensor {base_name}")
        if suffix.startswith("stride"):
            idx = int(suffix[len("stride") :])
            if name in bindings:
                return int(bindings[name])
            stride_slots = _stride_slot_count(base_name)
            if stride_slots == 1:
                return 1
            strides = list(tensor.stride())
            if 0 <= idx < len(strides):
                return int(strides[idx])
            if not strides and idx == 0:
                return 1
            raise KeyError(f"cannot resolve {name} for tensor {base_name}")
        raise KeyError(f"unsupported tensor descriptor arg {name}")

    for name in arg_names:
        if name in tensors:
            spec = tensors[name] if isinstance(tensors.get(name), dict) else {}
            base_name = _tensor_alias_base(name)
            args.append(_resolve_tensor(base_name, spec))
        elif _tensor_alias_base(name) in tensors:
            base_name = _tensor_alias_base(name)
            spec = tensors[base_name] if isinstance(tensors.get(base_name), dict) else {}
            tensor = _resolve_tensor(base_name, spec)
            suffix = _descriptor_suffix(name, base_name)
            if suffix == "aligned":
                args.append(tensor)
            else:
                args.append(_descriptor_value(name, base_name, tensor))
        elif name in scalars:
            dt = str(scalars[name])
            if name not in bindings:
                raise KeyError(f"missing scalar binding for {name}")
            if dt == "f32":
                args.append(float(bindings[name]))
            else:
                args.append(int(bindings[name]))
        else:
            if name not in bindings:
                raise KeyError(f"missing binding for {name}")
            args.append(int(bindings[name]))
    return args, inputs_torch, outputs_torch


def _guided_postprocess_spec(io_spec: dict[str, Any]) -> dict[str, Any]:
    raw = io_spec.get("output_postprocess") if isinstance(io_spec.get("output_postprocess"), dict) else {}
    return {str(k): dict(v) for k, v in dict(raw or {}).items() if str(k).strip() and isinstance(v, dict)}


def _apply_guided_postprocess(
    *,
    io_spec: dict[str, Any],
    baseline: dict[str, np.ndarray],
    guided_outputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    out = {str(k): np.asarray(v) for k, v in dict(guided_outputs or {}).items()}
    for logical_name, spec in _guided_postprocess_spec(io_spec).items():
        op = str(spec.get("op") or "").strip().lower()
        if op == "row_mean":
            source = str(spec.get("source") or logical_name).strip()
            if source not in out:
                continue
            row_values = np.asarray(out[source], dtype=np.float32)
            out[logical_name] = np.asarray(np.mean(row_values, dtype=np.float32), dtype=np.float32)
            continue
        if op != "masked_row_mean":
            continue
        source = str(spec.get("source") or logical_name).strip()
        target_name = str(spec.get("target") or "").strip()
        ignore_name = str(spec.get("ignore_index") or "").strip()
        if source not in out or target_name not in baseline or ignore_name not in baseline:
            continue
        row_values = np.asarray(out[source], dtype=np.float32)
        target = np.asarray(baseline[target_name])
        ignore_index = np.asarray(baseline[ignore_name]).reshape(-1)
        ignore_value = int(ignore_index[0]) if ignore_index.size else 0
        valid_mask = np.asarray(target != ignore_value)
        denom = float(np.count_nonzero(valid_mask))
        if denom <= 0.0:
            denom = 1.0
        out[logical_name] = np.asarray(np.sum(row_values, dtype=np.float32) / denom, dtype=np.float32)
    return out


def _make_guided_postprocess_runner(
    *,
    io_spec: dict[str, Any],
    baseline: dict[str, np.ndarray],
    outputs_torch: dict[str, torch.Tensor],
) -> Any | None:
    spec = _guided_postprocess_spec(io_spec)
    if not spec:
        return None
    target_cache: dict[str, torch.Tensor] = {}
    for logical_name, cfg in spec.items():
        op = str(cfg.get("op") or "").strip().lower()
        if op == "row_mean":
            source = str(cfg.get("source") or logical_name).strip()
            if source not in outputs_torch:
                continue
            source_tensor = outputs_torch[source]

            def _runner(source_tensor=source_tensor):
                _ = torch.mean(source_tensor)

            return _runner
        if op != "masked_row_mean":
            continue
        source = str(cfg.get("source") or logical_name).strip()
        target_name = str(cfg.get("target") or "").strip()
        ignore_name = str(cfg.get("ignore_index") or "").strip()
        if source not in outputs_torch or target_name not in baseline or ignore_name not in baseline:
            continue
        target_cache[target_name] = torch.as_tensor(np.asarray(baseline[target_name]), device="cuda", dtype=torch.int64).contiguous()
        ignore_idx_arr = np.asarray(baseline[ignore_name]).reshape(-1)
        ignore_value = int(ignore_idx_arr[0]) if ignore_idx_arr.size else 0
        valid_count = max(1, int(np.count_nonzero(np.asarray(baseline[target_name]) != ignore_value)))
        denom = torch.tensor(float(valid_count), device="cuda", dtype=torch.float32)
        source_tensor = outputs_torch[source]

        def _runner(source_tensor=source_tensor, denom=denom):
            _ = torch.sum(source_tensor) / denom

        return _runner
    return None


def _make_guided_output_init_runner(*, io_spec: dict[str, Any], outputs_torch: dict[str, torch.Tensor]) -> Any | None:
    spec = io_spec.get("output_init") if isinstance(io_spec.get("output_init"), dict) else {}
    init_items: list[tuple[torch.Tensor, float]] = []
    for name, cfg in dict(spec or {}).items():
        tensor = outputs_torch.get(str(name))
        if tensor is None:
            continue
        if isinstance(cfg, dict):
            op = str(cfg.get("op") or "").strip().lower()
            if op == "fill" and "value" in cfg:
                init_items.append((tensor, float(cfg.get("value") or 0.0)))
        elif isinstance(cfg, (int, float)):
            init_items.append((tensor, float(cfg)))
    if not init_items:
        return None

    def _runner(init_items=tuple(init_items)):
        for tensor, value in init_items:
            tensor.fill_(float(value))

    return _runner


def _load_guided_realizations(report: dict[str, Any]) -> list[dict[str, Any]]:
    org = report.get("org") if isinstance(report.get("org"), dict) else {}
    compile_checks = list(org.get("compile_checks") or [])
    if not compile_checks:
        plan_path_raw = str(org.get("plan_path") or "").strip()
        if plan_path_raw:
            plan_path = Path(plan_path_raw)
            if plan_path.is_file():
                try:
                    plan_json = json.loads(plan_path.read_text(encoding="utf-8"))
                    compile_checks = list(plan_json.get("compile_checks") or [])
                except Exception:
                    compile_checks = []
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(compile_checks):
        if not isinstance(item, dict) or not bool(item.get("ok")):
            continue
        contract_path_raw = str(item.get("contract_path") or "").strip()
        if not contract_path_raw:
            continue
        contract_path = Path(contract_path_raw)
        if not contract_path.is_file():
            continue
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        try:
            lowered = lower_cuda_contract_to_kernel(
                dict(contract),
                shape_bindings=dict(((report.get("baseline") or {}).get("shapes") or {})),
            )
        except Exception:
            continue
        out.append(
            {
                "rank": idx,
                "candidate": str(item.get("candidate") or ""),
                "kernel_kind": str(item.get("kernel_kind") or ""),
                "bindings": dict(item.get("bindings") or {}),
                "contract_path": str(contract_path),
                "ptx_path": str(lowered.get("cuda_ptx_path") or (contract.get("executable") or {}).get("path") or item.get("ptx_path") or ""),
                "entry": str(lowered.get("kernel_name") or (contract.get("executable") or {}).get("entry") or item.get("entry") or ""),
                "io_spec": dict(lowered.get("io_spec") or {}),
                "shape_bindings": dict(lowered.get("bindings") or {}),
                "output_names": [str(x) for x in list(lowered.get("output_names") or []) if str(x).strip()],
                "launch": _parse_launch_dict(lowered.get("launch")),
                "ptx_text": lowered.get("cuda_ptx") or "",
            }
        )
    return out


def _bench_module_launch(
    *,
    compiled_module: Any,
    args: list[Any],
    launch: CudaLaunch,
    warmup: int,
    iters: int,
    repeats: int,
    prelaunch_runner: Any | None = None,
    postprocess_runner: Any | None = None,
) -> tuple[float, list[float]]:
    launch_args = [*args, int(launch.grid[0]), int(launch.grid[1]), int(launch.grid[2]), int(launch.block[0]), int(launch.block[1]), int(launch.block[2]), int(launch.shared_mem)]
    for _ in range(int(warmup)):
        if prelaunch_runner is not None:
            prelaunch_runner()
        compiled_module.launch(*launch_args)
        if postprocess_runner is not None:
            postprocess_runner()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            for _ in range(int(iters)):
                if prelaunch_runner is not None:
                    prelaunch_runner()
                compiled_module.launch(*launch_args)
                if postprocess_runner is not None:
                    postprocess_runner()
        torch.cuda.synchronize()
    except Exception:
        g = None
    times_ns: list[float] = []
    for _ in range(max(1, int(repeats))):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if g is not None:
            g.replay()
        else:
            for _ in range(int(iters)):
                if prelaunch_runner is not None:
                    prelaunch_runner()
                compiled_module.launch(*launch_args)
                if postprocess_runner is not None:
                    postprocess_runner()
        end.record()
        torch.cuda.synchronize()
        times_ns.append(float(start.elapsed_time(end)) * 1.0e6 / float(iters))
    return float(statistics.median(times_ns)), times_ns


def _bench_native(fn, *, warmup: int, iters: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(int(warmup)):
        fn()
    torch.cuda.synchronize()
    graphs: list[torch.cuda.CUDAGraph] = []
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(int(iters)):
                fn()
        torch.cuda.synchronize()
        graphs.append(g)
    except Exception:
        graphs = []
    times_ns: list[float] = []
    for _ in range(max(1, int(repeats))):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if graphs:
            graphs[0].replay()
        else:
            for _ in range(int(iters)):
                fn()
        end.record()
        torch.cuda.synchronize()
        times_ns.append(float(start.elapsed_time(end)) * 1.0e6 / float(iters))
    return float(statistics.median(times_ns)), times_ns


def _qps(ns_per_iter: float) -> float:
    return float(1.0e9 / float(ns_per_iter)) if float(ns_per_iter) > 0.0 else 0.0


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _pick_io_value(io: dict[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        key = str(name).strip()
        if key and key in io:
            return np.asarray(io[key])
    raise KeyError(str(names[0] if names else ""))


def _native_callable(kernel: str, baseline: dict[str, np.ndarray]):
    from pipeline.triton.providers.liger.specs import (
        cross_entropy_forward,
        geglu_forward,
        fused_add_rms_norm_forward,
        fused_linear_cross_entropy_forward,
        fused_linear_jsd_forward,
        fused_neighborhood_attention_forward,
        jsd_forward,
        kldiv_forward_triton,
        liger_fused_neighborhood_attention,
        liger_fused_linear_cross_entropy,
        liger_fused_linear_jsd,
        liger_dyt_fwd,
        liger_mhc_forward,
        liger_multi_token_attention,
        liger_poly_norm,
        liger_tvd,
        group_norm_forward,
        layer_norm_forward,
        llama4_rope_forward,
        LigerTiledGEGLUMLP,
        qwen2vl_mrope_forward,
        rms_norm_forward,
        rope_forward,
        triton_grpo_loss,
        _sparsemax_forward,
        _softmax_forward,
        swiglu_forward,
    )

    if kernel == "liger_swiglu":
        a = torch.as_tensor(np.asarray(baseline["a"]), device="cuda", dtype=torch.float32).contiguous()
        b = torch.as_tensor(np.asarray(baseline["b"]), device="cuda", dtype=torch.float32).contiguous()

        def _fn():
            swiglu_forward(a, b)

        return _fn
    if kernel == "liger_rms_norm":
        x = torch.as_tensor(np.asarray(baseline["X"]), device="cuda", dtype=torch.float32).contiguous()
        w = torch.as_tensor(np.asarray(baseline["W"]), device="cuda", dtype=torch.float32).contiguous()

        def _fn():
            rms_norm_forward(x, w, 1.0e-5, 0.0, "none", False)

        return _fn
    if kernel == "liger_fused_add_rms_norm":
        x = torch.as_tensor(np.asarray(baseline["X"]), device="cuda", dtype=torch.float32).contiguous()
        r = torch.as_tensor(np.asarray(baseline["R"]), device="cuda", dtype=torch.float32).contiguous()
        w = torch.as_tensor(np.asarray(baseline["W"]), device="cuda", dtype=torch.float32).contiguous()

        def _fn():
            fused_add_rms_norm_forward(x, r, w, 1.0e-5, 0.0, "none")

        return _fn
    if kernel == "liger_rope":
        q = torch.as_tensor(np.asarray(baseline["q"]), device="cuda", dtype=torch.float32).contiguous()
        k = torch.as_tensor(np.asarray(baseline["k"]), device="cuda", dtype=torch.float32).contiguous()
        cos = torch.as_tensor(np.asarray(baseline["cos"]), device="cuda", dtype=torch.float32).contiguous()
        sin = torch.as_tensor(np.asarray(baseline["sin"]), device="cuda", dtype=torch.float32).contiguous()

        def _fn():
            rope_forward(q, k, cos, sin)

        return _fn
    if kernel == "liger_cross_entropy":
        x = torch.as_tensor(np.asarray(baseline["input"]), device="cuda", dtype=torch.float32).contiguous()
        target = torch.as_tensor(np.asarray(baseline["target"]), device="cuda", dtype=torch.int64).contiguous()

        def _fn():
            cross_entropy_forward(x, target, None, -100, 0.0, 0.0, "mean", None, False, return_token_accuracy=False, return_predicted_tokens=False)

        return _fn
    if kernel == "liger_geglu":
        a = torch.as_tensor(np.asarray(baseline["a"]), device="cuda", dtype=torch.float32).contiguous()
        b = torch.as_tensor(np.asarray(baseline["b"]), device="cuda", dtype=torch.float32).contiguous()

        def _fn():
            geglu_forward(a, b)

        return _fn
    if kernel == "liger_group_norm":
        x = torch.as_tensor(np.asarray(baseline["X"]), device="cuda", dtype=torch.float32).contiguous()
        w = torch.as_tensor(np.asarray(baseline["W"]), device="cuda", dtype=torch.float32).contiguous()
        b = torch.as_tensor(np.asarray(baseline["B"]), device="cuda", dtype=torch.float32).contiguous()
        num_groups = int(np.asarray(baseline["num_groups"]).reshape(()))

        def _fn():
            group_norm_forward(x, int(x.shape[1]), num_groups, w, b, 1.0e-5)

        return _fn
    if kernel == "liger_dyt":
        x = torch.as_tensor(np.asarray(baseline["X"]), device="cuda", dtype=torch.float32).contiguous()
        alpha = torch.as_tensor(np.asarray(baseline["Alpha"]).reshape(1), device="cuda", dtype=torch.float32).contiguous()
        gamma = torch.as_tensor(np.asarray(baseline["Gamma"]), device="cuda", dtype=torch.float32).contiguous()
        beta = torch.as_tensor(np.asarray(baseline["Beta"]), device="cuda", dtype=torch.float32).contiguous()

        def _fn():
            liger_dyt_fwd(x, alpha, gamma, beta)

        return _fn
    if kernel == "liger_layer_norm":
        x = torch.as_tensor(np.asarray(baseline["X"]), device="cuda", dtype=torch.float32).contiguous()
        w = torch.as_tensor(np.asarray(baseline["W"]), device="cuda", dtype=torch.float32).contiguous()
        b = torch.as_tensor(np.asarray(baseline["B"]), device="cuda", dtype=torch.float32).contiguous()

        def _fn():
            layer_norm_forward(x, w, b, 1.0e-5)

        return _fn
    if kernel == "liger_softmax":
        x = torch.as_tensor(np.asarray(baseline["X"]), device="cuda", dtype=torch.float32).contiguous()

        def _fn():
            _softmax_forward(x)

        return _fn
    if kernel == "liger_qwen2vl_mrope":
        q = torch.as_tensor(np.asarray(baseline["q"]), device="cuda", dtype=torch.float32).contiguous()
        k = torch.as_tensor(np.asarray(baseline["k"]), device="cuda", dtype=torch.float32).contiguous()
        cos = torch.as_tensor(np.asarray(baseline["cos"]), device="cuda", dtype=torch.float32).contiguous()
        sin = torch.as_tensor(np.asarray(baseline["sin"]), device="cuda", dtype=torch.float32).contiguous()
        mrope_section = [int(x) for x in np.asarray(baseline["mrope_section"]).reshape(-1).tolist()]

        def _fn():
            qwen2vl_mrope_forward(q, k, cos, sin, mrope_section)

        return _fn
    if kernel == "liger_sparsemax":
        x = torch.as_tensor(np.asarray(baseline["X"]), device="cuda", dtype=torch.float32).contiguous()

        def _fn():
            _sparsemax_forward(x, -1)

        return _fn
    if kernel == "liger_kl_div":
        x = torch.as_tensor(np.asarray(baseline["input"]), device="cuda", dtype=torch.float32).contiguous()
        y = torch.as_tensor(np.asarray(baseline["target"]), device="cuda", dtype=torch.float32).contiguous()

        def _fn():
            kldiv_forward_triton(x, y, False, "batchmean", 1.0e-10)

        return _fn
    if kernel == "liger_jsd":
        x = torch.as_tensor(np.asarray(baseline["input"]), device="cuda", dtype=torch.float32).contiguous()
        y = torch.as_tensor(np.asarray(baseline["target"]), device="cuda", dtype=torch.float32).contiguous()
        labels = torch.as_tensor(np.asarray(baseline["shift_labels"]), device="cuda", dtype=torch.int64).contiguous()

        def _fn():
            jsd_forward(x, y, labels, 0.5, -100, True)

        return _fn
    if kernel == "liger_fused_linear_cross_entropy":
        x = torch.as_tensor(np.asarray(baseline["input"]), device="cuda", dtype=torch.float32).contiguous()
        w = torch.as_tensor(np.asarray(baseline["weight"]), device="cuda", dtype=torch.float32).contiguous()
        target = torch.as_tensor(np.asarray(baseline["target"]), device="cuda", dtype=torch.int64).contiguous()

        def _fn():
            liger_fused_linear_cross_entropy(x, w, target, None, None, -100, 0.0, 0.0, "mean", None, False)

        return _fn
    if kernel == "liger_fused_linear_jsd":
        student_input = torch.as_tensor(np.asarray(baseline["student_input"]), device="cuda", dtype=torch.float32).contiguous()
        student_weight = torch.as_tensor(np.asarray(baseline["student_weight"]), device="cuda", dtype=torch.float32).contiguous()
        teacher_input = torch.as_tensor(np.asarray(baseline["teacher_input"]), device="cuda", dtype=torch.float32).contiguous()
        teacher_weight = torch.as_tensor(np.asarray(baseline["teacher_weight"]), device="cuda", dtype=torch.float32).contiguous()
        labels = torch.as_tensor(np.asarray(baseline["shift_labels"]), device="cuda", dtype=torch.int64).contiguous()

        def _fn():
            liger_fused_linear_jsd(student_input, student_weight, teacher_input, teacher_weight, labels, 0.5, -100, 1.0)

        return _fn
    if kernel == "liger_fused_neighborhood_attention":
        query = torch.as_tensor(np.asarray(baseline["query"]), device="cuda", dtype=torch.float32).contiguous()
        key = torch.as_tensor(np.asarray(baseline["key"]), device="cuda", dtype=torch.float32).contiguous()
        value = torch.as_tensor(np.asarray(baseline["value"]), device="cuda", dtype=torch.float32).contiguous()
        kernel_size = int(np.asarray(baseline["kernel_size"]).reshape(()))
        dilation = int(np.asarray(baseline["dilation"]).reshape(()))

        def _fn():
            liger_fused_neighborhood_attention(query, key, value, kernel_size=kernel_size, dilation=dilation, scale=None)

        return _fn
    if kernel == "liger_grpo_loss":
        logits = torch.as_tensor(np.asarray(baseline["logits"]), device="cuda", dtype=torch.float32).contiguous()
        old_logp = torch.as_tensor(np.asarray(baseline["old_logp"]), device="cuda", dtype=torch.float32).contiguous()
        ref_logp = torch.as_tensor(np.asarray(baseline["ref_logp"]), device="cuda", dtype=torch.float32).contiguous()
        completion_ids = torch.as_tensor(np.asarray(baseline["completion_ids"]), device="cuda", dtype=torch.int64).contiguous()
        advantages = torch.as_tensor(np.asarray(baseline["advantages"]), device="cuda", dtype=torch.float32).contiguous()
        completion_mask = torch.as_tensor(np.asarray(baseline["completion_mask"]), device="cuda", dtype=torch.float32).contiguous()

        def _fn():
            triton_grpo_loss(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, reduce=True)

        return _fn
    if kernel == "liger_llama4_rope":
        def _half_split_to_interleaved(arr: np.ndarray) -> np.ndarray:
            x = np.asarray(arr, dtype=np.float32)
            half = int(x.shape[-1]) // 2
            real = x[..., :half]
            imag = x[..., half:]
            out = np.empty(x.shape[:-1] + (half * 2,), dtype=np.float32)
            out[..., 0::2] = real
            out[..., 1::2] = imag
            return out

        q_np = np.asarray(baseline["q"])
        k_np = np.asarray(baseline["k"])
        if "freqs_cis" in baseline:
            freqs_cis = torch.as_tensor(np.asarray(baseline["freqs_cis"]), device="cuda", dtype=torch.complex64).contiguous()
        else:
            cos = np.asarray(baseline["cos"], dtype=np.float32)
            sin = np.asarray(baseline["sin"], dtype=np.float32)
            freqs_cis = torch.complex(
                torch.as_tensor(cos, device="cuda", dtype=torch.float32).contiguous(),
                torch.as_tensor(sin, device="cuda", dtype=torch.float32).contiguous(),
            )
        q = torch.as_tensor(_half_split_to_interleaved(q_np), device="cuda", dtype=torch.float32).contiguous()
        k = torch.as_tensor(_half_split_to_interleaved(k_np), device="cuda", dtype=torch.float32).contiguous()

        def _fn():
            llama4_rope_forward(q, k, freqs_cis)

        return _fn
    if kernel == "liger_mhc":
        x = torch.as_tensor(np.asarray(baseline["X"]), device="cuda", dtype=torch.float32).contiguous()
        phi = torch.as_tensor(np.asarray(baseline["Phi"]), device="cuda", dtype=torch.float32).contiguous()
        bias = torch.as_tensor(np.asarray(baseline["B"]), device="cuda", dtype=torch.float32).contiguous()
        alpha_pre = torch.as_tensor(np.asarray(baseline["AlphaPre"]).reshape(()), device="cuda", dtype=torch.float32)
        alpha_post = torch.as_tensor(np.asarray(baseline["AlphaPost"]).reshape(()), device="cuda", dtype=torch.float32)
        alpha_res = torch.as_tensor(np.asarray(baseline["AlphaRes"]).reshape(()), device="cuda", dtype=torch.float32)
        layer_w = torch.as_tensor(np.asarray(baseline["LayerW"]), device="cuda", dtype=torch.float32).contiguous()
        layer = torch.nn.Linear(int(layer_w.shape[1]), int(layer_w.shape[0]), bias=False, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            layer.weight.copy_(layer_w)

        def _fn():
            liger_mhc_forward(x, layer, phi, bias, alpha_pre, alpha_post, alpha_res, allow_fp32=True, tmax=8)

        return _fn
    if kernel == "liger_multi_token_attention":
        scores = torch.as_tensor(np.asarray(baseline["scores"]), device="cuda", dtype=torch.float32).contiguous()
        weight = torch.as_tensor(np.asarray(baseline["weight"]), device="cuda", dtype=torch.float32).contiguous()
        bias = torch.as_tensor(np.asarray(baseline["bias"]), device="cuda", dtype=torch.float32).contiguous()
        groups = int(np.asarray(baseline["groups"]).reshape(()))
        kernel_size = int(np.asarray(baseline["kernel_size"]).reshape(()))

        def _fn():
            liger_multi_token_attention(scores, weight, bias, 1, kernel_size // 2, 1, groups, False)

        return _fn
    if kernel == "liger_poly_norm":
        x = torch.as_tensor(np.asarray(baseline["X"]), device="cuda", dtype=torch.float32).contiguous()
        w = torch.as_tensor(np.asarray(baseline["W"]), device="cuda", dtype=torch.float32).contiguous()
        b = torch.as_tensor(np.asarray(baseline["B"]).reshape(()), device="cuda", dtype=torch.float32)

        def _fn():
            liger_poly_norm(x, w, b, 1.0e-6, True)

        return _fn
    if kernel == "liger_tiled_mlp":
        x = torch.as_tensor(np.asarray(baseline["X"]), device="cuda", dtype=torch.float32).contiguous()
        gate_w = torch.as_tensor(np.asarray(baseline["GateW"]), device="cuda", dtype=torch.float32).contiguous()
        up_w = torch.as_tensor(np.asarray(baseline["UpW"]), device="cuda", dtype=torch.float32).contiguous()
        down_w = torch.as_tensor(np.asarray(baseline["DownW"]), device="cuda", dtype=torch.float32).contiguous()
        num_shards = int(np.asarray(baseline["num_shards"]).reshape(()))
        cfg = SimpleNamespace(hidden_size=int(gate_w.shape[0]), intermediate_size=int(gate_w.shape[1]), hidden_act="gelu_pytorch_tanh")
        mlp = LigerTiledGEGLUMLP(config=cfg, num_shards=num_shards).to("cuda").to(torch.float32)
        with torch.no_grad():
            mlp.gate_proj.weight.copy_(gate_w.transpose(0, 1).contiguous())
            mlp.up_proj.weight.copy_(up_w.transpose(0, 1).contiguous())
            mlp.down_proj.weight.copy_(down_w.transpose(0, 1).contiguous())

        def _fn():
            mlp(x)

        return _fn
    if kernel == "liger_tvd":
        x = torch.as_tensor(np.asarray(baseline["input"]), device="cuda", dtype=torch.float32).contiguous()
        y = torch.as_tensor(np.asarray(baseline["target"]), device="cuda", dtype=torch.float32).contiguous()

        def _fn():
            liger_tvd(x, y, None, "batchmean", -100)

        return _fn
    raise KeyError(f"unsupported kernel={kernel}")


def _compare_guided_outputs(*, kernel: str, baseline: dict[str, np.ndarray], guided_outputs: dict[str, np.ndarray]) -> dict[str, float]:
    if kernel == "liger_swiglu":
        return {"c": _max_abs_diff(guided_outputs["c"], baseline["c"])}
    if kernel == "liger_rms_norm":
        return {
            "Y": _max_abs_diff(guided_outputs["Y"], baseline["Y"]),
            "RSTD": _max_abs_diff(guided_outputs["RSTD"], baseline["RSTD"]),
        }
    if kernel == "liger_fused_add_rms_norm":
        return {
            "Y": _max_abs_diff(guided_outputs["Y"], baseline["Y"]),
            "S": _max_abs_diff(guided_outputs["S"], baseline["S"]),
            "RSTD": _max_abs_diff(guided_outputs["RSTD"], baseline["RSTD"]),
        }
    if kernel == "liger_rope":
        return {
            "q_out": _max_abs_diff(guided_outputs["q_out"], baseline["q_out"]),
            "k_out": _max_abs_diff(guided_outputs["k_out"], baseline["k_out"]),
        }
    if kernel == "liger_cross_entropy":
        return {"loss": _max_abs_diff(guided_outputs["loss"], baseline["loss"])}
    if kernel == "liger_geglu":
        return {"c": _max_abs_diff(guided_outputs["c"], baseline["c"])}
    if kernel == "liger_group_norm":
        return {
            "Y": _max_abs_diff(_pick_io_value(guided_outputs, "Y"), _pick_io_value(baseline, "Y")),
            "Mean": _max_abs_diff(_pick_io_value(guided_outputs, "Mean"), _pick_io_value(baseline, "Mean")),
            "RSTD": _max_abs_diff(
                _pick_io_value(guided_outputs, "RSTD", "Rstd"),
                _pick_io_value(baseline, "RSTD", "Rstd"),
            ),
        }
    if kernel == "liger_dyt":
        return {"Y": _max_abs_diff(_pick_io_value(guided_outputs, "Y"), _pick_io_value(baseline, "Y"))}
    if kernel == "liger_layer_norm":
        return {
            "Y": _max_abs_diff(_pick_io_value(guided_outputs, "Y"), _pick_io_value(baseline, "Y")),
            "Mean": _max_abs_diff(_pick_io_value(guided_outputs, "Mean"), _pick_io_value(baseline, "Mean")),
            "RSTD": _max_abs_diff(
                _pick_io_value(guided_outputs, "RSTD", "Rstd"),
                _pick_io_value(baseline, "RSTD", "Rstd"),
            ),
        }
    if kernel == "liger_softmax":
        return {"Y": _max_abs_diff(guided_outputs["Y"], baseline["Y"])}
    if kernel == "liger_qwen2vl_mrope":
        return {
            "q_out": _max_abs_diff(guided_outputs["q_out"], baseline["q_out"]),
            "k_out": _max_abs_diff(guided_outputs["k_out"], baseline["k_out"]),
        }
    if kernel == "liger_sparsemax":
        return {"Y": _max_abs_diff(guided_outputs["Y"], baseline["Y"])}
    if kernel == "liger_kl_div":
        return {"loss": _max_abs_diff(guided_outputs["loss"], baseline["loss"])}
    if kernel == "liger_jsd":
        return {"loss": _max_abs_diff(guided_outputs["loss"], baseline["loss"])}
    if kernel == "liger_fused_linear_cross_entropy":
        return {"loss": _max_abs_diff(_pick_io_value(guided_outputs, "loss"), _pick_io_value(baseline, "loss"))}
    if kernel == "liger_fused_linear_jsd":
        return {"loss": _max_abs_diff(_pick_io_value(guided_outputs, "loss"), _pick_io_value(baseline, "loss"))}
    if kernel == "liger_fused_neighborhood_attention":
        return {"Y": _max_abs_diff(_pick_io_value(guided_outputs, "Y"), _pick_io_value(baseline, "Y"))}
    if kernel == "liger_grpo_loss":
        return {"loss": _max_abs_diff(_pick_io_value(guided_outputs, "loss"), _pick_io_value(baseline, "loss"))}
    if kernel == "liger_llama4_rope":
        return {
            "q_out": _max_abs_diff(_pick_io_value(guided_outputs, "q_out"), _pick_io_value(baseline, "q_out")),
            "k_out": _max_abs_diff(_pick_io_value(guided_outputs, "k_out"), _pick_io_value(baseline, "k_out")),
        }
    if kernel == "liger_mhc":
        return {"Y": _max_abs_diff(_pick_io_value(guided_outputs, "Y"), _pick_io_value(baseline, "Y"))}
    if kernel == "liger_multi_token_attention":
        return {"Y": _max_abs_diff(_pick_io_value(guided_outputs, "Y"), _pick_io_value(baseline, "Y"))}
    if kernel == "liger_poly_norm":
        return {"Y": _max_abs_diff(_pick_io_value(guided_outputs, "Y"), _pick_io_value(baseline, "Y"))}
    if kernel == "liger_tiled_mlp":
        return {"Y": _max_abs_diff(_pick_io_value(guided_outputs, "Y"), _pick_io_value(baseline, "Y"))}
    if kernel == "liger_tvd":
        return {"loss": _max_abs_diff(_pick_io_value(guided_outputs, "loss"), _pick_io_value(baseline, "loss"))}
    raise KeyError(f"unsupported kernel={kernel}")


def _guided_error_score(max_abs: dict[str, float]) -> float:
    vals = [float(v) for v in max_abs.values()]
    return max(vals) if vals else math.inf


def _pick_best_guided(candidates: list[dict[str, Any]], *, tol: float = 1.0e-5) -> dict[str, Any]:
    if not candidates:
        return {
            "candidate": "",
            "kernel_kind": "",
            "bindings": {},
            "contract_path": "",
            "ptx_path": "",
            "guided_ns": 0.0,
            "guided_qps": 0.0,
            "guided_repeats_ns": [],
            "max_abs": {},
            "error_score": math.inf,
            "ok": False,
            "error": "guided_candidate_missing",
        }
    correct = [c for c in candidates if float(c.get("error_score", math.inf)) <= float(tol)]
    pool = correct if correct else candidates
    return max(pool, key=lambda c: float(c.get("guided_qps") or 0.0))


def _run_one(spec, *, out_root: Path, warmup: int, iters: int, repeats: int, shape_overrides: dict[str, int] | None = None) -> dict[str, Any]:
    if shape_overrides:
        spec = replace(spec, canonical_shapes={**dict(spec.canonical_shapes), **dict(shape_overrides)})
    out_dir = out_root / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    _bootstrap_seed_file(kernel=spec.name, out_dir=out_dir, suffix="intent_seed.json")
    _bootstrap_seed_file(kernel=spec.name, out_dir=out_dir, suffix="org_seed.json")
    report_path = out_dir / f"{spec.name}.json"
    report: dict[str, Any]
    pipeline_error = ""
    try:
        report = run_pipeline_for_spec(
            spec,
            out_dir=out_dir,
            cases_limit=1,
            triton_provider="native",
            backend_target="cuda_5090d",
            execution_policy=None,
        )
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        pipeline_error = f"{type(exc).__name__}: {exc}"
        report = {
            "kernel": spec.name,
            "baseline": {"shapes": dict(spec.canonical_shapes), "seed": 0},
            "org": {"error": pipeline_error},
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    baseline_npz = out_dir / f"{spec.name}.baseline.npz"
    if baseline_npz.is_file():
        baseline = dict(np.load(baseline_npz, allow_pickle=False))
    else:
        baseline_shapes = dict(((report.get("baseline") or {}).get("shapes") or {}) or dict(spec.canonical_shapes))
        baseline_seed = int(((report.get("baseline") or {}).get("seed") or 0))
        baseline = {str(k): np.asarray(v) for k, v in dict(spec.runner(TestCase(shapes=baseline_shapes, dtypes={}, seed=baseline_seed))).items()}
    guided_rows: list[dict[str, Any]] = []
    for realization in _load_guided_realizations(report):
        output_names = list(realization["output_names"])
        baseline_for_candidate = _with_io_aliases_for_names(sorted(set(list(baseline.keys()) + list(output_names))), baseline)
        try:
            compiled_module = load_cuda_ptx_module(
                kernel_name=str(realization.get("entry") or spec.name),
                ptx=realization.get("ptx_text") or "",
                io_spec=dict(realization.get("io_spec") or {}),
            )
            args, _inputs_torch, outputs_torch = _build_guided_tensors(
                io_spec=dict(realization.get("io_spec") or {}),
                baseline=baseline_for_candidate,
                bindings=dict(realization.get("shape_bindings") or {}),
                outputs=output_names,
            )
            postprocess_runner = _make_guided_postprocess_runner(
                io_spec=dict(realization.get("io_spec") or {}),
                baseline=baseline_for_candidate,
                outputs_torch=outputs_torch,
            )
            prelaunch_runner = _make_guided_output_init_runner(
                io_spec=dict(realization.get("io_spec") or {}),
                outputs_torch=outputs_torch,
            )
            ns_guided, guided_repeats = _bench_module_launch(
                compiled_module=compiled_module,
                args=args,
                launch=realization["launch"],
                warmup=warmup,
                iters=iters,
                repeats=repeats,
                prelaunch_runner=prelaunch_runner,
                postprocess_runner=postprocess_runner,
            )
            launch = realization["launch"]
            launch_args = [
                *args,
                int(launch.grid[0]),
                int(launch.grid[1]),
                int(launch.grid[2]),
                int(launch.block[0]),
                int(launch.block[1]),
                int(launch.block[2]),
                int(launch.shared_mem),
            ]
            if prelaunch_runner is not None:
                prelaunch_runner()
            compiled_module.launch(*launch_args)
            if postprocess_runner is not None:
                postprocess_runner()
            torch.cuda.synchronize()
            guided_outputs = {str(k): v.detach().cpu().numpy() for k, v in outputs_torch.items()}
            guided_outputs = _with_io_aliases_for_names(sorted(set(list(guided_outputs.keys()) + list(output_names))), guided_outputs)
            guided_outputs = _apply_guided_postprocess(
                io_spec=dict(realization.get("io_spec") or {}),
                baseline=baseline_for_candidate,
                guided_outputs=guided_outputs,
            )
            max_abs = _compare_guided_outputs(kernel=spec.name, baseline=baseline_for_candidate, guided_outputs=guided_outputs)
            guided_rows.append(
                {
                    "candidate": realization["candidate"],
                    "kernel_kind": realization["kernel_kind"],
                    "bindings": dict(realization.get("bindings") or {}),
                    "contract_path": realization["contract_path"],
                    "ptx_path": realization["ptx_path"],
                    "guided_ns": float(ns_guided),
                    "guided_qps": _qps(ns_guided),
                    "guided_repeats_ns": guided_repeats,
                    "max_abs": max_abs,
                    "error_score": _guided_error_score(max_abs),
                    "ok": True,
                    "error": "",
                }
            )
        except Exception as exc:
            guided_rows.append(
                {
                    "candidate": realization["candidate"],
                    "kernel_kind": realization["kernel_kind"],
                    "bindings": dict(realization.get("bindings") or {}),
                    "contract_path": realization["contract_path"],
                    "ptx_path": realization["ptx_path"],
                    "guided_ns": 0.0,
                    "guided_qps": 0.0,
                    "guided_repeats_ns": [],
                    "max_abs": {},
                    "error_score": math.inf,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    best_guided = _pick_best_guided(guided_rows)
    native_fn = _native_callable(spec.name, baseline)
    ns_native, native_repeats = _bench_native(native_fn, warmup=warmup, iters=iters, repeats=repeats)
    org = report.get("org") if isinstance(report.get("org"), dict) else {}
    remote_source = org.get("remote_source") if isinstance(org.get("remote_source"), dict) else {}
    guided_status = {
        "ok": bool(best_guided.get("ok")) if not pipeline_error else False,
        "error": str(pipeline_error or best_guided.get("error") or org.get("error") or org.get("reason") or ""),
    }
    return {
        "kernel": spec.name,
        "shapes": dict(spec.canonical_shapes),
        "report_path": str(report_path),
        "native_ns": float(ns_native),
        "guided_ns": float(best_guided["guided_ns"]),
        "native_qps": _qps(ns_native),
        "guided_qps": float(best_guided["guided_qps"]),
        "ratio": (float(best_guided["guided_qps"]) / _qps(ns_native) if _qps(ns_native) > 0.0 else 0.0),
        "max_abs": dict(best_guided["max_abs"]),
        "native_repeats_ns": native_repeats,
        "guided_repeats_ns": list(best_guided["guided_repeats_ns"]),
        "guided_candidate": {
            "candidate": str(best_guided["candidate"]),
            "kernel_kind": str(best_guided["kernel_kind"]),
            "bindings": dict(best_guided["bindings"]),
            "contract_path": str(best_guided["contract_path"]),
            "ptx_path": str(best_guided["ptx_path"]),
            "error_score": float(best_guided["error_score"]),
        },
        "guided_status": guided_status,
        "guided_candidates": guided_rows,
        "remote_source": remote_source,
    }


def _parse_bindings_json(raw: str) -> dict[str, int]:
    if not str(raw or "").strip():
        return {}
    data = json.loads(str(raw))
    if not isinstance(data, dict):
        raise TypeError("--bindings-json must decode to an object")
    out: dict[str, int] = {}
    for k, v in data.items():
        key = str(k).strip()
        if not key:
            continue
        out[key] = int(v)
    return out


def _bootstrap_seed_file(*, kernel: str, out_dir: Path, suffix: str) -> bool:
    target = out_dir / f"{kernel}.{suffix}"
    if target.is_file():
        return True
    artifacts_root = ROOT / "artifacts"
    candidates = [
        p
        for p in artifacts_root.glob(f"**/{kernel}/{kernel}.{suffix}")
        if p.is_file() and out_dir not in p.parents
    ]
    if candidates:
        source = max(candidates, key=lambda p: p.stat().st_mtime)
        text = source.read_text(encoding="utf-8")
        if suffix == "org_seed.json":
            try:
                payload = json.loads(text)
            except Exception:
                payload = None
            if isinstance(payload, dict) and _env_flag("INTENTIR_ORG_BLINDFOLD"):
                label = str(os.environ.get("INTENTIR_ORG_BLINDFOLD_LABEL") or "").strip() or "target_kernel_func"
                payload["kernel"] = label
                if isinstance(payload.get("org"), dict):
                    payload["org"] = dict(payload["org"])
                    payload["org"]["kernel"] = label
                if isinstance(payload.get("raw_json"), dict):
                    payload["raw_json"] = dict(payload["raw_json"])
                    payload["raw_json"]["kernel"] = label
                text = json.dumps(payload, indent=2, ensure_ascii=False)
        target.write_text(text, encoding="utf-8")
        return True
    if suffix == "org_seed.json":
        org_candidates = [
            p
            for p in artifacts_root.glob(f"**/{kernel}/{kernel}.org.json")
            if p.is_file() and out_dir not in p.parents
        ]
        if not org_candidates:
            return False
        source = max(org_candidates, key=lambda p: p.stat().st_mtime)
        org_payload = json.loads(source.read_text(encoding="utf-8"))
        if _env_flag("INTENTIR_ORG_BLINDFOLD"):
            label = str(os.environ.get("INTENTIR_ORG_BLINDFOLD_LABEL") or "").strip() or "target_kernel_func"
            if isinstance(org_payload, dict):
                org_payload = dict(org_payload)
                org_payload["kernel"] = label
        seed_payload = {
            "schema_version": "org_seed_v1",
            "generated_at": "",
            "kernel": (
                str(os.environ.get("INTENTIR_ORG_BLINDFOLD_LABEL") or "").strip() or "target_kernel_func"
                if _env_flag("INTENTIR_ORG_BLINDFOLD")
                else str(kernel)
            ),
            "triton_provider": "native",
            "backend_target": "cuda_5090d",
            "org": org_payload,
            "raw_json": org_payload,
            "llm_trace": {},
            "quality": {"diff_ok": True, "static_ok": True, "contract_level": "semantic"},
        }
        target.write_text(json.dumps(seed_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", nargs="*", default=list(DEFAULT_KERNELS))
    ap.add_argument("--out", default=str(ARTIFACT_ROOT / "latest"))
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--bindings-json", default="{}")
    args = ap.parse_args()

    os.environ.setdefault("INTENTIR_REAL_MLIR", "1")
    os.environ.setdefault("INTENTIR_CUDA_REAL_MLIR_ALLOW_UNKNOWN", "1")
    os.environ.setdefault("INTENTIR_ORG_MODE", "apply")
    os.environ.setdefault("INTENTIR_ORG_SEED_POLICY", "force_llm")
    os.environ.setdefault("INTENTIR_ORG_REMOTE_SOURCE_ENABLE", "1")

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    spec_map = {spec.name: spec for spec in liger_kernel_specs()}
    bindings_override = _parse_bindings_json(str(args.bindings_json))
    rows: list[dict[str, Any]] = []
    for name in list(args.kernels):
        spec = spec_map[str(name)]
        row = _run_one(
            spec,
            out_root=out_root,
            warmup=int(args.warmup),
            iters=int(args.iters),
            repeats=int(args.repeats),
            shape_overrides=bindings_override,
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    summary = {"rows": rows}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
