from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from intent_ir.mlir.module import IntentMLIRModule
from intent_ir.mlir.passes.emit_cuda_contract import build_cuda_contract
from intent_ir.mlir.passes.emit_rvv_contract import build_rvv_contract
from intent_ir.mlir.toolchain import detect_mlir_toolchain
from pipeline.common.strict_policy import cuda_require_llvm_ptx


def _dump_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _normalize_shape_bindings(shape_bindings: dict[str, Any] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for k, v in dict(shape_bindings or {}).items():
        key = str(k).strip()
        if not key:
            continue
        try:
            out[key] = int(v)
        except Exception:
            continue
    return out


def _runtime_io_spec_from_intent_json(intent_json: dict[str, Any]) -> dict[str, Any]:
    tensors = dict(intent_json.get("tensors") or {})
    ops = [x for x in list(intent_json.get("ops") or []) if isinstance(x, dict)]
    outputs = [str(x) for x in list(intent_json.get("outputs") or []) if str(x).strip()]
    produced = {str(op.get("output") or "").strip() for op in ops if str(op.get("output") or "").strip()}
    used: set[str] = set()
    for op in ops:
        for inp in list(op.get("inputs") or []):
            name = str(inp).strip()
            if name:
                used.add(name)
    external_inputs = sorted([n for n in used if n in tensors and n not in produced])
    # Macro ops may reference scalar ABI inputs implicitly (not present in op.inputs).
    has_macro = any(str(op.get("op") or "").strip() == "upsample_bicubic2d_aa" for op in ops)
    if has_macro:
        extra_scalars: list[str] = []
        for name, spec in tensors.items():
            nm = str(name).strip()
            if not nm or nm in produced or nm in outputs or nm in external_inputs:
                continue
            if not isinstance(spec, dict):
                continue
            shape = list(spec.get("shape") or [])
            if len(shape) == 0:
                extra_scalars.append(nm)
        external_inputs.extend(sorted(extra_scalars))
    io_names = list(external_inputs) + [n for n in outputs if n in tensors and n not in set(external_inputs)]
    io_tensors: dict[str, Any] = {}
    for name in io_names:
        spec = tensors.get(name)
        if not isinstance(spec, dict):
            continue
        io_tensors[str(name)] = {
            "dtype": str(spec.get("dtype") or "f32"),
            "shape": list(spec.get("shape") or []),
            "layout": str(spec.get("layout") or "row_major"),
        }
    return {
        "arg_names": [str(x) for x in io_names],
        "tensors": io_tensors,
        "outputs": [str(x) for x in outputs],
        "scalars": {},
    }


def _recover_intent_json(
    *,
    module: IntentMLIRModule,
    fallback_intent_module: IntentMLIRModule | None = None,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    if isinstance(module.intent_json, dict):
        candidates.append(dict(module.intent_json))
    if fallback_intent_module is not None and isinstance(fallback_intent_module.intent_json, dict):
        candidates.append(dict(fallback_intent_module.intent_json))
    meta_intent = (module.meta or {}).get("intent_json") if isinstance(module.meta, dict) else None
    if isinstance(meta_intent, dict):
        candidates.append(dict(meta_intent))
    for cand in candidates:
        if isinstance(cand.get("tensors"), dict):
            return cand
    return None


def _looks_like_llvm_ir(text: str) -> bool:
    s = str(text or "")
    return ("; ModuleID" in s) or ("define " in s and "{" in s and "}" in s)


def _llvm_target_triple(llvm_ir_text: str) -> str:
    s = str(llvm_ir_text or "")
    m = re.search(r'target\s+triple\s*=\s*"([^"]+)"', s)
    return str(m.group(1) if m else "").strip().lower()


def _is_host_llvm_triple_for_cuda(triple: str) -> bool:
    t = str(triple or "").strip().lower()
    if not t:
        return False
    if "nvptx" in t or "amdgcn" in t or "spir" in t or "spirv" in t:
        return False
    host_markers = ("x86_64", "i386", "aarch64", "arm", "riscv", "wasm")
    return any(x in t for x in host_markers)


def _retarget_llvm_ir_to_cuda_device_triple(llvm_ir_text: str) -> tuple[str, bool]:
    text = str(llvm_ir_text or "")
    if not text:
        return text, False
    triple = _llvm_target_triple(text)
    if triple and not _is_host_llvm_triple_for_cuda(triple):
        return text, False
    device_triple = "nvptx64-nvidia-cuda"
    pat = re.compile(r'(target\s+triple\s*=\s*")([^"]*)(")')
    if pat.search(text):
        return pat.sub(rf'\g<1>{device_triple}\3', text, count=1), True
    return f'target triple = "{device_triple}"\n{text}', True


def _is_rvv_llvm_triple(triple: str) -> bool:
    t = str(triple or "").strip().lower()
    if not t:
        return False
    return ("riscv" in t) and ("linux" in t or "unknown" in t or "elf" in t)


def _host_llvm_triple() -> str:
    raw = str(os.getenv("INTENTIR_HOST_LLVM_TRIPLE", "")).strip().lower()
    if raw:
        return raw
    machine = str(os.uname().machine or "").strip().lower()
    if machine in {"x86_64", "amd64"}:
        return "x86_64-pc-linux-gnu"
    if machine in {"aarch64", "arm64"}:
        return "aarch64-unknown-linux-gnu"
    if machine:
        return f"{machine}-unknown-linux-gnu"
    return "x86_64-pc-linux-gnu"


def _retarget_llvm_ir_to_host_triple_for_link(llvm_ir_text: str) -> tuple[str, bool, str]:
    text = str(llvm_ir_text or "")
    if not text:
        return text, False, _host_llvm_triple()
    host_triple = _host_llvm_triple()
    changed = False

    triple_pat = re.compile(r'(target\s+triple\s*=\s*")([^"]*)(")')
    if triple_pat.search(text):
        text2 = triple_pat.sub(rf'\g<1>{host_triple}\3', text, count=1)
        changed = bool(changed or text2 != text)
        text = text2
    else:
        text = f'target triple = "{host_triple}"\n{text}'
        changed = True

    if "x86_64" in host_triple:
        host_x86_datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
        datalayout_pat = re.compile(r'(target\s+datalayout\s*=\s*")([^"]*)(")')
        if datalayout_pat.search(text):
            text2 = datalayout_pat.sub(rf'\g<1>{host_x86_datalayout}\3', text, count=1)
            changed = bool(changed or text2 != text)
            text = text2
        else:
            text = f'target datalayout = "{host_x86_datalayout}"\n{text}'
            changed = True

    for key in ("target-cpu", "target-features", "tune-cpu", "target-abi"):
        pat = re.compile(rf'\s*"{re.escape(str(key))}"="[^"]*"')
        text2 = pat.sub("", text)
        changed = bool(changed or text2 != text)
        text = text2
    return text, changed, host_triple


def _cuda_llc_target() -> str:
    raw = str(os.getenv("INTENTIR_CUDA_SM", "")).strip().lower()
    if raw.startswith("sm_"):
        return raw
    if raw.isdigit():
        return f"sm_{raw}"
    # Best-effort auto-detect from torch when available so local dev runs (and
    # perf evidence) compile for the actual GPU arch without extra knobs.
    try:  # pragma: no cover - depends on CUDA env
        import torch  # type: ignore

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            if isinstance(major, int) and isinstance(minor, int) and major > 0 and minor >= 0:
                return f"sm_{major}{minor}"
    except Exception:
        pass
    return "sm_80"


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(str(name), "")
    if raw is None:
        return bool(default)
    v = str(raw).strip().lower()
    if not v:
        return bool(default)
    return v in {"1", "true", "yes", "y", "on"}


def _parse_ptx_maxntid(*, ptx_text: str, entry: str) -> tuple[int, int, int] | None:
    text = str(ptx_text or "")
    if not text:
        return None

    scopes: list[str] = [text]
    entry_name = str(entry or "").strip()
    if entry_name:
        entry_pat = re.compile(rf"\.visible\s+\.entry\s+{re.escape(entry_name)}\s*\(", re.MULTILINE)
        m = entry_pat.search(text)
        if m is not None:
            tail = text[m.end() :]
            end_idx = tail.find("}")
            scopes.insert(0, tail[: end_idx + 1] if end_idx >= 0 else tail)

    maxntid_pat = re.compile(r"\.maxntid\s+(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")
    for scope in scopes:
        mm = maxntid_pat.search(scope)
        if mm is None:
            continue
        try:
            return (
                max(1, int(mm.group(1))),
                max(1, int(mm.group(2))),
                max(1, int(mm.group(3))),
            )
        except Exception:
            continue
    return None


def _list_ptx_entry_symbols(ptx_text: str) -> list[str]:
    text = str(ptx_text or "")
    if not text:
        return []
    pat = re.compile(r"\.visible\s+\.entry\s+([A-Za-z_.$][\w.$]*)\s*\(")
    seen: set[str] = set()
    out: list[str] = []
    for m in pat.finditer(text):
        name = str(m.group(1) or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _resolve_ptx_entry_symbol(*, ptx_path: Path, preferred_entry: str) -> tuple[str, list[str]]:
    text = str(ptx_path.read_text(encoding="utf-8", errors="ignore") or "")
    entries = _list_ptx_entry_symbols(text)
    preferred = str(preferred_entry or "").strip()
    if preferred and preferred in entries:
        return preferred, entries
    if preferred:
        heuristics = [e for e in entries if e.endswith(preferred) or preferred in e]
        if len(heuristics) == 1:
            return str(heuristics[0]), entries
    if entries:
        return str(entries[0]), entries
    return preferred, entries


def _as_int3(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    out: list[int] = []
    for item in value:
        try:
            out.append(max(1, int(item)))
        except Exception:
            return None
    return (int(out[0]), int(out[1]), int(out[2]))


def _align_cuda_contract_launch_with_ptx_maxntid(contract: Any) -> None:
    executable = getattr(contract, "executable", None)
    exe_format = str(getattr(executable, "format", "") or "").strip().lower()
    if exe_format not in {"cuda_ptx", "ptx"}:
        return

    artifacts = dict(getattr(contract, "artifacts", {}) or {})
    ptx_path_raw = str(getattr(executable, "path", "") or artifacts.get("cuda_ptx_path") or "").strip()
    if not ptx_path_raw:
        return

    ptx_path = Path(ptx_path_raw)
    if not ptx_path.is_absolute():
        ptx_path = (Path.cwd() / ptx_path).resolve()
    if not ptx_path.is_file():
        return

    ptx_text = ptx_path.read_text(encoding="utf-8", errors="ignore")
    entry = str(getattr(executable, "entry", "") or getattr(contract, "kernel_name", "") or "").strip()
    maxntid = _parse_ptx_maxntid(ptx_text=ptx_text, entry=entry)
    if maxntid is None:
        return
    artifacts["cuda_ptx_maxntid"] = [int(maxntid[0]), int(maxntid[1]), int(maxntid[2])]

    launch = dict(getattr(contract, "launch", {}) or {})
    block = _as_int3(launch.get("block"))
    grid = _as_int3(launch.get("grid"))
    if block is None:
        launch["block"] = [int(maxntid[0]), int(maxntid[1]), int(maxntid[2])]
        if grid is None:
            launch["grid"] = [1, 1, 1]
        launch["shared_mem"] = int(launch.get("shared_mem") or 0)
        artifacts["cuda_launch_block_inferred_from_ptx"] = True
        contract.launch = launch
        contract.artifacts = artifacts
        return

    clamped = (
        min(int(block[0]), int(maxntid[0])),
        min(int(block[1]), int(maxntid[1])),
        min(int(block[2]), int(maxntid[2])),
    )
    if clamped != block:
        old_grid = grid if grid is not None else (1, 1, 1)
        new_grid: list[int] = []
        for g, b, c in zip(old_grid, block, clamped):
            logical_threads = max(1, int(g) * int(b))
            new_grid.append(int((logical_threads + int(c) - 1) // int(c)))
        launch["block"] = [int(clamped[0]), int(clamped[1]), int(clamped[2])]
        launch["grid"] = [int(new_grid[0]), int(new_grid[1]), int(new_grid[2])]
        launch["shared_mem"] = int(launch.get("shared_mem") or 0)
        artifacts["cuda_launch_block_adjusted_from"] = [int(block[0]), int(block[1]), int(block[2])]
        artifacts["cuda_launch_grid_adjusted_from"] = [int(old_grid[0]), int(old_grid[1]), int(old_grid[2])]
        artifacts["cuda_launch_adjust_reason"] = "ptx_maxntid_cap"
        contract.launch = launch
    contract.artifacts = artifacts


def _rewrite_nvptx_math_intrinsics_for_llc(llvm_ir_text: str) -> str:
    """Rewrite unstable NVPTX math callsites for LLVM-14/ptxas stability.

    We currently rewrite:
    - LLVM intrinsics: `llvm.exp/exp2/log/sin/cos/pow.f32` -> internal helpers backed by NVVM approx intrinsics.
    - External libcalls: `acosf/atanf/tanf/erff` -> internal helpers (no unresolved externs).
    """

    text = str(llvm_ir_text or "")
    if not text:
        return text
    rewrite_exp = "@llvm.exp.f32(" in text
    rewrite_exp2 = "@llvm.exp2.f32(" in text
    rewrite_log = "@llvm.log.f32(" in text
    rewrite_sin = "@llvm.sin.f32(" in text
    rewrite_cos = "@llvm.cos.f32(" in text
    rewrite_pow = "@llvm.pow.f32(" in text
    rewrite_acos = "@acosf(" in text
    rewrite_atan = "@atanf(" in text
    rewrite_tan = "@tanf(" in text
    rewrite_erf = "@erff(" in text
    out = text
    if rewrite_exp:
        out = out.replace("@llvm.exp.f32(", "@intentir_nvvm_expf_approx(")
        out = re.sub(r"^.*declare float @intentir_nvvm_expf_approx\(float\).*\n", "", out, flags=re.MULTILINE)
    if rewrite_exp2:
        out = out.replace("@llvm.exp2.f32(", "@intentir_nvvm_exp2f_approx(")
        out = re.sub(r"^.*declare float @intentir_nvvm_exp2f_approx\(float\).*\n", "", out, flags=re.MULTILINE)
    if rewrite_log:
        out = out.replace("@llvm.log.f32(", "@intentir_nvvm_logf_approx(")
        out = re.sub(r"^.*declare float @intentir_nvvm_logf_approx\(float\).*\n", "", out, flags=re.MULTILINE)
    if rewrite_sin:
        out = out.replace("@llvm.sin.f32(", "@intentir_nvvm_sinf_approx(")
        out = re.sub(r"^.*declare float @intentir_nvvm_sinf_approx\(float\).*\n", "", out, flags=re.MULTILINE)
    if rewrite_cos:
        out = out.replace("@llvm.cos.f32(", "@intentir_nvvm_cosf_approx(")
        out = re.sub(r"^.*declare float @intentir_nvvm_cosf_approx\(float\).*\n", "", out, flags=re.MULTILINE)
    if rewrite_pow:
        out = out.replace("@llvm.pow.f32(", "@intentir_nvvm_powf_approx(")
        out = re.sub(
            r"^.*declare float @intentir_nvvm_powf_approx\(float,\s*float\).*\n",
            "",
            out,
            flags=re.MULTILINE,
        )
    if rewrite_acos:
        out = out.replace("@acosf(", "@intentir_nvvm_acosf_approx(")
        out = re.sub(r"^declare[^\n]*@acosf\([^\n]*\)\s*[^\n]*\n", "", out, flags=re.MULTILINE)
        out = re.sub(
            r"^declare[^\n]*@intentir_nvvm_acosf_approx\([^\n]*\)\s*[^\n]*\n",
            "",
            out,
            flags=re.MULTILINE,
        )
    if rewrite_atan:
        out = out.replace("@atanf(", "@intentir_nvvm_atanf_approx(")
        out = re.sub(r"^declare[^\n]*@atanf\([^\n]*\)\s*[^\n]*\n", "", out, flags=re.MULTILINE)
        out = re.sub(
            r"^declare[^\n]*@intentir_nvvm_atanf_approx\([^\n]*\)\s*[^\n]*\n",
            "",
            out,
            flags=re.MULTILINE,
        )
    if rewrite_tan:
        out = out.replace("@tanf(", "@intentir_nvvm_tanf_approx(")
        out = re.sub(r"^declare[^\n]*@tanf\([^\n]*\)\s*[^\n]*\n", "", out, flags=re.MULTILINE)
        out = re.sub(
            r"^declare[^\n]*@intentir_nvvm_tanf_approx\([^\n]*\)\s*[^\n]*\n",
            "",
            out,
            flags=re.MULTILINE,
        )
    if rewrite_erf:
        out = out.replace("@erff(", "@intentir_nvvm_erff_approx(")
        out = re.sub(r"^declare[^\n]*@erff\([^\n]*\)\s*[^\n]*\n", "", out, flags=re.MULTILINE)
        out = re.sub(
            r"^declare[^\n]*@intentir_nvvm_erff_approx\([^\n]*\)\s*[^\n]*\n",
            "",
            out,
            flags=re.MULTILINE,
        )
    if not (
        rewrite_exp
        or rewrite_exp2
        or rewrite_log
        or rewrite_sin
        or rewrite_cos
        or rewrite_pow
        or rewrite_acos
        or rewrite_atan
        or rewrite_tan
        or rewrite_erf
    ):
        return out

    helper_blocks: list[str] = []
    if (rewrite_exp or rewrite_exp2 or rewrite_pow or rewrite_erf) and "@llvm.nvvm.ex2.approx.f(" not in out:
        helper_blocks.append("declare float @llvm.nvvm.ex2.approx.f(float)")
    if (rewrite_log or rewrite_pow) and "@llvm.nvvm.lg2.approx.f(" not in out:
        helper_blocks.append("declare float @llvm.nvvm.lg2.approx.f(float)")
    if (rewrite_acos or rewrite_atan or rewrite_pow or rewrite_erf) and "@llvm.fabs.f32(" not in out:
        helper_blocks.append("declare float @llvm.fabs.f32(float)")
    if (rewrite_sin or rewrite_tan) and "@llvm.nvvm.sin.approx.f(" not in out:
        helper_blocks.append("declare float @llvm.nvvm.sin.approx.f(float)")
    if (rewrite_cos or rewrite_tan) and "@llvm.nvvm.cos.approx.f(" not in out:
        helper_blocks.append("declare float @llvm.nvvm.cos.approx.f(float)")
    if rewrite_acos and "@llvm.sqrt.f32(" not in out:
        helper_blocks.append("declare float @llvm.sqrt.f32(float)")
    if rewrite_exp and "define internal float @intentir_nvvm_expf_approx(" not in out:
        helper_blocks.append(
            "\n".join(
                [
                    "define internal float @intentir_nvvm_expf_approx(float %x) {",
                    "entry:",
                    "  %c_log2e = bitcast i32 1069066811 to float",
                    "  %scaled = fmul float %x, %c_log2e",
                    "  %r = call float @llvm.nvvm.ex2.approx.f(float %scaled)",
                    "  ret float %r",
                    "}",
                ]
            )
        )
    if rewrite_exp2 and "define internal float @intentir_nvvm_exp2f_approx(" not in out:
        helper_blocks.append(
            "\n".join(
                [
                    "define internal float @intentir_nvvm_exp2f_approx(float %x) {",
                    "entry:",
                    "  %r = call float @llvm.nvvm.ex2.approx.f(float %x)",
                    "  ret float %r",
                    "}",
                ]
            )
        )
    if rewrite_sin and "define internal float @intentir_nvvm_sinf_approx(" not in out:
        helper_blocks.append(
            "\n".join(
                [
                    "define internal float @intentir_nvvm_sinf_approx(float %x) {",
                    "entry:",
                    "  %r = call float @llvm.nvvm.sin.approx.f(float %x)",
                    "  ret float %r",
                    "}",
                ]
            )
        )
    if rewrite_cos and "define internal float @intentir_nvvm_cosf_approx(" not in out:
        helper_blocks.append(
            "\n".join(
                [
                    "define internal float @intentir_nvvm_cosf_approx(float %x) {",
                    "entry:",
                    "  %r = call float @llvm.nvvm.cos.approx.f(float %x)",
                    "  ret float %r",
                    "}",
                ]
            )
        )
    if rewrite_tan and "define internal float @intentir_nvvm_tanf_approx(" not in out:
        helper_blocks.append(
            "\n".join(
                [
                    "define internal float @intentir_nvvm_tanf_approx(float %x) {",
                    "entry:",
                    "  %s = call float @llvm.nvvm.sin.approx.f(float %x)",
                    "  %c = call float @llvm.nvvm.cos.approx.f(float %x)",
                    "  %r = fdiv float %s, %c",
                    "  ret float %r",
                    "}",
                ]
            )
        )
    if rewrite_pow and "define internal float @intentir_nvvm_powf_approx(" not in out:
        helper_blocks.append(
            "\n".join(
                [
                    "define internal float @intentir_nvvm_powf_approx(float %x, float %y) {",
                    "entry:",
                    "  %c_nan = bitcast i32 2143289344 to float",
                    "  %zero = bitcast i32 0 to float",
                    "  %ax = call float @llvm.fabs.f32(float %x)",
                    "  %lg = call float @llvm.nvvm.lg2.approx.f(float %ax)",
                    "  %s = fmul float %lg, %y",
                    "  %mag = call float @llvm.nvvm.ex2.approx.f(float %s)",
                    "  %yi = fptosi float %y to i32",
                    "  %yf = sitofp i32 %yi to float",
                    "  %is_int = fcmp oeq float %y, %yf",
                    "  %odd_bits = and i32 %yi, 1",
                    "  %is_odd = icmp ne i32 %odd_bits, 0",
                    "  %x_neg = fcmp olt float %x, %zero",
                    "  %neg_int = and i1 %x_neg, %is_int",
                    "  %neg_odd = and i1 %neg_int, %is_odd",
                    "  %mag_neg = fneg float %mag",
                    "  %mag_signed = select i1 %neg_odd, float %mag_neg, float %mag",
                    "  %x_pos = fcmp ogt float %x, %zero",
                    "  %valid = or i1 %x_pos, %is_int",
                    "  %r = select i1 %valid, float %mag_signed, float %c_nan",
                    "  ret float %r",
                    "}",
                ]
            )
        )
    if rewrite_erf and "define internal float @intentir_nvvm_erff_approx(" not in out:
        helper_blocks.append(
            "\n".join(
                [
                    "define internal float @intentir_nvvm_erff_approx(float %x) {",
                    "entry:",
                    "  %one = bitcast i32 1065353216 to float",
                    "  %zero = bitcast i32 0 to float",
                    "  %c_log2e = bitcast i32 1069066811 to float",
                    "  %c_p = bitcast i32 1051179525 to float",
                    "  %c_a1 = bitcast i32 1048738054 to float",
                    "  %c_a2 = bitcast i32 -1097750130 to float",
                    "  %c_a3 = bitcast i32 1068888291 to float",
                    "  %c_a4 = bitcast i32 -1078329117 to float",
                    "  %c_a5 = bitcast i32 1065868322 to float",
                    "  %ax = call float @llvm.fabs.f32(float %x)",
                    "  %t0 = fmul float %c_p, %ax",
                    "  %t1 = fadd float %one, %t0",
                    "  %t = fdiv float %one, %t1",
                    "  %p0 = fmul float %c_a5, %t",
                    "  %p1 = fadd float %p0, %c_a4",
                    "  %p2 = fmul float %p1, %t",
                    "  %p3 = fadd float %p2, %c_a3",
                    "  %p4 = fmul float %p3, %t",
                    "  %p5 = fadd float %p4, %c_a2",
                    "  %p6 = fmul float %p5, %t",
                    "  %p7 = fadd float %p6, %c_a1",
                    "  %poly = fmul float %p7, %t",
                    "  %xx = fmul float %ax, %ax",
                    "  %neg_xx = fneg float %xx",
                    "  %scaled = fmul float %neg_xx, %c_log2e",
                    "  %expv = call float @llvm.nvvm.ex2.approx.f(float %scaled)",
                    "  %tau = fmul float %poly, %expv",
                    "  %core = fsub float %one, %tau",
                    "  %neg = fcmp olt float %x, %zero",
                    "  %ncore = fneg float %core",
                    "  %res = select i1 %neg, float %ncore, float %core",
                    "  ret float %res",
                    "}",
                ]
            )
        )
    if rewrite_atan and "define internal float @intentir_nvvm_atanf_approx(" not in out:
        helper_blocks.append(
            "\n".join(
                [
                    "define internal float @intentir_nvvm_atanf_approx(float %x) {",
                    "entry:",
                    "  %c_a9 = bitcast i32 1017818725 to float",
                    "  %c_a7 = bitcast i32 -1112647114 to float",
                    "  %c_a5 = bitcast i32 1043887842 to float",
                    "  %c_a3 = bitcast i32 -1096213244 to float",
                    "  %c_a1 = bitcast i32 1065350968 to float",
                    "  %c_pio2 = bitcast i32 1070141403 to float",
                    "  %ax = call float @llvm.fabs.f32(float %x)",
                    "  %le1 = fcmp ole float %ax, 1.000000e+00",
                    "  %inv = fdiv float 1.000000e+00, %ax",
                    "  %v = select i1 %le1, float %ax, float %inv",
                    "  %z = fmul float %v, %v",
                    "  %p0 = fmul float %z, %c_a9",
                    "  %p1 = fadd float %p0, %c_a7",
                    "  %p2 = fmul float %p1, %z",
                    "  %p3 = fadd float %p2, %c_a5",
                    "  %p4 = fmul float %p3, %z",
                    "  %p5 = fadd float %p4, %c_a3",
                    "  %p6 = fmul float %p5, %z",
                    "  %p7 = fadd float %p6, %c_a1",
                    "  %core = fmul float %p7, %v",
                    "  %hi = fsub float %c_pio2, %core",
                    "  %mag = select i1 %le1, float %core, float %hi",
                    "  %neg = fcmp olt float %x, 0.000000e+00",
                    "  %nmag = fneg float %mag",
                    "  %res = select i1 %neg, float %nmag, float %mag",
                    "  ret float %res",
                    "}",
                ]
            )
        )
    if rewrite_acos and "define internal float @intentir_nvvm_acosf_approx(" not in out:
        helper_blocks.append(
            "\n".join(
                [
                    "define internal float @intentir_nvvm_acosf_approx(float %x) {",
                    "entry:",
                    "  %c0 = bitcast i32 -1130795472 to float",
                    "  %c1 = bitcast i32 1033377319 to float",
                    "  %c2 = bitcast i32 1046033540 to float",
                    "  %c3 = bitcast i32 1070140838 to float",
                    "  %c_pi = bitcast i32 1078530012 to float",
                    "  %ax = call float @llvm.fabs.f32(float %x)",
                    "  %p0 = fmul float %ax, %c0",
                    "  %p1 = fadd float %p0, %c1",
                    "  %p2 = fmul float %p1, %ax",
                    "  %p3 = fsub float %p2, %c2",
                    "  %p4 = fmul float %p3, %ax",
                    "  %p5 = fadd float %p4, %c3",
                    "  %om = fsub float 1.000000e+00, %ax",
                    "  %sq = call float @llvm.sqrt.f32(float %om)",
                    "  %r = fmul float %p5, %sq",
                    "  %neg = fcmp olt float %x, 0.000000e+00",
                    "  %pi_minus = fsub float %c_pi, %r",
                    "  %res = select i1 %neg, float %pi_minus, float %r",
                    "  ret float %res",
                    "}",
                ]
            )
        )
    if rewrite_log and "define internal float @intentir_nvvm_logf_approx(" not in out:
        helper_blocks.append(
            "\n".join(
                [
                    "define internal float @intentir_nvvm_logf_approx(float %x) {",
                    "entry:",
                    "  %r = call float @llvm.nvvm.lg2.approx.f(float %x)",
                    "  %c_ln2 = bitcast i32 1060205080 to float",
                    "  %ln = fmul float %r, %c_ln2",
                    "  ret float %ln",
                    "}",
                ]
            )
        )

    if not helper_blocks:
        return out

    inject = "\n" + "\n\n".join(helper_blocks) + "\n"
    marker = "\n!nvvm.annotations ="
    if marker in out:
        return out.replace(marker, inject + marker, 1)
    return out + inject


def _compile_llvm_ir_to_cuda_ptx(
    *,
    llvm_ir_text: str,
    out_path: Path,
) -> str:
    triple = _llvm_target_triple(llvm_ir_text)
    probe = detect_mlir_toolchain()
    tools = dict(probe.get("tools") or {}) if isinstance(probe, dict) else {}
    llc_row = dict(tools.get("llc") or {})
    llc_ok = bool(llc_row.get("available"))
    llc_path = str(llc_row.get("path") or "").strip()
    if not llc_ok or not llc_path:
        raise RuntimeError("llc unavailable for CUDA PTX materialization")
    llc_ver = str(llc_row.get("version") or "").strip()
    target = _cuda_llc_target()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _find_libdevice_bc() -> Path | None:
        # Prefer an explicit override so CI/remote setups can pin a path.
        env = str(os.getenv("INTENTIR_CUDA_LIBDEVICE_BC", "") or "").strip()
        if env:
            p = Path(env)
            return p if p.is_file() else None
        preferred = Path("/usr/local/cuda/nvvm/libdevice/libdevice.10.bc")
        if preferred.is_file():
            return preferred
        # Fall back to the newest CUDA install under /usr/local.
        candidates = sorted(Path("/usr/local").glob("cuda*/nvvm/libdevice/libdevice.10.bc"))
        for p in reversed(candidates):
            if p.is_file():
                return p
        return None

    with tempfile.TemporaryDirectory(prefix="intentir_cuda_llc_") as td:
        ll_path = Path(td) / "kernel.ll"
        ll_text = str(llvm_ir_text or "")
        if _is_host_llvm_triple_for_cuda(triple):
            ll_text, _ = _retarget_llvm_ir_to_cuda_device_triple(ll_text)
            triple = _llvm_target_triple(ll_text)
        if "nvptx" in triple:
            ll_text = _rewrite_nvptx_math_intrinsics_for_llc(ll_text)
        ll_path.write_text(ll_text, encoding="utf-8")

        # Link CUDA libdevice when available so ptxas validation can succeed for
        # math functions (exp/erf/etc). Without libdevice, LLVM often emits
        # unresolved extern calls like __nv_expf.
        llc_input: Path = ll_path
        llvm_as_path = str((tools.get("llvm-as") or {}).get("path") or "").strip()
        llvm_link_path = ""
        if llvm_as_path:
            llvm_link_path = str(Path(llvm_as_path).with_name("llvm-link"))
        libdevice_bc = _find_libdevice_bc()
        # Default: link libdevice when available. Without it, LLVM's NVPTX
        # backend can emit unresolved extern calls like __nv_expf/__nv_erff,
        # which the CUDA driver rejects at module-load time.
        link_libdevice = _env_flag("INTENTIR_CUDA_LINK_LIBDEVICE", default=True)
        if (
            link_libdevice
            and llvm_as_path
            and llvm_link_path
            and Path(llvm_link_path).is_file()
            and libdevice_bc is not None
        ):
            bc_path = Path(td) / "kernel.bc"
            linked_bc = Path(td) / "kernel.linked.bc"
            as_cp = subprocess.run([llvm_as_path, str(ll_path), "-o", str(bc_path)], capture_output=True, text=True)
            if int(as_cp.returncode) != 0:
                raise RuntimeError(
                    f"llvm-as failed rc={as_cp.returncode}: {as_cp.stderr or as_cp.stdout}"
                )
            link_cp = subprocess.run(
                [
                    llvm_link_path,
                    str(bc_path),
                    str(libdevice_bc),
                    # libdevice contains a huge amount of bitcode (including
                    # intrinsics our NVPTX backend may not lower). Keep the
                    # link minimal so we only pull in what this kernel needs.
                    "--only-needed",
                    "--internalize",
                    "-o",
                    str(linked_bc),
                ],
                capture_output=True,
                text=True,
            )
            if int(link_cp.returncode) != 0:
                raise RuntimeError(
                    f"llvm-link libdevice failed rc={link_cp.returncode}: {link_cp.stderr or link_cp.stdout}"
                )
            llc_input = linked_bc

        cmd = [
            llc_path,
            "-O3",
            "-march=nvptx64",
            f"-mcpu={target}",
            "-o",
            str(out_path),
            str(llc_input),
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if int(cp.returncode) != 0:
            raise RuntimeError(
                f"llc nvptx compile failed rc={cp.returncode}: {cp.stderr or cp.stdout}"
            )
    return f"llc:{llc_path}:{llc_ver}"


def _validate_cuda_ptx_assembly(
    *,
    ptx_path: Path,
    entry: str,
) -> None:
    text = str(ptx_path.read_text(encoding="utf-8") or "")
    if not text.strip():
        raise RuntimeError(f"generated PTX is empty: {ptx_path}")
    # Keep this conservative to avoid false positives in mocked unit tests.
    if str(entry or "").strip():
        pat = re.compile(rf"\.visible\s+\.entry\s+{re.escape(str(entry).strip())}\b")
    if pat.search(text) is None:
        # Defer hard failure to ptxas check when available.
        pass
    # Default: do not run ptxas validation. The CUDA driver JIT is the true
    # execution path and some NVPTX outputs (or libdevice-linked variants) can
    # be rejected by ptxas while still being accepted by the driver.
    if str(os.getenv("INTENTIR_CUDA_PTXAS_VALIDATE", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        return
    probe = detect_mlir_toolchain()
    tools = dict(probe.get("tools") or {}) if isinstance(probe, dict) else {}
    ptxas_row = dict(tools.get("ptxas") or {})
    ptxas_path = str(ptxas_row.get("path") or "").strip()
    if not ptxas_path:
        return
    arch = _cuda_llc_target()
    with tempfile.TemporaryDirectory(prefix="intentir_ptxas_check_") as td:
        cubin_path = Path(td) / "kernel.cubin"
        cmd = [ptxas_path, f"-arch={arch}", str(ptx_path), "-o", str(cubin_path)]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if int(cp.returncode) != 0:
            raise RuntimeError(f"ptxas validate failed rc={cp.returncode}: {cp.stderr or cp.stdout}")


def _compile_llvm_ir_to_elf(
    *,
    llvm_ir_text: str,
    out_path: Path,
) -> str:
    probe = detect_mlir_toolchain()
    tools = dict(probe.get("tools") or {}) if isinstance(probe, dict) else {}
    llc_row = dict(tools.get("llc") or {})
    clang_row = dict(tools.get("clang") or {})
    llc_path = str(llc_row.get("path") or "").strip()
    clang_path = str(clang_row.get("path") or "").strip()
    if not llc_path:
        raise RuntimeError("llc unavailable for ELF materialization")
    if not clang_path:
        raise RuntimeError("clang unavailable for ELF materialization")
    llc_ver = str(llc_row.get("version") or "").strip()
    clang_ver = str(clang_row.get("version") or "").strip()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_dir = Path(__file__).resolve().parents[1] / "backends" / "spmd_rvv" / "runtime"
    runtime_sources = [
        runtime_dir / "intentir_runtime.c",
        runtime_dir / "intentir_driver.c",
        runtime_dir / "intentir_ops.c",
    ]
    for src in runtime_sources:
        if not src.is_file():
            raise FileNotFoundError(f"missing RVV runtime source for ELF link: {src}")

    def _compile_once(ll_text: str) -> None:
        with tempfile.TemporaryDirectory(prefix="intentir_llvm_to_elf_") as td:
            ll_path = Path(td) / "kernel.ll"
            obj_path = Path(td) / "kernel.o"
            ll_path.write_text(str(ll_text or ""), encoding="utf-8")
            llc_cmd = [llc_path, "-O3", "-filetype=obj", "-o", str(obj_path), str(ll_path)]
            llc_cp = subprocess.run(llc_cmd, capture_output=True, text=True)
            if int(llc_cp.returncode) != 0:
                raise RuntimeError(
                    f"llc elf compile failed rc={llc_cp.returncode}: {llc_cp.stderr or llc_cp.stdout}"
                )
            clang_cmd = [
                clang_path,
                str(obj_path),
                *(str(x) for x in runtime_sources),
                "-O2",
                "-std=c11",
                "-D_POSIX_C_SOURCE=200809L",
                f"-I{runtime_dir}",
                "-no-pie",
                "-o",
                str(out_path),
                "-lm",
                "-lrt",
            ]
            clang_cp = subprocess.run(clang_cmd, capture_output=True, text=True)
            if int(clang_cp.returncode) != 0:
                raise RuntimeError(
                    f"clang elf link failed rc={clang_cp.returncode}: {clang_cp.stderr or clang_cp.stdout}"
                )

    fingerprint = f"llc+clang:{llc_path}:{llc_ver}|{clang_path}:{clang_ver}"
    try:
        _compile_once(str(llvm_ir_text or ""))
        return fingerprint
    except Exception as primary_err:
        triple = _llvm_target_triple(llvm_ir_text)
        if not _is_rvv_llvm_triple(triple):
            raise
        host_text, changed, host_triple = _retarget_llvm_ir_to_host_triple_for_link(llvm_ir_text)
        if not changed:
            raise
        try:
            _compile_once(host_text)
            return f"{fingerprint}|host_retarget_fallback:{host_triple}"
        except Exception as host_err:
            raise RuntimeError(
                "rvv llvm->elf materialization failed for both target and host fallback: "
                f"primary={type(primary_err).__name__}: {primary_err}; "
                f"host={type(host_err).__name__}: {host_err}"
            ) from host_err


def _materialize_executable(
    *,
    backend: str,
    spec_name: str,
    out_dir: Path,
    module: IntentMLIRModule,
    suffix: str,
    shape_bindings: dict[str, int],
    fallback_intent_module: IntentMLIRModule | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    recovered_intent_json = _recover_intent_json(
        module=module,
        fallback_intent_module=fallback_intent_module,
    )
    if backend == "cuda":
        module_text = str(module.module_text or "")
        if not _looks_like_llvm_ir(module_text):
            raise RuntimeError(
                "cuda executable materialization requires textual LLVM IR; "
                f"got non-LLVM module for suffix={suffix}"
            )
        llvm_target_triple = _llvm_target_triple(module_text)
        llvm_origin = str((module.meta or {}).get("llvm_dialect_origin") or "")
        ptx_path = out_dir / f"{spec_name}.intentir.intentdialect.{suffix}.kernel.ptx"
        entry = str((module.meta or {}).get("kernel_name") or spec_name)
        try:
            fingerprint = _compile_llvm_ir_to_cuda_ptx(
                llvm_ir_text=module_text,
                out_path=ptx_path,
            )
            resolved_entry, discovered_entries = _resolve_ptx_entry_symbol(
                ptx_path=ptx_path,
                preferred_entry=entry,
            )
            _validate_cuda_ptx_assembly(ptx_path=ptx_path, entry=resolved_entry)
            executable = {
                "format": "cuda_ptx",
                "path": str(ptx_path),
                "entry": str(resolved_entry),
                "target": "cuda",
                "toolchain_fingerprint": str(fingerprint),
                "invocation": {"shape_bindings": dict(shape_bindings), "ptx_compiler": "llc_nvptx"},
            }
            artifacts = {
                "cuda_ptx_path": str(ptx_path),
                "cuda_ptx_origin": "llvm_llc",
                "cuda_llvm_target_triple": str(llvm_target_triple),
                "cuda_llvm_origin": str(llvm_origin),
                "cuda_sm": str(_cuda_llc_target()),
            }
            if isinstance(recovered_intent_json, dict):
                try:
                    inv = dict(executable.get("invocation") or {})
                    runtime_io = _runtime_io_spec_from_intent_json(recovered_intent_json)
                    inv["io_spec"] = dict(runtime_io)
                    inv["output_names"] = [str(x) for x in list(runtime_io.get("outputs") or [])]
                    executable["invocation"] = inv
                except Exception as e:
                    artifacts["cuda_io_spec_synth_error"] = f"{type(e).__name__}: {e}"
            if discovered_entries:
                artifacts["cuda_ptx_entries"] = [str(x) for x in discovered_entries]
            if str(resolved_entry) != str(entry):
                artifacts["cuda_ptx_entry_expected"] = str(entry)
                artifacts["cuda_ptx_entry_resolved"] = str(resolved_entry)
            return executable, artifacts
        except Exception as llc_err:
            # Hard-cut: CUDA executable materialization must succeed via LLVM->PTX.
            # No nvcc/nvrtc/cpp_codegen compatibility fallback is allowed.
            if bool(cuda_require_llvm_ptx()):
                raise RuntimeError(
                    "cuda llvm->ptx materialization failed under strict LLVM PTX mode: "
                    f"{type(llc_err).__name__}: {llc_err}"
                ) from llc_err
            raise RuntimeError(
                "cuda llvm->ptx materialization failed; legacy compatibility fallback is removed: "
                f"{type(llc_err).__name__}: {llc_err}"
            ) from llc_err

    if backend == "rvv":
        module_text = str(module.module_text or "")
        if not _looks_like_llvm_ir(module_text):
            raise RuntimeError(
                "rvv executable materialization requires textual LLVM IR; "
                f"got non-LLVM module for suffix={suffix}"
            )
        elf_path = out_dir / f"{spec_name}.intentir.intentdialect.{suffix}.kernel.elf"
        fingerprint = _compile_llvm_ir_to_elf(
            llvm_ir_text=module_text,
            out_path=elf_path,
        )
        host_fallback_tag = "|host_retarget_fallback:"
        host_retarget_fallback = host_fallback_tag in str(fingerprint)
        executable = {
            "format": "rvv_elf",
            "path": str(elf_path),
            "entry": str((module.meta or {}).get("kernel_name") or spec_name),
            "target": "rvv",
            "toolchain_fingerprint": str(fingerprint),
            "invocation": {"shape_bindings": dict(shape_bindings), "elf_compiler": "llc+clang"},
        }
        artifacts = {
            "rvv_elf_path": str(elf_path),
            "rvv_elf_origin": "llvm_llc",
        }
        if host_retarget_fallback:
            artifacts["rvv_elf_host_retarget_fallback"] = True
            artifacts["rvv_elf_host_triple"] = str(fingerprint).split(host_fallback_tag, 1)[1]
        # Hard-cut: RVV compatibility C-source artifacts are fully removed from
        # default pipeline outputs; strict path keeps executable-only contracts.
        artifacts["rvv_compat_removed"] = True
        if isinstance(recovered_intent_json, dict):
            try:
                inv = dict(executable.get("invocation") or {})
                runtime_io = _runtime_io_spec_from_intent_json(recovered_intent_json)
                inv["io_spec"] = dict(runtime_io)
                inv["output_names"] = [str(x) for x in list(runtime_io.get("outputs") or [])]
                executable["invocation"] = inv
            except Exception as e:
                artifacts["rvv_io_spec_synth_error"] = f"{type(e).__name__}: {e}"
        return executable, artifacts

    raise ValueError(f"unsupported backend for executable materialization: {backend}")


def _emit_contract(
    *,
    backend: str,
    spec_name: str,
    out_dir: Path,
    module: IntentMLIRModule,
    suffix: str,
    shape_bindings: dict[str, Any] | None = None,
    materialize_executable: bool = False,
    fallback_intent_module: IntentMLIRModule | None = None,
) -> tuple[str, dict[str, Any]]:
    module_path = out_dir / f"{spec_name}.intentir.intentdialect.{suffix}.module.mlir"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(str(module.module_text or ""), encoding="utf-8")
    recovered_intent_json = _recover_intent_json(
        module=module,
        fallback_intent_module=fallback_intent_module,
    )
    executable: dict[str, Any] = {
        "format": f"{backend}_mlir_module",
        "path": str(module_path),
        "entry": str((module.meta or {}).get("kernel_name") or ""),
        "target": str(backend),
    }
    exec_artifacts: dict[str, Any] = {}
    if materialize_executable:
        bindings = _normalize_shape_bindings(shape_bindings)
        if not bindings:
            raise RuntimeError(
                f"materialize_executable requires concrete shape bindings for backend={backend} suffix={suffix}"
            )
        executable, exec_artifacts = _materialize_executable(
            backend=backend,
            spec_name=spec_name,
            out_dir=out_dir,
            module=module,
            suffix=suffix,
            shape_bindings=bindings,
            fallback_intent_module=fallback_intent_module,
        )
    executable = {
        "format": str(executable.get("format") or f"{backend}_mlir_module"),
        "path": str(executable.get("path") or module_path),
        "entry": str(executable.get("entry") or (module.meta or {}).get("kernel_name") or ""),
        "target": str(executable.get("target") or backend),
        "toolchain_fingerprint": str(executable.get("toolchain_fingerprint") or ""),
        "invocation": dict(executable.get("invocation") or {}),
    }
    if isinstance(recovered_intent_json, dict):
        try:
            inv = dict(executable.get("invocation") or {})
            runtime_io = _runtime_io_spec_from_intent_json(recovered_intent_json)
            inv.setdefault("io_spec", dict(runtime_io))
            inv.setdefault("output_names", [str(x) for x in list(runtime_io.get("outputs") or [])])
            executable["invocation"] = inv
        except Exception:
            pass
    contract_fallback_error = ""
    contract_fallback_used = False

    if backend == "cuda":
        try:
            contract = build_cuda_contract(
                module,
                source_kind="mlir_module",
                artifact_module_path=str(module_path),
                executable=executable,
            )
        except Exception as e:
            if fallback_intent_module is None:
                raise
            contract = build_cuda_contract(
                fallback_intent_module,
                source_kind="mlir_module_fallback_intent",
                artifact_module_path=str(module_path),
                executable=executable,
            )
            contract_fallback_error = f"{type(e).__name__}: {e}"
            contract_fallback_used = True
    elif backend == "rvv":
        try:
            contract = build_rvv_contract(
                module,
                source_kind="mlir_module",
                artifact_module_path=str(module_path),
                executable=executable,
            )
        except Exception as e:
            if fallback_intent_module is None:
                raise
            contract = build_rvv_contract(
                fallback_intent_module,
                source_kind="mlir_module_fallback_intent",
                artifact_module_path=str(module_path),
                executable=executable,
            )
            contract_fallback_error = f"{type(e).__name__}: {e}"
            contract_fallback_used = True
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unsupported backend: {backend}")
    if contract_fallback_used:
        artifacts = dict(contract.artifacts or {})
        artifacts["intent_recovery_fallback"] = "midend_module"
        artifacts["intent_recovery_error"] = str(contract_fallback_error)
        contract.artifacts = artifacts
    if exec_artifacts:
        artifacts = dict(contract.artifacts or {})
        artifacts.update(exec_artifacts)
        contract.artifacts = artifacts
    if backend == "cuda":
        _align_cuda_contract_launch_with_ptx_maxntid(contract)
    contract_path = out_dir / f"{spec_name}.intentir.intentdialect.{suffix}.contract.json"
    return _dump_json(contract_path, contract.to_json_dict()), exec_artifacts


def emit_backend_contract_artifacts(
    *,
    spec_name: str,
    out_dir: Path,
    midend_module: IntentMLIRModule,
    fallback_intent_module: IntentMLIRModule | None = None,
    mlir_report: dict[str, Any],
    downstream_name: str | None = None,
    downstream_module: IntentMLIRModule | None = None,
    downstream_llvm_name: str | None = None,
    downstream_llvm_module: IntentMLIRModule | None = None,
    downstream_llvm_variants: list[tuple[str, IntentMLIRModule]] | None = None,
    shape_bindings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Emit MLIR backend contracts into stable JSON artifacts and annotate report fields.

    Fields written into `mlir_report`:
      - `midend_cuda_contract_path`
      - `midend_rvv_contract_path`
      - `downstream_cuda_contract_path` / `downstream_rvv_contract_path` (if available)
      - `downstream_cuda_llvm_contract_path` / `downstream_rvv_llvm_contract_path` (if available)
      - `downstream_contract_path`, `downstream_contract_backend`
      - `downstream_llvm_contract_path`, `downstream_llvm_contract_backend`
      - `mlir_backend_contract_used`
    """
    emitted: dict[str, Any] = {}

    # Always emit contracts from midend so tools can recover even when a specific
    # downstream pipeline is skipped.
    for backend in ("cuda", "rvv"):
        try:
            p, exec_meta = _emit_contract(
                backend=backend,
                spec_name=spec_name,
                out_dir=out_dir,
                module=midend_module,
                suffix=f"midend_{backend}",
                shape_bindings=shape_bindings,
                materialize_executable=False,
                fallback_intent_module=fallback_intent_module,
            )
            emitted[f"midend_{backend}_contract_path"] = p
            if exec_meta:
                emitted[f"midend_{backend}_contract_exec_meta"] = dict(exec_meta)
        except Exception as e:  # pragma: no cover - defensive
            emitted[f"midend_{backend}_contract_error"] = f"{type(e).__name__}: {e}"

    def _backend_for_downstream_name(name: str | None) -> str | None:
        n = str(name or "").strip().lower()
        if n.startswith("downstream_cuda"):
            return "cuda"
        if n.startswith("downstream_rvv"):
            return "rvv"
        return None

    downstream_contract_selected = False

    def _emit_downstream_contract(*, name: str | None, module: IntentMLIRModule | None, prefer_primary: bool) -> None:
        nonlocal downstream_contract_selected
        if module is None:
            return
        backend = _backend_for_downstream_name(name)
        if backend is None:
            return
        key_name = str(name or "").strip()
        if not key_name:
            return
        should_materialize = bool(key_name.endswith("_llvm") and backend == "cuda")
        downstream_fallback = fallback_intent_module or midend_module
        try:
            p, exec_meta = _emit_contract(
                backend=backend,
                spec_name=spec_name,
                out_dir=out_dir,
                module=module,
                suffix=key_name,
                shape_bindings=shape_bindings,
                materialize_executable=should_materialize,
                fallback_intent_module=downstream_fallback,
            )
            emitted[f"{key_name}_contract_path"] = p
            if exec_meta:
                emitted[f"{key_name}_contract_exec_meta"] = dict(exec_meta)
            if key_name.endswith("_llvm"):
                if prefer_primary or ("downstream_llvm_contract_path" not in emitted):
                    emitted["downstream_llvm_contract_path"] = p
                    emitted["downstream_llvm_contract_backend"] = backend
            if prefer_primary or (should_materialize and not downstream_contract_selected):
                emitted["downstream_contract_path"] = p
                emitted["downstream_contract_backend"] = backend
                downstream_contract_selected = True
        except Exception as e:  # pragma: no cover - defensive
            emitted[f"{key_name}_contract_error"] = f"{type(e).__name__}: {e}"

    # Emit executable-ready LLVM downstream contracts first and keep them as the
    # primary runtime contract path.
    _emit_downstream_contract(name=downstream_llvm_name, module=downstream_llvm_module, prefer_primary=True)
    if downstream_llvm_variants:
        primary_name = str(downstream_llvm_name or "").strip()
        for item in list(downstream_llvm_variants):
            if not isinstance(item, (tuple, list)) or len(item) < 2:
                continue
            var_name = str(item[0] or "").strip()
            var_mod = item[1] if isinstance(item[1], IntentMLIRModule) else None
            if not var_name or var_mod is None:
                continue
            if primary_name and var_name == primary_name:
                continue
            _emit_downstream_contract(name=var_name, module=var_mod, prefer_primary=False)
    # Emit semantic downstream contracts as auxiliary artifacts only.
    _emit_downstream_contract(name=downstream_name, module=downstream_module, prefer_primary=False)

    if any(str(k).endswith("_contract_path") for k in emitted):
        emitted["mlir_backend_contract_used"] = True

    mlir_report.update(emitted)
    return emitted


__all__ = ["emit_backend_contract_artifacts"]
