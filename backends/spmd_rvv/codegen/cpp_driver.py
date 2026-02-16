from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from intent_ir.ir import IntentFunction
from backends.common.cpp_build import (
    ensure_cmake_binary_built,
    resolve_binary_path,
    resolve_build_root,
)


def _cpp_codegen_dir() -> Path:
    # backends/spmd_rvv/cpp_codegen (C++ host tool)
    return Path(__file__).resolve().parents[1] / "cpp_codegen"


def _cpp_codegen_build_dir() -> Path:
    return resolve_build_root(
        _cpp_codegen_dir(),
        env_var="INTENTIR_CPP_CODEGEN_BUILD_DIR",
        namespace="cpp_codegen",
    )


def _cpp_codegen_bin(*, build_type: str) -> Path:
    return resolve_binary_path(
        _cpp_codegen_build_dir(),
        build_type=str(build_type),
        binary_name="intentir_codegen",
    )


def ensure_cpp_codegen_built(*, build_type: str = "Release") -> Path:
    """
    Ensure the C++ IntentIR->C codegen binary is built and return its path.

    The backend tool lives in `backends/spmd_rvv/cpp_codegen/` and parses IntentIR
    JSON directly (no extra backend IR). Python remains orchestration only.
    """
    src_dir = _cpp_codegen_dir()
    build_dir = _cpp_codegen_build_dir() / str(build_type).lower()
    bin_path = _cpp_codegen_bin(build_type=build_type)
    return ensure_cmake_binary_built(
        source_dir=src_dir,
        build_dir=build_dir,
        binary_path=bin_path,
        build_type=str(build_type),
        label="cpp codegen",
    )


def lower_intent_to_c_with_files_cpp(
    intent: IntentFunction,
    *,
    shape_bindings: Mapping[str, Any],
    atol: float = 1e-3,
    rtol: float = 1e-3,
    mode: str = "verify",
    build_type: str = "Release",
) -> str:
    """
    Lower `IntentFunction` to standalone C by invoking the C++ backend codegen.

    The generated C reads `<tensor>.bin` inputs, computes outputs, compares
    against `<output>_ref.bin`, and prints PASS/FAIL.
    """
    bin_path = ensure_cpp_codegen_built(build_type=build_type)

    # The C++ tool expects a plain JSON mapping {symbol: int}.
    shapes: dict[str, int] = {}
    for k, v in dict(shape_bindings).items():
        try:
            shapes[str(k)] = int(v)
        except Exception:
            continue

    intent_json = intent.to_json_dict()

    with tempfile.TemporaryDirectory(prefix="intentir_cpp_codegen_") as td:
        td_path = Path(td)
        intent_path = td_path / "intent.json"
        shapes_path = td_path / "shapes.json"
        intent_path.write_text(json.dumps(intent_json, indent=2))
        shapes_path.write_text(json.dumps(shapes, indent=2))

        cmd = [
            str(bin_path),
            "--intent",
            str(intent_path),
            "--shapes",
            str(shapes_path),
            "--mode",
            str(mode),
            "--atol",
            str(float(atol)),
            "--rtol",
            str(float(rtol)),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"cpp codegen failed (rc={res.returncode}):\n{res.stderr or res.stdout}")
        if not res.stdout.strip():
            raise RuntimeError("cpp codegen returned empty C source on stdout")
        return res.stdout


def lower_intent_to_c_with_files(
    intent: IntentFunction,
    *,
    shape_bindings: Mapping[str, Any],
    atol: float = 1e-3,
    rtol: float = 1e-3,
    mode: str = "verify",
    build_type: str = "Release",
) -> str:
    return lower_intent_to_c_with_files_cpp(
        intent,
        shape_bindings=shape_bindings,
        atol=float(atol),
        rtol=float(rtol),
        mode=str(mode),
        build_type=str(build_type),
    )


__all__ = ["ensure_cpp_codegen_built", "lower_intent_to_c_with_files_cpp", "lower_intent_to_c_with_files"]
