from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from intent_ir.ir import IntentFunction


def _cpp_codegen_dir() -> Path:
    # backends/spmd_rvv/cpp_codegen (C++ host tool)
    return Path(__file__).resolve().parents[1] / "cpp_codegen"


def _cpp_codegen_build_dir() -> Path:
    override = os.getenv("INTENTIR_CPP_CODEGEN_BUILD_DIR")
    if override:
        return Path(override).expanduser().resolve()

    # Keep build artifacts out of the repo tree by default.
    cache_root = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser()
    src_dir = _cpp_codegen_dir()
    src_tag = hashlib.sha1(str(src_dir).encode("utf-8")).hexdigest()[:10]
    return (cache_root / "intentir" / "cpp_codegen" / src_tag).resolve()


def _cpp_codegen_bin(*, build_type: str) -> Path:
    return _cpp_codegen_build_dir() / str(build_type).lower() / "intentir_codegen"


def ensure_cpp_codegen_built(*, build_type: str = "Release") -> Path:
    """
    Ensure the C++ IntentIR->C codegen binary is built and return its path.

    The backend tool lives in `backends/spmd_rvv/cpp_codegen/` and parses IntentIR
    JSON directly (no extra backend IR). Python remains orchestration only.
    """
    bin_path = _cpp_codegen_bin(build_type=build_type)
    src_dir = _cpp_codegen_dir()
    build_dir = _cpp_codegen_build_dir()
    build_dir = build_dir / str(build_type).lower()
    build_dir.mkdir(parents=True, exist_ok=True)

    def sources_newer_than_bin() -> bool:
        if not bin_path.exists():
            return True
        try:
            bin_mtime = bin_path.stat().st_mtime
        except FileNotFoundError:
            return True
        for p in src_dir.rglob("*"):
            if build_dir in p.parents:
                continue
            if not p.is_file():
                continue
            if p.name == "CMakeLists.txt" or p.suffix in {".cpp", ".cc", ".c", ".h", ".hpp", ".cmake"}:
                try:
                    if p.stat().st_mtime > bin_mtime:
                        return True
                except FileNotFoundError:
                    continue
        return False

    if not sources_newer_than_bin():
        return bin_path

    cfg = [
        "cmake",
        "-S",
        str(src_dir),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    res = subprocess.run(cfg, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"cpp codegen: cmake configure failed:\n{res.stderr or res.stdout}")

    build = ["cmake", "--build", str(build_dir), "-j"]
    res = subprocess.run(build, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"cpp codegen: cmake build failed:\n{res.stderr or res.stdout}")

    if not bin_path.exists():
        raise RuntimeError(f"cpp codegen: build succeeded but binary not found at {bin_path}")
    return bin_path


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


__all__ = ["ensure_cpp_codegen_built", "lower_intent_to_c_with_files_cpp"]
