from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping

from intent_ir.ir import IntentFunction


def _cpp_codegen_dir() -> Path:
    # backends/cuda/cpp_codegen (C++ host tool)
    return Path(__file__).resolve().parents[1] / "cpp_codegen"


def _cpp_codegen_build_dir() -> Path:
    override = os.getenv("INTENTIR_CUDA_CPP_CODEGEN_BUILD_DIR")
    if override:
        return Path(override).expanduser().resolve()

    # Keep build artifacts out of the repo tree by default.
    cache_root = Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser()
    src_dir = _cpp_codegen_dir()
    src_tag = hashlib.sha1(str(src_dir).encode("utf-8")).hexdigest()[:10]
    return (cache_root / "intentir" / "cuda_cpp_codegen" / src_tag).resolve()


def _cpp_codegen_bin(*, build_type: str) -> Path:
    return _cpp_codegen_build_dir() / str(build_type).lower() / "intentir_cuda_codegen"


def ensure_cpp_codegen_built(*, build_type: str = "Release") -> Path:
    """
    Ensure the C++ IntentIR->CUDA codegen binary is built and return its path.

    The backend tool lives in `backends/cuda/cpp_codegen/` and parses IntentIR
    JSON directly. Python remains orchestration only.
    """
    bin_path = _cpp_codegen_bin(build_type=build_type)
    src_dir = _cpp_codegen_dir()
    build_dir = _cpp_codegen_build_dir() / str(build_type).lower()
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
        raise RuntimeError(f"cuda cpp codegen: cmake configure failed:\n{res.stderr or res.stdout}")

    build = ["cmake", "--build", str(build_dir), "-j"]
    res = subprocess.run(build, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"cuda cpp codegen: cmake build failed:\n{res.stderr or res.stdout}")

    if not bin_path.exists():
        raise RuntimeError(f"cuda cpp codegen: build succeeded but binary not found at {bin_path}")
    return bin_path


def lower_intent_to_cuda_kernel_cpp(
    intent: IntentFunction,
    *,
    bindings: Mapping[str, Any],
    build_type: str = "Release",
) -> Dict[str, Any]:
    """
    Lower `IntentFunction` into a single CUDA kernel by invoking the C++ backend codegen.

    Returns a plain JSON-like dict with keys compatible with `CudaLoweredKernel`:
      - kernel_name, cuda_src, io_spec, launch, output_names, bindings
    """
    bin_path = ensure_cpp_codegen_built(build_type=build_type)
    intent_json = intent.to_json_dict()
    bindings_json = dict(bindings)

    with tempfile.TemporaryDirectory(prefix="intentir_cuda_cpp_codegen_") as td:
        td_path = Path(td)
        intent_path = td_path / "intent.json"
        bindings_path = td_path / "bindings.json"
        intent_path.write_text(json.dumps(intent_json, indent=2))
        bindings_path.write_text(json.dumps(bindings_json, indent=2))

        cmd = [str(bin_path), "--intent", str(intent_path), "--bindings", str(bindings_path)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"cuda cpp codegen failed (rc={res.returncode}):\n{res.stderr or res.stdout}")
        out = (res.stdout or "").strip()
        if not out:
            raise RuntimeError("cuda cpp codegen returned empty output on stdout")
        try:
            j = json.loads(out)
        except Exception as e:
            raise RuntimeError(f"cuda cpp codegen returned non-JSON output:\n{out[:400]}") from e
        if not isinstance(j, dict):
            raise RuntimeError("cuda cpp codegen output must be a JSON object")
        return j


__all__ = ["ensure_cpp_codegen_built", "lower_intent_to_cuda_kernel_cpp"]
