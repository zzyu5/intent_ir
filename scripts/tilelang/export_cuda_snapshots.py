"""
Offline TileLang -> CUDA snapshot exporter.

Goal: keep the CUDA pipeline runtime-independent from TileLang.

This script is the *only* place that imports TileLang/TIR to:
  - compile TileLang PrimFunc to CUDA C + PTX
  - extract launch config (grid/block) from TIR `T.launch_thread(...)`
  - write a self-contained snapshot under `kernels/cuda/ops/snapshots/`

The CUDA frontend (`pipeline/cuda/core.py`) consumes only these snapshots:
  - `<name>.cu`
  - `<name>.ptx`
  - `<name>.cuda_snapshot.json` (extended schema; includes io_spec + launch)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontends.tilelang.cuda_export import export_tilelang_cuda
from frontends.tilelang.runtime import infer_written_global_buffers
from intent_ir.ir import Dim, IntentFunction


_LAUNCH_THREAD_RE = re.compile(
    r'^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*T\.launch_thread\(\"(?P<tag>(?:blockIdx|threadIdx)\.(?:x|y|z))\",\s*(?P<extent>.+)\)\s*$'
)


def _shape_list_from_intent_dim_list(dims: List[Dim]) -> List[int | str]:
    out: List[int | str] = []
    for d in dims:
        if d.kind == "const":
            out.append(int(d.value))
        elif d.kind == "sym":
            out.append(str(d.value))
        else:
            # Best-effort: keep as string.
            out.append(str(d.value))
    return out


def _merge_semantic_io_spec(*, runtime_io: Dict[str, Any], intent: IntentFunction, arg_names: List[str]) -> Dict[str, Any]:
    """
    Merge:
      - runtime io_spec (superset of pointer params; may have constant shapes)
      - deterministic IntentIR shapes (symbolic; may omit runtime-only buffers)

    The result stays a superset of runtime pointer params (required for signature parsing),
    while recovering symbolic shapes (needed for LLM semantics and casegen bindings).
    """
    rt_tensors = runtime_io.get("tensors") if isinstance(runtime_io.get("tensors"), dict) else {}
    rt_scalars = runtime_io.get("scalars") if isinstance(runtime_io.get("scalars"), dict) else {}

    tensors: Dict[str, Dict[str, Any]] = {}
    scalars: Dict[str, str] = {str(k): str(v) for k, v in rt_scalars.items()}

    # Start from runtime tensor specs (must cover all pointer params).
    for name, ts in rt_tensors.items():
        if not isinstance(ts, dict):
            continue
        tensors[str(name)] = dict(ts)

    # Overlay tensor shapes from the deterministic intent for IO tensors that exist there.
    for name in list(tensors.keys()):
        if name not in intent.tensors:
            continue
        t = intent.tensors[name]
        tensors[name]["dtype"] = str(t.dtype)
        tensors[name]["shape"] = _shape_list_from_intent_dim_list(list(t.shape))
        # Ensure all symbolic dims appear in scalars (even if runtime signature drops them).
        for d in list(t.shape):
            if d.kind == "sym":
                scalars.setdefault(str(d.value), "i32")

    # Ensure all scalar arg_names exist.
    for a in arg_names:
        if a not in tensors and a not in scalars:
            scalars[str(a)] = "i32"

    return {"arg_names": list(arg_names), "tensors": tensors, "scalars": scalars}


def _strip_trailing_comment(expr: str) -> str:
    s = str(expr).strip()
    if "#" in s:
        s = s.split("#", 1)[0].strip()
    return s


def _extract_launch_threads_from_tir(prim_func: Any) -> Dict[str, str]:
    txt = str(prim_func.script(show_meta=False))
    ext: Dict[str, str] = {}
    for ln in txt.splitlines():
        m = _LAUNCH_THREAD_RE.match(ln.strip())
        if not m:
            continue
        tag = str(m.group("tag"))
        e = _strip_trailing_comment(str(m.group("extent")))
        ext[tag] = e
    return ext


def _default_launch(ext: Dict[str, str]) -> Dict[str, Any]:
    """
    Normalize parsed launch_thread extents into a JSON-friendly launch dict.
    """
    grid = [ext.get("blockIdx.x", "1"), ext.get("blockIdx.y", "1"), ext.get("blockIdx.z", "1")]
    block = [ext.get("threadIdx.x", "1"), ext.get("threadIdx.y", "1"), ext.get("threadIdx.z", "1")]
    return {"grid": grid, "block": block, "shared_mem": 0}


@dataclass(frozen=True)
class SnapshotMeta:
    name: str
    origin: str
    entry_name: str
    include_dirs: List[str]
    io_spec: Dict[str, Any]
    outputs: List[str]
    canonical_shapes: Dict[str, int]
    vary_axes: List[str]
    exclude_axes: List[str]
    launch: Dict[str, Any]
    notes: str = ""
    # Optional extra evidence to help LLM (pure text; CUDA pipeline must not parse it).
    semantic_guide: Optional[str] = None
    baseline: str = "cuda"  # "cuda" or "numpy"


def _export_one(
    *,
    spec: Any,
    out_dir: Path,
    refresh: bool,
) -> Tuple[SnapshotMeta, Path, Path, Path]:
    name = str(getattr(spec, "name"))
    prim_func = getattr(spec, "prim_func")
    arg_names = list(getattr(spec, "arg_names") or [])
    canonical_shapes = {str(k): int(v) for k, v in (getattr(spec, "canonical_shapes") or {}).items()}
    vary_axes = [str(x) for x in (getattr(spec, "vary_axes") or [])]
    exclude_axes = [str(x) for x in (getattr(spec, "exclude_axes") or [])]
    runtime_ref_ok = bool(getattr(spec, "runtime_ref_ok", True))
    baseline_kind = "cuda" if runtime_ref_ok else "numpy"

    out_dir.mkdir(parents=True, exist_ok=True)
    cu_path = out_dir / f"{name}.cu"
    ptx_path = out_dir / f"{name}.ptx"
    meta_path = out_dir / f"{name}.cuda_snapshot.json"

    if (not refresh) and cu_path.is_file() and ptx_path.is_file() and meta_path.is_file():
        # Load existing meta and trust it (for incremental workflow).
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict) and existing.get("io_spec") and existing.get("launch"):
            meta = SnapshotMeta(
                name=str(existing.get("name") or name),
                origin=str(existing.get("origin") or "tilelang"),
                entry_name=str(existing.get("entry_name") or existing.get("ptx_entry") or "main_kernel"),
                include_dirs=[str(x) for x in (existing.get("include_dirs") or [])],
                io_spec=dict(existing.get("io_spec") or {}),
                outputs=[str(x) for x in (existing.get("outputs") or [])],
                canonical_shapes={str(k): int(v) for k, v in (existing.get("canonical_shapes") or canonical_shapes).items()},
                vary_axes=[str(x) for x in (existing.get("vary_axes") or vary_axes)],
                exclude_axes=[str(x) for x in (existing.get("exclude_axes") or exclude_axes)],
                launch=dict(existing.get("launch") or {}),
                notes=str(existing.get("notes") or ""),
                semantic_guide=(str(existing.get("semantic_guide")) if isinstance(existing.get("semantic_guide"), str) else None),
                baseline=str(existing.get("baseline") or baseline_kind),
            )
            return meta, cu_path, ptx_path, meta_path

    # Compile TileLang kernel and export CUDA/PTX.
    #
    # Important: export into an artifacts temp dir, NOT under kernels/, so the
    # snapshot directory stays clean (no *.tilelang_kernel.cu / *.tilelang.ptx).
    tmp_dir = ROOT / "artifacts" / "tilelang_cuda_export_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    exp = export_tilelang_cuda(prim_func, out_dir=tmp_dir, stem=name)
    # Persist stable filenames expected by the CUDA pipeline.
    cu_path.write_text(str(exp.cuda_src), encoding="utf-8")
    ptx_path.write_text(str(exp.ptx_text), encoding="utf-8")

    # Build semantic io_spec by merging runtime pointer params + deterministic intent shapes.
    from frontends.tilelang.cuda_export import build_io_spec_from_tilelang_prim_func  # noqa: PLC0415

    runtime_io = build_io_spec_from_tilelang_prim_func(prim_func)
    intent = None
    try:
        intent = getattr(spec, "intent_builder")()
    except Exception:
        intent = None
    if intent is None:
        # Fallback: keep runtime io spec only.
        io_spec = dict(runtime_io)
        io_spec["arg_names"] = list(arg_names) if arg_names else list(runtime_io.get("arg_names") or [])
    else:
        io_spec = _merge_semantic_io_spec(runtime_io=runtime_io, intent=intent, arg_names=(arg_names or list(runtime_io.get("arg_names") or [])))

    outputs = list(infer_written_global_buffers(prim_func))
    launch_ext = _extract_launch_threads_from_tir(prim_func)
    launch = _default_launch(launch_ext)

    semantic_guide = None
    try:
        semantic_guide = str(prim_func.script(show_meta=False))
    except Exception:
        semantic_guide = None

    meta = SnapshotMeta(
        name=name,
        origin="tilelang",
        entry_name=str(exp.entry_name),
        include_dirs=[str(p) for p in (exp.include_dirs or [])],
        io_spec=io_spec,
        outputs=[str(x) for x in outputs],
        canonical_shapes=canonical_shapes,
        vary_axes=vary_axes,
        exclude_axes=exclude_axes,
        launch=launch,
        notes="Generated snapshot. Edit TileLang kernels under kernels/tilelang/ops/* and re-run this exporter.",
        semantic_guide=semantic_guide,
        baseline=baseline_kind,
    )

    meta_path.write_text(json.dumps(asdict(meta), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta, cu_path, ptx_path, meta_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["smoke", "coverage", "all"], default="smoke")
    ap.add_argument("--kernel", action="append", default=[], help="repeatable; filter by kernel name")
    ap.add_argument("--out-dir", default=str(ROOT / "kernels" / "cuda" / "ops" / "snapshots"))
    ap.add_argument("--refresh", action="store_true", help="re-export even if snapshots already exist")
    args = ap.parse_args()

    wanted = set(args.kernel or [])
    out_dir = Path(str(args.out_dir))

    # Build suite from TileLang pipeline specs (single source of truth).
    from pipeline.tilelang.core import coverage_kernel_specs, default_kernel_specs  # noqa: PLC0415

    if args.suite == "smoke":
        specs = list(default_kernel_specs())
    else:
        specs = list(coverage_kernel_specs())

    exported: List[Dict[str, Any]] = []
    for s in specs:
        if wanted and str(getattr(s, "name", "")) not in wanted:
            continue
        meta, cu_path, ptx_path, meta_path = _export_one(spec=s, out_dir=out_dir, refresh=bool(args.refresh))
        exported.append(
            {
                "name": meta.name,
                "baseline": meta.baseline,
                "cu": str(cu_path.relative_to(ROOT)),
                "ptx": str(ptx_path.relative_to(ROOT)),
                "meta": str(meta_path.relative_to(ROOT)),
            }
        )
        print(f"[export] {meta.name}: {cu_path.name} {ptx_path.name} {meta_path.name}", file=sys.stderr, flush=True)

    print(json.dumps({"ok": True, "count": len(exported), "exported": exported}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
