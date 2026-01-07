from __future__ import annotations

import difflib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

try:
    import torch
except Exception:
    torch = None

from frontends.triton.certificate import build_certificate_v2
from frontends.triton.facts import extract_facts
from pipeline import registry
from pipeline.triton.core import default_kernel_specs


def _repo_root() -> Path:
    # tests/frontends/triton/test_*.py -> repo root
    return Path(__file__).resolve().parents[3]


def _golden_dir() -> Path:
    return _repo_root() / "tests" / "golden" / "triton"


def _find_spec(name: str):
    for s in default_kernel_specs():
        if s.name == name:
            return s
    raise KeyError(f"unknown kernel spec: {name}")


def _extract_semantic_facts(spec, *, tmp_path: Path) -> dict:
    adapter = registry.get("triton")
    desc = adapter.build_descriptor(spec)
    desc.meta["artifact_dir"] = str(tmp_path)
    desc = adapter.ensure_artifacts(desc, spec)
    ttir_path = desc.artifacts.ttir_path
    if not ttir_path:
        raise RuntimeError("TTIR not produced; check Triton/CUDA availability")
    ttir = Path(str(ttir_path)).read_text(encoding="utf-8")
    facts = extract_facts(ttir)
    cert_v2 = build_certificate_v2(ttir, desc=desc, facts=facts)
    return dict(cert_v2.to_json_dict()["semantic_facts"])


def _diff(a: dict, b: dict) -> str:
    sa = json.dumps(a, indent=2, sort_keys=True).splitlines(keepends=True)
    sb = json.dumps(b, indent=2, sort_keys=True).splitlines(keepends=True)
    return "".join(difflib.unified_diff(sb, sa, fromfile="golden", tofile="current"))


def _triton_import_ok() -> bool:
    """
    Importing `triton` can hard-crash the interpreter on machines without a
    working CUDA/driver stack. Probe in a subprocess so pytest collection stays
    robust and we can skip cleanly.
    """
    if importlib.util.find_spec("triton") is None:
        return False
    try:
        p = subprocess.run(
            [sys.executable, "-c", "import triton; print('ok')"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=30,
        )
        return p.returncode == 0
    except Exception:
        return False


def _cuda_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


_TRITON_OK = _triton_import_ok()


@pytest.mark.skipif(not _TRITON_OK, reason="triton import probe failed (no CUDA/driver?)")
@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.parametrize("kernel_name", ["softmax_inner", "layer_norm_persistent"])
def test_triton_certificate_semantic_facts_golden(kernel_name: str, tmp_path: Path):
    """
    PR#8: lock only CertificateV2.semantic_facts (schedule_hints may drift).
    """
    spec = _find_spec(kernel_name)
    got = _extract_semantic_facts(spec, tmp_path=tmp_path / kernel_name)

    golden_dir = _golden_dir()
    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_path = golden_dir / f"{kernel_name}.semantic_facts.json"

    if os.environ.get("UPDATE_GOLDEN") == "1":
        golden_path.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    if not golden_path.exists():
        raise AssertionError(f"missing golden file: {golden_path} (run with UPDATE_GOLDEN=1)")

    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    assert got == expected, _diff(got, expected)
