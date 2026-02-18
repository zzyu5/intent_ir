from __future__ import annotations

import difflib
import json
import os
import shutil
from pathlib import Path

import pytest

from pipeline import registry
from pipeline.cuda.core import native_kernel_specs


def _repo_root() -> Path:
    # tests/frontends/cuda/test_*.py -> repo root
    return Path(__file__).resolve().parents[3]


def _golden_dir() -> Path:
    return _repo_root() / "tests" / "golden" / "cuda"


def _find_spec(name: str):
    for s in native_kernel_specs():
        if s.name == name:
            return s
    raise KeyError(f"unknown kernel spec: {name}")


def _nvcc_available() -> bool:
    return shutil.which("nvcc") is not None


def _extract_semantic_facts(spec, *, tmp_path: Path) -> dict:
    adapter = registry.get("cuda")
    desc = adapter.build_descriptor(spec)
    desc.meta["artifact_dir"] = str(tmp_path)
    desc = adapter.ensure_artifacts(desc, spec)
    facts = adapter.extract_facts(desc)
    cert_v2 = adapter.build_certificate(desc, facts, None)
    return dict(cert_v2.to_json_dict()["semantic_facts"])


def _diff(a: dict, b: dict) -> str:
    sa = json.dumps(a, indent=2, sort_keys=True).splitlines(keepends=True)
    sb = json.dumps(b, indent=2, sort_keys=True).splitlines(keepends=True)
    return "".join(difflib.unified_diff(sb, sa, fromfile="golden", tofile="current"))


@pytest.mark.skipif(not _nvcc_available(), reason="nvcc not available")
@pytest.mark.parametrize("kernel_name", ["vec_add", "transpose2d", "row_sum"])
def test_cuda_certificate_semantic_facts_golden(kernel_name: str, tmp_path: Path):
    """
    Lock only CertificateV2.semantic_facts (schedule_hints may drift).

    Note: semantic_facts stability depends on PTX parsing + NVCC stability.
    We compile with -O0 to keep patterns stable.
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
