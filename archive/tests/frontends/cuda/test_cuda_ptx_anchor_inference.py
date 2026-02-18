from __future__ import annotations

import shutil

import pytest

from pipeline import registry
from pipeline.cuda.core import native_kernel_specs


def _nvcc_available() -> bool:
    return shutil.which("nvcc") is not None


@pytest.mark.skipif(not _nvcc_available(), reason="nvcc not available")
def test_cuda_ptx_infers_dot_and_reduce_for_naive_gemm(tmp_path):
    spec = next(s for s in native_kernel_specs() if s.name == "naive_gemm")
    adapter = registry.get("cuda")
    desc = adapter.build_descriptor(spec)
    desc.meta["artifact_dir"] = str(tmp_path)
    desc = adapter.ensure_artifacts(desc, spec)
    facts = adapter.extract_facts(desc)
    cert = adapter.build_certificate(desc, facts, None)
    sf = cert.to_json_dict()["semantic_facts"]
    anchors = sf["anchors"]
    assert anchors.get("has_reduce") is True
    assert anchors.get("has_dot") is True

