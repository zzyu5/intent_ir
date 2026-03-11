from __future__ import annotations

import pytest

pytest.importorskip("triton")
pytest.importorskip("torch")

from pipeline.triton.core import _run_org_plugin


def test_org_mode_off_does_not_mutate_report(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("INTENTIR_ORG_MODE", raising=False)  # default=off
    report: dict[str, object] = {}
    _run_org_plugin(
        spec_name="flash_attention2d",
        out_dir=tmp_path,
        desc=None,
        intent=None,
        report=report,
        shape_bindings={},
        triton_provider="native",
        backend_target=None,
    )
    assert "org" not in report
