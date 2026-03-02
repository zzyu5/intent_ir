from __future__ import annotations

import pytest

from pipeline.common.evidence_mode import evidence_enabled, evidence_mode


def test_evidence_mode_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTENTIR_EVIDENCE_MODE", raising=False)
    assert evidence_mode() == "on"
    assert evidence_enabled() is True


def test_evidence_mode_off_disables_mlir_shadow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTENTIR_EVIDENCE_MODE", "off")
    monkeypatch.delenv("INTENTIR_MLIR_SHADOW", raising=False)
    from pipeline.triton import core as triton_core  # local import avoids triton import for unrelated tests

    assert evidence_mode() == "off"
    assert evidence_enabled() is False
    assert triton_core._mlir_shadow_mode_enabled() is False
