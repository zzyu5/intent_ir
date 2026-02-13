from __future__ import annotations

from pathlib import Path

from pipeline.triton.providers import get_provider_plugin


ROOT = Path(__file__).resolve().parents[3]


def test_core_does_not_import_legacy_flaggems_modules_directly() -> None:
    core = (ROOT / "pipeline" / "triton" / "core.py").read_text(encoding="utf-8")
    assert "from pipeline.triton.flaggems_" not in core
    assert "import pipeline.triton.flaggems_" not in core


def test_flaggems_plugin_resolved_from_provider_registry() -> None:
    plugin = get_provider_plugin("flaggems")
    assert plugin.name == "flaggems"
    assert plugin.require_source_and_state is True
