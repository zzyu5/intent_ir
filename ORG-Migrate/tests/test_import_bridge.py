from __future__ import annotations

from pipeline.triton.org_bridge import load_org_module


def test_import_bridge_loads_org_schema() -> None:
    mod = load_org_module("org.schema")
    assert getattr(mod, "OrgDoc").__name__ == "OrgDoc"
