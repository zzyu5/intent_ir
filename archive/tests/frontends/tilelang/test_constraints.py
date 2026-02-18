from __future__ import annotations

from pipeline.interfaces import KernelDescriptor

from frontends.common.evidence import AccessSummary, IndexExpr, Predicate
from frontends.tilelang.constraints import extract_constraints
from frontends.tilelang.facts import TileLangFacts


def test_tilelang_constraints_include_access_witness():
    desc = KernelDescriptor(schema_version="kernel_desc_v1.0", name="k", frontend="tilelang", source_kind="ir", source_text="")
    desc.launch = {"canonical_shapes": {"M": 16, "N": 64}}

    facts = TileLangFacts(
        schema_version="tilelang_facts_v1.0",
        anchors={"kernel_kind_hint": "unknown"},
        accesses=[
            AccessSummary(
                kind="load",
                tensor="X",
                dtype="f32",
                rank=2,
                index_exprs=[IndexExpr(terms={"r0": 64, "r1": 1}, const=0)],
                predicate=Predicate(clauses=["r1 < N"]),
            ),
            AccessSummary(
                kind="store",
                tensor="Y",
                dtype="f32",
                rank=2,
                index_exprs=[IndexExpr(terms={"r0": 64, "r1": 1}, const=0)],
                predicate=None,
            ),
        ],
        symbol_ranges={"r0": {"start": 0, "end": 16}, "r1": {"start": 0, "end": 64}},
        tile_hints=[16, 64],
        raw={},
    )

    c = extract_constraints(desc, facts)
    assert c.needs_mask is True
    assert "access_witness" in c.meta
    aw = c.meta["access_witness"]
    assert isinstance(aw, dict)
    assert isinstance(aw.get("accesses"), list)
