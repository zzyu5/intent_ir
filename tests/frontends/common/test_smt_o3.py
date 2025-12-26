from frontends.common.evidence import AccessSummary, IndexExpr, Predicate
from frontends.common.smt_o3 import check_mask_implies_inbounds


def test_o3_pass_with_affine_nonneg_minbound():
    # idx = pid0 + r0, predicate includes idx < M, and pid0,r0 are non-negative.
    accesses = [
        AccessSummary(
            kind="load",
            tensor="X",
            dtype="fp32",
            rank=1,
            index_exprs=[IndexExpr(terms={"pid0": 1, "r0": 1}, const=0)],
            predicate=Predicate(clauses=["pid0 + r0 < M"]),
            address_space="global",
        )
    ]
    rep = check_mask_implies_inbounds(accesses, shape_hints={"M": 8}, symbol_ranges={"r0": {"start": 0, "end": 4}})
    assert rep.status == "PASS"
    assert rep.failed == 0


def test_o3_unknown_when_missing_upper_bound_clause():
    accesses = [
        AccessSummary(
            kind="load",
            tensor="X",
            dtype="fp32",
            rank=1,
            index_exprs=[IndexExpr(terms={"pid0": 1}, const=0)],
            predicate=Predicate(clauses=["pid0 >= 0"]),
            address_space="global",
        )
    ]
    rep = check_mask_implies_inbounds(accesses, shape_hints={"M": 8})
    assert rep.status == "UNKNOWN"
    assert rep.unknown >= 1


def test_o3_fail_emits_counterexample_model():
    # idx = pid0 - 1 can be negative even if idx < M holds; produce a counterexample.
    accesses = [
        AccessSummary(
            kind="load",
            tensor="X",
            dtype="fp32",
            rank=1,
            index_exprs=[IndexExpr(terms={"pid0": 1}, const=-1)],
            predicate=Predicate(clauses=["pid0 - 1 < M"]),
            address_space="global",
        )
    ]
    rep = check_mask_implies_inbounds(accesses, shape_hints={"M": 1})
    assert rep.status == "FAIL"
    assert rep.failed >= 1
    # Ensure at least one dim has a concrete model.
    found = False
    for a in rep.access_checks:
        for d in a.dims:
            if d.status == "FAIL" and d.counterexample is not None:
                found = True
                assert "pid0" in d.counterexample.assignments
    assert found

