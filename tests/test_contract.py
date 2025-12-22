import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from triton_frontend.facts import TTIRFacts, extract_constraints, extract_facts
from triton_frontend.contract import ContractReport, evaluate_contract


def test_contract_out_when_no_anchor():
    ttir = "module { tt.func @foo() { tt.return } }"
    facts = extract_facts(ttir)
    report = evaluate_contract(facts)
    assert report.level == "OUT_OF_SCOPE"
    assert "no dot or reduce" in report.reasons[0]


def test_contract_out_when_atomic():
    ttir = "module { tt.func @foo() { %0 = tt.atomic_rmw %ptr, %val {addi} : tensor<1xf32> } }"
    facts = extract_facts(ttir)
    report = evaluate_contract(facts)
    assert report.level == "OUT_OF_SCOPE"
    assert report.signals["has_atomic"] is True


def test_contract_full_for_mock_dot_with_masks():
    ttir = """
    tt.func @mm(%A: tensor<*xf32>, %B: tensor<*xf32>) {
      %ld1 = tt.load %A, %mask : tensor<*xf32>
      %ld2 = tt.load %B, %mask : tensor<*xf32>
      %d = tt.dot %ld1, %ld2
      tt.store %A, %d, %mask
      tt.return
    }
    """
    facts = extract_facts(ttir)
    constraints = extract_constraints(ttir, facts)
    report = evaluate_contract(facts, constraints)
    assert report.level == "FULL"
    assert report.kernel_kind_hint == "matmul"


def test_contract_partial_when_missing_mask():
    ttir = """
    tt.func @mm(%A: tensor<*xf32>, %B: tensor<*xf32>) {
      %ld1 = tt.load %A : tensor<*xf32>
      %ld2 = tt.load %B : tensor<*xf32>
      %d = tt.dot %ld1, %ld2
      tt.store %A, %d : tensor<*xf32>
      tt.return
    }
    """
    facts = extract_facts(ttir)
    report = evaluate_contract(facts)
    assert report.level == "PARTIAL"


def test_facts_counts():
    ttir = """
    %0 = tt.load %A, %mask
    %1 = tt.load %B
    tt.store %C, %0
    tt.reduce %0
    """
    facts = extract_facts(ttir)
    assert facts.op_counts.get("load") == 2
    assert facts.op_counts.get("store") == 1
    assert facts.has_reduce is True
