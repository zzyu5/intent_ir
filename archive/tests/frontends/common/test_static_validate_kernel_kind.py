from __future__ import annotations

from frontends.common.certificate_v2 import SemanticCertificateV2
from frontends.common.static_validate import static_validate
from intent_ir.ir import Dim, IntentFunction, Op, ScheduleSketch, TensorLayout, TensorType


def _matmul_intent() -> IntentFunction:
    rm = TensorLayout(kind="row_major", params={})
    tensors = {
        "A": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "K")], layout=rm),
        "B": TensorType(dtype="f32", shape=[Dim("sym", "K"), Dim("sym", "N")], layout=rm),
        "O": TensorType(dtype="f32", shape=[Dim("sym", "M"), Dim("sym", "N")], layout=rm),
    }
    ops = [Op(op="matmul", inputs=["A", "B"], output="O", attrs={})]
    return IntentFunction(
        name="naive_gemm",
        tensors=tensors,
        ops=ops,
        outputs=["O"],
        schedule=ScheduleSketch(),
        axis_roles={},
    )


def test_static_validate_dot_reduce_defaults_to_matmul() -> None:
    intent = _matmul_intent()
    cert = SemanticCertificateV2(
        schema_version="cert_v2.0",
        semantic_facts={"anchors": {"has_dot": True, "has_reduce": True}},
        schedule_hints={},
        meta={"contract": {"level": "PARTIAL", "reasons": []}},
    ).canonicalize()

    sv = static_validate(intent, cert)
    assert sv.ok
    assert "softmax op missing" not in sv.reasons
    assert "reduce op missing" not in sv.reasons


def test_static_validate_attention_hint_still_requires_softmax() -> None:
    intent = _matmul_intent()
    cert = SemanticCertificateV2(
        schema_version="cert_v2.0",
        semantic_facts={"anchors": {"has_dot": True, "has_reduce": True, "kernel_kind_hint": "attention"}},
        schedule_hints={},
        meta={"contract": {"level": "PARTIAL", "reasons": []}},
    ).canonicalize()

    sv = static_validate(intent, cert)
    assert not sv.ok
    assert "softmax op missing" in sv.reasons

