"""
TileLang MVP full pipeline runner (PR#9).

This mirrors the Triton pipeline shape, but uses the TileLang adapter:
  TileLang DSL -> CertificateV2 -> obligations -> contract -> IntentIR -> diff.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from frontends.common.contract_v2 import evaluate_contract_v2
from frontends.common.obligations import evaluate_obligations
from pipeline import registry as pipeline_registry
from pipeline.interfaces import FrontendConstraints
from verify.diff_runner import run_diff
from verify.gen_cases import GeneratedCases, TestCase, generate_cases_split
from verify.metamorphic import run_bounded_exhaustive, run_metamorphic_suite
from verify.mutation import run_mutation_kill

from intent_ir.parser import CandidateIntent
from intent_ir.ir.printer_mlir_like import print_mlir_like

from kernels.tilelang.ops.any_kernel_dim import any_kernel_dim_spec
from kernels.tilelang.ops.groupnorm import group_norm_kernel_spec
from kernels.tilelang.ops.softmax_inner import softmax_inner_spec
from kernels.tilelang.ops.layernorm import layer_norm_persistent_spec
from kernels.tilelang.ops.upsample_bicubic2d_aa import upsample_bicubic2d_aa_spec
from kernels.tilelang.ops._attn_fwd import attn_fwd_spec
from kernels.tilelang.ops.gemm import gemm_spec
from kernels.tilelang.ops.spec import TileLangKernelSpec


ROOT = Path(__file__).resolve().parents[2]


def mvp_kernel_specs() -> List[TileLangKernelSpec]:
    """
    PR#9 original MVP: one anchor-strong kernel to prove the pipeline can host TileLang.
    """
    return [gemm_spec()]


def default_kernel_specs() -> List[TileLangKernelSpec]:
    """
    Regression suite: mirror Triton's 6 representative kernels (by name).
    """
    return [
        any_kernel_dim_spec(),
        group_norm_kernel_spec(),
        attn_fwd_spec(),
        softmax_inner_spec(),
        layer_norm_persistent_spec(),
        upsample_bicubic2d_aa_spec(),
    ]


def run_pipeline_for_spec(
    spec: TileLangKernelSpec,
    *,
    out_dir: Path,
    cases_limit: int = 8,
    stage_c: bool = True,
    mutation_kill: bool = True,
) -> Dict[str, object]:
    report: Dict[str, object] = {"kernel": spec.name, "frontend": "tilelang"}
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = pipeline_registry.get("tilelang")

    # 1) Descriptor / facts / constraints / certificate
    desc = adapter.build_descriptor(spec)
    desc.meta["artifact_dir"] = str(out_dir)
    (out_dir / f"{spec.name}.tilelang_tir.py").write_text(desc.source_text, encoding="utf-8")
    report["descriptor"] = desc.to_json_dict()

    desc = adapter.ensure_artifacts(desc, spec)
    facts = adapter.extract_facts(desc)
    constraints: FrontendConstraints = adapter.extract_constraints(desc, facts)
    cert_v2 = adapter.build_certificate(desc, facts, constraints)

    obligations = evaluate_obligations(desc, cert_v2)
    cert_v2.semantic_facts["obligations"] = [o.to_json_dict() for o in obligations]
    contract = evaluate_contract_v2(desc, cert_v2, obligations, constraints=constraints)

    report["certificate_v2"] = cert_v2.to_json_dict()
    (out_dir / f"{spec.name}.certificate_v2.json").write_text(json.dumps(report["certificate_v2"], indent=2), encoding="utf-8")
    report["contract"] = {
        "level": contract.level,
        "reasons": list(contract.reasons),
        "assumptions": list(contract.assumptions),
        "signals": dict(contract.signals),
    }
    (out_dir / f"{spec.name}.contract.json").write_text(json.dumps(report["contract"], indent=2), encoding="utf-8")

    # 2) Deterministic "intent" (no LLM for TileLang MVP).
    intent = spec.intent_builder()
    (out_dir / f"{spec.name}.intentir.mlir").write_text(print_mlir_like(intent), encoding="utf-8")
    cand = CandidateIntent(intent=intent, llm_trace={"provider": "tilelang"})
    report["intent"] = intent.to_json_dict()

    # 3) Stage B: cases + diff
    cases_pack: GeneratedCases = generate_cases_split(
        cand.intent,
        constraints=constraints,
        limit=int(cases_limit),
        seed=0,
        axes=list(spec.vary_axes),
        exclude_axes=list(spec.exclude_axes or []),
        assumptions=list(contract.assumptions),
        base_shapes=dict(spec.canonical_shapes),
    )
    cases_in = list(cases_pack.in_contract)
    cases_out = list(cases_pack.out_of_contract)
    report["cases"] = {"in_contract": [dict(c.shapes) for c in cases_in], "out_of_contract": [dict(c.shapes) for c in cases_out]}

    diffs_in, _ = run_diff(cand.intent, spec.runner, cases_in)
    report["diff"] = {"ok": bool(diffs_in and all(d.ok for d in diffs_in)), "results": [d.__dict__ for d in diffs_in]}

    if stage_c:
        base_case = TestCase(shapes=dict(spec.canonical_shapes), dtypes={}, seed=0)
        meta = run_metamorphic_suite(spec.name, cand.intent, spec.runner, base_case=base_case)
        bounded = run_bounded_exhaustive(spec.name, cand.intent, spec.runner, max_cases=64)
        report["stage_c"] = {
            "metamorphic": {"ok": bool(meta.ok), "results": [r.__dict__ for r in meta.results]},
            "bounded_exhaustive": bounded.__dict__,
        }
    else:
        report["stage_c"] = {"skipped": True}

    if mutation_kill:
        base_case = TestCase(shapes=dict(spec.canonical_shapes), dtypes={}, seed=0)
        diff_cases = cases_in[:2] if cases_in else [base_case]
        metamorphic_base = cases_in[0] if cases_in else base_case
        mut = run_mutation_kill(
            spec.name,
            intent=cand.intent,
            run_ref_fn=spec.runner,
            diff_cases=diff_cases,
            metamorphic_base_case=metamorphic_base,
            static_validate_fn=None,
            n_mutants=8,
            seed=0,
        )
        report["mutation_kill"] = mut.__dict__
    else:
        report["mutation_kill"] = {"skipped": True}

    return report


__all__ = ["TileLangKernelSpec", "mvp_kernel_specs", "default_kernel_specs", "run_pipeline_for_spec"]
