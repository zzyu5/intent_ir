from __future__ import annotations

from org.mapping.cuda.matmul_fused_epilogue2d import plan_matmul_fused_epilogue2d
from org.mapping.hardware_model import build_hardware_model
from org.schema import validate_org_doc


def _org_matmul() -> object:
    return validate_org_doc(
        {
            "schema_version": "intentir_org_v1",
            "kernel": "matmul_fused_epilogue2d",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 32, "N": 32, "K": 32},
                "artifacts": {"ttgir_path": "matmul.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "operand_reuse", "summary": "reuse A/B tiles", "scope": "k_loop", "tensors": ["A", "B"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "mma_acceleration", "summary": "use matrix primitive", "scope": "mainloop", "tensors": ["A", "B"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "fused_epilogue_avoid_writeback", "summary": "keep epilogue fused", "scope": "epilogue", "tensors": ["bias"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "prefetch tiles", "scope": "k_loop", "tensors": ["A", "B"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "ab_tile_stage", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["tile_m", "tile_n", "tile_k"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "mma_core", "category": "primitive", "supports_goals": ["g1"], "attrs": {}, "dims": ["tile_m", "tile_n", "tile_k"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "epilogue_fused_writeback", "category": "fusion", "supports_goals": ["g2"], "attrs": {}, "dims": ["tile_m", "tile_n"], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "prefetch_pipeline", "category": "pipeline", "supports_goals": ["g3"], "attrs": {}, "dims": ["pipeline_stages"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "tile_m", "role": "m_tile", "candidates": [32, 64], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "tile_n", "role": "n_tile", "candidates": [16, 32], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "tile_k", "role": "k_tile", "candidates": [16, 32], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "pipeline_stages", "role": "pipeline_depth", "candidates": [2], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "matmul_mma_tf32_v1",
                "bindings": {"MMA_BM": 32, "MMA_BN": 32, "MMA_BK": 32},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "matmul.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    )


def test_backend_plan_matmul_fused_epilogue_chain() -> None:
    plan = plan_matmul_fused_epilogue2d(
        _org_matmul(),
        shape_bindings={"M": 32, "N": 32, "K": 32},
        source_oracle={"kernel_kind": "matmul_mma_tf32_v1", "bindings": {"MMA_BM": 32, "MMA_BN": 32, "MMA_BK": 32}},
        hardware_model=build_hardware_model(target="cuda_5090d", arch="sm120"),
        budget=8,
    )
    assert plan.selected_modules
    assert any(module.id == "mma_core" for module in plan.selected_modules)
    assert plan.candidates
    assert any(candidate.kernel_kind == "matmul_mma_tf32_v1" for candidate in plan.candidates)
    assert any(str(x).startswith("preserve:") for x in plan.notes)
