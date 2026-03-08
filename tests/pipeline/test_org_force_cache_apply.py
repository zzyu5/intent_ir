from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("triton")
pytest.importorskip("torch")

from intent_ir.ir import IntentFunction
from pipeline.interfaces import KernelArtifactBundle, KernelDescriptor
from pipeline.triton.core import _run_org_plugin
from pipeline.triton.org_bridge import load_org_attr


def _dummy_intent(name: str) -> IntentFunction:
    return IntentFunction.from_json_dict(
        {
            "name": str(name),
            "tensors": {"x": {"dtype": "f32", "shape": [1], "layout": "row_major"}, "Out": {"dtype": "f32", "shape": [1], "layout": "row_major"}},
            "ops": [{"op": "identity", "inputs": ["x"], "output": "Out"}],
            "outputs": ["Out"],
        }
    )


def _dummy_desc(*, kernel: str, ttgir_path: Path | None = None, ptx_path: Path | None = None) -> KernelDescriptor:
    desc = KernelDescriptor(schema_version="kernel_desc_v1.0", name=str(kernel), frontend="triton")
    desc.source_text = "def kernel(): pass"
    desc.artifacts = KernelArtifactBundle(ttgir_path=(str(ttgir_path) if ttgir_path is not None else None))
    if ptx_path is not None:
        desc.artifacts.extra["ptx_path"] = str(ptx_path)
    return desc


def _seed_payload(*, kernel: str) -> dict[str, object]:
    if kernel == "flash_attention2d":
        return {
            "schema_version": "intentir_org_v1",
            "kernel": "flash_attention2d",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
                "artifacts": {"ttgir_path": "flash.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep q/state resident", "scope": "kv_loop", "tensors": ["Q"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "online reduce", "scope": "softmax", "tensors": ["Out"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "avoid score matrix", "scope": "softmax", "tensors": ["scores"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "pipeline loads", "scope": "kv_loop", "tensors": ["K", "V"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "kv_tile_stage", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["tile_kv"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "online_softmax_reduce", "category": "communication", "supports_goals": ["g1", "g2"], "attrs": {}, "dims": ["score_warps"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "prefetch_pipeline", "category": "pipeline", "supports_goals": ["g3"], "attrs": {}, "dims": ["pipeline_stages"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "tile_kv", "role": "kv_tile", "candidates": [32, 64], "constraints": ["tile_kv <= KV_CTX"], "evidence_refs": ["e0"]},
                {"name": "score_warps", "role": "score_reduce", "candidates": [6, 4], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "pipeline_stages", "role": "pipeline_depth", "candidates": [2], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "attn2d_causal_softmax_v6",
                "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "flash.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    if kernel == "_attn_fwd":
        return {
            "schema_version": "intentir_org_v1",
            "kernel": "_attn_fwd",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"Z": 1, "q_numhead": 1, "kv_numhead": 1, "Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
                "artifacts": {"ttgir_path": "attn_fwd.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep q/state resident", "scope": "q_state", "tensors": ["Q"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "online reduce", "scope": "softmax", "tensors": ["Out"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "avoid score matrix", "scope": "scores", "tensors": ["scores"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "pipeline loads", "scope": "kv_loop", "tensors": ["K", "V"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "qkv_stage", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_m", "block_kv"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "online_softmax_reduce", "category": "communication", "supports_goals": ["g1", "g2"], "attrs": {}, "dims": ["block_kv"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "mask_causal_apply", "category": "communication", "supports_goals": ["g2"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "prefetch_pipeline", "category": "pipeline", "supports_goals": ["g3"], "attrs": {}, "dims": ["pipeline_stages"], "evidence_refs": ["e0"]},
            ],
            "dims": [
                {"name": "block_m", "role": "query_tile", "candidates": [8, 4], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "block_kv", "role": "kv_tile", "candidates": [32, 16], "constraints": [], "evidence_refs": ["e0"]},
                {"name": "pipeline_stages", "role": "pipeline_depth", "candidates": [2], "constraints": [], "evidence_refs": ["e0"]},
            ],
            "source_oracle": {
                "kernel_kind": "attn_fwd_tiled_v3",
                "bindings": {"ATTN_FWD_BLOCK_M": 8, "ATTN_FWD_BLOCK_KV": 32},
                "arch": "sm90",
                "compiler_stack": "python",
                "evidence_refs": ["e1"],
            },
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "attn_fwd.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    if kernel == "masked_softmax2d":
        return {
            "schema_version": "intentir_org_v1",
            "kernel": "masked_softmax2d",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 64},
                "artifacts": {"ttgir_path": "masked_softmax2d.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep row resident", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "row reduction", "scope": "softmax", "tensors": ["output"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "mask inline", "scope": "mask", "tensors": ["mask"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "vector row path", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "row_reduction", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "mask_apply", "category": "communication", "supports_goals": ["g2"], "attrs": {}, "dims": [], "evidence_refs": ["e0"]},
                {"id": "m3", "tag": "vector_row_path", "category": "mapping", "supports_goals": ["g3"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [{"name": "block_threads", "role": "thread_block", "candidates": [64, 128], "constraints": [], "evidence_refs": ["e0"]}],
            "source_oracle": {"kernel_kind": "row_masked_softmax_axis1_v1", "bindings": {}, "arch": "sm90", "compiler_stack": "python", "evidence_refs": ["e1"]},
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "masked_softmax2d.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    if kernel == "softmax_inner":
        return {
            "schema_version": "intentir_org_v1",
            "kernel": "softmax_inner",
            "source_context": {
                "frontend": "triton",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "shape_bindings": {"M": 4, "N": 64},
                "artifacts": {"ttgir_path": "softmax_inner.ttgir"},
            },
            "goals": [
                {"id": "g0", "tag": "resident_working_set", "summary": "keep row resident", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
                {"id": "g1", "tag": "streaming_softmax_state", "summary": "row reduction", "scope": "softmax", "tensors": ["output"], "evidence_refs": ["e0"]},
                {"id": "g2", "tag": "avoid_materialization", "summary": "avoid extra buffer", "scope": "softmax", "tensors": ["scores"], "evidence_refs": ["e0"]},
                {"id": "g3", "tag": "latency_hiding", "summary": "vector row path", "scope": "row", "tensors": ["input"], "evidence_refs": ["e0"]},
            ],
            "mechanisms": [
                {"id": "m0", "tag": "row_tile_resident", "category": "staging", "supports_goals": ["g0"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m1", "tag": "row_reduction", "category": "communication", "supports_goals": ["g1"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
                {"id": "m2", "tag": "vector_row_path", "category": "mapping", "supports_goals": ["g3"], "attrs": {}, "dims": ["block_threads"], "evidence_refs": ["e0"]},
            ],
            "dims": [{"name": "block_threads", "role": "thread_block", "candidates": [64, 128], "constraints": [], "evidence_refs": ["e0"]}],
            "source_oracle": {"kernel_kind": "row_softmax_axis1_triton_v1", "bindings": {"SOFTMAX_BLOCK_THREADS": 64}, "arch": "sm90", "compiler_stack": "python", "evidence_refs": ["e1"]},
            "evidence": [
                {"id": "e0", "kind": "ttgir_line", "path": "softmax_inner.ttgir:1", "summary": "ttgir evidence"},
                {"id": "e1", "kind": "tuning_db", "path": "cuda.jsonl", "summary": "source oracle"},
            ],
        }
    return {
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
            {"id": "g0", "tag": "operand_reuse", "summary": "reuse ab tiles", "scope": "k_loop", "tensors": ["A", "B"], "evidence_refs": ["e0"]},
            {"id": "g1", "tag": "mma_acceleration", "summary": "use mma", "scope": "mainloop", "tensors": ["A", "B"], "evidence_refs": ["e0"]},
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


def _write_seed(*, out_dir: Path, kernel: str) -> None:
    save_org_seed = load_org_attr("org.io", "save_org_seed")
    validate_org_doc = load_org_attr("org.schema", "validate_org_doc")
    org = validate_org_doc(_seed_payload(kernel=kernel))
    save_org_seed(
        path=out_dir / f"{kernel}.org_seed.json",
        kernel=str(kernel),
        triton_provider="native",
        backend_target="cuda_5090d",
        org=org,
        raw_json=org.to_json_dict(),
        llm_trace={"provider": "test", "cached": True},
        quality={"diff_ok": True, "static_ok": True, "contract_level": "test"},
    )


def test_force_cache_apply_flash_attention2d_requires_ttgir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    _write_seed(out_dir=tmp_path, kernel="flash_attention2d")
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="flash_attention2d",
        out_dir=tmp_path,
        desc=None,
        intent=_dummy_intent("flash_attention2d"),
        report=report,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("error") == "ttgir_missing"


def test_force_cache_apply_flash_attention2d_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.delenv("INTENTIR_ORG_SOURCE_ARCH", raising=False)
    _write_seed(out_dir=tmp_path, kernel="flash_attention2d")
    ttgir = tmp_path / "flash.ttgir"
    ptx = tmp_path / "flash.ptx"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @flash_attention2d_kernel(%Q_ptr: !tt.ptr<f32>) {\n    %pid_q = tt.get_program_id x : i32\n    %k_33 = tt.load %k_32, %k_25, %cst_2 : tensor<32x64x!tt.ptr<f32>, #blocked>\n    %m_ij = "tt.reduce"(%scores_46) <{axis = 0 : i32}> ({\n    ^bb0(%lhs: f32, %rhs: f32):\n      %max = arith.maxnumf %lhs, %rhs : f32\n      tt.reduce.return %max : f32\n    }) : (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> f32\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    ptx.write_text("cp.async.cg.shared.global;\nshfl.sync.bfly;\nbar.sync 0;\n", encoding="utf-8")
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="flash_attention2d",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="flash_attention2d", ttgir_path=ttgir, ptx_path=ptx),
        intent=_dummy_intent("flash_attention2d"),
        report=report,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    evidence_source = (report["org"] or {}).get("evidence_source", {})
    assert evidence_source.get("primary") == "ttgir"
    assert (report["org"] or {}).get("compiler_stack") == "python"
    assert (report["org"] or {}).get("compiler_cpp_wave") in {"", "wave2"}
    assert evidence_source.get("ptx_available") is True
    assert str(evidence_source.get("ptx_path") or "").endswith("flash.ptx")
    assert isinstance((report["org"] or {}).get("hardware_model"), dict)
    assert ((report["org"] or {}).get("hardware_model") or {}).get("arch_cluster") == "cuda_tc_mid_smem"
    source_oracle_facts = json.loads(Path(str((report["org"] or {}).get("source_oracle_facts_path"))).read_text(encoding="utf-8"))
    assert source_oracle_facts["available"] is True
    assert source_oracle_facts["oracle"]["arch"] == "sm90"
    assert source_oracle_facts["oracle"]["kernel_kind"] == "attn2d_causal_softmax_v6"
    assert source_oracle_facts["oracle"]["bindings"]["ATTN_SCORE_WARPS"] == 6
    assert Path(str((report["org"] or {}).get("ttgir_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("ptx_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("source_oracle_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("hardware_model_path"))).is_file()
    assert (tmp_path / "flash_attention2d.org_plan.json").is_file()


def test_force_cache_apply_attn_fwd_requires_ttgir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    _write_seed(out_dir=tmp_path, kernel="_attn_fwd")
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="_attn_fwd",
        out_dir=tmp_path,
        desc=None,
        intent=_dummy_intent("_attn_fwd"),
        report=report,
        shape_bindings={"Z": 1, "q_numhead": 1, "kv_numhead": 1, "Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("error") == "ttgir_missing"


def test_force_cache_apply_attn_fwd_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.delenv("INTENTIR_ORG_SOURCE_ARCH", raising=False)
    _write_seed(out_dir=tmp_path, kernel="_attn_fwd")
    ttgir = tmp_path / "attn_fwd.ttgir"
    ptx = tmp_path / "attn_fwd.ptx"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @_attn_fwd(%Q: !tt.ptr<f32>, %K: !tt.ptr<f32>, %V: !tt.ptr<f32>, %attn_mask: !tt.ptr<f32>) {\n    %q_0 = tt.load %Qv, %mask_q, %cst : tensor<16x64x!tt.ptr<f32>, #blocked>\n    %acc = scf.for %tile = %c0_i32 to %KV_CTX step %c16_i32 iter_args(%m = %neg_inf) -> (f32) {\n      %k_0 = tt.load %Kv, %mask_k, %cst : tensor<64x16x!tt.ptr<f32>, #blocked>\n      %v_0 = tt.load %Vv, %mask_v, %cst : tensor<16x64x!tt.ptr<f32>, #blocked>\n      %pred_causal = arith.cmpi sle, %kv, %q : i1\n      %m_ij = "tt.reduce"(%scores) <{axis = 1 : i32}> ({\n      ^bb0(%lhs: f32, %rhs: f32):\n        %max = arith.maxnumf %lhs, %rhs : f32\n        tt.reduce.return %max : f32\n      }) : (tensor<16x16xf32, #blocked>) -> tensor<16xf32, #blocked>\n      scf.yield %m\n    }\n    %dot = tt.dot %a, %b : tensor<16x64xf32, #blocked> * tensor<64x16xf32, #blocked>\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    ptx.write_text("cp.async.cg.shared.global;\nshfl.sync.bfly;\nbar.sync 0;\n", encoding="utf-8")
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="_attn_fwd",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="_attn_fwd", ttgir_path=ttgir, ptx_path=ptx),
        intent=_dummy_intent("_attn_fwd"),
        report=report,
        shape_bindings={"Z": 1, "q_numhead": 1, "kv_numhead": 1, "Q_CTX": 128, "KV_CTX": 128, "HEAD_DIM": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    source_oracle_facts = json.loads(Path(str((report["org"] or {}).get("source_oracle_facts_path"))).read_text(encoding="utf-8"))
    assert source_oracle_facts["available"] is True
    assert source_oracle_facts["oracle"]["kernel_kind"] == "attn_fwd_tiled_v3"
    assert Path(str((report["org"] or {}).get("ttgir_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("ptx_facts_path"))).is_file()
    assert (tmp_path / "_attn_fwd.org_plan.json").is_file()
    assert (tmp_path / "_attn_fwd.org_candidates.txt").is_file()


def test_force_cache_apply_masked_softmax2d_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    _write_seed(out_dir=tmp_path, kernel="masked_softmax2d")
    ttgir = tmp_path / "masked_softmax2d.ttgir"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @masked_softmax2d_kernel(%inp_ptr: !tt.ptr<f32>, %mask_ptr: !tt.ptr<i1>, %out_ptr: !tt.ptr<f32>, %M: i32, %N: i32) {\n    %x_12 = tt.load %x_11, %in_bounds_8, %cst : tensor<256x!tt.ptr<f32>, #blocked>\n    %m_15 = tt.load %m_14, %in_bounds_6, %cst_1 : tensor<256x!tt.ptr<i8>, #blocked>\n    %x_17 = arith.select %m_16, %x_12, %cst_0 : tensor<256xi1, #blocked>, tensor<256xf32, #blocked>\n    %mx = "tt.reduce"(%x_17) <{axis = 0 : i32}> ({\n    ^bb0(%lhs: f32, %rhs: f32):\n      %max = arith.maxnumf %lhs, %rhs : f32\n      tt.reduce.return %max : f32\n    }) : (tensor<256xf32, #blocked>) -> f32\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="masked_softmax2d",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="masked_softmax2d", ttgir_path=ttgir),
        intent=_dummy_intent("masked_softmax2d"),
        report=report,
        shape_bindings={"M": 4, "N": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    assert (tmp_path / "masked_softmax2d.org_plan.json").is_file()
    assert (tmp_path / "masked_softmax2d.org_candidates.txt").is_file()


def test_force_cache_apply_softmax_inner_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    _write_seed(out_dir=tmp_path, kernel="softmax_inner")
    ttgir = tmp_path / "softmax_inner.ttgir"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @softmax_kernel_inner(%output_ptr: !tt.ptr<f32>, %input_ptr: !tt.ptr<f32>, %M: i32, %N: i32) {\n    %inp = tt.load %input_ptrs, %mask_3, %cst : tensor<64x!tt.ptr<f32>, #blocked>\n    %m = "tt.reduce"(%inp) <{axis = 0 : i32}> ({\n    ^bb0(%lhs: f32, %rhs: f32):\n      %max = arith.maxnumf %lhs, %rhs : f32\n      tt.reduce.return %max : f32\n    }) : (tensor<64xf32, #blocked>) -> f32\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="softmax_inner",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="softmax_inner", ttgir_path=ttgir),
        intent=_dummy_intent("softmax_inner"),
        report=report,
        shape_bindings={"M": 4, "N": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    assert (tmp_path / "softmax_inner.org_plan.json").is_file()
    assert (tmp_path / "softmax_inner.org_candidates.txt").is_file()


def test_force_cache_apply_matmul_fused_epilogue_requires_ttgir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    _write_seed(out_dir=tmp_path, kernel="matmul_fused_epilogue2d")
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="matmul_fused_epilogue2d",
        out_dir=tmp_path,
        desc=None,
        intent=_dummy_intent("matmul_fused_epilogue2d"),
        report=report,
        shape_bindings={"M": 32, "N": 32, "K": 32},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("error") == "ttgir_missing"


def test_force_cache_apply_matmul_fused_epilogue_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.delenv("INTENTIR_ORG_SOURCE_ARCH", raising=False)
    _write_seed(out_dir=tmp_path, kernel="matmul_fused_epilogue2d")
    ttgir = tmp_path / "matmul.ttgir"
    ttgir.write_text(
        '#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>\nmodule attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n  tt.func public @matmul_fused_epilogue2d_kernel(%A: !tt.ptr<f32>) {\n    %pid_m = tt.get_program_id x : i32\n    %dot = tt.dot %a, %b : tensor<16x16xf32, #blocked> * tensor<16x16xf32, #blocked>\n    %layout = ttg.convert_layout %dot : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked>\n    tt.store %out, %layout : tensor<16x16x!tt.ptr<f32>, #blocked>\n    tt.return\n  }\n}\n',
        encoding="utf-8",
    )
    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="matmul_fused_epilogue2d",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="matmul_fused_epilogue2d", ttgir_path=ttgir),
        intent=_dummy_intent("matmul_fused_epilogue2d"),
        report=report,
        shape_bindings={"M": 32, "N": 32, "K": 32},
        triton_provider="native",
        backend_target="cuda_5090d",
    )
    assert (report["org"] or {}).get("evidence_source", {}).get("primary") == "ttgir"
    assert (report["org"] or {}).get("compiler_stack") == "python"
    assert (report["org"] or {}).get("compiler_cpp_wave") in {"", "wave2"}
    assert ((report["org"] or {}).get("hardware_model") or {}).get("arch_cluster") == "cuda_tc_mid_smem"
    source_oracle_facts = json.loads(Path(str((report["org"] or {}).get("source_oracle_facts_path"))).read_text(encoding="utf-8"))
    assert source_oracle_facts["available"] is True
    assert source_oracle_facts["oracle"]["arch"] == "sm90"
    assert source_oracle_facts["oracle"]["bindings"]["MMA_ASYNC_COPY"] == 1
    assert Path(str((report["org"] or {}).get("ttgir_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("ptx_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("source_oracle_facts_path"))).is_file()
    assert Path(str((report["org"] or {}).get("hardware_model_path"))).is_file()
    assert (tmp_path / "matmul_fused_epilogue2d.org_plan.json").is_file()
    assert (tmp_path / "matmul_fused_epilogue2d.org_candidates.txt").read_text(encoding="utf-8").splitlines()[3] == "matmul_tile_v2"
