from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_tool_module():
    tool_path = ROOT / "ORG-Migrate" / "tools" / "compare_source_oracle_vs_guided.py"
    spec = importlib.util.spec_from_file_location("org_compare_tool", tool_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_graph(out_root: Path, *, ok: bool | None, reason_code: str, reason_detail: str, skip_reason: str) -> None:
    perf_dir = out_root / "perf"
    perf_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": [
            {
                "ok": ok,
                "reason_code": reason_code,
                "reason_detail": reason_detail,
                "skip_reason": skip_reason,
                "count_in_denominator": False,
            }
        ]
    }
    (perf_dir / "gpu_perf_graph.json").write_text(json.dumps(payload), encoding="utf-8")


def test_compare_tool_writes_outcomes_and_metadata(tmp_path, monkeypatch) -> None:
    module = _load_tool_module()
    report_path = tmp_path / "flash_attention2d.json"
    plan_path = tmp_path / "flash_attention2d.org_plan.json"
    candidates_path = tmp_path / "flash_attention2d.org_candidates.txt"
    out_root = tmp_path / "compare"

    report = {
        "org": {
            "plan_path": str(plan_path),
            "candidates_txt_path": str(candidates_path),
            "arch": "sm120",
            "shape_bindings": {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
            "compiler_stack": "python",
            "compiler_cpp_wave": "",
            "evidence_source": {"primary": "ttgir", "ptx_available": True},
            "hardware_model": {"arch_cluster": "cuda_tc_mid_smem"},
        },
        "org_doc": {
            "source_context": {
                "shape_bindings": {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
            }
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    plan_path.write_text(json.dumps({"source_oracle": {"kernel_kind": "attn2d_causal_softmax_v6", "bindings": {"ATTN_BLOCK_KV": 64}}}), encoding="utf-8")
    candidates_path.write_text("attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64\n", encoding="utf-8")

    guided_root = out_root / "guided"
    source_root = out_root / "source_replay"
    _write_graph(source_root, ok=False, reason_code="intentir_unavailable", reason_detail="coverage failed", skip_reason="intentir_unavailable")

    def fake_run_tune(**kwargs):
        out_dir = Path(kwargs["out_root"])
        assert kwargs["compiler_stack"] == "python"
        assert kwargs["compiler_cpp_wave"] == ""
        if out_dir.name == "guided":
            return {
                "returncode": 0,
                "out_root": str(guided_root),
                "summary": {
                    "candidates": [
                        {
                            "kernel_kind": "attn2d_causal_softmax_v6",
                            "bindings": {"ATTN_BLOCK_KV": 64},
                            "ratio": 0.81,
                            "qps_native": 200.0,
                            "qps_intentir": 162.0,
                            "coverage_rc": 0,
                            "perf_rc": 0,
                        },
                        {
                            "kernel_kind": "attn2d_causal_softmax_v6",
                            "bindings": {"ATTN_BLOCK_KV": 64, "FLASH_ATTN_ASYNC_COPY": 1},
                            "ratio": None,
                            "coverage_rc": 0,
                            "perf_rc": 0,
                        }
                    ]
                },
            }
        if out_dir.name == "source_replay":
            return {
                "returncode": 1,
                "out_root": str(source_root),
                "summary": {
                    "candidates": [
                        {
                            "kernel_kind": "attn2d_causal_softmax_v6",
                            "bindings": {"ATTN_BLOCK_KV": 64},
                            "ratio": None,
                            "coverage_rc": 1,
                            "perf_rc": 1,
                        }
                    ]
                },
            }
        raise AssertionError(f"unexpected out_root {out_dir}")

    monkeypatch.setattr(module, "_run_tune", fake_run_tune)
    monkeypatch.setattr(module, "_resolve_source_candidate", lambda **_: "attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64")
    monkeypatch.setattr(module, "_resolve_target_oracle_candidate", lambda **_: "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_source_oracle_vs_guided.py",
            "--report",
            str(report_path),
            "--backend-target",
            "cuda_5090d",
            "--out-root",
            str(out_root),
        ],
    )

    assert module.main() == 0

    payload = json.loads((out_root / "comparison.json").read_text(encoding="utf-8"))
    assert payload["shape_bindings"] == {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64}
    assert payload["compiler_stack"] == "python"
    assert payload["compiler_cpp_wave"] == ""
    assert payload["guided_compiler_stack"] == "python"
    assert payload["source_compiler_stack"] == "python"
    assert payload["evidence_source"]["primary"] == "ttgir"
    assert payload["hardware_model"]["arch_cluster"] == "cuda_tc_mid_smem"
    assert payload["comparisons"]["guided_outcome"]["status"] == "ok"
    assert payload["comparisons"]["source_replay_outcome"]["status"] == "failed"
    assert payload["comparisons"]["source_replay_outcome"]["failure"]["reason_code"] == "intentir_unavailable"
    assert payload["comparisons"]["target_oracle_outcome"]["status"] == "candidate_unavailable"
    assert payload["comparisons"]["target_oracle_outcome"]["failure"]["reason_code"] == "candidate_unavailable"
    assert payload["source_candidate_origin"] == "plan.source_oracle"
    assert payload["comparisons"]["source_replay_analysis"]["status"] == "failed"
    assert payload["comparisons"]["source_replay_raw_ratio"] is None
    assert payload["comparisons"]["source_replay_portable_ratio"] is None
    assert payload["comparisons"]["guided_best_qps_intentir"] == 162.0
    assert payload["comparisons"]["guided_best_qps_native"] == 200.0
    txt = (out_root / "comparison.txt").read_text(encoding="utf-8")
    assert "hardware_cluster: cuda_tc_mid_smem" in txt
    assert "guided_best_qps_intentir: 162.0" in txt
    assert "guided_outcome: ok" in txt
    assert "source_replay_outcome: failed" in txt
    assert "target_oracle_outcome: candidate_unavailable" in txt


def test_compare_tool_detects_async_repair_from_guided_candidates(tmp_path, monkeypatch) -> None:
    module = _load_tool_module()
    report_path = tmp_path / "matmul_fused_epilogue2d.json"
    plan_path = tmp_path / "matmul_fused_epilogue2d.org_plan.json"
    candidates_path = tmp_path / "matmul_fused_epilogue2d.org_candidates.txt"
    out_root = tmp_path / "compare"

    report = {
        "org": {
            "plan_path": str(plan_path),
            "candidates_txt_path": str(candidates_path),
            "arch": "sm120",
            "shape_bindings": {"M": 32, "N": 32, "K": 32},
            "compiler_stack": "python",
            "compiler_cpp_wave": "",
            "evidence_source": {"primary": "ttgir"},
            "hardware_model": {"arch_cluster": "cuda_tc_mid_smem"},
        }
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    plan_path.write_text(json.dumps({"source_oracle": {"kernel_kind": "", "bindings": {}}}), encoding="utf-8")
    candidates_path.write_text("matmul_mma_tf32_v1:MMA_BK=32,MMA_BM=32,MMA_BN=32\n", encoding="utf-8")

    source_root = out_root / "source_replay"
    target_root = out_root / "target_oracle"
    _write_graph(source_root, ok=False, reason_code="lowering_missing_op", reason_detail="async path unsupported", skip_reason="intentir_unavailable")
    _write_graph(target_root, ok=False, reason_code="lowering_missing_op", reason_detail="async path unsupported", skip_reason="intentir_unavailable")

    def fake_run_tune(**kwargs):
        out_dir = Path(kwargs["out_root"])
        if out_dir.name == "guided":
            assert kwargs["compiler_stack"] == "python"
            assert kwargs["compiler_cpp_wave"] == ""
        if out_dir.name == "guided":
            return {
                "returncode": 0,
                "out_root": str(out_dir),
                "summary": {
                    "candidates": [
                        {
                            "kernel_kind": "matmul_mma_tf32_v1",
                            "bindings": {"MMA_ASYNC_COPY": 1, "MMA_BK": 32, "MMA_BM": 32, "MMA_BN": 32},
                            "ratio": None,
                            "coverage_rc": 0,
                            "perf_rc": 0,
                        },
                        {
                            "kernel_kind": "matmul_mma_tf32_v1",
                            "bindings": {"MMA_BK": 32, "MMA_BM": 32, "MMA_BN": 32},
                            "ratio": 1.01,
                            "coverage_rc": 0,
                            "perf_rc": 0,
                        },
                    ]
                },
            }
        if out_dir.name == "source_replay":
            assert kwargs["compiler_stack"] == "python"
            assert kwargs["compiler_cpp_wave"] == ""
            return {
                "returncode": 0,
                "out_root": str(source_root),
                "summary": {"candidates": [{"kernel_kind": "matmul_mma_tf32_v1", "bindings": {"MMA_ASYNC_COPY": 1, "MMA_BK": 32, "MMA_BM": 32, "MMA_BN": 32}, "ratio": None}]},
            }
        if out_dir.name == "target_oracle":
            assert kwargs["compiler_stack"] == "python"
            assert kwargs["compiler_cpp_wave"] == ""
            return {
                "returncode": 0,
                "out_root": str(target_root),
                "summary": {"candidates": [{"kernel_kind": "matmul_mma_tf32_v1", "bindings": {"MMA_ASYNC_COPY": 1, "MMA_BK": 32, "MMA_BM": 32, "MMA_BN": 32}, "ratio": None}]},
            }
        raise AssertionError(f"unexpected out_root {out_dir}")

    monkeypatch.setattr(module, "_run_tune", fake_run_tune)
    monkeypatch.setattr(module, "_resolve_source_candidate", lambda **_: "matmul_mma_tf32_v1:MMA_ASYNC_COPY=1,MMA_BK=32,MMA_BM=32,MMA_BN=32")
    monkeypatch.setattr(module, "_resolve_target_oracle_candidate", lambda **_: "matmul_mma_tf32_v1:MMA_ASYNC_COPY=1,MMA_BK=32,MMA_BM=32,MMA_BN=32")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_source_oracle_vs_guided.py",
            "--report",
            str(report_path),
            "--backend-target",
            "cuda_5090d",
            "--out-root",
            str(out_root),
        ],
    )

    assert module.main() == 0
    payload = json.loads((out_root / "comparison.json").read_text(encoding="utf-8"))
    source_analysis = payload["comparisons"]["source_replay_analysis"]
    target_analysis = payload["comparisons"]["target_oracle_analysis"]
    assert source_analysis["status"] == "requires_substitution"
    assert source_analysis["repair"]["repair_candidate"] == "matmul_mma_tf32_v1:MMA_BK=32,MMA_BM=32,MMA_BN=32"
    assert source_analysis["repair"]["reason"] == "async_binding_removed"
    assert target_analysis["status"] == "requires_substitution"
    assert payload["target_candidate_origin"] == "tuning_db:sm120"
    assert payload["comparisons"]["source_replay_raw_ratio"] is None
    assert payload["comparisons"]["source_replay_portable_ratio"] == 1.01
    assert payload["comparisons"]["target_oracle_portable_ratio"] == 1.01


def test_compare_tool_prefers_matmul_cluster_variant_shift_repair(tmp_path, monkeypatch) -> None:
    module = _load_tool_module()
    report_path = tmp_path / "matmul_fused_epilogue2d.json"
    plan_path = tmp_path / "matmul_fused_epilogue2d.org_plan.json"
    candidates_path = tmp_path / "matmul_fused_epilogue2d.org_candidates.txt"
    out_root = tmp_path / "compare"

    report = {
        "org": {
            "plan_path": str(plan_path),
            "candidates_txt_path": str(candidates_path),
            "arch": "sm120",
            "shape_bindings": {"M": 32, "N": 32, "K": 32},
            "compiler_stack": "python",
            "compiler_cpp_wave": "",
            "evidence_source": {"primary": "ttgir"},
            "hardware_model": {"arch_cluster": "cuda_tc_mid_smem"},
        }
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    plan_path.write_text(json.dumps({"source_oracle": {"kernel_kind": "", "bindings": {}}}), encoding="utf-8")
    candidates_path.write_text("matmul_tile_v2\n", encoding="utf-8")

    source_root = out_root / "source_replay"
    target_root = out_root / "target_oracle"
    _write_graph(source_root, ok=False, reason_code="lowering_missing_op", reason_detail="async path unsupported", skip_reason="intentir_unavailable")
    _write_graph(target_root, ok=False, reason_code="lowering_missing_op", reason_detail="async path unsupported", skip_reason="intentir_unavailable")

    def fake_run_tune(**kwargs):
        out_dir = Path(kwargs["out_root"])
        if out_dir.name == "guided":
            return {
                "returncode": 0,
                "out_root": str(out_dir),
                "summary": {
                    "candidates": [
                        {
                            "kernel_kind": "matmul_mma_tf32_v1",
                            "bindings": {"MMA_BK": 32, "MMA_BM": 32, "MMA_BN": 32},
                            "ratio": 0.91,
                            "coverage_rc": 0,
                            "perf_rc": 0,
                        },
                        {
                            "kernel_kind": "matmul_tile_v2",
                            "bindings": {},
                            "ratio": 1.09,
                            "coverage_rc": 0,
                            "perf_rc": 0,
                        },
                    ]
                },
            }
        if out_dir.name == "source_replay":
            return {
                "returncode": 0,
                "out_root": str(source_root),
                "summary": {"candidates": [{"kernel_kind": "matmul_mma_tf32_v1", "bindings": {"MMA_ASYNC_COPY": 1, "MMA_BK": 32, "MMA_BM": 32, "MMA_BN": 32}, "ratio": None}]},
            }
        if out_dir.name == "target_oracle":
            return {
                "returncode": 0,
                "out_root": str(target_root),
                "summary": {"candidates": [{"kernel_kind": "matmul_mma_tf32_v1", "bindings": {"MMA_ASYNC_COPY": 1, "MMA_BK": 32, "MMA_BM": 32, "MMA_BN": 32}, "ratio": None}]},
            }
        raise AssertionError(f"unexpected out_root {out_dir}")

    monkeypatch.setattr(module, "_run_tune", fake_run_tune)
    monkeypatch.setattr(module, "_resolve_source_candidate", lambda **_: "matmul_mma_tf32_v1:MMA_ASYNC_COPY=1,MMA_BK=32,MMA_BM=32,MMA_BN=32")
    monkeypatch.setattr(module, "_resolve_target_oracle_candidate", lambda **_: "matmul_mma_tf32_v1:MMA_ASYNC_COPY=1,MMA_BK=32,MMA_BM=32,MMA_BN=32")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_source_oracle_vs_guided.py",
            "--report",
            str(report_path),
            "--backend-target",
            "cuda_5090d",
            "--out-root",
            str(out_root),
        ],
    )

    assert module.main() == 0
    payload = json.loads((out_root / "comparison.json").read_text(encoding="utf-8"))
    assert payload["comparisons"]["source_replay_analysis"]["repair"]["reason"] == "cluster_variant_shift"
    assert payload["comparisons"]["source_replay_analysis"]["repair"]["repair_candidate"] == "matmul_tile_v2"
    assert payload["comparisons"]["source_replay_portable_ratio"] == 1.09
    assert payload["comparisons"]["target_oracle_portable_ratio"] == 1.09
    assert payload["comparisons"]["guided_vs_portable_target_oracle"] == 1.0
    assert payload["comparisons"]["source_replay_portable_outcome"]["status"] == "portable_repair_ok"
    txt = (out_root / "comparison.txt").read_text(encoding="utf-8")
    assert "source_replay_repair:" in txt
    assert "target_oracle_repair:" in txt
    assert "source_replay_portable:" in txt
    assert "target_oracle_portable:" in txt


def test_compare_tool_detects_flash_cluster_variant_repair(tmp_path, monkeypatch) -> None:
    module = _load_tool_module()
    report_path = tmp_path / "flash_attention2d.json"
    plan_path = tmp_path / "flash_attention2d.org_plan.json"
    candidates_path = tmp_path / "flash_attention2d.org_candidates.txt"
    out_root = tmp_path / "compare"

    report = {
        "org": {
            "plan_path": str(plan_path),
            "candidates_txt_path": str(candidates_path),
            "arch": "sm120",
            "shape_bindings": {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
            "compiler_stack": "python",
            "compiler_cpp_wave": "",
            "evidence_source": {"primary": "ttgir"},
            "hardware_model": {"arch_cluster": "cuda_tc_mid_smem"},
        }
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    plan_path.write_text(json.dumps({"source_oracle": {"kernel_kind": "attn2d_causal_softmax_v7", "bindings": {"ATTN_BLOCK_KV": 64, "FLASH_ATTN_ASYNC_COPY": 1}}}), encoding="utf-8")
    candidates_path.write_text("attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6\n", encoding="utf-8")

    def fake_run_tune(**kwargs):
        out_dir = Path(kwargs["out_root"])
        if out_dir.name == "guided":
            assert kwargs["compiler_stack"] == "python"
            assert kwargs["compiler_cpp_wave"] == ""
        if out_dir.name == "guided":
            return {
                "returncode": 0,
                "out_root": str(out_dir),
                "summary": {
                    "candidates": [
                        {"kernel_kind": "attn2d_causal_softmax_v6", "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6}, "ratio": 0.668, "coverage_rc": 0, "perf_rc": 0},
                        {"kernel_kind": "attn2d_causal_softmax_v7", "bindings": {"ATTN_BLOCK_KV": 64, "FLASH_ATTN_ASYNC_COPY": 1}, "ratio": 0.223, "coverage_rc": 0, "perf_rc": 0},
                    ]
                },
            }
        if out_dir.name in {"source_replay", "target_oracle"}:
            assert kwargs["compiler_stack"] == "python"
            assert kwargs["compiler_cpp_wave"] == ""
            return {
                "returncode": 0,
                "out_root": str(out_dir),
                "summary": {
                    "candidates": [
                        {"kernel_kind": "attn2d_causal_softmax_v7", "bindings": {"ATTN_BLOCK_KV": 64, "FLASH_ATTN_ASYNC_COPY": 1}, "ratio": 0.223, "coverage_rc": 0, "perf_rc": 0},
                    ]
                },
            }
        raise AssertionError(f"unexpected out_root {out_dir}")

    monkeypatch.setattr(module, "_run_tune", fake_run_tune)
    monkeypatch.setattr(module, "_resolve_source_candidate", lambda **_: "attn2d_causal_softmax_v7:ATTN_BLOCK_KV=64,FLASH_ATTN_ASYNC_COPY=1")
    monkeypatch.setattr(module, "_resolve_target_oracle_candidate", lambda **_: "attn2d_causal_softmax_v7:ATTN_BLOCK_KV=64,FLASH_ATTN_ASYNC_COPY=1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_source_oracle_vs_guided.py",
            "--report",
            str(report_path),
            "--backend-target",
            "cuda_5090d",
            "--out-root",
            str(out_root),
        ],
    )

    assert module.main() == 0
    payload = json.loads((out_root / "comparison.json").read_text(encoding="utf-8"))
    assert payload["comparisons"]["source_replay_analysis"]["status"] == "requires_substitution"
    assert payload["comparisons"]["source_replay_analysis"]["repair"]["reason"] == "cluster_variant_shift"
    assert payload["comparisons"]["source_replay_portable_ratio"] == 0.668
    assert payload["comparisons"]["guided_vs_source_replay_portable"] == 1.0


def test_make_outcome_reports_process_error_without_graph(tmp_path) -> None:
    module = _load_tool_module()
    result = {
        "returncode": 2,
        "out_root": str(tmp_path / "missing"),
    }
    outcome = module._make_outcome(result)
    assert outcome["status"] == "process_error"
    assert outcome["failure"]["reason_code"] == "tune_returncode_nonzero"


def test_compare_tool_uses_source_compiler_stack_from_plan(tmp_path, monkeypatch) -> None:
    module = _load_tool_module()
    report_path = tmp_path / "flash_attention2d.json"
    plan_path = tmp_path / "flash_attention2d.org_plan.json"
    candidates_path = tmp_path / "flash_attention2d.org_candidates.txt"
    out_root = tmp_path / "compare"
    report = {
        "org": {
            "plan_path": str(plan_path),
            "candidates_txt_path": str(candidates_path),
            "arch": "sm120",
            "shape_bindings": {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
            "compiler_stack": "python",
            "compiler_cpp_wave": "",
            "evidence_source": {"primary": "ttgir"},
            "hardware_model": {"arch_cluster": "cuda_tc_mid_smem"},
        }
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    plan_path.write_text(
        json.dumps({"source_oracle": {"kernel_kind": "attn2d_causal_softmax_v7", "bindings": {"ATTN_BLOCK_KV": 64, "FLASH_ATTN_ASYNC_COPY": 1}, "compiler_stack": "cpp_plugin"}}),
        encoding="utf-8",
    )
    candidates_path.write_text("attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6\n", encoding="utf-8")

    seen: list[tuple[str, str, str]] = []

    def fake_run_tune(**kwargs):
        out_dir = Path(kwargs["out_root"])
        seen.append((out_dir.name, str(kwargs["compiler_stack"]), str(kwargs["compiler_cpp_wave"])))
        return {
            "returncode": 0,
            "out_root": str(out_dir),
            "summary": {"candidates": [{"kernel_kind": "attn2d_causal_softmax_v6", "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6}, "ratio": 0.67}]},
        }

    monkeypatch.setattr(module, "_run_tune", fake_run_tune)
    monkeypatch.setattr(module, "_resolve_source_candidate", lambda **_: "attn2d_causal_softmax_v7:ATTN_BLOCK_KV=64,FLASH_ATTN_ASYNC_COPY=1")
    monkeypatch.setattr(module, "_resolve_target_oracle_candidate", lambda **_: "attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_source_oracle_vs_guided.py",
            "--report",
            str(report_path),
            "--backend-target",
            "cuda_5090d",
            "--out-root",
            str(out_root),
        ],
    )
    assert module.main() == 0
    assert ("guided", "python", "") in seen
    assert ("source_replay", "cpp_plugin", "") in seen
    assert ("target_oracle", "python", "") in seen


def test_compare_tool_infers_guided_compiler_stack_from_candidate_header(tmp_path, monkeypatch) -> None:
    module = _load_tool_module()
    report_path = tmp_path / "flash_attention2d.json"
    plan_path = tmp_path / "flash_attention2d.org_plan.json"
    candidates_path = tmp_path / "flash_attention2d.org_candidates.txt"
    out_root = tmp_path / "compare"
    report = {
        "org": {
            "plan_path": str(plan_path),
            "candidates_txt_path": str(candidates_path),
            "arch": "sm120",
            "shape_bindings": {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
            "compiler_cpp_wave": "",
            "evidence_source": {"primary": "ttgir"},
            "hardware_model": {"arch_cluster": "cuda_tc_mid_smem"},
        }
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    plan_path.write_text(json.dumps({"source_oracle": {"kernel_kind": "", "bindings": {}, "compiler_stack": "cpp_plugin"}}), encoding="utf-8")
    candidates_path.write_text("# kernel=flash_attention2d target=cuda_5090d budget=8 compiler_stack=cpp_plugin arch=sm120\nattn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6\n", encoding="utf-8")

    seen: list[tuple[str, str, str]] = []

    def fake_run_tune(**kwargs):
        out_dir = Path(kwargs["out_root"])
        seen.append((out_dir.name, str(kwargs["compiler_stack"]), str(kwargs["compiler_cpp_wave"])))
        return {
            "returncode": 0,
            "out_root": str(out_dir),
            "summary": {"candidates": [{"kernel_kind": "attn2d_causal_softmax_v6", "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6}, "ratio": 0.67}]},
        }

    monkeypatch.setattr(module, "_run_tune", fake_run_tune)
    monkeypatch.setattr(module, "_resolve_source_candidate", lambda **_: "")
    monkeypatch.setattr(module, "_resolve_target_oracle_candidate", lambda **_: "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_source_oracle_vs_guided.py",
            "--report",
            str(report_path),
            "--backend-target",
            "cuda_5090d",
            "--out-root",
            str(out_root),
        ],
    )
    assert module.main() == 0
    assert ("guided", "cpp_plugin", "") in seen


def test_compare_tool_propagates_cpp_wave(tmp_path, monkeypatch) -> None:
    module = _load_tool_module()
    report_path = tmp_path / "flash_attention2d.json"
    plan_path = tmp_path / "flash_attention2d.org_plan.json"
    candidates_path = tmp_path / "flash_attention2d.org_candidates.txt"
    out_root = tmp_path / "compare"
    report = {
        "org": {
            "plan_path": str(plan_path),
            "candidates_txt_path": str(candidates_path),
            "arch": "sm120",
            "shape_bindings": {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
            "compiler_stack": "cpp_plugin",
            "compiler_cpp_wave": "wave3",
            "evidence_source": {"primary": "ttgir"},
            "hardware_model": {"arch_cluster": "cuda_tc_mid_smem"},
        }
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    plan_path.write_text(json.dumps({"source_oracle": {"kernel_kind": "", "bindings": {}, "compiler_stack": "cpp_plugin"}}), encoding="utf-8")
    candidates_path.write_text("# kernel=flash_attention2d target=cuda_5090d budget=8 compiler_stack=cpp_plugin arch=sm120\nattn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6\n", encoding="utf-8")

    seen: list[tuple[str, str, str]] = []

    def fake_run_tune(**kwargs):
        out_dir = Path(kwargs["out_root"])
        seen.append((out_dir.name, str(kwargs["compiler_stack"]), str(kwargs["compiler_cpp_wave"])))
        return {
            "returncode": 0,
            "out_root": str(out_dir),
            "summary": {"candidates": [{"kernel_kind": "attn2d_causal_softmax_v6", "bindings": {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6}, "ratio": 0.67}]},
        }

    monkeypatch.setattr(module, "_run_tune", fake_run_tune)
    monkeypatch.setattr(module, "_resolve_source_candidate", lambda **_: "")
    monkeypatch.setattr(module, "_resolve_target_oracle_candidate", lambda **_: "")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_source_oracle_vs_guided.py",
            "--report",
            str(report_path),
            "--backend-target",
            "cuda_5090d",
            "--out-root",
            str(out_root),
        ],
    )
    assert module.main() == 0
    assert ("guided", "cpp_plugin", "wave3") in seen
