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
            "evidence_source": {"primary": "ttgir", "ptx_available": True},
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
    assert payload["evidence_source"]["primary"] == "ttgir"
    assert payload["comparisons"]["guided_outcome"]["status"] == "ok"
    assert payload["comparisons"]["source_replay_outcome"]["status"] == "failed"
    assert payload["comparisons"]["source_replay_outcome"]["failure"]["reason_code"] == "intentir_unavailable"
    assert payload["comparisons"]["target_oracle_outcome"]["status"] == "candidate_unavailable"
    assert payload["comparisons"]["target_oracle_outcome"]["failure"]["reason_code"] == "candidate_unavailable"
    txt = (out_root / "comparison.txt").read_text(encoding="utf-8")
    assert "guided_outcome: ok" in txt
    assert "source_replay_outcome: failed" in txt
    assert "target_oracle_outcome: candidate_unavailable" in txt


def test_make_outcome_reports_process_error_without_graph(tmp_path) -> None:
    module = _load_tool_module()
    result = {
        "returncode": 2,
        "out_root": str(tmp_path / "missing"),
    }
    outcome = module._make_outcome(result)
    assert outcome["status"] == "process_error"
    assert outcome["failure"]["reason_code"] == "tune_returncode_nonzero"
