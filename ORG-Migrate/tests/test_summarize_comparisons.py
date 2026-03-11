from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_summarize_comparisons_writes_jsonl_and_csv(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "comparison.json").write_text(
        json.dumps(
            {
                "kernel": "flash_attention2d",
                "backend_target": "cuda_5090d",
                "source_arch": "sm90",
                "target_arch": "sm120",
                "compiler_stack": "python",
                "compiler_cpp_wave": "",
                "guided_compiler_stack": "python",
                "source_compiler_stack": "cpp_plugin",
                "target_compiler_stack": "python",
                "shape_bindings": {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
                "evidence_source": {"primary": "ttgir"},
                "hardware_model": {"arch_cluster": "cuda_tc_mid_smem"},
                "source_candidate": "attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6",
                "source_candidate_origin": "tuning_db:sm90",
                "target_candidate": "attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6",
                "target_candidate_origin": "tuning_db:sm120",
                "comparisons": {
                    "guided_best_ratio": 0.8,
                    "guided_best_qps_intentir": 162.0,
                    "guided_best_qps_native": 200.0,
                    "guided_requested_sm": "sm_120",
                    "guided_effective_sm": "sm_86",
                    "guided_downleveled": True,
                    "source_replay_raw_ratio": None,
                    "source_replay_raw_qps_intentir": None,
                    "source_replay_raw_qps_native": None,
                    "source_replay_requested_sm": None,
                    "source_replay_effective_sm": None,
                    "source_replay_downleveled": None,
                    "source_replay_portable_ratio": 0.75,
                    "source_replay_portable_qps_intentir": 150.0,
                    "source_replay_portable_qps_native": 200.0,
                    "source_replay_portable_requested_sm": "sm_120",
                    "source_replay_portable_effective_sm": "sm_86",
                    "source_replay_portable_downleveled": True,
                    "target_oracle_raw_ratio": 0.9,
                    "target_oracle_raw_qps_intentir": 180.0,
                    "target_oracle_raw_qps_native": 200.0,
                    "target_oracle_requested_sm": "sm_120",
                    "target_oracle_effective_sm": "sm_86",
                    "target_oracle_downleveled": True,
                    "target_oracle_portable_ratio": 0.9,
                    "target_oracle_portable_qps_intentir": 180.0,
                    "target_oracle_portable_qps_native": 200.0,
                    "target_oracle_portable_requested_sm": "sm_120",
                    "target_oracle_portable_effective_sm": "sm_86",
                    "target_oracle_portable_downleveled": True,
                    "shared_native_qps": 200.0,
                    "native_qps_spread_ratio": 1.0,
                    "guided_shared_native_ratio": 0.8,
                    "source_replay_portable_shared_native_ratio": 0.75,
                    "target_oracle_portable_shared_native_ratio": 0.9,
                    "source_replay_best_ratio": 0.7,
                    "target_oracle_best_ratio": 0.9,
                    "guided_vs_source_replay_raw": None,
                    "guided_vs_source_replay_portable": 1.0666666667,
                    "guided_vs_target_oracle_raw": 0.88,
                    "guided_vs_portable_target_oracle": 0.88,
                    "guided_vs_source_replay": 1.14,
                    "guided_vs_target_oracle": 0.88,
                    "guided_first_candidate": {"kernel_kind": "attn2d_causal_softmax_v6", "bindings": {"ATTN_BLOCK_KV": 64}},
                    "source_replay_first_candidate": {"kernel_kind": "attn2d_causal_softmax_v6", "bindings": {"ATTN_BLOCK_KV": 64}},
                    "target_oracle_first_candidate": {"kernel_kind": "attn2d_causal_softmax_v6", "bindings": {"ATTN_BLOCK_KV": 64}},
                    "guided_failure": {"reason_code": "", "reason_detail": "", "ok": True},
                    "source_replay_failure": {"reason_code": "", "reason_detail": "", "ok": True},
                    "target_oracle_failure": {"reason_code": "", "reason_detail": "", "ok": True},
                    "guided_outcome": {"status": "ok", "returncode": 0},
                    "source_replay_outcome": {"status": "failed", "returncode": 1},
                    "target_oracle_outcome": {"status": "candidate_unavailable", "returncode": None},
                    "source_replay_portable_outcome": {
                        "status": "portable_repair_ok",
                        "candidate": "attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6",
                        "reason": "async_binding_removed",
                    },
                    "target_oracle_portable_outcome": {
                        "status": "raw_replayable",
                        "candidate": "attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6",
                        "reason": "raw_replayable",
                    },
                    "source_replay_analysis": {
                        "status": "requires_substitution",
                        "repair": {
                            "repair_candidate": "attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6",
                            "reason": "async_binding_removed",
                        },
                    },
                    "target_oracle_analysis": {"status": "candidate_unavailable", "repair": {}},
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "summary"
    cmd = [
        sys.executable,
        str(ROOT / "ORG-Migrate" / "tools" / "summarize_comparisons.py"),
        "--root",
        str(tmp_path / "runs"),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    jsonl_path = out_dir / "comparison_table.jsonl"
    csv_path = out_dir / "comparison_table.csv"
    assert jsonl_path.is_file()
    assert csv_path.is_file()
    row = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["kernel"] == "flash_attention2d"
    assert row["guided_best_ratio"] == 0.8
    assert row["guided_best_qps_intentir"] == 162.0
    assert row["guided_effective_sm"] == "sm_86"
    assert row["shared_native_qps"] == 200.0
    assert row["compiler_stack"] == "python"
    assert row["compiler_cpp_wave"] == ""
    assert row["source_compiler_stack"] == "cpp_plugin"
    assert row["evidence_primary"] == "ttgir"
    assert row["hardware_cluster"] == "cuda_tc_mid_smem"
    assert row["guided_outcome"] == "ok"
    assert row["source_outcome"] == "failed"
    assert row["target_outcome"] == "candidate_unavailable"
    assert row["source_candidate_origin"] == "tuning_db:sm90"
    assert row["source_analysis"] == "requires_substitution"
    assert row["source_repair_reason"] == "async_binding_removed"
    assert row["source_replay_portable_ratio"] == 0.75
    assert row["source_portable_outcome"] == "portable_repair_ok"
