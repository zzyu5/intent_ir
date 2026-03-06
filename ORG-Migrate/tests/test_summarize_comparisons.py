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
                "source_candidate": "attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6",
                "target_candidate": "attn2d_causal_softmax_v6:ATTN_BLOCK_KV=64,ATTN_SCORE_WARPS=6",
                "comparisons": {
                    "guided_best_ratio": 0.8,
                    "source_replay_best_ratio": 0.7,
                    "target_oracle_best_ratio": 0.9,
                    "guided_vs_source_replay": 1.14,
                    "guided_vs_target_oracle": 0.88,
                    "guided_first_candidate": {"kernel_kind": "attn2d_causal_softmax_v6", "bindings": {"ATTN_BLOCK_KV": 64}},
                    "source_replay_failure": {"reason_code": "", "reason_detail": "", "ok": True},
                    "target_oracle_failure": {"reason_code": "", "reason_detail": "", "ok": True},
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
