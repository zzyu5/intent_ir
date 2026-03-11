from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    script_path = ROOT / "scripts" / "intentir.py"
    spec = importlib.util.spec_from_file_location("intentir_cli", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_read_candidate_file_skips_comments_and_inline_comments(tmp_path: Path) -> None:
    mod = _load_module()
    f = tmp_path / "candidates.txt"
    f.write_text(
        "\n".join(
            [
                "# header line",
                "",
                "  attn2d_causal_softmax_v6:ATTN_BLOCK_KV=32,ATTN_SCORE_WARPS=6  ",
                "attn2d_causal_softmax_v7:ATTN_BLOCK_KV=64  # prefer v7",
                "   # full-line comment with leading spaces",
                "attn2d_causal_softmax_v7",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert mod._read_candidate_file(str(f)) == [
        (3, "attn2d_causal_softmax_v6:ATTN_BLOCK_KV=32,ATTN_SCORE_WARPS=6"),
        (4, "attn2d_causal_softmax_v7:ATTN_BLOCK_KV=64"),
        (6, "attn2d_causal_softmax_v7"),
    ]


def test_parse_candidate_list_reports_file_and_line(tmp_path: Path) -> None:
    mod = _load_module()
    f = tmp_path / "bad.txt"
    f.write_text(
        "\n".join(
            [
                "attn2d_causal_softmax_v6:ATTN_BLOCK_KV=32",
                "bad:ATTN_BLOCK_KV=not_int",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as e:
        _ = mod._parse_candidate_list([], [str(f)])
    msg = str(e.value)
    assert str(f) in msg
    assert ":2" in msg
    assert "non-int value" in msg


def test_tune_parser_accepts_candidate_file(tmp_path: Path) -> None:
    mod = _load_module()
    f = tmp_path / "cands.txt"
    f.write_text("attn2d_causal_softmax_v6:ATTN_BLOCK_KV=32,ATTN_SCORE_WARPS=6\n", encoding="utf-8")
    ap = mod._build_parser()
    args = ap.parse_args(
        [
            "tune",
            "--backend-target",
            "cuda_h100",
            "--kernel",
            "flash_attention2d",
            "--candidate-file",
            str(f),
        ]
    )
    assert getattr(args, "candidate_file") == [str(f)]

