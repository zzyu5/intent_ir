from __future__ import annotations

from pathlib import Path

from pipeline.common.tuning_db import load_tuning_db_jsonl, resolve_tuning_entries


def test_tuning_db_last_match_wins_for_same_compiler_stack(tmp_path: Path) -> None:
    db = tmp_path / "cuda.jsonl"
    db.write_text(
        "\n".join(
                [
                    '{"backend":"cuda","compiler_stack":"cpp_plugin","kernel":"flash_attention2d","arch":"sm120","when":{"HEAD_DIM":64},"bindings":{"ATTN_BLOCK_KV":64,"ATTN_SCORE_WARPS":6,"FLASH_ATTN_ASYNC_COPY":1},"kernel_kind":"attn2d_causal_softmax_v7"}',
                    '{"backend":"cuda","compiler_stack":"cpp_plugin","kernel":"flash_attention2d","arch":"sm120","when":{"HEAD_DIM":64},"bindings":{"ATTN_BLOCK_KV":64,"ATTN_SCORE_WARPS":6,"FLASH_ATTN_ASYNC_COPY":0},"kernel_kind":"attn2d_causal_softmax_v6"}',
                ]
        )
        + "\n",
        encoding="utf-8",
    )
    mapping = load_tuning_db_jsonl(path=db, backend="cuda")
    entries = mapping[("flash_attention2d", "sm120")]
    bindings, kernel_kind = resolve_tuning_entries(
        entries,
        shape_bindings={"HEAD_DIM": 64, "Q_CTX": 64, "KV_CTX": 64},
        compiler_stack="cpp_plugin",
    )
    assert kernel_kind == "attn2d_causal_softmax_v6"
    assert bindings == {"ATTN_BLOCK_KV": 64, "ATTN_SCORE_WARPS": 6, "FLASH_ATTN_ASYNC_COPY": 0}
