from __future__ import annotations

from org.facts.ttgir import extract_ttgir_mechanism_facts


TTGIR_FIXTURE = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_fused_epilogue2d_kernel(%A: !tt.ptr<f32>, %B: !tt.ptr<f32>, %C: !tt.ptr<f32>) {
    %pid_m = tt.get_program_id x : i32
    %dot = tt.dot %a, %b : tensor<16x16xf32, #blocked> * tensor<16x16xf32, #blocked>
    %layout = ttg.convert_layout %dot : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked>
    tt.store %out, %layout : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
""".strip()


def test_ttgir_facts_matmul_fused_epilogue2d() -> None:
    facts = extract_ttgir_mechanism_facts(TTGIR_FIXTURE, kernel_name="matmul_fused_epilogue2d", artifact_path="matmul.ttgir")
    mechanisms = dict(facts.get("mechanisms") or {})
    assert mechanisms["tiling.blocked_layout"]["present"] is True
    assert mechanisms["mapping.program_axes"]["attrs"]["axes"] == ["x"]
    assert mechanisms["primitive.mma"]["present"] is True
    assert mechanisms["fusion.epilogue_fused_writeback"]["present"] is True
    assert mechanisms["primitive.dot_op"]["present"] is True
    assert mechanisms["primitive.dot_op"]["attrs"]["reduction_scope"] == "warp"
    assert mechanisms["layout.output_convert"]["present"] is True
    assert mechanisms["layout.output_convert"]["attrs"]["layout_convert_sites"] == 1
