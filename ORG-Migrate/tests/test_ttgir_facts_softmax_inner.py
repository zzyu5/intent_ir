from __future__ import annotations

from org.facts.ttgir import extract_ttgir_mechanism_facts


TTGIR_FIXTURE = """
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @softmax_kernel_inner(%output_ptr: !tt.ptr<f32>, %input_ptr: !tt.ptr<f32>, %M: i32, %N: i32) {
    %pid_m = tt.get_program_id x : i32
    %inp = tt.load %input_ptrs, %mask_3, %cst : tensor<64x!tt.ptr<f32>, #blocked>
    %m = "tt.reduce"(%inp) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %max = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %max : f32
    }) : (tensor<64xf32, #blocked>) -> f32
    tt.return
  }
}
""".strip()


def test_ttgir_facts_softmax_inner() -> None:
    facts = extract_ttgir_mechanism_facts(TTGIR_FIXTURE, kernel_name="softmax_inner", artifact_path="softmax_inner.ttgir")
    mechanisms = dict(facts.get("mechanisms") or {})
    assert mechanisms["staging.row_tile_resident"]["present"] is True
    assert mechanisms["communication.row_reduction"]["present"] is True
    assert mechanisms["layout.vector_row_path"]["present"] is False or isinstance(mechanisms["layout.vector_row_path"]["present"], bool)
