from __future__ import annotations

from org.facts.ttgir import extract_ttgir_mechanism_facts


TTGIR_FIXTURE = """
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @masked_softmax2d_kernel(%inp_ptr: !tt.ptr<f32>, %mask_ptr: !tt.ptr<i1>, %out_ptr: !tt.ptr<f32>, %M: i32, %N: i32) {
    %pid_m = tt.get_program_id x : i32
    %x_12 = tt.load %inp_ptr, %in_bounds_8, %cst : tensor<256x!tt.ptr<f32>, #blocked>
    %m_15 = tt.load %m_14, %in_bounds_6, %cst_1 : tensor<256x!tt.ptr<i8>, #blocked>
    %x_17 = arith.select %m_16, %x_12, %cst_0 : tensor<256xi1, #blocked>, tensor<256xf32, #blocked>
    %mx = "tt.reduce"(%x_17) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %max = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %max : f32
    }) : (tensor<256xf32, #blocked>) -> f32
    tt.return
  }
}
""".strip()


def test_ttgir_facts_masked_softmax2d() -> None:
    facts = extract_ttgir_mechanism_facts(TTGIR_FIXTURE, kernel_name="masked_softmax2d", artifact_path="masked_softmax2d.ttgir")
    mechanisms = dict(facts.get("mechanisms") or {})
    assert mechanisms["staging.row_tile_resident"]["present"] is True
    assert mechanisms["communication.row_reduction"]["present"] is True
    assert mechanisms["communication.mask_apply"]["present"] is True
