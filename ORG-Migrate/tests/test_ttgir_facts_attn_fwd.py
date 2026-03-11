from __future__ import annotations

from org.facts.ttgir import extract_ttgir_mechanism_facts


TTGIR_FIXTURE = """
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd(%Q: !tt.ptr<f32>, %K: !tt.ptr<f32>, %V: !tt.ptr<f32>, %attn_mask: !tt.ptr<f32>) {
    %q_0 = tt.load %Qv, %mask_q, %cst : tensor<16x64x!tt.ptr<f32>, #blocked>
    %acc = scf.for %tile = %c0_i32 to %KV_CTX step %c16_i32 iter_args(%m = %neg_inf) -> (f32) {
      %k_0 = tt.load %Kv, %mask_k, %cst : tensor<64x16x!tt.ptr<f32>, #blocked>
      %v_0 = tt.load %Vv, %mask_v, %cst : tensor<16x64x!tt.ptr<f32>, #blocked>
      %pred_causal = arith.cmpi sle, %kv, %q : i1
      %m_ij = "tt.reduce"(%scores) <{axis = 1 : i32}> ({
      ^bb0(%lhs: f32, %rhs: f32):
        %max = arith.maxnumf %lhs, %rhs : f32
        tt.reduce.return %max : f32
      }) : (tensor<16x16xf32, #blocked>) -> tensor<16xf32, #blocked>
      scf.yield %m
    }
    %dot = tt.dot %a, %b : tensor<16x64xf32, #blocked> * tensor<64x16xf32, #blocked>
    tt.return
  }
}
""".strip()


def test_ttgir_facts_attn_fwd() -> None:
    facts = extract_ttgir_mechanism_facts(TTGIR_FIXTURE, kernel_name="_attn_fwd", artifact_path="attn_fwd.ttgir")
    mechanisms = dict(facts.get("mechanisms") or {})
    assert mechanisms["staging.q_resident_state"]["present"] is True
    assert mechanisms["staging.kv_streamed_tiles"]["present"] is True
    assert mechanisms["communication.streaming_softmax"]["present"] is True
    assert mechanisms["communication.mask_causal"]["present"] is True
    assert mechanisms["primitive.dot_op"]["present"] is True
    assert mechanisms["mapping.program_axes"]["present"] in {False, True}
