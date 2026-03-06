from __future__ import annotations

from org.facts.ttgir import extract_ttgir_mechanism_facts


TTGIR_FIXTURE = """
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @flash_attention2d_kernel(%Q_ptr: !tt.ptr<f32>, %K_ptr: !tt.ptr<f32>, %V_ptr: !tt.ptr<f32>, %Out_ptr: !tt.ptr<f32>, %sm_scale: f32) {
    %pid_q = tt.get_program_id x : i32
    %k_33 = tt.load %k_32, %k_25, %cst_2 : tensor<32x64x!tt.ptr<f32>, #blocked>
    %m_ij = "tt.reduce"(%scores_46) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %max = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %max : f32
    }) : (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> f32
    tt.return
  }
}
""".strip()


def test_ttgir_facts_flash_attention2d() -> None:
    facts = extract_ttgir_mechanism_facts(TTGIR_FIXTURE, kernel_name="flash_attention2d", artifact_path="flash.ttgir")
    mechanisms = dict(facts.get("mechanisms") or {})
    assert facts["schema_version"] == "org_mechanism_facts_v1"
    assert mechanisms["tiling.blocked_layout"]["present"] is True
    assert mechanisms["staging.local_or_shared"]["present"] is True
    assert mechanisms["mapping.program_axes"]["attrs"]["axes"] == ["x"]
    assert mechanisms["mapping.warp_or_cta"]["attrs"]["num_warps"] == 4
    assert mechanisms["communication.reduction"]["present"] is True
