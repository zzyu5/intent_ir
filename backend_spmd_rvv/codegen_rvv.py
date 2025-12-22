"""
Experimental RVV codegen (Task6) for simple kernels.

Historically this module contained per-kernel RVV templates (e.g. reduce_any).
The current pipeline uses `backend_spmd_rvv.lower_c.lower_intent_to_c_with_files`
to lower the full IntentIR ops list into a standalone C program (RVV-enabled
where possible).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from intent_ir.ir_types import IntentFunction, IntentIRValidationError, Op
from .tiling_search import TileChoice
from .hardware_profile import RVVHardwareProfile
from .lower_c import lower_intent_to_c_with_files


def _find_reduce_any(intent: IntentFunction) -> Optional[Op]:
    for op in intent.ops:
        if op.op == "reduce_any":
            return op
    return None


def generate_reduce_any(intent: IntentFunction, shape_bindings: Dict[str, int]) -> str:
    op = _find_reduce_any(intent)
    if not op:
        raise IntentIRValidationError("reduce_any op not found for RVV codegen")
    inp = op.inputs[0] if op.inputs else "inp"
    out = op.output or "out"
    M = shape_bindings.get("M")
    N = shape_bindings.get("N")
    if not (M and N):
        raise IntentIRValidationError("shape_bindings must include M and N for reduce_any")
    lines = []
    lines.append("#include <stdint.h>")
    lines.append("#include <stddef.h>")
    lines.append("#include <riscv_vector.h>")
    lines.append("")
    lines.append("/* RVV reduce_any: out[m] = any(inp[m, :]) */")
    lines.append("void reduce_any_rvv(const float* inp, uint8_t* out, int M, int N) {")
    lines.append("  for (int m = 0; m < M; ++m) {")
    lines.append("    int n = 0;")
    lines.append("    const float* base = inp + m * N;")
    lines.append("    while (n < N) {")
    lines.append("      size_t vl = __riscv_vsetvl_e32m1(N - n);")
    lines.append("      vfloat32m1_t v = __riscv_vle32_v_f32m1(base + n, vl);")
    lines.append("      vbool32_t msk = __riscv_vmfne_vf_f32m1_b32(v, 0.0f, vl);")
    lines.append("      int idx = __riscv_vfirst_m_b32(msk, vl);")
    lines.append("      if (idx >= 0) { out[m] = 1; goto next_row; }")
    lines.append("      n += vl;")
    lines.append("    }")
    lines.append("    out[m] = 0;")
    lines.append("next_row: ;")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append("int main() {")
    lines.append(f"  const int M = {M};")
    lines.append(f"  const int N = {N};")
    lines.append("  float inp[M * N];")
    lines.append("  for (int i = 0; i < M * N; ++i) inp[i] = (i % 3 == 0) ? 1.0f : 0.0f;")
    lines.append("  uint8_t out[M];")
    lines.append("  reduce_any_rvv(inp, out, M, N);")
    lines.append("  // return sum of outputs as checksum")
    lines.append("  int s = 0; for (int i = 0; i < M; ++i) s += out[i];")
    lines.append("  return (s == M) ? 0 : 1;")
    lines.append("}")
    return "\n".join(lines)


def _format_c_array(name: str, arr: np.ndarray, c_type: str) -> Sequence[str]:
    flat = arr.reshape(-1)
    elems = ", ".join(str(x) for x in flat)
    return [f"static {c_type} {name}[{flat.size}] = {{{elems}}};"]


def generate_reduce_any_with_data(intent: IntentFunction, inputs: Dict[str, np.ndarray], outputs: Dict[str, np.ndarray], shape_bindings: Dict[str, int]) -> str:
    """
    Generate RVV C for reduce_any with embedded inputs/expected outputs.
    Returns C source that exits 0 on exact match, else 1.
    """
    if "inp" not in inputs or "out" not in outputs:
        raise IntentIRValidationError("reduce_any with data requires inputs['inp'] and outputs['out']")
    inp_arr = np.asarray(inputs["inp"], dtype=np.float32)
    out_arr = np.asarray(outputs["out"], dtype=np.uint8)
    M = shape_bindings.get("M")
    N = shape_bindings.get("N")
    if not (M and N):
        raise IntentIRValidationError("shape_bindings must include M and N")
    lines = []
    lines.append("#include <stdint.h>")
    lines.append("#include <stddef.h>")
    lines.append("#include <stdio.h>")
    lines.append("#include <riscv_vector.h>")
    lines.append("")
    lines.extend(_format_c_array("inp_ref", inp_arr, "float"))
    lines.extend(_format_c_array("out_ref", out_arr, "uint8_t"))
    lines.append("")
    lines.append("void reduce_any_rvv(const float* inp, uint8_t* out, int M, int N) {")
    lines.append("  for (int m = 0; m < M; ++m) {")
    lines.append("    int n = 0;")
    lines.append("    const float* base = inp + m * N;")
    lines.append("    out[m] = 0;")
    lines.append("    while (n < N) {")
    lines.append("      size_t vl = __riscv_vsetvl_e32m1(N - n);")
    lines.append("      vfloat32m1_t v = __riscv_vle32_v_f32m1(base + n, vl);")
    lines.append("      vbool32_t msk = __riscv_vmfne_vf_f32m1_b32(v, 0.0f, vl);")
    lines.append("      int idx = __riscv_vfirst_m_b32(msk, vl);")
    lines.append("      if (idx >= 0) { out[m] = 1; break; }")
    lines.append("      n += vl;")
    lines.append("    }")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append("int main() {")
    lines.append(f"  const int M = {int(M)};")
    lines.append(f"  const int N = {int(N)};")
    lines.append("  uint8_t out[M];")
    lines.append("  reduce_any_rvv(inp_ref, out, M, N);")
    lines.append("  int sum_got = 0, sum_ref = 0;")
    lines.append("  for (int i = 0; i < M; ++i) { sum_got += (int)out[i]; sum_ref += (int)out_ref[i]; }")
    lines.append("  for (int i = 0; i < M; ++i) {")
    lines.append("    if (out[i] != out_ref[i]) {")
    lines.append("      fprintf(stderr, \"mismatch at %d: got %u expected %u\\n\", i, (unsigned)out[i], (unsigned)out_ref[i]);")
    lines.append("      fprintf(stderr, \"sum_got=%d sum_ref=%d\\n\", sum_got, sum_ref);")
    lines.append("      fprintf(stderr, \"out_got:\");")
    lines.append("      for (int j = 0; j < M; ++j) fprintf(stderr, \" %u\", (unsigned)out[j]);")
    lines.append("      fprintf(stderr, \"\\nout_ref:\");")
    lines.append("      for (int j = 0; j < M; ++j) fprintf(stderr, \" %u\", (unsigned)out_ref[j]);")
    lines.append("      fprintf(stderr, \"\\n\");")
    lines.append("      return 1;")
    lines.append("    }")
    lines.append("  }")
    lines.append("  printf(\"PASS reduce_any M=%d N=%d sum=%d\\n\", M, N, sum_got);")
    lines.append("  printf(\"out:\");")
    lines.append("  for (int i = 0; i < M; ++i) printf(\" %u\", (unsigned)out[i]);")
    lines.append("  printf(\"\\n\");")
    lines.append("  return 0;")
    lines.append("}")
    return "\n".join(lines)


def generate_groupnorm_with_files(intent: IntentFunction, *, shape_bindings: Dict[str, int], atol: float = 1e-3, rtol: float = 1e-3) -> str:
    """
    Generate a self-contained C program that:
    - reads X/W/B + expected Y/Mean/Rstd from local .bin files (float32, C-order)
    - computes groupnorm in the canonical view X/Y:[N,C,HW], stats:[N,G]
    - compares with tolerances and prints max_abs/max_rel + first mismatch

    Expected files (in current working directory):
      X.bin, W.bin, B.bin, Y_ref.bin, Mean_ref.bin, Rstd_ref.bin
    """
    N = int(shape_bindings.get("N", 0))
    C = int(shape_bindings.get("C", 0))
    HW = int(shape_bindings.get("HW", 0))
    G = int(shape_bindings.get("num_groups", shape_bindings.get("group", 0)))
    if not (N and C and HW and G):
        raise IntentIRValidationError("groupnorm codegen requires N,C,HW,num_groups bindings")
    if C % G != 0:
        raise IntentIRValidationError(f"groupnorm requires C divisible by num_groups: C={C} G={G}")
    group_size = int(shape_bindings.get("group_size", C // G))
    eps = 1e-5
    # Prefer an explicit eps const if present.
    for op in intent.ops:
        if op.op == "const" and op.output and op.output.lower().startswith("eps"):
            v = op.attrs.get("value")
            if isinstance(v, (int, float, np.number)):
                eps = float(v)
            break

    lines: list[str] = []
    lines.append("#include <math.h>")
    lines.append("#include <stdint.h>")
    lines.append("#include <stddef.h>")
    lines.append("#include <stdio.h>")
    lines.append("#include <stdlib.h>")
    lines.append("")
    lines.append("static int read_f32(const char* path, float* dst, size_t n) {")
    lines.append("  FILE* f = fopen(path, \"rb\");")
    lines.append("  if (!f) { perror(path); return 0; }")
    lines.append("  size_t got = fread(dst, sizeof(float), n, f);")
    lines.append("  fclose(f);")
    lines.append("  return got == n;")
    lines.append("}")
    lines.append("")
    lines.append("static int compare_f32(const char* name, const float* got, const float* ref, size_t n, float atol, float rtol) {")
    lines.append("  double max_abs = 0.0, max_rel = 0.0;")
    lines.append("  size_t worst = 0;")
    lines.append("  for (size_t i = 0; i < n; ++i) {")
    lines.append("    double a = (double)got[i];")
    lines.append("    double b = (double)ref[i];")
    lines.append("    double abs_e = fabs(a - b);")
    lines.append("    double rel_e = abs_e / (fabs(b) + 1e-8);")
    lines.append("    if (abs_e > max_abs) { max_abs = abs_e; max_rel = rel_e; worst = i; }")
    lines.append("  }")
    lines.append("  int ok = (max_abs <= atol) || (max_rel <= rtol);")
    lines.append("  printf(\"%s: ok=%d max_abs=%g max_rel=%g worst_i=%zu got=%g ref=%g\\n\",")
    lines.append("         name, ok, max_abs, max_rel, worst, (double)got[worst], (double)ref[worst]);")
    lines.append("  return ok;")
    lines.append("}")
    lines.append("")
    lines.append("int main() {")
    lines.append(f"  const int N = {N};")
    lines.append(f"  const int C = {C};")
    lines.append(f"  const int HW = {HW};")
    lines.append(f"  const int G = {G};")
    lines.append(f"  const int GS = {group_size};")
    lines.append(f"  const float eps = {eps:.10g}f;")
    lines.append(f"  const float atol = {float(atol):.10g}f;")
    lines.append(f"  const float rtol = {float(rtol):.10g}f;")
    lines.append("  size_t Xn = (size_t)N * (size_t)C * (size_t)HW;")
    lines.append("  float* X = (float*)malloc(sizeof(float) * Xn);")
    lines.append("  float* W = (float*)malloc(sizeof(float) * (size_t)C);")
    lines.append("  float* B = (float*)malloc(sizeof(float) * (size_t)C);")
    lines.append("  float* Y = (float*)malloc(sizeof(float) * Xn);")
    lines.append("  float* Yref = (float*)malloc(sizeof(float) * Xn);")
    lines.append("  float* Mean = (float*)malloc(sizeof(float) * (size_t)N * (size_t)G);")
    lines.append("  float* Rstd = (float*)malloc(sizeof(float) * (size_t)N * (size_t)G);")
    lines.append("  float* MeanRef = (float*)malloc(sizeof(float) * (size_t)N * (size_t)G);")
    lines.append("  float* RstdRef = (float*)malloc(sizeof(float) * (size_t)N * (size_t)G);")
    lines.append("  if (!X||!W||!B||!Y||!Yref||!Mean||!Rstd||!MeanRef||!RstdRef) { fprintf(stderr, \"alloc failed\\n\"); return 2; }")
    lines.append("  if (!read_f32(\"X.bin\", X, Xn)) return 2;")
    lines.append("  if (!read_f32(\"W.bin\", W, (size_t)C)) return 2;")
    lines.append("  if (!read_f32(\"B.bin\", B, (size_t)C)) return 2;")
    lines.append("  if (!read_f32(\"Y_ref.bin\", Yref, Xn)) return 2;")
    lines.append("  if (!read_f32(\"Mean_ref.bin\", MeanRef, (size_t)N*(size_t)G)) return 2;")
    lines.append("  if (!read_f32(\"Rstd_ref.bin\", RstdRef, (size_t)N*(size_t)G)) return 2;")
    lines.append("")
    lines.append("  // groupnorm: for each (n,g), reduce over c in group and hw.")
    lines.append("  for (int n = 0; n < N; ++n) {")
    lines.append("    for (int g = 0; g < G; ++g) {")
    lines.append("      double sum = 0.0;")
    lines.append("      for (int gs = 0; gs < GS; ++gs) {")
    lines.append("        int c = g * GS + gs;")
    lines.append("        for (int hw = 0; hw < HW; ++hw) {")
    lines.append("          size_t idx = ((size_t)n * (size_t)C + (size_t)c) * (size_t)HW + (size_t)hw;")
    lines.append("          sum += (double)X[idx];")
    lines.append("        }")
    lines.append("      }")
    lines.append("      double denom = (double)GS * (double)HW;")
    lines.append("      float mean = (float)(sum / denom);")
    lines.append("      Mean[(size_t)n * (size_t)G + (size_t)g] = mean;")
    lines.append("      double var_sum = 0.0;")
    lines.append("      for (int gs = 0; gs < GS; ++gs) {")
    lines.append("        int c = g * GS + gs;")
    lines.append("        for (int hw = 0; hw < HW; ++hw) {")
    lines.append("          size_t idx = ((size_t)n * (size_t)C + (size_t)c) * (size_t)HW + (size_t)hw;")
    lines.append("          double d = (double)X[idx] - (double)mean;")
    lines.append("          var_sum += d * d;")
    lines.append("        }")
    lines.append("      }")
    lines.append("      float var = (float)(var_sum / denom);")
    lines.append("      float rstd = 1.0f / sqrtf(var + eps);")
    lines.append("      Rstd[(size_t)n * (size_t)G + (size_t)g] = rstd;")
    lines.append("      for (int gs = 0; gs < GS; ++gs) {")
    lines.append("        int c = g * GS + gs;")
    lines.append("        float w = W[c];")
    lines.append("        float b = B[c];")
    lines.append("        for (int hw = 0; hw < HW; ++hw) {")
    lines.append("          size_t idx = ((size_t)n * (size_t)C + (size_t)c) * (size_t)HW + (size_t)hw;")
    lines.append("          float xn = (X[idx] - mean) * rstd;")
    lines.append("          Y[idx] = xn * w + b;")
    lines.append("        }")
    lines.append("      }")
    lines.append("    }")
    lines.append("  }")
    lines.append("")
    lines.append("  int okY = compare_f32(\"Y\", Y, Yref, Xn, atol, rtol);")
    lines.append("  int okM = compare_f32(\"Mean\", Mean, MeanRef, (size_t)N*(size_t)G, atol, rtol);")
    lines.append("  int okR = compare_f32(\"Rstd\", Rstd, RstdRef, (size_t)N*(size_t)G, atol, rtol);")
    lines.append("  int ok = okY && okM && okR;")
    lines.append("  printf(ok ? \"PASS groupnorm\\n\" : \"FAIL groupnorm\\n\");")
    lines.append("  return ok ? 0 : 1;")
    lines.append("}")
    return "\n".join(lines)


def generate_attention_with_files(intent: IntentFunction, *, shape_bindings: Dict[str, int], atol: float = 1e-2, rtol: float = 1e-2) -> str:
    """
    Generate a C program that reads Q/K/V/attn_mask/sm_scale and expected Out from .bin files,
    computes standard attention (matmul+softmax+matmul) in float32, and compares with tolerance.

    Expected files:
      Q.bin, K.bin, V.bin, attn_mask.bin, sm_scale.bin, Out_ref.bin
    """
    B = int(shape_bindings.get("batch", 0))
    H = int(shape_bindings.get("q_numhead", shape_bindings.get("H", 0)))
    HK = int(shape_bindings.get("kv_numhead", H))
    QCTX = int(shape_bindings.get("Q_CTX", 0))
    KCTX = int(shape_bindings.get("KV_CTX", 0))
    D = int(shape_bindings.get("HEAD_DIM", 0))
    if not (B and H and QCTX and KCTX and D):
        raise IntentIRValidationError("attention codegen requires batch,q_numhead,kv_numhead,Q_CTX,KV_CTX,HEAD_DIM bindings")
    if HK != H:
        # Keep it explicit; extend later if needed.
        raise IntentIRValidationError(f"attention codegen currently requires kv_numhead == q_numhead (got {HK} vs {H})")

    lines: list[str] = []
    lines.append("#include <math.h>")
    lines.append("#include <stdint.h>")
    lines.append("#include <stddef.h>")
    lines.append("#include <stdio.h>")
    lines.append("#include <stdlib.h>")
    lines.append("")
    lines.append("static int read_f32(const char* path, float* dst, size_t n) {")
    lines.append("  FILE* f = fopen(path, \"rb\");")
    lines.append("  if (!f) { perror(path); return 0; }")
    lines.append("  size_t got = fread(dst, sizeof(float), n, f);")
    lines.append("  fclose(f);")
    lines.append("  return got == n;")
    lines.append("}")
    lines.append("")
    lines.append("static int compare_f32(const char* name, const float* got, const float* ref, size_t n, float atol, float rtol) {")
    lines.append("  double max_abs = 0.0, max_rel = 0.0; size_t worst = 0;")
    lines.append("  for (size_t i = 0; i < n; ++i) {")
    lines.append("    double a = (double)got[i]; double b = (double)ref[i];")
    lines.append("    double abs_e = fabs(a - b); double rel_e = abs_e / (fabs(b) + 1e-8);")
    lines.append("    if (abs_e > max_abs) { max_abs = abs_e; max_rel = rel_e; worst = i; }")
    lines.append("  }")
    lines.append("  int ok = (max_abs <= atol) || (max_rel <= rtol);")
    lines.append("  printf(\"%s: ok=%d max_abs=%g max_rel=%g worst_i=%zu got=%g ref=%g\\n\", name, ok, max_abs, max_rel, worst, (double)got[worst], (double)ref[worst]);")
    lines.append("  return ok;")
    lines.append("}")
    lines.append("")
    lines.append("static inline size_t idx4(int b,int h,int i,int j,int H,int I,int J) {")
    lines.append("  return ((size_t)b*(size_t)H + (size_t)h)*(size_t)I*(size_t)J + (size_t)i*(size_t)J + (size_t)j;")
    lines.append("}")
    lines.append("")
    lines.append("int main() {")
    lines.append(f"  const int B = {B};")
    lines.append(f"  const int H = {H};")
    lines.append(f"  const int Q = {QCTX};")
    lines.append(f"  const int K = {KCTX};")
    lines.append(f"  const int D = {D};")
    lines.append(f"  const float atol = {float(atol):.10g}f;")
    lines.append(f"  const float rtol = {float(rtol):.10g}f;")
    lines.append("  size_t Qn = (size_t)B*(size_t)H*(size_t)Q*(size_t)D;")
    lines.append("  size_t Kn = (size_t)B*(size_t)H*(size_t)K*(size_t)D;")
    lines.append("  size_t Mn = (size_t)B*(size_t)H*(size_t)Q*(size_t)K;")
    lines.append("  size_t On = (size_t)B*(size_t)H*(size_t)Q*(size_t)D;")
    lines.append("  float* q = (float*)malloc(sizeof(float)*Qn);")
    lines.append("  float* k = (float*)malloc(sizeof(float)*Kn);")
    lines.append("  float* v = (float*)malloc(sizeof(float)*Kn);")
    lines.append("  float* mask = (float*)malloc(sizeof(float)*Mn);")
    lines.append("  float* qk = (float*)malloc(sizeof(float)*Mn);")
    lines.append("  float* w = (float*)malloc(sizeof(float)*Mn);")
    lines.append("  float* out = (float*)malloc(sizeof(float)*On);")
    lines.append("  float* out_ref = (float*)malloc(sizeof(float)*On);")
    lines.append("  float sm_scale = 1.0f;")
    lines.append("  if (!q||!k||!v||!mask||!qk||!w||!out||!out_ref) { fprintf(stderr, \"alloc failed\\n\"); return 2; }")
    lines.append("  if (!read_f32(\"Q.bin\", q, Qn)) return 2;")
    lines.append("  if (!read_f32(\"K.bin\", k, Kn)) return 2;")
    lines.append("  if (!read_f32(\"V.bin\", v, Kn)) return 2;")
    lines.append("  if (!read_f32(\"attn_mask.bin\", mask, Mn)) return 2;")
    lines.append("  if (!read_f32(\"sm_scale.bin\", &sm_scale, 1)) return 2;")
    lines.append("  if (!read_f32(\"Out_ref.bin\", out_ref, On)) return 2;")
    lines.append("")
    lines.append("  // QK = Q @ K^T (over D)")
    lines.append("  for (int b = 0; b < B; ++b) {")
    lines.append("    for (int h = 0; h < H; ++h) {")
    lines.append("      for (int qi = 0; qi < Q; ++qi) {")
    lines.append("        for (int kj = 0; kj < K; ++kj) {")
    lines.append("          double acc = 0.0;")
    lines.append("          for (int d = 0; d < D; ++d) {")
    lines.append("            size_t q_idx = idx4(b,h,qi,d,H,Q,D);")
    lines.append("            size_t k_idx = idx4(b,h,kj,d,H,K,D);")
    lines.append("            acc += (double)q[q_idx] * (double)k[k_idx];")
    lines.append("          }")
    lines.append("          size_t m_idx = idx4(b,h,qi,kj,H,Q,K);")
    lines.append("          float x = (float)acc * sm_scale + mask[m_idx];")
    lines.append("          qk[m_idx] = x;")
    lines.append("        }")
    lines.append("      }")
    lines.append("    }")
    lines.append("  }")
    lines.append("")
    lines.append("  // softmax over last dim K")
    lines.append("  for (int b = 0; b < B; ++b) {")
    lines.append("    for (int h = 0; h < H; ++h) {")
    lines.append("      for (int qi = 0; qi < Q; ++qi) {")
    lines.append("        double mx = -1e30;")
    lines.append("        for (int kj = 0; kj < K; ++kj) {")
    lines.append("          size_t m_idx = idx4(b,h,qi,kj,H,Q,K);")
    lines.append("          double vv = (double)qk[m_idx]; if (vv > mx) mx = vv;")
    lines.append("        }")
    lines.append("        double sum = 0.0;")
    lines.append("        for (int kj = 0; kj < K; ++kj) {")
    lines.append("          size_t m_idx = idx4(b,h,qi,kj,H,Q,K);")
    lines.append("          double e = exp((double)qk[m_idx] - mx);")
    lines.append("          w[m_idx] = (float)e;")
    lines.append("          sum += e;")
    lines.append("        }")
    lines.append("        double inv = 1.0 / sum;")
    lines.append("        for (int kj = 0; kj < K; ++kj) {")
    lines.append("          size_t m_idx = idx4(b,h,qi,kj,H,Q,K);")
    lines.append("          w[m_idx] = (float)((double)w[m_idx] * inv);")
    lines.append("        }")
    lines.append("      }")
    lines.append("    }")
    lines.append("  }")
    lines.append("")
    lines.append("  // Out = W @ V (over K)")
    lines.append("  for (int b = 0; b < B; ++b) {")
    lines.append("    for (int h = 0; h < H; ++h) {")
    lines.append("      for (int qi = 0; qi < Q; ++qi) {")
    lines.append("        for (int d = 0; d < D; ++d) {")
    lines.append("          double acc = 0.0;")
    lines.append("          for (int kj = 0; kj < K; ++kj) {")
    lines.append("            size_t w_idx = idx4(b,h,qi,kj,H,Q,K);")
    lines.append("            size_t v_idx = idx4(b,h,kj,d,H,K,D);")
    lines.append("            acc += (double)w[w_idx] * (double)v[v_idx];")
    lines.append("          }")
    lines.append("          size_t o_idx = idx4(b,h,qi,d,H,Q,D);")
    lines.append("          out[o_idx] = (float)acc;")
    lines.append("        }")
    lines.append("      }")
    lines.append("    }")
    lines.append("  }")
    lines.append("")
    lines.append("  int ok = compare_f32(\"Out\", out, out_ref, On, atol, rtol);")
    lines.append("  printf(ok ? \"PASS attention\\n\" : \"FAIL attention\\n\");")
    lines.append("  return ok ? 0 : 1;")
    lines.append("}")
    return "\n".join(lines)


def generate_rvv(intent: IntentFunction, *, tile: Optional[TileChoice] = None, profile: Optional[RVVHardwareProfile] = None, shape_bindings: Optional[Dict[str, int]] = None) -> str:
    """
    Dispatch RVV codegen based on intent ops. Currently only reduce_any is supported.
    """
    if shape_bindings is None:
        shape_bindings = {}
    # reduce_any path
    if _find_reduce_any(intent):
        return generate_reduce_any(intent, shape_bindings)
    raise IntentIRValidationError("RVV codegen: unsupported intent (only reduce_any is implemented)")


__all__ = [
    "generate_rvv",
    "generate_reduce_any_with_data",
    "generate_groupnorm_with_files",
    "generate_attention_with_files",
    "lower_intent_to_c_with_files",
]
