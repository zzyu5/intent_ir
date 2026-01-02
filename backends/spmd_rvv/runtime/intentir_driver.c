#include "intentir_driver.h"

int intentir_alloc(IntentirBufferDesc* bufs, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    IntentirBufferDesc* b = &bufs[i];
    if (!b->ptr) return 0;
    if (b->bytes == 0) b->bytes = 1;
    void* p = malloc(b->bytes);
    if (!p) {
      fprintf(stderr, "alloc failed: %s bytes=%zu\n", (b->name ? b->name : "(null)"), b->bytes);
      return 0;
    }
    *b->ptr = p;
  }
  return 1;
}

int intentir_alloc_and_load_inputs(IntentirBufferDesc* inputs, size_t n) {
  if (!intentir_alloc(inputs, n)) return 0;
  for (size_t i = 0; i < n; ++i) {
    IntentirBufferDesc* b = &inputs[i];
    if (!b->name || !b->ptr || !*b->ptr) return 0;
    char path[512];
    snprintf(path, sizeof(path), "%s.bin", b->name);
    if (!intentir_read_bytes(path, *b->ptr, b->bytes)) return 0;
  }
  return 1;
}

int intentir_compare_outputs_with_refs(const IntentirBufferDesc* outputs, size_t n, float atol, float rtol) {
  int ok_all = 1;
  for (size_t i = 0; i < n; ++i) {
    const IntentirBufferDesc* b = &outputs[i];
    if (!b->name || !b->ptr || !*b->ptr) return 0;
    if (b->bytes == 0) return 0;
    char path[512];
    snprintf(path, sizeof(path), "%s_ref.bin", b->name);
    void* ref = malloc(b->bytes);
    if (!ref) {
      fprintf(stderr, "alloc failed: %s_ref bytes=%zu\n", b->name, b->bytes);
      return 0;
    }
    if (!intentir_read_bytes(path, ref, b->bytes)) {
      free(ref);
      return 0;
    }
    int ok = 0;
    if (b->dtype == INTENTIR_DTYPE_U8) {
      ok = intentir_compare_u8(b->name, (const uint8_t*)(*b->ptr), (const uint8_t*)ref, b->bytes);
    } else {
      ok = intentir_compare_f32(b->name, (const float*)(*b->ptr), (const float*)ref, b->bytes / sizeof(float), atol, rtol);
    }
    if (!ok) ok_all = 0;
    free(ref);
  }
  return ok_all;
}

void intentir_maybe_bench(IntentirComputeFn compute, double matmul_flops_total) {
  if (!compute) return;
  int bench_iters = 0;
  int bench_warmup = 1;
  const char* b = getenv("INTENTIR_BENCH_ITERS");
  if (b) bench_iters = atoi(b);
  const char* w = getenv("INTENTIR_BENCH_WARMUP");
  if (w) bench_warmup = atoi(w);
  if (bench_iters <= 0) return;
  if (bench_warmup < 0) bench_warmup = 0;
  for (int i = 0; i < bench_warmup; ++i) compute();
  uint64_t t0 = intentir_now_ns();
  for (int i = 0; i < bench_iters; ++i) compute();
  uint64_t t1 = intentir_now_ns();
  double ns_total = (double)(t1 - t0);
  double ns_per_iter = ns_total / (double)bench_iters;
  double gflops = (ns_per_iter > 0.0) ? (matmul_flops_total / ns_per_iter) : 0.0;
  printf("INTENTIR_BENCH {\"iters\":%d,\"warmup\":%d,\"ns_total\":%llu,\"ns_per_iter\":%.1f,\"matmul_flops\":%.0f,\"matmul_gflops\":%.6f}\n",
         bench_iters, bench_warmup, (unsigned long long)(t1 - t0), ns_per_iter, matmul_flops_total, gflops);
}

