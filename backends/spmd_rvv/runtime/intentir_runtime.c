#include "intentir_runtime.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void intentir_runtime_init(void) {
#ifdef _OPENMP
  const char* t = getenv("INTENTIR_OMP_THREADS");
  if (t && *t) {
    int n = atoi(t);
    if (n > 0) {
      // Keep behavior deterministic for benchmarking.
      omp_set_dynamic(0);
      omp_set_num_threads(n);
    }
  }
#endif
}

int intentir_read_bytes(const char* path, void* dst, size_t bytes) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return 0;
  }
  size_t got = fread(dst, 1, bytes, f);
  fclose(f);
  return got == bytes;
}

int intentir_compare_f32(const char* name, const float* got, const float* ref, size_t n, float atol, float rtol) {
  double max_abs = 0.0, max_rel = 0.0;
  size_t worst = 0;
  for (size_t i = 0; i < n; ++i) {
    double a = (double)got[i];
    double b = (double)ref[i];
    double abs_e = fabs(a - b);
    double rel_e = abs_e / (fabs(b) + 1e-8);
    if (abs_e > max_abs) {
      max_abs = abs_e;
      max_rel = rel_e;
      worst = i;
    }
  }
  int ok = (max_abs <= (double)atol) || (max_rel <= (double)rtol);
  printf("%s: ok=%d max_abs=%g max_rel=%g worst_i=%zu got=%g ref=%g\n", name, ok, max_abs, max_rel, worst,
         (double)got[worst], (double)ref[worst]);
  return ok;
}

int intentir_compare_u8(const char* name, const uint8_t* got, const uint8_t* ref, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (got[i] != ref[i]) {
      fprintf(stderr, "%s mismatch at %zu: got=%u ref=%u\n", name, i, (unsigned)got[i], (unsigned)ref[i]);
      return 0;
    }
  }
  printf("%s: ok=1 (exact)\n", name);
  return 1;
}

uint64_t intentir_now_ns(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
