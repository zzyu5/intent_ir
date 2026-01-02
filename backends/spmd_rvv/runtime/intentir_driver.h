#ifndef INTENTIR_DRIVER_H
#define INTENTIR_DRIVER_H

#include "intentir_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum IntentirDType {
  INTENTIR_DTYPE_F32 = 0,
  INTENTIR_DTYPE_U8 = 1,
} IntentirDType;

typedef struct IntentirBufferDesc {
  const char* name;  // base name for {name}.bin / {name}_ref.bin
  void** ptr;        // storage for buffer pointer (set by alloc/load)
  size_t bytes;      // number of bytes to allocate/read/compare
  IntentirDType dtype;
} IntentirBufferDesc;

int intentir_alloc(IntentirBufferDesc* bufs, size_t n);
int intentir_alloc_and_load_inputs(IntentirBufferDesc* inputs, size_t n);
int intentir_compare_outputs_with_refs(const IntentirBufferDesc* outputs, size_t n, float atol, float rtol);

typedef void (*IntentirComputeFn)(void);
void intentir_maybe_bench(IntentirComputeFn compute, double matmul_flops_total);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // INTENTIR_DRIVER_H

