# Intent MLIR Dialect (Transitional Spec v0)

This document defines the transitional Intent dialect used in the dual-track
IntentIR -> MLIR migration.

## Design goals

1. Keep a lossless bridge to current `IntentFunction` during migration.
2. Enable pass orchestration (`upstream/midend/downstream_*`) with traceable artifacts.
3. Stay backend-agnostic at dialect level.

## Module shape

```mlir
module attributes {intent.dialect_version = "intent_dialect_v0"} {
  intent.func @kernel_name() {
    %tmp = intent.add(%a, %b) {kind="elementwise", tier="core"} : !intent.tensor<?Mx?Nx f32>
    intent.return %tmp
  }
  // intentir_json_begin
  // <base64(json payload)>
  // intentir_json_end
}
```

Notes:

1. The `intentir_json_*` payload is migration-only and guarantees roundtrip.
2. When backend direct MLIR lowering is complete, this payload can be removed.

## Pass pipelines

1. `upstream.yaml`: symbol/provider normalization
2. `midend.yaml`: macro expansion + canonicalization + cse-like simplification
3. `downstream_cuda.yaml`: backend legalize + provider/backend metadata attach
4. `downstream_rvv.yaml`: backend legalize + provider/backend metadata attach

Pass trace artifacts:

1. Each pass records `before_stats` / `after_stats` (ops/tensors/outputs/symbols).
2. Pipeline trace records `input_stats` / `output_stats`.
3. Backend drivers propagate MLIR bridge metrics (`input_ir_kind`, `mlir_parse_ms`).

## Current limitations

1. `mlir-opt` integration is optional; python passes are default during migration.
2. The textual dialect is intentionally conservative and oriented for bridge safety.
