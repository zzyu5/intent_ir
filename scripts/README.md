# `scripts/` Governance (Active Entrypoints Only)

Primary user entrypoint:

- `scripts/intentir.py`
  - `suite`: full/category/smoke runs
  - `kernel`: single-kernel run
  - `env`: dependency/environment probe

Examples:

```bash
python scripts/intentir.py suite --suite flaggems-full196 --progress-style chunk --pipeline-timeout-sec 1800 --run-rvv-remote --rvv-host 192.168.8.72 --rvv-user ubuntu --rvv-use-key
python scripts/intentir.py suite --suite triton-smoke --kernel tanh2d
python scripts/intentir.py kernel --frontend triton --kernel tanh2d
python scripts/intentir.py tilelang export-cuda-snapshots --kernel matmul --refresh
python scripts/intentir.py env
```

This repo uses `scripts/CATALOG.json` as the single source of truth for script ownership and lifecycle.

Validate catalog:

```bash
python scripts/validate_catalog.py
```

## Active Workflow / Lane Entrypoints

- `scripts/flaggems/sync_feature_list_mixed.py`: build mixed-track feature list from registry + task templates.
- `scripts/flaggems/build_workflow_state.py`: refresh `current_status.json` and `session_context.json`.
- `scripts/flaggems/plan_next_batch.py`: pick active batch for a lane.
- `scripts/flaggems/start_session.py`: validate lane context before implementation.
- `scripts/flaggems/end_session.py`: persist progress/handoff and evidence paths.

## Active Coverage Integrity Entrypoints

- `scripts/flaggems/build_coverage_batches.py`: derive fixed 7-family coverage batches from registry.
- `scripts/flaggems/run_coverage_batches.py`: run all category batches (resume-capable) and aggregate to full196 evidence.
- `scripts/flaggems/aggregate_coverage_batches.py`: aggregate family artifacts into one full196 run/status/integrity chain.
- `scripts/triton/flaggems_full_pipeline_verify.py`: provider pipeline verification.
- `scripts/flaggems/run_multibackend_matrix.py`: pipeline + RVV local + CUDA local + converge in one run.
- `scripts/flaggems/converge_status.py`: converge provider/RVV/CUDA into status report.
- `scripts/flaggems/recompute_coverage_integrity.py`: recompute trustable full-coverage integrity from artifacts.
- `scripts/flaggems/coverage_report.py`: emit machine-readable coverage summary.

## Active IR Architecture Entrypoints

- `scripts/intentir/check_primitive_reuse.py`: primitive reuse + provider leakage guard.
- `scripts/intentir/check_macro_composition.py`: macro composition guard.
- `scripts/intentir/report_mapping_complexity.py`: family-level mapping complexity report.
- IR guards support both `--registry` and `--mlir-manifest` sources so migration quality can run on intent-dialect MLIR artifacts directly.
- `scripts/flaggems/run_ir_arch_batch.py`: run ir_arch lane batch.

## Active Backend Compiler Entrypoints

- `scripts/backend_codegen_smoke.py`: RVV local smoke from pipeline artifacts.
- `scripts/cuda_backend_smoke.py`: CUDA local smoke from pipeline artifacts.
- `scripts/rvv_remote_suite.py`: remote RVV suite execution.
- `scripts/rvv_remote_run.py`: single-kernel remote RVV run.
- `scripts/flaggems/run_backend_compiler_batch.py`: backend compiler lane batch runner.
- `scripts/flaggems/compute_stage_timing_breakdown.py`: stage timing breakdown artifact.
- `scripts/flaggems/compute_stage_timing_delta.py`: timing delta report.
- `scripts/flaggems/export_schedule_profiles.py`: schedule profile export.

## Active Gates / Nightly Entrypoints

- `scripts/flaggems/check_batch_gate.py`: lane gate checker.
- `scripts/flaggems/ci_gate.py`: aggregate lane gates.
- `scripts/flaggems/nightly_maintenance.py`: nightly matrix + ci gate + workflow snapshot refresh.
- `scripts/flaggems/install_systemd_nightly.py`: install/render systemd timer templates.
