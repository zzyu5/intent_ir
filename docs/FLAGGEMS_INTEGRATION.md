# FlagGems Integration (Triton Provider Path)

FlagGems is integrated as a Triton `provider`, not a separate frontend.

## Coverage Baseline

- Source of truth: `flag_gems.ops.__all__`
- Frozen registry: `pipeline/triton/flaggems_registry.json`
- Registry builder: `pipeline/triton/flaggems_registry.py`
- Generation command:

```bash
PYTHONPATH=. python scripts/flaggems/generate_registry.py
```

Registry states are normalized into exactly five values:

- `dual_pass`
- `rvv_only`
- `cuda_only`
- `blocked_ir`
- `blocked_backend`

## Pipeline Entry

Run Triton full pipeline with FlagGems provider:

```bash
PYTHONPATH=. python scripts/triton/full_pipeline_verify.py \
  --provider flaggems \
  --flaggems-opset deterministic_forward \
  --backend-target rvv \
  --no-use-llm
```

Equivalent unified entry:

```bash
PYTHONPATH=. python scripts/full_pipeline_verify.py \
  --frontend triton \
  --triton-provider flaggems \
  --flaggems-opset deterministic_forward \
  --backend-target cuda_h100
```

## Metadata and Preflight

When `provider=flaggems`:

- Function-level meta includes:
  - `provider=flaggems`
  - `source_op`
  - `capability_state`
  - `backend_target` (if set)
- Op-level `meta` carries the same fields.
- Pipeline report includes:
  - `provider_meta_validation`
  - `backend_preflight`

## Capability and Reporting

- Backend capability API:
  - `backends/capability.py`
  - targets: `rvv`, `cuda_h100`, `cuda_5090d`
- Coverage report command:

```bash
PYTHONPATH=. python scripts/flaggems/coverage_report.py
```

Default output:
- `artifacts/flaggems_coverage/coverage_report.json`

## Multi-Backend Matrix

Run a full local matrix (pipeline + RVV local smoke + CUDA local smoke + status convergence):

```bash
PYTHONPATH=. python scripts/flaggems/run_multibackend_matrix.py \
  --suite smoke \
  --flaggems-opset deterministic_forward \
  --backend-target rvv
```

Converge status using existing result JSONs:

```bash
PYTHONPATH=. python scripts/flaggems/converge_status.py \
  --rvv-json artifacts/flaggems_matrix/rvv_local.json \
  --cuda-json artifacts/flaggems_matrix/cuda_local.json \
  --out artifacts/flaggems_matrix/status_converged.json
```
