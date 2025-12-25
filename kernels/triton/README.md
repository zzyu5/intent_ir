# `kernels/triton/`

Real Triton kernel testcases used by the end-to-end pipeline.

Layout:

- `ops/`: kernel modules (copied/minimized from upstream projects)
- `support/`: minimal runtime stubs and small helper utilities

The pipeline loads kernels by module path (see `pipeline/core.py`).

