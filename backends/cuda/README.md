# `backends/cuda/` (planned)

CUDA backend for lowering (expanded) IntentIR into runnable GPU code.

Design goals:
- Consume `intent_ir.ir.IntentFunction` (same IR as RVV backend).
- Produce CUDA kernels that can be benchmarked on a local NVIDIA GPU.
- Reuse the existing Torch CUDA extension runner (`frontends/cuda/runtime.py`) for
  compilation + launch in the MVP.

Status: under active development (MVP: AI-Bench8 kernels for paper E5.2-on-CUDA).

