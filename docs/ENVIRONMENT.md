# Environment & Dependencies

This repo mixes three kinds of dependencies:

## Python (core)

- Python 3.10+ recommended
- Install: `pip install -r requirements/dev.txt`
- Fully pinned install: `pip install -r requirements/lock/dev.lock.txt`
- Quick check: `python scripts/check_env.py`

## Triton pipeline (GPU)

To run `scripts/triton/full_pipeline_verify.py` you need:

- CUDA-capable GPU + working CUDA driver
- `torch` (CUDA build)
- `triton`

The pipeline uses Triton dump env vars:

- `TRITON_KERNEL_DUMP=1`
- `TRITON_DUMP_DIR=...`
- `TRITON_CACHE_DIR=...`

These are set automatically by `pipeline/triton/core.py`.

## Remote RVV execution (Task6)

To run `scripts/rvv_remote_run.py` you need:

- An SSH-accessible RVV host with `gcc` that supports `-march=rv64gcv`
- Python package `paramiko` on the local machine

You can validate the remote toolchain with:

- `python scripts/check_env.py --remote-host <host> --remote-user <user> --use-key`

Secrets are not committed:

- LLM provider keys live in `intent_ir/llm_providers.local.json` (gitignored)
- SSH password can be provided via `INTENTIR_SSH_PASSWORD` (or typed when prompted), or use `--use-key` if you configured SSH keys

## Docker (optional)

- CPU-only dev image: see `docker/README.md`
