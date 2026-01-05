# Requirements

This repo uses a small set of always-needed Python packages, plus optional GPU frontends.

## Files

- `requirements/core.txt`: core runtime deps (NumPy + HTTP + SSH)
- `requirements/dev.txt`: dev/test deps (pytest) + core
- `requirements/gpu.txt`: optional GPU frontend deps (Torch/Triton/TileLang) — versions depend on your CUDA stack

## Install

- Core + tests: `pip install -r requirements/dev.txt`
- Optional GPU pipelines: install CUDA-compatible Torch first, then `pip install -r requirements/gpu.txt`

For a quick sanity check, run: `python scripts/check_env.py`.
