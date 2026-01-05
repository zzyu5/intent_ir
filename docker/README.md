# Docker (optional)

This repo can be used without Docker. Docker is provided for:

- reproducible CI runs (`pytest -q`)
- a clean environment for remote-RVV tooling (`paramiko`) and backend codegen

## CPU image (tests + remote tooling)

Build:

- `docker build -f docker/Dockerfile.cpu -t intentir:cpu .`

Run unit tests:

- `docker run --rm -it -v "$PWD":/repo -w /repo intentir:cpu pytest -q`

This image does **not** include CUDA/Triton/TileLang.

## GPU image

Not provided in-repo because Torch/Triton/TileLang versions depend on your CUDA stack.
Recommended approach is to start from an NVIDIA base image and install:

- a CUDA-compatible Torch build
- `requirements/gpu.txt`
