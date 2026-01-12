"""
CUDA kernel coverage set (source-only).

Each op is a real `.cu` file plus a tiny Python metadata module:
- `<kernel>.cu`: CUDA kernel source (what users should edit/read)
- `<kernel>.py`: IO spec + path to the `.cu` file (used by the pipeline)
"""
