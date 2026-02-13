"""
Provider-scoped re-export for FlagGems semantic registry builder.
"""

from pipeline.triton.flaggems_registry import (
    DEFAULT_FLAGGEMS_OPSET,
    DEFAULT_REGISTRY_PATH,
    build_registry,
    infer_flaggems_commit_from_src,
    load_flaggems_all_ops,
    write_registry,
)

__all__ = [
    "DEFAULT_FLAGGEMS_OPSET",
    "DEFAULT_REGISTRY_PATH",
    "build_registry",
    "infer_flaggems_commit_from_src",
    "load_flaggems_all_ops",
    "write_registry",
]
