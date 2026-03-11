# MLIR Backends

This directory is the explicit dispatcher layer for the two MLIR backend paths:

- `python_stack.py`: Python real-MLIR / legacy LLVM pipeline selection.
- `cpp_plugin_stack.py`: out-of-tree C++ MLIR plugin wave membership and miss policy.
- `router.py`: the single route selector used by Triton, CUDA, and TileLang pipeline entrypoints.

The lowering implementations still live in their native homes:

- Python lowering passes: `intent_ir/mlir/passes/`
- C++ plugin passes: `compiler/intentir_mlir_plugin/`

What moved here is the decision logic, not the lowering semantics.
