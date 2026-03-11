# `tests/core/`

IntentIR semantic-core tests only.

This directory is reserved for:

- IR type/schema invariants
- parser and scalar-provenance behavior
- interpreter semantics
- tolerance and strict-policy helpers

It must stay independent from ORG plugin logic and from backend-specific MLIR lowering tests.
