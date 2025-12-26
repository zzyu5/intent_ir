# `kernels/tilelang/`

MVP TileLang kernels used to validate multi-frontend pipeline wiring.

For PR#9, the "TileLang source" is a small JSON DSL that explicitly describes:
- semantic anchors (matmul/reduce/etc)
- canonical memory accesses (loads/stores + index_exprs + predicates)

This is intentionally minimal and deterministic; it can later be replaced with a
real TileLang AST parser without changing CertificateV2/obligations/verify.

