# IntentIR Formal Semantics (v1.1+)

This document defines the **denotational semantics** of IntentIR as implemented by:

- `intent_ir/ir/ir_types.py` (types + validation)
- `verify/interpreter.py` (reference interpreter)

It exists to make the project’s verification claims explicit (paper-facing) and to
separate “semantic meaning” from “schedule/performance hints”.

## 1. Core Model

### 1.1 Shapes, indices, and tensors

Let a (rank-`r`) shape be a tuple `S = (s0, …, s{r-1})` of positive integers.

A tensor value `T` of shape `S` is a total function:

`T : Π_{k=0..r-1} [0, s_k) -> V`

where `V` is a dtype-dependent value set (e.g., `float32`, `int32`, `bool`).

### 1.2 Symbols and bindings

IntentIR allows symbolic dimensions (e.g., `"M"`, `"N"`). A **shape binding**
is a mapping `β : Sym -> ℕ+` that assigns concrete sizes to symbols.

Instantiating a tensor type with binding `β` produces a concrete shape by
replacing each `Dim(sym)` with `β(sym)` and each `Dim(const)` with that integer.

### 1.3 Programs

An `IntentFunction` defines:

- input tensors (by name + type)
- a list of ops `op[0..K-1]` producing SSA-like values
- a list of outputs

Program evaluation is deterministic:

`⟦f⟧(inputs) -> outputs`

and is implemented by the interpreter in `verify/interpreter.py`.

## 2. Op Semantics (selected)

All ops are pure (no mutation). Layout is not semantically observable unless an
op explicitly depends on it (currently none do; `layout_cast` is a no-op).

### 2.1 Constants and identity

- `const(value, dtype?) -> O`: `⟦O⟧ = value` broadcast to its declared shape.
- `identity(X) -> O`: `⟦O⟧ = ⟦X⟧`.

### 2.2 Elementwise ops

For elementwise binary ops (e.g., `add/sub/mul/div/max/min/lt/le/...`), semantics
is pointwise over a common index space with broadcasting.

IntentIR prefers **explicit** broadcasting via `broadcast_in_dim` when ranks differ,
but the interpreter also handles common NumPy-style broadcast patterns and uses
declared shape symbols to disambiguate some 1D-vs-2D cases (see comments in
`verify/interpreter.py`).

Unary elementwise ops (e.g., `exp/relu/rsqrt/abs/floor/not`) are pointwise.

### 2.3 Reshape / transpose / broadcast

- `reshape(X) {shape=[...]} -> O`: reinterprets the linearized storage of `X`
  into the new shape; requires total element count to match.
- `transpose(X) {perm=[...]} -> O`: `O[i_perm[k]] = X[i_k]` (permutes axes).
- `broadcast_in_dim(X) {out_shape=[...], broadcast_dims=[...]} -> O`:
  places each input axis `k` onto output axis `broadcast_dims[k]` and broadcasts
  along the remaining axes.

### 2.4 Reduce ops

Let `dims` be the reduction axes set.

- `reduce_sum(X) {dims/axes, keepdims, scale?} -> O`:
  `O = scale * Σ_{j in dims} X[...]` (with `keepdims` controlling rank).
- `reduce_max(X) {dims/axes, keepdims} -> O`: pointwise max over `dims`.
- `reduce_any(X) {dims/axes, keepdims} -> O`: boolean OR over `dims`.

`scale` may be numeric or a derived symbol expression (e.g., `"num_elements"`),
resolved under the current shape bindings.

### 2.5 Matmul

`matmul(A, B) {accum_dtype?, epilogue?} -> C`:

For `A` shape `[M,K]`, `B` shape `[K,N]`:

`C[m,n] = Σ_{k=0..K-1} cast(accum_dtype, A[m,k]) * cast(accum_dtype, B[k,n])`

then cast to output dtype and apply `epilogue` if present (an elementwise graph).

### 2.6 Softmax

`softmax(X) {axis} -> O` uses the numerically-stable form:

`m = reduce_max(X, axis, keepdims=True)`

`e = exp(X - m)`

`s = reduce_sum(e, axis, keepdims=True)`

`O = e / s`

### 2.7 Gather

`gather(data, idx0, idx1, ...) -> O` indexes `data` at integer indices per axis.
Out-of-bounds behavior is **undefined** unless guarded by masks in the source
frontend; in verification, TTIR/TileLang constraints and case generation aim to
exercise boundary conditions and reject invalid indexings.

### 2.8 Macro ops

Macro ops (e.g., `upsample_bicubic2d_aa`) are **semantic** ops: they define a
single high-level meaning that can be lowered into primitive ops by the compiler.

Formally, each macro op `M` has a denotation `⟦M⟧` and a lowering `lower(M)` such that:

`⟦M⟧ == ⟦lower(M)⟧`

The project’s claim is that macro lowering is semantics-preserving.

## 3. Schedule Semantics (non-semantic)

`attach_schedule` / `IntentFunction.schedule` fields (tile sizes, vec width,
pipeline depth, axis bindings, memory hints) are **performance hints** only:

- they do not change `⟦f⟧`
- they may restrict the allowed set of backend implementations (e.g., a backend
  may reject schedules it cannot honor)

## 4. Verification Semantics (what we prove/test)

The pipeline reports multiple layers:

1) **Syntax/shape well-formedness**: `IntentFunction.validate()`.
2) **Static obligations** (frontend constraints, best-effort): TTIR/TileLang-derived
   contracts, plus bounded SMT checks. See `docs/SMT_O3_SOUNDNESS.md`.
3) **Dynamic differential testing**: compare interpreter vs frontend baseline on
   generated cases (random + boundary + metamorphic + bounded exhaustive).

### 4.1 Equality notion (floating point)

For float tensors we use approximate equality:

`allclose(ref, got; atol, rtol)`

Tolerances are inferred from op mix + dtype (see `verify/tolerances.py`), and
additional NaN/Inf probes are performed for selected kernels (`verify/numerical_stability.py`).

### 4.2 Trust model

The current system is **translation validation** (per-kernel evidence), not a
full formal proof:

- Static checks can return `PASS/FAIL/UNKNOWN` (bounded search is incomplete).
- Dynamic checks increase confidence and can produce counterexamples.
- A future “complete core” can integrate an SMT solver (e.g., Z3) for integer
  index/bound reasoning.
