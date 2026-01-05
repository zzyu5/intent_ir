# SMT O3 (mask ⇒ inbounds) Soundness Notes

This document clarifies the semantics and limitations of the MVP SMT/O3 checker in
`frontends/common/smt_o3.py`.

## What is checked?

For each masked memory access, we try to validate the obligation:

`predicate(mask) ⇒ (0 <= index < upper_bound)`

In the current MVP, `upper_bound` is taken from a predicate clause that directly
looks like `idx < UB` (or equivalent). The checker then focuses on proving (or
refuting) **non-negativity** (`idx >= 0`) under the predicate.

## Status semantics

### `PASS`
- We found a **sound** proof that `idx >= 0` holds when the predicate holds.
- Proof sources:
  - An explicit clause like `idx >= 0`; or
  - A derived lower bound using declared/assumed lower bounds for variables
    (e.g. `pid* >= 0`, `r* >= 0`, `DIM >= 1`).
- The witness includes which clause/proof was used and (when relevant) the
  lower-bound assumptions.

### `FAIL`
- We found a **concrete counterexample model** (variable assignments) such that:
  - all predicate clauses evaluate to `True`, and
  - `idx < 0` (so `idx >= 0` is violated).
- `FAIL` is sound for the returned counterexample.
- The witness includes the bounded-search configuration and stats.

### `UNKNOWN`
`UNKNOWN` means “insufficient evidence/proof”, **not** “probably safe”.

Typical reasons:
- missing an explicit `idx < UB` clause (upper bound not recoverable);
- non-affine predicate clauses (unsupported grammar);
- the lower-bound proof requires stronger reasoning; and
- bounded search did not find a counterexample.

When bounded search is involved, the witness explicitly states that the search is
**bounded/incomplete** and provides:
- per-variable domain summaries (values / min / max / range source),
- `max_models` and how many models were checked,
- whether the search exhausted the bounded domain or was truncated.

## Bounded search domains (incomplete by design)

The checker does **deterministic bounded model search** to find counterexamples.
Domains are constructed from (in priority order):
- `symbol_ranges` (frontend-provided ranges for symbols like `r0`);
- `shape_hints` (kernel “canonical shapes” for dimension symbols like `M/N/K`);
- defaults for `pid*`, `r*`, `arg*`, and unknown symbols.

Even if the search exhausts the constructed finite domain, counterexamples may
still exist outside it. This is the core **soundness/completeness gap** of the
MVP checker.

## Practical guidance

- If you see `UNKNOWN`, treat it as “needs more evidence / stronger checker”.
- If you want `FAIL` to be easier to find, increase:
  - `max_counterexample_search`, and/or
  - provide better `symbol_ranges` / `shape_hints` via the frontend.
- For a future “paper-complete” story, integrate a real SMT solver (e.g. Z3)
  as an optional backend and keep the bounded checker as a fast fallback.

