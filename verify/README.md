# `verify/` (Verification)

Verification compares:

- a **reference runner** (e.g., Triton launch output)
vs
- an **IntentIR interpreter** (semantic execution)

and uses TTIR-derived constraints/certificates to strengthen case generation.

## Key modules

- `gen_cases.py`: case generation (edge/boundary/bounded-exhaustive).
- `interpreter.py`: execute IntentIR ops over numpy arrays.
- `diff_runner.py`: compare reference vs interpreter outputs with tolerances + shape checks.
- `metamorphic.py`: metamorphic relations (Stage C).
- `mutation.py`: mutation-kill harness (Stage C).

The full pipeline is orchestrated by `pipeline/core.py`.
