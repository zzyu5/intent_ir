# `intent_ir/` (IntentIR Core)

This package defines IntentIR and the parser/validator + macro-lowering used by
all frontends. Frontend-specific prompt construction lives under `frontends/`.

## Key modules

- `ir/ir_types.py`: IntentIR dataclasses/types + validation.
- `parser/parser_llm.py`: robust JSON parser/validator for LLM outputs (Task2).
- `llm/llm_client.py` + `llm/llm_extract.py`: LLM client + “LLM response → JSON object” helpers (frontend-agnostic).
- `ir/printer_mlir_like.py`: MLIR-like pretty printer (Task1.5).
- `macros/macro_spec.py`: structured macro attrs (impl-level details).
- `macros/macro_lowering/` + `macros/macro_expand.py`: compiler-style macro lowering into primitive ops.

## Data model

The stable interchange is the IntentIR JSON shape produced by the LLM:

- `tensors`: named tensors with `dtype`, `shape`, `layout`
- `ops`: ordered list of ops (primitive or semantic macro ops)
- `outputs`: list of output tensor names
- `schedule`: optional schedule sketch (tile sizes, axis bindings, memory hints)

The parser enforces schema correctness so later stages can rely on it.
