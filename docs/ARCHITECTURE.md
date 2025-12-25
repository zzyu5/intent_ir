# Architecture

IntentIR is organized as a pipeline with stable intermediate artifacts:

## Pipeline (Triton today)

1. **Source acquisition** (frontend)
   - Input: a Triton kernel function (DSL source).
   - Output: `*.triton_src.txt` (for debugging / reproducibility).

2. **LLM extraction → IntentIR JSON** (Task1/2)
   - `frontends/triton/llm_intent.py` builds the Triton prompt from kernel DSL.
   - `intent_ir/parser/parser_llm.py` parses/validates the model output into `IntentFunction`.
   - Output: `artifacts/full_pipeline_verify/<kernel>.json` contains `intent` + `intent_expanded`.

3. **Macro expansion (compiler step)** (Task1.5 + Task4.5)
   - If the model emits semantic macro ops (e.g., `upsample_bicubic2d_aa`), the compiler expands them into primitive ops using structured `attrs.impl`.

4. **TTIR compile + dump** (Task3)
   - The Triton frontend launches the kernel once with controlled env vars so TTIR is dumped.
   - Output: `*.ttir` plus other Triton dump artifacts.

5. **TTIR facts/contract/certificate** (Task4)
   - Extract explainable facts + constraints and build a certificate with obligations/witnesses.
   - Output: included in the per-kernel report JSON.

6. **Verification** (Task5)
   - Generate targeted cases (bounded + edge cases).
   - Run: Triton baseline runner vs IntentIR interpreter; compare outputs (and run metamorphic/mutation-kill).
   - Output: diff report + counterexamples in the report JSON.

7. **Backend lowering** (Task6)
   - Lower IntentIR ops to standalone C (via C++ codegen tool) and run on the RVV host.
   - Compare remote results with the baseline runner outputs from step 4.

## Extension Points

To add a new frontend (e.g., CUDA C / TileLang), aim for the same artifacts:

- A stable “kernel source string” for the LLM prompt
- A baseline runner that produces input/output tensors
- Optional: a frontend IR dump (like TTIR) from which you can extract constraints

IntentIR + verification + backend can then be reused with minimal changes.
