"""
Triton frontend: end-to-end status script for Tasks 1–5 on selected real kernels.

This script is intended for debugging and quick visibility of:
- LLM extraction → IntentIR (printed)
- TTIR dump extraction
- Task4 facts/contract
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontends.triton.llm_intent import extract_intent_json
from intent_ir.parser import parse_candidate_json, CandidateIntent
from intent_ir.ir.printer_mlir_like import print_mlir_like
from frontends.triton.facts import extract_facts
from frontends.triton.contract import evaluate_contract
from kernels.triton.tools.ttir_launch import (
    launch_any_kernel_dim,
    launch_group_norm_kernel,
    launch_attention_kernel,
)


def _kernel_src(mod_name: str, fn_name: str) -> str:
    mod = __import__(mod_name, fromlist=[fn_name])
    fn = getattr(mod, fn_name)
    # Handle Triton Autotuner/JITFunction wrappers
    candidate = getattr(fn, "fn", fn)
    if hasattr(candidate, "src"):
        return str(candidate.src)
    return inspect.getsource(candidate)


def _prepare_dump_dir() -> Path:
    dump_dir = Path(os.environ.get("TRITON_DUMP_DIR", "/tmp/triton_dump"))
    dump_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRITON_KERNEL_DUMP", "1")
    os.environ["TRITON_DUMP_DIR"] = str(dump_dir)
    return dump_dir


def _find_latest_ttir(dump_dir: Path, name_hint: str) -> Optional[Path]:
    ttirs = sorted(dump_dir.rglob("*.ttir"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in ttirs:
        if name_hint in p.name:
            return p
    return ttirs[0] if ttirs else None


def _run_one(op_name: str, module: str, fn_name: str, launcher: Callable[[], object]) -> None:
    print(f"\n=== {op_name} ===")
    dump_dir = _prepare_dump_dir()

    cand: Optional[CandidateIntent] = None
    if not os.environ.get("SKIP_LLM"):
        src = _kernel_src(module, fn_name)
        try:
            llm_json = extract_intent_json(
                src,
                kernel_name=fn_name,
                extra_instruction=(
                    "Focus on this jit kernel. Every output tensor must be produced by an op. "
                    "Include mean/rstd or other writes. Model scalars/constants (eps/group_size/etc) "
                    "as const ops or attrs; never leave them implicit."
                ),
                temperature=0,
                max_tokens=2000,
            )
            cand = parse_candidate_json(llm_json)
            print("LLM JSON:\n", json.dumps(cand.raw_json, indent=2))
            print("\nIntent IR (MLIR-like):\n", print_mlir_like(cand.intent))
        except Exception as e:
            print("LLM/parse failed:", e)

    try:
        launcher()
    except Exception as e:
        print("Kernel launch failed:", e)
        return

    ttir_path = _find_latest_ttir(dump_dir, fn_name)
    if not ttir_path:
        print("No TTIR found under", dump_dir)
        return
    ttir_text = ttir_path.read_text()
    facts = extract_facts(ttir_text)
    contract = evaluate_contract(facts)
    print(f"TTIR path: {ttir_path} (lines={facts.raw_summary.get('num_lines')})")
    print("Contract:", contract.level, "reasons:", contract.reasons)
    print("Op counts:", facts.op_counts)
    print("Sample TTIR lines:")
    for line in ttir_text.splitlines()[:8]:
        print("  ", line)


def main() -> None:
    _run_one("ANY (reduce_any)", "kernels.triton.ops.any", "any_kernel_dim", launch_any_kernel_dim)
    _run_one("GroupNorm", "kernels.triton.ops.groupnorm", "group_norm_kernel", launch_group_norm_kernel)
    _run_one("Attention _attn_fwd", "kernels.triton.ops.attention", "_attn_fwd", launch_attention_kernel)


if __name__ == "__main__":
    main()
