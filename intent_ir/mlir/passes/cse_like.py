from __future__ import annotations

from intent_ir.ir import IntentFunction, Op
from intent_ir.mlir.convert_from_intent import to_mlir
from intent_ir.mlir.convert_to_intent import to_intent
from intent_ir.mlir.module import IntentMLIRModule


def cse_like(module: IntentMLIRModule, **_: object) -> IntentMLIRModule:
    """
    Lightweight CSE-like cleanup for migration stage.

    We only fold exact duplicate pure ops with same op/inputs/attrs where the
    previous result can be reused safely.
    """
    intent = to_intent(module)
    seen: dict[tuple[str, tuple[str, ...], str], str] = {}
    rewritten: list[Op] = []
    replace: dict[str, str] = {}

    def _map_inp(x: str) -> str:
        return str(replace.get(str(x), str(x)))

    for op in list(intent.ops or []):
        inputs = [_map_inp(x) for x in list(op.inputs or [])]
        op2 = Op(op=str(op.op), inputs=inputs, output=str(op.output), attrs=dict(op.attrs or {}), meta=dict(op.meta or {}))
        key = (op2.op, tuple(op2.inputs), repr(sorted(op2.attrs.items(), key=lambda kv: kv[0])))
        if op2.op in {"const", "identity", "cast", "add", "sub", "mul", "div", "max", "min", "relu", "rsqrt", "exp"}:
            prev = seen.get(key)
            if prev is not None:
                replace[op2.output] = str(prev)
                continue
            seen[key] = op2.output
        rewritten.append(op2)

    outputs = [replace.get(str(o), str(o)) for o in list(intent.outputs or [])]
    dedup_intent = IntentFunction(
        name=str(intent.name),
        tensors=dict(intent.tensors),
        ops=rewritten,
        outputs=outputs,
        contract=intent.contract,
        parallel_axes=list(intent.parallel_axes),
        schedule=intent.schedule,
        meta=dict(intent.meta or {}),
        axis_roles=dict(intent.axis_roles or {}),
    )
    out = to_mlir(dedup_intent, provenance=dict(module.provenance or {}))
    out.meta = dict(out.meta or {})
    out.meta["cse_like_applied"] = True
    return out

