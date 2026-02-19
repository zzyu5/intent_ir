from __future__ import annotations

from typing import Any

from intent_ir.mlir.module import IntentMLIRModule
from .emit_cuda_contract import _dim_to_json, _schedule_to_json, _tensor_io_spec
from intent_ir.mlir.convert_to_intent import to_intent
from intent_ir.ir import IntentFunction
from backends.common.mlir_contract import MlirBackendContract


def build_rvv_contract_from_intent(
    intent: IntentFunction,
    *,
    source_kind: str,
    source_module: IntentMLIRModule | None = None,
) -> MlirBackendContract:
    op_names = [str(getattr(op, "op", "")) for op in list(intent.ops or []) if str(getattr(op, "op", ""))]
    tensor_shapes: dict[str, list[object]] = {}
    for name, tensor in dict(intent.tensors or {}).items():
        tensor_shapes[str(name)] = [_dim_to_json(d) for d in list(getattr(tensor, "shape", []) or [])]
    artifacts: dict[str, Any] = {}
    if source_module is not None:
        artifacts["dialect_version"] = str(source_module.dialect_version)
        artifacts["symbols"] = [str(x) for x in list(source_module.symbols or []) if str(x).strip()]
    contract = MlirBackendContract(
        backend="rvv",
        kernel_name=str(intent.name),
        io_spec={"tensors": _tensor_io_spec(intent.tensors), "outputs": [str(x) for x in list(intent.outputs or [])]},
        launch={},
        schedule=_schedule_to_json(intent.schedule),
        artifacts=artifacts,
        reason_context={"source_kind": str(source_kind), "mlir_backend_contract_used": True},
        op_names=op_names,
        tensor_shapes=tensor_shapes,
        intent_json=intent.to_json_dict(),
    )
    return contract


def build_rvv_contract(
    module: IntentMLIRModule,
    *,
    source_kind: str = "mlir_module",
) -> MlirBackendContract:
    intent = to_intent(module)
    return build_rvv_contract_from_intent(intent, source_kind=source_kind, source_module=module)


def emit_rvv_contract(module: IntentMLIRModule, **_: object) -> IntentMLIRModule:
    contract = build_rvv_contract(module, source_kind="mlir_module")
    out = IntentMLIRModule(
        module_text=str(module.module_text),
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )
    out.meta["rvv_contract"] = contract.to_json_dict()
    return out


__all__ = [
    "build_rvv_contract",
    "build_rvv_contract_from_intent",
    "emit_rvv_contract",
]
