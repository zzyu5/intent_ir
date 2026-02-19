from __future__ import annotations

from typing import Any, Mapping

from backends.common.mlir_contract import MlirBackendContract
from intent_ir.ir import IntentFunction, TensorType
from intent_ir.mlir.convert_to_intent import to_intent
from intent_ir.mlir.module import IntentMLIRModule


def _dim_to_json(dim: Any) -> Any:
    kind = getattr(dim, "kind", None)
    if kind == "sym":
        return str(getattr(dim, "value"))
    if kind == "const":
        try:
            return int(getattr(dim, "value"))
        except Exception:
            return str(getattr(dim, "value"))
    if isinstance(dim, int):
        return int(dim)
    return str(dim)


def _tensor_io_spec(tensors: Mapping[str, TensorType]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, tensor in dict(tensors).items():
        out[str(name)] = {
            "dtype": str(getattr(tensor, "dtype", "f32")),
            "shape": [_dim_to_json(d) for d in list(getattr(tensor, "shape", []) or [])],
            "layout": str(getattr(getattr(tensor, "layout", None), "kind", "row_major")),
        }
    return out


def _schedule_to_json(schedule: Any) -> dict[str, Any]:
    if schedule is None:
        return {}
    out: dict[str, Any] = {}
    for key in ("tile_m", "tile_n", "tile_k", "vec_width", "pipeline_depth"):
        val = getattr(schedule, key, None)
        if val is not None:
            out[key] = val
    axis_bindings = getattr(schedule, "axis_bindings", None)
    if isinstance(axis_bindings, dict) and axis_bindings:
        out["axis_bindings"] = {str(k): str(v) for k, v in axis_bindings.items()}
    parallel_axes = getattr(schedule, "parallel_axes", None)
    if isinstance(parallel_axes, list) and parallel_axes:
        out["parallel_axes"] = [str(x) for x in parallel_axes]
    return out


def build_cuda_contract_from_intent(
    intent: IntentFunction,
    *,
    source_kind: str,
    source_module: IntentMLIRModule | None = None,
) -> MlirBackendContract:
    op_names = [str(getattr(op, "op", "")) for op in list(intent.ops or []) if str(getattr(op, "op", ""))]
    tensor_shapes: dict[str, list[Any]] = {}
    for name, tensor in dict(intent.tensors or {}).items():
        tensor_shapes[str(name)] = [_dim_to_json(d) for d in list(getattr(tensor, "shape", []) or [])]
    artifacts: dict[str, Any] = {}
    if source_module is not None:
        artifacts["dialect_version"] = str(source_module.dialect_version)
        artifacts["symbols"] = [str(x) for x in list(source_module.symbols or []) if str(x).strip()]
    return MlirBackendContract(
        backend="cuda",
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


def build_cuda_contract(
    module: IntentMLIRModule,
    *,
    source_kind: str = "mlir_module",
) -> MlirBackendContract:
    intent = to_intent(module)
    return build_cuda_contract_from_intent(intent, source_kind=source_kind, source_module=module)


def emit_cuda_contract(module: IntentMLIRModule, **_: object) -> IntentMLIRModule:
    contract = build_cuda_contract(module, source_kind="mlir_module")
    out = IntentMLIRModule(
        module_text=str(module.module_text),
        dialect_version=str(module.dialect_version),
        provenance=dict(module.provenance or {}),
        symbols=list(module.symbols or []),
        meta=dict(module.meta or {}),
        intent_json=(dict(module.intent_json) if isinstance(module.intent_json, dict) else None),
    )
    out.meta["cuda_contract"] = contract.to_json_dict()
    return out


__all__ = [
    "build_cuda_contract",
    "build_cuda_contract_from_intent",
    "emit_cuda_contract",
]
