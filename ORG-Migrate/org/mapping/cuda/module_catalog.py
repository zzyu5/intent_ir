from __future__ import annotations

from org.backend_plan import BackendModule, BackendModuleEdge
from org.mapping.hardware_model import HardwareModel


PASS_SEQUENCE: list[str] = [
    "goal_to_mechanism_pass",
    "residency_pass",
    "pipeline_pass",
    "communication_pass",
    "layout_pass",
]


def flash_attention2d_catalog(hardware_model: HardwareModel) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    modules = [
        BackendModule(
            id="q_resident_state",
            kind="staging",
            provides=["flash.q_resident_state"],
            params=["ATTN_BLOCK_KV"],
            constraints=["HEAD_DIM == 64"],
        ),
        BackendModule(
            id="kv_tile_stage",
            kind="staging",
            provides=["flash.kv_tile_stage"],
            params=["ATTN_BLOCK_KV"],
            constraints=["ATTN_BLOCK_KV <= KV_CTX"],
        ),
        BackendModule(
            id="kv_shared_stage",
            kind="staging",
            provides=["flash.kv_shared_stage"],
            params=["FLASH_KV_SHARED_STAGE"],
            constraints=["FLASH_KV_SHARED_STAGE in {1}"],
        ),
        BackendModule(
            id="online_softmax_reduce",
            kind="communication",
            provides=["flash.online_softmax_reduce"],
            params=["ATTN_SCORE_WARPS"],
            constraints=["ATTN_SCORE_WARPS in {2,4,6}"],
        ),
        BackendModule(
            id="prefetch_pipeline",
            kind="pipeline",
            provides=["flash.prefetch_pipeline"],
            params=["FLASH_ATTN_ASYNC_COPY"],
            constraints=(["supports_async_copy"] if hardware_model.supports_async_copy else []),
        ),
        BackendModule(
            id="output_accumulator",
            kind="primitive",
            provides=["flash.output_accumulator"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="backend_v6",
            kind="template",
            provides=["backend.kernel_kind.attn2d_causal_softmax_v6"],
            requires=["flash.q_resident_state", "flash.kv_tile_stage", "flash.online_softmax_reduce", "flash.output_accumulator"],
            params=["ATTN_BLOCK_KV", "ATTN_SCORE_WARPS"],
            constraints=["HEAD_DIM == 64"],
        ),
        BackendModule(
            id="backend_v7",
            kind="template",
            provides=["backend.kernel_kind.attn2d_causal_softmax_v7"],
            requires=["flash.q_resident_state", "flash.kv_tile_stage", "flash.output_accumulator"],
            params=["ATTN_BLOCK_KV"],
            constraints=["HEAD_DIM == 64"],
        ),
        BackendModule(
            id="backend_v8",
            kind="template",
            provides=["backend.kernel_kind.attn2d_causal_softmax_v8"],
            requires=["flash.q_resident_state", "flash.kv_tile_stage", "flash.kv_shared_stage", "flash.output_accumulator"],
            params=["ATTN_BLOCK_KV", "FLASH_KV_SHARED_STAGE"],
            constraints=["HEAD_DIM == 64"],
        ),
        BackendModule(
            id="backend_v9",
            kind="template",
            provides=["backend.kernel_kind.attn2d_causal_softmax_v9"],
            requires=["flash.q_resident_state", "flash.kv_tile_stage", "flash.output_accumulator"],
            params=["ATTN_BLOCK_KV"],
            constraints=["HEAD_DIM == 64"],
        ),
    ]
    edges = [
        BackendModuleEdge(src="backend_v6", dst="q_resident_state", edge_type="uses"),
        BackendModuleEdge(src="backend_v6", dst="kv_tile_stage", edge_type="uses"),
        BackendModuleEdge(src="backend_v6", dst="online_softmax_reduce", edge_type="uses"),
        BackendModuleEdge(src="backend_v6", dst="output_accumulator", edge_type="uses"),
        BackendModuleEdge(src="backend_v7", dst="q_resident_state", edge_type="uses"),
        BackendModuleEdge(src="backend_v7", dst="kv_tile_stage", edge_type="uses"),
        BackendModuleEdge(src="backend_v7", dst="output_accumulator", edge_type="uses"),
        BackendModuleEdge(src="backend_v8", dst="q_resident_state", edge_type="uses"),
        BackendModuleEdge(src="backend_v8", dst="kv_tile_stage", edge_type="uses"),
        BackendModuleEdge(src="backend_v8", dst="kv_shared_stage", edge_type="uses"),
        BackendModuleEdge(src="backend_v8", dst="output_accumulator", edge_type="uses"),
        BackendModuleEdge(src="backend_v9", dst="q_resident_state", edge_type="uses"),
        BackendModuleEdge(src="backend_v9", dst="kv_tile_stage", edge_type="uses"),
        BackendModuleEdge(src="backend_v9", dst="output_accumulator", edge_type="uses"),
        BackendModuleEdge(src="backend_v7", dst="prefetch_pipeline", edge_type="optional"),
        BackendModuleEdge(src="backend_v6", dst="prefetch_pipeline", edge_type="optional"),
        BackendModuleEdge(src="backend_v8", dst="prefetch_pipeline", edge_type="optional"),
        BackendModuleEdge(src="backend_v9", dst="prefetch_pipeline", edge_type="optional"),
    ]
    return modules, edges, list(PASS_SEQUENCE)


def attn_fwd_catalog(hardware_model: HardwareModel) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    modules = [
        BackendModule(
            id="qkv_stage",
            kind="staging",
            provides=["attn_fwd.qkv_stage"],
            params=["ATTN_FWD_BLOCK_M", "ATTN_FWD_BLOCK_KV"],
            constraints=["HEAD_DIM == 64"],
        ),
        BackendModule(
            id="online_softmax_reduce",
            kind="communication",
            provides=["attn_fwd.online_softmax_reduce"],
            params=["ATTN_FWD_BLOCK_KV"],
            constraints=["ATTN_FWD_BLOCK_KV <= KV_CTX"],
        ),
        BackendModule(
            id="mask_causal_apply",
            kind="communication",
            provides=["attn_fwd.mask_causal_apply"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="prefetch_pipeline",
            kind="pipeline",
            provides=["attn_fwd.prefetch_pipeline"],
            params=[],
            constraints=(["supports_async_copy"] if hardware_model.supports_async_copy else []),
        ),
        BackendModule(
            id="output_accumulator",
            kind="primitive",
            provides=["attn_fwd.output_accumulator"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="backend_attn_fwd_tiled_v3",
            kind="template",
            provides=["backend.kernel_kind.attn_fwd_tiled_v3"],
            requires=["attn_fwd.qkv_stage", "attn_fwd.online_softmax_reduce", "attn_fwd.mask_causal_apply", "attn_fwd.output_accumulator"],
            params=["ATTN_FWD_BLOCK_M", "ATTN_FWD_BLOCK_KV"],
            constraints=["HEAD_DIM == 64"],
        ),
        BackendModule(
            id="backend_attn_fwd_softmax_v2",
            kind="template",
            provides=["backend.kernel_kind.attn_fwd_softmax_v2"],
            requires=["attn_fwd.online_softmax_reduce", "attn_fwd.mask_causal_apply", "attn_fwd.output_accumulator"],
            params=[],
            constraints=["HEAD_DIM == 64"],
        ),
        BackendModule(
            id="backend_attn_fwd_softmax_v1",
            kind="template",
            provides=["backend.kernel_kind.attn_fwd_softmax_v1"],
            requires=["attn_fwd.mask_causal_apply", "attn_fwd.output_accumulator"],
            params=[],
            constraints=["HEAD_DIM == 64"],
        ),
    ]
    edges = [
        BackendModuleEdge(src="backend_attn_fwd_tiled_v3", dst="qkv_stage", edge_type="uses"),
        BackendModuleEdge(src="backend_attn_fwd_tiled_v3", dst="online_softmax_reduce", edge_type="uses"),
        BackendModuleEdge(src="backend_attn_fwd_tiled_v3", dst="mask_causal_apply", edge_type="uses"),
        BackendModuleEdge(src="backend_attn_fwd_tiled_v3", dst="output_accumulator", edge_type="uses"),
        BackendModuleEdge(src="backend_attn_fwd_tiled_v3", dst="prefetch_pipeline", edge_type="optional"),
        BackendModuleEdge(src="backend_attn_fwd_softmax_v2", dst="online_softmax_reduce", edge_type="uses"),
        BackendModuleEdge(src="backend_attn_fwd_softmax_v2", dst="mask_causal_apply", edge_type="uses"),
        BackendModuleEdge(src="backend_attn_fwd_softmax_v2", dst="output_accumulator", edge_type="uses"),
        BackendModuleEdge(src="backend_attn_fwd_softmax_v1", dst="mask_causal_apply", edge_type="uses"),
        BackendModuleEdge(src="backend_attn_fwd_softmax_v1", dst="output_accumulator", edge_type="uses"),
    ]
    return modules, edges, list(PASS_SEQUENCE)


def row_softmax_catalog(hardware_model: HardwareModel, *, masked: bool) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    prefix = "masked_softmax" if masked else "softmax_inner"
    modules = [
        BackendModule(
            id=f"{prefix}_row_tile_resident",
            kind="staging",
            provides=[f"{prefix}.row_tile_resident"],
            params=["SOFTMAX_BLOCK_THREADS"],
            constraints=[],
        ),
        BackendModule(
            id=f"{prefix}_row_reduction",
            kind="communication",
            provides=[f"{prefix}.row_reduction"],
            params=["SOFTMAX_BLOCK_THREADS"],
            constraints=[],
        ),
        BackendModule(
            id=f"{prefix}_online_safe_math_reduction",
            kind="communication",
            provides=[f"{prefix}.online_safe_math_reduction"],
            params=["SOFTMAX_BLOCK_THREADS"],
            constraints=[],
        ),
        BackendModule(
            id=f"{prefix}_vector_row_path",
            kind="mapping",
            provides=[f"{prefix}.vector_row_path"],
            params=["SOFTMAX_BLOCK_THREADS", "SOFTMAX_VECTOR_WIDTH"],
            constraints=[],
        ),
        BackendModule(
            id=f"{prefix}_full_row_vector_resident",
            kind="primitive",
            provides=[f"{prefix}.full_row_vector_resident"],
            params=["SOFTMAX_FULL_ROW_VECTOR", "SOFTMAX_BLOCK_THREADS", "SOFTMAX_VECTOR_WIDTH"],
            constraints=["SOFTMAX_FULL_ROW_VECTOR in {1}", "SOFTMAX_VECTOR_WIDTH in {4}"],
        ),
        BackendModule(
            id=f"{prefix}_mask_apply",
            kind="communication",
            provides=[f"{prefix}.mask_apply"],
            params=[],
            constraints=[],
        ) if masked else BackendModule(id=f"{prefix}_noop", kind="primitive", provides=[f"{prefix}.noop"], params=[], constraints=[]),
        BackendModule(
            id=f"{prefix}_backend_triton_v1",
            kind="template",
            provides=["backend.kernel_kind.row_softmax_axis1_triton_v1"],
            requires=[f"{prefix}.row_tile_resident", f"{prefix}.row_reduction", f"{prefix}.online_safe_math_reduction", f"{prefix}.vector_row_path"],
            params=["SOFTMAX_BLOCK_THREADS"],
            constraints=[],
        ),
        BackendModule(
            id=f"{prefix}_backend_v1",
            kind="template",
            provides=["backend.kernel_kind.row_softmax_axis1_v1" if not masked else "backend.kernel_kind.row_masked_softmax_axis1_v1"],
            requires=[f"{prefix}.row_reduction", f"{prefix}.online_safe_math_reduction"],
            params=[],
            constraints=[],
        ),
    ]
    if not masked:
        modules.append(
            BackendModule(
                id=f"{prefix}_backend_fullrow_v2",
                kind="template",
                provides=["backend.kernel_kind.row_softmax_axis1_v2"],
                requires=[
                    f"{prefix}.row_tile_resident",
                    f"{prefix}.row_reduction",
                    f"{prefix}.online_safe_math_reduction",
                    f"{prefix}.vector_row_path",
                    f"{prefix}.full_row_vector_resident",
                ],
                params=["SOFTMAX_BLOCK_THREADS", "SOFTMAX_VECTOR_WIDTH", "SOFTMAX_FULL_ROW_VECTOR"],
                constraints=["SOFTMAX_FULL_ROW_VECTOR in {1}", "SOFTMAX_VECTOR_WIDTH in {4}"],
            )
        )
    edges = [
        BackendModuleEdge(src=f"{prefix}_backend_triton_v1", dst=f"{prefix}_row_tile_resident", edge_type="uses"),
        BackendModuleEdge(src=f"{prefix}_backend_triton_v1", dst=f"{prefix}_row_reduction", edge_type="uses"),
        BackendModuleEdge(src=f"{prefix}_backend_triton_v1", dst=f"{prefix}_online_safe_math_reduction", edge_type="uses"),
        BackendModuleEdge(src=f"{prefix}_backend_triton_v1", dst=f"{prefix}_vector_row_path", edge_type="uses"),
        BackendModuleEdge(src=f"{prefix}_backend_v1", dst=f"{prefix}_row_reduction", edge_type="uses"),
        BackendModuleEdge(src=f"{prefix}_backend_v1", dst=f"{prefix}_online_safe_math_reduction", edge_type="uses"),
    ]
    if not masked:
        edges.extend(
            [
                BackendModuleEdge(src=f"{prefix}_backend_fullrow_v2", dst=f"{prefix}_row_tile_resident", edge_type="uses"),
                BackendModuleEdge(src=f"{prefix}_backend_fullrow_v2", dst=f"{prefix}_row_reduction", edge_type="uses"),
                BackendModuleEdge(src=f"{prefix}_backend_fullrow_v2", dst=f"{prefix}_online_safe_math_reduction", edge_type="uses"),
                BackendModuleEdge(src=f"{prefix}_backend_fullrow_v2", dst=f"{prefix}_vector_row_path", edge_type="uses"),
                BackendModuleEdge(src=f"{prefix}_backend_fullrow_v2", dst=f"{prefix}_full_row_vector_resident", edge_type="uses"),
            ]
        )
    if masked:
        edges.append(BackendModuleEdge(src=f"{prefix}_backend_triton_v1", dst=f"{prefix}_mask_apply", edge_type="uses"))
        edges.append(BackendModuleEdge(src=f"{prefix}_backend_v1", dst=f"{prefix}_mask_apply", edge_type="uses"))
    return modules, edges, list(PASS_SEQUENCE)


def ai_bench_softmax_catalog(hardware_model: HardwareModel) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    modules = [
        BackendModule(id="ai_softmax_row_tile_resident", kind="staging", provides=["ai_softmax.row_tile_resident"], params=["SOFTMAX_BLOCK_THREADS"], constraints=[]),
        BackendModule(id="ai_softmax_row_reduction", kind="communication", provides=["ai_softmax.row_reduction"], params=["SOFTMAX_BLOCK_THREADS"], constraints=[]),
        BackendModule(id="ai_softmax_vector_row_path", kind="mapping", provides=["ai_softmax.vector_row_path"], params=["SOFTMAX_BLOCK_THREADS", "SOFTMAX_VEC4"], constraints=[]),
        BackendModule(id="ai_softmax_power2_padding", kind="mapping", provides=["ai_softmax.power2_padding"], params=["SOFTMAX_BLOCK_THREADS"], constraints=[]),
        BackendModule(
            id="ai_softmax_backend_vec4_v2",
            kind="template",
            provides=["backend.kernel_kind.row_softmax_axis1_vec4_v2"],
            requires=["ai_softmax_row_tile_resident", "ai_softmax_row_reduction", "ai_softmax_vector_row_path"],
            params=["SOFTMAX_BLOCK_THREADS", "SOFTMAX_VEC4"],
            constraints=["SOFTMAX_BLOCK_THREADS == 256", "SOFTMAX_VEC4 == 1"],
        ),
        BackendModule(
            id="ai_softmax_backend_v1",
            kind="template",
            provides=["backend.kernel_kind.row_softmax_axis1_v1"],
            requires=["ai_softmax_row_tile_resident", "ai_softmax_row_reduction", "ai_softmax_power2_padding"],
            params=[],
            constraints=[],
        ),
    ]
    edges = [
        BackendModuleEdge(src="ai_softmax_backend_vec4_v2", dst="ai_softmax_row_tile_resident", edge_type="uses"),
        BackendModuleEdge(src="ai_softmax_backend_vec4_v2", dst="ai_softmax_row_reduction", edge_type="uses"),
        BackendModuleEdge(src="ai_softmax_backend_vec4_v2", dst="ai_softmax_vector_row_path", edge_type="uses"),
        BackendModuleEdge(src="ai_softmax_backend_v1", dst="ai_softmax_row_tile_resident", edge_type="uses"),
        BackendModuleEdge(src="ai_softmax_backend_v1", dst="ai_softmax_row_reduction", edge_type="uses"),
        BackendModuleEdge(src="ai_softmax_backend_v1", dst="ai_softmax_power2_padding", edge_type="uses"),
    ]
    return modules, edges, list(PASS_SEQUENCE)


def ai_bench_matmul_catalog(hardware_model: HardwareModel) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    modules = [
        BackendModule(id="ai_matmul_operand_tile_stage", kind="staging", provides=["ai_matmul.operand_tile_stage"], params=["MMA_BM", "MMA_BN", "MMA_BK"], constraints=[]),
        BackendModule(id="ai_matmul_mma_core", kind="primitive", provides=["ai_matmul.mma_core"], params=["MMA_BM", "MMA_BN", "MMA_BK"], constraints=(["supports_mma"] if hardware_model.supports_mma else [])),
        BackendModule(id="ai_matmul_async_prefetch", kind="pipeline", provides=["ai_matmul.async_prefetch"], params=["MMA_ASYNC_COPY"], constraints=(["supports_async_copy"] if hardware_model.supports_async_copy else [])),
        BackendModule(id="ai_matmul_tile_fallback", kind="primitive", provides=["ai_matmul.tile_fallback"], params=[], constraints=[]),
        BackendModule(
            id="ai_matmul_backend_mma_v2",
            kind="template",
            provides=["backend.kernel_kind.matmul_mma_tf32_v2"],
            requires=["ai_matmul_operand_tile_stage", "ai_matmul_mma_core"],
            params=["MMA_BM", "MMA_BN", "MMA_BK", "MMA_ASYNC_COPY"],
            constraints=["MMA_BM%16==0", "MMA_BN%16==0", "MMA_BK%8==0"],
        ),
        BackendModule(
            id="ai_matmul_backend_mma_v1",
            kind="template",
            provides=["backend.kernel_kind.matmul_mma_tf32_v1"],
            requires=["ai_matmul_operand_tile_stage", "ai_matmul_mma_core"],
            params=["MMA_BM", "MMA_BN", "MMA_BK"],
            constraints=["MMA_BM%16==0", "MMA_BN%16==0", "MMA_BK%8==0"],
        ),
        BackendModule(
            id="ai_matmul_backend_global_v1",
            kind="template",
            provides=["backend.kernel_kind.matmul_mma_tf32_global_v1"],
            requires=["ai_matmul_operand_tile_stage", "ai_matmul_mma_core"],
            params=["MMA_BM", "MMA_BN", "MMA_BK"],
            constraints=["MMA_BM%16==0", "MMA_BN%16==0", "MMA_BK%8==0"],
        ),
        BackendModule(
            id="ai_matmul_backend_tile_v2",
            kind="template",
            provides=["backend.kernel_kind.matmul_tile_v2"],
            requires=["ai_matmul_tile_fallback"],
            params=[],
            constraints=[],
        ),
    ]
    edges = [
        BackendModuleEdge(src="ai_matmul_backend_mma_v2", dst="ai_matmul_operand_tile_stage", edge_type="uses"),
        BackendModuleEdge(src="ai_matmul_backend_mma_v2", dst="ai_matmul_mma_core", edge_type="uses"),
        BackendModuleEdge(src="ai_matmul_backend_mma_v2", dst="ai_matmul_async_prefetch", edge_type="optional"),
        BackendModuleEdge(src="ai_matmul_backend_mma_v1", dst="ai_matmul_operand_tile_stage", edge_type="uses"),
        BackendModuleEdge(src="ai_matmul_backend_mma_v1", dst="ai_matmul_mma_core", edge_type="uses"),
        BackendModuleEdge(src="ai_matmul_backend_global_v1", dst="ai_matmul_operand_tile_stage", edge_type="uses"),
        BackendModuleEdge(src="ai_matmul_backend_global_v1", dst="ai_matmul_mma_core", edge_type="uses"),
        BackendModuleEdge(src="ai_matmul_backend_tile_v2", dst="ai_matmul_tile_fallback", edge_type="uses"),
    ]
    return modules, edges, list(PASS_SEQUENCE)


def masked_attention2d_catalog(hardware_model: HardwareModel) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    modules = [
        BackendModule(id="masked_attn_q_resident_state", kind="staging", provides=["masked_attn.q_resident_state"], params=[], constraints=[]),
        BackendModule(
            id="masked_attn_tiny_kv_stage",
            kind="staging",
            provides=["masked_attn.tiny_kv_stage"],
            params=["ATTN_SCORE_WARPS", "MASKED_ATTN_SHARED_STAGE", "MASKED_ATTN_VECTOR_WIDTH"],
            constraints=[],
        ),
        BackendModule(id="masked_attn_mask_causal_apply", kind="communication", provides=["masked_attn.mask_causal_apply"], params=[], constraints=[]),
        BackendModule(id="masked_attn_parallel_softmax", kind="communication", provides=["masked_attn.parallel_softmax"], params=["ATTN_SCORE_WARPS"], constraints=[]),
        BackendModule(id="masked_attn_vector_dot_fragment", kind="mapping", provides=["masked_attn.vector_dot_fragment"], params=["ATTN_SCORE_WARPS"], constraints=[]),
        BackendModule(
            id="masked_attn_backend_v18",
            kind="template",
            provides=["backend.kernel_kind.attn2d_causal_softmax_v18"],
            requires=["masked_attn_q_resident_state", "masked_attn_tiny_kv_stage", "masked_attn_mask_causal_apply", "masked_attn_parallel_softmax"],
            params=[],
            constraints=["Q_CTX==16", "KV_CTX==16", "HEAD_DIM==16"],
        ),
        BackendModule(
            id="masked_attn_backend_v14",
            kind="template",
            provides=["backend.kernel_kind.attn2d_causal_softmax_v14"],
            requires=["masked_attn_q_resident_state", "masked_attn_tiny_kv_stage", "masked_attn_mask_causal_apply", "masked_attn_parallel_softmax"],
            params=[],
            constraints=["HEAD_DIM==16"],
        ),
        BackendModule(
            id="masked_attn_backend_v10",
            kind="template",
            provides=["backend.kernel_kind.attn2d_causal_softmax_v10"],
            requires=["masked_attn_q_resident_state", "masked_attn_tiny_kv_stage", "masked_attn_mask_causal_apply", "masked_attn_vector_dot_fragment"],
            params=["ATTN_SCORE_WARPS"],
            constraints=["HEAD_DIM==16"],
        ),
    ]
    edges = [
        BackendModuleEdge(src="masked_attn_backend_v18", dst="masked_attn_q_resident_state", edge_type="uses"),
        BackendModuleEdge(src="masked_attn_backend_v18", dst="masked_attn_tiny_kv_stage", edge_type="uses"),
        BackendModuleEdge(src="masked_attn_backend_v18", dst="masked_attn_mask_causal_apply", edge_type="uses"),
        BackendModuleEdge(src="masked_attn_backend_v18", dst="masked_attn_parallel_softmax", edge_type="uses"),
        BackendModuleEdge(src="masked_attn_backend_v14", dst="masked_attn_q_resident_state", edge_type="uses"),
        BackendModuleEdge(src="masked_attn_backend_v14", dst="masked_attn_tiny_kv_stage", edge_type="uses"),
        BackendModuleEdge(src="masked_attn_backend_v14", dst="masked_attn_mask_causal_apply", edge_type="uses"),
        BackendModuleEdge(src="masked_attn_backend_v14", dst="masked_attn_parallel_softmax", edge_type="uses"),
        BackendModuleEdge(src="masked_attn_backend_v10", dst="masked_attn_q_resident_state", edge_type="uses"),
        BackendModuleEdge(src="masked_attn_backend_v10", dst="masked_attn_tiny_kv_stage", edge_type="uses"),
        BackendModuleEdge(src="masked_attn_backend_v10", dst="masked_attn_mask_causal_apply", edge_type="uses"),
        BackendModuleEdge(src="masked_attn_backend_v10", dst="masked_attn_vector_dot_fragment", edge_type="uses"),
    ]
    return modules, edges, list(PASS_SEQUENCE)


def row_reduction_catalog(
    hardware_model: HardwareModel,
    *,
    reduction_kind: str,
) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    prefix = f"row_{reduction_kind}"
    kernel_kind = f"row_{reduction_kind}_axis1_v2"
    modules = [
        BackendModule(
            id=f"{prefix}_row_tile_resident",
            kind="staging",
            provides=[f"{prefix}.row_tile_resident"],
            params=["ROW_REDUCE_BLOCK_THREADS", "ROW_REDUCE_VECTOR_WIDTH"],
            constraints=[],
        ),
        BackendModule(
            id=f"{prefix}_vector_row_load",
            kind="mapping",
            provides=[f"{prefix}.vector_row_load"],
            params=["ROW_REDUCE_VECTOR_WIDTH"],
            constraints=["ROW_REDUCE_VECTOR_WIDTH in {1,2,4}"],
        ),
        BackendModule(
            id=f"{prefix}_warp_reduction_tree",
            kind="communication",
            provides=[f"{prefix}.warp_reduction_tree"],
            params=["ROW_REDUCE_BLOCK_THREADS"],
            constraints=["ROW_REDUCE_BLOCK_THREADS in {32,64,128,256}"],
        ),
        BackendModule(
            id=f"{prefix}_shared_warp_exchange",
            kind="staging",
            provides=[f"{prefix}.shared_warp_exchange"],
            params=["ROW_REDUCE_SHARED_STAGE"],
            constraints=(["shared_mem_kb >= 32"] if int(hardware_model.shared_mem_kb or 0) >= 32 else []),
        ),
        BackendModule(
            id=f"{prefix}_writeback",
            kind="primitive",
            provides=[f"{prefix}.writeback"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id=f"{prefix}_backend_v2",
            kind="template",
            provides=[f"backend.kernel_kind.{kernel_kind}"],
            requires=[
                f"{prefix}.row_tile_resident",
                f"{prefix}.warp_reduction_tree",
                f"{prefix}.writeback",
            ],
            params=["ROW_REDUCE_BLOCK_THREADS", "ROW_REDUCE_VECTOR_WIDTH", "ROW_REDUCE_SHARED_STAGE"],
            constraints=[],
        ),
    ]
    edges = [
        BackendModuleEdge(src=f"{prefix}_backend_v2", dst=f"{prefix}_row_tile_resident", edge_type="uses"),
        BackendModuleEdge(src=f"{prefix}_backend_v2", dst=f"{prefix}_warp_reduction_tree", edge_type="uses"),
        BackendModuleEdge(src=f"{prefix}_backend_v2", dst=f"{prefix}_writeback", edge_type="uses"),
        BackendModuleEdge(src=f"{prefix}_backend_v2", dst=f"{prefix}_vector_row_load", edge_type="optional"),
        BackendModuleEdge(src=f"{prefix}_backend_v2", dst=f"{prefix}_shared_warp_exchange", edge_type="optional"),
    ]
    return modules, edges, list(PASS_SEQUENCE)


def elementwise2d_catalog(
    hardware_model: HardwareModel,
    *,
    op_kind: str,
) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    prefix = f"elementwise_{str(op_kind).strip()}"
    primitive_tag = f"{prefix}_primitive"
    modules = [
        BackendModule(
            id=f"{prefix}_tile_resident",
            kind="staging",
            provides=[f"{prefix}.tile_resident"],
            params=["ELEMENTWISE_BLOCK_THREADS", "ELEMENTWISE_VECTOR_WIDTH"],
            constraints=[],
        ),
        BackendModule(
            id=f"{prefix}_vector_global_io",
            kind="mapping",
            provides=[f"{prefix}.vector_global_io"],
            params=["ELEMENTWISE_VECTOR_WIDTH"],
            constraints=["ELEMENTWISE_VECTOR_WIDTH in {1,2,4}"],
        ),
        BackendModule(
            id=f"{prefix}_two_axis_grid_mapping",
            kind="mapping",
            provides=[f"{prefix}.two_axis_grid_mapping"],
            params=["ELEMENTWISE_BLOCK_THREADS"],
            constraints=["ELEMENTWISE_BLOCK_THREADS in {64,128,256,512}"],
        ),
        BackendModule(
            id=f"{prefix}_masked_edge_handling",
            kind="communication",
            provides=[f"{prefix}.masked_edge_handling"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id=primitive_tag,
            kind="primitive",
            provides=[f"{prefix}.{primitive_tag}"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id=f"{prefix}_backend_v1",
            kind="template",
            provides=["backend.kernel_kind.elementwise_v1"],
            requires=[
                f"{prefix}.tile_resident",
                f"{prefix}.two_axis_grid_mapping",
                f"{prefix}.{primitive_tag}",
            ],
            params=["ELEMENTWISE_BLOCK_THREADS", "ELEMENTWISE_VECTOR_WIDTH"],
            constraints=[],
        ),
    ]
    edges = [
        BackendModuleEdge(src=f"{prefix}_backend_v1", dst=f"{prefix}_tile_resident", edge_type="uses"),
        BackendModuleEdge(src=f"{prefix}_backend_v1", dst=f"{prefix}_two_axis_grid_mapping", edge_type="uses"),
        BackendModuleEdge(src=f"{prefix}_backend_v1", dst=primitive_tag, edge_type="uses"),
        BackendModuleEdge(src=f"{prefix}_backend_v1", dst=f"{prefix}_vector_global_io", edge_type="optional"),
        BackendModuleEdge(src=f"{prefix}_backend_v1", dst=f"{prefix}_masked_edge_handling", edge_type="optional"),
    ]
    return modules, edges, list(PASS_SEQUENCE)


def layer_norm_persistent_catalog(hardware_model: HardwareModel) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    modules = [
        BackendModule(
            id="layer_norm_row_tile_resident",
            kind="staging",
            provides=["layer_norm.row_tile_resident"],
            params=["LAYER_NORM_BLOCK_THREADS", "LAYER_NORM_VECTOR_WIDTH", "LAYER_NORM_PERSISTENT_ROW"],
            constraints=[],
        ),
        BackendModule(
            id="layer_norm_warp_statistics",
            kind="communication",
            provides=["layer_norm.warp_statistics"],
            params=["LAYER_NORM_BLOCK_THREADS"],
            constraints=["LAYER_NORM_BLOCK_THREADS in {32,64,128,256}"],
        ),
        BackendModule(
            id="layer_norm_multi_output_stats_resident",
            kind="communication",
            provides=["layer_norm.multi_output_stats_resident"],
            params=["LAYER_NORM_BLOCK_THREADS"],
            constraints=["LAYER_NORM_BLOCK_THREADS in {32,64,128,256}"],
        ),
        BackendModule(
            id="layer_norm_register_stage",
            kind="staging",
            provides=["layer_norm.register_stage"],
            params=["LAYER_NORM_VECTOR_WIDTH"],
            constraints=["LAYER_NORM_VECTOR_WIDTH in {1,2,4}"],
        ),
        BackendModule(
            id="layer_norm_full_row_vector_resident",
            kind="primitive",
            provides=["layer_norm.full_row_vector_resident"],
            params=["LAYER_NORM_FULL_ROW_VECTOR", "LAYER_NORM_BLOCK_THREADS", "LAYER_NORM_VECTOR_WIDTH"],
            constraints=["LAYER_NORM_FULL_ROW_VECTOR in {1}", "LAYER_NORM_VECTOR_WIDTH in {4}"],
        ),
        BackendModule(
            id="layer_norm_persistent_row_cache",
            kind="staging",
            provides=["layer_norm.persistent_row_cache"],
            params=["LAYER_NORM_PERSISTENT_ROW"],
            constraints=(["shared_mem_kb >= 64"] if int(hardware_model.shared_mem_kb or 0) >= 64 else []),
        ),
        BackendModule(
            id="layer_norm_affine_epilogue",
            kind="fusion",
            provides=["layer_norm.affine_epilogue"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="layer_norm_backend_v1",
            kind="template",
            provides=["backend.kernel_kind.layer_norm_axis1_v1"],
            requires=[
                "layer_norm.row_tile_resident",
                "layer_norm.warp_statistics",
                "layer_norm.multi_output_stats_resident",
                "layer_norm.affine_epilogue",
            ],
            params=["LAYER_NORM_BLOCK_THREADS", "LAYER_NORM_VECTOR_WIDTH", "LAYER_NORM_PERSISTENT_ROW"],
            constraints=[],
        ),
        BackendModule(
            id="layer_norm_backend_v2",
            kind="template",
            provides=["backend.kernel_kind.layer_norm_axis1_v2"],
            requires=[
                "layer_norm.row_tile_resident",
                "layer_norm.warp_statistics",
                "layer_norm.multi_output_stats_resident",
                "layer_norm.full_row_vector_resident",
                "layer_norm.affine_epilogue",
            ],
            params=["LAYER_NORM_BLOCK_THREADS", "LAYER_NORM_VECTOR_WIDTH", "LAYER_NORM_FULL_ROW_VECTOR"],
            constraints=["LAYER_NORM_FULL_ROW_VECTOR in {1}", "LAYER_NORM_VECTOR_WIDTH in {4}"],
        ),
    ]
    edges = [
        BackendModuleEdge(src="layer_norm_backend_v1", dst="layer_norm_row_tile_resident", edge_type="uses"),
        BackendModuleEdge(src="layer_norm_backend_v1", dst="layer_norm_warp_statistics", edge_type="uses"),
        BackendModuleEdge(src="layer_norm_backend_v1", dst="layer_norm_multi_output_stats_resident", edge_type="uses"),
        BackendModuleEdge(src="layer_norm_backend_v1", dst="layer_norm_affine_epilogue", edge_type="uses"),
        BackendModuleEdge(src="layer_norm_backend_v1", dst="layer_norm_register_stage", edge_type="optional"),
        BackendModuleEdge(src="layer_norm_backend_v1", dst="layer_norm_persistent_row_cache", edge_type="optional"),
        BackendModuleEdge(src="layer_norm_backend_v2", dst="layer_norm_row_tile_resident", edge_type="uses"),
        BackendModuleEdge(src="layer_norm_backend_v2", dst="layer_norm_warp_statistics", edge_type="uses"),
        BackendModuleEdge(src="layer_norm_backend_v2", dst="layer_norm_multi_output_stats_resident", edge_type="uses"),
        BackendModuleEdge(src="layer_norm_backend_v2", dst="layer_norm_full_row_vector_resident", edge_type="uses"),
        BackendModuleEdge(src="layer_norm_backend_v2", dst="layer_norm_affine_epilogue", edge_type="uses"),
        BackendModuleEdge(src="layer_norm_backend_v2", dst="layer_norm_register_stage", edge_type="optional"),
    ]
    return modules, edges, list(PASS_SEQUENCE)


def rms_norm2d_catalog(hardware_model: HardwareModel) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    _ = hardware_model
    modules = [
        BackendModule(
            id="rms_norm_row_tile_resident",
            kind="staging",
            provides=["rms_norm.row_tile_resident"],
            params=["RMS_NORM_BLOCK_THREADS", "RMS_NORM_VECTOR_WIDTH"],
            constraints=[],
        ),
        BackendModule(
            id="rms_norm_warp_statistics",
            kind="communication",
            provides=["rms_norm.warp_statistics"],
            params=["RMS_NORM_BLOCK_THREADS"],
            constraints=[],
        ),
        BackendModule(
            id="rms_norm_cta_statistics",
            kind="communication",
            provides=["rms_norm.cta_statistics"],
            params=["RMS_NORM_BLOCK_THREADS"],
            constraints=[],
        ),
        BackendModule(
            id="rms_norm_vector_row_io",
            kind="primitive",
            provides=["rms_norm.vector_row_io"],
            params=["RMS_NORM_VECTOR_WIDTH"],
            constraints=[],
        ),
        BackendModule(
            id="rms_norm_full_row_vector_resident",
            kind="primitive",
            provides=["rms_norm.full_row_vector_resident"],
            params=["RMS_NORM_FULL_ROW_VECTOR", "RMS_NORM_BLOCK_THREADS", "RMS_NORM_VECTOR_WIDTH"],
            constraints=["RMS_NORM_FULL_ROW_VECTOR in {1}"],
        ),
        BackendModule(
            id="rms_norm_affine_epilogue",
            kind="fusion",
            provides=["rms_norm.affine_epilogue"],
            params=["RMS_NORM_VECTOR_WIDTH"],
            constraints=[],
        ),
        BackendModule(
            id="rms_norm_backend_v2",
            kind="template",
            provides=["backend.kernel_kind.rms_norm_axis1_v2"],
            requires=[
                "rms_norm.row_tile_resident",
                "rms_norm.cta_statistics",
                "rms_norm.affine_epilogue",
            ],
            params=["RMS_NORM_BLOCK_THREADS", "RMS_NORM_VECTOR_WIDTH"],
            constraints=[],
        ),
        BackendModule(
            id="rms_norm_backend_v3",
            kind="template",
            provides=["backend.kernel_kind.rms_norm_axis1_v3"],
            requires=[
                "rms_norm.row_tile_resident",
                "rms_norm.warp_statistics",
                "rms_norm.affine_epilogue",
            ],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="rms_norm_backend_v4",
            kind="template",
            provides=["backend.kernel_kind.rms_norm_axis1_v4"],
            requires=[
                "rms_norm.row_tile_resident",
                "rms_norm.cta_statistics",
                "rms_norm.full_row_vector_resident",
                "rms_norm.affine_epilogue",
            ],
            params=["RMS_NORM_FULL_ROW_VECTOR", "RMS_NORM_BLOCK_THREADS", "RMS_NORM_VECTOR_WIDTH"],
            constraints=["RMS_NORM_FULL_ROW_VECTOR in {1}"],
        ),
    ]
    edges = [
        BackendModuleEdge(src="rms_norm_backend_v2", dst="rms_norm_row_tile_resident", edge_type="uses"),
        BackendModuleEdge(src="rms_norm_backend_v2", dst="rms_norm_cta_statistics", edge_type="uses"),
        BackendModuleEdge(src="rms_norm_backend_v2", dst="rms_norm_affine_epilogue", edge_type="uses"),
        BackendModuleEdge(src="rms_norm_backend_v2", dst="rms_norm_vector_row_io", edge_type="optional"),
        BackendModuleEdge(src="rms_norm_backend_v3", dst="rms_norm_row_tile_resident", edge_type="uses"),
        BackendModuleEdge(src="rms_norm_backend_v3", dst="rms_norm_warp_statistics", edge_type="uses"),
        BackendModuleEdge(src="rms_norm_backend_v3", dst="rms_norm_affine_epilogue", edge_type="uses"),
        BackendModuleEdge(src="rms_norm_backend_v4", dst="rms_norm_row_tile_resident", edge_type="uses"),
        BackendModuleEdge(src="rms_norm_backend_v4", dst="rms_norm_cta_statistics", edge_type="uses"),
        BackendModuleEdge(src="rms_norm_backend_v4", dst="rms_norm_full_row_vector_resident", edge_type="uses"),
        BackendModuleEdge(src="rms_norm_backend_v4", dst="rms_norm_affine_epilogue", edge_type="uses"),
        BackendModuleEdge(src="rms_norm_backend_v4", dst="rms_norm_vector_row_io", edge_type="optional"),
    ]
    return modules, edges, list(PASS_SEQUENCE)


def group_norm_kernel_catalog(hardware_model: HardwareModel) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    modules = [
        BackendModule(
            id="group_norm_group_tile_resident",
            kind="staging",
            provides=["group_norm.group_tile_resident"],
            params=["GROUP_NORM_BLOCK_THREADS", "GROUP_NORM_VECTOR_WIDTH"],
            constraints=[],
        ),
        BackendModule(
            id="group_norm_warp_reduction",
            kind="communication",
            provides=["group_norm.warp_reduction"],
            params=["GROUP_NORM_BLOCK_THREADS"],
            constraints=["GROUP_NORM_BLOCK_THREADS in {64,128,256}"],
        ),
        BackendModule(
            id="group_norm_online_normalization",
            kind="fusion",
            provides=["group_norm.online_normalization"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="group_norm_affine_fused_epilogue",
            kind="fusion",
            provides=["group_norm.affine_fused_epilogue"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="group_norm_vector_group_io",
            kind="mapping",
            provides=["group_norm.vector_group_io"],
            params=["GROUP_NORM_VECTOR_WIDTH"],
            constraints=["GROUP_NORM_VECTOR_WIDTH in {1,2,4}"],
        ),
        BackendModule(
            id="group_norm_backend_v1",
            kind="template",
            provides=["backend.kernel_kind.group_norm_v1"],
            requires=[
                "group_norm.group_tile_resident",
                "group_norm.warp_reduction",
                "group_norm.online_normalization",
                "group_norm.affine_fused_epilogue",
            ],
            params=["GROUP_NORM_BLOCK_THREADS", "GROUP_NORM_VECTOR_WIDTH"],
            constraints=[],
        ),
    ]
    edges = [
        BackendModuleEdge(src="group_norm_backend_v1", dst="group_norm_group_tile_resident", edge_type="uses"),
        BackendModuleEdge(src="group_norm_backend_v1", dst="group_norm_warp_reduction", edge_type="uses"),
        BackendModuleEdge(src="group_norm_backend_v1", dst="group_norm_online_normalization", edge_type="uses"),
        BackendModuleEdge(src="group_norm_backend_v1", dst="group_norm_affine_fused_epilogue", edge_type="uses"),
        BackendModuleEdge(src="group_norm_backend_v1", dst="group_norm_vector_group_io", edge_type="optional"),
    ]
    return modules, edges, list(PASS_SEQUENCE)


def matmul_fused_epilogue2d_catalog(hardware_model: HardwareModel) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    modules = [
        BackendModule(
            id="ab_tile_stage",
            kind="staging",
            provides=["matmul.ab_tile_stage"],
            params=["MMA_BM", "MMA_BN", "MMA_BK"],
            constraints=["MMA_BM%16==0", "MMA_BN%16==0", "MMA_BK%8==0"],
        ),
        BackendModule(
            id="mma_core",
            kind="primitive",
            provides=["matmul.mma_core"],
            params=["MMA_BM", "MMA_BN", "MMA_BK"],
            constraints=(["supports_mma"] if hardware_model.supports_mma else []),
        ),
        BackendModule(
            id="epilogue_fused_writeback",
            kind="fusion",
            provides=["matmul.epilogue_fused_writeback"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="prefetch_pipeline",
            kind="pipeline",
            provides=["matmul.prefetch_pipeline"],
            params=["MMA_ASYNC_COPY"],
            constraints=(["supports_async_copy"] if hardware_model.supports_async_copy else []),
        ),
        BackendModule(
            id="backend_mma_v1",
            kind="template",
            provides=["backend.kernel_kind.matmul_mma_tf32_v1"],
            requires=["matmul.ab_tile_stage", "matmul.mma_core", "matmul.epilogue_fused_writeback"],
            params=["MMA_BM", "MMA_BN", "MMA_BK", "MMA_ASYNC_COPY"],
            constraints=[],
        ),
        BackendModule(
            id="backend_tile_v2",
            kind="template",
            provides=["backend.kernel_kind.matmul_tile_v2"],
            requires=["matmul.ab_tile_stage", "matmul.epilogue_fused_writeback"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="backend_tile_v1",
            kind="template",
            provides=["backend.kernel_kind.matmul_tile_v1"],
            requires=["matmul.ab_tile_stage", "matmul.epilogue_fused_writeback"],
            params=[],
            constraints=[],
        ),
    ]
    edges = [
        BackendModuleEdge(src="backend_mma_v1", dst="ab_tile_stage", edge_type="uses"),
        BackendModuleEdge(src="backend_mma_v1", dst="mma_core", edge_type="uses"),
        BackendModuleEdge(src="backend_mma_v1", dst="epilogue_fused_writeback", edge_type="uses"),
        BackendModuleEdge(src="backend_mma_v1", dst="prefetch_pipeline", edge_type="optional"),
        BackendModuleEdge(src="backend_tile_v2", dst="ab_tile_stage", edge_type="uses"),
        BackendModuleEdge(src="backend_tile_v2", dst="epilogue_fused_writeback", edge_type="uses"),
        BackendModuleEdge(src="backend_tile_v1", dst="ab_tile_stage", edge_type="uses"),
        BackendModuleEdge(src="backend_tile_v1", dst="epilogue_fused_writeback", edge_type="uses"),
    ]
    return modules, edges, list(PASS_SEQUENCE)


def rope_view_catalog(hardware_model: HardwareModel) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    del hardware_model
    modules = [
        BackendModule(
            id="rope_logical_view",
            kind="layout",
            provides=["rope.logical_view"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="rope_rotation",
            kind="primitive",
            provides=["rope.rotation"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="rope_backend_v1",
            kind="template",
            provides=["backend.kernel_kind.rope_dual_v1"],
            requires=["rope.logical_view", "rope.rotation"],
            params=[],
            constraints=[],
        ),
    ]
    edges = [
        BackendModuleEdge(src="rope_backend_v1", dst="rope_logical_view", edge_type="uses"),
        BackendModuleEdge(src="rope_backend_v1", dst="rope_rotation", edge_type="uses"),
    ]
    return modules, edges, list(PASS_SEQUENCE)


def cfg_masked_row_reduce_catalog(hardware_model: HardwareModel) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    del hardware_model
    modules = [
        BackendModule(
            id="cfg_masked_row_tile_resident",
            kind="staging",
            provides=["cfg_masked_row.row_tile_resident"],
            params=["CFG_ROW_BLOCK_THREADS", "CFG_ROW_VECTOR_WIDTH"],
            constraints=[],
        ),
        BackendModule(
            id="cfg_masked_row_reduction",
            kind="communication",
            provides=["cfg_masked_row.row_reduction"],
            params=["CFG_ROW_BLOCK_THREADS"],
            constraints=[],
        ),
        BackendModule(
            id="cfg_masked_vector_io",
            kind="mapping",
            provides=["cfg_masked_row.masked_vector_io"],
            params=["CFG_ROW_VECTOR_WIDTH"],
            constraints=["CFG_ROW_VECTOR_WIDTH in {1,2,4}"],
        ),
        BackendModule(
            id="cfg_masked_label_gather",
            kind="primitive",
            provides=["cfg_masked_row.label_gather"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="cfg_masked_branch_predicate",
            kind="communication",
            provides=["cfg_masked_row.branch_predicate"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="cfg_masked_register_residency",
            kind="staging",
            provides=["cfg_masked_row.register_residency"],
            params=["CFG_ROW_BLOCK_THREADS", "CFG_ROW_VECTOR_WIDTH"],
            constraints=[],
        ),
        BackendModule(
            id="cfg_masked_atomic_finalize",
            kind="fusion",
            provides=["cfg_masked_row.atomic_finalize"],
            params=[],
            constraints=[],
        ),
        BackendModule(
            id="cfg_masked_row_backend_v1",
            kind="template",
            provides=["backend.kernel_kind.cfg_masked_row_reduce_v1"],
            requires=[
                "cfg_masked_row.row_reduction",
                "cfg_masked_row.label_gather",
                "cfg_masked_row.branch_predicate",
                "cfg_masked_row.atomic_finalize",
            ],
            params=["CFG_ROW_BLOCK_THREADS", "CFG_ROW_VECTOR_WIDTH"],
            constraints=[],
        ),
    ]
    edges = [
        BackendModuleEdge(src="cfg_masked_row_backend_v1", dst="cfg_masked_row_tile_resident", edge_type="optional"),
        BackendModuleEdge(src="cfg_masked_row_backend_v1", dst="cfg_masked_vector_io", edge_type="optional"),
        BackendModuleEdge(src="cfg_masked_row_backend_v1", dst="cfg_masked_register_residency", edge_type="optional"),
        BackendModuleEdge(src="cfg_masked_row_backend_v1", dst="cfg_masked_row_reduction", edge_type="uses"),
        BackendModuleEdge(src="cfg_masked_row_backend_v1", dst="cfg_masked_label_gather", edge_type="uses"),
        BackendModuleEdge(src="cfg_masked_row_backend_v1", dst="cfg_masked_branch_predicate", edge_type="uses"),
        BackendModuleEdge(src="cfg_masked_row_backend_v1", dst="cfg_masked_atomic_finalize", edge_type="uses"),
    ]
    return modules, edges, list(PASS_SEQUENCE)


def cross_entropy_loss_catalog(hardware_model: HardwareModel) -> tuple[list[BackendModule], list[BackendModuleEdge], list[str]]:
    return cfg_masked_row_reduce_catalog(hardware_model)


__all__ = [
    "PASS_SEQUENCE",
    "flash_attention2d_catalog",
    "attn_fwd_catalog",
    "row_softmax_catalog",
    "row_reduction_catalog",
    "elementwise2d_catalog",
    "layer_norm_persistent_catalog",
    "rms_norm2d_catalog",
    "group_norm_kernel_catalog",
    "matmul_fused_epilogue2d_catalog",
    "rope_view_catalog",
    "cfg_masked_row_reduce_catalog",
    "cross_entropy_loss_catalog",
]
