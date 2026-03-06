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
    ]
    edges = [
        BackendModuleEdge(src="backend_v6", dst="q_resident_state", edge_type="uses"),
        BackendModuleEdge(src="backend_v6", dst="kv_tile_stage", edge_type="uses"),
        BackendModuleEdge(src="backend_v6", dst="online_softmax_reduce", edge_type="uses"),
        BackendModuleEdge(src="backend_v6", dst="output_accumulator", edge_type="uses"),
        BackendModuleEdge(src="backend_v7", dst="q_resident_state", edge_type="uses"),
        BackendModuleEdge(src="backend_v7", dst="kv_tile_stage", edge_type="uses"),
        BackendModuleEdge(src="backend_v7", dst="output_accumulator", edge_type="uses"),
        BackendModuleEdge(src="backend_v7", dst="prefetch_pipeline", edge_type="optional"),
        BackendModuleEdge(src="backend_v6", dst="prefetch_pipeline", edge_type="optional"),
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


__all__ = ["PASS_SEQUENCE", "flash_attention2d_catalog", "matmul_fused_epilogue2d_catalog"]
