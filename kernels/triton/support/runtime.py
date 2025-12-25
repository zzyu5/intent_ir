"""
Minimal runtime stubs to satisfy Triton op imports.
"""

from __future__ import annotations

import contextlib
import types
from typing import Any, Iterable

import torch
import triton


def get_tuned_config(name: str) -> Iterable[Any]:
    # Provide a simple default config for autotune use
    if name == "attention":
        return [
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "STAGE": 2, "HAS_ATTN_MASK": 0, "PRE_LOAD_V": 0}, num_warps=4, num_stages=2),
        ]
    if name == "any":
        return [
            triton.Config({"BLOCK_M": 8, "BLOCK_N": 16}, num_warps=2, num_stages=2),
        ]
    if name == "groupnorm":
        return [
            triton.Config({"BLOCK_GROUP_SIZE": 8, "BLOCK_HW_SIZE": 8}, num_warps=4, num_stages=2),
        ]
    if name == "softmax_inner":
        return [
            triton.Config({"TILE_N": 128, "ONE_TILE_PER_CTA": 1}, num_warps=4, num_stages=2),
        ]
    if name == "softmax_non_inner":
        return [
            triton.Config({"TILE_N": 32, "TILE_K": 32, "ONE_TILE_PER_CTA": 1}, num_warps=4, num_stages=2),
        ]
    if name == "upsample_bicubic2d_aa":
        return [
            triton.Config({"BLOCK_X": 16, "BLOCK_Y": 16}, num_warps=4, num_stages=2),
        ]
    if name == "layer_norm_persistent":
        return [
            triton.Config({"TILE_N": 1024}, num_warps=4, num_stages=2),
        ]
    if name == "layer_norm_loop":
        return [
            triton.Config({"TILE_N": 1024}, num_warps=4, num_stages=2),
        ]
    if name == "layer_norm_backward":
        return [
            triton.Config({"BLOCK_ROW_SIZE": 16, "BLOCK_COL_SIZE": 128}, num_warps=4, num_stages=2),
        ]
    return [triton.Config({}, num_warps=4, num_stages=2)]


class _TorchDeviceFn:
    @contextlib.contextmanager
    def device(self, dev):
        if dev is None:
            yield
        else:
            with torch.cuda.device(dev):
                yield


torch_device_fn = _TorchDeviceFn()

def _pow2_ceil(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def get_heuristic_config(name: str) -> dict[str, Any]:
    """
    Triton `@triton.heuristics` expects a dict[str, callable(args)->value].
    We keep these heuristics conservative and mostly used for bringing up
    copied kernels that rely on heuristic meta-params.
    """
    if name == "softmax_inner":
        return {
            "TILE_N": lambda args: min(_pow2_ceil(int(args["N"])), 1024),
            "ONE_TILE_PER_CTA": lambda args: int(int(args["N"]) <= 1024),
        }
    if name in {"softmax_non_inner", "softmax_backward_non_inner"}:
        return {
            "TILE_N": lambda args: min(_pow2_ceil(int(args["N"])), 128),
            "TILE_K": lambda args: min(_pow2_ceil(int(args["K"])), 128),
            "ONE_TILE_PER_CTA": lambda args: 1,
        }
    if name == "softmax_backward_inner":
        return {
            "TILE_M": lambda args: 16,
            "TILE_N": lambda args: min(_pow2_ceil(int(args["N"])), 1024),
            "ONE_TILE_PER_CTA": lambda args: int(int(args["N"]) <= 1024),
        }
    return {}


# Some copied kernels expect `runtime.device.name` to exist.
device = types.SimpleNamespace(name="cuda")


__all__ = ["get_tuned_config", "get_heuristic_config", "torch_device_fn", "device"]
