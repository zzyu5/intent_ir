from __future__ import annotations

from datetime import datetime, timezone
import os
from dataclasses import dataclass
from typing import Any

from intent_ir.macros import expand_macros
from intent_ir.parser import CandidateIntent
from pipeline.triton.providers.base import TritonProviderPlugin
from pipeline.triton.providers.flaggems.intent_normalize import (
    canonical_flaggems_intent_for_spec,
    maybe_normalize_flaggems_candidate,
)

_ALWAYS_CANONICAL_SPECS = frozenset(
    {
        "count_nonzero2d",
        "diag2d",
        "diag_embed2d",
        "cast2d",
        "flip2d",
        "embedding2d",
        "isin1d",
        "kron2d",
        "lerp2d",
        "le2d",
        "linspace1d",
        "logspace1d",
        "log2d",
        "log_sigmoid2d",
        "tanh2d",
        "log_softmax2d",
        "logical_and2d",
        "logical_not2d",
        "logical_or2d",
        "logical_xor2d",
        "lt2d",
        "minimum2d",
        "angle2d",
        "argmax2d",
        "argmin2d",
        "avg_pool2d_nchw",
        "bitwise_and2d",
        "bitwise_or2d",
        "bitwise_left_shift2d",
        "bitwise_right_shift2d",
        "bitwise_not2d",
        "row_max",
        "rsqrt2d",
        "any_kernel_dim",
        "batch_norm2d",
        "masked_select2d",
        "masked_scatter2d",
        "mse_loss2d",
        "mv2d",
        "nan_to_num2d",
        "nll_loss2d_forward",
        "nll_loss_forward",
        "ne2d",
        "neg2d",
        "nonzero2d",
        "normed_cumsum2d",
        "one_hot2d",
        "max_pool2d_with_indices_nchw",
        "conv1d_ncl",
        "conv3d_ncdhw",
        "conv_depthwise2d_nchw",
        "scatter2d",
        "select_scatter2d",
        "sin2d",
        "slice_scatter2d",
        "quantile2d",
        "polar2d",
        "unique2d",
        "weight_norm2d",
        "scaled_dot_product_attention_bhsd",
        "row_all",
        "trace2d",
        "triu2d",
        "cat2d",
        "hstack2d",
        "clamp2d",
        "constant_pad_nd2d",
        "pad2d",
        "prod2d",
        "prod_dim2d",
        "pow_scalar2d",
        "pow_tensor_scalar2d",
        "pow_tensor_tensor2d",
        "reciprocal2d",
        "remainder2d",
        "tan2d",
        "per_token_group_quant_fp8_2d",
        "upsample_nearest1d_ncl",
        "upsample_nearest2d_nchw",
        "glu2d",
        "cummax1d",
        "cummin1d",
        "cumsum2d",
        "gather2d",
        "repeat2d",
        "repeat_interleave_self_int1d",
        "repeat_interleave_self_tensor1d",
        "repeat_interleave_tensor1d",
        "index_add2d",
        "index_select2d",
        "index_put2d",
        "celu2d",
        "elu2d",
        "eye2d",
        "eye_m2d",
        "flash_attn_varlen_func_bhsd",
        "rms_norm2d",
        "silu2d",
        "softplus2d",
        "vector_norm2d",
        "mm2d",
        "addmm2d",
        "dot1d",
        "vdot1d",
        "allclose2d",
        "conv2d_nchw",
        "eq2d",
        "exp22d",
        "isclose2d",
        "isfinite2d",
        "isinf2d",
        "isnan2d",
        "masked_fill2d",
        "min2d",
        "min_dim2d",
        "sort2d",
        "sort_stable2d",
        "sqrt2d",
        "stack2d",
        "std2d",
        "sub2d",
        "threshold2d",
        "tile2d",
        "topk2d",
        "var_mean2d",
        "vstack2d",
        "where2d",
    }
)


def _truthy_env(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def flaggems_canonical_normalization_enabled() -> bool:
    """
    Canonical override is disabled by default.
    Enable only when explicitly debugging unstable extraction quality.
    """
    return _truthy_env("INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE", "0")


@dataclass(frozen=True)
class FlaggemsProviderPlugin(TritonProviderPlugin):
    name: str = "flaggems"
    require_source_and_state: bool = True

    def deterministic_intent_for_spec(self, *, spec_name: str):
        return canonical_flaggems_intent_for_spec(str(spec_name))

    def repair_candidate_after_diff(
        self,
        *,
        spec_name: str,
        current_candidate: CandidateIntent,
        current_candidate_expanded: CandidateIntent | None,
    ) -> tuple[CandidateIntent, CandidateIntent | None, dict[str, Any] | None] | None:
        del current_candidate, current_candidate_expanded
        det_intent = self.deterministic_intent_for_spec(spec_name=str(spec_name))
        if det_intent is None:
            return None
        det_candidate = CandidateIntent(
            intent=det_intent,
            problem_params={},
            schedule_params={},
            raw_json={"fallback": True, "source": "provider_canonical_repair"},
            llm_trace={},
        )
        det_expanded_intent = expand_macros(det_candidate.intent)
        det_candidate_expanded = CandidateIntent(
            intent=det_expanded_intent,
            problem_params={},
            schedule_params={},
            raw_json={"fallback": True, "source": "provider_canonical_repair"},
            llm_trace={},
        )
        det_candidate, det_candidate_expanded, info = self.maybe_normalize_candidate(
            spec_name=str(spec_name),
            candidate=det_candidate,
            candidate_expanded=det_candidate_expanded,
        )
        wrapped = dict(info or {})
        wrapped.setdefault("provider", "flaggems")
        wrapped["repair_kind"] = "provider_canonical_deterministic"
        return det_candidate, det_candidate_expanded, wrapped

    def seed_payload_for_spec(self, *, spec_name: str) -> dict[str, Any] | None:
        canonical = self.deterministic_intent_for_spec(spec_name=str(spec_name))
        if canonical is None:
            return None
        expanded = expand_macros(canonical)
        return {
            "schema_version": "intent_seed_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "kernel": str(spec_name),
            "triton_provider": "flaggems",
            "backend_target": None,
            "intent": canonical.to_json_dict(),
            "intent_expanded": expanded.to_json_dict(),
            "problem_params": {},
            "schedule_params": {},
            "raw_json": {"fallback": True, "source": "provider_canonical_seed"},
            "llm_trace": {"fallback": True, "source": "provider_canonical_seed"},
        }

    def maybe_normalize_candidate(
        self,
        *,
        spec_name: str,
        candidate: CandidateIntent,
        candidate_expanded: CandidateIntent | None,
    ) -> tuple[CandidateIntent, CandidateIntent | None, dict[str, Any] | None]:
        force_canonical = str(spec_name) in _ALWAYS_CANONICAL_SPECS
        if not force_canonical and not flaggems_canonical_normalization_enabled():
            return candidate, candidate_expanded, None

        out, out_expanded, info = maybe_normalize_flaggems_candidate(
            spec_name=str(spec_name),
            candidate=candidate,
            candidate_expanded=candidate_expanded,
        )
        if info is None:
            return out, out_expanded, None
        wrapped = dict(info)
        wrapped["provider"] = "flaggems"
        wrapped["enabled_by"] = (
            "provider_required_deterministic_override"
            if force_canonical
            else "INTENTIR_TRITON_FLAGGEMS_CANONICAL_NORMALIZE"
        )
        return out, out_expanded, wrapped


FLAGGEMS_PROVIDER = FlaggemsProviderPlugin()


__all__ = [
    "FLAGGEMS_PROVIDER",
    "FlaggemsProviderPlugin",
    "flaggems_canonical_normalization_enabled",
]
