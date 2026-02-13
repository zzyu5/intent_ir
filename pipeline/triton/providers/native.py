from __future__ import annotations

from pipeline.triton.providers.base import TritonProviderPlugin


NATIVE_PROVIDER = TritonProviderPlugin(name="native", require_source_and_state=False)


__all__ = ["NATIVE_PROVIDER"]
