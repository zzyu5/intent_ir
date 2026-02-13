from __future__ import annotations

from pipeline.triton.providers.base import TritonProviderPlugin
from pipeline.triton.providers.flaggems import FLAGGEMS_PROVIDER
from pipeline.triton.providers.native import NATIVE_PROVIDER


_PROVIDERS: dict[str, TritonProviderPlugin] = {
    "native": NATIVE_PROVIDER,
    "flaggems": FLAGGEMS_PROVIDER,
}


def get_provider_plugin(provider_name: str | None) -> TritonProviderPlugin:
    p = str(provider_name or "native").strip().lower() or "native"
    plugin = _PROVIDERS.get(p)
    if plugin is not None:
        return plugin
    # Unknown providers still use generic metadata behavior without hardcoding.
    return TritonProviderPlugin(name=p, require_source_and_state=False)


def registered_providers() -> tuple[str, ...]:
    return tuple(sorted(_PROVIDERS.keys()))


__all__ = ["get_provider_plugin", "registered_providers"]
