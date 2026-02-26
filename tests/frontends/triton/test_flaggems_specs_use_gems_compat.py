from __future__ import annotations

from contextlib import contextmanager

from pipeline.triton.providers.flaggems import specs


class _UseGemsLegacy:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    @contextmanager
    def use_gems(self, *args, **kwargs):
        self.calls.append((tuple(args), dict(kwargs)))
        if kwargs:
            raise TypeError("use_gems.__init__() got an unexpected keyword argument 'include'")
        yield


class _UseGemsModern:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    @contextmanager
    def use_gems(self, *args, **kwargs):
        self.calls.append((tuple(args), dict(kwargs)))
        yield


def test_flaggems_use_gems_compat_falls_back_for_legacy_signature(monkeypatch) -> None:
    legacy = _UseGemsLegacy()
    monkeypatch.setattr(specs, "flag_gems", legacy)
    with specs._flaggems_use_gems(include=["add"]):
        pass
    assert legacy.calls == [
        ((), {"include": ["add"]}),
        ((), {}),
    ]


def test_flaggems_use_gems_compat_prefers_include_for_modern_signature(monkeypatch) -> None:
    modern = _UseGemsModern()
    monkeypatch.setattr(specs, "flag_gems", modern)
    with specs._flaggems_use_gems(include=["add"]):
        pass
    assert modern.calls == [
        ((), {"include": ["add"]}),
    ]
