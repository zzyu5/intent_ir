from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path

from pipeline.triton.providers.flaggems.registry import ensure_flaggems_importable


class _BrokenEditableFinder(importlib.abc.MetaPathFinder):
    __module__ = "_flag_gems_editable"

    def __init__(self, missing_init: Path) -> None:
        self._missing_init = Path(missing_init)
        self.known_source_files = {"flag_gems": str(self._missing_init)}

    def find_spec(self, fullname: str, path=None, target=None):  # type: ignore[override]
        if fullname != "flag_gems":
            return None
        return importlib.util.spec_from_file_location(
            "flag_gems",
            str(self._missing_init),
            submodule_search_locations=[str(self._missing_init.parent)],
        )


def test_ensure_flaggems_importable_repairs_broken_editable_finder(monkeypatch, tmp_path: Path) -> None:
    src = tmp_path / "fg_src"
    pkg_dir = src / "flag_gems"
    pkg_dir.mkdir(parents=True)
    init_py = pkg_dir / "__init__.py"
    init_py.write_text("VALUE = 7\n", encoding="utf-8")

    broken = _BrokenEditableFinder(tmp_path / "missing" / "flag_gems" / "__init__.py")
    monkeypatch.setenv("FLAGGEMS_SRC", str(src))
    monkeypatch.setattr(sys, "meta_path", [broken, *list(sys.meta_path)], raising=False)
    for key in list(sys.modules):
        if key == "flag_gems" or key.startswith("flag_gems."):
            monkeypatch.delitem(sys.modules, key, raising=False)

    ensure_flaggems_importable(None)
    import flag_gems  # noqa: PLC0415

    assert Path(str(flag_gems.__file__)).resolve() == init_py.resolve()
    assert broken not in sys.meta_path
