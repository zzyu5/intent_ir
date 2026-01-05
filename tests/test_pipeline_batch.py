from dataclasses import dataclass

from pipeline.run import process_batch


@dataclass(frozen=True)
class _Item:
    name: str


def test_process_batch_continues_on_error_and_calls_hook():
    items = [_Item("a"), _Item("b"), _Item("c")]
    seen = []

    def process_one(it: _Item):
        if it.name == "b":
            raise ValueError("boom")
        return {"ok": True, "kernel": it.name}

    def on_error(it: _Item, err: dict) -> None:
        seen.append((it.name, (err.get("error") or {}).get("type")))

    results = process_batch(items, process_one, name_fn=lambda i: i.name, on_error=on_error)
    assert [r.get("kernel") for r in results] == ["a", "b", "c"]
    assert results[1]["ok"] is False
    assert results[1]["error"]["type"] == "ValueError"
    assert seen == [("b", "ValueError")]

