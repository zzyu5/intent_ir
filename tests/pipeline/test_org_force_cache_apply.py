from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("triton")
pytest.importorskip("torch")

from intent_ir.ir import IntentFunction
from pipeline.interfaces import KernelArtifactBundle, KernelDescriptor
from pipeline.triton.org_bridge import load_org_attr
from pipeline.triton.core import _run_org_plugin


def _dummy_intent(name: str) -> IntentFunction:
    # _run_org_plugin builds an intent_summary even when using cache; supply a minimal valid IntentFunction.
    return IntentFunction.from_json_dict(
        {
            "name": str(name),
            "tensors": {
                "x": {"dtype": "f32", "shape": [1], "layout": "row_major"},
                "Out": {"dtype": "f32", "shape": [1], "layout": "row_major"},
            },
            "ops": [{"op": "identity", "inputs": ["x"], "output": "Out"}],
            "outputs": ["Out"],
        }
    )


def _dummy_desc(*, kernel: str, ttgir_text: str | None = None, ttgir_path: Path | None = None) -> KernelDescriptor:
    desc = KernelDescriptor(schema_version="kernel_desc_v1.0", name=str(kernel), frontend="triton")
    desc.source_text = "def kernel(): pass"
    desc.artifacts = KernelArtifactBundle(
        ttgir_text=(str(ttgir_text) if ttgir_text is not None else None),
        ttgir_path=(str(ttgir_path) if ttgir_path is not None else None),
    )
    desc.launch = {"canonical_shapes": {"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64}}
    return desc


def _write_seed(*, out_dir: Path, kernel: str, org_payload: object) -> Path:
    validate_org_doc = load_org_attr("org.schema", "validate_org_doc")
    save_org_seed = load_org_attr("org.io", "save_org_seed")
    seed_path = out_dir / f"{kernel}.org_seed.json"
    org = validate_org_doc(org_payload)
    save_org_seed(
        path=seed_path,
        kernel=str(kernel),
        triton_provider="native",
        backend_target="cuda_5090d",
        org=org,
        raw_json=(org.to_json_dict()),
        llm_trace={"provider": "test", "cached": True},
        quality={"diff_ok": True, "static_ok": True, "contract_level": "test"},
    )
    return seed_path


def test_force_cache_extract_writes_org_doc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "extract")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")

    _write_seed(
        out_dir=tmp_path,
        kernel="flash_attention2d",
        org_payload={
            "schema_version": "intentir_org_v1",
            "kernel": "flash_attention2d",
            "nodes": [
                {
                    "id": "n0",
                    "node_type": "tiling",
                    "why": ["resident_working_set"],
                    "how": ["scratchpad_staging"],
                    "dims": ["ATTN_BLOCK_KV", "ATTN_SCORE_WARPS"],
                    "constraints": [],
                    "evidence": [{"kind": "extra", "path": "extra.shape_bindings"}],
                }
            ],
            "edges": [],
        },
    )

    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="flash_attention2d",
        out_dir=tmp_path,
        desc=None,
        intent=_dummy_intent("flash_attention2d"),
        report=report,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )

    assert (tmp_path / "flash_attention2d.org.json").is_file()
    assert not (tmp_path / "flash_attention2d.org_candidates.txt").exists()
    assert report.get("org") and isinstance(report.get("org"), dict)
    assert bool((report["org"] or {}).get("cache_used")) is True


def test_force_cache_apply_flash_attention2d_requires_ttgir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")

    _write_seed(
        out_dir=tmp_path,
        kernel="flash_attention2d",
        org_payload={
            "schema_version": "intentir_org_v1",
            "kernel": "flash_attention2d",
            "nodes": [
                {
                    "id": "n0",
                    "node_type": "overlap_pipeline",
                    "why": ["avoid_recompute"],
                    "how": ["double_buffering"],
                    "dims": [{"name": "ATTN_BLOCK_KV", "allowed": [64]}],
                    "constraints": [],
                    "evidence": [{"kind": "extra", "path": "extra.shape_bindings"}],
                }
            ],
            "edges": [],
        },
    )

    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="flash_attention2d",
        out_dir=tmp_path,
        desc=None,
        intent=_dummy_intent("flash_attention2d"),
        report=report,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )

    assert report.get("org")
    assert (report["org"] or {}).get("error") == "ttgir_missing"
    assert not (tmp_path / "flash_attention2d.org_candidates.txt").exists()


def test_force_cache_apply_flash_attention2d_uses_ttgir_primary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("INTENTIR_ORG_MODE", "apply")
    monkeypatch.setenv("INTENTIR_ORG_SEED_POLICY", "force_cache")
    monkeypatch.setenv("INTENTIR_COMPILER_STACK", "python")
    monkeypatch.setenv("INTENTIR_CUDA_SM", "sm_120")
    monkeypatch.setenv("INTENTIR_ORG_BUDGET", "8")

    _write_seed(
        out_dir=tmp_path,
        kernel="flash_attention2d",
        org_payload={
            "schema_version": "intentir_org_v1",
            "kernel": "flash_attention2d",
            "nodes": [
                {
                    "id": "n0",
                    "node_type": "tiling",
                    "why": [],
                    "how": [],
                    "dims": ["ATTN_BLOCK_KV", "ATTN_SCORE_WARPS"],
                    "constraints": [],
                    "evidence": [{"kind": "extra", "path": "extra.shape_bindings"}],
                }
            ],
            "edges": [],
        },
    )

    ttgir_path = tmp_path / "flash_attention2d.ttgir"
    ttgir_path.write_text(
        (
            '#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>\n'
            'module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {\n'
            '  tt.func public @flash_attention2d_kernel(%Q_ptr: !tt.ptr<f32>, %K_ptr: !tt.ptr<f32>, %V_ptr: !tt.ptr<f32>, %Out_ptr: !tt.ptr<f32>, %sm_scale: f32) {\n'
            '    %pid_q = tt.get_program_id x : i32\n'
            '    %k_33 = tt.load %k_32, %k_25, %cst_2 : tensor<32x64x!tt.ptr<f32>, #blocked>\n'
            '    tt.return\n'
            '  }\n'
            '}\n'
        ),
        encoding="utf-8",
    )

    report: dict[str, object] = {"diff": {"ok": True}, "static_validation": {"ok": True}}
    _run_org_plugin(
        spec_name="flash_attention2d",
        out_dir=tmp_path,
        desc=_dummy_desc(kernel="flash_attention2d", ttgir_path=ttgir_path),
        intent=_dummy_intent("flash_attention2d"),
        report=report,
        shape_bindings={"Q_CTX": 64, "KV_CTX": 64, "HEAD_DIM": 64},
        triton_provider="native",
        backend_target="cuda_5090d",
    )

    org_report = dict(report.get("org") or {})
    assert org_report.get("evidence_source", {}).get("primary") == "ttgir"
    assert (tmp_path / "flash_attention2d.org_plan.json").is_file()
    assert (tmp_path / "flash_attention2d.org_candidates.txt").is_file()
    plan = json.loads((tmp_path / "flash_attention2d.org_plan.json").read_text(encoding="utf-8"))
    assert plan.get("schema_version") == "intentir_backend_plan_v1"
    assert plan.get("candidates")
