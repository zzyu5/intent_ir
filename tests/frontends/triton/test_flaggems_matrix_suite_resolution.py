from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_matrix_module():
    script_path = ROOT / "scripts" / "flaggems" / "run_multibackend_matrix.py"
    spec = importlib.util.spec_from_file_location("run_multibackend_matrix", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_suite_auto_promotes_to_coverage_for_non_smoke_kernel() -> None:
    mod = _load_matrix_module()
    suite, kernels = mod._resolve_suite_and_kernel_filter(
        requested_suite="smoke",
        requested_kernels=["eye2d"],
        flaggems_opset="deterministic_forward",
        backend_target="rvv",
    )
    assert suite == "coverage"
    assert kernels == ["eye2d"]


def test_smoke_suite_keeps_smoke_for_smoke_kernel() -> None:
    mod = _load_matrix_module()
    suite, kernels = mod._resolve_suite_and_kernel_filter(
        requested_suite="smoke",
        requested_kernels=["add2d"],
        flaggems_opset="deterministic_forward",
        backend_target="rvv",
    )
    assert suite == "smoke"
    assert kernels == ["add2d"]


def test_unknown_kernel_rejected_with_clear_error() -> None:
    mod = _load_matrix_module()
    with pytest.raises(SystemExit) as exc:
        mod._resolve_suite_and_kernel_filter(
            requested_suite="coverage",
            requested_kernels=["not_a_real_kernel"],
            flaggems_opset="deterministic_forward",
            backend_target="rvv",
        )
    assert "unknown kernel(s)" in str(exc.value)


def test_load_active_semantic_ops_reads_items(tmp_path: Path) -> None:
    mod = _load_matrix_module()
    active = tmp_path / "active_batch.json"
    active.write_text(
        json.dumps(
            {
                "schema_version": "flaggems_active_batch_v1",
                "items": [
                    {"semantic_op": "diag"},
                    {"semantic_op": "angle"},
                    {"semantic_op": "diag"},
                    {"semantic_op": ""},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert mod._load_active_semantic_ops(active) == ["diag", "angle"]


def test_matrix_forwards_cuda_stage_timeout_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mod = _load_matrix_module()
    recorded_cmds: list[list[str]] = []

    def _fake_run(cmd: list[str], *, cwd: Path):
        _ = cwd
        recorded_cmds.append(list(cmd))
        if "--out" in cmd:
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"results": []}), encoding="utf-8")
        return 0, "", ""

    monkeypatch.setattr(mod, "_run", _fake_run)
    monkeypatch.setattr(mod, "_suite_kernel_names", lambda **kwargs: ["angle2d"])
    monkeypatch.setattr(mod, "_load_active_semantic_ops", lambda _: ["angle"])
    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "angle2d.json").write_text(json.dumps({"diff": {"ok": True}}), encoding="utf-8")
    out_dir = tmp_path / "matrix_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_multibackend_matrix.py",
            "--suite",
            "coverage",
            "--kernel",
            "angle2d",
            "--skip-pipeline",
                "--skip-rvv",
                "--active-batch",
                str(tmp_path / "active.json"),
                "--pipeline-out-dir",
                str(pipeline_dir),
                "--seed-cache-dir",
                str(tmp_path / "seed"),
            "--out-dir",
            str(out_dir),
            "--cuda-timeout-sec",
            "111",
            "--cuda-compile-timeout-sec",
            "222",
            "--cuda-launch-timeout-sec",
            "333",
            "--cuda-runtime-backend",
            "nvrtc",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert int(exc.value.code) == 0
    cuda_cmds = [c for c in recorded_cmds if "scripts/cuda_backend_smoke.py" in c]
    assert len(cuda_cmds) == 1
    cuda_cmd = cuda_cmds[0]
    assert cuda_cmd[cuda_cmd.index("--timeout-sec") + 1] == "111"
    assert cuda_cmd[cuda_cmd.index("--compile-timeout-sec") + 1] == "222"
    assert cuda_cmd[cuda_cmd.index("--launch-timeout-sec") + 1] == "333"
    assert cuda_cmd[cuda_cmd.index("--runtime-backend") + 1] == "nvrtc"
    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["cuda_timeout_sec"] == 111
    assert summary["cuda_compile_timeout_sec"] == 222
    assert summary["cuda_launch_timeout_sec"] == 333
    assert summary["cuda_runtime_backend"] == "nvrtc"


def test_matrix_skips_backend_stages_when_provider_report_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mod = _load_matrix_module()
    recorded_cmds: list[list[str]] = []

    def _fake_run(cmd: list[str], *, cwd: Path):
        _ = cwd
        recorded_cmds.append(list(cmd))
        if "--out" in cmd:
            out = Path(cmd[cmd.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"results": []}), encoding="utf-8")
        return 0, "", ""

    monkeypatch.setattr(mod, "_run", _fake_run)
    monkeypatch.setattr(mod, "_suite_kernel_names", lambda **kwargs: ["tile2d"])
    monkeypatch.setattr(mod, "_load_active_semantic_ops", lambda _: ["tile"])
    pipeline_dir = tmp_path / "pipeline"
    out_dir = tmp_path / "matrix_out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_multibackend_matrix.py",
            "--suite",
            "coverage",
            "--kernel",
            "tile2d",
            "--active-batch",
            str(tmp_path / "active.json"),
            "--pipeline-out-dir",
            str(pipeline_dir),
            "--seed-cache-dir",
            str(tmp_path / "seed"),
            "--out-dir",
            str(out_dir),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert int(exc.value.code) == 1

    pipeline_cmds = [c for c in recorded_cmds if "scripts/triton/flaggems_full_pipeline_verify.py" in c]
    assert len(pipeline_cmds) == 1
    assert "--strict-kernel-failure" in pipeline_cmds[0]

    assert not any("scripts/backend_codegen_smoke.py" in c for c in recorded_cmds)
    assert not any("scripts/cuda_backend_smoke.py" in c for c in recorded_cmds)

    converge_cmds = [c for c in recorded_cmds if "scripts/flaggems/converge_status.py" in c]
    assert len(converge_cmds) == 1
    converge_cmd = converge_cmds[0]
    assert converge_cmd[converge_cmd.index("--scope-mode") + 1] == "active_only"
    assert converge_cmd[converge_cmd.index("--scope-semantic-ops") + 1] == "tile"
    summary = json.loads((out_dir / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["missing_provider_reports"] == ["tile2d"]
