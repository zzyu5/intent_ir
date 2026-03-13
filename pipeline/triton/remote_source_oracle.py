from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Mapping

from pipeline.interfaces import KernelDescriptor


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ROOT.parent
DEFAULT_LIGER_ROOT = WORKSPACE_ROOT / "Liger-Kernel"
REMOTE_RUNNER = ROOT / "scripts" / "triton" / "liger_remote_source_oracle_runner.py"
DEFAULT_REMOTE_SSH = "kingdom@211.87.236.70"

_SUPPORTED_KERNELS = {
    "liger_swiglu",
    "liger_rms_norm",
    "liger_fused_add_rms_norm",
    "liger_rope",
    "liger_cross_entropy",
    "liger_geglu",
    "liger_layer_norm",
    "liger_softmax",
    "liger_group_norm",
    "liger_dyt",
    "liger_qwen2vl_mrope",
    "liger_sparsemax",
    "liger_kl_div",
    "liger_jsd",
    "liger_fused_linear_cross_entropy",
    "liger_fused_linear_jsd",
    "liger_fused_neighborhood_attention",
    "liger_grpo_loss",
    "liger_llama4_rope",
    "liger_mhc",
    "liger_multi_token_attention",
    "liger_poly_norm",
    "liger_tiled_mlp",
    "liger_tvd",
}


def remote_source_enabled() -> bool:
    raw = str(os.getenv("INTENTIR_ORG_REMOTE_SOURCE_ENABLE", "") or "").strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    return bool(str(os.getenv("INTENTIR_ORG_REMOTE_SOURCE_SSH", DEFAULT_REMOTE_SSH) or "").strip())


def _remote_source_allow_fallback() -> bool:
    raw = str(os.getenv("INTENTIR_ORG_REMOTE_SOURCE_ALLOW_FALLBACK", "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _liger_local_root() -> Path:
    raw = str(os.getenv("INTENTIR_ORG_REMOTE_SOURCE_LIGER_ROOT", "") or "").strip()
    if raw:
        return Path(raw).resolve()
    return DEFAULT_LIGER_ROOT.resolve()


def _ssh_target() -> str:
    raw = str(os.getenv("INTENTIR_ORG_REMOTE_SOURCE_SSH", DEFAULT_REMOTE_SSH) or "").strip()
    if not raw:
        raise RuntimeError("INTENTIR_ORG_REMOTE_SOURCE_SSH is not set and no default remote target is configured")
    return raw


def _remote_base_dir() -> str:
    raw = str(os.getenv("INTENTIR_ORG_REMOTE_SOURCE_BASE_DIR", "") or "").strip()
    if raw:
        return raw
    return "/tmp/intentir_remote_source_oracle"


def _run_local(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=(str(cwd) if cwd is not None else None),
        capture_output=True,
        text=True,
        check=False,
    )


def _run_local_checked(cmd: list[str], *, cwd: Path | None = None, what: str) -> subprocess.CompletedProcess[str]:
    proc = _run_local(cmd, cwd=cwd)
    if int(proc.returncode) != 0:
        raise RuntimeError(f"{what} failed rc={proc.returncode}: {proc.stderr or proc.stdout}")
    return proc


def _remote_run_checked(ssh_target: str, remote_cmd: str, *, what: str) -> subprocess.CompletedProcess[str]:
    return _run_local_checked(["ssh", ssh_target, remote_cmd], what=what)


def _canonical_remote_bindings(spec_name: str, shape_bindings: Mapping[str, int] | None) -> dict[str, int]:
    raw = {str(k): int(v) for k, v in dict(shape_bindings or {}).items() if str(k).strip()}
    defaults = {
        "liger_swiglu": {"M": 65536, "N": 256},
        "liger_rms_norm": {"M": 2048, "N": 32768},
        "liger_fused_add_rms_norm": {"M": 2048, "N": 32768},
        "liger_rope": {"B": 2, "QH": 32, "KH": 8, "S": 2048, "HD": 128},
        "liger_cross_entropy": {"BT": 2048, "V": 4096},
        "liger_geglu": {"M": 65536, "N": 256},
        "liger_layer_norm": {"M": 2048, "N": 4096},
        "liger_softmax": {"M": 2048, "N": 4096},
        "liger_group_norm": {"N": 32, "C": 512, "HW": 64, "num_groups": 32},
        "liger_dyt": {"M": 2048, "N": 4096},
        "liger_qwen2vl_mrope": {"B": 2, "QH": 32, "KH": 8, "S": 2048, "HD": 128},
        "liger_sparsemax": {"M": 2048, "N": 4096},
        "liger_kl_div": {"BT": 2048, "V": 4096},
        "liger_jsd": {"BT": 2048, "V": 4096},
        "liger_fused_linear_cross_entropy": {"BT": 2048, "H": 2048, "V": 4096},
        "liger_fused_linear_jsd": {"BT": 2048, "H": 2048, "V": 4096},
        "liger_fused_neighborhood_attention": {"B": 1, "QH": 8, "S": 512, "HD": 64, "kernel_size": 7, "dilation": 1},
        "liger_grpo_loss": {"B": 4, "T": 512, "V": 4096},
        "liger_llama4_rope": {"B": 1, "QH": 32, "KH": 8, "S": 2048, "HD": 64},
        "liger_mhc": {"B": 2, "T": 512, "HC": 4, "C": 128},
        "liger_multi_token_attention": {"B": 2, "CIN": 4, "COUT": 4, "L": 128, "K": 3, "groups": 1},
        "liger_poly_norm": {"M": 2048, "N": 4096},
        "liger_tiled_mlp": {"B": 1, "S": 4096, "H": 2048, "I": 5632, "num_shards": 4},
        "liger_tvd": {"BT": 2048, "V": 4096},
    }
    out = dict(defaults.get(str(spec_name), {}))
    out.update(raw)
    return out


def _job_token(spec_name: str, bindings: Mapping[str, int]) -> str:
    payload = json.dumps(
        {
            "kernel": str(spec_name),
            "bindings": {str(k): int(v) for k, v in dict(bindings).items()},
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _patch_descriptor_to_remote_only(
    desc: KernelDescriptor,
    *,
    manifest: Mapping[str, Any],
    local_paths: Mapping[str, Path],
) -> None:
    if not isinstance(desc.meta, dict):
        desc.meta = {}
    if not isinstance(desc.artifacts.extra, dict):
        desc.artifacts.extra = {}
    ttir_path = local_paths.get("ttir")
    ttgir_path = local_paths.get("ttgir")
    ptx_path = local_paths.get("ptx")
    llir_path = local_paths.get("llir")
    cubin_path = local_paths.get("cubin")
    if ttir_path is not None and ttir_path.is_file():
        desc.artifacts.ttir_path = str(ttir_path)
        desc.artifacts.ttir_text = None
        desc.meta["ttir_original_path"] = str(ttir_path)
    if ttgir_path is not None and ttgir_path.is_file():
        desc.artifacts.ttgir_path = str(ttgir_path)
        desc.artifacts.ttgir_text = None
        desc.meta["ttgir_original_path"] = str(ttgir_path)
    if ptx_path is not None and ptx_path.is_file():
        desc.artifacts.ptx_text = None
        desc.artifacts.extra["ptx_path"] = str(ptx_path)
        desc.meta["ptx_original_path"] = str(ptx_path)
    if llir_path is not None and llir_path.is_file():
        desc.artifacts.llvm_ir_text = None
        desc.artifacts.extra["llvm_ir_path"] = str(llir_path)
        desc.meta["llvm_ir_original_path"] = str(llir_path)
    if cubin_path is not None and cubin_path.is_file():
        desc.artifacts.extra["cubin_path"] = str(cubin_path)
        desc.meta["cubin_original_path"] = str(cubin_path)
    # Force ORG evidence to consume only the remote artifact set.
    desc.frontend_facts = {}
    desc.frontend_constraints = {}
    desc.meta["remote_source_oracle"] = dict(manifest)


def apply_remote_source_oracle(
    *,
    spec_name: str,
    out_dir: Path,
    desc: KernelDescriptor | None,
    shape_bindings: Mapping[str, int] | None,
) -> dict[str, Any] | None:
    if desc is None or not remote_source_enabled():
        return None
    kernel = str(spec_name).strip()
    if kernel not in _SUPPORTED_KERNELS:
        result = {
            "enabled": True,
            "available": False,
            "reason": "unsupported_kernel",
            "kernel": kernel,
        }
        if not _remote_source_allow_fallback():
            raise RuntimeError(f"remote source oracle unsupported for kernel={kernel}")
        return result

    ssh_target = _ssh_target()
    liger_root = _liger_local_root()
    if not liger_root.is_dir():
        raise FileNotFoundError(f"local Liger root missing: {liger_root}")
    local_src_root = liger_root / "src"
    if not local_src_root.is_dir():
        raise FileNotFoundError(f"local Liger src missing: {local_src_root}")
    if not REMOTE_RUNNER.is_file():
        raise FileNotFoundError(f"remote oracle runner missing: {REMOTE_RUNNER}")

    bindings = _canonical_remote_bindings(kernel, shape_bindings)
    token = _job_token(kernel, bindings)
    remote_base = _remote_base_dir().rstrip("/")
    remote_root = f"{remote_base}/{kernel}_{token}"
    remote_src_dir = f"{remote_root}/src"
    remote_out_dir = f"{remote_root}/out"
    remote_runner_path = f"{remote_root}/liger_remote_source_oracle_runner.py"
    local_root = Path(out_dir) / "remote_source_oracle" / kernel
    local_root.mkdir(parents=True, exist_ok=True)
    bindings_path = local_root / "bindings.json"
    bindings_path.write_text(json.dumps(bindings, indent=2, ensure_ascii=False), encoding="utf-8")

    _remote_run_checked(
        ssh_target,
        f"mkdir -p {shlex.quote(remote_root)} {shlex.quote(remote_out_dir)}",
        what="remote mkdir",
    )
    tar_proc = subprocess.Popen(
        ["tar", "czf", "-", "-C", str(liger_root), "src"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    ssh_proc = subprocess.run(
        [
            "ssh",
            ssh_target,
            f"mkdir -p {shlex.quote(remote_root)} && tar xzf - -C {shlex.quote(remote_root)}",
        ],
        stdin=tar_proc.stdout,
        capture_output=True,
        text=True,
        check=False,
    )
    if tar_proc.stdout is not None:
        tar_proc.stdout.close()
    tar_stderr = tar_proc.stderr.read().decode("utf-8", errors="ignore") if tar_proc.stderr is not None else ""
    tar_rc = tar_proc.wait()
    if int(tar_rc) != 0 or int(ssh_proc.returncode) != 0:
        raise RuntimeError(
            "remote source sync failed: "
            f"tar_rc={tar_rc} ssh_rc={ssh_proc.returncode} tar={tar_stderr} ssh={ssh_proc.stderr or ssh_proc.stdout}"
        )

    _run_local_checked(
        ["scp", str(REMOTE_RUNNER), f"{ssh_target}:{remote_runner_path}"],
        what="scp remote oracle runner",
    )
    remote_bindings_json = json.dumps(bindings, ensure_ascii=False, sort_keys=True)
    remote_cmd = (
        f"PYTHONPATH={shlex.quote(remote_src_dir)}:$PYTHONPATH "
        f"python3 {shlex.quote(remote_runner_path)} "
        f"--kernel {shlex.quote(kernel)} "
        f"--out-dir {shlex.quote(remote_out_dir)} "
        f"--bindings-json {shlex.quote(remote_bindings_json)}"
    )
    remote_exec = _remote_run_checked(ssh_target, remote_cmd, what="remote source compile")
    remote_log_path = local_root / "remote_exec.log"
    remote_log_path.write_text(
        json.dumps(
            {
                "ssh_target": ssh_target,
                "command": remote_cmd,
                "stdout": str(remote_exec.stdout or ""),
                "stderr": str(remote_exec.stderr or ""),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    manifest_local = local_root / "manifest.json"
    _run_local_checked(
        ["scp", f"{ssh_target}:{remote_out_dir}/manifest.json", str(manifest_local)],
        what="scp remote manifest",
    )
    manifest = json.loads(manifest_local.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise RuntimeError("remote manifest is not a JSON object")

    local_paths: dict[str, Path] = {}
    for key in ("ttir", "ttgir", "ptx", "llir", "cubin"):
        rel = str((manifest.get("artifacts") or {}).get(key) or "").strip()
        if not rel:
            continue
        remote_path = f"{ssh_target}:{rel}"
        local_path = local_root / Path(rel).name
        _run_local_checked(["scp", remote_path, str(local_path)], what=f"scp remote {key}")
        local_paths[key] = local_path

    _patch_descriptor_to_remote_only(desc, manifest=manifest, local_paths=local_paths)
    result = {
        "enabled": True,
        "available": True,
        "kernel": kernel,
        "ssh_target": ssh_target,
        "remote_root": remote_root,
        "manifest_path": str(manifest_local),
        "bindings": bindings,
        "source_arch": str(manifest.get("source_arch") or ""),
        "artifacts": {k: str(v) for k, v in local_paths.items()},
        "runner_stdout": str(remote_exec.stdout or "").strip(),
        "runner_stderr": str(remote_exec.stderr or "").strip(),
    }
    return result


__all__ = ["apply_remote_source_oracle", "remote_source_enabled"]
