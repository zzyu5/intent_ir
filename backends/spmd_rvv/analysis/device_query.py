"""
Device profile query helpers.

- load_profile: load from JSON or built-in presets.
- query_remote_device: SSH to a remote RVV host and run a probe script (password or key).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

from .hardware_profile import RVVHardwareProfile


DEVICE_PROFILES: Dict[str, Dict] = {
    "generic_rvv_256": {
        "num_cores": 4,
        "rvv_vlen_bits": 256,
        "frequency_ghz": 1.5,
        "mem_bandwidth_gbps": 12.0,
        "l1d_cache_kb": 32,
        "l2_cache_kb": 512,
    },
    "generic_rvv_512": {
        "num_cores": 4,
        "rvv_vlen_bits": 512,
        "frequency_ghz": 1.5,
        "mem_bandwidth_gbps": 16.0,
        "l1d_cache_kb": 32,
        "l2_cache_kb": 1024,
    },
}


def load_profile(name_or_path: str) -> RVVHardwareProfile:
    p = Path(name_or_path)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return RVVHardwareProfile(**data)
    if name_or_path in DEVICE_PROFILES:
        return RVVHardwareProfile(**DEVICE_PROFILES[name_or_path])
    raise ValueError(f"unknown profile {name_or_path}")


def query_remote_device(
    host: str,
    *,
    user: str = "ubuntu",
    password: Optional[str] = None,
    port: int = 22,
    probe_script: str | Path = None,
    timeout: int = 60,
) -> RVVHardwareProfile:
    """
    SSH to a remote device and run a probe script that prints JSON profile.

    - If paramiko is available and password is given, use paramiko.
    - Otherwise fall back to system ssh (expects key-based login).
    """
    # Default probe lives in repo-root `scripts/rvv_probe.sh`.
    probe_path = (
        Path(probe_script)
        if probe_script
        else Path(__file__).resolve().parents[3] / "scripts" / "rvv_probe.sh"
    )
    if not probe_path.exists():
        raise FileNotFoundError(f"probe script not found: {probe_path}")
    script_content = probe_path.read_text(encoding="utf-8")

    if password:
        try:
            import paramiko  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"paramiko not available for password auth: {e}")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(hostname=host, port=port, username=user, password=password, timeout=timeout)
        stdin, stdout, stderr = client.exec_command("bash -s", timeout=timeout)
        stdin.write(script_content)
        stdin.channel.shutdown_write()
        out = stdout.read().decode("utf-8")
        err = stderr.read().decode("utf-8")
        client.close()
        if err:
            raise RuntimeError(f"remote probe stderr: {err}")
    else:
        ssh_cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            f"{user}@{host}",
            "bash -s",
        ]
        proc = subprocess.run(ssh_cmd, input=script_content.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"ssh failed: {proc.stderr.decode('utf-8')}")
        out = proc.stdout.decode("utf-8")

    data = json.loads(out.strip())
    return RVVHardwareProfile(**data)


__all__ = ["load_profile", "DEVICE_PROFILES", "query_remote_device"]
