"""
Environment validation script (P1 gap fix).

This script is intentionally conservative: it reports what is available in the
current environment and optionally verifies remote RVV toolchains.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""


def _run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return 0, str(out).strip()
    except subprocess.CalledProcessError as e:
        return int(e.returncode), (e.output or "").strip()
    except FileNotFoundError:
        return 127, ""


def _check_python() -> CheckResult:
    v = sys.version_info
    ok = (v.major, v.minor) >= (3, 10)
    return CheckResult("python", ok, detail=f"{v.major}.{v.minor}.{v.micro}", hint="need Python>=3.10" if not ok else "")


def _check_import(mod: str, *, required: bool, hint: str) -> CheckResult:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", None)
        detail = f"ok{(' ' + str(ver)) if ver else ''}"
        return CheckResult(mod, True, detail=detail)
    except Exception as e:
        return CheckResult(mod, False, detail=f"{type(e).__name__}: {e}", hint=(hint if required else f"optional: {hint}"))


def _check_tool(name: str, cmd: list[str], *, required: bool, hint: str) -> CheckResult:
    rc, out = _run_cmd(cmd)
    ok = rc == 0
    if ok:
        line = out.splitlines()[0] if out else "ok"
        return CheckResult(name, True, detail=line)
    return CheckResult(name, False, detail="not found" if rc == 127 else f"rc={rc}", hint=(hint if required else f"optional: {hint}"))


def _check_remote_rvv(
    *,
    host: str,
    user: str,
    port: int,
    use_key: bool,
    password: Optional[str],
) -> list[CheckResult]:
    out: list[CheckResult] = []
    try:
        import paramiko
    except Exception as e:
        return [CheckResult("remote", False, detail=f"paramiko missing: {type(e).__name__}: {e}", hint="pip install paramiko")]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=str(host), port=int(port), username=str(user), password=(None if use_key else password), timeout=20)
    except Exception as e:
        return [CheckResult("remote_ssh", False, detail=f"{type(e).__name__}: {e}", hint="check host/user/auth and network")]

    def exec_ok(cmd: str) -> tuple[bool, str]:
        try:
            stdin, stdout, stderr = client.exec_command(cmd, timeout=30)
            out_s = stdout.read().decode("utf-8", errors="replace")
            err_s = stderr.read().decode("utf-8", errors="replace")
            rc = stdout.channel.recv_exit_status()
            txt = (out_s or err_s).strip()
            return (rc == 0), (txt.splitlines()[0] if txt else f"rc={rc}")
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    ok, detail = exec_ok("gcc --version")
    out.append(CheckResult("remote_gcc", ok, detail=detail, hint="install gcc on remote host" if not ok else ""))

    # Quick probe: can gcc accept rv64gcv flags (compile-only).
    ok2, detail2 = exec_ok("printf 'int main(){return 0;}' | gcc -x c - -c -o /tmp/intentir_env_check.o -march=rv64gcv 2>/dev/null && echo ok")
    out.append(
        CheckResult(
            "remote_rvv_march",
            bool(ok2),
            detail=str(detail2),
            hint="need gcc with -march=rv64gcv support on the RVV host" if not ok2 else "",
        )
    )

    client.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="treat optional components as required (fail if missing)")
    ap.add_argument("--remote-host", default=None)
    ap.add_argument("--remote-user", default="ubuntu")
    ap.add_argument("--remote-port", type=int, default=22)
    ap.add_argument("--use-key", action="store_true")
    ap.add_argument("--remote-password", default=None)
    args = ap.parse_args()

    print(f"platform: {platform.platform()}")
    print(f"cwd: {os.getcwd()}")

    required = True
    opt = bool(args.strict)

    checks: list[CheckResult] = []
    checks.append(_check_python())
    checks.append(_check_import("numpy", required=required, hint="pip install -r requirements/core.txt"))
    checks.append(_check_import("pytest", required=required, hint="pip install -r requirements/dev.txt"))
    checks.append(_check_import("requests", required=required, hint="pip install -r requirements/core.txt"))
    checks.append(_check_import("paramiko", required=required, hint="pip install -r requirements/core.txt"))

    # Optional pipelines.
    checks.append(_check_import("torch", required=opt, hint="install torch (CUDA build) or use requirements/gpu.txt"))
    checks.append(_check_import("triton", required=opt, hint="pip install triton (CUDA) or use requirements/gpu.txt"))
    checks.append(_check_import("tilelang", required=opt, hint="pip install tilelang (CUDA) or use requirements/gpu.txt"))

    # Build tools (required for C++ codegen).
    checks.append(_check_tool("cmake", ["cmake", "--version"], required=required, hint="install cmake"))
    checks.append(_check_tool("g++", ["g++", "--version"], required=required, hint="install g++"))
    checks.append(_check_tool("gcc", ["gcc", "--version"], required=required, hint="install gcc"))

    if args.remote_host:
        checks.extend(
            _check_remote_rvv(
                host=str(args.remote_host),
                user=str(args.remote_user),
                port=int(args.remote_port),
                use_key=bool(args.use_key),
                password=(None if bool(args.use_key) else (args.remote_password or os.getenv("INTENTIR_SSH_PASSWORD"))),
            )
        )

    ok_all = True
    for c in checks:
        status = "OK" if c.ok else "FAIL"
        print(f"[{status}] {c.name}: {c.detail}")
        if (not c.ok) and c.hint:
            print(f"  hint: {c.hint}")
        ok_all = ok_all and bool(c.ok)

    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
