"""
Render and install workflow/flaggems nightly systemd unit files.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _render(template: str, *, user: str, repo_root: Path) -> str:
    return (
        template.replace("{{USER}}", str(user))
        .replace("{{REPO_ROOT}}", str(repo_root))
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--service-template",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "flaggems-nightly.service.example"),
    )
    ap.add_argument(
        "--timer-template",
        type=Path,
        default=(ROOT / "workflow" / "flaggems" / "flaggems-nightly.timer.example"),
    )
    ap.add_argument("--repo-root", type=Path, default=ROOT)
    ap.add_argument("--user", default=os.getenv("USER", "unknown"))
    ap.add_argument(
        "--systemd-dir",
        type=Path,
        default=(Path.home() / ".config" / "systemd" / "user"),
        help="Target systemd unit directory (default: ~/.config/systemd/user).",
    )
    ap.add_argument("--service-name", default="flaggems-nightly.service")
    ap.add_argument("--timer-name", default="flaggems-nightly.timer")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    service_src = _read(args.service_template)
    timer_src = _read(args.timer_template)
    service_rendered = _render(service_src, user=str(args.user), repo_root=Path(args.repo_root))
    timer_rendered = _render(timer_src, user=str(args.user), repo_root=Path(args.repo_root))

    out_dir = Path(args.systemd_dir)
    service_out = out_dir / str(args.service_name)
    timer_out = out_dir / str(args.timer_name)
    if bool(args.dry_run):
        print(f"[dry-run] would write: {service_out}")
        print(f"[dry-run] would write: {timer_out}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    service_out.write_text(service_rendered, encoding="utf-8")
    timer_out.write_text(timer_rendered, encoding="utf-8")
    print(f"Rendered: {service_out}")
    print(f"Rendered: {timer_out}")
    print("Next steps:")
    print("  systemctl --user daemon-reload")
    print(f"  systemctl --user enable --now {args.timer_name}")
    print(f"  systemctl --user status {args.timer_name}")


if __name__ == "__main__":
    main()
