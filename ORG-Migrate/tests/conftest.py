from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORG_ROOT = ROOT / "ORG-Migrate"

for path in (ORG_ROOT, ROOT):
    sp = str(path)
    if sp not in sys.path:
        sys.path.insert(0, sp)
