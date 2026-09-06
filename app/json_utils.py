from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    """Load a JSON file while tolerating UTF-8 BOM-prefixed content."""
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf-8-sig")
    return json.loads(raw_text)
