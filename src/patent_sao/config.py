from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    root = config_path.parent.parent
    config["_root"] = root
    for key, value in config.get("paths", {}).items():
        config["paths"][key] = str((root / value).resolve())
    return config


def ensure_parent(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output
