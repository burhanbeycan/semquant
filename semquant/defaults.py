from __future__ import annotations

from importlib import resources
from typing import Any, Dict

import yaml


def load_default_config() -> Dict[str, Any]:
    """Load the built-in default YAML config shipped with the package."""
    with resources.files("semquant.configs").joinpath("default.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
