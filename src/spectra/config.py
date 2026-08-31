"""Carregamento centralizado de configuração e caminhos do projeto."""

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Carrega o YAML do projeto ou um arquivo explicitamente informado."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
