import os
from pathlib import Path
from typing import Union

import yaml

# Default config lives next to this file: utils/config/Config.yaml.
# Override with the NASMESHLIB_CONFIG environment variable.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "Config.yaml"


def load_config(config_path: Union[str, Path, None] = None) -> dict:
    """Load the YAML config and resolve every '*dir*' / '*output*' entry to an absolute path.

    A missing or empty config file returns an empty dict, so callers fall back to their defaults.
    """
    config_path = Path(config_path or os.environ.get("NASMESHLIB_CONFIG", DEFAULT_CONFIG_PATH)).resolve()
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    for k, v in config.items():
        if isinstance(v, str) and ("dir" in k or "output" in k):
            config[k] = str(Path(v).resolve())

    return config
