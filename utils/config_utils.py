import yaml
from pathlib import Path
from typing import Union
config_path = r"<local-path>"
def load_config(config_path: Union[str,Path] = config_path) -> dict:
    """
    YAML dosyasını yükler ve içindeki tüm 'dir' ve 'output' anahtarlarını mutlak yola çevirir.
    """

    config_path = Path(config_path).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for k, v in config.items():
        if isinstance(v,str) and ("dir" in k or "output" in k):
            config[k] = str(Path(v).resolve())
            
    return config