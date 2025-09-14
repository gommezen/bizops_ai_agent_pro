import datetime as dt
from pathlib import Path

import yaml


def utc_timestamp() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def new_run_dir(base="data/runs") -> Path:
    p = Path(base) / utc_timestamp()
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path="configs/config.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8"))
