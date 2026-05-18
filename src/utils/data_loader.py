import json
from pathlib import Path
from typing import Dict
import pandas as pd


def load_config(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file tidak ditemukan: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_raw_data(config: Dict) -> pd.DataFrame:
    raw_path = Path(config["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Data CSV tidak ditemukan: {raw_path}")

    df = pd.read_csv(raw_path)
    return df


def save_dataframe(df: pd.DataFrame, path: str, index: bool = False) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)
