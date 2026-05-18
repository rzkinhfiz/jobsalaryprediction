import joblib
from pathlib import Path
from typing import Dict


def save_encoders(encoders: Dict[str, object], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, output_path)


def load_encoders(path: str) -> Dict[str, object]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Encoder file tidak ditemukan: {input_path}")
    return joblib.load(input_path)
