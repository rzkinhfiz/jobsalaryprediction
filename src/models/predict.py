from typing import Dict
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.data_loader import load_config
from src.models.model_pipeline import load_model, load_scaler
from src.utils.encoders import load_encoders


def load_artifacts(config_path: str) -> Dict[str, object]:
    config = load_config(config_path)
    model_path = Path(config["model"]["output_dir"]) / config["model"]["model_filename"]
    scaler_path = Path(config["model"]["output_dir"]) / config["model"]["scaler_filename"]
    feature_scaler_path = Path(config["model"]["output_dir"]) / config["model"]["feature_scaler_filename"]
    encoder_path = Path(config["model"]["output_dir"]) / config["model"]["encoder_filename"]

    return {
        "config": config,
        "model": load_model(str(model_path)),
        "scaler": load_scaler(str(scaler_path)),
        "feature_scaler": load_scaler(str(feature_scaler_path)),
        "encoders": load_encoders(str(encoder_path)),
    }


def preprocess_input(raw_input: Dict[str, object], artifacts: Dict[str, object]) -> pd.DataFrame:
    config = artifacts["config"]
    categorical_cols = config["features"]["categorical_columns"]
    numeric_cols = config["features"]["numeric_columns"]

    input_df = pd.DataFrame([raw_input])
    required_cols = categorical_cols + numeric_cols
    missing_cols = [col for col in required_cols if col not in input_df.columns]
    if missing_cols:
        raise ValueError(f"Input masih kurang kolom: {missing_cols}")

    encoders = artifacts["encoders"]
    for col in categorical_cols:
        encoder = encoders.get(col)
        if not encoder:
            raise ValueError(f"Encoder untuk kolom '{col}' tidak ditemukan")
        input_df[col] = encoder.transform(input_df[col].astype(str))

    input_df[numeric_cols] = input_df[numeric_cols].astype(float)
    scaled_columns = config["features"]["scaled_columns"]
    input_df[scaled_columns] = artifacts["feature_scaler"].transform(input_df[scaled_columns])

    ordered_columns = categorical_cols + numeric_cols
    return input_df[ordered_columns]


def predict_salary(raw_input: Dict[str, object], artifacts: Dict[str, object]) -> np.ndarray:
    processed = preprocess_input(raw_input, artifacts)
    y_pred_log = artifacts["model"].predict(processed)
    return np.expm1(y_pred_log)
