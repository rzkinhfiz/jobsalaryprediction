from pathlib import Path
from typing import Tuple
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int = 100, random_state: int = 42) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def save_model(model, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(path: str):
    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    return joblib.load(model_path)


def save_scaler(scaler, path: str) -> None:
    save_model(scaler, path)


def load_scaler(path: str):
    return load_model(path)


def save_feature_importance(model, feature_names: list, path: str) -> None:
    importance_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    importance_df = importance_df.sort_values(by="importance", ascending=False)
    save_model(importance_df, path)
