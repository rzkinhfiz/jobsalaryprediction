from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def log_transform_target(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    if (df[target_column] <= 0).any():
        raise ValueError("Semua nilai target harus positif untuk log transformasi.")

    df[f"{target_column}_log"] = np.log1p(df[target_column])
    return df


def encode_categorical_columns(df: pd.DataFrame, categorical_columns: List[str]) -> Tuple[pd.DataFrame, dict]:
    encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder
    return df, encoders


def split_features_target(df: pd.DataFrame, target_column: str, drop_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    drop_columns = drop_columns or []
    X = df.drop(columns=[target_column] + drop_columns)
    y = df[target_column].copy()
    return X, y


def scale_numeric_features(X_train: pd.DataFrame, X_test: pd.DataFrame, numeric_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

    return X_train_scaled, X_test_scaled, scaler


def scale_feature_columns(X_train: pd.DataFrame, X_test: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[feature_columns] = scaler.fit_transform(X_train[feature_columns])
    X_test_scaled[feature_columns] = scaler.transform(X_test[feature_columns])

    return X_train_scaled, X_test_scaled, scaler
