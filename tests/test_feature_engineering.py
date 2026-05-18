import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.features.feature_engineering import (
    log_transform_target,
    encode_categorical_columns,
    split_features_target,
    scale_numeric_features,
)


def test_log_transform_target():
    df = pd.DataFrame({"salary": [100.0, 200.0, 300.0]})
    df_transformed = log_transform_target(df.copy(), "salary")

    assert "salary_log" in df_transformed.columns
    assert df_transformed["salary_log"].iloc[0] == np.log1p(100.0)


def test_encode_categorical_columns():
    df = pd.DataFrame({"job_title": ["Data Scientist", "Data Analyst"]})
    df_encoded, encoders = encode_categorical_columns(df.copy(), ["job_title"])

    assert "job_title" in encoders
    assert df_encoded["job_title"].dtype == int
    assert set(df_encoded["job_title"].tolist()) == {0, 1}


def test_split_features_target():
    df = pd.DataFrame({"experience_years": [1, 2], "salary_log": [4.0, 5.0]})
    X, y = split_features_target(df, "salary_log")

    assert "salary_log" not in X.columns
    assert y.tolist() == [4.0, 5.0]


def test_scale_numeric_features():
    X_train = pd.DataFrame({"experience_years": [1.0, 2.0], "skills_count": [3.0, 4.0]})
    X_test = pd.DataFrame({"experience_years": [5.0], "skills_count": [6.0]})
    X_train_scaled, X_test_scaled, scaler = scale_numeric_features(X_train, X_test, ["experience_years", "skills_count"])

    assert isinstance(scaler, StandardScaler)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
