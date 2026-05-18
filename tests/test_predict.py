import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.models.predict import preprocess_input, predict_salary
from src.utils.encoders import save_encoders
from src.utils.data_loader import load_config
from src.models.model_pipeline import save_model, save_scaler
from pathlib import Path


def create_dummy_artifacts(tmp_path):
    config = load_config("config/config.json")
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    X_train = pd.DataFrame({
        "job_title": [0, 1],
        "education_level": [0, 1],
        "industry": [0, 1],
        "company_size": [0, 1],
        "location": [0, 1],
        "remote_work": [0, 1],
        "experience_years": [1.0, 2.0],
        "skills_count": [3.0, 4.0],
        "certifications": [0.0, 1.0],
    })
    y_train = pd.Series(np.log1p([100.0, 200.0]))
    model.fit(X_train, y_train)

    scaler = StandardScaler()
    scaler.fit(X_train[config["features"]["scaled_columns"]])

    encoders = {
        "job_title": LabelEncoder().fit(["Data Scientist", "Data Analyst"]),
        "education_level": LabelEncoder().fit(["Bachelor", "Master"]),
        "industry": LabelEncoder().fit(["Technology", "Finance"]),
        "company_size": LabelEncoder().fit(["Small", "Large"]),
        "location": LabelEncoder().fit(["USA", "Canada"]),
        "remote_work": LabelEncoder().fit(["Yes", "No"]),
    }

    artifact_dir = tmp_path / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, str(artifact_dir / config["model"]["model_filename"]))
    save_scaler(scaler, str(artifact_dir / config["model"]["scaler_filename"]))
    save_scaler(scaler, str(artifact_dir / config["model"]["feature_scaler_filename"]))
    save_encoders(encoders, str(artifact_dir / config["model"]["encoder_filename"]))

    return artifact_dir, config


def test_preprocess_input(tmp_path, monkeypatch):
    artifact_dir, config = create_dummy_artifacts(tmp_path)
    monkeypatch.setattr("src.models.predict.load_config", lambda path: config)

    raw_input = {
        "job_title": "Data Scientist",
        "education_level": "Bachelor",
        "industry": "Technology",
        "company_size": "Small",
        "location": "USA",
        "remote_work": "Yes",
        "experience_years": 1.0,
        "skills_count": 3.0,
        "certifications": 0.0,
    }

    feature_scaler = StandardScaler().fit(pd.DataFrame(
        {
            "job_title": [0, 1],
            "education_level": [0, 1],
            "industry": [0, 1],
            "company_size": [0, 1],
            "location": [0, 1],
            "remote_work": [0, 1],
            "experience_years": [1.0, 2.0],
            "skills_count": [3.0, 4.0],
            "certifications": [0.0, 1.0],
        }
    ))
    artifacts = {
        "config": config,
        "model": None,
        "scaler": None,
        "feature_scaler": feature_scaler,
        "encoders": {
            "job_title": LabelEncoder().fit(["Data Scientist", "Data Analyst"]),
            "education_level": LabelEncoder().fit(["Bachelor", "Master"]),
            "industry": LabelEncoder().fit(["Technology", "Finance"]),
            "company_size": LabelEncoder().fit(["Small", "Large"]),
            "location": LabelEncoder().fit(["USA", "Canada"]),
            "remote_work": LabelEncoder().fit(["Yes", "No"]),
        },
    }

    processed = preprocess_input(raw_input, artifacts)
    assert processed.shape == (1, 9)
    assert set(processed.columns) == set(config["features"]["categorical_columns"] + config["features"]["numeric_columns"])


def test_predict_salary(tmp_path, monkeypatch):
    artifact_dir, config = create_dummy_artifacts(tmp_path)
    config["model"]["output_dir"] = str(artifact_dir)
    monkeypatch.setattr("src.models.predict.load_config", lambda path: config)

    raw_input = {
        "job_title": "Data Scientist",
        "education_level": "Bachelor",
        "industry": "Technology",
        "company_size": "Small",
        "location": "USA",
        "remote_work": "Yes",
        "experience_years": 1.0,
        "skills_count": 3.0,
        "certifications": 0.0,
    }

    # This test checks the prediction flow by loading actual artifacts.
    from src.models.predict import load_artifacts as load_artifacts_fn
    artifacts = load_artifacts_fn("config/config.json")

    prediction = predict_salary(raw_input, artifacts)
    assert prediction.shape == (1,)
    assert prediction[0] > 0
