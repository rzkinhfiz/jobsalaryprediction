import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.models.model_pipeline import train_random_forest, save_model, load_model


def test_train_random_forest(tmp_path):
    X_train = pd.DataFrame({"feature_1": [1.0, 2.0, 3.0], "feature_2": [4.0, 5.0, 6.0]})
    y_train = pd.Series([10.0, 20.0, 30.0])
    model = train_random_forest(X_train, y_train, n_estimators=10, random_state=42)

    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 10
    assert model.score(X_train, y_train) >= 0.0


def test_save_and_load_model(tmp_path):
    X_train = pd.DataFrame({"feature_1": [1.0, 2.0, 3.0], "feature_2": [4.0, 5.0, 6.0]})
    y_train = pd.Series([10.0, 20.0, 30.0])
    model = train_random_forest(X_train, y_train, n_estimators=10, random_state=42)

    model_path = tmp_path / "model.pkl"
    save_model(model, str(model_path))

    loaded_model = load_model(str(model_path))
    assert isinstance(loaded_model, RandomForestRegressor)
    assert loaded_model.predict(X_train).shape == (3,)
