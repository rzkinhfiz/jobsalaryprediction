import json
import pandas as pd
from pathlib import Path
from src.utils.data_loader import load_config, load_raw_data, save_dataframe


def test_load_config(tmp_path):
    config_file = tmp_path / "config.json"
    config_data = {"data": {"raw_path": "data/raw/job_salary_prediction_dataset.csv"}}
    config_file.write_text(json.dumps(config_data), encoding="utf-8")

    loaded = load_config(str(config_file))
    assert loaded["data"]["raw_path"] == "data/raw/job_salary_prediction_dataset.csv"


def test_save_dataframe(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3]})
    output_file = tmp_path / "output.csv"
    save_dataframe(df, str(output_file))

    assert output_file.exists()
    loaded = pd.read_csv(output_file)
    assert loaded.shape == (3, 1)
    assert loaded["a"].tolist() == [1, 2, 3]
