# Usage Guide

## Load Trained Model and Artifacts

```python
from src.models.predict import load_artifacts

artifacts = load_artifacts('config/config.json')
```

This loads:
- `model`: trained Random Forest model
- `feature_scaler`: scaler for encoded + numeric features
- `encoders`: label encoders for categorical columns
- `config`: project metadata and paths

## Single Prediction

```python
raw_input = {
    'job_title': 'Data Scientist',
    'education_level': 'Master',
    'industry': 'Technology',
    'company_size': 'Large',
    'location': 'USA',
    'remote_work': 'Yes',
    'experience_years': 5,
    'skills_count': 6,
    'certifications': 2,
}

salary_prediction = predict_salary(raw_input, artifacts)[0]
print(f"Prediksi gaji: Rp {salary_prediction:,.0f}")
```

## Batch Predictions

```python
import pandas as pd
from src.models.predict import load_artifacts, predict_salary

artifacts = load_artifacts('config/config.json')

batch_df = pd.read_csv('data/predictions_batch.csv')
results = []
for _, row in batch_df.iterrows():
    raw_input = row.to_dict()
    results.append(predict_salary(raw_input, artifacts)[0])

batch_df['predicted_salary'] = results
batch_df.to_csv('data/predictions_results.csv', index=False)
```

## Direct Artifact Loading

If you need lower-level control:

```python
import joblib

model = joblib.load('models/random_forest_salary_model.pkl')
feature_scaler = joblib.load('models/feature_scaler.pkl')
encoders = joblib.load('models/label_encoders.pkl')
```

## Preprocessing Steps for Inference

1. Convert categorical columns using the same label encoders:
   - `job_title`
   - `education_level`
   - `industry`
   - `company_size`
   - `location`
   - `remote_work`
2. Scale all model features using `feature_scaler`.
3. Predict with the trained model.
4. Back-transform the output with `np.expm1()`.

## Example with Direct Preprocessing

```python
import joblib
import numpy as np
import pandas as pd

model = joblib.load('models/random_forest_salary_model.pkl')
feature_scaler = joblib.load('models/feature_scaler.pkl')
encoders = joblib.load('models/label_encoders.pkl')

raw_input = {
    'job_title': 'Data Scientist',
    'education_level': 'Master',
    'industry': 'Technology',
    'company_size': 'Large',
    'location': 'USA',
    'remote_work': 'Yes',
    'experience_years': 5,
    'skills_count': 6,
    'certifications': 2,
}

input_df = pd.DataFrame([raw_input])
for col, encoder in encoders.items():
    input_df[col] = encoder.transform(input_df[col].astype(str))

input_scaled = feature_scaler.transform(input_df)
y_pred_log = model.predict(input_scaled)
salary = np.expm1(y_pred_log)[0]
print(f"Prediksi gaji: Rp {salary:,.0f}")
```

## Notes

- Use `load_artifacts('config/config.json')` to ensure inference uses the same configuration as training.
- Use `np.expm1()` to convert log-scale predictions back to the original salary scale.
- If new categories appear in production, the label encoders must be retrained or updated.

## Troubleshooting

### Error: "Model and scaler dimensions don't match"
- Pastikan feature order sama saat training dan prediction
- Check feature names: `model.n_features_in_`

### Error: "LabelEncoder classes mismatch"
- Reuse exact encoder yang digunakan saat training
- Jangan fit encoder baru pada data baru

### Prediction unreasonable values
- Verify input range: pastikan dalam normal range dari training data
- Check scaling: pastikan scaler fitted dengan training data
- Inspect features: pastikan categorical values valid
