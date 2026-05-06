# Usage Guide

## Loading & Using Trained Model

### Method 1: Direct Model Loading

```python
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load('models/random_forest_salary_model.pkl')

# Load scaler (jika ada)
scaler = joblib.load('models/scaler.pkl')  # If saved separately
```

### Method 2: Using Wrapper Function

```python
from src.models.predict import load_model, predict_salary

model, scaler = load_model()
prediction = predict_salary(features, model, scaler)
```

## Making Predictions

### Single Prediction

```python
# Define input features
input_data = {
    'job_title': 'Data Scientist',
    'education': 'Master',
    'experience': 5,
    'remote_status': 'Remote'
}

# Convert to DataFrame
df_input = pd.DataFrame([input_data])

# Preprocess (encoding + scaling)
# 1. Apply same LabelEncoder yang digunakan training
# 2. Scale menggunakan fitted scaler

X_scaled = scaler.transform(X_processed)

# Predict
y_pred_log = model.predict(X_scaled)

# Back-transform ke skala asli
salary_prediction = np.expm1(y_pred_log)[0]

print(f"Prediksi Gaji: Rp {salary_prediction:,.0f}")
```

### Batch Predictions

```python
import pandas as pd

# Load multiple records
df_batch = pd.read_csv('data/predictions_batch.csv')

# Preprocess all records
X_batch = preprocess_features(df_batch)
X_batch_scaled = scaler.transform(X_batch)

# Predict
y_pred_log_batch = model.predict(X_batch_scaled)
y_pred_batch = np.expm1(y_pred_log_batch)

# Add predictions to DataFrame
df_batch['predicted_salary'] = y_pred_batch

# Save results
df_batch.to_csv('data/predictions_results.csv', index=False)
```

## Feature Encoding Reference

### Categorical Features

Jika Anda memiliki LabelEncoder files:

```python
import joblib

# Load encoders
le_job_title = joblib.load('models/le_job_title.pkl')
le_education = joblib.load('models/le_education.pkl')
le_remote = joblib.load('models/le_remote_status.pkl')

# Encode categorical features
job_encoded = le_job_title.transform(['Data Scientist'])[0]
education_encoded = le_education.transform(['Master'])[0]
remote_encoded = le_remote.transform(['Remote'])[0]
```

### Numeric Features
Langsung gunakan nilai numerik tanpa transformasi (kecuali scaling).

## Complete Example

```python
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ===== LOAD MODEL & ARTIFACTS =====
model = joblib.load('models/random_forest_salary_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# ===== PREPARE INPUT =====
input_features = {
    'job_title': 'Machine Learning Engineer',
    'education': 'Bachelor',
    'experience': 3,
    'remote_status': 'Hybrid',
    'age': 28
}

# Convert to DataFrame
df = pd.DataFrame([input_features])

# ===== ENCODE CATEGORICAL FEATURES =====
le_job = joblib.load('models/le_job_title.pkl')
le_edu = joblib.load('models/le_education.pkl')
le_rem = joblib.load('models/le_remote_status.pkl')

df['job_title'] = le_job.transform(df['job_title'])
df['education'] = le_edu.transform(df['education'])
df['remote_status'] = le_rem.transform(df['remote_status'])

# ===== SCALE FEATURES =====
X_scaled = scaler.transform(df)

# ===== PREDICT =====
y_pred_log = model.predict(X_scaled)
salary = np.expm1(y_pred_log)[0]

print(f"Predicted Salary: Rp {salary:,.0f}")
print(f"Log-scale prediction: {y_pred_log[0]:.4f}")
```

## Confidence Intervals (Advanced)

Menggunakan Random Forest untuk estimasi uncertainty:

```python
# Get predictions dari individual trees
predictions_individual = np.array([
    tree.predict(X_scaled) for tree in model.estimators_
])

# Hitung confidence intervals (95%)
mean_pred = predictions_individual.mean(axis=0)
std_pred = predictions_individual.std(axis=0)

lower_bound = np.expm1(mean_pred - 1.96 * std_pred)
upper_bound = np.expm1(mean_pred + 1.96 * std_pred)

print(f"Prediction: Rp {salary:,.0f}")
print(f"95% CI: [Rp {lower_bound:,.0f} - Rp {upper_bound:,.0f}]")
```

## Feature Importance

Melihat features yang paling berpengaruh:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Get feature importances
importances = model.feature_importances_
feature_names = ['job_title', 'education', 'experience', 'remote_status', 'age']

# Create DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance Score')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

print(importance_df)
```

## API Endpoint (FastAPI Example)

```python
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model on startup
model = joblib.load('models/random_forest_salary_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.post("/predict")
async def predict(features: dict):
    """
    Input: {
        'job_title': 'Data Scientist',
        'education': 'Master',
        'experience': 5
    }
    """
    # Process features
    X = np.array([list(features.values())])
    X_scaled = scaler.transform(X)
    
    # Predict
    y_pred_log = model.predict(X_scaled)
    salary = float(np.expm1(y_pred_log)[0])
    
    return {"predicted_salary": salary}
```

Run dengan: `uvicorn app:app --reload`

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
