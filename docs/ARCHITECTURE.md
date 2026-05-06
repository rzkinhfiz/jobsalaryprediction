# Project Architecture

## Project Overview

**Job Salary Prediction** adalah Machine Learning project untuk memprediksi gaji pekerjaan berdasarkan features seperti job title, education, experience, dan lainnya. Project menggunakan **Random Forest Regressor** dengan data preprocessing yang comprehensive.

## Folder Structure

```
dsproject_jobsalarypredict/
│
├── .github/                          # GitHub configuration
│   └── copilot-instructions.md      # Copilot setup instructions
│
├── data/                             # Data storage
│   ├── raw/                          # Raw input data
│   │   └── job_salary_prediction_dataset.csv
│   └── processed/                    # Processed data (after preprocessing)
│
├── notebooks/                        # Jupyter notebooks
│   └── job_salary_prediction.ipynb   # Main EDA & modeling notebook
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── models/                       # Model training & prediction
│   │   ├── __init__.py
│   │   ├── train.py                 # Model training script
│   │   ├── predict.py               # Model inference script
│   │   └── evaluate.py              # Model evaluation
│   ├── features/                     # Feature engineering
│   │   ├── __init__.py
│   │   └── preprocessing.py         # Data preprocessing functions
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── helpers.py               # Helper functions
│       └── constants.py             # Constants & configurations
│
├── models/                           # Saved trained models
│   ├── random_forest_salary_model.pkl    # Trained Random Forest
│   ├── scaler.pkl                        # StandardScaler (if saved)
│   ├── le_job_title.pkl                 # LabelEncoder for job_title
│   ├── le_education.pkl                 # LabelEncoder for education
│   └── le_remote_status.pkl             # LabelEncoder for remote_status
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py         # Test data preprocessing
│   ├── test_model.py                 # Test model training/prediction
│   └── test_utils.py                 # Test utility functions
│
├── docs/                             # Documentation
│   ├── MODEL.md                      # Model documentation
│   ├── DATA_DICTIONARY.md            # Feature descriptions
│   ├── SETUP.md                      # Installation guide
│   ├── USAGE.md                      # How to use model
│   ├── RESULTS.md                    # Model performance results
│   └── ARCHITECTURE.md               # This file
│
├── config/                           # Configuration files
│   ├── config.yaml                   # Model hyperparameters
│   └── paths.yaml                    # File paths configuration
│
├── .gitignore                        # Git ignore rules
├── README.md                         # Project overview
├── requirements.txt                  # Python dependencies
└── setup.py (optional)               # Package setup

```

## Data Pipeline

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT DATA                            │
│              (job_salary_prediction_dataset.csv)             │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    DATA EXPLORATION (EDA)                   │
│  - Summary statistics                                       │
│  - Distribution analysis                                    │
│  - Missing value detection                                  │
│  - Correlation analysis                                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   DATA PREPROCESSING                        │
│  - Handle missing values                                    │
│  - Outlier detection & handling                             │
│  - Log transformation (salary)                              │
│  - Train-test split (80-20)                                 │
└────────────────────────────┬────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
        ┌─────────────────┐      ┌─────────────────┐
        │   TRAIN DATA    │      │   TEST DATA     │
        │  (80% records)  │      │  (20% records)  │
        └────────┬────────┘      └────────┬────────┘
                 │                        │
                 │ (Fit)                  │ (Transform only)
                 ▼                        ▼
        ┌─────────────────┐      ┌─────────────────┐
        │  CATEGORICAL    │      │  CATEGORICAL    │
        │  ENCODING       │      │  ENCODING       │
        │ (LabelEncoder)  │      │                 │
        └────────┬────────┘      └────────┬────────┘
                 │                        │
                 │ (Fit)                  │ (Transform only)
                 ▼                        ▼
        ┌─────────────────┐      ┌─────────────────┐
        │  FEATURE        │      │  FEATURE        │
        │  SCALING        │      │  SCALING        │
        │(StandardScaler) │      │                 │
        └────────┬────────┘      └────────┬────────┘
                 │                        │
        ┌────────▼────────┐      ┌────────▼────────┐
        │  X_train_scaled │      │  X_test_scaled  │
        │  y_train        │      │  y_test         │
        └────────┬────────┘      └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   MODEL TRAINING│
        │   (RF, n=100)   │
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │ TRAINED MODEL   │
        │   (saved.pkl)   │
        └────────┬────────┘
                 │
                 │ (Predict)
                 ▼
        ┌─────────────────────────┐        ┌─────────────────────────┐
        │  PREDICTIONS (log-scale)│───────▶│ BACK-TRANSFORM (expm1)  │
        └────────┬────────────────┘        └────────┬────────────────┘
                 │                                   │
                 ▼                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │         MODEL EVALUATION                            │
        │  - MAE, MSE, MAPE, R²                               │
        │  - Prediction vs Actual plots                       │
        │  - Residual analysis                                │
        │  - Feature importance                               │
        └─────────────────────────────────────────────────────┘

```

## Code Workflow

### 1. Data Loading & Exploration
```python
# Location: notebooks/job_salary_prediction.ipynb (Cell 1-10)
df = pd.read_csv('data/raw/job_salary_prediction_dataset.csv')
# Check: shape, dtypes, missing values, descriptive stats
```

### 2. Data Preprocessing
```python
# Location: notebooks/job_salary_prediction.ipynb (Cell 11-20)
# - Categorical encoding
# - Outlier handling (log transformation)
# - Feature scaling (after train-test split)
```

### 3. Model Training
```python
# Location: notebooks/job_salary_prediction.ipynb (Cell 21-25)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
```

### 4. Model Evaluation
```python
# Location: notebooks/job_salary_prediction.ipynb (Cell 26-30)
y_pred = model.predict(X_test_scaled)
# Calculate metrics: MAE, MSE, MAPE, R²
# Generate plots: scatter, residual
```

### 5. Model Persistence
```python
# Location: notebooks/job_salary_prediction.ipynb (Cell 31)
import joblib
joblib.dump(model, 'models/random_forest_salary_model.pkl')
```

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.8+ | Primary programming language |
| **Data Handling** | Pandas, NumPy | Data manipulation & computation |
| **ML Framework** | Scikit-learn | Model training & evaluation |
| **Visualization** | Matplotlib, Seaborn | Data visualization |
| **Notebooks** | Jupyter | Interactive analysis |
| **Serialization** | Joblib | Model & scaler persistence |
| **Testing** | pytest | Unit testing framework |
| **Version Control** | Git | Code versioning |

## Configuration Management

### Config Files Location
- `config/config.yaml` - Model hyperparameters
- `config/paths.yaml` - File paths
- `.env` - Environment variables (optional)

Example `config.yaml`:
```yaml
model:
  type: RandomForest
  n_estimators: 100
  max_depth: null
  random_state: 42

preprocessing:
  test_size: 0.2
  random_state: 42
  scale_features: true

paths:
  raw_data: data/raw/
  processed_data: data/processed/
  models: models/
```

## Dependencies Management

### requirements.txt
- Core dependencies untuk running notebook & training
- Installed via: `pip install -r requirements.txt`

### requirements-dev.txt (Optional)
- Development dependencies: testing, linting, formatting
- Installed via: `pip install -r requirements-dev.txt`

## Model Versioning

### Version Control Strategy
```
models/
├── random_forest_salary_model_v1.0.pkl  # Initial version
├── random_forest_salary_model_v1.1.pkl  # Minor improvements
├── random_forest_salary_model_v2.0.pkl  # Major improvements
└── random_forest_salary_model_CURRENT.pkl  # Symlink to active model
```

### Metadata Tracking
Store model info dalam `models/MANIFEST.json`:
```json
{
  "model_id": "rf_salary_v1.0",
  "date_trained": "2024-05-06",
  "metrics": {"mae": 5000, "r2": 0.82},
  "features": ["job_title", "experience", ...],
  "preprocessors": ["scaler.pkl", "le_job_title.pkl", ...]
}
```

## Deployment Architecture

### Local Development
```
Developer Machine
├── Jupyter Notebook (for experimentation)
├── Unit tests (pytest)
└── Model evaluation scripts
```

### Production Deployment (Optional)
```
Production Server
├── FastAPI/Flask app
├── Trained model loaded
├── Inference endpoint
└── Monitoring & logging
```

## Error Handling & Logging

### Logging Structure
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Training started")
logger.warning("Feature X has high missing rate")
logger.error("Model training failed")
```

### Exception Handling
- Input validation errors
- Model loading errors
- Prediction errors
- Data preprocessing errors

## Performance Optimization

### Memory Optimization
- Use data types efficiently (int32 vs int64)
- Delete unnecessary objects
- Use generators for large datasets

### Speed Optimization
- Vectorized operations (NumPy/Pandas)
- Parallel processing (n_jobs=-1 in sklearn)
- Model caching

## Monitoring & Maintenance

### Production Monitoring
- Prediction latency tracking
- Model accuracy drift detection
- Input data distribution changes
- Error rate monitoring

### Retraining Strategy
- **Trigger**: When R² drops below threshold
- **Frequency**: Monthly or when new data available
- **Process**: Retrain on combined historical + new data

## Security Considerations

- ✓ Don't commit sensitive data
- ✓ Use `.gitignore` for data & models
- ✓ Validate user inputs before prediction
- ✓ Implement access control for model API

## Documentation Standards

- Code comments untuk complex logic
- Docstrings untuk functions
- README untuk project overview
- Architecture docs untuk system design
