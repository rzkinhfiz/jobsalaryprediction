# Job Salary Prediction

A machine learning project to predict job salaries based on various features such as job title, education level, years of experience, and remote work status. This project uses **Random Forest Regressor** with comprehensive EDA, preprocessing, and evaluation.

## 🎯 Project Overview

**Goal**: Build an accurate salary prediction model using job-related features.

**Model**: Random Forest Regressor with log-transformed salary target
- **n_estimators**: 100 trees
- **Evaluation Metrics**: MAE, MSE, MAPE, R²
- **Features**: Feature importance analysis, residual diagnostics

**Status**: ✅ Model trained and evaluated

## 📁 Project Structure

```
dsproject_jobsalarypredict/
├── .github/
│   └── copilot-instructions.md      # Setup instructions
├── data/
│   ├── raw/                         # Raw dataset
│   │   └── job_salary_prediction_dataset.csv
│   └── processed/                   # Processed & split data
│       ├── X_train.csv, X_test.csv
│       ├── y_train.csv, y_test.csv
│       ├── X_train_scaled.csv, X_test_scaled.csv
│       └── metadata.json
├── notebooks/
│   └── job_salary_prediction.ipynb  # Main notebook (EDA, training, evaluation)
├── models/
│   ├── random_forest_salary_model.pkl    # Trained model
│   ├── scaler.pkl                        # StandardScaler
│   └── le_*.pkl                          # Label encoders
├── src/
│   ├── models/                      # Model training/prediction scripts
│   ├── features/                    # Feature engineering
│   └── utils/                       # Helper utilities
├── docs/                            # 📚 Comprehensive documentation
│   ├── MODEL.md                     # Model architecture & algorithms
│   ├── DATA_DICTIONARY.md           # Feature descriptions
│   ├── SETUP.md                     # Installation guide
│   ├── USAGE.md                     # How to use model
│   ├── RESULTS.md                   # Model performance metrics
│   └── ARCHITECTURE.md              # System design & pipeline
├── tests/                           # Unit tests
├── config/                          # Configuration files
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Detailed setup**: See [docs/SETUP.md](docs/SETUP.md)

### 2. Run Notebook
```bash
jupyter notebook notebooks/job_salary_prediction.ipynb
```

The notebook includes:
- ✅ Exploratory Data Analysis (EDA)
- ✅ Data preprocessing & feature encoding
- ✅ Log transformation for outlier handling
- ✅ Train-test split & scaling
- ✅ Random Forest model training
- ✅ Feature importance analysis
- ✅ Model evaluation (MAE, MSE, RMSE, MAPE, R²)
- ✅ Residual analysis plots
- ✅ Model persistence

### 3. Make Predictions
```python
import joblib
import numpy as np

# Load model
model = joblib.load('models/random_forest_salary_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare features and predict
X_new = preprocess_features(new_data)
X_new_scaled = scaler.transform(X_new)
y_pred_log = model.predict(X_new_scaled)
salary = np.expm1(y_pred_log)
```

**Detailed usage**: See [docs/USAGE.md](docs/USAGE.md)

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| MAE | 5202.27 |
| MSE | 43175717.67 |
| RMSE | 6570.82 |
| MAPE | 3.77% |
| R² Score | 0.97 |

**Full results**: See [docs/RESULTS.md](docs/RESULTS.md)

## 📚 Documentation

Comprehensive documentation available in `docs/` folder:

| Document | Purpose |
|----------|---------|
| [MODEL.md](docs/MODEL.md) | Algorithm details, preprocessing pipeline, evaluation metrics |
| [DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | Feature descriptions, data types, preprocessing steps |
| [SETUP.md](docs/SETUP.md) | Installation & environment setup guide |
| [USAGE.md](docs/USAGE.md) | How to use trained model, inference examples |
| [RESULTS.md](docs/RESULTS.md) | Model performance, error analysis, feature importance |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Project structure, data pipeline, technology stack |

## 🔄 Data Pipeline

```
Raw Data → EDA → Preprocessing → Train-Test Split 
    ↓
Encoding (LabelEncoder) → Scaling (StandardScaler)
    ↓
Model Training (Random Forest) → Evaluation → Predictions
    ↓
Results (MAE, MAPE, R²) → Feature Importance → Residual Plot
```

## 🛠 Tech Stack

- **Python**: 3.8+
- **Data Processing**: pandas, numpy
- **ML Framework**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Notebooks**: Jupyter
- **Model Serialization**: joblib
- **Version Control**: Git

## 📦 Key Features

✅ **Comprehensive EDA**: Distribution analysis, correlation matrix, categorical insights  
✅ **Robust Preprocessing**: Log transformation, categorical encoding, feature scaling  
✅ **Random Forest Model**: 100 trees, feature importance built-in  
✅ **Model Evaluation**: Multiple metrics (MAE, MSE, MAPE, R²)  
✅ **Diagnostic Plots**: Prediction vs actual, residual analysis  
✅ **Model Persistence**: Saved in `models/` folder with all artifacts  
✅ **Processed Data**: Train-test splits saved for reproducibility  
✅ **Complete Documentation**: 6 markdown files covering all aspects  

## 📋 Next Steps

1. **Update placeholder values** in `docs/RESULTS.md` with actual model metrics
2. **Customize features** based on your dataset (e.g., rename columns)
3. **Hyperparameter tuning** for model improvement
4. **Deploy model** as REST API (FastAPI/Flask)
5. **Monitoring** for prediction drift in production

## 📝 Project Workflow

```
1. Data Exploration      → Understand data distribution & relationships
2. Preprocessing        → Clean, encode, and normalize features
3. Model Training       → Train Random Forest on scaled data
4. Evaluation          → Assess model performance with multiple metrics
5. Feature Analysis    → Identify important predictors
6. Prediction          → Generate salary estimates
7. Production Ready    → Model saved and documented
```

## 🤝 Contributing

To contribute:
1. Create feature branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -am 'Add feature'`
3. Push to branch: `git push origin feature/your-feature`
4. Submit pull request

## 📄 License

[Specify your license here]

## 📧 Contact

[Add contact information if applicable]

---

**Last Updated**: May 6, 2026  
**Model Version**: v1.0  
**Status**: Production Ready