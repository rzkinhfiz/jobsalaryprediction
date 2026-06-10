# Job Salary Prediction

A machine learning project to predict job salaries based on various features such as job title, education level, years of experience, and remote work status. This project uses **Random Forest Regressor** with comprehensive EDA, preprocessing, and evaluation.

## 🎯 Project Overview

**Goal**: Build an accurate salary prediction model using job-related features.

**Model**: Random Forest Regressor with log-transformed salary target
- **n_estimators**: 50 trees
- **max_depth**: 20
- **min_samples_leaf**: 5
- **Evaluation Metrics**: MAE, MSE, MAPE, R²
- **Features**: Feature importance analysis, residual diagnostics

**Live Demo**: https://jobsalaryprediction-rzkinhfizproject.streamlit.app/

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
│   ├── feature_scaler.pkl                # StandardScaler artifact
│   └── label_encoders.pkl                # Label encoders
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

### 2. Run Notebook or Launch App
```bash
jupyter notebook notebooks/job_salary_prediction.ipynb
```

Or run the Streamlit app:
```bash
streamlit run app.py
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
from src.models.predict import load_artifacts, predict_salary

artifacts = load_artifacts('config/config.json')
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
print(f"Prediksi gaji: $ {salary_prediction:,.0f}")
```

**Detailed usage**: See [docs/USAGE.md](docs/USAGE.md)

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| MAE | 5462.14 |
| MSE | 47893565.64 |
| RMSE | 6920.52 |
| MAPE | 3.94% |
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

1. **Review model metrics** and verify them against notebook results
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

Created by Rizki Nurhafizd Achmad

## 📧 Contact

| Platform | Tautan |
| :--- | :--- |
| **LinkedIn** | [rizki-nurhafizd](https://www.linkedin.com/in/rizki-nurhafizd/) |
| **GitHub** | [@rzkinhfiz](https://github.com/rzkinhfiz) |
| **Web Portfolio** | [My Website](https://v0-rzkinhfiz-porto.vercel.app/) |

---

**Last Updated**: May 18, 2026  
**Model Version**: v2.0  
**Status**: Production Ready
