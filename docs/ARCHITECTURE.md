# Project Architecture

## Project Overview

**Job Salary Prediction** adalah project machine learning yang memprediksi gaji pekerjaan berdasarkan fitur pekerjaan dan konteks perusahaan. Proyek ini menggunakan **Random Forest Regressor** dengan preprocessing yang mencakup log-transformasi target, LabelEncoder untuk fitur kategorikal, dan StandardScaler untuk fitur input.

## Folder Structure

```
dsproject_jobsalarypredict/
├── .github/
│   └── copilot-instructions.md
├── config/
│   └── config.json
├── data/
│   ├── raw/
│   │   └── job_salary_prediction_dataset.csv
│   └── processed/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA_DICTIONARY.md
│   ├── MODEL.md
│   ├── RESULTS.md
│   ├── SETUP.md
│   └── USAGE.md
├── models/
│   ├── random_forest_salary_model.pkl
│   ├── feature_scaler.pkl
│   └── label_encoders.pkl
├── notebooks/
│   └── job_salary_prediction.ipynb
├── src/
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── model_pipeline.py
│   │   └── predict.py
│   └── utils/
│       ├── data_loader.py
│       └── encoders.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_feature_engineering.py
│   ├── test_model_pipeline.py
│   └── test_predict.py
├── README.md
└── requirements.txt
```

## Data Pipeline

### Workflow

1. Load raw data dari `data/raw/job_salary_prediction_dataset.csv`
2. Lakukan EDA untuk memahami distribusi, korelasi, dan kualitas data
3. Preprocess data:
   - log-transform target `salary`
   - encode kategori dengan `LabelEncoder`
   - split train/test 80/20
   - scale semua fitur menggunakan `StandardScaler`
4. Train model Random Forest
5. Evaluate model dengan MAE, MSE, RMSE, MAPE, dan R²
6. Save artifacts model, scaler, dan encoders

## Code Layers

- `src/utils/data_loader.py`: load config, load raw data, save processed data
- `src/utils/encoders.py`: save/load LabelEncoder dictionary
- `src/features/feature_engineering.py`: log transform, encoding, split, scaling
- `src/models/model_pipeline.py`: train, save, load model and scaler
- `src/models/predict.py`: inference pipeline and preprocessing

## Inference Flow

1. Load `config/config.json`
2. Load model artifact dan encoders
3. Transform categorical input dengan LabelEncoder
4. Scale semua fitur input dengan `feature_scaler`
5. Prediksi `y_pred_log`
6. Konversi hasil ke skala asli dengan `np.expm1()`

## Configuration

- File utama: `config/config.json`
- Menyimpan paths, artifact names, training parameters, dan metadata fitur

## Notes

- Gunakan `config/config.json` untuk memastikan preprocessing konsisten antara training dan inference
- Semua kode sumber berada di `src/`
- Notebook utama ada di `notebooks/job_salary_prediction.ipynb`
- Unit tests ada di `tests/`
