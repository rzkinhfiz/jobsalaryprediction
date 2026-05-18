# Model Documentation

## Overview
Model ini menggunakan **Random Forest Regressor** untuk memprediksi `salary` berdasarkan fitur pekerjaan dan perusahaan. Target `salary` ditransformasikan menjadi `salary_log` dengan `np.log1p()` untuk mengurangi skewness pada distribusi.

## Model Architecture

### Algorithm: Random Forest Regressor
- **Type**: Ensemble learning
- **Base estimators**: Decision trees
- **n_estimators**: 100
- **random_state**: 42
- **Objective**: Regression

### Why Random Forest?
- Menangani hubungan non-linear antara fitur dan target
- Tahan terhadap outlier pada fitur dan target
- Memberikan feature importance
- Cocok untuk dataset campuran numeric dan encoded categorical

## Data Preprocessing

### 1. Log Transformation
```python
df['salary_log'] = np.log1p(df['salary'])
```
- Mengurangi skewness pada distribusi gaji
- Membuat target lebih stabil untuk regresi

### 2. Categorical Encoding
- Menggunakan `LabelEncoder` untuk setiap kolom kategorikal
- Kolom yang dienkode:
  - `job_title`
  - `education_level`
  - `industry`
  - `company_size`
  - `location`
  - `remote_work`

### 3. Feature Scaling
- Semua fitur input diskalakan dengan `StandardScaler`
- Scaling dilakukan setelah train/test split
- `feature_scaler` disimpan untuk inference produksi

### 4. Train-Test Split
- `test_size`: 0.2
- `random_state`: 42
- Urutan: split → encode → scale → train

## Model Training

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X_train_scaled, y_train)
```

### Input Features
- Semua fitur pada `X` setelah preprocessing
- Target yang dilatih: `salary_log`

## Model Evaluation

### Metrics
| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| MSE | Mean Squared Error |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| R² | Explained variance score |

### Prediction Flow
1. Preprocess input dengan label encoders
2. Scale fitur menggunakan `feature_scaler`
3. Prediksi `y_pred_log`
4. Back-transform ke nilai asli:
```python
salary = np.expm1(y_pred_log)
```

## Saved Artifacts

- `models/random_forest_salary_model.pkl`
- `models/feature_scaler.pkl`
- `models/label_encoders.pkl`

## Practical Notes

- `RandomForestRegressor` sering kali tidak memerlukan scaling, tetapi di sini scaling diterapkan untuk menjaga konsistensi numerik dan inference yang stabil.
- Label encoders disimpan agar model inference menggunakan representasi kategori yang sama seperti saat training.
- Bergantung pada deployment, persiapkan strategi untuk menangani kategori baru yang tidak dikenal.

## Strengths
- Interpretasi feature importance
- Robust terhadap outlier
- Prediksi non-linear

## Limitations
- Model ukuran bisa cukup besar
- Perlu perhatian pada kategori baru saat inference
- Tidak baik untuk prediksi di luar rentang data training

## Maintenance
- Simpan dan versi artifact model
- Monitor error dan drift data
- Retrain model saat data baru tersedia
