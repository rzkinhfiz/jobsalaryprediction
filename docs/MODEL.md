# Model Documentation

## Overview
Model prediksi gaji pekerjaan menggunakan algoritma **Random Forest Regressor** dengan log transformation pada target variable untuk menangani distribusi yang skewed.

## Model Architecture

### Algorithm: Random Forest Regressor
- **Type**: Ensemble learning method
- **Base estimators**: Decision trees
- **Number of trees**: 100
- **Random state**: 42 (untuk reproducibility)
- **Objective**: Regression - memprediksi nilai gaji secara kontinyu

### Why Random Forest?
- Robust terhadap outliers
- Menangani non-linear relationships
- Feature importance built-in
- Tidak memerlukan feature scaling (tree-based model)
- Good generalization capability

## Data Preprocessing Pipeline

### 1. Log Transformation
```python
df['salary_log'] = np.log1p(df['salary'])
```
**Alasan**: Mengatasi skewness pada distribusi gaji agar lebih normal

### 2. Categorical Encoding
- Menggunakan `LabelEncoder` untuk mengkonversi categorical features ke numeric
- Features yang dienkode: `job_title`, `education`, `remote_status`, dll

### 3. Feature Scaling (StandardScaler)
- Fitur numeric di-scale ke mean=0, std=1
- **Penting**: Scaling dilakukan SETELAH train/test split
- Training set: `fit_transform()`
- Test set: `transform()` (menggunakan statistics dari training set)

### 4. Train-Test Split
- **Test size**: 20% (default)
- **Random state**: 42
- **Urutan**: Split → Scale → Train

## Model Training

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)
```

**Features yang digunakan**:
- Semua features setelah preprocessing kecuali target variable

## Model Evaluation

### Metrics

| Metric | Formula | Interpretasi |
|--------|---------|--------------|
| **MAE** | $\frac{1}{n}\sum\|y_{true} - y_{pred}\|$ | Rata-rata error absolut dalam rupiah |
| **MSE** | $\frac{1}{n}\sum(y_{true} - y_{pred})^2$ | Penalti untuk error besar |
| **RMSE** | $\sqrt{MSE}$ | Root mean squared error |
| **MAPE** | $\frac{1}{n}\sum\|\frac{y_{true} - y_{pred}}{y_{true}}\| \times 100\%$ | Error dalam persentase |
| **R² Score** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proporsi variance yang dijelaskan (0-1) |

### Prediction Process
1. Input features di-scale menggunakan scaler yang telah dilatih
2. Model memprediksi di log scale
3. Back-transform menggunakan `np.expm1()` ke skala asli

## Model Artifacts

### Saved Model
- **File**: `models/random_forest_salary_model.pkl`
- **Format**: Joblib (pickle)
- **Size**: Bergantung pada jumlah features
- **Load**: `joblib.load('models/random_forest_salary_model.pkl')`

### Additional Artifacts
- Scaler object untuk normalisasi features
- Label encoders untuk categorical variables
- Feature names untuk reference

## Performance Considerations

### Strengths
✓ Interpretable melalui feature importance  
✓ Robust terhadap outliers dan missing values  
✓ Tidak memerlukan feature normalization untuk prediction  
✓ Fast inference time  

### Limitations
✗ Dapat overfit pada training data  
✗ Boros memory untuk dataset besar  
✗ Tidak optimal untuk extrapolation di luar training range  

## Hyperparameter Tuning (Optional)

Untuk meningkatkan performa, Anda dapat tuning:

```python
# Grid search example
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
```

## Monitoring & Maintenance

- **Retraining**: Lakukan ketika ada data baru yang signifikan
- **Validation**: Monitor prediction errors secara periodik
- **Data Drift**: Cek apakah distribution data input berubah
