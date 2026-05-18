# Model Results & Evaluation

## Training Summary

### Dataset Split
- **Total Records**: 250000
- **Training Set**: 80% (200000 records)
- **Test Set**: 20% (50000 records)
- **Train-Test Random State**: 42

### Model Configuration
- **Algorithm**: Random Forest Regressor
- **n_estimators**: 100
- **max_depth**: None
- **random_state**: 42

## Performance Metrics

### Test Set Results

| Metric | Value |
|--------|-------|
| **MAE** | 5202.27 |
| **MSE** | 43175717.67 |
| **RMSE** | 6570.82 |
| **MAPE** | 3.77% |
| **R² Score** | 0.97 |

**Interpretation**: Model berhasil memprediksi gaji dengan kesalahan rata-rata di bawah Rp 6.000 dan tingkat akurasi sangat tinggi pada dataset ini.

## Error Analysis

### Residual Summary

- **Mean Error**: 363.42
- **Median Absolute Error**: 4348.45
- **Std Dev of Errors**: 6560.77
- **Min Error**: -36822.32
- **Max Error**: 33079.07

### Key Insights
- Residuals tersebar di sekitar nol, menunjukkan model cenderung seimbang antara underestimation dan overestimation.
- Error absolut sedikit lebih besar di nilai gaji tinggi, tetapi persentase error tetap rendah.

## Feature Importance

### Top Predictors

| Feature | Importance |
|---------|------------|
| `location` | 35.14% |
| `experience_years` | 19.61% |
| `job_title` | 15.91% |
| `company_size` | 15.08% |
| `education_level` | 9.35% |
| `skills_count` | 2.60% |
| `certifications` | 1.00% |
| `industry` | 0.73% |
| `remote_work` | 0.57% |

### Interpretation
- `location` dan `experience_years` adalah prediktor paling kuat.
- `job_title` dan `company_size` juga memberi kontribusi besar pada hasil prediksi.
- Fitur-fitur seperti `skills_count`, `certifications`, `industry`, dan `remote_work` berkontribusi lebih kecil, tetapi tetap berguna untuk model.

## Residual Analysis

### Patterns
- Residuals tidak menunjukkan pola heteroscedastic yang kuat.
- Titik-titik prediksi menyebar merata di sekitar garis nol.
- Outlier relatif terbatas dan model masih cukup stabil.

### Plot Interpretation
- Titik di dekat garis diagonal pada scatter plot berarti prediksi mendekati nilai aktual.
- Penyebaran yang lebih lebar pada nilai ekstrem menunjukkan ketidakpastian lebih tinggi pada gaji besar.

## Notes on Model Robustness

- Model menggunakan log-transformasi target sehingga prediksi dikembalikan ke skala asli dengan `np.expm1()`.
- Fitur diskalakan dengan `StandardScaler` setelah encoding, sehingga semua input model berada pada rentang numerik yang konsisten.
- Label encoders disimpan untuk memastikan inferensi konsisten di production.

## Saved Artifacts

- `models/random_forest_salary_model.pkl`
- `models/feature_scaler.pkl`
- `models/label_encoders.pkl`

## Recommendations

### Next improvements
1. Tuning hyperparameter dengan Grid Search atau Random Search
2. Menguji algoritma lain seperti Gradient Boosting atau XGBoost
3. Menambahkan fitur domain khusus industri
4. Mengumpulkan data baru untuk memvalidasi drift

### Deployment
- Simpan model, scaler, dan encoder bersama untuk inference reproducible.
- Pantau input data baru agar nilai kategori baru tidak merusak label encoder.

## Version History

| Version | Date | Notes |
|---------|------|-------|
| v1.0 | 2026-05-18 | Updated documentation and artifact persistence for `feature_scaler` and label encoders |

**Last Updated**: May 18, 2026

