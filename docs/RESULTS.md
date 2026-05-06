# Model Results & Evaluation

## Training Summary

### Dataset Split
- **Total Records**: 250000
- **Training Set**: 80% (200000 records)
- **Test Set**: 20% (50000 records)
- **Train-Test Random State**: 42

### Model Configuration
- **Algorithm**: Random Forest Regressor
- **n_estimators**: 100 trees
- **max_depth**: None (unlimited)
- **min_samples_split**: 2
- **min_samples_leaf**: 1
- **random_state**: 42

## Performance Metrics

### Test Set Results

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **MAE** | 5202.27 | Rata-rata error $ 5202.27 |
| **MSE** | 43175717.67 | Mean squared error |
| **RMSE** | 6570.82 | Root mean squared error |
| **MAPE** | 3.77% | Rata-rata error 3.77% |
| **R² Score** | 0.97 | Model menjelaskan 97% variance |

### Confidence Interpretation
- ✓ R² > 0.7: Good model
- ✓ R² 0.5-0.7: Moderate model
- ✓ R² < 0.5: Weak model
- **Our Model**: R² = 0.97

## Error Analysis

### Error Distribution
```
Mean Error (ME): 363.42
Median Absolute Error: 4348.45
Std Dev of Errors: 6560.77
Min Error: -36822.32
Max Error: 33079.07
```

### Error by Salary Range

| Salary Range | Count | MAE | MAPE |
|--------------|-------|----------------|---------|
| < 100k | 5253 | 4928.48 | 5.87% |
| 100k - 150k | 23193 | 4921.73 | 3.91% |
| 150k - 200k | 17520 | 5349.44 | 3.14% |
| > 200k | 4034 | 6532.55 | 2.96% |

**Insight**: Model error bervariasi menurut kisaran gaji. Error absolut cenderung sedikit meningkat pada gaji lebih tinggi, tetapi persentase error (MAPE) berkurang pada rentang >200k karena nilai dasar yang lebih besar.

## Feature Importance

### Top Features

```
1. location           35.14%
2. experience_years   19.61%
3. job_title          15.91%
4. company_size       15.08%
5. education_level     9.35%
6. skills_count        2.60%
7. certifications      1.00%
8. industry            0.73%
9. remote_work         0.57%
```

### Feature Importance Insights
- **Lokasi pekerjaan** (`location`) adalah faktor paling dominan, menunjukkan bahwa regionalisasi gaji sangat kuat dalam dataset ini.
- **Pengalaman kerja** (`experience_years`) adalah predictor kedua terkuat, sejalan dengan ekspektasi bahwa pengalaman meningkatkan gaji.
- **Job title** dan **company size** juga memiliki kontribusi besar, menandakan bahwa jabatan dan ukuran perusahaan mempengaruhi skala kompensasi.
- **Education level** tetap penting, tetapi dampaknya lebih kecil dibandingkan faktor pengalaman dan lokasi.
- Fitur seperti `skills_count`, `certifications`, `industry`, dan `remote_work` memiliki pengaruh lebih kecil; ini bisa menjadi kandidat untuk simplifikasi model jika diperlukan.
- Kesimpulan: model lebih sensitif terhadap konteks pekerjaan (lokasi, jabatan, ukuran perusahaan, pengalaman) daripada atribut tambahan yang lebih detil.

## Residual Analysis

### Residual Statistics
```
Mean of Residuals: 363.42
Std Dev: 6560.77
Min Residual: -36822.32
Max Residual: 33079.07
```

### Residual Patterns
- ✓ Residuals appear randomly distributed around zero
- ✓ No strong heteroscedasticity detected
- ✓ Outliers dalam acceptable range

### Issues Detected
- [ ] Systematic bias untuk certain salary ranges
- [ ] Heteroscedasticity (non-constant variance)
- [ ] Autocorrelation (jika time-series data)

## Predictions vs Actual

### Scatter Plot Interpretation
- Points close to diagonal line = good predictions
- Deviations indicate prediction errors
- Larger spreads pada extreme values = model uncertainty

### Under/Over-prediction
- **Underestimation**: Model tends to predict lower salary
- **Overestimation**: Model tends to predict higher salary
- **Balanced**: Mix of both

## Comparison with Baseline

### Baseline Model: Mean Predictor
```
Baseline MAE:  [Rp X]
Model MAE:     [Rp Y]
Improvement:   [Z%]

Baseline MAPE: [X%]
Model MAPE:    [Y%]
```

## Model Robustness

### Cross-Validation Results (Optional)

```python
# 5-Fold Cross-Validation
Fold 1: MAE = [Value]
Fold 2: MAE = [Value]
Fold 3: MAE = [Value]
Fold 4: MAE = [Value]
Fold 5: MAE = [Value]
Mean CV MAE: [Value] ± [Std]
```

**Insight**: Model stability across different data splits

## Overfitting Assessment

### Training vs Test Performance
```
Training MAE: [Value]
Test MAE:     [Value]
Difference:   [Value] (< 10% gap = good)
```

- ✓ No significant overfitting detected
- ⚠ Minor overfitting present
- ✗ Significant overfitting detected

## Recommendations

### Model Improvements
1. [ ] Hyperparameter tuning (Grid Search / Random Search)
2. [ ] Add more features or engineer new ones
3. [ ] Collect more training data
4. [ ] Remove low-importance features
5. [ ] Try ensemble methods (Gradient Boosting, XGBoost)

### Data Quality
1. [ ] Handle outliers more aggressively
2. [ ] Investigate data quality issues
3. [ ] Rebalance feature distribution
4. [ ] Add domain-specific features

### Deployment Considerations
1. [ ] Monitor prediction errors in production
2. [ ] Retrain model periodically with new data
3. [ ] Implement prediction confidence thresholds
4. [ ] Set up alerts untuk anomalous predictions

## Production Metrics

### Expected Performance in Production
- **Target MAE**: Rp [Value]
- **Acceptable MAPE**: [Value]%
- **Prediction Latency**: < 100ms
- **Model Size**: ~[X] MB

### Success Criteria
- Model meets or exceeds target metrics
- Predictions used by [stakeholder] team
- User feedback positive

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2024-XX-XX | Initial model with 100 trees, R²=[Value] |
| v1.1 | 2024-XX-XX | Improved preprocessing, R²=[Value] |
| v2.0 | 2024-XX-XX | Added feature engineering, R²=[Value] |

---

**Last Updated**: May 6, 2026  
**Model ID**: random_forest_salary_model_v1.0  
**Status**: Production Ready
