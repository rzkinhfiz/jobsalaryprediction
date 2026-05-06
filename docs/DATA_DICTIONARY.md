# Data Dictionary

## Dataset Overview
- **Source**: `data/raw/job_salary_prediction_dataset.csv`
- **Format**: CSV
- **Rows**: [Bergantung pada dataset Anda]
- **Columns**: [Daftar di bawah]

## Feature Descriptions

### Target Variable

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `salary` | Numeric (float/int) | [Min-Max] | Gaji pekerjaan dalam rupiah atau mata uang lokal |

### Categorical Features

| Column | Type | Unique Values | Description | Preprocessing |
|--------|------|----------------|-------------|---|
| `job_title` | String | Bervariasi | Posisi/jabatan pekerjaan (e.g., Data Scientist, Software Engineer) | LabelEncoder |
| `education` | String | 3-5 | Tingkat pendidikan (e.g., High School, Bachelor, Master, PhD) | LabelEncoder |
| `remote_status` or `remote_work` | String/Boolean | 2-3 | Status pekerjaan remote (e.g., Remote, On-site, Hybrid) | LabelEncoder |
| `location` or `city` | String | Bervariasi | Lokasi geografis pekerjaan | LabelEncoder |

### Numeric Features

| Column | Type | Range | Unit | Description |
|--------|------|-------|------|-------------|
| `experience` | Int/Float | 0-50+ | Tahun | Pengalaman kerja dalam tahun |
| `age` | Int | 18-70+ | Tahun | Usia karyawan |
| `years_in_current_role` | Int | 0-30+ | Tahun | Lama bekerja di posisi saat ini |

## Data Quality & Preprocessing

### Missing Values Handling
- **Strategy**: [Drop/Fill/Impute]
- **Columns affected**: [Sebutkan jika ada]

### Outliers
- **Detection**: Visualisasi dengan boxplot
- **Handling**: Log transformation pada salary
- **Method**: IQR method atau statistical bounds

### Feature Engineering
- **Log transformation**: `salary_log = log1p(salary)`
- **Scaling**: StandardScaler pada numeric features
  - Mean = 0, Std = 1
  - Applied AFTER train-test split

### Categorical Encoding
```python
LabelEncoder() untuk setiap categorical column
Mapping disimpan untuk inference later
```

## Data Statistics

### Salary Distribution (Original Scale)
```
Count:    [N rows]
Mean:     [Mean value]
Median:   [Median value]
Std Dev:  [Standard deviation]
Min:      [Minimum value]
Max:      [Maximum value]
```

### Salary Distribution (Log Scale)
```
Lebih normal (closer to Gaussian distribution)
Reduces impact of extreme values
```

## Data Integrity Checks

### Validation Rules
- [ ] Salary > 0
- [ ] Experience >= 0
- [ ] Age >= 18
- [ ] No duplicate records
- [ ] Categorical values dalam whitelist

### Known Issues
- [Sebutkan jika ada issue yang diketahui]
- [e.g., Missing values di column X]
- [e.g., Outliers di education category]

## Class Imbalance (Jika Applicable)
- Regression task (tidak ada class imbalance)
- Salary distribution cenderung skewed (handled dengan log transformation)

## Data Leakage Prevention
- ✓ Scaling dilakukan setelah train-test split
- ✓ Target variable tidak digunakan sebagai feature
- ✓ Temporal ordering (jika ada) dipertahankan
