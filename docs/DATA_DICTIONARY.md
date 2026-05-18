# Data Dictionary

## Dataset Overview
- **Source**: `data/raw/job_salary_prediction_dataset.csv`
- **Format**: CSV
- **Rows**: 250000
- **Columns**: 10
- **Missing values**: 0 di semua kolom

## Dataset Summary
- **Target**: `salary`
- **Feature types**: 6 categorical, 4 numeric
- **Salary range**: 31.867 - 333.046
- **Salary mean**: 145.718
- **Salary median**: 143.453
- **Salary std**: 37.408
- **Salary distribution**: skewed positif, log transform digunakan untuk stabilisasi

## Feature Descriptions

### Target Variable

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `salary` | Integer | 31.867 - 333.046 | Gaji pekerjaan dalam ribuan unit mata uang (misalnya Rp 31.867 sampai Rp 333.046) |

### Input Features

| Column | Type | Unique Values | Description | Preprocessing |
|--------|------|---------------|-------------|---------------|
| `job_title` | String | 12 | Posisi pekerjaan | LabelEncoder |
| `experience_years` | Integer | 21 | Jumlah tahun pengalaman kerja | StandardScaler |
| `education_level` | String | 5 | Tingkat pendidikan | LabelEncoder |
| `skills_count` | Integer | 19 | Jumlah skills atau kompetensi yang dimiliki | StandardScaler |
| `industry` | String | 10 | Industri tempat bekerja | LabelEncoder |
| `company_size` | String | 5 | Ukuran perusahaan | LabelEncoder |
| `location` | String | 10 | Lokasi pekerjaan atau negara | LabelEncoder |
| `remote_work` | String | 3 | Status remote: `Hybrid`, `No`, `Yes` | LabelEncoder |
| `certifications` | Integer | 6 | Jumlah sertifikasi profesional | StandardScaler |

## Categorical Feature Values

### `job_title`
- AI Engineer
- Data Analyst
- Frontend Developer
- Business Analyst
- Product Manager
- Backend Developer
- Machine Learning Engineer
- DevOps Engineer
- Software Engineer
- Cybersecurity Analyst
- Data Scientist
- Cloud Engineer

### `education_level`
- Bachelor
- PhD
- High School
- Diploma
- Master

### `industry`
- Healthcare
- Telecom
- Media
- Retail
- Manufacturing
- Education
- Finance
- Technology
- Consulting
- Government

### `company_size`
- Medium
- Small
- Large
- Enterprise
- Startup

### `location`
- India
- Australia
- Singapore
- Canada
- Sweden
- USA
- Netherlands
- Remote
- Germany
- UK

### `remote_work`
- Hybrid
- No
- Yes

## Numeric Feature Summary

| Column | Type | Min | 25% | Median | 75% | Max |
|--------|------|-----|-----|--------|-----|-----|
| `experience_years` | Integer | 1 | 5 | 10 | 15 | 20 |
| `skills_count` | Integer | 0 | 1 | 4 | 6 | 19 |
| `certifications` | Integer | 0 | 0 | 1 | 2 | 5 |
| `salary` | Integer | 31.867 | 119.358 | 143.453 | 169.492 | 333.046 |

> Catatan: semua numeric feature distandarisasi menggunakan `StandardScaler` setelah pemisahan train-test untuk menghindari data leakage.

## Preprocessing Notes

### Cleaning
- Tidak ada missing values yang perlu ditangani.
- Data bersih dari nilai null pada semua kolom.

### Encoding
- Semua kolom kategorikal di-encode dengan `LabelEncoder`.
- Nilai kategorikal disimpan agar inference konsisten.

### Transformation
- `salary` ditransformasikan menjadi `salary_log` menggunakan `np.log1p(salary)`.
- Transformasi ini mengurangi efek outlier dan membuat target lebih mendekati distribusi normal.

### Train-Test Split
- Data dibagi menjadi 80% training dan 20% testing.
- `random_state=42` untuk reproduksibilitas.

### Scaling
- `StandardScaler` digunakan pada fitur numerik:
  - `experience_years`
  - `skills_count`
  - `certifications`
- Scaling dilakukan setelah train-test split.

## Data Integrity Checks

- **Tidak ada missing values** di dataset.
- **Format kolom** konsisten dengan ekspektasi: 4 numeric, 6 categorical.
- **Range salary** positif, cocok untuk log1p transform.
- **Jumlah data** besar (250k baris) mendukung model yang stabil.

## Recommended Updates

- Simpan mapping LabelEncoder untuk setiap fitur kategorikal di `models/`.
- Simpan `StandardScaler` sebagai artifact untuk inference.
- Catat nilai min/max atau distribusi target untuk monitoring drift.
