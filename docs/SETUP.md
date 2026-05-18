# Setup & Installation Guide

## Prerequisites

- **Python**: 3.8 atau lebih baru
- **pip**: Package manager for Python
- **git**: Version control (optional)

## Step 1: Clone Repository (Optional)

```bash
git clone <repository-url>
cd dsproject_jobsalarypredict
```

## Step 2: Create Virtual Environment

### Option A: Using venv
```bash
python -m venv venv
source venv/bin/activate
```

### Option B: Using conda
```bash
conda create -n jobsalary python=3.9
conda activate jobsalary
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Key Libraries Installed
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning
- **matplotlib**: Data visualization
- **seaborn**: Statistical visualization
- **jupyter**: Interactive notebooks
- **joblib**: Model serialization
- **pytest**: Unit testing

## Step 4: Verify Installation

```bash
python --version
python -c "import pandas, sklearn, numpy, joblib; print('All packages imported successfully!')"
```

## Step 5: Prepare Data

1. Place raw dataset in `data/raw/`:
   ```bash
   data/raw/job_salary_prediction_dataset.csv
   ```

2. Ensure folder structure:
   ```bash
   data/
   ├── raw/
   └── processed/
   ```

## Step 6: Run the Notebook

```bash
jupyter notebook
```

Open the notebook at `notebooks/job_salary_prediction.ipynb`.

## Optional: Run Source Code

If you want to use the Python package directly, run from repo root:

```bash
python -m pytest tests/
```

## Configuration

Project settings are stored in:

- `config/config.json`

This file contains dataset paths, model artifact names, training parameters, and feature metadata.

## Running Tests

```bash
python -m pytest tests/
```

## Troubleshooting

### Command not found: python
```bash
python3 -m venv venv
python3 -m pip install -r requirements.txt
```

### ModuleNotFoundError
```bash
which python
pip install -r requirements.txt
```

### Jupyter kernel not found
```bash
python -m ipykernel install --user --name jobsalary --display-name "Python (Job Salary)"
```

### Permission denied on Linux/Mac
```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

## Notes

- The project uses `config/config.json`, not YAML files.
- Model artifacts are saved in `models/`.
- Processed data and metadata are stored in `data/processed/`.
