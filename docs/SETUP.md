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
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Option B: Using conda
```bash
# Create conda environment
conda create -n jobsalary python=3.9

# Activate environment
conda activate jobsalary
```

## Step 3: Install Dependencies

```bash
# Install from requirements.txt
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

## Step 4: Verify Installation

```bash
# Test Python
python --version

# Test key packages
python -c "import pandas, sklearn, numpy; print('All packages imported successfully!')"
```

## Step 5: Download/Prepare Data

1. Place raw dataset di folder `data/raw/`
   ```
   data/raw/job_salary_prediction_dataset.csv
   ```

2. Folder structure harus:
   ```
   data/
   ├── raw/
   │   └── job_salary_prediction_dataset.csv
   └── processed/
       └── (akan dibuat setelah processing)
   ```

## Step 6: Run Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open browser ke: http://localhost:8888

# Navigate ke: notebooks/job_salary_prediction.ipynb
```

## Alternative: Using VS Code

1. **Install extensions**:
   - Python (Microsoft)
   - Jupyter (Microsoft)
   - Pylance (optional)

2. **Select kernel**:
   - Open notebook
   - Click kernel selector (top right)
   - Choose your virtual environment

## Troubleshooting

### Issue: Command not found: python
**Solution**: 
```bash
# Use python3 instead
python3 -m venv venv
python3 -m pip install -r requirements.txt
```

### Issue: ModuleNotFoundError
**Solution**:
```bash
# Verify virtual environment is activated
which python  # Should show path inside venv/

# Reinstall packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Jupyter kernel not found
**Solution**:
```bash
# Install ipython kernel
python -m ipykernel install --user --name jobsalary --display-name "Python (Job Salary)"

# Then restart Jupyter and select new kernel
```

### Issue: Permission denied on Linux/Mac
**Solution**:
```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

## Environment Variables (Optional)

Create `.env` file untuk configuration:
```
DATA_PATH=data/raw/
MODEL_PATH=models/
RANDOM_STATE=42
```

Load di Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
data_path = os.getenv('DATA_PATH')
```

## Running Tests (Optional)

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_model.py -v

# Run with coverage
pytest --cov=src tests/
```

## Development Setup

If contributing to project:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run linters
black src/
flake8 src/
```

## Next Steps

1. ✓ Instalasi dependencies
2. Run EDA notebook: `notebooks/job_salary_prediction.ipynb`
3. Train model (cell by cell di notebook)
4. Evaluate hasil di folder `results/` atau `logs/`
5. Deploy model (jika needed)
