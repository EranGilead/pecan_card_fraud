## Credit Card Fraud Detection

Local starter for experimenting with credit card fraud detection: data exploration, feature engineering, model training, and evaluation.

### Environment
- Python 3 (see `.python-version` if present)
- Create and activate venv: `python3 -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Freeze updates: `pip freeze > requirements.txt`

### Project layout (initial)
- `data/` keep raw/intermediate datasets (not tracked by git)
- `notebooks/` exploratory work
- `src/` reusable code (preprocessing, features, models)
- `reports/` figures/metrics from experiments

### Next steps
- Pull/download anonymized credit card fraud dataset (e.g., Kaggle or public ULB dataset) into `data/raw/`.
- Start an EDA notebook in `notebooks/` to inspect class imbalance and basic stats.
- Implement preprocessing and modeling pipelines in `src/` using `scikit-learn` + `imbalanced-learn`.
