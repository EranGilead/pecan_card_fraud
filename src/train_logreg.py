"""
Train/evaluate a baseline logistic regression model on creditcard.csv.

Usage:
    source .venv/bin/activate
    python -m src.train_logreg --data-path data/creditcard.csv --out-dir reports/models

Outputs:
    - metrics.json: ROC AUC, PR AUC, recall@k, precision@k, and class balance
    - coef.csv: model coefficients per feature (after scaling)
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data(path: pathlib.Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column.")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def recall_at_precision_threshold(y_true: np.ndarray, y_score: np.ndarray, precision_target: float = 0.9) -> float:
    """Compute recall at the smallest threshold where precision >= target."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(precision >= precision_target)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(recall[valid]))


def top_k_metrics(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> Dict[str, float]:
    """Compute precision and recall at top-k highest scores."""
    k = min(k, len(y_score))
    idx = np.argsort(y_score)[::-1][:k]
    top_true = y_true[idx]
    precision_at_k = float(top_true.mean())
    recall_at_k = float(top_true.sum() / y_true.sum()) if y_true.sum() > 0 else 0.0
    return {"precision_at_k": precision_at_k, "recall_at_k": recall_at_k, "k": float(k)}


def train_logreg(X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, Dict[str, float]]:
    """Train logistic regression with standard scaling and class weights."""
    num_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_features)],
        remainder="drop",
    )
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=500,
        class_weight="balanced",
        n_jobs=1,  # avoid sandbox issues with semaphores
    )
    pipe = Pipeline([("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X, y)
    return pipe, {"n_features": len(num_features)}


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Compute AUCs and retrieval metrics."""
    scores = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "recall_at_prec90": recall_at_precision_threshold(y_test.to_numpy(), scores, precision_target=0.9),
    }
    metrics.update(top_k_metrics(y_test.to_numpy(), scores, k=100))
    metrics.update({"positives": int(y_test.sum()), "negatives": int((1 - y_test).sum())})
    return metrics


def save_coefficients(model: Pipeline, feature_names: pd.Index, out_path: pathlib.Path) -> None:
    """Save model coefficients aligned to original features."""
    logreg = model.named_steps["model"]
    coefs = logreg.coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df.to_csv(out_path, index=False)


def main(data_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model, info = train_logreg(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    metrics.update(info)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    save_coefficients(model, X.columns, out_dir / "coef.csv")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved coefficients to {out_dir / 'coef.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train logistic regression baseline on creditcard.csv.")
    parser.add_argument("--data-path", type=pathlib.Path, default=pathlib.Path("data/creditcard.csv"))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("reports/models"))
    args = parser.parse_args()
    main(args.data_path, args.out_dir)
