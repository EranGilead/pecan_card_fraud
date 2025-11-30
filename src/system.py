"""
Serve fraud probability for a new transaction using the best available model.

Workflow:
1) Choose best experiment from experiments_summary.json (by PR AUC).
2) Load a saved model/threshold if present; otherwise train and save.
3) Predict fraud probability for the provided transaction.

Usage:
    source .venv/bin/activate
    python -m src.system --transaction-file new_txn.json --data-path data/creditcard.csv --summary reports/models/experiments_summary.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from src.train_logreg import fit_and_score_logreg
from src.train_hgb import fit_and_score_hgb
from src.train_xgb import fit_and_score_xgb


def load_summary(path: pathlib.Path) -> Dict[str, Any]:
    """Load experiments summary JSON."""
    return json.loads(path.read_text())


def select_best_experiment(summary: Dict[str, Any], metric: str = "pr_auc") -> str:
    """Select the experiment name with the highest metric.

    Args:
        summary: Loaded experiments summary dict.
        metric: Metric key to rank by (default pr_auc).

    Returns:
        Best experiment name.
    """
    best_name = None
    best_val = -np.inf
    for name, exp in summary.get("experiments", {}).items():
        val = exp.get("metrics", {}).get(metric)
        if val is not None and val > best_val:
            best_val = val
            best_name = name
    if best_name is None:
        raise ValueError("No experiments found with metric {}".format(metric))
    return best_name


def train_model(experiment: str, data_path: pathlib.Path) -> Tuple[Any, float, pd.Index]:
    """Train the chosen experiment and return model, threshold, and feature order."""
    if experiment == "baseline":
        model, y_test, scores, thr = fit_and_score_logreg(data_path, mode="baseline")
    elif experiment == "gridsearch":
        model, y_test, scores, thr = fit_and_score_logreg(data_path, mode="gridsearch")
    elif experiment == "smote":
        model, y_test, scores, thr = fit_and_score_logreg(data_path, mode="smote")
    elif experiment == "hgb":
        model, y_test, scores, thr = fit_and_score_hgb(data_path)
    elif experiment == "xgb":
        model, y_test, scores, thr = fit_and_score_xgb(data_path)
    else:
        raise ValueError(f"Unsupported experiment: {experiment}")
    # Capture feature order from training data (model expects these columns)
    if hasattr(model, "feature_names_in_"):
        feature_order = pd.Index(model.feature_names_in_)
    else:
        feature_order = pd.Index(y_test.index)  # fallback, unlikely used
    return model, thr, feature_order


def load_or_train_model(
    experiment: str,
    data_path: pathlib.Path,
    model_dir: pathlib.Path,
) -> Tuple[Any, float, pd.Index]:
    """Load model/threshold if saved, else train and save them."""
    model_path = model_dir / "model.joblib"
    threshold_path = model_dir / "threshold.json"
    feature_path = model_dir / "features.json"

    if model_path.exists() and threshold_path.exists() and feature_path.exists():
        model = joblib.load(model_path)
        threshold = json.loads(threshold_path.read_text())["threshold"]
        feature_order = pd.Index(json.loads(feature_path.read_text()))
        return model, threshold, feature_order

    model, threshold, feature_order = train_model(experiment, data_path)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    threshold_path.write_text(json.dumps({"threshold": threshold}))
    feature_path.write_text(json.dumps(list(feature_order)))
    return model, threshold, feature_order


def prepare_features(tx: Dict[str, Any], feature_order: pd.Index) -> pd.DataFrame:
    """Align transaction dict to model feature order, filling missing with zeros."""
    row = {k: tx.get(k, 0) for k in feature_order}
    return pd.DataFrame([row])[feature_order]


def predict_probability(
    model: Any,
    threshold: float,
    features: pd.DataFrame,
) -> Dict[str, float]:
    """Predict fraud probability and label at threshold."""
    proba = float(model.predict_proba(features)[:, 1][0])
    pred = int(proba >= threshold)
    return {"probability": proba, "predicted_label": pred, "threshold_used": threshold}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a new transaction for fraud probability.")
    parser.add_argument("--transaction-file", type=pathlib.Path, required=True, help="JSON file with feature:value mapping.")
    parser.add_argument("--data-path", type=pathlib.Path, default=pathlib.Path("data/creditcard.csv"))
    parser.add_argument(
        "--summary",
        type=pathlib.Path,
        default=pathlib.Path("reports/models/experiments_summary.json"),
        help="Path to experiments_summary.json to pick best model.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="pr_auc",
        help="Metric key in summary to select best model (default: pr_auc).",
    )
    args = parser.parse_args()

    tx = json.loads(args.transaction_file.read_text())
    summary = load_summary(args.summary)
    best_exp = select_best_experiment(summary, metric=args.metric)
    model_dir = args.summary.parent / best_exp
    model, threshold, feature_order = load_or_train_model(best_exp, args.data_path, model_dir)
    features = prepare_features(tx, feature_order)
    result = predict_probability(model, threshold, features)
    result["model_used"] = best_exp
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
