"""
Train/evaluate a gradient boosting baseline (HistGradientBoostingClassifier) on creditcard.csv.

Uses a small grid over learning rate / max_depth / max_iter, class_weight for imbalance,
and picks a threshold on a validation split to achieve precision >= 0.9 with maximum recall.

Usage:
    source .venv/bin/activate
    python -m src.train_hgb --data-path data/creditcard.csv --out-dir reports/models/hgb
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from src.tools import evaluate_scores, find_threshold_for_precision


def load_data(path: pathlib.Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset and split features/labels.

    Args:
        path: Path to creditcard.csv.

    Returns:
        Tuple of (X, y) where X excludes Class and y is the target.
    """
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column.")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def train_and_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    precision_target: float = 0.9,
) -> Tuple[HistGradientBoostingClassifier, Dict[str, float], Dict[str, float]]:
    """Fit HGB over a small grid and select threshold for target precision.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        precision_target: Precision level to satisfy on validation.

    Returns:
        (best_model, best_stats, meta) where best_stats contains threshold/precision/recall.
    """
    grid = [
        {"learning_rate": 0.05, "max_depth": 3, "max_iter": 400},
        {"learning_rate": 0.1, "max_depth": 3, "max_iter": 300},
        {"learning_rate": 0.1, "max_depth": None, "max_iter": 300},
    ]
    best = {"recall": -1.0, "precision": 0.0, "threshold": None, "params": None, "pr_auc": -1.0}
    best_model = None
    for params in grid:
        clf = HistGradientBoostingClassifier(
            class_weight={0: 1.0, 1: 20.0},
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            max_iter=params["max_iter"],
            max_leaf_nodes=None,
            validation_fraction=None,  # disable internal split to control ours
            random_state=42,
        )
        clf.fit(X_train, y_train)
        val_scores = clf.predict_proba(X_val)[:, 1]
        thr, prec, rec = find_threshold_for_precision(y_val.to_numpy(), val_scores, precision_target)
        pr_auc_val = average_precision_score(y_val, val_scores)
        if rec > best["recall"] or (rec == best["recall"] and pr_auc_val > best["pr_auc"]):
            best.update({"recall": rec, "precision": prec, "threshold": thr, "params": params, "pr_auc": pr_auc_val})
            best_model = clf
    return best_model, best, {"precision_target": precision_target}


def main(data_path: pathlib.Path, out_dir: pathlib.Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_data(data_path)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )

    model, best_val, meta = train_and_select(X_train, y_train, X_val, y_val, precision_target=0.9)
    test_scores = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_scores(
        y_test,
        test_scores,
        threshold=best_val["threshold"],
        amounts=X_test.get("Amount"),
    )
    metrics.update(
        {
            "val_precision_at_threshold": best_val["precision"],
            "val_recall_at_threshold": best_val["recall"],
            "threshold_prec_target": best_val["threshold"],
            "best_params": best_val["params"],
        }
    )
    metrics.update(meta)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Feature importances dump
    # Permutation importances on a sample of test to keep runtime reasonable
    sample_size = min(5000, len(X_test))
    perm = permutation_importance(
        model,
        X_test.sample(n=sample_size, random_state=42),
        y_test.sample(n=sample_size, random_state=42),
        n_repeats=3,
        random_state=42,
        n_jobs=1,
        scoring="average_precision",
    )
    fi_df = pd.DataFrame({"feature": X.columns, "importance": perm.importances_mean})
    fi_df.sort_values("importance", ascending=False).to_csv(out_dir / "feature_importances.csv", index=False)
    print(f"[hgb] Saved feature importances to {out_dir / 'feature_importances.csv'}")

    print(f"[hgb] Saved metrics to {metrics_path}")
    return {"name": "hgb", "metrics": metrics, "params": best_val["params"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HistGradientBoosting baseline on creditcard.csv.")
    parser.add_argument("--data-path", type=pathlib.Path, default=pathlib.Path("data/creditcard.csv"))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("reports/models/hgb"))
    args = parser.parse_args()
    main(args.data_path, args.out_dir)
