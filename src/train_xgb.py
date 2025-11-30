"""
Train/evaluate an XGBoost classifier on creditcard.csv.

Uses a small grid over learning_rate / max_depth / n_estimators and scale_pos_weight,
then selects a threshold on a validation split to achieve precision >= 0.9 with maximum recall.

Usage:
    source .venv/bin/activate
    python -m src.train_xgb --data-path data/creditcard.csv --out-dir reports/models/xgb
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from src.tools import evaluate_scores, find_threshold_for_precision


def load_data(path: pathlib.Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Expected 'Class' column.")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def top_k_metrics(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> Dict[str, float]:
    k = min(k, len(y_score))
    idx = np.argsort(y_score)[::-1][:k]
    top_true = y_true[idx]
    precision_at_k = float(top_true.mean())
    recall_at_k = float(top_true.sum() / y_true.sum()) if y_true.sum() > 0 else 0.0
    return {"precision_at_k": precision_at_k, "recall_at_k": recall_at_k, "k": float(k)}


def find_threshold_for_precision(
    y_true: np.ndarray, y_score: np.ndarray, precision_target: float
) -> Tuple[Optional[float], float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(precision >= precision_target)[0]
    if len(valid) == 0:
        return None, 0.0, 0.0
    best_idx = valid[np.argmax(recall[valid])]
    thr = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    return float(thr), float(precision[best_idx]), float(recall[best_idx])


def train_and_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    precision_target: float = 0.9,
) -> Tuple[XGBClassifier, Dict[str, float]]:
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    base_spw = neg / pos if pos > 0 else 1.0
    grid = [
        {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 300, "scale_pos_weight": base_spw},
        {"learning_rate": 0.05, "max_depth": 4, "n_estimators": 400, "scale_pos_weight": base_spw},
        {"learning_rate": 0.1, "max_depth": 4, "n_estimators": 400, "scale_pos_weight": base_spw},
    ]
    best = {"recall": -1.0, "precision": 0.0, "threshold": None, "params": None, "pr_auc": -1.0}
    best_model = None
    for params in grid:
        clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=1,
            random_state=42,
            **params,
        )
        clf.fit(X_train, y_train)
        val_scores = clf.predict_proba(X_val)[:, 1]
        thr, prec, rec = find_threshold_for_precision(y_val.to_numpy(), val_scores, precision_target)
        pr_auc_val = average_precision_score(y_val, val_scores)
        if rec > best["recall"] or (rec == best["recall"] and pr_auc_val > best["pr_auc"]):
            best.update({"recall": rec, "precision": prec, "threshold": thr, "params": params, "pr_auc": pr_auc_val})
            best_model = clf
    return best_model, best


def main(data_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_data(data_path)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )

    model, best_val = train_and_select(X_train, y_train, X_val, y_val, precision_target=0.9)
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
            "precision_target": 0.9,
        }
    )

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Feature importances (gain)
    fi = model.get_booster().get_score(importance_type="gain")
    fi_df = pd.DataFrame(
        [
            {"feature": f, "importance": float(v)}
            for f, v in fi.items()
        ]
    ).sort_values("importance", ascending=False)
    fi_df.to_csv(out_dir / "feature_importances.csv", index=False)

    print(f"[xgb] Saved metrics to {metrics_path}")
    print(f"[xgb] Saved feature importances to {out_dir / 'feature_importances.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost classifier on creditcard.csv.")
    parser.add_argument("--data-path", type=pathlib.Path, default=pathlib.Path("data/creditcard.csv"))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("reports/models/xgb"))
    args = parser.parse_args()
    main(args.data_path, args.out_dir)
