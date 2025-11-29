"""
Train/evaluate logistic regression on creditcard.csv with two experiment modes:
- baseline: balanced class weight, single split, no threshold tuning.
- gridsearch: class weight/C grid with validation threshold chosen for precision>=0.9.

Usage:
    source .venv/bin/activate
    python -m src.train_logreg --data-path data/creditcard.csv --out-dir reports/models --mode baseline
    python -m src.train_logreg --data-path data/creditcard.csv --out-dir reports/models --mode gridsearch

Outputs (per mode under out-dir/mode):
    - metrics.json: ROC AUC, PR AUC, retrieval metrics, params/threshold (grid)
    - coef.csv: model coefficients per feature (after scaling)
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Optional, Tuple

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


def find_threshold_for_precision(
    y_true: np.ndarray, y_score: np.ndarray, precision_target: float
) -> Tuple[Optional[float], float, float]:
    """Find threshold achieving precision >= target with maximum recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(precision >= precision_target)[0]
    if len(valid) == 0:
        return None, 0.0, 0.0
    best_idx = valid[np.argmax(recall[valid])]
    # thresholds array is len-1 of precision/recall; guard for edge case
    thr = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    return float(thr), float(precision[best_idx]), float(recall[best_idx])


def top_k_metrics(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> Dict[str, float]:
    """Compute precision and recall at top-k highest scores."""
    k = min(k, len(y_score))
    idx = np.argsort(y_score)[::-1][:k]
    top_true = y_true[idx]
    precision_at_k = float(top_true.mean())
    recall_at_k = float(top_true.sum() / y_true.sum()) if y_true.sum() > 0 else 0.0
    return {"precision_at_k": precision_at_k, "recall_at_k": recall_at_k, "k": float(k)}


def train_logreg(
    X: pd.DataFrame, y: pd.Series, class_weight: object, C: float
) -> Tuple[Pipeline, Dict[str, float]]:
    """Train logistic regression with standard scaling and class weights."""
    num_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_features)],
        remainder="drop",
    )
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        max_iter=500,
        class_weight=class_weight,
        n_jobs=1,  # avoid sandbox issues with semaphores
    )
    pipe = Pipeline([("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X, y)
    return pipe, {"n_features": len(num_features)}


def evaluate(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute AUCs, retrieval metrics, and optional thresholded precision/recall."""
    scores = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
    }
    metrics.update(top_k_metrics(y_test.to_numpy(), scores, k=100))
    metrics.update({"positives": int(y_test.sum()), "negatives": int((1 - y_test).sum())})
    if threshold is not None:
        preds = (scores >= threshold).astype(int)
        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics.update({"precision_at_threshold": precision, "recall_at_threshold": recall})
    return metrics


def evaluate_with_recall_at_prec(y_true: pd.Series, scores: np.ndarray, precision_target: float) -> float:
    """Recall at the best threshold achieving at least the target precision."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    valid = np.where(precision >= precision_target)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(recall[valid]))


def save_coefficients(model: Pipeline, feature_names: pd.Index, out_path: pathlib.Path) -> None:
    """Save model coefficients aligned to original features."""
    logreg = model.named_steps["model"]
    coefs = logreg.coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df.to_csv(out_path, index=False)


def run_gridsearch(data_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Grid search over class weights/C, pick threshold for precision>=0.9 on val."""
    X, y = load_data(data_path)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
    )

    weight_grid = [1, 5, 10, 20]
    c_grid = [0.01, 0.1, 1.0, 10.0]
    best = {
        "recall_at_prec90": -1.0,
        "pr_auc": -1.0,
        "threshold": None,
        "C": None,
        "weight": None,
        "precision_at_threshold": 0.0,
    }
    for w in weight_grid:
        class_weight = {0: 1.0, 1: float(w)}
        for c in c_grid:
            model, _ = train_logreg(X_train, y_train, class_weight=class_weight, C=c)
            val_scores = model.predict_proba(X_val)[:, 1]
            thr, prec, rec = find_threshold_for_precision(
                y_val.to_numpy(), val_scores, precision_target=0.9
            )
            pr_auc_val = average_precision_score(y_val, val_scores)
            if rec > best["recall_at_prec90"] or (
                rec == best["recall_at_prec90"] and pr_auc_val > best["pr_auc"]
            ):
                best.update(
                    {
                        "recall_at_prec90": rec,
                        "precision_at_threshold": prec,
                        "threshold": thr,
                        "C": c,
                        "weight": w,
                        "pr_auc": pr_auc_val,
                    }
                )

    best_class_weight = {0: 1.0, 1: float(best["weight"])}
    final_model, info = train_logreg(
        X_trainval, y_trainval, class_weight=best_class_weight, C=best["C"]
    )
    test_metrics = evaluate(final_model, X_test, y_test, threshold=best["threshold"])
    test_metrics.update(
        {
            "recall_at_prec90_val": best["recall_at_prec90"],
            "precision_at_prec90_val": best["precision_at_threshold"],
            "threshold_prec90": best["threshold"],
            "best_weight": best["weight"],
            "best_C": best["C"],
        }
    )
    test_metrics.update(info)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2))
    save_coefficients(final_model, X.columns, out_dir / "coef.csv")
    print(f"[gridsearch] Saved metrics to {metrics_path}")
    print(f"[gridsearch] Saved coefficients to {out_dir / 'coef.csv'}")
    return {"name": "gridsearch", "metrics": test_metrics}


def run_baseline(data_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Baseline: balanced class weight, C=1, single train/test split."""
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model, info = train_logreg(X_train, y_train, class_weight="balanced", C=1.0)
    scores = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "pr_auc": float(average_precision_score(y_test, scores)),
        "recall_at_prec90": evaluate_with_recall_at_prec(y_test, scores, precision_target=0.9),
    }
    metrics.update(top_k_metrics(y_test.to_numpy(), scores, k=100))
    metrics.update({"positives": int(y_test.sum()), "negatives": int((1 - y_test).sum())})
    metrics.update(info)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    save_coefficients(model, X.columns, out_dir / "coef.csv")
    print(f"[baseline] Saved metrics to {metrics_path}")
    print(f"[baseline] Saved coefficients to {out_dir / 'coef.csv'}")
    return {"name": "baseline", "metrics": metrics}


def run_all(data_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Run both baseline and gridsearch, saving a combined summary."""
    baseline_res = run_baseline(data_path, out_dir / "baseline")
    grid_res = run_gridsearch(data_path, out_dir / "gridsearch")
    summary = {
        "experiments": {
            baseline_res["name"]: baseline_res,
            grid_res["name"]: grid_res,
        }
    }
    summary_path = out_dir / "experiments_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[all] Wrote combined summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train logistic regression on creditcard.csv (baseline or gridsearch)."
    )
    parser.add_argument("--data-path", type=pathlib.Path, default=pathlib.Path("data/creditcard.csv"))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("reports/models"))
    parser.add_argument(
        "--mode",
        choices=["baseline", "gridsearch", "all"],
        default="gridsearch",
        help="baseline: balanced class weight, no threshold tuning; gridsearch: weight/C grid + threshold at prec>=0.9",
    )
    args = parser.parse_args()
    if args.mode == "baseline":
        run_baseline(args.data_path, args.out_dir / "baseline")
    elif args.mode == "gridsearch":
        run_gridsearch(args.data_path, args.out_dir / "gridsearch")
    else:
        run_all(args.data_path, args.out_dir)
