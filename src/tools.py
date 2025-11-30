from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def add_cost_metrics(
    metrics: Dict[str, float],
    y_true: pd.Series,
    preds: pd.Series,
    amounts: Optional[pd.Series],
    threshold_used: Optional[float] = None,
) -> None:
    """Update metrics dict with FP/FN counts and amount-based costs."""
    if amounts is None:
        return
    fp_mask = (preds == 1) & (y_true == 0)
    fn_mask = (preds == 0) & (y_true == 1)
    fp_amount = float(amounts[fp_mask].sum())
    fn_amount = float(amounts[fn_mask].sum())
    metrics.update(
        {
            "cost_fp_amount": fp_amount,
            "cost_fn_amount": fn_amount,
            "cost_total_amount": fp_amount + fn_amount,
            "fp_count": int(fp_mask.sum()),
            "fn_count": int(fn_mask.sum()),
        }
    )
    if threshold_used is not None:
        metrics["threshold_used"] = float(threshold_used)


def top_k_metrics(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> Dict[str, float]:
    """Precision/recall at top-k scores."""
    k = min(k, len(y_score))
    idx = np.argsort(y_score)[::-1][:k]
    top_true = y_true[idx]
    precision_at_k = float(top_true.mean())
    recall_at_k = float(top_true.sum() / y_true.sum()) if y_true.sum() > 0 else 0.0
    return {"precision_at_k": precision_at_k, "recall_at_k": recall_at_k, "k": float(k)}


def find_threshold_for_precision(
    y_true: np.ndarray, y_score: np.ndarray, precision_target: float
) -> Tuple[Optional[float], float, float]:
    """Find threshold achieving precision >= target with maximum recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    valid = np.where(precision >= precision_target)[0]
    if len(valid) == 0:
        return None, 0.0, 0.0
    best_idx = valid[np.argmax(recall[valid])]
    thr = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    return float(thr), float(precision[best_idx]), float(recall[best_idx])


def evaluate_scores(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: Optional[float] = None,
    amounts: Optional[pd.Series] = None,
    precision_target: Optional[float] = None,
) -> Dict[str, float]:
    """Compute core metrics and optional cost accounting."""
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }
    metrics.update(top_k_metrics(y_true.to_numpy(), y_score, k=100))
    metrics.update({"positives": int(y_true.sum()), "negatives": int((1 - y_true).sum())})

    if precision_target is not None and threshold is None:
        thr, prec, rec = find_threshold_for_precision(y_true.to_numpy(), y_score, precision_target)
        metrics.update(
            {
                "threshold_prec_target": thr,
                "precision_at_threshold": prec,
                "recall_at_threshold": rec,
                "precision_target": precision_target,
            }
        )
        if thr is not None:
            threshold = thr

    if threshold is not None:
        preds = (y_score >= threshold).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics.update({"precision_at_threshold": precision, "recall_at_threshold": recall})
        add_cost_metrics(metrics, y_true, preds, amounts, threshold_used=threshold)
    elif amounts is not None:
        # default threshold 0.5 for cost accounting if none provided
        preds = (y_score >= 0.5).astype(int)
        add_cost_metrics(metrics, y_true, preds, amounts, threshold_used=0.5)

    return metrics
