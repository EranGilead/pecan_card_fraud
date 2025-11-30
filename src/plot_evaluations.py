"""
Generate evaluation plots from experiments_summary.json:
- PR AUC / ROC AUC curves and bars
- Precision/Recall at tuned threshold (where available)
- Costs (FP/FN/total amount)

Usage:
    source .venv/bin/activate
    python -m src.plot_evaluations --summary reports/models/experiments_summary.json --out-dir reports/figures
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier


def load_summary(path: pathlib.Path) -> pd.DataFrame:
    """Load experiments_summary.json into a DataFrame."""
    data = json.loads(path.read_text())
    rows = []
    for name, exp in data.get("experiments", {}).items():
        m = exp.get("metrics", {})
        row = {"model": name}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


def load_data(path: pathlib.Path):
    """Load dataset and split features/labels."""
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def _barplot(df: pd.DataFrame, x: str, y: str, title: str, out_path: pathlib.Path, fmt: str = ".3f") -> None:
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(data=df, x=x, y=y, palette="mako")
    ax.bar_label(ax.containers[0], fmt=f"%{fmt}")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_pr_roc(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    metric_df = df[["model", "pr_auc", "roc_auc"]].dropna()
    metric_df = metric_df.sort_values("pr_auc", ascending=False)
    _barplot(metric_df, x="model", y="pr_auc", title="PR AUC by Model", out_path=out_dir / "pr_auc.png")
    _barplot(metric_df, x="model", y="roc_auc", title="ROC AUC by Model", out_path=out_dir / "roc_auc.png")


def plot_threshold_metrics(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    cols = ["model", "precision_at_threshold", "recall_at_threshold"]
    if not set(cols[1:]).issubset(df.columns):
        return
    tdf = df[cols].dropna()
    if tdf.empty:
        return
    tdf = tdf.sort_values("precision_at_threshold", ascending=False)
    plt.figure(figsize=(6, 4))
    tdf_melt = tdf.melt(id_vars="model", value_vars=["precision_at_threshold", "recall_at_threshold"], var_name="metric")
    ax = sns.barplot(data=tdf_melt, x="model", y="value", hue="metric", palette="Set2")
    ax.set_ylim(0, 1)
    ax.set_title("Precision/Recall at Tuned Threshold")
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall_threshold.png", dpi=200)
    plt.close()
    print(f"Saved {out_dir / 'precision_recall_threshold.png'}")


def plot_costs(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    cost_cols = ["cost_fp_amount", "cost_fn_amount", "cost_total_amount"]
    if not set(cost_cols).issubset(df.columns):
        return
    cdf = df[["model"] + cost_cols].dropna()
    if cdf.empty:
        return
    cdf = cdf.sort_values("cost_total_amount")
    # stacked bar for FP/FN
    plt.figure(figsize=(6, 4))
    bottom = cdf["cost_fp_amount"]
    ax = plt.gca()
    ax.bar(cdf["model"], cdf["cost_fp_amount"], label="FP cost", color="#4a90e2")
    ax.bar(cdf["model"], cdf["cost_fn_amount"], bottom=cdf["cost_fp_amount"], label="FN cost", color="#d0021b")
    ax.set_ylabel("Amount")
    ax.set_title("Cost by Model (FP + FN)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "costs_stacked.png", dpi=200)
    plt.close()
    print(f"Saved {out_dir / 'costs_stacked.png'}")


def plot_precision_at_k(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    if "precision_at_k" not in df.columns:
        return
    pdf = df[["model", "precision_at_k", "recall_at_k"]].dropna()
    if pdf.empty:
        return
    pdf = pdf.sort_values("precision_at_k", ascending=False)
    plt.figure(figsize=(6, 4))
    mlt = pdf.melt(id_vars="model", value_vars=["precision_at_k", "recall_at_k"], var_name="metric")
    ax = sns.barplot(data=mlt, x="model", y="value", hue="metric", palette="muted")
    ax.set_title("Top-K Metrics (k from data)")
    plt.tight_layout()
    plt.savefig(out_dir / "topk_metrics.png", dpi=200)
    plt.close()
    print(f"Saved {out_dir / 'topk_metrics.png'}")


def train_and_scores_baseline(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
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
        n_jobs=1,
    )
    pipe = Pipeline([("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    scores = pipe.predict_proba(X_test)[:, 1]
    return y_test, scores


def train_and_scores_grid(X, y, best_weight: float, best_c: float):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    num_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_features)],
        remainder="drop",
    )
    clf = LogisticRegression(
        penalty="l2",
        C=best_c,
        max_iter=500,
        class_weight={0: 1.0, 1: float(best_weight)},
        n_jobs=1,
    )
    pipe = Pipeline([("preprocess", preprocessor), ("model", clf)])
    pipe.fit(X_trainval, y_trainval)
    scores = pipe.predict_proba(X_test)[:, 1]
    return y_test, scores


def train_and_scores_smote(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    num_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), num_features)],
        remainder="drop",
    )
    pipe = ImbPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("smote", SMOTE(random_state=42)),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    max_iter=500,
                    class_weight=None,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    scores = pipe.predict_proba(X_test)[:, 1]
    return y_test, scores


def train_and_scores_hgb(X, y, params: dict):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = HistGradientBoostingClassifier(
        class_weight={0: 1.0, 1: 20.0},
        learning_rate=params.get("learning_rate", 0.1),
        max_depth=params.get("max_depth", 3),
        max_iter=params.get("max_iter", 300),
        random_state=42,
    )
    clf.fit(X_trainval, y_trainval)
    scores = clf.predict_proba(X_test)[:, 1]
    return y_test, scores


def train_and_scores_xgb(X, y, params: dict):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=1,
        random_state=42,
        learning_rate=params.get("learning_rate", 0.1),
        max_depth=params.get("max_depth", 4),
        n_estimators=params.get("n_estimators", 400),
        scale_pos_weight=params.get("scale_pos_weight"),
    )
    clf.fit(X_trainval, y_trainval)
    scores = clf.predict_proba(X_test)[:, 1]
    return y_test, scores


def plot_curves(summary_df: pd.DataFrame, data_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    X, y = load_data(data_path)
    curves = []
    for _, row in summary_df.iterrows():
        model = row["model"]
        if model == "baseline":
            y_true, scores = train_and_scores_baseline(X, y)
        elif model == "gridsearch":
            y_true, scores = train_and_scores_grid(X, y, row.get("best_weight", 1), row.get("best_C", 1.0))
        elif model == "smote":
            y_true, scores = train_and_scores_smote(X, y)
        elif model == "hgb":
            y_true, scores = train_and_scores_hgb(X, y, row.get("best_params", {}))
        elif model == "xgb":
            y_true, scores = train_and_scores_xgb(X, y, row.get("best_params", {}))
        else:
            continue
        fpr, tpr, _ = roc_curve(y_true, scores)
        prec, rec, _ = precision_recall_curve(y_true, scores)
        curves.append({"model": model, "fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec})

    # ROC curve
    plt.figure(figsize=(6, 5))
    for c in curves:
        plt.plot(c["fpr"], c["tpr"], label=c["model"])
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=200)
    plt.close()

    # PR curve
    plt.figure(figsize=(6, 5))
    for c in curves:
        plt.plot(c["recall"], c["precision"], label=c["model"])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=200)
    plt.close()


def main(summary_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_summary(summary_path)
    plot_pr_roc(df, out_dir)
    plot_threshold_metrics(df, out_dir)
    plot_costs(df, out_dir)
    plot_precision_at_k(df, out_dir)
    # Curves require raw data to recompute scores
    data_path = pathlib.Path("data/creditcard.csv")
    if data_path.exists():
        plot_curves(df, data_path, out_dir)
    else:
        print("Data file data/creditcard.csv not found; skipping curve plots.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot evaluation metrics across experiments.")
    parser.add_argument("--summary", type=pathlib.Path, default=pathlib.Path("reports/models/experiments_summary.json"))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("reports/figures"))
    args = parser.parse_args()
    main(args.summary, args.out_dir)
