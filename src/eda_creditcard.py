"""
EDA utilities for the credit card fraud dataset (creditcard.csv).

Run from project root:
    python -m src.eda_creditcard --data-path data/raw/creditcard.csv --out-dir reports/figures
"""
from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import matplotlib

# Use non-interactive backend for headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def load_data(path: pathlib.Path) -> pd.DataFrame:
    """Load the credit card dataset."""
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Expected a 'Class' column with labels (0=legit, 1=fraud).")
    return df


def summarize(df: pd.DataFrame) -> str:
    """Basic dataset info and missing value counts."""
    total = len(df)
    fraud = int(df["Class"].sum())
    legit = total - fraud
    lines = [
        f"Rows: {total:,}",
        f"Legit: {legit:,} ({legit / total:.4%}) | Fraud: {fraud:,} ({fraud / total:.4%})",
    ]
    missing = df.isna().sum()
    nonzero_missing = missing[missing > 0]
    if not nonzero_missing.empty:
        lines.append("\nMissing values:")
        lines.append(nonzero_missing.sort_values(ascending=False).to_string())
    else:
        lines.append("\nNo missing values detected.")
    summary_text = "\n".join(lines)
    print(summary_text)
    return summary_text


def plot_class_balance(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """Bar plot of class counts (imbalanced dataset)."""
    counts = df["Class"].value_counts().rename(index={0: "Legit", 1: "Fraud"}).reset_index()
    counts.columns = ["Class", "Count"]
    ax = sns.barplot(
        data=counts,
        x="Class",
        y="Count",
        hue="Class",
        palette=["#4a90e2", "#d0021b"],
        legend=False,
    )
    ax.bar_label(ax.containers[0], fmt="%.0f")
    ax.set_ylabel("Count")
    ax.set_title("Class Balance")
    ax.set_yscale("log")  # imbalance is large; log scale keeps both visible
    plt.tight_layout()
    out_path = out_dir / "class_balance.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_amount_distribution(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """Log-scaled amount distribution split by class (zeros included via small shift)."""
    if "Amount" not in df.columns:
        return
    min_positive = df.loc[df["Amount"] > 0, "Amount"].min()
    shift = (min_positive / 10) if pd.notnull(min_positive) else 1e-3
    plot_df = df.copy()
    plot_df["Amount_plot"] = df["Amount"].where(df["Amount"] > 0, shift)
    ax = sns.histplot(
        data=plot_df,
        x="Amount_plot",
        hue="Class",
        element="step",
        stat="density",
        common_norm=False,
        palette={0: "#4a90e2", 1: "#d0021b"},
        log_scale=(True, False),
    )
    ax.set_title("Transaction Amount Distribution (log x-axis, zeros shifted)")
    ax.set_xlabel(f"Amount (log scale, zeros -> {shift:g})")
    plt.tight_layout()
    out_path = out_dir / "amount_distribution.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_time_by_class(df: pd.DataFrame, out_dir: pathlib.Path) -> None:
    """Histogram of transaction time by class."""
    if "Time" not in df.columns:
        return
    ax = sns.histplot(
        data=df,
        x="Time",
        hue="Class",
        element="step",
        stat="count",
        common_norm=False,
        palette={0: "#4a90e2", 1: "#d0021b"},
        bins=50,
    )
    ax.set_title("Transaction Time vs Class")
    ax.set_ylabel("Count (log scale)")
    ax.set_yscale("log")
    plt.tight_layout()
    out_path = out_dir / "time_by_class.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_feature_correlations(
    df: pd.DataFrame, out_dir: pathlib.Path, top_k: int = 10
) -> pd.Series:
    """Plot and return top-k absolute correlations with the target."""
    # Exclude target from features
    features = df.drop(columns=["Class"], errors="ignore")
    corr = features.corrwith(df["Class"]).abs().sort_values(ascending=False)
    top_corr = corr.head(top_k)
    corr_df = top_corr.reset_index()
    corr_df.columns = ["Feature", "Corr"]
    ax = sns.barplot(
        data=corr_df,
        x="Corr",
        y="Feature",
        hue="Feature",
        palette="magma",
        legend=False,
    )
    ax.set_title(f"Top {len(top_corr)} Features by |Correlation| with Fraud Class")
    ax.set_xlabel("Absolute correlation")
    plt.tight_layout()
    out_path = out_dir / "top_feature_correlations.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")
    return top_corr


def plot_pairplot(df: pd.DataFrame, out_dir: pathlib.Path, features: pd.Index, max_points: int = 2000) -> None:
    """Pairwise scatter/diagonal KDE for selected features; samples for speed."""
    if features.empty:
        return
    sample = df if len(df) <= max_points else df.sample(n=max_points, random_state=42)
    pair_df = sample[list(features) + ["Class"]].copy()
    pair_df["Class"] = pair_df["Class"].replace({0: "Legit", 1: "Fraud"})
    g = sns.pairplot(
        pair_df,
        vars=list(features),
        hue="Class",
        diag_kind="kde",
        corner=True,
        plot_kws={"s": 10, "alpha": 0.4},
        palette={"Legit": "#4a90e2", "Fraud": "#d0021b"},
    )
    g.fig.suptitle(f"Pairwise feature separation ({len(sample):,} samples)", y=1.02)
    out_path = out_dir / "pairplot.png"
    g.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)
    print(f"Saved {out_path}")


def write_summary(out_dir: pathlib.Path, summary_text: str, top_corr: pd.Series) -> None:
    """Write a concise summary of key findings to summary.txt."""
    lines = [summary_text, "\nTop feature correlations with fraud (abs):"]
    lines.append(top_corr.to_string())
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(lines))
    print(f"Wrote summary to {summary_path}")


def run_eda(data_path: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Execute EDA workflow and generate figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(data_path)
    summary_text = summarize(df)
    sns.set_theme(style="whitegrid")
    plot_class_balance(df, out_dir)
    plot_amount_distribution(df, out_dir)
    plot_time_by_class(df, out_dir)
    top_corr = plot_feature_correlations(df, out_dir)
    # Use top correlated features for pairwise visualization
    pair_features = top_corr.index[:5]
    plot_pairplot(df, out_dir, pair_features)
    write_summary(out_dir, summary_text, top_corr)


def parse_args() -> Tuple[pathlib.Path, pathlib.Path]:
    parser = argparse.ArgumentParser(description="EDA for creditcard.csv fraud dataset.")
    parser.add_argument(
        "--data-path",
        type=pathlib.Path,
        default=pathlib.Path("data/raw/creditcard.csv"),
        help="Path to creditcard.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("reports/figures"),
        help="Directory to write plots",
    )
    args = parser.parse_args()
    return args.data_path, args.out_dir


if __name__ == "__main__":
    data_path, out_dir = parse_args()
    run_eda(data_path, out_dir)
