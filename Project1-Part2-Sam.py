"""
California Housing — diagnosing and handling the $500,001 price cap
===================================================================

Pipeline
  1. Load sklearn's California Housing data and split it 80/20 into train/test sets.
  2. Baseline: ordinary Linear Regression on the raw data -> MSE, R².
  3. Cleaning pipeline: detect the price ceiling from the TRAINING data (the maximum
     value, repeated far more often than any value just below it) and remove the
     capped rows from training.
  4. Second Linear Regression on the cleaned training data -> MSE, R².
     Both models are also scored on the same uncapped test homes for a fair comparison.

Outputs (written to ./outputs)
  baseline_predicted_vs_actual.png   Predicted (x) vs. Actual (y): the cap shows as a horizontal line
  model_metrics.csv                  MSE and R² for each model / test sample
  bias_by_price_band.csv             Mean prediction error by actual-price band, both models
  defense_report_results.md          Row counts and metrics

Target units: MedHouseVal is in $100,000s, so the cap appears as 5.00001.

Usage
  python regression.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXPECTED_CAP = 5.00001  # documented ceiling, used only as a sanity check

PRICE_BANDS = [0, 1, 2, 3, 4, 5.0]
BAND_LABELS = ["< $100k", "$100-200k", "$200-300k", "$300-400k", "$400-500k"]


def detect_cap(y, k_neighbors=10, min_share=0.002, min_ratio=20.0):
    """
    Detect a ceiling on the target. A real price distribution thins out toward its
    maximum; a capped one piles every value above the ceiling onto the maximum.
    Flags the max when it holds >= min_share of rows AND is >= min_ratio times as
    common as the k distinct values just below it.
    """
    y_max = float(y.max())
    at_max = np.isclose(y, y_max)
    n_at_max = int(at_max.sum())
    neighbor_counts = y[~at_max].value_counts().sort_index().iloc[-k_neighbors:]
    spike_ratio = n_at_max / max(float(neighbor_counts.mean()), 1e-9)
    share = n_at_max / len(y)
    is_capped = share >= min_share and spike_ratio >= min_ratio
    return y_max, n_at_max, share, spike_ratio, is_capped


def get_metrics(model, X, y):
    preds = model.predict(X)
    return float(mean_squared_error(y, preds)), float(r2_score(y, preds))


def make_baseline_plot(y_true, y_pred, capped_mask, cap, save_path):
    """Predicted on x, actual on y, so the $500k cap shows as a horizontal line."""
    y_true = np.asarray(y_true)
    capped_mask = np.asarray(capped_mask)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(y_pred[~capped_mask], y_true[~capped_mask], alpha=0.3, s=12,
               label=f"Uncapped (n={(~capped_mask).sum():,})")
    ax.scatter(y_pred[capped_mask], y_true[capped_mask], alpha=0.6, s=14, color="red",
               label=f"Capped at $500,001 (n={capped_mask.sum():,})")
    lo = min(y_pred.min(), y_true.min())
    hi = max(y_pred.max(), y_true.max())
    ax.plot([lo, hi], [lo, hi], "k--", label="Ideal 1:1")
    ax.axhline(cap, color="red", linewidth=1, alpha=0.5)
    ax.annotate("Horizontal artifact: every home worth more than \\$500k\n"
                "is recorded as exactly \\$500,001",
                xy=(np.median(y_pred[capped_mask]), cap), xytext=(0.03, 0.93),
                textcoords="axes fraction", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("Predicted Price ($100k)")
    ax.set_ylabel("Actual Price ($100k)")
    ax.set_title("Baseline Linear Regression: Predicted vs. Actual (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def bias_by_band(model, X, y):
    """Mean (predicted - actual) per actual-price band, in dollars. >0 = overpriced."""
    pred = pd.Series(model.predict(X), index=y.index)
    band = pd.cut(y, PRICE_BANDS, labels=BAND_LABELS, include_lowest=True)
    err = (pred - y).groupby(band, observed=False)
    return pd.DataFrame({"n": err.size(), "mean_error_$": err.mean() * 100_000,
                         "mean_pred_$": pred.groupby(band, observed=False).mean() * 100_000})


def write_results_report(path, **kwargs):
    with open(path, "w") as f:
        f.write("# California Housing Cap Analysis Report\n\n")
        for k, v in kwargs.items():
            f.write(f"- **{k}**: {v}\n")


def main():
    # Load dataset
    data = fetch_california_housing(as_frame=True)
    features = data.data
    target = data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Detect the ceiling from the TRAINING data only (no test-set leakage)
    PRICE_CAP, n_at_cap, share_at_cap, spike_ratio, is_capped = detect_cap(y_train)
    if not is_capped:
        raise ValueError("No price ceiling detected in the training target.")
    if not np.isclose(PRICE_CAP, EXPECTED_CAP):
        print(f"Warning: detected cap {PRICE_CAP} differs from documented {EXPECTED_CAP}")
    capped_mask_all = target >= PRICE_CAP - 1e-9

    capped_train_mask = y_train >= PRICE_CAP - 1e-9
    capped_test_mask = y_test >= PRICE_CAP - 1e-9
    uncapped_test_mask = ~capped_test_mask

    # 1) Baseline model
    baseline_model = LinearRegression()
    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)

    baseline_full_metrics = (
        float(mean_squared_error(y_test, baseline_predictions)),
        float(r2_score(y_test, baseline_predictions)),
    )
    baseline_uncapped_metrics = get_metrics(
        baseline_model,
        X_test.loc[uncapped_test_mask],
        y_test.loc[uncapped_test_mask],
    )

    # 2) Cleaning: remove capped labels from the training set only.
    clean_train_mask = ~capped_train_mask
    X_train_clean = X_train.loc[clean_train_mask]
    y_train_clean = y_train.loc[clean_train_mask]

    cleaned_model = LinearRegression()
    cleaned_model.fit(X_train_clean, y_train_clean)

    # Evaluate both models on exactly the same uncapped test observations.
    cleaned_uncapped_metrics = get_metrics(
        cleaned_model,
        X_test.loc[uncapped_test_mask],
        y_test.loc[uncapped_test_mask],
    )

    # 3) Plot the baseline predictions for the entire test set, including caps.
    plot_path = OUTPUT_DIR / "baseline_predicted_vs_actual.png"
    make_baseline_plot(
        y_test,
        baseline_predictions,
        capped_test_mask,
        PRICE_CAP,
        plot_path,
    )

    # Bias by price band: both models, same uncapped test homes
    Xu, yu = X_test.loc[uncapped_test_mask], y_test.loc[uncapped_test_mask]
    base_bias = bias_by_band(baseline_model, Xu, yu)
    clean_bias = bias_by_band(cleaned_model, Xu, yu)
    bias = pd.DataFrame({
        "n": base_bias["n"],
        "baseline_mean_error_$": base_bias["mean_error_$"],
        "cleaned_mean_error_$": clean_bias["mean_error_$"],
        # Same homes, so this gap isolates what training on capped labels did:
        "cap_effect_on_prediction_$": base_bias["mean_pred_$"] - clean_bias["mean_pred_$"],
    })
    bias_path = OUTPUT_DIR / "bias_by_price_band.csv"
    bias.round(0).to_csv(bias_path, index_label="price_band")

    # 4) Save a metrics table for easy submission or transfer to the report.
    metrics_rows = [
        {
            "model": "Baseline",
            "evaluation_sample": "Entire test set",
            "n_evaluated": len(y_test),
            "n_capped_test": int(capped_test_mask.sum()),
            "MSE": baseline_full_metrics[0],
            "R_squared": baseline_full_metrics[1],
        },
        {
            "model": "Baseline",
            "evaluation_sample": "Uncapped test homes only",
            "n_evaluated": int(uncapped_test_mask.sum()),
            "n_capped_test": int(capped_test_mask.sum()),
            "MSE": baseline_uncapped_metrics[0],
            "R_squared": baseline_uncapped_metrics[1],
        },
        {
            "model": "Cleaned",
            "evaluation_sample": "Uncapped test homes only",
            "n_evaluated": int(uncapped_test_mask.sum()),
            "n_capped_test": int(capped_test_mask.sum()),
            "MSE": cleaned_uncapped_metrics[0],
            "R_squared": cleaned_uncapped_metrics[1],
        },
    ]
    metrics_path = OUTPUT_DIR / "model_metrics.csv"
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    report_path = OUTPUT_DIR / "defense_report_results.md"
    write_results_report(
        report_path,
        detected_cap=PRICE_CAP,
        cap_spike_ratio=round(spike_ratio, 1),
        n_rows=len(target),
        n_capped=int(capped_mask_all.sum()),
        n_train=len(y_train),
        n_capped_train=int(capped_train_mask.sum()),
        n_clean_train=len(y_train_clean),
        n_test=len(y_test),
        n_capped_test=int(capped_test_mask.sum()),
        n_uncapped_test=int(uncapped_test_mask.sum()),
        baseline_full=baseline_full_metrics,
        baseline_uncapped=baseline_uncapped_metrics,
        cleaned_uncapped=cleaned_uncapped_metrics,
    )

    print("\nCAP EFFECT CHECK")
    print(f"Total rows: {len(target):,}")
    print(f"Rows at the $500,001 cap: {int(capped_mask_all.sum()):,}")
    print(f"Training rows before cleaning: {len(y_train):,}")
    print(f"Capped training rows removed: {int(capped_train_mask.sum()):,}")
    print(f"Training rows after cleaning: {len(y_train_clean):,}")

    print("\nMODEL PERFORMANCE")
    print("Baseline, entire test set:          "
          f"MSE={baseline_full_metrics[0]:.6f}, R^2={baseline_full_metrics[1]:.6f}")
    print("Baseline, uncapped test homes:      "
          f"MSE={baseline_uncapped_metrics[0]:.6f}, R^2={baseline_uncapped_metrics[1]:.6f}")
    print("Cleaned, uncapped test homes:        "
          f"MSE={cleaned_uncapped_metrics[0]:.6f}, R^2={cleaned_uncapped_metrics[1]:.6f}")

    print(f"\nDetected cap: {PRICE_CAP} ({n_at_cap:,} training rows, {share_at_cap:.1%}; "
          f"{spike_ratio:.0f}x as common as the values just below it)")
    print("\nBIAS BY PRICE BAND (uncapped test homes, $; >0 = overpriced)")
    print(bias.round(0).to_string())

    print("\nSaved files:")
    print(f"- {plot_path}")
    print(f"- {metrics_path}")
    print(f"- {report_path}")
    print(f"- {bias_path}")


if __name__ == "__main__":
    main()
