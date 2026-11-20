#presets

from __future__ import annotations

#Part 1 of Project 1 - CSCI 270

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from sklearn.datasets import load_breast_cancer

#loading dataset
cancer = load_breast_cancer(as_frame=True)

X = cancer.data
y = cancer.target

print("Dataset shape:", X.shape)
print("Target names:", cancer.target_names)

print("\nFirst 5 rows of the dataset:")
print(X.head())

print("\nClass distribution:")
print(y.value_counts())

from sklearn.model_selection import train_test_split

y = (cancer.target == 0).astype(int)  # Convert to binary classification (malignant=1, benign=0


#split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

print("\nClass distribution in training set:")
print(y_train.value_counts())

print("\nClass distribution in testing set:")
print(y_test.value_counts())

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (

accuracy_score,
recall_score,
confusion_matrix,
classification_report
)

# create the baseline model

baseline_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

#train
baseline_model.fit(X_train, y_train)

#make predicitons
baseline_predicitons = baseline_model.predict(X_test)

#Evaluate the model
baseline_accuracy = accuracy_score(y_test, baseline_predicitons)
baseline_recall = recall_score(y_test, baseline_predicitons)

print("\nBaseline Model Evaluation:")
print("Accuracy:", baseline_accuracy)
print("Recall:", baseline_recall)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, baseline_predicitons))

print("\nClassification Report:")
print(classification_report(
    y_test, 
    baseline_predicitons, 
    target_names=['Benign', 'Malignant']
    ))

# recall focused model

# Get the probability of a malignant tumor
malignant_probabilities = baseline_model.predict_proba(X_test)[:, 1]

# Set the decision threshold
threshold = 0.30
    #decision threshold was initially lower, but was raised to 0.30 after testing various thresholds
    

# Make predictions using the new threshold
recall_predictions = (malignant_probabilities >= threshold).astype(int)

# Evaluate the model
recall_accuracy = accuracy_score(y_test, recall_predictions)
recall_recall = recall_score(y_test, recall_predictions)

print("\n=== RECALL-FOCUSED MODEL ===")
print("Decision Threshold:", threshold)
print("Accuracy:", recall_accuracy)
print("Malignant Recall:", recall_recall)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, recall_predictions))

# Threshold comparison - part 1

print("\n=== THRESHOLD COMPARISON ===")

for threshold in [0.5, 0.4, 0.3, 0.25, 0.2, 0.1]:

    predictions = (
        malignant_probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(y_test, predictions)

    recall = recall_score(y_test, predictions)

    matrix = confusion_matrix(y_test, predictions)

    false_negatives = matrix[1, 0]

    false_positives = matrix[0, 1]

    print("\nThreshold:", threshold)
    print("Accuracy:", round(accuracy, 4))
    print("Malignant Recall:", round(recall, 4))
    print("False Negatives:", false_negatives)
    print("False Positives:", false_positives)


# Precision curve - part 1

precision_values, recall_values, thresholds = precision_recall_curve(
    y_test,
    malignant_probabilities
)

plt.figure(figsize=(10, 7))

# Plot the precision-recall curve
plt.plot(
    recall_values,
    precision_values,
    color="blue",
    linewidth=2,
    label="Precision-Recall Curve"
)

# Find the point closest to threshold 0.30
threshold = 0.30
index = np.argmin(np.abs(thresholds - threshold))

precision = precision_values[index]
recall = recall_values[index]

# Highlight the 0.30 threshold
plt.scatter(
    recall,
    precision,
    color="red",
    s=120,
    edgecolors="black",
    zorder=5,
    label="Threshold = 0.30"
)

# Label the point
plt.annotate(
    "Threshold 0.30",
    (recall, precision),
    xytext=(10, 10),
    textcoords="offset points",
    fontsize=10
)

print(
    f"Threshold 0.30: "
    f"Precision = {precision:.4f}, "
    f"Recall = {recall:.4f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Breast Cancer Classification")

plt.xlim(0, 1.05)
plt.ylim(0, 1.05)

plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("part1_precision_recall_curve.png", dpi=300)

plt.show()

#Part 2 of Project 1 - CSCI 270

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


import argparse
import json
from pathlib import Path


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

#part 3 of project 1 - CSCI 270

# Owen Catron
# CSCI 270 PROJECT 1 PART 3 SUBMISSION

# Import necessary libraries
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

import os

# Acquire directory to identify where the CSV is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Mall_Customers.csv')

# initialize variables to be used for iterative K values on both clusters
K_range = range(2, 11)

# 1. Age vs. Spending Score (1-100)

inertias1 = [] # for the elbow method in the first clustering example
kmeans1_k5 = None # initializes the variable to save for later when graphing the best K

# reads CSV and assigns to a variable
df1 = pd.read_csv(csv_path, usecols=['Age', 'Spending Score (1-100)'])
X = df1

print(f"For the Age vs. Spending Score model, here are the k values 2-10.")

for k in K_range:
    # initializes K-Means model
    kmeans1 = KMeans(n_clusters=k, random_state=42)
    kmeans1.fit(X)

    inertias1.append(kmeans1.inertia_) # tracking inertia for the elbow curve

    score1 = silhouette_score(X, kmeans1.labels_) # calculates silhouette score

    print(f"Cluster centroids:\n{kmeans1.cluster_centers_}")
    print(f"K-value: {k}")
    print(f"Silhouette Score: {score1:.3f}")

    if k == 5:
        kmeans1_k5 = kmeans1 # recalls the variable and assigns to optimal K of 5
        silhouette1_k5 = score1 # saves for silhouette comparison

    if k == 6:
        silhouette1_k6 = score1 # saves for silhouette comparison

# sets up first window to display scatter (K-Means) and elbow method (inertia vs K)
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5)) 
fig1.canvas.manager.set_window_title("Figure 1: Age vs. Spending Score")

# Scatter of first clustering
axes1[0].scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans1_k5.labels_, cmap='rainbow')
axes1[0].scatter(kmeans1_k5.cluster_centers_[:, 0], 
kmeans1_k5.cluster_centers_[:, 1], s=300, marker='X', c='black', label='Centroids')
axes1[0].set_title("K-Means Clustering (K=5): Age vs. Spending Score")
axes1[0].set_xlabel('Age')
axes1[0].set_ylabel('Spending Score (1-100)')
axes1[0].legend()

# Elbow method for first clustering
axes1[1].plot(K_range, inertias1, 'bo-')
axes1[1].set_title('Elbow Method for Optimal K in Age vs. Spending Score')
axes1[1].set_xlabel('Number of clusters (K)')
axes1[1].set_ylabel('Inertia')
axes1[1].grid(True)

plt.tight_layout()
plt.show() 

# 2. Annual Income (k$) vs. Spending Score (1-100)

inertias2 = [] # for the elbow method in the second clustering example
kmeans2_k5 = None # initializes the variable to save for later when graphing the best K

# reads CSV and assigns to a variable
df2 = pd.read_csv(csv_path, usecols=['Annual Income (k$)', 'Spending Score (1-100)'])
y = df2

print(f"\nFor the Annual Income vs. Spending Score model, here are the k values 2-10.")

for k in K_range:
    # initializes second K-Means model
    kmeans2 = KMeans(n_clusters=k, random_state=42)
    kmeans2.fit(y)

    inertias2.append(kmeans2.inertia_) # tracking inertia for the elbow curve

    score2 = silhouette_score(y, kmeans2.labels_) # calculates silhouette score

    print(f"Cluster centroids:\n{kmeans2.cluster_centers_}")
    print(f"K-value: {k}")
    print(f"Silhouette Score: {score2:.3f}")
    
    if k == 5:
        kmeans2_k5 = kmeans2 # recalls variable and assigns to optimal K of 5
        silhouette2_k5 = score2 # saves for silhouette comparison

    if k == 6:
        silhouette2_k6 = score2 # saves for silhouette comparison

# sets up second window to display scatter (K-Means) and elbow method (inertia vs K)
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.canvas.manager.set_window_title("Figure 2: Annual Income vs. Spending Score")

# Scatter of second clustering
axes2[0].scatter(y.iloc[:, 0], y.iloc[:, 1], c=kmeans2_k5.labels_, cmap='rainbow')
axes2[0].scatter(kmeans2_k5.cluster_centers_[:, 0], 
kmeans2_k5.cluster_centers_[:, 1], s=300, marker='X', c='black', label='Centroids')
axes2[0].set_title("K-Means Clustering (K=5): Annual Income vs. Spending Score")
axes2[0].set_xlabel('Annual Income (k$)')
axes2[0].set_ylabel('Spending Score (1-100)')
axes2[0].legend()

# Elbow method for second clustering
axes2[1].plot(K_range, inertias2, 'bo-')
axes2[1].set_title('Elbow Method for Optimal K in Annual Income vs. Spending Score')
axes2[1].set_xlabel('Number of clusters (K)')
axes2[1].set_ylabel('Inertia')
axes2[1].grid(True)

plt.tight_layout()
plt.show() 

# 3. Silhouette Score, Optimal K Value, and Business Pitch Justifications

# definition and formula
print(f"\n--- Silhouette Score Definition ---")
print(f"\nThe silhouette score determines how dense and correllated the data points are for each cluster, "
    f"and also measures how well each cluster is separated from its nearest neighbors.")
print(f"The formula is given by the difference between separation and cohesion "
    f"over the maximum of the two values.")

# importance of comparison
print(f"\n--- Importance of Silhouette Comparisons ---")
print(f"\nWhen comparing the silhouette scores, it is relevant to consider the higher value "
    f"of each K-Means integer when comparing against K = 5 and K = 6.")
print(f"The higher the silhouette score decimal (closest to 1) indicates how strongly the clusters "
    f"are separated, and how common it is for clusters to overlap with other clusters.")

# identifying each silhouette score value to determine which clustering is ideal
print(f"\n--- Silhouette Score Value Comparisons ---")
print(f"\nFor K = 5 in the Age vs. Spending Score model, the silhouette score is: {silhouette1_k5:.3f}"
    f"\nFor K = 6 in the Age vs. Spending Score model, the silhouette score is: {silhouette1_k6:.3f}")
print(f"For K = 5 in the Annual Income vs. Spending Score model, the silhouette score is: {silhouette2_k5:.3f}"
    f"\nFor K = 6 in the Annual Income vs. Spending Score model, the silhouette score is: {silhouette2_k6:.3f}")

# technical justification of K = 5 vs K = 6 for both clustering samples
print(f"\n--- Selected Optimized K Value Technical Justification ---")
print(f"\nWe chose K = 5 for both the Age vs. Spending Score and the Annual Income vs. Spending Score clustering models.")
print(f"As shown in the comparison for both clustering models, none of the colored data clusters"
f" have significantly noticeable overlap with one another.")
print(f"The silhouette scores for both clustering models are higher for K = 5 when compared to K = 6.")
print(f"Both are close to being reasonable for explaining the distribution of their clustering, "
f"although the borders in each model are visually narrow. Even still, there is no overlap, unlike minor outliers or misclassification in K = 6.")
print(f"The most important observation made comes with looking at the elbow method curve in Annual Income vs. Spending Score.")
print(f"This curve has steep rates of decline between integers K = 2 and K = 5, until the ranges of K = 5 to K = 10.")
print(f"This is the purpose of the elbow curve, to locate where"
" the rate of decline begins to diminish, which indicates the data point with the optimal K value.")
print(f"While the Age vs. Spending Score model observes an optimal K of 4 or 6, the stark falloff "
f"between Income and Score in terms of inertia vs. K-values is more defined.")
print(f"This constitutes for an argumentative technical analysis for either K = 5 or K = 6, although the "
    f"insights from being a mall owner have greater support to K = 5 for its narrowly defined business strategy.")

# BUSINESS PITCH: explaining the K = 5 decision on number of clusters for models
print(f"\n--- BUSINESS DEFENSE PITCH: K = 5 CLUSTERS ---")
print(f"\nWhen considering modeling the data for Age and Annual Income, each against Spending Score,"
    f" we need to recognize the target demographics for the marketing team to develop a competitive strategy.")
print(f"Viewing the Age vs. Spending Score graph at K = 5 clusters helps us, the analysts, realize portions "
    f"of the data whose locations have feature similarities by concise sectioning in the scatter plot.")
print(f"The red, blue, yellow, purple, and green clusters have centroids that can easily categorize "
    f"and have easier interpretability of this data. Having any additional cluster counts (e.g. 6)"
    f" would be a broader model of analyzing the correlation between two features, which are more likely "
    f"to contain overlap, meaning an unclear grouping of characteristics. This means a harder time appealing customers in each group.")
print(f"Notably, for the centroids of the Annual Income vs. Spending score graph, a vast majority have "
    f"greater point distance variations, with the exception of the purple cluster.")
print(f"By this example, it is important for the marketing team to identify closely related variables for their customer bases,"
    f" which is why it can be determined for them that most customers of a salary of $40,000 to approx. $65,000 "
    f"would have a spending score of 40 to approx. 60. As the mall owner, we need to trust that the marketing team "
    f"will be able to notice patterns in the data for segmentation and advertising purposes, like with this example.")
print(f"To give another example, on the Age. vs. Spending Score model, the general patterns seem to outline how "
    f"the younger demographics tend to have a higher spending score, while older generations would be less "
    f"inclined to spend, thus having a lower spending score. This becomes important to note for determining frequent mall visitors.")
print(f"The scatter data, as it stands, have easy to distinguish clusters at a K score of 5.")
print(f"More clusters would correspond to the marketing team having to focus on additional, possibly redundant "
    f"generalizations of features, leading to a potentially less effective competitive strategy.")
print(f"While not shown, when analyzing the K = 6 scatter plots, the data points already begin to become potentially mismatched - "
    f"not only having overlap, but miscolored points as well, only to justify the need for a specific number of circular clusters.")
print(f"From a business viewpoint, K = 6 leaves the data harder to distinguish.\n")

print(f"\nIn short, K = 5 is a more valuable representation for a marketing team to develop a business strategy.")
print(f"The 5 clusters are articulated so that we can outline their demographics as well as their quantity of visitors, "
    f"so the outcomes for the strategy (advertising, coupons, pricing of vendors or stores) can ultimately attract more customers,"
    f" based on the data in the scatter plots that have been gathered so far.")

#part 4 of project 1 - CSCI 270


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Aotizhongxin monitoring station
df = pd.read_csv(
    "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
)

# Combine date columns into a datetime index
df["datetime"] = pd.to_datetime(
    df[["year", "month", "day", "hour"]]
)

df = df.set_index("datetime")

# Isolate PM2.5 measurements
pm25 = df["PM2.5"]


# Investigate consecutive missing PM2.5 measurements

missing = pm25.isna()

# Identify groups of consecutive missing values
groups = (missing != missing.shift()).cumsum()

# Calculate the length of each missing-data gap
missing_runs = missing[missing].groupby(groups[missing]).size()

print("\nMISSING DATA GAP ANALYSIS")
print("Number of separate missing gaps:", len(missing_runs))
print("Longest missing gap:", missing_runs.max(), "hours")
print("Average missing gap:", round(missing_runs.mean(), 2), "hours")

print("\nTen longest missing gaps:")
print(missing_runs.sort_values(ascending=False).head(10))

# Categorize missing gaps by length
print("\nGAP LENGTH DISTRIBUTION")
print("1-6 hours:", ((missing_runs >= 1) & (missing_runs <= 6)).sum())
print("7-24 hours:", ((missing_runs >= 7) & (missing_runs <= 24)).sum())
print("25-72 hours:", ((missing_runs >= 25) & (missing_runs <= 72)).sum())
print("Over 72 hours:", (missing_runs > 72).sum())


# Check for missing values
print("Total observations:", len(pm25))
print("Missing PM2.5 values:", pm25.isna().sum())

# Handle missing values using linear interpolation  

# Handle missing PM2.5 measurements

# Identify missing values
missing = pm25.isna()

# Identify groups of consecutive missing values
groups = (missing != missing.shift()).cumsum()

# Calculate the length of each missing gap
gap_lengths = missing.groupby(groups).transform("sum")

# Identify missing gaps of 6 hours or shorter
short_missing = missing & (gap_lengths <= 6)

# Perform linear interpolation using timestamps
interpolated = pm25.interpolate(
    method="time",
    limit_area="inside"
)

# Fill only short gaps
pm25_clean = pm25.copy()
pm25_clean.loc[short_missing] = interpolated.loc[short_missing]

# Display correction results
print("\nMISSING DATA CORRECTION")
print("Values interpolated:", short_missing.sum())
print("Values still missing:", pm25_clean.isna().sum())


# Resample hourly measurements into daily averages

# Calculate daily PM2.5 averages

daily_pm25 = pm25_clean.resample("D").mean()

# Count valid hourly measurements per day
daily_counts = pm25.resample("D").count()

# Require at least 18 valid hourly measurements
daily_pm25[daily_counts < 18] = np.nan

print("\nDAILY AVERAGE ANALYSIS")
print("Total days:", len(daily_pm25))
print("Days with valid averages:", daily_pm25.notna().sum())
print("Days without valid averages:", daily_pm25.isna().sum())


# Fit a linear trend using actual elapsed calendar days

valid_daily = daily_pm25.dropna()

# Calculate days since the first valid observation
x = (valid_daily.index - valid_daily.index[0]).days
y = valid_daily.values

# Calculate best-fit line
slope, intercept = np.polyfit(x, y, 1)

trend = slope * x + intercept

# Calculate annual change
annual_change = slope * 365.25

print("\nCORRECTED TREND ANALYSIS")
print("Daily trend:", round(slope, 6), "µg/m³ per day")
print("Estimated annual change:", round(annual_change, 4), "µg/m³ per year")


# Plot daily averages and trend line
plt.figure(figsize=(12, 6))

plt.scatter(
    valid_daily.index,
    valid_daily.values,
    s=8,
    alpha=0.4,
    label="Daily PM2.5 averages"
)

plt.plot(
    valid_daily.index,
    trend,
    color="red",
    linewidth=2,
    label="Linear trend line"
)

plt.title("Daily PM2.5 Concentration at Aotizhongxin")
plt.xlabel("Date")
plt.ylabel("PM2.5 Concentration (µg/m³)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("PM25_Corrected_Graph.png", dpi=300)
plt.show()
