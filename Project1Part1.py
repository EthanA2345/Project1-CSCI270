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
    #decision threshold was initially lower, but was raised to 3.0 after testing various thresholds
    

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

