"""
Artificial Intelligence & Machine Learning - Task 5
Ensemble Learning, Model Optimization & Real-World ML Pipeline
Dataset: Breast Cancer (sklearn built-in)
"""

# ============================================================
# Part A: Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# Part B: Load Dataset
# ============================================================
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset shape:", X.shape)
print("Class distribution:\n", pd.Series(y).value_counts())

# ============================================================
# Part C: Train-Test Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# Part D: Feature Scaling
# (Needed for Logistic Regression; tree-based models don't require it)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# Part E: Baseline Model (Logistic Regression)
# ============================================================
print("\n" + "=" * 60)
print("Baseline Model: Logistic Regression")
print("=" * 60)

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_scaled, y_train)

y_pred_lr = model_lr.predict(X_test_scaled)
print(classification_report(y_test, y_pred_lr))

# ============================================================
# Part F: Random Forest Model
# ============================================================
print("\n" + "=" * 60)
print("Model 2: Random Forest Classifier")
print("=" * 60)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# ============================================================
# Part G: Gradient Boosting Model
# ============================================================
print("\n" + "=" * 60)
print("Model 3: Gradient Boosting Classifier")
print("=" * 60)

gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
print(classification_report(y_test, y_pred_gb))

# ============================================================
# Part H: Hyperparameter Tuning (GridSearchCV)
# ============================================================
print("\n" + "=" * 60)
print("Hyperparameter Tuning: GridSearchCV on Random Forest")
print("=" * 60)

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 5, 7],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best CV F1 Score: {:.4f}".format(grid.best_score_))

best_rf = grid.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
print("\nTuned Random Forest - Test Performance:")
print(classification_report(y_test, y_pred_best_rf))

# ============================================================
# Part I: Model Comparison
# ============================================================
def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
    }


results = [
    evaluate_model("Logistic Regression (Baseline)", y_test, y_pred_lr),
    evaluate_model("Random Forest", y_test, y_pred_rf),
    evaluate_model("Gradient Boosting", y_test, y_pred_gb),
    evaluate_model("Random Forest (Tuned)", y_test, y_pred_best_rf),
]

results_df = pd.DataFrame(results).sort_values(by="F1-score", ascending=False)
print("\n" + "=" * 60)
print("Model Comparison Summary")
print("=" * 60)
print(results_df.to_string(index=False))

# Bar chart comparison
results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
    kind="bar", figsize=(10, 6)
)
plt.title("Model Comparison - Breast Cancer Classification")
plt.ylabel("Score")
plt.ylim(0.8, 1.0)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()

# ============================================================
# Part J: Final Model Selection
# ============================================================
best_model_row = results_df.iloc[0]
print("\n" + "=" * 60)
print("Final Model Selection")
print("=" * 60)
print(f"Best performing model: {best_model_row['Model']}")
print(f"Accuracy:  {best_model_row['Accuracy']:.4f}")
print(f"Precision: {best_model_row['Precision']:.4f}")
print(f"Recall:    {best_model_row['Recall']:.4f}")
print(f"F1-score:  {best_model_row['F1-score']:.4f}")

print(
    """
Justification:
- Ensemble models (Random Forest, Gradient Boosting) generally outperform
  the single Logistic Regression baseline because they combine multiple
  weak/strong learners, reducing variance (Random Forest) and bias
  (Gradient Boosting).
- Hyperparameter tuning via GridSearchCV further improves stability and
  generalization by selecting the optimal n_estimators/max_depth
  combination validated through cross-validation.
- The final model is chosen based on the highest F1-score, balancing
  precision and recall, which is important for medical diagnosis tasks
  such as breast cancer classification where both false positives and
  false negatives carry real costs.
"""
)

# ============================================================
# Optional: Feature Importance (Random Forest)
# ============================================================
importances = pd.Series(best_rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 6))
top_features.sort_values().plot(kind="barh")
plt.title("Top 10 Feature Importances - Tuned Random Forest")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()

print("\nScript finished. Plots saved as 'model_comparison.png' and 'feature_importance.png'.")
