import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score
)

print("=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

# ======================================
# LOAD TEST DATA
# ======================================

X_test = pd.read_csv(
    "data/processed/X_test.csv"
)

y_test = pd.read_csv(
    "data/processed/y_test.csv"
).squeeze()

print("\nTest dataset loaded!")

# ======================================
# LOAD MODELS
# ======================================

logistic_model = joblib.load(
    "models/logistic_regression.pkl"
)

rf_model = joblib.load(
    "models/random_forest.pkl"
)

xgb_model = joblib.load(
    "models/xgboost_model.pkl"
)

print("All models loaded!")

# ======================================
# PROBABILITIES
# ======================================

logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

rf_prob = rf_model.predict_proba(X_test)[:, 1]

xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

# ======================================
# AUC
# ======================================

logistic_auc = roc_auc_score(
    y_test,
    logistic_prob
)

rf_auc = roc_auc_score(
    y_test,
    rf_prob
)

xgb_auc = roc_auc_score(
    y_test,
    xgb_prob
)

print("\nAUC Scores:")

print(
    f"Logistic Regression: {logistic_auc:.4f}"
)

print(
    f"Random Forest: {rf_auc:.4f}"
)

print(
    f"XGBoost: {xgb_auc:.4f}"
)

# ======================================
# ROC CURVES
# ======================================

logistic_fpr, logistic_tpr, _ = roc_curve(
    y_test,
    logistic_prob
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_prob
)

xgb_fpr, xgb_tpr, _ = roc_curve(
    y_test,
    xgb_prob
)

# ======================================
# PLOT
# ======================================

plt.figure(figsize=(8, 6))

plt.plot(
    logistic_fpr,
    logistic_tpr,
    label=f"Logistic (AUC={logistic_auc:.3f})"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC={rf_auc:.3f})"
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    label=f"XGBoost (AUC={xgb_auc:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()

plt.savefig(
    "reports/roc_curve.png"
)

plt.show()

print("\nROC Curve saved!")

print(
    "reports/roc_curve.png"
)

print("\n" + "=" * 60)
print("PHASE 8.2 COMPLETED")
print("=" * 60)