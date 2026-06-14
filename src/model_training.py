import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("LOGISTIC REGRESSION TRAINING")
print("=" * 60)

# ==================================================
# LOAD DATA
# ==================================================

print("\nLoading datasets...")

X_train = pd.read_csv(
    "data/processed/X_train.csv"
)

X_test = pd.read_csv(
    "data/processed/X_test.csv"
)

y_train = pd.read_csv(
    "data/processed/y_train.csv"
).squeeze()

y_test = pd.read_csv(
    "data/processed/y_test.csv"
).squeeze()

print("Datasets loaded successfully!")

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

# ==================================================
# TRAIN MODEL
# ==================================================

print("\nTraining Logistic Regression...")

model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Model trained successfully!")

# ==================================================
# PREDICTION
# ==================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

print("Predictions completed!")

# ==================================================
# EVALUATION
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(f"{accuracy:.4f}")

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "No Churn",
        "Churn"
    ]
)

print(report)

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

# ==================================================
# SAVE REPORT
# ==================================================

with open(
    "reports/classification_report.txt",
    "w"
) as f:
    f.write(report)

print("\nClassification report saved!")

# ==================================================
# SAVE MODEL
# ==================================================

joblib.dump(
    model,
    "models/logistic_regression.pkl"
)

print("Model saved successfully!")

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

print("\nFeature Coefficients:")

coefficients = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": model.coef_[0]
})

coefficients = coefficients.sort_values(
    by="Coefficient",
    ascending=False
)

print(coefficients)

print("\nTop 5 Churn Drivers:")

print(
    coefficients.head(5)
)

print("\nTop 5 Retention Drivers:")

print(
    coefficients.tail(5)
)

# ==================================================
# PROBABILITY EXAMPLE
# ==================================================

print("\nExample Churn Probabilities:")

example_probs = pd.DataFrame({
    "Actual": y_test.iloc[:10].values,
    "Prediction": y_pred[:10],
    "Churn_Probability": (
        y_prob[:10] * 100
    ).round(2)
})

print(example_probs)

print("\n" + "=" * 60)
print("PHASE 5 REBUILD COMPLETED")
print("=" * 60)