import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("RANDOM FOREST TRAINING")
print("=" * 60)

# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading datasets...")

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

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

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully!")

# ==========================================
# PREDICTION
# ==========================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

print("Predictions completed!")

# ==========================================
# EVALUATION
# ==========================================

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

# ==========================================
# SAVE REPORT
# ==========================================

with open(
    "reports/random_forest_report.txt",
    "w"
) as f:
    f.write(report)

print("\nReport saved successfully!")

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/random_forest.pkl"
)

print("Model saved successfully!")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print("\nFeature Importance:")

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

print("\nTop 5 Churn Drivers:")

print(
    importance.head()
)

# ==========================================
# EXAMPLE PROBABILITIES
# ==========================================

print("\nExample Churn Probabilities:")

examples = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Prediction": y_pred[:10],
    "Churn_Probability":
        (y_prob[:10] * 100).round(2)
})

print(examples)

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("RANDOM FOREST COMPLETED")
print("=" * 60)