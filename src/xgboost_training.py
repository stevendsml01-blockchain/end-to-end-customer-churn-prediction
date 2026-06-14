import pandas as pd
import joblib

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 60)
print("XGBOOST TRAINING")
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

# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining XGBoost...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

print("Model trained successfully!")

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# ==========================================
# EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(round(accuracy, 4))

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
    "reports/xgboost_report.txt",
    "w"
) as f:
    f.write(report)

print("\nReport saved successfully!")

# ==========================================
# SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/xgboost_model.pkl"
)

print("Model saved successfully!")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")

print(importance)

print("\nTop 5 Churn Drivers:")

print(importance.head())

# ==========================================
# EXAMPLE PROBABILITIES
# ==========================================

print("\nExample Churn Probabilities:")

examples = pd.DataFrame({
    "Actual": y_test.iloc[:10],
    "Prediction": y_pred[:10],
    "Churn_Probability":
        (y_prob[:10] * 100).round(2)
})

print(examples)

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("XGBOOST COMPLETED")
print("=" * 60)