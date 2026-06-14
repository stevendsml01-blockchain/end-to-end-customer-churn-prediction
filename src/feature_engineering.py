import pandas as pd
import joblib
from sklearn.model_selection import train_test_split


print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# ==================================================
# LOAD DATASET
# ==================================================

print("\nLoading labeled dataset...")

df = pd.read_csv(
    "data/processed/customer_churn_labeled.csv"
)

print("Dataset loaded successfully!")
print(f"Dataset Shape: {df.shape}")

# ==================================================
# DROP UNNECESSARY COLUMNS
# ==================================================

print("\nDropping unnecessary columns...")

df = df.drop(columns=["Customer_ID"])

print("Customer_ID removed.")

# ==================================================
# SPLIT FEATURES AND TARGET
# ==================================================

print("\nSeparating features and target...")

X = df.drop(columns=["Churn", "Risk_Score", "Churn_Probability"])
y = df["Churn"]

print(f"Features Shape: {X.shape}")
print(f"Target Shape: {y.shape}")

# ==================================================
# ONE-HOT ENCODING
# ==================================================

print("\nApplying one-hot encoding...")

categorical_columns = [
    "Gender",
    "Contract",
    "Payment_Method",
    "Internet_Service"
]

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=True
)

print("Encoding completed.")

print("\nEncoded Feature Shape:")
print(X.shape)

print("\nFeature Names:")

for column in X.columns:
    print(column)

# ==================================================
# TRAIN-TEST SPLIT
# ==================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Train-test split completed.")

print(f"\nX_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")

print(f"\ny_train: {y_train.shape}")
print(f"y_test : {y_test.shape}")

# ==================================================
# SAVE FILES
# ==================================================

print("\nSaving processed files...")

X_train.to_csv(
    "data/processed/X_train.csv",
    index=False
)

X_test.to_csv(
    "data/processed/X_test.csv",
    index=False
)

y_train.to_frame().to_csv(
    "data/processed/y_train.csv",
    index=False
)

y_test.to_frame().to_csv(
    "data/processed/y_test.csv",
    index=False
)

print("Files saved successfully!")

# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print("\nGenerated Files:")

print("data/processed/X_train.csv")
print("data/processed/X_test.csv")
print("data/processed/y_train.csv")
print("data/processed/y_test.csv")

print("\nDataset is ready for model training!")

joblib.dump(
    X.columns.tolist(),
    "models/feature_columns.pkl"
)

print(
    "Feature columns saved successfully!"
)