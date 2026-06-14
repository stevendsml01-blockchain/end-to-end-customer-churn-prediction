import pandas as pd
import joblib

print("=" * 60)
print("CHURN PREDICTION")
print("=" * 60)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(
    "models/logistic_regression.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

print("\nModel loaded successfully!")

# ==========================================
# NEW CUSTOMER DATA
# ==========================================

customer = {
    "Gender": "Male",
    "Senior_Citizen": 0,
    "Tenure": 8,
    "Monthly_Charges": 85,
    "Total_Charges": 680,
    "Contract": "Month-to-month",
    "Payment_Method": "Electronic Check",
    "Internet_Service": "Fiber Optic"
}

print("\nCustomer Data:")

for key, value in customer.items():
    print(f"{key}: {value}")

# ==========================================
# CONVERT TO DATAFRAME
# ==========================================

customer_df = pd.DataFrame([customer])

# ==========================================
# ONE-HOT ENCODING
# ==========================================

customer_df = pd.get_dummies(
    customer_df,
    drop_first=True
)

# ==========================================
# MATCH TRAINING COLUMNS
# ==========================================

customer_df = customer_df.reindex(
    columns=feature_columns,
    fill_value=0
)

# ==========================================
# PREDICTION
# ==========================================

prediction = model.predict(
    customer_df
)[0]

probability = (
    model.predict_proba(customer_df)[0][1]
    * 100
)

# ==========================================
# DISPLAY RESULT
# ==========================================

print("\nPrediction:")

if prediction == 1:
    print("Customer WILL CHURN")
else:
    print("Customer will NOT churn")

print("\nProbability:")

print(f"{probability:.2f}%")

print("\nRisk Level:")

if probability >= 70:
    print("HIGH RISK")

elif probability >= 40:
    print("MEDIUM RISK")

else:
    print("LOW RISK")

print("\n" + "=" * 60)
print("PHASE 9 COMPLETED")
print("=" * 60)