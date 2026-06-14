import pandas as pd
import numpy as np

print("=" * 60)
print("GENERATING REALISTIC CHURN LABELS")
print("=" * 60)

# ==================================================
# LOAD CLEANED DATASET
# ==================================================

df = pd.read_csv(
    "data/processed/cleaned_customer_churn.csv"
)

print("\nDataset loaded successfully!")
print(f"Rows: {len(df)}")

# ==================================================
# RANDOM SEED
# ==================================================

np.random.seed(42)

# ==================================================
# GENERATE RISK SCORE
# ==================================================

df["Risk_Score"] = 0

# Contract
df.loc[
    df["Contract"] == "Month-to-month",
    "Risk_Score"
] += 3

df.loc[
    df["Contract"] == "One Year",
    "Risk_Score"
] += 1

# Tenure
df.loc[
    df["Tenure"] <= 12,
    "Risk_Score"
] += 2

# Monthly Charges
df.loc[
    df["Monthly_Charges"] > 70,
    "Risk_Score"
] += 2

# Payment Method
df.loc[
    df["Payment_Method"] == "Electronic Check",
    "Risk_Score"
] += 1

# Internet Service
df.loc[
    df["Internet_Service"] == "Fiber Optic",
    "Risk_Score"
] += 1

print("\nRisk scores generated successfully!")

# ==================================================
# CONVERT RISK SCORE TO PROBABILITY
# ==================================================

probability_mapping = {
    0: 0.05,
    1: 0.10,
    2: 0.20,
    3: 0.30,
    4: 0.40,
    5: 0.50,
    6: 0.60,
    7: 0.70,
    8: 0.80,
    9: 0.90,
    10: 0.95
}

df["Churn_Probability"] = (
    df["Risk_Score"]
    .map(probability_mapping)
    .fillna(0.95)
)

# ==================================================
# GENERATE REALISTIC CHURN LABELS
# ==================================================

random_numbers = np.random.rand(len(df))

df["Churn"] = (
    random_numbers < df["Churn_Probability"]
).astype(int)

print("\nRealistic churn labels generated!")

# ==================================================
# DISPLAY DISTRIBUTION
# ==================================================

print("\nChurn Distribution:")

print(
    df["Churn"]
    .value_counts()
    .rename({
        0: "No Churn",
        1: "Churn"
    })
)

print("\nChurn Percentage:")

print(
    round(
        df["Churn"]
        .value_counts(normalize=True) * 100,
        2
    )
)

# ==================================================
# DISPLAY RISK ANALYSIS
# ==================================================

print("\nAverage Churn Rate by Risk Score:")

risk_analysis = (
    df.groupby("Risk_Score")["Churn"]
    .mean()
    .round(3)
    * 100
)

print(risk_analysis)

# ==================================================
# SAVE DATASET
# ==================================================

df.to_csv(
    "data/processed/customer_churn_labeled.csv",
    index=False
)

print("\nLabeled dataset saved successfully!")

print(
    "\nOutput File:"
)

print(
    "data/processed/customer_churn_labeled.csv"
)

print("\n" + "=" * 60)
print("PHASE 3 REBUILD COMPLETED")
print("=" * 60)