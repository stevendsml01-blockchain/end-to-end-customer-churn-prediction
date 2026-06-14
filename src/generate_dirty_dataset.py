import pandas as pd
import numpy as np
import random

# Reproducibility
random.seed(42)
np.random.seed(42)

NUM_CUSTOMERS = 5000

print("=" * 50)
print("GENERATING DIRTY CUSTOMER CHURN DATASET")
print("=" * 50)

# -----------------------------
# Generate Base Dataset
# -----------------------------

customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, NUM_CUSTOMERS + 1)]

genders = np.random.choice(
    ["Male", "Female"],
    size=NUM_CUSTOMERS,
    p=[0.5, 0.5]
)

senior_citizen = np.random.choice(
    [0, 1],
    size=NUM_CUSTOMERS,
    p=[0.84, 0.16]
)

tenure = np.random.randint(1, 73, NUM_CUSTOMERS)

monthly_charges = np.round(
    np.random.uniform(20, 120, NUM_CUSTOMERS),
    2
)

total_charges = np.round(
    tenure * monthly_charges,
    2
)

contracts = np.random.choice(
    ["Month-to-month", "One Year", "Two Year"],
    size=NUM_CUSTOMERS,
    p=[0.55, 0.25, 0.20]
)

payment_methods = np.random.choice(
    [
        "Electronic Check",
        "Bank Transfer",
        "Credit Card",
        "Mailed Check"
    ],
    size=NUM_CUSTOMERS
)

internet_services = np.random.choice(
    [
        "Fiber Optic",
        "DSL",
        "No"
    ],
    size=NUM_CUSTOMERS,
    p=[0.45, 0.45, 0.10]
)

df = pd.DataFrame({
    "Customer_ID": customer_ids,
    "Gender": genders,
    "Senior_Citizen": senior_citizen,
    "Tenure": tenure,
    "Monthly_Charges": monthly_charges,
    "Total_Charges": total_charges,
    "Contract": contracts,
    "Payment_Method": payment_methods,
    "Internet_Service": internet_services
})

print(f"Base dataset created: {len(df)} rows")

# -----------------------------
# Introduce Missing Values
# -----------------------------

print("\nIntroducing missing values...")

missing_columns = [
    "Gender",
    "Monthly_Charges",
    "Payment_Method",
    "Internet_Service"
]

for col in missing_columns:
    indices = np.random.choice(
        df.index,
        size=100,
        replace=False
    )
    df.loc[indices, col] = np.nan

print("Missing values inserted.")

# -----------------------------
# Introduce Typographical Errors
# -----------------------------

print("\nIntroducing inconsistent categories...")

gender_typos = {
    "Male": ["male", "MALE", "M"],
    "Female": ["female", "FEMALE", "F"]
}

for idx in np.random.choice(df.index, 150, replace=False):
    current = df.loc[idx, "Gender"]

    if current in gender_typos:
        df.loc[idx, "Gender"] = random.choice(
            gender_typos[current]
        )

contract_typos = {
    "Month-to-month": [
        "month to month",
        "Month to month"
    ],
    "One Year": [
        "one year",
        "1 Year"
    ],
    "Two Year": [
        "two year",
        "2 Year"
    ]
}

for idx in np.random.choice(df.index, 120, replace=False):
    current = df.loc[idx, "Contract"]

    if current in contract_typos:
        df.loc[idx, "Contract"] = random.choice(
            contract_typos[current]
        )

payment_typos = {
    "Electronic Check": [
        "electronic check",
        "E-Check"
    ]
}

for idx in np.random.choice(df.index, 80, replace=False):
    current = df.loc[idx, "Payment_Method"]

    if current in payment_typos:
        df.loc[idx, "Payment_Method"] = random.choice(
            payment_typos[current]
        )

print("Category inconsistencies inserted.")

# -----------------------------
# Introduce Invalid Values
# -----------------------------

print("\nIntroducing invalid numerical values...")

invalid_tenure = np.random.choice(
    df.index,
    30,
    replace=False
)

df.loc[invalid_tenure, "Tenure"] = np.random.choice(
    [-5, -2, 150, 999],
    size=30
)

invalid_monthly = np.random.choice(
    df.index,
    30,
    replace=False
)

df.loc[invalid_monthly, "Monthly_Charges"] = np.random.choice(
    [-50, -10, 1500, 99999],
    size=30
)

invalid_total = np.random.choice(
    df.index,
    30,
    replace=False
)

df.loc[invalid_total, "Total_Charges"] = np.random.choice(
    [-500, -100, 250000],
    size=30
)

print("Invalid values inserted.")

# -----------------------------
# Introduce Duplicate Records
# -----------------------------

print("\nIntroducing duplicate records...")

duplicates = df.sample(
    n=75,
    random_state=42
)

df = pd.concat(
    [df, duplicates],
    ignore_index=True
)

print(f"Duplicate records inserted: {len(duplicates)}")

# -----------------------------
# Save Dataset
# -----------------------------

output_path = "data/raw/customer_churn.csv"

df.to_csv(
    output_path,
    index=False
)

print("\nDataset successfully saved!")

print(f"Output file: {output_path}")

print("\nFINAL DATASET SUMMARY")

print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\nMissing values:")

print(df.isnull().sum())

print("\nDuplicate records:")

print(df.duplicated().sum())

print("\nDataset generation completed.")

print("=" * 50)