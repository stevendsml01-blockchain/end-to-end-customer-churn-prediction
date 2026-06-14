import pandas as pd

print("=" * 60)
print("CUSTOMER CHURN DATA PREPROCESSING")
print("=" * 60)

# ==================================================
# LOAD DATASET
# ==================================================

print("\nLoading raw dataset...")

df = pd.read_csv("data/raw/customer_churn.csv")

print("Dataset loaded successfully!")

# ==================================================
# DATA INSPECTION
# ==================================================

print("\n" + "=" * 60)
print("DATA INSPECTION")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nFirst 5 Rows:")
print(df.head())

print("\nNumerical Statistics:")
print(df.describe())

# ==================================================
# MISSING VALUE TREATMENT
# ==================================================

print("\n" + "=" * 60)
print("MISSING VALUE TREATMENT")
print("=" * 60)

print("\nHandling missing values...")

# Gender
if "Gender" in df.columns:
    mode_gender = df["Gender"].mode()[0]
    df["Gender"] = df["Gender"].fillna(mode_gender)

# Monthly Charges
if "Monthly_Charges" in df.columns:
    median_monthly = df["Monthly_Charges"].median()
    df["Monthly_Charges"] = df["Monthly_Charges"].fillna(
        median_monthly
    )

# Payment Method
if "Payment_Method" in df.columns:
    mode_payment = df["Payment_Method"].mode()[0]
    df["Payment_Method"] = df["Payment_Method"].fillna(
        mode_payment
    )

# Internet Service
if "Internet_Service" in df.columns:
    mode_internet = df["Internet_Service"].mode()[0]
    df["Internet_Service"] = df["Internet_Service"].fillna(
        mode_internet
    )

print("Missing values handled successfully!")

print("\nRemaining Missing Values:")
print(df.isnull().sum())

# ==================================================
# DUPLICATE REMOVAL
# ==================================================

print("\n" + "=" * 60)
print("DUPLICATE REMOVAL")
print("=" * 60)

duplicates_before = df.duplicated().sum()

print(f"\nDuplicate records before removal: {duplicates_before}")

df = df.drop_duplicates()

duplicates_after = df.duplicated().sum()

print(f"Duplicate records after removal: {duplicates_after}")

print(f"Current dataset size: {len(df)}")

# ==================================================
# CATEGORY STANDARDIZATION
# ==================================================

print("\n" + "=" * 60)
print("CATEGORY STANDARDIZATION")
print("=" * 60)

gender_mapping = {
    "male": "Male",
    "MALE": "Male",
    "M": "Male",
    "female": "Female",
    "FEMALE": "Female",
    "F": "Female"
}

df["Gender"] = df["Gender"].replace(gender_mapping)

contract_mapping = {
    "month-to-month": "Month-to-month",
    "Month To Month": "Month-to-month",
    "one year": "One Year",
    "ONE YEAR": "One Year",
    "two year": "Two Year",
    "TWO YEAR": "Two Year"
}

df["Contract"] = df["Contract"].replace(contract_mapping)

internet_mapping = {
    "fiber optic": "Fiber Optic",
    "Fiber optic": "Fiber Optic",
    "dsl": "DSL"
}

df["Internet_Service"] = df[
    "Internet_Service"
].replace(internet_mapping)

print("\nGender Categories:")
print(sorted(df["Gender"].unique()))

print("\nContract Categories:")
print(sorted(df["Contract"].unique()))

print("\nInternet Service Categories:")
print(sorted(df["Internet_Service"].unique()))

# ==================================================
# INVALID VALUE HANDLING
# ==================================================

print("\n" + "=" * 60)
print("INVALID VALUE HANDLING")
print("=" * 60)

print("\nBefore Cleaning:")

print(
    f"Tenure Range: "
    f"{df['Tenure'].min()} to {df['Tenure'].max()}"
)

print(
    f"Monthly Charges Range: "
    f"{df['Monthly_Charges'].min()} "
    f"to {df['Monthly_Charges'].max()}"
)

print(
    f"Total Charges Range: "
    f"{df['Total_Charges'].min()} "
    f"to {df['Total_Charges'].max()}"
)

tenure_before = len(df)

df = df[
    (df["Tenure"] >= 0) &
    (df["Tenure"] <= 72)
]

print(
    f"\nRows removed due to invalid tenure: "
    f"{tenure_before - len(df)}"
)

monthly_before = len(df)

df = df[
    (df["Monthly_Charges"] >= 0) &
    (df["Monthly_Charges"] <= 200)
]

print(
    f"Rows removed due to invalid monthly charges: "
    f"{monthly_before - len(df)}"
)

total_before = len(df)

df = df[
    (df["Total_Charges"] >= 0)
]

print(
    f"Rows removed due to invalid total charges: "
    f"{total_before - len(df)}"
)

print("\nAfter Cleaning:")

print(
    f"Tenure Range: "
    f"{df['Tenure'].min()} to {df['Tenure'].max()}"
)

print(
    f"Monthly Charges Range: "
    f"{df['Monthly_Charges'].min()} "
    f"to {df['Monthly_Charges'].max()}"
)

print(
    f"Total Charges Range: "
    f"{df['Total_Charges'].min()} "
    f"to {df['Total_Charges'].max()}"
)

print(f"\nRemaining Rows: {len(df)}")

# ==================================================
# FINAL VALIDATION
# ==================================================

print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nGender Categories:")
print(sorted(df["Gender"].unique()))

print("\nContract Categories:")
print(sorted(df["Contract"].unique()))

print("\nInternet Service Categories:")
print(sorted(df["Internet_Service"].unique()))

if (
    df.isnull().sum().sum() == 0
    and df.duplicated().sum() == 0
):
    print("\nVALIDATION PASSED!")
    print("Dataset is ready for the next phase.")
else:
    print("\nVALIDATION FAILED!")
    print("Please review the dataset.")


# ==========================================
# CATEGORY STANDARDIZATION
# ==========================================

print("\nStandardizing categorical values...")

# Contract
contract_mapping = {
    "Month-to-month": "Month-to-month",
    "month to month": "Month-to-month",
    "Month to month": "Month-to-month",

    "One Year": "One Year",
    "1 Year": "One Year",

    "Two Year": "Two Year",
    "2 Year": "Two Year"
}

df["Contract"] = df["Contract"].replace(contract_mapping)


payment_mapping = {
    "Electronic Check": "Electronic Check",
    "electronic check": "Electronic Check",
    "E-Check": "Electronic Check",

    "Credit Card": "Credit Card",
    "credit card": "Credit Card",

    "Bank Transfer": "Bank Transfer",
    "bank transfer": "Bank Transfer",

    "Mailed Check": "Mailed Check",
    "mailed check": "Mailed Check"
}

df["Payment_Method"] = (
    df["Payment_Method"]
    .astype(str)
    .str.strip()
    .replace(payment_mapping)
)

# Internet Service
internet_mapping = {
    "Fiber Optic": "Fiber Optic",
    "fiber optic": "Fiber Optic",

    "DSL": "DSL",
    "dsl": "DSL",

    "No": "No",
    "No Internet": "No"
}

df["Internet_Service"] = df["Internet_Service"].replace(internet_mapping)


print("Categorical values standardized successfully!")


# ==================================================
# SAVE CLEAN DATASET
# ==================================================

df.to_csv(
    "data/processed/cleaned_customer_churn.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
print(
    "Output File: "
    "data/processed/cleaned_customer_churn.csv"
)

print("\n" + "=" * 60)
print("DATA PREPROCESSING COMPLETED")
print("=" * 60)