# End-to-End Customer Churn Prediction

An end-to-end Machine Learning project for predicting customer churn using multiple classification models and translating the findings into actionable business recommendations.

---

## Project Overview

Customer churn prediction helps businesses identify customers who are likely to discontinue their services. By detecting high-risk customers early, companies can take proactive actions to improve customer retention and reduce revenue loss.

This project demonstrates a complete machine learning workflow, including:

- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Model comparison
- Customer churn prediction
- Business insights generation
- Actionable recommendations

---

## Models Used

The following models were developed and evaluated:

- Logistic Regression
- Random Forest
- XGBoost

---

## Project Structure

```
end-to-end-customer-churn-prediction/
│
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── generate_churn_labels.py
│   ├── generate_dirty_dataset.py
│   ├── model_training.py
│   ├── random_forest.py
│   ├── xgboost_training.py
│   ├── model_evaluation.py
│   └── predict.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Dataset

The dataset contains customer demographic information and service usage patterns, including:

- Gender
- Senior Citizen Status
- Contract Type
- Internet Service
- Payment Method
- Monthly Charges
- Total Charges
- Tenure

### Target Variable

- Churn (0 = No Churn, 1 = Churn)

---

## Model Performance

### AUC Scores

| Model | AUC Score |
|---------|-----------|
| Logistic Regression | 0.679 |
| Random Forest | 0.638 |
| XGBoost | 0.685 |

### Best Model

XGBoost achieved the highest AUC score among the evaluated models.

---

## Example Prediction

Example output:

```
Prediction:
Customer WILL CHURN

Probability:
55.46%

Risk Level:
MEDIUM RISK
```

---

## Visualizations

Generated reports include:

- ROC Curve Comparison
- Confusion Matrix
- Feature Importance Analysis
- Classification Reports

---

# Business Insights

The objective of this project is not only to build predictive models but also to generate insights that support business decision-making.

Based on the feature importance analysis, several key findings emerged:

### 1. Contract Type Is the Strongest Churn Driver

Customers with long-term contracts (one-year and two-year contracts) are significantly less likely to churn.

This suggests that customers on month-to-month contracts represent the highest-risk segment.

### 2. Fiber Optic Customers Show Higher Churn Risk

Fiber optic service was identified as one of the strongest predictors of churn.

Possible explanations include:

- Higher customer expectations,
- Service quality issues,
- Greater price sensitivity among premium users.

### 3. Higher Monthly Charges Increase Churn Probability

Customers paying higher monthly fees tend to exhibit higher churn risk.

Pricing and perceived value appear to influence customer retention.

### 4. New Customers Are More Vulnerable

Tenure plays an important role in predicting churn.

Customers with shorter tenure are more likely to leave compared to long-standing customers.

---

# Business Recommendations

Based on these findings, businesses can take several actions:

### Retention Campaigns for Month-to-Month Customers

Encourage customers to switch to longer-term contracts through:

- Discounts,
- Loyalty programs,
- Contract renewal incentives.

### Improve Fiber Optic Customer Experience

Conduct investigations into:

- Service complaints,
- Network quality,
- Customer satisfaction surveys.

### Strengthen Early Customer Engagement

Focus on retaining new customers by providing:

- Welcome offers,
- Personalized onboarding,
- Follow-up support during the first months.

### Proactive Intervention Using Churn Scores

Customers identified by the model as high-risk should be prioritized for retention efforts.

Possible actions include:

- Personalized promotions,
- Customer support outreach,
- Upselling relevant services,
- Resolving complaints before cancellation occurs.

---

## Potential Business Impact

Assume a company has:

- 100,000 customers,
- A churn rate of 20%,
- Average monthly revenue of $80 per customer.

If the churn model helps prevent only 5% of expected churn, the company could retain:

```
100,000 × 20% × 5%
= 1,000 customers
```

Estimated revenue preserved:

```
1,000 × $80
= $80,000 per month
≈ $960,000 per year
```

This demonstrates how machine learning can directly contribute to measurable business value.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/stevendsml01-blockchain/end-to-end-customer-churn-prediction.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Train Logistic Regression:

```bash
python src/model_training.py
```

Train Random Forest:

```bash
python src/random_forest.py
```

Train XGBoost:

```bash
python src/xgboost_training.py
```

Evaluate Models:

```bash
python src/model_evaluation.py
```

Run Predictions:

```bash
python src/predict.py
```

---

## Future Improvements

Potential enhancements include:

- Hyperparameter tuning,
- Cross-validation,
- SMOTE for class imbalance,
- Streamlit deployment,
- Docker containerization,
- Cloud deployment.

---

## Author

**Christians Steven Zoe**

Data Scientist | Machine Learning Enthusiast

GitHub:
https://github.com/stevendsml01-blockchain