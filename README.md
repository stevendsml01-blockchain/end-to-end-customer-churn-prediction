# End-to-End Customer Churn Prediction

An end-to-end Machine Learning project for predicting customer churn using multiple classification models.

## Project Overview

Customer churn prediction helps businesses identify customers who are likely to leave their services. Early identification allows companies to take preventive actions and improve customer retention.

This project demonstrates a complete machine learning workflow, including:

- Data preprocessing
- Feature engineering
- Model training
- Model evaluation
- Model comparison
- Customer churn prediction
- Model deployment preparation

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

Target Variable:

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

Possible future enhancements include:

- Hyperparameter tuning
- Cross-validation
- SMOTE for class imbalance
- Streamlit deployment
- Docker containerization
- Cloud deployment

---

## Author

Christians Steven Zoe

Data Scientist | Machine Learning Enthusiast

GitHub:
https://github.com/stevendsml01-blockchain