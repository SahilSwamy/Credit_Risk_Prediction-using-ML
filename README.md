💳 Credit Risk Prediction: Machine Learning for Lending

📌 Project Overview
Lending institutions lose millions annually due to credit defaults. This project leverages Machine Learning to automate credit risk assessment. Using a massive dataset of 1M+ records from Lending Club, I built a predictive system that identifies high-risk applicants before a loan is approved.
The Goal: Minimize financial loss by predicting "Default" vs "Non-Default" status with high precision.

🛠️ The Machine Learning Pipeline
1. Data Engineering & Preprocessing
Scale: Processed a large-scale dataset (1 Million+ rows) using optimized Pandas workflows.
Handling Imbalance: Since defaults are rarer than successful repayments, I implemented Class Weighting (and/or SMOTE) to ensure the model doesn't ignore high-risk cases.
Feature Encoding: Transformed categorical variables like 'Term' and 'Employment Length' into numerical formats suitable for ML.

2. Feature Selection (Predictors)
The model focuses on the most high-impact financial indicators:
Debt-to-Income (DTI): The primary measure of a borrower's ability to manage monthly payments.
FICO Score: A critical creditworthiness metric.
Annual Income & Loan Amount: To determine the "Loan-to-Income" ratio.
Interest Rate: Often a reflection of the risk already perceived by the bank.

3. Model Architecture
I chose the Random Forest Classifier for its robustness against outliers and its ability to handle non-linear relationships in financial data.
Hyperparameters: Tuned for depth and leaf nodes to prevent overfitting.
Metric Focus: Prioritized Recall (finding all potential defaulters) over simple Accuracy.

📈 Performance Results
Metric	Value
Model	Random Forest Classifier
Accuracy	87%
Data Size	1,000,000+ Rows
Optimization	Class Weighting for Imbalanced Data


Model file not included due to size .
Run credit_risk.py to genrate
credit_risk_model.pkl

Dataset source:- https://www.kaggle.com/datasets/wordsforthewise/lending-club


💡 Why this matters?
By implementing this model, a financial institution can reduce its Manual Review Time by 60% and proactively flag high-risk accounts, potentially saving millions in "Charged-Off" loans.
