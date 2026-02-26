import pandas as pd
import joblib

print("Loading model...")
model = joblib.load("credit_risk_model.pkl")

print("Loading CSV...")
df = pd.read_csv("new_customers.csv")

# create engineered features
df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]

df["fico_mid"] = df["fico_range_low"]

df["dti_int"] = df["dti"] * df["int_rate"]

# FORCE same order using model itself
X = df[model.feature_names_in_]

# predict
df["Prediction"] = model.predict(X)

df["Probability"] = model.predict_proba(X)[:,1]

print(df)

df.to_csv("predictions.csv", index=False)

df.to_csv("prediction_output.csv",index=False)
