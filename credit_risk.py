import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler

# ---------------- LOAD DATA ----------------
print("Loading dataset...")
df = pd.read_csv("lending_club.csv", low_memory=False)
print("Original Shape:", df.shape)

# ---------------- SELECT COLUMNS ----------------
columns_needed = [
    "loan_amnt", "term", "int_rate", "emp_length",
    "annual_inc", "dti", "fico_range_low", "loan_status"
]
df = df[columns_needed]

# ---------------- CLEAN DATA ----------------
df["term"] = df["term"].astype(str).str.extract(r"(\d+)").astype(float)
df["int_rate"] = df["int_rate"].astype(str).str.replace("%","",regex=False).astype(float)
df["emp_length"] = df["emp_length"].astype(str).str.extract(r"(\d+)").astype(float)

bad_status = ["Charged Off", "Default", "Late (31-120 days)"]
df["default"] = df["loan_status"].apply(lambda x: 1 if x in bad_status else 0)
df = df.drop(columns=["loan_status"])
df = df.dropna()
print("Cleaned Shape:", df.shape)

# ---------------- FEATURE ENGINEERING ----------------
df["fico_mid"] = df["fico_range_low"] + 25
df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"]
df["dti_int"] = df["dti"] * df["int_rate"]

features = ["loan_amnt","term","int_rate","emp_length","annual_inc","dti",
            "fico_range_low","fico_mid","loan_to_income","dti_int"]
target = "default"

X = df[features]
y = df[target]

# ---------------- TRAIN-TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training shape:", X_train.shape, "Testing shape:", X_test.shape)

# ---------------- RANDOM OVERSAMPLING ----------------
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
print("After Oversampling, Training shape:", X_train_res.shape)

# ---------------- HIGH ACCURACY RANDOM FOREST MODEL ----------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\nTraining High Accuracy Model...")

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("Model training completed")

# normal prediction (NO threshold tuning)
y_pred = rf_model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------- FEATURE IMPORTANCE ----------------
importance = rf_model.feature_importances_
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values("Importance", ascending=False)

print("\nFeature Importance:\n", feature_importance)

plt.figure(figsize=(8,5))
plt.barh(feature_importance["Feature"], feature_importance["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()
plt.show()

# # ---------------- BATCH PREDICTION FOR POWER BI ----------------
df_powerbi = df.copy()
rf_probs_full = rf_model.predict_proba(df_powerbi[features])[:,1]
df_powerbi["default_prob"] = rf_probs_full
df_powerbi["risk_level"] = df_powerbi["default_prob"].apply(
    lambda p: "HIGH RISK" if p>=0.60 else ("MEDIUM RISK" if p>=0.35 else "LOW RISK")
)

df_powerbi.to_csv("credit_risk_powerbi.csv", index=False)
print("Prediction completed and saved as credit_risk_powerbi.csv")


import joblib

joblib.dump(rf_model, "credit_risk_model.pkl")

print("Model saved successfully")