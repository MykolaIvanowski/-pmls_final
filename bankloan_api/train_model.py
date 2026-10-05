import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


# load training data set
df = pd.read_csv("BANK LOAN.csv")


# Define feature and target
X = df.drop(columns=["SN", "DEFAULTER"])
y = df["DEFAULTER"]


# random forest 500
model = RandomForestClassifier(
    n_estimators=500,
    random_state=42
)
# Train model
model.fit(X, y)

# Save trained model
joblib.dump(model, "model.pkl")


print("Model trained successfully.")
print(f"Number of trees: {model.n_estimators}")
print(f"Features: {list(X.columns)}")
print("Model saved as model.pkl")



# use test data
test_df = pd.read_csv("BANK LOAN_TEST.csv")

# shink columns
X_test = test_df.drop(columns=["SN"], errors='ignore')
X_test = X_test[["AGE","EMPLOY","ADDRESS","DEBTINC","CREDDEBT","OTHDEBT"]]

# predict default possibility
proba = model.predict_proba(X_test)[:,1]  # [:,1]

test_df["Default_Probability"] = proba
test_df["Prediction"] = (proba > 0.5).astype(int)

test_df.to_csv("BankLoan_Test_Predicted.csv", index=False)
test_df.to_excel("BankLoan_Test_Predicted.xlsx", index=False)

print(test_df.head())
print("Test predictions saved -> BankLoan_Test_Predicted.csv")