from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

# Load trained model
model = joblib.load("model.pkl")


# Create FastAPI application
app = FastAPI(
    title="Bank Loan Default Prediction API",
    description="Predicts probability of loan default",
    version="1.0.0"
)


# Input data schema
class CustomerData(BaseModel):
    AGE: float
    EMPLOY: float
    ADDRESS: float
    DEBTINC: float
    CREDDEBT: float
    OTHDEBT: float


@app.get("/")
def root():
    return {
        "message": "Bank Loan Default Prediction API is running"
    }


@app.get("/predict")
def predict_batch():
    """
    Loads BankLoan_Test.csv, uses model.pkl to calculate probability,
    returns table for Excel macro
    """
    if model is None:
        return {"error": "model.pkl not found"}

    # Load test data
    test_path = "BANK LOAN_TEST.csv"
    if not os.path.exists(test_path):
        for name in ["BANK LOAN TEST.csv", "BankLoan_Test.csv", "bankloan_test.csv"]:
            if os.path.exists(name):
                test_path = name
                break

    test = pd.read_csv(test_path)

    # feature models
    feature_cols = ["AGE", "EMPLOY", "ADDRESS", "DEBTINC", "CREDDEBT", "OTHDEBT"]
    X_test = test[feature_cols]

    # the probability of default
    probs = model.predict_proba(X_test)[:, 1]

    test_result = test.copy()
    test_result["Default_Probability"] = probs.round(4)
    test_result["Prediction"] = (probs > 0.5).astype(int)

    # return data
    return test_result.to_dict(orient="records")