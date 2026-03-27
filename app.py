from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.predict import predict

app = FastAPI(title = "Fraud Detection API")
class Transaction(BaseModel):
    TransactionAmt: float
    ProductCD: str
    P_emaildomain: str
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    df = pd.DataFrame([transaction.dict()])
    proba, Label = predict(df)
    return {"fraud_probability": proba[0], "is_fraud": int(Label[0])}

@app.get("/health")
def health():
    return {"status": "ok"}

