"""
FastAPI service to expose fraud prediction endpoint.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/fraud_ensemble.joblib")

app = FastAPI(title="Insurance Fraud Detection API")

model = joblib.load(MODEL_PATH)


class Claim(BaseModel):
    customer_age: int = Field(..., ge=18, le=100)
    claim_amount: float = Field(..., ge=0)
    num_previous_claims: int = Field(..., ge=0)
    claim_type: str
    has_police_report: str


@app.get("/")
async def root():
    return {"service": "fraud-detection-ai", "status": "ok"}


@app.post("/predict_fraud")
async def predict_fraud(claim: Claim):
    df = pd.DataFrame(
        [
            {
                "customer_age": claim.customer_age,
                "claim_amount": claim.claim_amount,
                "num_previous_claims": claim.num_previous_claims,
                "claim_type": claim.claim_type,
                "has_police_report": claim.has_police_report,
            }
        ]
    )

    proba = model.predict_proba(df)[0][1]
    is_fraud = bool(proba > 0.5)

    return {
        "fraud_probability": float(proba),
        "is_fraud": is_fraud,
    }
