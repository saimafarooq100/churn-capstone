"""Churn-scoring service (provided: the capstone is about the pipeline, not the service)."""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

model = joblib.load(os.getenv("MODEL_PATH", "model/model.joblib"))
with open(os.getenv("META_PATH", "model/metadata.json")) as f:
    meta = json.load(f)
FEATURES = meta["features"]


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(gt=0, le=1000)
    contract_type: int = Field(ge=0, le=2)
    support_calls_90d: int = Field(ge=0, le=100)
    late_payments_12m: int = Field(ge=0, le=50)
    has_security_addon: int = Field(ge=0, le=1)
    age: int = Field(ge=18, le=110)


app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify(status="ok", model_version=meta["version"])


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify(error="bad_request", message="Body must be a JSON object"), 400
    try:
        customer = Customer(**payload)
    except ValidationError as e:
        details = [{"field": ".".join(str(p) for p in err["loc"]), "message": err["msg"]} for err in e.errors()]
        return jsonify(error="validation_failed", details=details), 422
    p = float(model.predict_proba(pd.DataFrame([customer.model_dump()], columns=FEATURES))[0, 1])
    return jsonify(churn_probability=round(p, 4), model_version=meta["version"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
