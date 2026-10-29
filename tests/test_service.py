from app import app

client = app.test_client()
SAFE = {"tenure_months": 48, "monthly_charges": 45.0, "contract_type": 2, "support_calls_90d": 0,
        "late_payments_12m": 0, "has_security_addon": 1, "age": 52}
RISKY = {"tenure_months": 2, "monthly_charges": 110.0, "contract_type": 0, "support_calls_90d": 5,
         "late_payments_12m": 3, "has_security_addon": 0, "age": 25}


def test_health_is_ok():
    assert client.get("/health").status_code == 200


def test_valid_customer_gets_a_probability():
    r = client.post("/predict", json=SAFE)
    assert r.status_code == 200 and 0 <= r.get_json()["churn_probability"] <= 1


def test_risky_customer_scores_higher():
    low = client.post("/predict", json=SAFE).get_json()["churn_probability"]
    high = client.post("/predict", json=RISKY).get_json()["churn_probability"]
    assert high > low


def test_invalid_customer_is_422():
    assert client.post("/predict", json={**SAFE, "age": 5}).status_code == 422
