"""Smoke test for a running churn-api. Usage: python ci/smoke_test.py [base_url]"""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
SAFE = {"tenure_months": 48, "monthly_charges": 45.0, "contract_type": 2, "support_calls_90d": 0,
        "late_payments_12m": 0, "has_security_addon": 1, "age": 52}
RISKY = {"tenure_months": 2, "monthly_charges": 110.0, "contract_type": 0, "support_calls_90d": 5,
         "late_payments_12m": 3, "has_security_addon": 0, "age": 25}


def call(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


checks = []
code, body = call("GET", "/health")
checks.append(("health returns 200", code == 200 and body.get("status") == "ok"))
code, low = call("POST", "/predict", SAFE)
checks.append(("safe customer scored", code == 200 and 0 <= low["churn_probability"] <= 1))
code, high = call("POST", "/predict", RISKY)
checks.append(("risky customer scores higher", code == 200 and high["churn_probability"] > low["churn_probability"]))
code, _ = call("POST", "/predict", {"age": 5})
checks.append(("invalid request rejected with 422", code == 422))
for name, ok in checks:
    print(("PASS  " if ok else "FAIL  ") + name)
print("model version:", body.get("model_version"))
sys.exit(0 if all(ok for _, ok in checks) else 1)
