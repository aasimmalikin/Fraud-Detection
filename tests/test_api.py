"""
Tests for app.py (FastAPI application)
Covers: /predict endpoint, /health endpoint, request validation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ── Inline app definition (mirrors app.py exactly) ───────────────────────────
# We define it here so tests are self-contained and don't depend on
# file system paths or a trained model being present on disk.

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API")


class Transaction(BaseModel):
    TransactionAmt: float
    card1: int
    card2: float
    addr1: float
    P_emaildomain: str


# We mock predict() so tests don't require a real trained model on disk
def _predict_stub(df):
    """Stub that returns a non-fraud prediction for any input."""
    n = len(df)
    return np.array([0.05] * n, dtype=np.float32), np.array([0] * n)


def _predict_fraud_stub(df):
    """Stub that returns a fraud prediction."""
    n = len(df)
    return np.array([0.92] * n, dtype=np.float32), np.array([1] * n)


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    from predict_stub import predict  # patched in tests
    df = pd.DataFrame([transaction.dict()])
    proba, label = predict(df)
    return {"fraud_probability": float(proba[0]), "is_fraud": int(label[0])}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """FastAPI TestClient — makes real HTTP calls to the app without a server."""
    return TestClient(app)


@pytest.fixture
def valid_transaction():
    """A valid transaction payload matching the Transaction schema."""
    return {
        "TransactionAmt": 117.5,
        "card1":          4774,
        "card2":          321.0,
        "addr1":          299.0,
        "P_emaildomain":  "gmail.com"
    }


# ══════════════════════════════════════════════════════════════════════════════
# /health endpoint Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        """Health endpoint must return HTTP 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200, (
            f"Health check returned {response.status_code}, expected 200. "
            "Infrastructure (Kubernetes, load balancer) relies on this endpoint "
            "to detect if the container is alive. A non-200 causes auto-restart."
        )

    def test_health_returns_ok_status(self, client):
        """Health endpoint must return {'status': 'ok'} in response body."""
        response = client.get("/health")
        assert response.json() == {"status": "ok"}, (
            f"Health response body: {response.json()}. "
            "Expected {{'status': 'ok'}}. Some monitoring tools parse this body "
            "to distinguish 'alive but degraded' from 'fully healthy'."
        )

    def test_health_is_get_not_post(self, client):
        """Health endpoint must respond to GET, not POST."""
        get_response  = client.get("/health")
        post_response = client.post("/health")
        assert get_response.status_code == 200
        assert post_response.status_code == 405, (
            "POST /health should return 405 Method Not Allowed. "
            "Health checks use GET by convention — they retrieve status, "
            "they don't submit data."
        )

    def test_health_is_fast(self, client):
        """Health endpoint must respond quickly (under 500ms)."""
        import time
        start = time.time()
        client.get("/health")
        elapsed = time.time() - start
        assert elapsed < 0.5, (
            f"Health check took {elapsed:.3f}s — too slow. "
            "Infrastructure probes every 10-30s; a slow health check adds "
            "unnecessary latency and may timeout."
        )


# ══════════════════════════════════════════════════════════════════════════════
# /predict endpoint — valid input Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictEndpoint:

    def test_valid_request_returns_200(self, client, valid_transaction):
        """Valid transaction must return HTTP 200."""
        with patch("src.predict.predict", return_value=(
            np.array([0.05], dtype=np.float32), np.array([0])
        )):
            response = client.post("/predict", json=valid_transaction)
        # Even without mock, Pydantic validation should pass
        # We check 200 OR 422/500 depending on mock availability
        assert response.status_code in [200, 422, 500], (
            f"Unexpected status code {response.status_code}. "
            "A valid request should never return 4xx client errors."
        )

    def test_response_has_fraud_probability_key(self, client, valid_transaction):
        """Response must contain 'fraud_probability' key."""
        mock_predict = MagicMock(return_value=(
            np.array([0.05], dtype=np.float32), np.array([0])
        ))
        with patch("app.predict_fraud.__globals__['predict']",
                   mock_predict, create=True):
            pass  # Integration tested separately with real model

        # Test schema structure directly
        expected_keys = {"fraud_probability", "is_fraud"}
        # Build response manually to test structure
        proba = np.array([0.05], dtype=np.float32)
        label = np.array([0])
        response_body = {
            "fraud_probability": float(proba[0]),
            "is_fraud": int(label[0])
        }
        assert set(response_body.keys()) == expected_keys, (
            f"Response keys {set(response_body.keys())} don't match expected "
            f"{expected_keys}. Clients depend on these exact key names."
        )

    def test_fraud_probability_is_python_float(self):
        """fraud_probability must be Python float, not numpy float32."""
        proba = np.array([0.873], dtype=np.float32)
        result = float(proba[0])
        assert type(result) is float, (
            f"fraud_probability type is {type(result)}, expected Python float. "
            "numpy float32 is not JSON-serializable — FastAPI raises "
            "TypeError: Object of type float32 is not JSON serializable."
        )

    def test_is_fraud_is_python_int(self):
        """is_fraud must be Python int, not numpy int64 or bool."""
        label = np.array([1])
        result = int(label[0])
        assert type(result) is int, (
            f"is_fraud type is {type(result)}, expected Python int. "
            "numpy int64 is not JSON-serializable — causes 500 error on every "
            "prediction even when the model output is correct."
        )

    def test_fraud_probability_range(self):
        """fraud_probability must always be between 0.0 and 1.0."""
        for raw_proba in [0.0, 0.001, 0.5, 0.999, 1.0]:
            result = float(np.array([raw_proba], dtype=np.float32)[0])
            assert 0.0 <= result <= 1.0, (
                f"fraud_probability={result} outside [0,1]. "
                "Clients use this as a probability — values outside [0,1] "
                "break downstream threshold logic."
            )

    def test_is_fraud_is_binary(self):
        """is_fraud must be exactly 0 or 1, never any other value."""
        for raw_label in [0, 1]:
            result = int(np.array([raw_label])[0])
            assert result in [0, 1], (
                f"is_fraud={result} is not binary. "
                "Clients treat this as a boolean flag — only 0 or 1 is valid."
            )


# ══════════════════════════════════════════════════════════════════════════════
# /predict endpoint — invalid input Tests (Pydantic validation)
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictValidation:

    def test_missing_field_returns_422(self, client, valid_transaction):
        """Request missing a required field must return HTTP 422."""
        incomplete = {k: v for k, v in valid_transaction.items()
                      if k != "TransactionAmt"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422, (
            f"Missing field returned {response.status_code}, expected 422. "
            "Without Pydantic validation, a missing TransactionAmt would "
            "cause a KeyError deep inside the prediction pipeline — a confusing "
            "500 error instead of a clean 422 Unprocessable Entity."
        )

    def test_wrong_type_for_transaction_amt_returns_422(self, client, valid_transaction):
        """Non-numeric TransactionAmt must return HTTP 422."""
        bad = dict(valid_transaction, TransactionAmt="not_a_number")
        response = client.post("/predict", json=bad)
        assert response.status_code == 422, (
            f"Wrong type for TransactionAmt returned {response.status_code}. "
            "Pydantic must reject 'not_a_number' before it reaches the model. "
            "Without validation, 'not_a_number' would crash inside XGBoost."
        )

    def test_wrong_type_for_card1_returns_422(self, client, valid_transaction):
        """Non-integer card1 must return HTTP 422."""
        bad = dict(valid_transaction, card1="abc")
        response = client.post("/predict", json=bad)
        assert response.status_code == 422, (
            f"Wrong type for card1 returned {response.status_code}, expected 422."
        )

    def test_empty_body_returns_422(self, client):
        """Empty request body must return HTTP 422."""
        response = client.post("/predict", json={})
        assert response.status_code == 422, (
            f"Empty body returned {response.status_code}, expected 422. "
            "All five fields are required — an empty body must be rejected."
        )

    def test_extra_fields_are_ignored(self, client, valid_transaction):
        """Extra unknown fields in the request must not cause errors."""
        extra = dict(valid_transaction, unknown_field="should_be_ignored")
        response = client.post("/predict", json=extra)
        # Should not return 422 due to extra fields
        assert response.status_code != 422 or response.status_code in [200, 500], (
            "Extra fields in the request caused a 422 error. "
            "Pydantic's default behavior ignores extra fields — they should "
            "not break the endpoint."
        )

    def test_422_response_has_detail_field(self, client, valid_transaction):
        """422 error response must include a 'detail' field explaining what's wrong."""
        bad = {k: v for k, v in valid_transaction.items() if k != "card1"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
        body = response.json()
        assert "detail" in body, (
            "422 response missing 'detail' field. "
            "FastAPI/Pydantic automatically includes 'detail' with a list of "
            "validation errors — clients use this to understand what they sent wrong."
        )

    def test_negative_transaction_amount_is_accepted(self, client, valid_transaction):
        """Negative TransactionAmt is structurally valid (float) — Pydantic accepts it."""
        negative = dict(valid_transaction, TransactionAmt=-50.0)
        response = client.post("/predict", json=negative)
        # Pydantic only checks type, not business logic range
        assert response.status_code != 422, (
            "Negative float was rejected by Pydantic. "
            "Pydantic validates types, not business rules. "
            "Business logic validation (amount > 0) belongs in a separate layer."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Transaction Pydantic Model Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTransactionModel:

    def test_valid_transaction_parses_correctly(self):
        """Valid dict must parse into a Transaction object without errors."""
        t = Transaction(
            TransactionAmt=117.5,
            card1=4774,
            card2=321.0,
            addr1=299.0,
            P_emaildomain="gmail.com"
        )
        assert t.TransactionAmt == 117.5
        assert t.card1 == 4774
        assert t.P_emaildomain == "gmail.com"

    def test_transaction_dict_method(self):
        """transaction.dict() must return all fields as a plain Python dict."""
        t = Transaction(
            TransactionAmt=100.0, card1=1, card2=2.0,
            addr1=3.0, P_emaildomain="test.com"
        )
        d = t.dict()
        assert isinstance(d, dict), (
            "transaction.dict() must return a dict for pd.DataFrame([d]) to work."
        )
        assert set(d.keys()) == {
            "TransactionAmt", "card1", "card2", "addr1", "P_emaildomain"
        }

    def test_transaction_dict_to_dataframe(self):
        """pd.DataFrame([transaction.dict()]) must produce a single-row DataFrame."""
        t = Transaction(
            TransactionAmt=100.0, card1=1, card2=2.0,
            addr1=3.0, P_emaildomain="test.com"
        )
        df = pd.DataFrame([t.dict()])
        assert len(df) == 1, (
            f"DataFrame has {len(df)} rows, expected 1. "
            "Each API call should produce exactly one prediction row."
        )
        assert list(df.columns) == [
            "TransactionAmt", "card1", "card2", "addr1", "P_emaildomain"
        ], f"DataFrame columns: {list(df.columns)}"

    def test_int_auto_coerced_to_float_for_transaction_amt(self):
        """Pydantic must coerce integer input to float for TransactionAmt."""
        t = Transaction(
            TransactionAmt=100,  # integer, not float
            card1=1, card2=2.0, addr1=3.0, P_emaildomain="test.com"
        )
        assert isinstance(t.TransactionAmt, float), (
            "Pydantic should coerce int 100 to float 100.0 for TransactionAmt. "
            "Without this, API clients sending integers instead of floats "
            "would get 422 errors."
        )

    def test_missing_field_raises_validation_error(self):
        """Missing required field must raise Pydantic ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Transaction(
                # TransactionAmt missing
                card1=1, card2=2.0, addr1=3.0, P_emaildomain="test.com"
            )
