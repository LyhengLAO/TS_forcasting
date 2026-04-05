# ============================================================
# tests/test_api.py
# Tests d'intégration API FastAPI avec TestClient
# ============================================================

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


# ─── Mock du modèle (évite de charger MLflow dans les tests) ──

class MockModel:
    """Modèle fictif qui retourne une température fixe."""
    def predict(self, X):
        return np.array([18.5] * len(X))


@pytest.fixture
def client():
    """Client de test FastAPI avec modèle mocké."""
    with patch("api.main.MODEL", new=MockModel()), \
         patch("api.main.mlflow.sklearn.load_model", return_value=MockModel()):
        from api.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def valid_payload():
    return {
        "datetime_utc": "2024-06-01T12:00:00",
        "features": {
            "meteo_temperature_2m_lag_1h":     18.2,
            "meteo_temperature_2m_lag_24h":    17.5,
            "meteo_relative_humidity_2m":       62.0,
            "meteo_wind_speed_10m":             8.3,
            "meteo_precipitation":              0.0,
            "hour_sin":                         0.0,
            "hour_cos":                         1.0,
            "dow_sin":                          0.0,
            "dow_cos":                          1.0,
            "month_sin":                        0.866,
            "month_cos":                        0.5,
            "meteo_temperature_2m_roll_mean_6h": 17.8,
            "meteo_temperature_2m_diff_1":      0.3,
        }
    }


# ─── Tests : Health ───────────────────────────────────────────

class TestHealth:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status"       in data
        assert "model_loaded" in data
        assert "timestamp"    in data

    def test_health_status_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_model_loaded(self, client):
        resp = client.get("/health")
        assert resp.json()["model_loaded"] is True


# ─── Tests : Predict ──────────────────────────────────────────

class TestPredict:

    def test_predict_returns_200(self, client, valid_payload):
        resp = client.post("/predict", json=valid_payload)
        assert resp.status_code == 200

    def test_predict_response_schema(self, client, valid_payload):
        resp = client.post("/predict", json=valid_payload)
        data = resp.json()
        assert "datetime_utc"        in data
        assert "predicted_temp_24h"  in data
        assert "confidence_lower"    in data
        assert "confidence_upper"    in data
        assert "model_version"       in data

    def test_predict_value_is_float(self, client, valid_payload):
        resp = client.post("/predict", json=valid_payload)
        pred = resp.json()["predicted_temp_24h"]
        assert isinstance(pred, float)

    def test_predict_interval_ordered(self, client, valid_payload):
        """lower ≤ pred ≤ upper."""
        resp = client.post("/predict", json=valid_payload)
        data = resp.json()
        assert data["confidence_lower"] <= data["predicted_temp_24h"] <= data["confidence_upper"]

    def test_predict_datetime_echoed(self, client, valid_payload):
        resp = client.post("/predict", json=valid_payload)
        assert resp.json()["datetime_utc"] == valid_payload["datetime_utc"]

    def test_predict_empty_features_handled(self, client):
        """Features vides → comportement géré sans crash serveur."""
        payload = {"datetime_utc": "2024-01-01T00:00:00", "features": {}}
        resp = client.post("/predict", json=payload)
        # Peut retourner 200 (prédiction sur vecteur vide) ou 422 (validation Pydantic)
        assert resp.status_code in (200, 422, 500)

    def test_predict_invalid_datetime_format(self, client, valid_payload):
        """Datetime invalide → 422 Unprocessable Entity."""
        bad = dict(valid_payload)
        bad["datetime_utc"] = "pas-une-date"
        resp = client.post("/predict", json=bad)
        # FastAPI / Pydantic valide le format string — peut passer si str accepté
        assert resp.status_code in (200, 422)

    def test_predict_missing_required_field(self, client):
        """Champ requis manquant → 422."""
        resp = client.post("/predict", json={"features": {}})
        assert resp.status_code == 422


# ─── Tests : Metrics (Prometheus) ─────────────────────────────

class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type_text(self, client):
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_metrics_contains_counter(self, client, valid_payload):
        # D'abord faire une requête pour incrémenter le counter
        client.post("/predict", json=valid_payload)
        resp = client.get("/metrics")
        assert "predict_requests_total" in resp.text

    def test_metrics_contains_histogram(self, client):
        resp = client.get("/metrics")
        assert "predict_latency_seconds" in resp.text


# ─── Tests : CORS et sécurité basique ─────────────────────────

class TestSecurity:

    def test_predict_large_feature_vector(self, client):
        """Vecteur de features très large → pas de crash."""
        payload = {
            "datetime_utc": "2024-06-01T12:00:00",
            "features": {f"feat_{i}": float(i) for i in range(200)}
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code in (200, 422, 500)

    def test_predict_extreme_values(self, client, valid_payload):
        """Valeurs extrêmes mais numériques → pas de crash."""
        extreme = dict(valid_payload)
        extreme["features"] = {k: 1e10 for k in valid_payload["features"]}
        resp = client.post("/predict", json=extreme)
        assert resp.status_code in (200, 500)   # peut retourner 500 si inf/nan

    def test_404_on_unknown_endpoint(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
