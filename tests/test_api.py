from fastapi.testclient import TestClient

from src.api.app import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


def test_predict_endpoint():
    with TestClient(app) as client:
        payload = {"text": "I love studying in Denmark, but winter is challenging."}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert "translation" in data
        assert isinstance(data["translation"], str)
