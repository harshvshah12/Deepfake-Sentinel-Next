from fastapi.testclient import TestClient
from api.index import app

client = TestClient(app)

def test_health_is_explicit_about_model_configuration():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_configured" in response.json()

def test_metrics_do_not_claim_fake_benchmarks():
    response = client.get("/model-metrics")
    assert response.status_code == 200
    assert response.json()["evaluation"] is None

def test_predict_rejects_non_image_upload():
    response = client.post("/predict", files={"file": ("bad.txt", b"not an image", "text/plain")})
    assert response.status_code in (400, 415, 422)
