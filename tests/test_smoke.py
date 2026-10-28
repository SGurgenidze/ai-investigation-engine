
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"



def test_ask_endpoint_returns_answer_or_abstains():
    response = client.post(
        "/ask",
        json={"question": "What happened in the incident?"},
    )

    assert response.status_code == 200

    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "abstained" in data

    if data["abstained"]:
        assert data["answer"] is None
        assert data["citations"] == []
    else:
        assert isinstance(data["answer"], str)
        assert len(data["citations"]) > 0
