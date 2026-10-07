"""Smoke tests for the inference server.

Uses FastAPI's TestClient, which runs the app in-process (including the
startup model-loading), so no separate server needs to be running.
Run with: python test_server.py
"""

from fastapi.testclient import TestClient

from server import app


def main() -> None:
    with TestClient(app) as client:
        # Health check reports ready once the model is loaded.
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok", r.text

        # A valid request returns a well-formed prediction.
        r = client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["predicted_class"] in {"setosa", "versicolor", "virginica"}
        assert 0.0 <= body["confidence"] <= 1.0
        assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4

        # Missing fields are rejected before reaching the model.
        r = client.post("/predict", json={"sepal_length": 5.1})
        assert r.status_code == 422, r.text

    print("All tests passed.")


if __name__ == "__main__":
    main()
