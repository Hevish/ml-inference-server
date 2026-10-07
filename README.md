# ML Inference Server

A minimal, production-shaped REST service that integrates a trained
PyTorch model into a running system. A client sends feature values over
HTTP and receives a class prediction with calibrated probabilities.

The model itself is deliberately trivial (a small MLP on the Iris
dataset) — the focus of this project is the **integration**: loading a
trained model into a web service, validating input, serving predictions
efficiently, and exposing a health check for orchestration.

## Architecture

```
client ──HTTP POST /predict──▶  FastAPI app  ──▶  PyTorch model (loaded once)
                                     │
                                     ├─ Pydantic validates the request (422 on bad input)
                                     ├─ applies the training-time feature normalisation
                                     └─ returns JSON: class + confidence + per-class probabilities
```

Three files separate the concerns:

| File | Responsibility |
|------|----------------|
| `model.py` | Model architecture + shared constants (imported by both training and serving, so they can never diverge) |
| `train.py` | Trains the model, saves `model.pt` and the normalisation stats (`preprocess.json`) |
| `server.py` | Loads the artefacts once at startup and serves `/predict` and `/health` |

## Running it

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (produces model.pt and preprocess.json). Takes a few seconds.
python train.py
# -> Test accuracy: 1.000

# 3. Start the inference server
uvicorn server:app --host 0.0.0.0 --port 8000
```

The server loads the model once at startup, then serves requests.
Interactive API docs are available at `http://localhost:8000/docs`.

## Using the API

**Health check** (used by load balancers / orchestrators for readiness):

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

**Prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

Response:

```json
{
  "predicted_class": "setosa",
  "confidence": 0.9997,
  "probabilities": {
    "setosa": 0.9997,
    "versicolor": 0.0003,
    "virginica": 0.0000001
  }
}
```

Malformed input (missing or out-of-range fields) is rejected with HTTP
`422` and a description of the problem, before it ever reaches the model.

## Design decisions

- **Model loaded once, at startup.** The FastAPI `lifespan` handler loads
  the weights and preprocessing stats a single time and keeps them in
  memory. Reloading the model per request is the most common and most
  expensive mistake in naive serving.
- **Inference in `eval()` mode under `torch.no_grad()`.** No autograd
  graph is built, so there is no per-request memory growth and inference
  is faster.
- **Preprocessing travels with the model.** The feature mean/std computed
  on the training set are saved alongside the weights and reapplied at
  inference. This prevents training/serving skew — a subtle, high-impact
  class of production bug where the server transforms inputs differently
  from how the model was trained.
- **Schema validation at the boundary.** Pydantic models define the
  request and response contracts, so invalid input fails fast with a
  clear error rather than producing a silent, wrong prediction.
- **Separate health endpoint.** `/health` reports readiness without
  running an inference, so an orchestrator can route traffic only once
  the model is actually loaded.

## Taking this to production

Natural next steps beyond this minimal example:

- **Containerise** with a Dockerfile (slim Python base, copy artefacts,
  run uvicorn with multiple workers behind the container) for reproducible
  deployment on Kubernetes e.g.,.
- **Batch inference** : accept a list of inputs per request to amortise
  overhead under load.
- **Model versioning** : serve the model version in the response and load
  artefacts from a model registry rather than local files.
- **Observability** : add request latency and prediction-distribution
  metrics (e.g. Prometheus) to detect drift and performance regressions.

## Testing

```bash
python test_server.py
```

Exercises the health check, a valid prediction, and input validation
using FastAPI's `TestClient` (no running server required).
