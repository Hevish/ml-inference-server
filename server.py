"""FastAPI inference server for the trained Iris classifier.

Design notes (the production-relevant bits):
  * The model is loaded ONCE at startup (lifespan handler) and reused
    across requests, rather than reloaded per request. Loading weights
    per request is the most common performance mistake in naive serving.
  * Inference runs under torch.no_grad() with the model in eval() mode,
    so no autograd graph is built and no memory leaks across requests.
  * Request and response bodies are validated by Pydantic schemas, so
    malformed input is rejected with a clear 422 before it reaches the model.
  * A /health endpoint lets an orchestrator (Kubernetes, a load balancer)
    check readiness without running an inference.
"""

import json
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel, Field

from model import IrisNet

# Populated at startup, read by the request handlers.
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and preprocessing stats once, before serving traffic."""
    model = IrisNet()
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()

    with open("preprocess.json") as f:
        pre = json.load(f)

    STATE["model"] = model
    STATE["mean"] = torch.tensor(pre["mean"])
    STATE["std"] = torch.tensor(pre["std"])
    STATE["class_names"] = pre["class_names"]
    yield
    STATE.clear()


app = FastAPI(title="Iris Inference Server", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    """One flower's four measurements, in centimetres."""

    sepal_length: float = Field(..., ge=0, le=20)
    sepal_width: float = Field(..., ge=0, le=20)
    petal_length: float = Field(..., ge=0, le=20)
    petal_width: float = Field(..., ge=0, le=20)


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health() -> dict:
    """Readiness probe: 'ok' only once the model is loaded."""
    return {"status": "ok" if "model" in STATE else "loading"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    features = torch.tensor(
        [[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]]
    )
    # Apply the SAME standardisation used in training.
    features = (features - STATE["mean"]) / STATE["std"]

    with torch.no_grad():
        logits = STATE["model"](features)
        probs = F.softmax(logits, dim=1).squeeze(0)

    idx = int(probs.argmax())
    class_names = STATE["class_names"]
    return PredictResponse(
        predicted_class=class_names[idx],
        confidence=float(probs[idx]),
        probabilities={name: float(p) for name, p in zip(class_names, probs)},
    )
