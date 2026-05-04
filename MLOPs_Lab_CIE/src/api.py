import os
import json
import pickle
import logging
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOGS_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI(title="BallotCast API")

with open(os.path.join(MODELS_DIR, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

PREDICTIONS_LOG = os.path.join(LOGS_DIR, "predictions.jsonl")


class VoterFeatures(BaseModel):
    constituency_population: int = Field(..., ge=50000, le=500000)
    candidates_count: int = Field(..., ge=3, le=15)
    literacy_rate: float = Field(..., ge=50.0, le=95.0)
    is_urban: int = Field(..., ge=0, le=1)


@app.get("/ping")
def ping():
    return {"status": "operational", "service": "BallotCast API"}


@app.post("/estimate")
def estimate(features: VoterFeatures):
    data = [[
        features.constituency_population,
        features.candidates_count,
        features.literacy_rate,
        features.is_urban,
    ]]
    prediction = float(model.predict(data)[0])

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": features.model_dump(),
        "prediction": round(prediction, 4),
    }
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {"voter_turnout_pct": round(prediction, 4)}


def save_step2_result():
    test_input = {"constituency_population": 326365, "candidates_count": 9, "literacy_rate": 76.6, "is_urban": 0}
    prediction = float(model.predict([[
        test_input["constituency_population"],
        test_input["candidates_count"],
        test_input["literacy_rate"],
        test_input["is_urban"],
    ]])[0])

    result = {
        "health_endpoint": "/ping",
        "predict_endpoint": "/estimate",
        "port": 8500,
        "health_response": {"status": "operational", "service": "BallotCast API"},
        "test_input": test_input,
        "prediction": round(prediction, 4),
    }
    with open(os.path.join(RESULTS_DIR, "step2_s4.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved results/step2_s4.json  (prediction={round(prediction, 4)})")


if __name__ == "__main__":
    save_step2_result()
    uvicorn.run("api:app", host="0.0.0.0", port=8500, reload=False)
