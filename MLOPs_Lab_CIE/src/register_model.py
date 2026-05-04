import os
import json
import mlflow
from mlflow.tracking import MlflowClient

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REGISTERED_MODEL_NAME = "ballotcast-voter-turnout-pct-predictor"
EXPERIMENT_NAME = "ballotcast-voter-turnout-pct"


def register():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(os.path.join(MODELS_DIR, "best_model_info.json"), "r") as f:
        info = json.load(f)

    best_run_id = info["run_id"]
    best_rmse = info["rmse"]

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    run = client.get_run(best_run_id)
    model_uri = f"runs:/{best_run_id}/model"

    mv = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    result = {
        "registered_model_name": REGISTERED_MODEL_NAME,
        "version": int(mv.version),
        "run_id": best_run_id,
        "source_metric": "rmse",
        "source_metric_value": best_rmse,
    }

    out_path = os.path.join(RESULTS_DIR, "step4_s6.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Registered model '{REGISTERED_MODEL_NAME}' version {mv.version}")
    print(f"Run ID: {best_run_id}")
    print(f"RMSE:   {best_rmse}")
    print("Saved results/step4_s6.json")


if __name__ == "__main__":
    register()
