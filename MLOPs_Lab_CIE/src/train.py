import os
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "training_data.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

EXPERIMENT_NAME = "ballotcast-voter-turnout-pct"
FEATURES = ["constituency_population", "candidates_count", "literacy_rate", "is_urban"]
TARGET = "voter_turnout_pct"


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4), "mape": round(mape, 4)}


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment(EXPERIMENT_NAME)

    models_config = [
        {
            "name": "LinearRegression",
            "model": LinearRegression(),
            "params": {"fit_intercept": True},
        },
        {
            "name": "RandomForest",
            "model": RandomForestRegressor(n_estimators=100, random_state=42),
            "params": {"n_estimators": 100, "random_state": 42},
        },
    ]

    results = []
    run_ids = {}

    for cfg in models_config:
        with mlflow.start_run(run_name=cfg["name"]) as run:
            mlflow.set_tag("project_phase", "model_selection")
            mlflow.log_params(cfg["params"])

            cfg["model"].fit(X_train, y_train)
            y_pred = cfg["model"].predict(X_test)
            metrics = compute_metrics(y_test.values, y_pred)

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(cfg["model"], artifact_path="model")

            run_ids[cfg["name"]] = run.info.run_id
            results.append({"name": cfg["name"], **metrics})

            model_path = os.path.join(MODELS_DIR, f"{cfg['name'].lower()}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(cfg["model"], f)

            print(f"{cfg['name']}: MAE={metrics['mae']}, RMSE={metrics['rmse']}, R2={metrics['r2']}, MAPE={metrics['mape']}")

    best = min(results, key=lambda x: x["rmse"])

    best_model_path = os.path.join(MODELS_DIR, f"{best['name'].lower()}.pkl")
    best_model_dest = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(best_model_path, "rb") as f:
        best_model = pickle.load(f)
    with open(best_model_dest, "wb") as f:
        pickle.dump(best_model, f)

    with open(os.path.join(MODELS_DIR, "best_model_info.json"), "w") as f:
        json.dump({"name": best["name"], "run_id": run_ids[best["name"]], "rmse": best["rmse"]}, f)

    step1 = {
        "experiment_name": EXPERIMENT_NAME,
        "models": results,
        "best_model": best["name"],
        "best_metric_name": "rmse",
        "best_metric_value": best["rmse"],
    }
    with open(os.path.join(RESULTS_DIR, "step1_s1.json"), "w") as f:
        json.dump(step1, f, indent=2)

    print(f"\nBest model: {best['name']} (RMSE={best['rmse']})")
    print("Saved results/step1_s1.json")


if __name__ == "__main__":
    train()
