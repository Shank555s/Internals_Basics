import os
import json
import pandas as pd

LOGS_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "predictions.jsonl")
TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "training_data.csv")
NEW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "new_data.csv")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

THRESHOLDS = {
    "constituency_population": 94684.34,
    "candidates_count": 2.21,
}


def run_monitor():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_pop_mean = round(train_df["constituency_population"].mean(), 2)
    train_cand_mean = round(train_df["candidates_count"].mean(), 2)

    new_df = pd.read_csv(NEW_DATA_PATH)
    live_pop_mean = round(new_df["constituency_population"].mean(), 2)
    live_cand_mean = round(new_df["candidates_count"].mean(), 2)

    records = []
    with open(LOGS_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    total = len(records)
    predictions = [r["prediction"] for r in records]
    mean_prediction = round(sum(predictions) / len(predictions), 4) if predictions else 0.0

    pop_shift = round(abs(live_pop_mean - train_pop_mean), 2)
    cand_shift = round(abs(live_cand_mean - train_cand_mean), 2)

    alerts = [
        {
            "feature": "constituency_population",
            "train_mean": train_pop_mean,
            "live_mean": live_pop_mean,
            "shift": pop_shift,
            "threshold": THRESHOLDS["constituency_population"],
            "status": "ALERT" if pop_shift > THRESHOLDS["constituency_population"] else "OK",
        },
        {
            "feature": "candidates_count",
            "train_mean": train_cand_mean,
            "live_mean": live_cand_mean,
            "shift": cand_shift,
            "threshold": THRESHOLDS["candidates_count"],
            "status": "ALERT" if cand_shift > THRESHOLDS["candidates_count"] else "OK",
        },
    ]

    drift_detected = any(a["status"] == "ALERT" for a in alerts)

    result = {
        "total_predictions": total,
        "mean_prediction": mean_prediction,
        "drift_detected": drift_detected,
        "alerts": alerts,
    }

    out_path = os.path.join(RESULTS_DIR, "step3_s5.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Total predictions logged: {total}")
    print(f"Mean prediction:         {mean_prediction}")
    print(f"Drift detected:          {drift_detected}")
    for a in alerts:
        print(f"  [{a['status']}] {a['feature']}: train={a['train_mean']}, live={a['live_mean']}, shift={a['shift']} (threshold={a['threshold']})")
    print("Saved results/step3_s5.json")


if __name__ == "__main__":
    run_monitor()
