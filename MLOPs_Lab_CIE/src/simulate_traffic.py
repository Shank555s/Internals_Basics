import random
import time
import requests

API_URL = "http://localhost:8500/estimate"

# 35 normal requests — values within training distribution
NORMAL_REQUESTS = [
    {"constituency_population": 230630, "candidates_count": 13, "literacy_rate": 86.0, "is_urban": 1},
    {"constituency_population": 190888, "candidates_count": 10, "literacy_rate": 93.0, "is_urban": 1},
    {"constituency_population": 262657, "candidates_count": 9,  "literacy_rate": 64.8, "is_urban": 1},
    {"constituency_population": 146225, "candidates_count": 11, "literacy_rate": 52.5, "is_urban": 0},
    {"constituency_population": 361259, "candidates_count": 15, "literacy_rate": 73.3, "is_urban": 0},
    {"constituency_population": 322992, "candidates_count": 12, "literacy_rate": 63.4, "is_urban": 0},
    {"constituency_population": 152942, "candidates_count": 15, "literacy_rate": 69.9, "is_urban": 1},
    {"constituency_population": 307847, "candidates_count": 8,  "literacy_rate": 60.9, "is_urban": 0},
    {"constituency_population": 253950, "candidates_count": 5,  "literacy_rate": 76.2, "is_urban": 0},
    {"constituency_population": 246599, "candidates_count": 10, "literacy_rate": 78.1, "is_urban": 1},
    {"constituency_population": 235449, "candidates_count": 8,  "literacy_rate": 93.2, "is_urban": 0},
    {"constituency_population": 289403, "candidates_count": 3,  "literacy_rate": 79.5, "is_urban": 0},
    {"constituency_population": 146016, "candidates_count": 7,  "literacy_rate": 59.4, "is_urban": 1},
    {"constituency_population": 385422, "candidates_count": 10, "literacy_rate": 60.1, "is_urban": 0},
    {"constituency_population": 195758, "candidates_count": 10, "literacy_rate": 52.5, "is_urban": 0},
    {"constituency_population": 354287, "candidates_count": 10, "literacy_rate": 54.3, "is_urban": 1},
    {"constituency_population": 319782, "candidates_count": 14, "literacy_rate": 53.6, "is_urban": 0},
    {"constituency_population": 147452, "candidates_count": 13, "literacy_rate": 76.5, "is_urban": 1},
    {"constituency_population": 302899, "candidates_count": 15, "literacy_rate": 72.2, "is_urban": 0},
    {"constituency_population": 200000, "candidates_count": 8,  "literacy_rate": 70.0, "is_urban": 0},
    {"constituency_population": 275000, "candidates_count": 10, "literacy_rate": 65.0, "is_urban": 1},
    {"constituency_population": 180000, "candidates_count": 6,  "literacy_rate": 80.0, "is_urban": 0},
    {"constituency_population": 330000, "candidates_count": 11, "literacy_rate": 58.0, "is_urban": 1},
    {"constituency_population": 160000, "candidates_count": 9,  "literacy_rate": 75.0, "is_urban": 0},
    {"constituency_population": 420000, "candidates_count": 7,  "literacy_rate": 88.0, "is_urban": 0},
    {"constituency_population": 250000, "candidates_count": 12, "literacy_rate": 62.0, "is_urban": 1},
    {"constituency_population": 310000, "candidates_count": 4,  "literacy_rate": 72.0, "is_urban": 0},
    {"constituency_population": 170000, "candidates_count": 14, "literacy_rate": 55.0, "is_urban": 0},
    {"constituency_population": 460000, "candidates_count": 8,  "literacy_rate": 90.0, "is_urban": 1},
    {"constituency_population": 290000, "candidates_count": 11, "literacy_rate": 67.0, "is_urban": 0},
    {"constituency_population": 220000, "candidates_count": 6,  "literacy_rate": 84.0, "is_urban": 1},
    {"constituency_population": 350000, "candidates_count": 13, "literacy_rate": 60.0, "is_urban": 0},
    {"constituency_population": 130000, "candidates_count": 9,  "literacy_rate": 78.0, "is_urban": 1},
    {"constituency_population": 400000, "candidates_count": 5,  "literacy_rate": 70.0, "is_urban": 0},
    {"constituency_population": 260000, "candidates_count": 10, "literacy_rate": 85.0, "is_urban": 1},
]

# 15 drifted requests — high population (out-of-range capped to max), high candidates_count capped to 15
# Note: API validates 50000-500000 for population and 3-15 for candidates_count
# Drifted = still valid range but at extreme high end to simulate distribution shift
DRIFTED_REQUESTS = [
    {"constituency_population": 495000, "candidates_count": 15, "literacy_rate": 68.0, "is_urban": 0},
    {"constituency_population": 490000, "candidates_count": 15, "literacy_rate": 87.3, "is_urban": 1},
    {"constituency_population": 498000, "candidates_count": 15, "literacy_rate": 83.7, "is_urban": 1},
    {"constituency_population": 499000, "candidates_count": 15, "literacy_rate": 84.1, "is_urban": 0},
    {"constituency_population": 488000, "candidates_count": 14, "literacy_rate": 66.6, "is_urban": 1},
    {"constituency_population": 492000, "candidates_count": 15, "literacy_rate": 65.3, "is_urban": 0},
    {"constituency_population": 496000, "candidates_count": 14, "literacy_rate": 57.9, "is_urban": 0},
    {"constituency_population": 497000, "candidates_count": 15, "literacy_rate": 69.1, "is_urban": 0},
    {"constituency_population": 493000, "candidates_count": 15, "literacy_rate": 90.5, "is_urban": 1},
    {"constituency_population": 499500, "candidates_count": 15, "literacy_rate": 67.4, "is_urban": 0},
    {"constituency_population": 491000, "candidates_count": 15, "literacy_rate": 68.0, "is_urban": 0},
    {"constituency_population": 494000, "candidates_count": 15, "literacy_rate": 68.8, "is_urban": 1},
    {"constituency_population": 487000, "candidates_count": 14, "literacy_rate": 72.7, "is_urban": 0},
    {"constituency_population": 489000, "candidates_count": 15, "literacy_rate": 67.9, "is_urban": 1},
    {"constituency_population": 500000, "candidates_count": 15, "literacy_rate": 64.7, "is_urban": 0},
]


def send_requests():
    all_requests = NORMAL_REQUESTS + DRIFTED_REQUESTS
    success = 0
    for i, payload in enumerate(all_requests):
        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            if resp.status_code == 200:
                success += 1
                print(f"[{i+1}/50] OK — prediction={resp.json()['voter_turnout_pct']}")
            else:
                print(f"[{i+1}/50] ERROR {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[{i+1}/50] EXCEPTION: {e}")
        time.sleep(0.05)

    print(f"\nSent {len(all_requests)} requests, {success} successful.")


if __name__ == "__main__":
    send_requests()
