"""Generate a SIMULATED trip dataset for smoke-testing the pipeline.

WARNING: labels come from a hand-written stochastic choice model, not from real
travellers. Use this only to check that the code runs. For the paper, replace
data/trips.csv with real data (e.g. a Google Form survey) using the same columns.
"""
import argparse
import numpy as np
import pandas as pd

MODES = ["bus", "metro", "auto", "cab", "bike"]


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dist = np.round(np.clip(rng.gamma(2.0, 3.0, n), 0.5, 35), 1)
    hour = rng.integers(5, 24, n)
    weekend = rng.binomial(1, 0.29, n)
    weather = rng.choice(["clear", "rain", "hot"], n, p=[0.65, 0.15, 0.20])
    peak = (((hour >= 8) & (hour <= 10)) | ((hour >= 17) & (hour <= 20))) & (weekend == 0)
    traffic = np.where(
        peak,
        rng.choice(["medium", "high"], n, p=[0.3, 0.7]),
        rng.choice(["low", "medium", "high"], n, p=[0.5, 0.35, 0.15]),
    )
    budget = rng.choice(["low", "medium", "high"], n, p=[0.4, 0.4, 0.2])
    group = rng.integers(1, 6, n)
    luggage = rng.binomial(1, 0.2, n)

    night = ((hour >= 22) | (hour < 6)).astype(float)
    wet = (weather == "rain").astype(float)
    hot = (weather == "hot").astype(float)
    jam = (traffic == "high").astype(float)
    b_low = (budget == "low").astype(float)
    b_mid = (budget == "medium").astype(float)
    b_high = (budget == "high").astype(float)

    u = np.zeros((n, 5))
    u[:, 0] = 1.2 - 0.05 * dist + 1.0 * b_low - 0.8 * b_high - 0.6 * jam - 0.5 * luggage - 0.3 * (group >= 4)
    u[:, 1] = (-0.5 + 0.15 * np.minimum(dist, 12) - 0.1 * np.maximum(dist - 20, 0)
               + 0.5 * jam + 0.4 * (wet + hot) + 0.2 * b_mid - 1.5 * night - 0.3 * luggage)
    u[:, 2] = (1.0 - 0.18 * dist + 0.4 * b_mid + 0.4 * ((group >= 2) & (group <= 3))
               + 0.4 * luggage + 0.3 * wet + 0.2 * night)
    u[:, 3] = (-1.0 + 0.04 * dist + 1.6 * b_high - 1.0 * b_low + 0.8 * (group >= 4)
               + 0.8 * luggage + 0.4 * (wet + hot) + 0.8 * night)
    u[:, 4] = (0.8 - 0.25 * dist + 0.3 * b_low - 1.0 * wet - 0.4 * hot
               - 0.4 * (group - 1) - 1.0 * luggage + 0.2 * jam)

    p = np.exp(u - u.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    idx = np.minimum((p.cumsum(axis=1) < rng.random(n)[:, None]).sum(axis=1), 4)

    return pd.DataFrame({
        "distance_km": dist, "hour": hour, "is_weekend": weekend,
        "weather": weather, "traffic": traffic, "budget": budget,
        "group_size": group, "has_luggage": luggage,
        "mode": [MODES[i] for i in idx],
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/trips_synthetic.csv")
    a = ap.parse_args()
    df = generate(a.n, a.seed)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} simulated rows to {a.out}")
    print(df["mode"].value_counts())
