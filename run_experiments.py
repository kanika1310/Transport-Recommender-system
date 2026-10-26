"""Train, compare and ablate transport-mode recommenders.

Usage:
    python src/run_experiments.py --data data/trips.csv
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "mode"
NUM = ["distance_km", "hour", "is_weekend", "group_size", "has_luggage"]
CAT = ["weather", "traffic", "budget"]
ALL_FEATURES = NUM + CAT

# Feature groups used in the ablation study (distance is always kept).
GROUPS = {
    "time (hour, is_weekend)": ["hour", "is_weekend"],
    "environment (weather, traffic)": ["weather", "traffic"],
    "user/trip (budget, group_size, has_luggage)": ["budget", "group_size", "has_luggage"],
}


class RuleBaseline(BaseEstimator, ClassifierMixin):
    """Hand-written if-else recommender, similar to what a simple app might do."""

    def fit(self, X, y):
        self.classes_ = np.array(sorted(set(y)))
        return self

    def _rule(self, row):
        if row["distance_km"] <= 3:
            return "auto"
        if row["budget"] == "low":
            return "bus"
        if row["traffic"] == "high" and row["distance_km"] > 5:
            return "metro"
        return "cab"

    def predict(self, X):
        return np.array([self._rule(r) for _, r in X.iterrows()])

    def predict_proba(self, X):
        pred = self.predict(X)
        proba = np.full((len(pred), len(self.classes_)), 0.02)
        for i, p in enumerate(pred):
            if p in self.classes_:
                proba[i, list(self.classes_).index(p)] = 1.0
        return proba / proba.sum(axis=1, keepdims=True)


def make_pre(cols):
    num = [c for c in NUM if c in cols]
    cat = [c for c in CAT if c in cols]
    parts = []
    if num:
        parts.append(("num", StandardScaler(), num))
    if cat:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    return ColumnTransformer(parts)


def build_models(cols, seed):
    return {
        "Majority class": DummyClassifier(strategy="prior"),
        "Rule-based": RuleBaseline(),
        "Logistic Regression": Pipeline([("pre", make_pre(cols)),
                                         ("clf", LogisticRegression(max_iter=1000))]),
        "Random Forest": Pipeline([("pre", make_pre(cols)),
                                   ("clf", RandomForestClassifier(n_estimators=300, random_state=seed))]),
        "Gradient Boosting": Pipeline([("pre", make_pre(cols)),
                                       ("clf", GradientBoostingClassifier(random_state=seed))]),
    }


def top_k_scorer(k):
    def scorer(est, X, y):
        proba = est.predict_proba(X)
        top = est.classes_[np.argsort(proba, axis=1)[:, -k:]]
        return float(np.mean([yi in row for yi, row in zip(y, top)]))
    return scorer


SCORING = {"accuracy": "accuracy", "macro_f1": "f1_macro", "top2_accuracy": top_k_scorer(2)}


def evaluate(model, X, y, cv):
    res = cross_validate(model, X, y, cv=cv, scoring=SCORING)
    return {m: (res[f"test_{m}"].mean(), res[f"test_{m}"].std()) for m in SCORING}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/trips.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    if not os.path.exists(a.data):
        raise SystemExit(f"{a.data} not found. Put your dataset there, or for a dry run use "
                         "`python src/generate_synthetic.py` and pass --data data/trips_synthetic.csv")
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.data)
    missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in dataset: {missing}")
    df = df.dropna(subset=ALL_FEATURES + [TARGET])
    X, y = df[ALL_FEATURES], df[TARGET]
    cv = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=a.seed)

    # 1) model comparison
    rows = []
    for name, model in build_models(ALL_FEATURES, a.seed).items():
        s = evaluate(model, X, y, cv)
        rows.append({"model": name, **{f"{m}_mean": v[0] for m, v in s.items()},
                     **{f"{m}_std": v[1] for m, v in s.items()}})
        print(f"{name:22s} acc={s['accuracy'][0]:.3f}  macroF1={s['macro_f1'][0]:.3f}  top2={s['top2_accuracy'][0]:.3f}")
    comp = pd.DataFrame(rows)
    comp.to_csv(f"{a.out}/model_comparison.csv", index=False)

    # 2) ablation with Random Forest
    abl = []
    full = evaluate(build_models(ALL_FEATURES, a.seed)["Random Forest"], X, y, cv)
    abl.append({"setting": "all features", "macro_f1": full["macro_f1"][0], "accuracy": full["accuracy"][0]})
    for gname, gcols in GROUPS.items():
        keep = [c for c in ALL_FEATURES if c not in gcols]
        s = evaluate(build_models(keep, a.seed)["Random Forest"], X[keep], y, cv)
        abl.append({"setting": f"without {gname}", "macro_f1": s["macro_f1"][0], "accuracy": s["accuracy"][0]})
    s = evaluate(build_models(["distance_km"], a.seed)["Random Forest"], X[["distance_km"]], y, cv)
    abl.append({"setting": "distance only", "macro_f1": s["macro_f1"][0], "accuracy": s["accuracy"][0]})
    abl = pd.DataFrame(abl)
    abl.to_csv(f"{a.out}/ablation.csv", index=False)
    print("\nAblation (Random Forest):\n" + abl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 3) plots
    ax = comp.plot.bar(x="model", y=["accuracy_mean", "macro_f1_mean", "top2_accuracy_mean"], figsize=(8, 4))
    ax.set_ylabel("score (5-fold CV mean)"); ax.set_xlabel(""); ax.legend(["Accuracy", "Macro-F1", "Top-2 acc."])
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(f"{a.out}/model_comparison.png", dpi=200); plt.close()

    rf = build_models(ALL_FEATURES, a.seed)["Random Forest"]
    pred = cross_val_predict(rf, X, y, cv=cv)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, pred, ax=ax, colorbar=False, cmap="Blues")
    plt.tight_layout(); plt.savefig(f"{a.out}/confusion_matrix_rf.png", dpi=200); plt.close()

    rf.fit(X, y)
    names = rf.named_steps["pre"].get_feature_names_out()
    imp = pd.Series(rf.named_steps["clf"].feature_importances_, index=names).sort_values().tail(12)
    imp.plot.barh(figsize=(6, 4)); plt.xlabel("Random Forest importance"); plt.tight_layout()
    plt.savefig(f"{a.out}/feature_importance.png", dpi=200); plt.close()

    with open(f"{a.out}/run_info.json", "w") as f:
        json.dump({"data": a.data, "rows": int(len(df)), "classes": sorted(y.unique().tolist()),
                   "class_counts": y.value_counts().to_dict(), "folds": a.folds, "seed": a.seed}, f, indent=2)
    print(f"\nSaved tables, plots and run_info.json to {a.out}/")


if __name__ == "__main__":
    main()
