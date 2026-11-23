"""
Train the real-estate price model.

Usage:
    python train.py                       # uses data/india_real_estate_synthetic_120k.xlsx
    python train.py --data path/to.xlsx

Outputs (in models/):
    price_model.joblib   - dict with median / low / high models
    metadata.json        - dropdown options, valid ranges, metrics, feature importance
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"

# NOTE: price_per_sqft_inr is deliberately NOT a feature. It is derived from the
# target (price_inr / area_sqft), so using it would leak the answer.
CATEGORICAL = ["city", "locality_key", "property_type", "furnishing"]
BINARY = ["gated_community", "lift_available", "gym_available", "swimming_pool"]
NUMERIC = [
    "bedrooms", "bathrooms", "area_sqft", "property_age_years", "floor_number",
    "total_floors", "parking_spaces", "distance_to_metro_km",
    "distance_to_school_km", "distance_to_hospital_km",
]
FEATURES = CATEGORICAL + BINARY + NUMERIC


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Shared by training and the web app so both see identical features."""
    out = df.copy()
    # Two locality names (Hebbal, Indira Nagar) exist in two different cities,
    # so locality must be keyed together with city.
    out["locality_key"] = out["city"].astype(str) + "|" + out["locality"].astype(str)
    for c in BINARY:
        out[c] = (out[c].astype(str).str.lower() == "yes").astype(int)
    return out[FEATURES]


def make_model(**kw):
    return HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=30, categorical_features="from_dtype", random_state=42, **kw,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "india_real_estate_synthetic_120k.xlsx"))
    args = ap.parse_args()

    t0 = time.time()
    print("Loading data...")
    raw = pd.read_excel(args.data, sheet_name="Real_Estate_Data")
    raw = raw.dropna(subset=["price_inr"]).drop_duplicates("listing_id")
    print(f"  {len(raw):,} rows")

    X = prepare(raw)
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    y = np.log(raw["price_inr"].values)  # log target: prices span 1.8M - 187M

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training median model...")
    med = make_model(loss="squared_error").fit(X_tr, y_tr)
    print("Training low (10th pct) / high (90th pct) models...")
    low = make_model(loss="quantile", quantile=0.10).fit(X_tr, y_tr)
    high = make_model(loss="quantile", quantile=0.90).fit(X_tr, y_tr)

    pred = np.exp(med.predict(X_te))
    true = np.exp(y_te)
    lo, hi = np.exp(low.predict(X_te)), np.exp(high.predict(X_te))
    metrics = {
        "r2": float(r2_score(true, pred)),
        "r2_log": float(r2_score(y_te, np.log(pred))),
        "mae_inr": float(mean_absolute_error(true, pred)),
        "mape_pct": float(mean_absolute_percentage_error(true, pred) * 100),
        "median_ape_pct": float(np.median(np.abs(pred - true) / true) * 100),
        "interval_coverage_pct": float(np.mean((true >= lo) & (true <= hi)) * 100),
        "train_rows": int(len(X_tr)), "test_rows": int(len(X_te)),
    }
    print("Metrics:", json.dumps(metrics, indent=2))

    print("Computing feature importance (permutation, 5k sample)...")
    idx = np.random.RandomState(0).choice(len(X_te), size=5000, replace=False)
    pi = permutation_importance(med, X_te.iloc[idx], y_te[idx], n_repeats=3, random_state=0, n_jobs=-1)
    importance = sorted(
        ({"feature": f, "importance": float(v)} for f, v in zip(FEATURES, pi.importances_mean)),
        key=lambda d: -d["importance"],
    )

    # ---- metadata for the UI ----
    non_plot = raw[raw["property_type"] != "Plot"]
    city_loc = (raw.groupby("city")["locality"].apply(lambda s: sorted(s.unique())).to_dict())
    city_state = raw.drop_duplicates("city").set_index("city")["state"].to_dict()
    city_stats = raw.groupby("city").agg(
        median_price=("price_inr", "median"), median_psf=("price_per_sqft_inr", "median"),
        listings=("price_inr", "size")).round(0).reset_index().to_dict("records")
    meta = {
        "cities": sorted(city_loc),
        "city_localities": city_loc,
        "city_state": city_state,
        "property_types": sorted(raw["property_type"].unique()),
        "furnishing": sorted(raw["furnishing"].unique()),
        "ranges": {
            "area_sqft": [int(raw.area_sqft.min()), int(raw.area_sqft.max())],
            "bedrooms": [1, int(non_plot.bedrooms.max())],
            "bathrooms": [1, int(non_plot.bathrooms.max())],
            "property_age_years": [0, int(raw.property_age_years.max())],
            "total_floors": [1, int(raw.total_floors.max())],
            "parking_spaces": [0, int(raw.parking_spaces.max())],
            "distance_to_metro_km": [0.1, 25.0],
            "distance_to_school_km": [0.1, 12.0],
            "distance_to_hospital_km": [0.1, 15.0],
        },
        "defaults": {
            "distance_to_metro_km": float(raw.distance_to_metro_km.median()),
            "distance_to_school_km": float(raw.distance_to_school_km.median()),
            "distance_to_hospital_km": float(raw.distance_to_hospital_km.median()),
        },
        "metrics": metrics,
        "feature_importance": importance,
        "city_stats": city_stats,
        "trained_rows": int(len(raw)),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"median": med, "low": low, "high": high, "features": FEATURES,
                 "categories": {c: list(X[c].cat.categories) for c in CATEGORICAL}},
                MODEL_DIR / "price_model.joblib", compress=3)
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Done in {time.time() - t0:.0f}s -> models/")


if __name__ == "__main__":
    main()
