"""
Flask app: serves the website and a JSON prediction API.

Run:   python app.py            (dev, http://127.0.0.1:5000)
Prod:  gunicorn app:app -b 0.0.0.0:8000
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from train import BINARY, CATEGORICAL, prepare

ROOT = Path(__file__).parent
app = Flask(__name__)

BUNDLE = joblib.load(ROOT / "models" / "price_model.joblib")
META = json.loads((ROOT / "models" / "metadata.json").read_text())
RANGES = META["ranges"]


def fmt_inr(v: float) -> str:
    """Indian-style short format: 1.25 Cr / 45.30 L."""
    if v >= 1e7:
        return f"₹{v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"₹{v / 1e5:.2f} L"
    return f"₹{v:,.0f}"


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def parse_payload(p: dict):
    """Validate + normalise user input. Returns (row_dict, error)."""
    city = p.get("city")
    if city not in META["city_localities"]:
        return None, "Please choose a valid city."
    locality = p.get("locality")
    if locality not in META["city_localities"][city]:
        return None, "Please choose a locality that belongs to the selected city."
    ptype = p.get("property_type")
    if ptype not in META["property_types"]:
        return None, "Please choose a valid property type."
    furnishing = p.get("furnishing", "Unfurnished")
    if furnishing not in META["furnishing"]:
        return None, "Invalid furnishing option."

    try:
        area = float(p.get("area_sqft"))
    except (TypeError, ValueError):
        return None, "Area must be a number."
    lo, hi = RANGES["area_sqft"]
    if not lo <= area <= hi:
        return None, f"Area must be between {lo} and {hi} sq ft (the range the model was trained on)."

    def num(key, default=0.0):
        try:
            return float(p.get(key, default))
        except (TypeError, ValueError):
            return float(default)

    def yes(key):
        return "Yes" if str(p.get(key, "No")).lower() in ("yes", "true", "1", "on") else "No"

    is_plot = ptype == "Plot"
    row = {
        "city": city, "locality": locality, "property_type": ptype,
        "area_sqft": area, "furnishing": furnishing,
        "parking_spaces": int(clamp(num("parking_spaces"), *RANGES["parking_spaces"])),
        "gated_community": yes("gated_community"),
        "gym_available": yes("gym_available"),
        "swimming_pool": yes("swimming_pool"),
        "property_age_years": int(clamp(num("property_age_years"), *RANGES["property_age_years"])),
        "distance_to_metro_km": clamp(num("distance_to_metro_km", META["defaults"]["distance_to_metro_km"]), *RANGES["distance_to_metro_km"]),
        "distance_to_school_km": clamp(num("distance_to_school_km", META["defaults"]["distance_to_school_km"]), *RANGES["distance_to_school_km"]),
        "distance_to_hospital_km": clamp(num("distance_to_hospital_km", META["defaults"]["distance_to_hospital_km"]), *RANGES["distance_to_hospital_km"]),
    }
    if is_plot:
        # Dataset convention: plots have no rooms / floors / lift.
        row.update(bedrooms=0, bathrooms=0, floor_number=0, total_floors=0, lift_available="No")
    else:
        total = int(clamp(num("total_floors", 1), *RANGES["total_floors"]))
        row.update(
            bedrooms=int(clamp(num("bedrooms", 2), *RANGES["bedrooms"])),
            bathrooms=int(clamp(num("bathrooms", 2), *RANGES["bathrooms"])),
            total_floors=total,
            floor_number=int(clamp(num("floor_number", 0), 0, total)),
            lift_available=yes("lift_available"),
        )
    return row, None


def to_frame(row: dict) -> pd.DataFrame:
    X = prepare(pd.DataFrame([row]))
    for c in CATEGORICAL:  # use the exact category set seen in training
        X[c] = pd.Categorical(X[c], categories=BUNDLE["categories"][c])
    return X[BUNDLE["features"]]


def predict_row(row: dict):
    X = to_frame(row)
    mid = float(np.exp(BUNDLE["median"].predict(X)[0]))
    low = float(np.exp(BUNDLE["low"].predict(X)[0]))
    high = float(np.exp(BUNDLE["high"].predict(X)[0]))
    low, high = min(low, mid), max(high, mid)  # keep the band around the estimate
    return mid, low, high


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    keep = ("cities", "city_localities", "city_state", "property_types", "furnishing",
            "ranges", "defaults", "metrics", "feature_importance", "city_stats", "trained_rows")
    return jsonify({k: META[k] for k in keep})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    row, err = parse_payload(request.get_json(silent=True) or {})
    if err:
        return jsonify({"error": err}), 400
    mid, low, high = predict_row(row)
    area = row["area_sqft"]

    # Comparison with the city's typical rate
    city_psf = next((c["median_psf"] for c in META["city_stats"] if c["city"] == row["city"]), None)
    psf = mid / area
    return jsonify({
        "price": round(mid), "low": round(low), "high": round(high),
        "price_label": fmt_inr(mid), "low_label": fmt_inr(low), "high_label": fmt_inr(high),
        "price_per_sqft": round(psf),
        "city_median_psf": city_psf,
        "vs_city_pct": round((psf / city_psf - 1) * 100, 1) if city_psf else None,
        "inputs": row,
    })


@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    """Price at several areas (for the sensitivity chart)."""
    row, err = parse_payload(request.get_json(silent=True) or {})
    if err:
        return jsonify({"error": err}), 400
    lo, hi = RANGES["area_sqft"]
    base = row["area_sqft"]
    areas = sorted({int(clamp(base * f, lo, hi)) for f in np.linspace(0.5, 1.5, 11)})
    pts = []
    for a in areas:
        r = dict(row, area_sqft=float(a))
        pts.append({"area": a, "price": round(predict_row(r)[0])})
    return jsonify({"points": pts})


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
