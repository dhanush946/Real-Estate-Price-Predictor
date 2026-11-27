# PriceWise India – Real Estate Price Predictor

Web app that predicts property prices from your 120k-row dataset.

## Structure
```
real-estate-predictor/
├── data/india_real_estate_synthetic_120k.xlsx   # your dataset
├── train.py            # trains the model, writes models/
├── app.py              # Flask server + JSON API
├── models/             # price_model.joblib, metadata.json (generated)
├── templates/index.html
├── static/style.css, static/app.js
├── requirements.txt, Dockerfile, Procfile
```

## Run it
```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train.py        # ~1 min; regenerates models/ for YOUR scikit-learn version
python app.py          # open http://127.0.0.1:5000
```
Run `train.py` once on your machine so the saved model matches your installed scikit-learn.

## API
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/meta` | GET | dropdown options, metrics, city stats |
| `/api/predict` | POST | JSON of property fields -> price, low/high range, price/sqft |
| `/api/whatif` | POST | same input -> price at 50%-150% of the area (chart) |

Example:
```bash
curl -X POST localhost:5000/api/predict -H "Content-Type: application/json" -d '{
 "city":"Pune","locality":"Baner","property_type":"Apartment","area_sqft":1200,
 "bedrooms":2,"bathrooms":2,"property_age_years":5,"floor_number":5,"total_floors":15,
 "furnishing":"Semi-Furnished","parking_spaces":1,"gated_community":"Yes",
 "lift_available":"Yes","gym_available":"No","swimming_pool":"No"}'
```
(Use localities that exist in the dataset for the chosen city; see `/api/meta`.)

## Modelling notes
- **Target:** `log(price_inr)` (prices range ₹18 L – ₹18.7 Cr).
- **Leakage avoided:** `price_per_sqft_inr` is excluded because it is `price / area`.
- **Model:** scikit-learn `HistGradientBoostingRegressor` (native categorical support), plus two quantile models (10th/90th percentile) for the price range.
- **Locality** is keyed as `city|locality` because "Hebbal" and "Indira Nagar" occur in two cities each.
- **Plots** are handled per the data dictionary: bedrooms, bathrooms, floors and lift are forced to 0/No.
- **Hold-out result (20% split):** R² ≈ 0.94, median error ≈ 9%, MAE ≈ ₹18.6 L. The 10–90% band covers ≈ 74% of test cases, a bit narrower than the nominal 80%, so treat it as approximate.
- The data is **synthetic**: results show how the model behaves on this data, not real market accuracy.

## Deploy
- **Docker:** `docker build -t pricewise . && docker run -p 8000:8000 pricewise`
- **Render / Railway / Heroku:** uses `Procfile` (`gunicorn app:app`). Commit `models/` or run `python train.py` in the build step.

## Ideas to extend
Save prediction history in SQLite, add a map of localities, compare two properties, export a PDF report, or retrain with real listing data (keep the same column names).
