/**
 * Shared pricing model logic, used by the API server (netlify/server) and the
 * training script (scripts/train.mjs) so both see identical features.
 *
 * The model is a log-linear (hedonic) regression:
 *
 *   log(price) = intercept
 *              + b_area * log(area_sqft)
 *              + locality_offset[city|locality]
 *              + property_type_effect + furnishing_effect
 *              + sum(binary amenity effects) + sum(numeric coefficients * value)
 *
 * The 10th / 90th percentile band comes from the residual distribution seen in
 * training. Parameters live in model/model.json.
 */

export const BINARY = ["gated_community", "lift_available", "gym_available", "swimming_pool"];
export const NUMERIC = [
  "bedrooms", "bathrooms", "property_age_years", "floor_number", "total_floors",
  "parking_spaces", "distance_to_metro_km", "distance_to_school_km", "distance_to_hospital_km",
];

// Locality names repeat across cities, so a locality is always keyed with its city.
export const localityKey = (city, locality) => `${city}|${locality}`;

export const isYes = (v) => ["yes", "true", "1", "on", "y"].includes(String(v ?? "").trim().toLowerCase());

export function logPrice(model, row) {
  const c = model.coef;
  let z = c.intercept + c.log_area * Math.log(row.area_sqft);
  z += c.locality[localityKey(row.city, row.locality)] ?? c.city_fallback?.[row.city] ?? 0;
  z += c.property_type[row.property_type] ?? 0;
  z += c.furnishing[row.furnishing] ?? 0;
  for (const k of BINARY) if (isYes(row[k])) z += c.binary[k] ?? 0;
  for (const k of NUMERIC) z += (c.numeric[k] ?? 0) * (Number(row[k]) || 0);
  return z;
}

export function predict(model, row) {
  const z = logPrice(model, row);
  const q = model.residual_quantiles;
  const mid = Math.exp(z);
  return {
    mid,
    low: Math.min(mid, Math.exp(z + q.low)),
    high: Math.max(mid, Math.exp(z + q.high)),
  };
}
