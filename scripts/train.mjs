/**
 * Train the price model from the Excel dataset and write model/model.json.
 *
 * Usage:
 *   node scripts/train.mjs                    # looks for the dataset in data/ or the repo root
 *   node scripts/train.mjs --data path/to.xlsx
 *
 * Runs automatically during every Netlify deploy. If no valid dataset is found,
 * the committed model/model.json (an indicative baseline) is kept unchanged, so
 * the deploy never fails because the dataset is missing.
 *
 * Expected sheet: "Real_Estate_Data" (falls back to the first sheet) with the
 * columns city, state, locality, property_type, area_sqft, price_inr, ... as in
 * india_real_estate_synthetic_120k.xlsx.
 */
import { existsSync, openSync, readSync, closeSync, writeFileSync, readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import ExcelJS from "exceljs";
import { BINARY, NUMERIC, localityKey, isYes, logPrice } from "../src/predict.mjs";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT = join(ROOT, "model", "model.json");
const FILE = "india_real_estate_synthetic_120k.xlsx";

function findDataset() {
  const i = process.argv.indexOf("--data");
  const candidates = i > -1 ? [process.argv[i + 1]] : [join(ROOT, "data", FILE), join(ROOT, FILE)];
  for (const p of candidates) {
    if (!p || !existsSync(p)) continue;
    // A real .xlsx is a zip archive and starts with "PK".
    const fd = openSync(p, "r");
    const buf = Buffer.alloc(2);
    readSync(fd, buf, 0, 2, 0);
    closeSync(fd);
    if (buf.toString() === "PK") return p;
    console.warn(`[train] ${p} is not a valid .xlsx file (it is not a zip archive), skipping.`);
  }
  return null;
}

function cellValue(v) {
  if (v && typeof v === "object") {
    if ("result" in v) return v.result; // formula
    if ("richText" in v) return v.richText.map((t) => t.text).join("");
    if ("text" in v) return v.text;
  }
  return v;
}

async function readRows(path) {
  const reader = new ExcelJS.stream.xlsx.WorkbookReader(path, { sharedStrings: "cache", worksheets: "emit" });
  let fallback = null; // first sheet, used if no "Real_Estate_Data" sheet exists
  for await (const sheet of reader) {
    const isTarget = sheet.name === "Real_Estate_Data";
    if (!isTarget && fallback) {
      for await (const _ of sheet); // drain sheets we don't need
      continue;
    }
    let header = null;
    const rows = [];
    for await (const r of sheet) {
      const vals = r.values.map(cellValue);
      if (!header) {
        header = vals.map((h) => String(h ?? "").trim());
        continue;
      }
      const obj = {};
      header.forEach((h, idx) => { if (h) obj[h] = vals[idx]; });
      rows.push(obj);
    }
    if (isTarget) return rows;
    fallback = rows;
  }
  return fallback ?? [];
}

// ---------- small linear algebra helpers ----------
function solve(A, b) {
  // Gaussian elimination with partial pivoting (A is small and square).
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c] || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / d;
      if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / (row[i] || 1e-12));
}

const quantile = (sorted, q) => {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
};
const median = (arr) => quantile([...arr].sort((a, b) => a - b), 0.5);
const std = (arr) => {
  const m = arr.reduce((s, v) => s + v, 0) / arr.length;
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
};

function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- training ----------
function clean(raw) {
  const seen = new Set();
  const out = [];
  for (const r of raw) {
    const price = Number(r.price_inr), area = Number(r.area_sqft);
    if (!(price > 0) || !(area > 0) || !r.city || !r.locality || !r.property_type) continue;
    if (r.listing_id != null) {
      if (seen.has(r.listing_id)) continue;
      seen.add(r.listing_id);
    }
    const row = {
      city: String(r.city).trim(), state: String(r.state ?? "").trim(), locality: String(r.locality).trim(),
      property_type: String(r.property_type).trim(), furnishing: String(r.furnishing ?? "Unfurnished").trim(),
      area_sqft: area, price_inr: price,
    };
    for (const k of BINARY) row[k] = isYes(r[k]) ? "Yes" : "No";
    for (const k of NUMERIC) row[k] = Number(r[k]) || 0;
    if (row.property_type === "Plot") {
      Object.assign(row, { bedrooms: 0, bathrooms: 0, floor_number: 0, total_floors: 0, lift_available: "No" });
    }
    out.push(row);
  }
  return out;
}

function fit(train, propertyTypes, furnishing) {
  // Dense design: [1, log(area), property type dummies, furnishing dummies, binaries, numerics]
  const ptCols = propertyTypes.slice(1), fuCols = furnishing.slice(1);
  const design = (r) => [
    1, Math.log(r.area_sqft),
    ...ptCols.map((p) => (r.property_type === p ? 1 : 0)),
    ...fuCols.map((f) => (r.furnishing === f ? 1 : 0)),
    ...BINARY.map((k) => (isYes(r[k]) ? 1 : 0)),
    ...NUMERIC.map((k) => r[k]),
  ];
  const X = train.map(design);
  const y = train.map((r) => Math.log(r.price_inr));
  const keys = train.map((r) => localityKey(r.city, r.locality));
  const p = X[0].length;
  const loc = {};
  let beta = new Array(p).fill(0);

  // Backfitting: alternate OLS on the dense features and per-locality mean offsets.
  for (let iter = 0; iter < 12; iter++) {
    const XtX = Array.from({ length: p }, () => new Array(p).fill(0));
    const Xty = new Array(p).fill(0);
    for (let i = 0; i < X.length; i++) {
      const xi = X[i], t = y[i] - (loc[keys[i]] ?? 0);
      for (let a = 0; a < p; a++) {
        Xty[a] += xi[a] * t;
        for (let b = a; b < p; b++) XtX[a][b] += xi[a] * xi[b];
      }
    }
    for (let a = 0; a < p; a++) { XtX[a][a] += 1e-6; for (let b = 0; b < a; b++) XtX[a][b] = XtX[b][a]; }
    beta = solve(XtX, Xty);

    const sum = {}, cnt = {};
    for (let i = 0; i < X.length; i++) {
      let fitted = 0;
      for (let a = 0; a < p; a++) fitted += X[i][a] * beta[a];
      sum[keys[i]] = (sum[keys[i]] ?? 0) + (y[i] - fitted);
      cnt[keys[i]] = (cnt[keys[i]] ?? 0) + 1;
    }
    for (const k in sum) loc[k] = sum[k] / (cnt[k] + 3); // light shrinkage for rare localities
  }

  let j = 0;
  const coef = { intercept: beta[j++], log_area: beta[j++], locality: loc, city_fallback: {},
    property_type: { [propertyTypes[0]]: 0 }, furnishing: { [furnishing[0]]: 0 }, binary: {}, numeric: {} };
  for (const pt of ptCols) coef.property_type[pt] = beta[j++];
  for (const f of fuCols) coef.furnishing[f] = beta[j++];
  for (const k of BINARY) coef.binary[k] = beta[j++];
  for (const k of NUMERIC) coef.numeric[k] = beta[j++];
  return coef;
}

const round = (obj, d = 6) => JSON.parse(JSON.stringify(obj, (_, v) => (typeof v === "number" ? +v.toFixed(d) : v)));

async function main() {
  const path = findDataset();
  if (!path) {
    console.log("[train] No dataset found (data/" + FILE + "). Keeping the existing model/model.json.");
    return;
  }
  const t0 = Date.now();
  console.log(`[train] Loading ${path} ...`);
  const rows = clean(await readRows(path));
  if (rows.length < 200) {
    console.warn(`[train] Only ${rows.length} usable rows found. Keeping the existing model/model.json.`);
    return;
  }
  console.log(`[train] ${rows.length.toLocaleString()} usable rows`);

  const rand = mulberry32(42);
  const shuffled = rows.map((r) => [rand(), r]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
  const cut = Math.floor(shuffled.length * 0.8);
  const train = shuffled.slice(0, cut), test = shuffled.slice(cut);

  const propertyTypes = [...new Set(rows.map((r) => r.property_type))].sort();
  const furnishing = [...new Set(rows.map((r) => r.furnishing))].sort();
  const unf = furnishing.indexOf("Unfurnished"); // use Unfurnished as the reference level
  if (unf > 0) furnishing.unshift(...furnishing.splice(unf, 1));
  const apt = propertyTypes.indexOf("Apartment");
  if (apt > 0) propertyTypes.unshift(...propertyTypes.splice(apt, 1));

  const coef = fit(train, propertyTypes, furnishing);
  const byCity = {};
  for (const [k, v] of Object.entries(coef.locality)) (byCity[k.split("|")[0]] ??= []).push(v);
  for (const [c, vs] of Object.entries(byCity)) coef.city_fallback[c] = median(vs);

  const draft = { coef, residual_quantiles: { low: 0, high: 0 } };
  const resid = train.map((r) => Math.log(r.price_inr) - logPrice(draft, r)).sort((a, b) => a - b);
  draft.residual_quantiles = { low: quantile(resid, 0.1), high: quantile(resid, 0.9) };

  // ---- hold-out metrics ----
  const z = test.map((r) => logPrice(draft, r));
  const truth = test.map((r) => r.price_inr), pred = z.map(Math.exp);
  const meanT = truth.reduce((s, v) => s + v, 0) / truth.length;
  const ssRes = truth.reduce((s, v, i) => s + (v - pred[i]) ** 2, 0);
  const ssTot = truth.reduce((s, v) => s + (v - meanT) ** 2, 0);
  const ape = truth.map((v, i) => Math.abs(pred[i] - v) / v);
  const metrics = {
    r2: 1 - ssRes / ssTot,
    mae_inr: truth.reduce((s, v, i) => s + Math.abs(v - pred[i]), 0) / truth.length,
    mape_pct: (ape.reduce((s, v) => s + v, 0) / ape.length) * 100,
    median_ape_pct: median(ape) * 100,
    interval_coverage_pct: (test.filter((r, i) => {
      const lo = Math.exp(z[i] + draft.residual_quantiles.low), hi = Math.exp(z[i] + draft.residual_quantiles.high);
      return r.price_inr >= lo && r.price_inr <= hi;
    }).length / test.length) * 100,
    train_rows: train.length, test_rows: test.length,
  };
  console.log("[train] Hold-out metrics:", JSON.stringify(round(metrics, 3)));

  // ---- feature importance: spread of each feature's contribution to log(price) ----
  const contrib = {
    locality: (r) => coef.locality[localityKey(r.city, r.locality)] ?? 0,
    area_sqft: (r) => coef.log_area * Math.log(r.area_sqft),
    property_type: (r) => coef.property_type[r.property_type] ?? 0,
    furnishing: (r) => coef.furnishing[r.furnishing] ?? 0,
    ...Object.fromEntries(BINARY.map((k) => [k, (r) => (isYes(r[k]) ? coef.binary[k] : 0)])),
    ...Object.fromEntries(NUMERIC.map((k) => [k, (r) => coef.numeric[k] * r[k]])),
  };
  const spread = Object.entries(contrib).map(([feature, fn]) => ({ feature, s: std(test.map(fn)) }));
  const total = spread.reduce((s, d) => s + d.s, 0) || 1;
  const feature_importance = spread
    .map(({ feature, s }) => ({ feature, importance: s / total }))
    .sort((a, b) => b.importance - a.importance);

  // ---- metadata for the UI ----
  const city_localities = {}, city_state = {}, cityRows = {};
  for (const r of rows) {
    (city_localities[r.city] ??= new Set()).add(r.locality);
    city_state[r.city] ??= r.state;
    (cityRows[r.city] ??= []).push(r);
  }
  for (const c in city_localities) city_localities[c] = [...city_localities[c]].sort();
  const nonPlot = rows.filter((r) => r.property_type !== "Plot");
  const max = (arr, k) => arr.reduce((m, r) => Math.max(m, r[k]), 0);
  const medianOf = (arr, k) => median(arr.map((r) => r[k]));

  const model = {
    source: "dataset",
    note: `Trained on ${rows.length.toLocaleString("en-IN")} listings from ${FILE}.`,
    trained_at: new Date().toISOString(),
    trained_rows: rows.length,
    cities: Object.keys(city_localities).sort(),
    city_localities, city_state,
    property_types: [...propertyTypes].sort(),
    furnishing: [...furnishing].sort(),
    ranges: {
      area_sqft: [Math.floor(Math.min(...rows.map((r) => r.area_sqft))), Math.ceil(max(rows, "area_sqft"))],
      bedrooms: [1, Math.max(1, max(nonPlot, "bedrooms"))],
      bathrooms: [1, Math.max(1, max(nonPlot, "bathrooms"))],
      property_age_years: [0, max(rows, "property_age_years")],
      total_floors: [1, Math.max(1, max(rows, "total_floors"))],
      parking_spaces: [0, max(rows, "parking_spaces")],
      distance_to_metro_km: [0.1, 25],
      distance_to_school_km: [0.1, 12],
      distance_to_hospital_km: [0.1, 15],
    },
    defaults: {
      distance_to_metro_km: +medianOf(rows, "distance_to_metro_km").toFixed(2),
      distance_to_school_km: +medianOf(rows, "distance_to_school_km").toFixed(2),
      distance_to_hospital_km: +medianOf(rows, "distance_to_hospital_km").toFixed(2),
    },
    coef: round(coef),
    residual_quantiles: round(draft.residual_quantiles),
    metrics: round(metrics, 4),
    feature_importance: round(feature_importance, 4),
    city_stats: Object.entries(cityRows).map(([city, rs]) => ({
      city,
      median_price: Math.round(medianOf(rs, "price_inr")),
      median_psf: Math.round(median(rs.map((r) => r.price_inr / r.area_sqft))),
      listings: rs.length,
    })).sort((a, b) => b.median_psf - a.median_psf),
  };
  writeFileSync(OUT, JSON.stringify(model, null, 2) + "\n");
  console.log(`[train] Done in ${((Date.now() - t0) / 1000).toFixed(1)}s -> model/model.json`);
}

main().catch((err) => {
  // Never break the deploy over training: keep the committed model instead.
  console.error("[train] Training failed, keeping the existing model/model.json:", err);
  try { JSON.parse(readFileSync(OUT, "utf8")); } catch { process.exit(1); }
});
