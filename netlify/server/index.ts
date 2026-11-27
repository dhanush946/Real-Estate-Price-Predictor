import type { Config } from "@netlify/server";
// @ts-ignore — plain JS module shared with the training script
import { predict } from "../../src/predict.mjs";
import model from "../../model/model.json" with { type: "json" };

type Row = Record<string, string | number>;
type Ranges = Record<string, [number, number]>;

const M = model as any;
const RANGES: Ranges = M.ranges;

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));

function fmtInr(v: number) {
  if (v >= 1e7) return `₹${(v / 1e7).toFixed(2)} Cr`;
  if (v >= 1e5) return `₹${(v / 1e5).toFixed(2)} L`;
  return `₹${Math.round(v).toLocaleString("en-IN")}`;
}

/** Validate + normalise user input. Mirrors the original Flask app's rules. */
function parsePayload(p: any): { row?: Row; error?: string } {
  const city = p?.city;
  if (!M.city_localities[city]) return { error: "Please choose a valid city." };
  const locality = p.locality;
  if (!M.city_localities[city].includes(locality))
    return { error: "Please choose a locality that belongs to the selected city." };
  const ptype = p.property_type;
  if (!M.property_types.includes(ptype)) return { error: "Please choose a valid property type." };
  const furnishing = p.furnishing ?? "Unfurnished";
  if (!M.furnishing.includes(furnishing)) return { error: "Invalid furnishing option." };

  const area = Number(p.area_sqft);
  if (p.area_sqft === "" || p.area_sqft == null || !Number.isFinite(area)) return { error: "Area must be a number." };
  const [aLo, aHi] = RANGES.area_sqft;
  if (area < aLo || area > aHi)
    return { error: `Area must be between ${aLo} and ${aHi} sq ft (the range the model was trained on).` };

  const num = (k: string, d = 0) => {
    const v = Number(p[k] ?? d);
    return Number.isFinite(v) ? v : d;
  };
  const yes = (k: string) => (["yes", "true", "1", "on"].includes(String(p[k] ?? "No").toLowerCase()) ? "Yes" : "No");
  const intIn = (k: string, d = 0) => Math.round(clamp(num(k, d), ...RANGES[k]));
  const kmIn = (k: string) => clamp(num(k, M.defaults[k]), ...RANGES[k]);

  const row: Row = {
    city, locality, property_type: ptype, area_sqft: area, furnishing,
    parking_spaces: intIn("parking_spaces"),
    gated_community: yes("gated_community"),
    gym_available: yes("gym_available"),
    swimming_pool: yes("swimming_pool"),
    property_age_years: intIn("property_age_years"),
    distance_to_metro_km: kmIn("distance_to_metro_km"),
    distance_to_school_km: kmIn("distance_to_school_km"),
    distance_to_hospital_km: kmIn("distance_to_hospital_km"),
  };
  if (ptype === "Plot") {
    // Dataset convention: plots have no rooms / floors / lift.
    Object.assign(row, { bedrooms: 0, bathrooms: 0, floor_number: 0, total_floors: 0, lift_available: "No" });
  } else {
    const total = intIn("total_floors", 1);
    Object.assign(row, {
      bedrooms: intIn("bedrooms", 2),
      bathrooms: intIn("bathrooms", 2),
      total_floors: total,
      floor_number: Math.round(clamp(num("floor_number", 0), 0, total)),
      lift_available: yes("lift_available"),
    });
  }
  return { row };
}

const json = (data: unknown, status = 200, headers: Record<string, string> = {}) =>
  Response.json(data, { status, headers });

async function readBody(req: Request) {
  try {
    return await req.json();
  } catch {
    return {};
  }
}

export default async (req: Request) => {
  const { pathname } = new URL(req.url);
  const route = pathname.replace(/\/+$/, "");

  if (route === "/api/health") return json({ status: "ok", model: M.source });

  if (route === "/api/meta" && req.method === "GET") {
    const keep = ["source", "note", "trained_at", "cities", "city_localities", "city_state", "property_types",
      "furnishing", "ranges", "defaults", "metrics", "feature_importance", "city_stats", "trained_rows"];
    return json(Object.fromEntries(keep.map((k) => [k, M[k]])), 200, {
      "Netlify-CDN-Cache-Control": "public, max-age=3600, durable",
    });
  }

  if (route === "/api/predict" || route === "/api/whatif") {
    if (req.method !== "POST") return json({ error: "Use POST with a JSON body." }, 405, { Allow: "POST" });
    const { row, error } = parsePayload(await readBody(req));
    if (error || !row) return json({ error }, 400);

    if (route === "/api/whatif") {
      // Price at 50%–150% of the entered area (for the sensitivity chart).
      const [lo, hi] = RANGES.area_sqft;
      const base = Number(row.area_sqft);
      const areas = [...new Set(Array.from({ length: 11 }, (_, i) => Math.round(clamp(base * (0.5 + i * 0.1), lo, hi))))]
        .sort((a, b) => a - b);
      const points = areas.map((a) => {
        const r = predict(M, { ...row, area_sqft: a });
        return { area: a, price: Math.round(r.mid), low: Math.round(r.low), high: Math.round(r.high) };
      });
      return json({ points });
    }

    const { mid, low, high } = predict(M, row);
    const area = Number(row.area_sqft);
    const cityPsf = M.city_stats.find((c: any) => c.city === row.city)?.median_psf ?? null;
    const psf = mid / area;
    return json({
      price: Math.round(mid), low: Math.round(low), high: Math.round(high),
      price_label: fmtInr(mid), low_label: fmtInr(low), high_label: fmtInr(high),
      price_per_sqft: Math.round(psf),
      city_median_psf: cityPsf,
      vs_city_pct: cityPsf ? Math.round((psf / cityPsf - 1) * 1000) / 10 : null,
      inputs: row,
    });
  }

  return json({ error: "Not found" }, 404);
};

export const config: Config = { path: "/api/*" };
