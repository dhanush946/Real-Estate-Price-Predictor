const $ = (id) => document.getElementById(id);
const form = $("form");
const fmt = (v) => (v >= 1e7 ? `₹${(v / 1e7).toFixed(2)} Cr` : v >= 1e5 ? `₹${(v / 1e5).toFixed(2)} L` : `₹${Math.round(v).toLocaleString("en-IN")}`);
let META;

function chips(el, name, values, checked) {
  el.innerHTML = values.map((v) => `<label class="chip"><input type="radio" name="${name}" value="${v}" ${v === checked ? "checked" : ""}/><span>${v}</span></label>`).join("");
}

function fillLocalities() {
  const city = $("city").value;
  $("locality").innerHTML = META.city_localities[city].map((l) => `<option>${l}</option>`).join("");
}

function payload() {
  const fd = new FormData(form);
  const p = Object.fromEntries(fd.entries());
  for (const k of ["gated_community", "lift_available", "gym_available", "swimming_pool"]) p[k] = fd.has(k) ? "Yes" : "No";
  return p;
}

async function post(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || "Something went wrong.");
  return data;
}

function renderWhatif(points, area) {
  const W = 560, H = 220, P = 40;
  const xs = points.map((p) => p.area), ys = points.flatMap((p) => [p.low, p.high]);
  const x = (v) => P + ((v - Math.min(...xs)) / (Math.max(...xs) - Math.min(...xs) || 1)) * (W - 2 * P);
  const y = (v) => H - P + 10 - ((v - Math.min(...ys)) / (Math.max(...ys) - Math.min(...ys) || 1)) * (H - 2 * P);
  const line = points.map((p, i) => `${i ? "L" : "M"}${x(p.area)},${y(p.price)}`).join("");
  const band = points.map((p, i) => `${i ? "L" : "M"}${x(p.area)},${y(p.high)}`).join("") + [...points].reverse().map((p) => `L${x(p.area)},${y(p.low)}`).join("") + "Z";
  $("whatif").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Price by area">
    <path d="${band}" class="band"/><path d="${line}" class="line"/>
    <line x1="${x(area)}" x2="${x(area)}" y1="${P - 20}" y2="${H - P + 10}" class="marker"/>
    ${points.map((p) => `<circle cx="${x(p.area)}" cy="${y(p.price)}" r="3"><title>${p.area} sq ft: ${fmt(p.price)}</title></circle>`).join("")}
    <text x="${P}" y="${H - 6}">${Math.min(...xs)} sq ft</text><text x="${W - P}" y="${H - 6}" text-anchor="end">${Math.max(...xs)} sq ft</text>
  </svg>`;
}

function renderCities(city, psf) {
  const stats = [...META.city_stats].sort((a, b) => b.median_psf - a.median_psf);
  const max = Math.max(psf, ...stats.map((s) => s.median_psf));
  $("cityBars").innerHTML = stats.map((s) => `<div class="bar ${s.city === city ? "on" : ""}">
    <span>${s.city}</span><div><i style="width:${(s.median_psf / max) * 100}%"></i>${s.city === city ? `<b style="left:${(psf / max) * 100}%"></b>` : ""}</div>
    <em>₹${s.median_psf.toLocaleString("en-IN")}</em></div>`).join("");
}

async function estimate(e) {
  e?.preventDefault();
  $("formError").hidden = true;
  $("submitBtn").disabled = true;
  try {
    const p = payload();
    const [res, wi] = await Promise.all([post("/api/predict", p), post("/api/whatif", p)]);
    $("emptyState").hidden = true;
    for (const id of ["estimateCard", "whatifCard", "cityCard"]) $(id).hidden = false;
    $("estimateFor").textContent = `${res.inputs.property_type} · ${res.inputs.locality}, ${res.inputs.city}`;
    $("price").textContent = res.price_label;
    $("low").textContent = res.low_label;
    $("high").textContent = res.high_label;
    $("rangeMark").style.left = `${((res.price - res.low) / (res.high - res.low || 1)) * 100}%`;
    $("psf").textContent = `₹${res.price_per_sqft.toLocaleString("en-IN")}`;
    $("cityPsf").textContent = res.city_median_psf ? `₹${res.city_median_psf.toLocaleString("en-IN")}` : "—";
    $("vsCity").textContent = res.vs_city_pct == null ? "—" : `${res.vs_city_pct > 0 ? "+" : ""}${res.vs_city_pct}%`;
    renderWhatif(wi.points, Number(res.inputs.area_sqft));
    renderCities(res.inputs.city, res.price_per_sqft);
  } catch (err) {
    $("formError").textContent = err.message;
    $("formError").hidden = false;
  } finally {
    $("submitBtn").disabled = false;
  }
}

function syncPlot() {
  const plot = form.property_type.value === "Plot";
  $("roomFields").hidden = plot;
  $("liftToggle").hidden = plot;
}

async function init() {
  try {
    META = await (await fetch("/api/meta")).json();
  } catch {
    $("modelPillText").textContent = "Model unavailable";
    return;
  }
  $("modelPillText").textContent = META.source === "dataset" ? `Trained on ${META.trained_rows.toLocaleString("en-IN")} listings` : "Baseline model";
  $("modelNote").textContent = META.note;
  $("cityCount").textContent = META.cities.length;
  $("city").innerHTML = META.cities.map((c) => `<option>${c}</option>`).join("");
  fillLocalities();
  chips($("propertyTypes"), "property_type", META.property_types, "Apartment");
  chips($("furnishing"), "furnishing", META.furnishing, "Semi-Furnished");

  const [lo, hi] = META.ranges.area_sqft;
  const area = $("area"), range = $("areaRange");
  Object.assign(range, { min: lo, max: Math.min(hi, 6000), step: 10, value: 1200 });
  Object.assign(area, { min: lo, max: hi, value: 1200 });
  $("areaHint").textContent = `${lo}–${hi} sq ft`;
  range.oninput = () => (area.value = range.value);
  area.oninput = () => (range.value = area.value);

  for (const k of ["distance_to_metro_km", "distance_to_school_km", "distance_to_hospital_km"]) {
    const input = form[k], out = document.querySelector(`output[data-for="${k}"]`);
    Object.assign(input, { min: META.ranges[k][0], max: META.ranges[k][1], value: META.defaults[k] });
    const show = () => (out.textContent = `${Number(input.value).toFixed(1)} km`);
    input.oninput = show;
    show();
  }

  if (META.metrics) {
    const m = META.metrics;
    $("metrics").innerHTML = `<h3>Hold-out accuracy</h3><p>R² ${m.r2.toFixed(3)} · median error ${m.median_ape_pct.toFixed(1)}% · range covers ${m.interval_coverage_pct.toFixed(0)}% of test listings</p>`;
  }
  $("importance").innerHTML = META.feature_importance.slice(0, 6).map((f) => `<li><span>${f.feature.replaceAll("_", " ")}</span><i style="width:${f.importance * 200}px"></i></li>`).join("");

  $("city").onchange = fillLocalities;
  $("propertyTypes").onchange = syncPlot;
  form.onsubmit = estimate;
  syncPlot();
}

init();
