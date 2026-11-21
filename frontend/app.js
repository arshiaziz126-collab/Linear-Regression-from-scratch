// GlycoCast website: talks to the Python backend (FastAPI) at /api/...
const $ = (id) => document.getElementById(id);
const fmt = (v) => +Number(v).toPrecision(4);
let INFO = null;
const pending = {};                     // one AbortController per kind of request
const timers = {};

// ---- tabs ------------------------------------------------------------------
document.querySelectorAll(".tab").forEach((btn) => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab, .panel").forEach((el) => el.classList.remove("active"));
  btn.classList.add("active");
  $(btn.dataset.tab).classList.add("active");
}));

async function post(kind, url, body) {
  if (pending[kind]) pending[kind].abort();       // cancel an older request still running
  pending[kind] = new AbortController();
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body), signal: pending[kind].signal,
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}
// wait a moment after the last change, so we don't send a request for every pixel
function soon(kind, fn) { clearTimeout(timers[kind]); timers[kind] = setTimeout(fn, 200); }

// ---- Predict ------------------------------------------------------------------
function mode() { return document.querySelector("input[name=mode]:checked").value; }

async function predict() {
  const body = mode() === "patient" ? { patient: Number($("patient").value) } : { values: {} };
  if (mode() === "custom") document.querySelectorAll("#inputs [data-name]").forEach((el) => (body.values[el.dataset.name] = Number(el.value)));
  document.body.classList.add("loading"); $("error").hidden = true;
  try {
    const r = await post("predict", "/api/predict", body);
    $("pred").textContent = Math.round(r.prediction);
    const pct = Math.min(Math.max((r.prediction - 25) / (346 - 25), 0.01), 1);
    $("fill").style.width = pct * 100 + "%";
    $("contrib").src = "data:image/png;base64," + r.chart_png;
    if (r.actual !== null) {
      $("actual").innerHTML = `Actual progression (hidden from the model): <b>${Math.round(r.actual)}</b>`;
      $("values").innerHTML = "<tr><th>Measurement</th><th>Value</th></tr>" +
        Object.entries(r.values).map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v)}</td></tr>`).join("");
    }
  } catch (e) {
    if (e.name !== "AbortError") { $("error").textContent = "Could not reach the model: " + e.message; $("error").hidden = false; }
  } finally { document.body.classList.remove("loading"); }
}

// ---- Playground ---------------------------------------------------------------
function lrOptions() { return $("scaled").checked ? INFO.lr_scaled : INFO.lr_raw; }

async function playground() {
  const opts = lrOptions();
  const lr = opts[Number($("lr").value)];
  $("lr-num").textContent = lr; $("steps-num").textContent = $("steps").value;
  $("play").style.opacity = 0.45;
  try {
    const r = await post("play", "/api/playground", { scaled: $("scaled").checked, steps: Number($("steps").value), lr });
    $("diverged").hidden = !r.diverged;
    $("play").hidden = r.diverged;
    if (r.diverged) { $("play-note").textContent = ""; return; }
    $("play").src = "data:image/png;base64," + r.chart_png;
    $("play-note").textContent = `After ${$("steps").value} steps: w = ${r.w.toFixed(2)}, b = ${r.b.toFixed(2)} ` +
      `(best: w = ${r.best_w.toFixed(2)}, b = ${r.best_b.toFixed(2)}). Gap in error: ${r.gap.toLocaleString(undefined, { maximumFractionDigits: 1 })}`;
  } catch (e) {
    if (e.name !== "AbortError") $("play-note").textContent = "Could not reach the model: " + e.message;
  } finally { $("play").style.opacity = 1; }
}

// ---- build the page from /api/info ------------------------------------------------
async function init() {
  INFO = await (await fetch("/api/info")).json();
  const r = INFO.results;

  $("patient").max = INFO.n_test - 1;
  $("patient").addEventListener("input", () => { $("patient-num").textContent = $("patient").value; soon("predict", predict); });
  $("range-note").textContent = `Dataset range 25 (low) – 346 (high). Average patient ≈ ${Math.round(INFO.average)}. ` +
    `Typical error ± ${Math.round(r.mae)}.`;

  $("inputs").innerHTML = INFO.inputs.map((s) => {
    if (s.name === "sex") {
      return `<div><div class="slider-head"><span>Sex (coded 1 / 2 in dataset)</span></div>
        <select data-name="sex"><option value="1">1</option><option value="2">2</option></select></div>`;
    }
    const step = Math.max(+((s.max - s.min) / 100).toFixed(3), 0.01);
    return `<div><div class="slider-head"><span>${s.label}</span><b data-out="${s.name}">${fmt(s.value)}</b></div>
      <input type="range" data-name="${s.name}" min="${s.min}" max="${s.max}" step="${step}" value="${s.value}"></div>`;
  }).join("");
  document.querySelectorAll("#inputs [data-name]").forEach((el) => el.addEventListener("input", () => {
    const out = document.querySelector(`[data-out="${el.dataset.name}"]`);
    if (out) out.textContent = fmt(el.value);
    soon("predict", predict);
  }));
  document.querySelectorAll("input[name=mode]").forEach((radio) => radio.addEventListener("change", () => {
    $("patient-box").hidden = mode() !== "patient";
    $("custom-box").hidden = mode() !== "custom";
    predict();
  }));

  // playground controls: the learning-rate slider picks from a list that depends on scaling
  const resetLr = () => {
    const opts = lrOptions();
    $("lr").max = opts.length - 1;
    $("lr").value = opts.indexOf($("scaled").checked ? 0.05 : 0.001);
  };
  $("scaled").addEventListener("change", () => { resetLr(); playground(); });
  ["steps", "lr"].forEach((id) => $(id).addEventListener("input", () => {
    $("lr-num").textContent = lrOptions()[Number($("lr").value)];
    $("steps-num").textContent = $("steps").value;
    soon("play", playground);
  }));
  resetLr();

  $("cards").innerHTML = [["R² (test)", r.r2.toFixed(3)], ["5-fold CV R²", r.cv_r2_mean.toFixed(3)], ["RMSE", r.rmse.toFixed(1)],
    ["MAE", r.mae.toFixed(1)], ["Baseline RMSE", r.baseline_rmse.toFixed(1)]]
    .map(([k, v]) => `<div class="card"><div class="label">${k}</div><div class="value">${v}</div></div>`).join("");
  $("sanity").innerHTML = "<tr><th>Check</th><th>Result</th></tr>" + [
    ["scikit-learn LinearRegression R² (same split)", `${r.sklearn_r2.toFixed(3)} (mine: ${r.r2.toFixed(3)})`],
    ["Largest weight difference vs scikit-learn", r.sklearn_coef_diff.toFixed(3)],
    ["Gradient descent vs exact normal equation", r.gd_vs_exact.toFixed(3)],
  ].map(([a, b]) => `<tr><td>${a}</td><td>${b}</td></tr>`).join("");
  $("honest").innerHTML = [
    `Test R² is <b>${r.r2.toFixed(2)}</b> and 5-fold CV R² is <b>${r.cv_r2_mean.toFixed(2)} ± ${r.cv_r2_std.toFixed(2)}</b>. Diabetes progression is noisy, and scores around 0.4–0.5 are normal for this dataset. scikit-learn gets exactly the same number.`,
    `The model is <b>${Math.round((1 - r.rmse / r.baseline_rmse) * 100)}% better (RMSE)</b> than always guessing the average patient.`,
    `Total cholesterol and LDL are about 90% correlated, so their weights partly cancel (multicollinearity).`,
  ].map((t) => `<li>${t}</li>`).join("");

  predict();
  playground();
}

init().catch((e) => { $("error").textContent = "Could not load the app: " + e.message; $("error").hidden = false; });
