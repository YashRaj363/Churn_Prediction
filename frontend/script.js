/* ===================================================================
   ChurnGuard — Frontend Logic
   =================================================================== */

const GAUGE_ARC = 306.31;

// ── Quick-fill profiles ──
const PROFILES = {
  low: {
    gender: "Male", SeniorCitizen: "0", Partner: "Yes", Dependents: "Yes",
    tenure: 68, PhoneService: "Yes", MultipleLines: "Yes",
    InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "Yes",
    DeviceProtection: "Yes", TechSupport: "Yes", StreamingTV: "No",
    StreamingMovies: "No", Contract: "Two year", PaperlessBilling: "No",
    PaymentMethod: "Credit card (automatic)", MonthlyCharges: 56.25, TotalCharges: 3825.0,
  },
  high: {
    gender: "Female", SeniorCitizen: "1", Partner: "No", Dependents: "No",
    tenure: 1, PhoneService: "Yes", MultipleLines: "No",
    InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No",
    DeviceProtection: "No", TechSupport: "No", StreamingTV: "Yes",
    StreamingMovies: "Yes", Contract: "Month-to-month", PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check", MonthlyCharges: 95.0, TotalCharges: 95.0,
  },
};

// ── Toast notifications ──
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.classList.add("leaving");
    toast.addEventListener("animationend", () => toast.remove());
  }, 3500);
}

// ── API helpers ──
async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `API error ${res.status}`);
  }
  return res.json();
}

// ── Health check ──
async function checkHealth() {
  const badge = document.getElementById("health-badge");
  try {
    const data = await api("/health");
    badge.classList.toggle("healthy", data.model_ready);
    badge.classList.toggle("unhealthy", !data.model_ready);
    badge.querySelector(".status-text").textContent = data.model_ready ? "Healthy" : "Model not loaded";
  } catch {
    badge.classList.add("unhealthy");
    badge.querySelector(".status-text").textContent = "Offline";
  }
}

// ── Model info ──
async function loadModelInfo() {
  try {
    const data = await api("/model");
    const badge = document.getElementById("model-badge");
    const version = data.model_version || "unknown";
    const type = (data.model_type || "").replace(/_/g, " ");
    badge.textContent = `${type} · v${version}`;
  } catch {
    /* silent */
  }
}

// ── Build request payload from form ──
function getFormData() {
  const form = document.getElementById("predict-form");
  const fd = new FormData(form);
  const data = {};

  const strings = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
  ];
  strings.forEach((f) => (data[f] = fd.get(f)));

  data.SeniorCitizen = parseInt(fd.get("SeniorCitizen"), 10);
  data.tenure = parseInt(fd.get("tenure"), 10);
  data.MonthlyCharges = parseFloat(fd.get("MonthlyCharges"));
  data.TotalCharges = parseFloat(fd.get("TotalCharges"));
  return data;
}

// ── Color for probability ──
function probColor(p) {
  if (p < 0.33) return "var(--green)";
  if (p < 0.66) return "var(--amber)";
  return "var(--red)";
}

// ── Animate number from 0 to target ──
function animateValue(el, target, suffix = "%", duration = 1200) {
  const start = performance.now();
  const from = 0;
  function tick(now) {
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
    const val = from + (target - from) * eased;
    el.textContent = val.toFixed(target < 10 ? 2 : 1) + suffix;
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

// ── Show prediction result ──
function showResult(result) {
  const placeholder = document.getElementById("result-placeholder");
  const content = document.getElementById("result-content");
  const card = document.getElementById("result-card");

  placeholder.style.display = "none";
  content.classList.remove("hidden");

  const prob = result.churn_probability;
  const pct = prob * 100;
  const risk = result.risk_band;
  const color = probColor(prob);

  // Gauge
  const gaugeFill = document.getElementById("gauge-fill");
  gaugeFill.style.stroke = color;
  // Reset then animate
  gaugeFill.style.transition = "none";
  gaugeFill.style.strokeDashoffset = GAUGE_ARC;
  // Force reflow
  gaugeFill.getBoundingClientRect();
  gaugeFill.style.transition = "stroke-dashoffset 1.4s cubic-bezier(0.4, 0, 0.2, 1), stroke 0.5s ease";
  gaugeFill.style.strokeDashoffset = GAUGE_ARC * (1 - prob);

  // Animate percentage text
  const gaugeText = document.getElementById("gauge-value");
  animateValue(gaugeText, pct, "%", 1300);

  // Verdict
  const verdict = document.getElementById("verdict");
  if (result.churn_prediction) {
    verdict.innerHTML = '⚠️ This customer is <strong style="color:var(--red)">likely to churn</strong>';
  } else {
    verdict.innerHTML = '✅ This customer is <strong style="color:var(--green)">likely to stay</strong>';
  }

  // Risk badge
  const badge = document.getElementById("risk-badge");
  badge.className = `risk-badge ${risk}`;
  badge.textContent = `${risk} risk`;
  // Re-trigger animation
  badge.style.animation = "none";
  badge.getBoundingClientRect();
  badge.style.animation = "";

  // Card glow
  card.className = "card result-card";
  card.classList.add(`risk-${risk}`);

  // Details
  document.getElementById("detail-prob").textContent = (prob * 100).toFixed(2) + "%";
  document.getElementById("detail-threshold").textContent = (result.decision_threshold * 100).toFixed(0) + "%";
  document.getElementById("detail-model").textContent = result.model_version;
}

// ── Predict ──
async function predict() {
  const btn = document.getElementById("predict-btn");
  const originalHTML = btn.innerHTML;
  btn.classList.add("loading");
  btn.innerHTML = '<span class="spinner"></span> Predicting…';

  try {
    const payload = getFormData();
    const result = await api("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    showResult(result);
    showToast("Prediction complete", "success");
  } catch (err) {
    showToast(err.message || "Prediction failed", "error");
  } finally {
    btn.classList.remove("loading");
    btn.innerHTML = originalHTML;
  }
}

// ── Fill profile ──
function fillProfile(key) {
  const profile = PROFILES[key];
  if (!profile) return;
  Object.entries(profile).forEach(([name, value]) => {
    const el = document.getElementById(name);
    if (el) el.value = value;
  });
  showToast(`Loaded ${key}-risk profile`, "info");
}

// ── Metrics ──
async function loadMetrics() {
  try {
    const data = await api("/metrics");
    document.getElementById("m-total").textContent = data.total_requests ?? "—";
    document.getElementById("m-errors").textContent =
      data.error_rate != null ? (data.error_rate * 100).toFixed(1) + "%" : "—";
    document.getElementById("m-latency").textContent =
      data.latency_ms_avg != null ? data.latency_ms_avg.toFixed(1) + " ms" : "—";
    document.getElementById("m-p95").textContent =
      data.latency_p95_ms != null ? data.latency_p95_ms.toFixed(1) + " ms" : "—";
    document.getElementById("m-churn-rate").textContent =
      data.predicted_churn_rate != null ? (data.predicted_churn_rate * 100).toFixed(1) + "%" : "—";
    document.getElementById("m-avg-prob").textContent =
      data.avg_churn_probability != null ? (data.avg_churn_probability * 100).toFixed(1) + "%" : "—";
  } catch (err) {
    showToast("Failed to load metrics", "error");
  }
}

// ── Drift ──
async function loadDrift() {
  const container = document.getElementById("drift-content");
  try {
    const data = await api("/monitoring/report");
    const drift = data.drift || {};

    let html = '<div class="drift-summary">';

    // Status badge
    if (drift.status === "ok") {
      const isDrifting = drift.dataset_drift;
      html += `<span class="drift-status-badge ${isDrifting ? "drifting" : "ok"}">
        ${isDrifting ? "⚠ Drift Detected" : "✓ No Drift"}</span>`;
      html += `<span class="drift-stat"><strong>${drift.n_features_drifting}</strong> / ${drift.n_features_checked} features drifting</span>`;
      html += `<span class="drift-stat">Samples: <strong>${drift.n_samples}</strong></span>`;
      html += "</div>";

      // Bar
      const frac = drift.drift_fraction || 0;
      const barClass = frac >= 0.3 ? "major" : frac >= 0.1 ? "moderate" : "";
      html += `<div class="drift-bar-container">
        <div class="drift-bar-label">Drift fraction: ${(frac * 100).toFixed(1)}% (threshold: 30%)</div>
        <div class="drift-bar"><div class="drift-bar-fill ${barClass}" style="width:${frac * 100}%"></div></div>
      </div>`;

      // Feature chips
      if (drift.features) {
        html += '<div class="drift-features-grid">';
        Object.entries(drift.features).forEach(([name, info]) => {
          const cls = info.drifting ? "drifting" : "ok";
          html += `<div class="drift-feature-chip ${cls}">
            <span class="feat-name">${name}</span>
            <span class="feat-psi">PSI ${info.psi.toFixed(3)}</span>
          </div>`;
        });
        html += "</div>";
      }
    } else if (drift.status === "insufficient_data") {
      html += `<span class="drift-status-badge pending">Insufficient Data</span>`;
      html += `<span class="drift-stat">${drift.n_samples || 0} samples (need ${drift.min_required || 50})</span>`;
      html += "</div>";
    } else {
      html += `<span class="drift-status-badge pending">${drift.status || "Unknown"}</span>`;
      html += "</div>";
    }

    // Alerts
    if (data.alerts && data.alerts.length > 0) {
      html += '<div style="margin-top:16px"><h3 class="section-title">Triggered Alerts</h3>';
      data.alerts.forEach((a) => {
        const color = a.level === "critical" ? "var(--red)" : "var(--amber)";
        html += `<div style="padding:8px 12px;margin-top:6px;border-left:3px solid ${color};background:rgba(255,255,255,0.02);border-radius:0 6px 6px 0;font-size:0.84rem">
          <strong style="color:${color};text-transform:uppercase">${a.level}</strong> · ${a.message}
        </div>`;
      });
      html += "</div>";
    }

    container.innerHTML = html;
  } catch (err) {
    showToast("Failed to load drift report", "error");
  }
}

// ── Init ──
document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadModelInfo();
  loadMetrics();

  // Form submit
  document.getElementById("predict-form").addEventListener("submit", (e) => {
    e.preventDefault();
    predict();
  });

  // Clear resets result panel
  document.getElementById("clear-btn").addEventListener("click", () => {
    const placeholder = document.getElementById("result-placeholder");
    const content = document.getElementById("result-content");
    const card = document.getElementById("result-card");
    placeholder.style.display = "";
    content.classList.add("hidden");
    card.className = "card result-card";
  });
});
