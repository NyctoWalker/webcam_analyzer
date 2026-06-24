/* Webcam Analyzer - frontend logic
 *
 * Two concerns:
 *   1. Live tab - poll /api/status every 500ms, drive Start/Stop buttons
 *   2. Analytics tab - fetch /api/stats, render summary cards + 4 charts
 *
 * Charts use Chart.js (loaded via <script> in index.html).
 */

(function () {
  "use strict";

  // DOM helpers
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // Tab switching
  $$(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      $$(".tab").forEach((b) => b.classList.toggle("active", b === btn));
      $$(".tab-panel").forEach((p) =>
        p.classList.toggle("active", p.id === `tab-${target}`),
      );
      if (target === "analytics") refreshAnalytics();
    });
  });

  // Live tab
  const btnStart = $("#btnStart");
  const btnStop  = $("#btnStop");
  const statusPill = $("#statusPill");
  const statusText = $("#statusText");
  const statusDot  = $("#statusDot");

  async function postJSON(url, body) {
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    return r.json();
  }

  async function fetchJSON(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url} -> ${r.status}`);
    return r.json();
  }

  btnStart.addEventListener("click", async () => {
    btnStart.disabled = true;
    try {
      const res = await postJSON("/api/start");
      if (!res.started && res.running) {
        setStatus("Already running", "running");
      }
    } catch (e) {
      setStatus(`Error: ${e.message}`, "error");
    }
    refreshStatus();
  });

  btnStop.addEventListener("click", async () => {
    btnStop.disabled = true;
    try {
      await postJSON("/api/stop");
    } catch (e) {
      setStatus(`Error: ${e.message}`, "error");
    }
    refreshStatus();
  });

  // Toggles (HUD overlay + video display)
  const toggleOverlay = $("#toggleOverlay");
  const toggleDisplay = $("#toggleDisplay");

  async function loadSettings() {
    try {
      const s = await fetchJSON("/api/settings");
      // Only set checked state if the checkbox hasn't been touched since
      // page load - avoids clobbering user input during a race.
      toggleOverlay.checked = !!s.overlay;
      toggleDisplay.checked = !!s.display;
    } catch (e) {
      // server may not be up yet - leave defaults
    }
  }

  async function pushSetting(key, value) {
    try {
      await postJSON("/api/settings", { [key]: value });
    } catch (e) {
      console.error(`Failed to update ${key}:`, e);
    }
  }

  toggleOverlay.addEventListener("change", (e) => {
    pushSetting("overlay", e.target.checked);
  });
  toggleDisplay.addEventListener("change", (e) => {
    pushSetting("display", e.target.checked);
  });

  loadSettings();

  function setStatus(text, kind) {
    statusText.textContent = text;
    statusPill.classList.remove("running", "error");
    if (kind === "running") statusPill.classList.add("running");
    else if (kind === "error") statusPill.classList.add("error");
  }

  async function refreshStatus() {
    try {
      const s = await fetchJSON("/api/status");
      if (s.last_error) {
        setStatus(`Error: ${s.last_error}`, "error");
      } else if (s.running) {
        setStatus("Running", "running");
      } else {
        setStatus(s.db_ready ? "Idle" : "DB down", s.db_ready ? "" : "error");
      }
      btnStart.disabled = s.running;
      btnStop.disabled  = !s.running;

      $("#liveBlinks").textContent         = s.blink_count ?? 0;
      $("#liveSmiles").textContent         = s.smile_count ?? 0;
      $("#liveSmileTime").textContent      = `${(s.smile_time_s ?? 0).toFixed(2)} s`;
      $("#liveBatchRemaining").textContent = `${Math.ceil(s.batch_remaining_s ?? 0)} s`;
      $("#liveFrames").textContent         = s.frames_processed ?? 0;

      // Reflect server-side toggle state, but only update the checkbox if
      // it isn't currently focused (i.e. the user isn't mid-click).
      // This keeps multiple browser tabs in sync.
      if (typeof s.overlay_enabled === "boolean" && document.activeElement !== toggleOverlay) {
        toggleOverlay.checked = s.overlay_enabled;
      }
      if (typeof s.display_enabled === "boolean" && document.activeElement !== toggleDisplay) {
        toggleDisplay.checked = s.display_enabled;
      }
    } catch (e) {
      setStatus(`Offline`, "error");
    }
  }

  // Poll status every 500 ms while Live tab is active
  setInterval(() => {
    if ($("#tab-live").classList.contains("active")) refreshStatus();
  }, 500);
  refreshStatus();

  // Analytics tab
  let currentRange = "24h";
  let charts = {};

  $$("#rangeButtons button").forEach((b) => {
    b.addEventListener("click", () => {
      $$("#rangeButtons button").forEach((x) => x.classList.toggle("active", x === b));
      currentRange = b.dataset.range;
      refreshAnalytics();
    });
  });

  $("#btnRefresh").addEventListener("click", refreshAnalytics);

  let autoRefreshTimer = null;
  $("#autoRefresh").addEventListener("change", (e) => {
    if (e.target.checked) {
      autoRefreshTimer = setInterval(refreshAnalytics, 30000);
      refreshAnalytics();
    } else if (autoRefreshTimer) {
      clearInterval(autoRefreshTimer);
      autoRefreshTimer = null;
    }
  });

  $("#bucketSelect").addEventListener("change", refreshAnalytics);

  async function refreshAnalytics() {
    const bucket = $("#bucketSelect").value;
    const url = `/api/stats?range=${encodeURIComponent(currentRange)}&bucket=${encodeURIComponent(bucket)}`;
    let data;
    try {
      data = await fetchJSON(url);
    } catch (e) {
      $("#analyticsEmpty").textContent = `Failed to load: ${e.message}`;
      $("#analyticsEmpty").hidden = false;
      return;
    }

    const ts = data.timeseries || [];
    const summary = data.summary || {};

    // Empty state
    const hasData = ts.length > 0 || summary.batch_count > 0;
    $("#analyticsEmpty").hidden = hasData;

    // Summary cards
    renderSummary(summary, data);

    // Charts
    const labels = ts.map((r) => formatBucketLabel(r.bucket, data.bucket));
    renderChart("chartBlinks", "blinks", labels, ts.map((r) => r.blinks), {
      color: "#4ea1ff",
      meta: "#metaBlinks",
      metaText: `total ${summary.blinks ?? 0} blinks`,
    });
    renderChart("chartSmiles", "smiles", labels, ts.map((r) => r.smiles), {
      color: "#44d58a",
      meta: "#metaSmiles",
      metaText: `total ${summary.smiles ?? 0} smiles`,
    });
    renderChart("chartSmileTime", "smile_time_s", labels, ts.map((r) => r.smile_time_s), {
      color: "#ffae3b",
      meta: "#metaSmileTime",
      metaText: `total ${(summary.smile_time_s ?? 0).toFixed(1)} s`,
    });
    renderLoudnessChart(labels, ts, summary, data);
  }

  function renderSummary(s, data) {
    const cards = [
      { label: "Batches",        value: s.batch_count ?? 0, sub: `in ${data.bucket_label}` },
      { label: "Captured time",  value: formatDuration(s.duration_s ?? 0), sub: "sum of batch durations" },
      { label: "Blinks",         value: s.blinks ?? 0, sub: `${ratePerHour(s.blinks, s.duration_s)} /h` },
      { label: "Smiles",         value: s.smiles ?? 0, sub: `${ratePerHour(s.smiles, s.duration_s)} /h` },
      { label: "Smile time",     value: formatDuration(s.smile_time_s ?? 0), sub: `${pct(s.smile_time_s, s.duration_s)} of capture` },
      { label: "Avg loudness",   value: s.avg_loudness != null ? `${s.avg_loudness.toFixed(1)} dB` : "—", sub: "RMS averaged" },
      { label: "Peak loudness",  value: s.max_loudness != null ? `${s.max_loudness.toFixed(1)} dB` : "—", sub: "max dBFS" },
    ];
    $("#summaryGrid").innerHTML = cards.map((c) => `
      <div class="summary-card">
        <div class="label">${c.label}</div>
        <div class="value">${c.value}</div>
        <div class="sub">${c.sub}</div>
      </div>
    `).join("");
  }

  function renderChart(canvasId, label, labels, values, opts) {
    const ctx = document.getElementById(canvasId).getContext("2d");
    if (charts[canvasId]) charts[canvasId].destroy();
    charts[canvasId] = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label,
          data: values,
          borderColor: opts.color,
          backgroundColor: opts.color + "22",
          tension: 0.25,
          fill: true,
          pointRadius: labels.length > 60 ? 0 : 2,
          pointHoverRadius: 4,
          borderWidth: 2,
        }],
      },
      options: chartOpts(opts.color),
    });
    if (opts.meta) $(opts.meta).textContent = opts.metaText || "";
  }

  function renderLoudnessChart(labels, ts, summary, data) {
    const ctx = document.getElementById("chartLoud").getContext("2d");
    if (charts.chartLoud) charts.chartLoud.destroy();
    charts.chartLoud = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "avg dBFS",
            data: ts.map((r) => r.avg_loudness),
            borderColor: "#9b6bff",
            backgroundColor: "#9b6bff22",
            tension: 0.25,
            fill: true,
            pointRadius: labels.length > 60 ? 0 : 2,
            borderWidth: 2,
          },
          {
            label: "max dBFS",
            data: ts.map((r) => r.max_loudness),
            borderColor: "#ff6b6b",
            backgroundColor: "transparent",
            tension: 0.25,
            fill: false,
            pointRadius: labels.length > 60 ? 0 : 2,
            borderWidth: 1.5,
            borderDash: [4, 4],
          },
        ],
      },
      options: chartOpts("#9b6bff"),
    });
    $("#metaLoud").textContent = `peak ${summary.max_loudness?.toFixed(1) ?? "—"} dB`;
  }

  function chartOpts(accent) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: { color: "#8b93a4", font: { size: 11 }, boxWidth: 12 },
        },
        tooltip: {
          backgroundColor: "#0f1115",
          borderColor: "#2a2f3a",
          borderWidth: 1,
          titleColor: "#e7e9ee",
          bodyColor: "#e7e9ee",
        },
      },
      scales: {
        x: {
          ticks: { color: "#8b93a4", maxRotation: 0, autoSkip: true, maxTicksLimit: 8, font: { size: 10 } },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
        y: {
          beginAtZero: true,
          ticks: { color: "#8b93a4", font: { size: 10 } },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
      },
    };
  }

  // Formatting helpers
  function formatBucketLabel(iso, bucketInterval) {
    if (!iso) return "";
    const d = new Date(iso);
    if (bucketInterval.includes("minute")) {
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    if (bucketInterval.includes("hour")) {
      return d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    }
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  }

  function formatDuration(sec) {
    if (sec == null || isNaN(sec)) return "—";
    if (sec < 60) return `${sec.toFixed(0)} s`;
    if (sec < 3600) return `${(sec / 60).toFixed(1)} min`;
    return `${(sec / 3600).toFixed(2)} h`;
  }

  function ratePerHour(count, sec) {
    if (!sec) return "0";
    return ((count / sec) * 3600).toFixed(1);
  }

  function pct(part, whole) {
    if (!whole) return "0%";
    return `${((part / whole) * 100).toFixed(1)}%`;
  }
})();
