/* ── Quiniela Mundial 2026 — main.js ── */

const FLAGS = {
  "Algeria":"🇩🇿","Argentina":"🇦🇷","Australia":"🇦🇺","Austria":"🇦🇹",
  "Belgium":"🇧🇪","Bosnia & Herzegovina":"🇧🇦","Brazil":"🇧🇷","Canada":"🇨🇦",
  "Cape Verde":"🇨🇻","Colombia":"🇨🇴","Croatia":"🇭🇷","Curaçao":"🇨🇼",
  "Czech Republic":"🇨🇿","DR Congo":"🇨🇩","Ecuador":"🇪🇨","Egypt":"🇪🇬",
  "England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","France":"🇫🇷","Germany":"🇩🇪","Ghana":"🇬🇭",
  "Haiti":"🇭🇹","Iran":"🇮🇷","Iraq":"🇮🇶","Ivory Coast":"🇨🇮",
  "Japan":"🇯🇵","Jordan":"🇯🇴","Mexico":"🇲🇽","Morocco":"🇲🇦",
  "Netherlands":"🇳🇱","New Zealand":"🇳🇿","Norway":"🇳🇴","Panama":"🇵🇦",
  "Paraguay":"🇵🇾","Portugal":"🇵🇹","Qatar":"🇶🇦","Saudi Arabia":"🇸🇦",
  "Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Senegal":"🇸🇳","South Africa":"🇿🇦","South Korea":"🇰🇷",
  "Spain":"🇪🇸","Sweden":"🇸🇪","Switzerland":"🇨🇭","Tunisia":"🇹🇳",
  "Turkey":"🇹🇷","USA":"🇺🇸","Uruguay":"🇺🇾","Uzbekistan":"🇺🇿",
};

const DAYS_ES = ["domingo","lunes","martes","miércoles","jueves","viernes","sábado"];
const MONTHS_ES = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto",
                   "septiembre","octubre","noviembre","diciembre"];

function flag(name) { return FLAGS[name] || "🏳️"; }

function formatDate(dateStr) {
  // dateStr = "2026-06-11"
  const [y, m, d] = dateStr.split("-").map(Number);
  const dt = new Date(y, m - 1, d);
  return `${DAYS_ES[dt.getDay()]} ${d} de ${MONTHS_ES[m-1]}`;
}

function formatTime(timeUtc) {
  // Show in local browser time
  const dt = new Date(timeUtc);
  return dt.toLocaleTimeString("es", { hour: "2-digit", minute: "2-digit" });
}

function matchStatus(timeUtc) {
  const now = Date.now();
  const start = new Date(timeUtc).getTime();
  const diff = now - start; // ms since kick-off
  if (diff > 0 && diff < 130 * 60 * 1000) return "live";   // < 130 min after start
  if (diff >= 130 * 60 * 1000) return "past";
  return "upcoming";
}

function eptsBadgeClass(ep) {
  if (ep >= 2.0) return "";
  if (ep >= 1.2) return "medium";
  return "low";
}

function probBars(probs) {
  if (!probs) return "";
  const hw = Math.round(probs.home_win * 100);
  const dr = Math.round(probs.draw * 100);
  const aw = 100 - hw - dr;
  return `
    <div class="prob-bar">
      <div class="seg-home" style="width:${hw}%"></div>
      <div class="seg-draw" style="width:${dr}%"></div>
      <div class="seg-away" style="width:${aw}%"></div>
    </div>
    <div class="prob-labels">
      <span title="Victoria local">${hw}%</span>
      <span title="Empate">${dr}%</span>
      <span title="Victoria visitante">${aw}%</span>
    </div>`;
}

function renderCard(m, filter) {
  const status = matchStatus(m.time_utc);
  if (filter === "upcoming" && status !== "upcoming") return "";
  if (filter === "group" && m.phase !== "group") return "";
  if (filter === "knockout" && m.phase !== "knockout") return "";

  const isTbd = m.tbd || !m.prediction;
  const pred = m.prediction || {};

  const statusPill = status === "live"
    ? `<span class="status-pill live">En curso</span>`
    : status === "past"
    ? `<span class="status-pill past">Finalizado</span>`
    : "";

  const scoreHtml = isTbd
    ? `<div class="score-main"><span>?</span><span class="score-sep">-</span><span>?</span></div>
       <span class="status-pill tbd">Por definir</span>`
    : `<div class="score-main">
         <span>${pred.home}</span>
         <span class="score-sep">-</span>
         <span>${pred.away}</span>
       </div>
       <span class="score-label">predicción</span>
       ${statusPill || `<span class="score-epts ${eptsBadgeClass(pred.expected_pts)}">~${pred.expected_pts?.toFixed(2)} pts</span>`}`;

  const groupLabel = m.group
    ? `<span class="match-group">${m.group}</span>`
    : `<span class="match-group">${m.round}</span>`;

  const phaseBadge = m.phase === "knockout"
    ? `<span class="match-phase-badge knockout">Eliminatoria</span>`
    : `<span class="match-phase-badge">Grupos</span>`;

  const srcBadge = !isTbd
    ? `<span class="src-badge ${m.source === 'historical_fallback' ? 'fallback' : ''}">${m.source === 'kalshi' ? 'Kalshi' : 'Histórico'}</span>`
    : "";

  const homeIsTbd = !FLAGS[m.home];
  const awayIsTbd = !FLAGS[m.away];

  return `
  <div class="match-card ${status === 'past' ? 'past' : ''} ${status === 'live' ? 'live' : ''}">
    <div class="match-meta">
      <span class="match-time">${formatTime(m.time_utc)}</span>
      ${groupLabel}
      ${phaseBadge}
    </div>
    <div class="team-home">
      <span class="team-name ${homeIsTbd ? 'tbd' : ''}">${m.home}</span>
      <span class="team-flag">${flag(m.home)}</span>
    </div>
    <div class="score-block">
      ${scoreHtml}
    </div>
    <div class="team-away">
      <span class="team-flag">${flag(m.away)}</span>
      <span class="team-name ${awayIsTbd ? 'tbd' : ''}">${m.away}</span>
    </div>
    <div class="prob-col">
      ${probBars(m.probabilities)}
      ${srcBadge}
    </div>
  </div>`;
}

function renderAll(data, filter) {
  const container = document.getElementById("matches");
  if (!container) return;

  // Group by date
  const byDate = {};
  for (const m of data.matches) {
    if (!byDate[m.date]) byDate[m.date] = [];
    byDate[m.date].push(m);
  }

  let html = "";
  for (const [date, matches] of Object.entries(byDate).sort()) {
    const cards = matches.map(m => renderCard(m, filter)).join("");
    if (!cards.trim()) continue;
    html += `
    <div class="date-section">
      <div class="date-header"><h2>${formatDate(date)}</h2></div>
      ${cards}
    </div>`;
  }
  container.innerHTML = html || "<p style='color:var(--text-muted);text-align:center;padding:40px'>No hay partidos para mostrar.</p>";
}

async function init() {
  let data;
  try {
    const res = await fetch("data/predictions.json");
    data = await res.json();
  } catch (e) {
    document.getElementById("matches").innerHTML = "<p style='color:red;padding:24px'>Error cargando predicciones.</p>";
    return;
  }

  // Updated badge
  const updEl = document.getElementById("last-updated");
  if (updEl && data.generated_at) {
    const d = new Date(data.generated_at);
    updEl.textContent = `Actualizado: ${d.toLocaleString("es")}`;
  }

  let activeFilter = "all";

  function applyFilter(f) {
    activeFilter = f;
    document.querySelectorAll(".chip").forEach(c => {
      c.classList.toggle("active", c.dataset.filter === f);
    });
    renderAll(data, f);
  }

  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => applyFilter(chip.dataset.filter));
  });

  applyFilter("all");
}

document.addEventListener("DOMContentLoaded", init);
