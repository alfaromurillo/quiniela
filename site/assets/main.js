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

const NAMES_ES = {
  "Algeria":"Argelia","Argentina":"Argentina","Australia":"Australia",
  "Austria":"Austria","Belgium":"Bélgica",
  "Bosnia & Herzegovina":"Bosnia y Herzegovina",
  "Brazil":"Brasil","Canada":"Canadá","Cape Verde":"Cabo Verde",
  "Colombia":"Colombia","Croatia":"Croacia","Curaçao":"Curazao",
  "Czech Republic":"Rep. Checa","DR Congo":"Congo RD",
  "Ecuador":"Ecuador","Egypt":"Egipto","England":"Inglaterra",
  "France":"Francia","Germany":"Alemania","Ghana":"Ghana",
  "Haiti":"Haití","Iran":"Irán","Iraq":"Irak",
  "Ivory Coast":"Costa de Marfil","Japan":"Japón","Jordan":"Jordania",
  "Mexico":"México","Morocco":"Marruecos","Netherlands":"Países Bajos",
  "New Zealand":"Nueva Zelanda","Norway":"Noruega","Panama":"Panamá",
  "Paraguay":"Paraguay","Portugal":"Portugal","Qatar":"Catar",
  "Saudi Arabia":"Arabia Saudita","Scotland":"Escocia","Senegal":"Senegal",
  "South Africa":"Sudáfrica","South Korea":"Corea del Sur","Spain":"España",
  "Sweden":"Suecia","Switzerland":"Suiza","Tunisia":"Túnez",
  "Turkey":"Turquía","USA":"EE.UU.","Uruguay":"Uruguay",
  "Uzbekistan":"Uzbekistán",
};

const DAYS_ES   = ["domingo","lunes","martes","miércoles","jueves","viernes","sábado"];
const MONTHS_ES = ["enero","febrero","marzo","abril","mayo","junio","julio","agosto",
                   "septiembre","octubre","noviembre","diciembre"];

// Result windows: minutes after kickoff before a match can be considered over
const RESULT_DELAY_GROUP    = 120;
const RESULT_DELAY_KNOCKOUT = 210;

function flag(name) { return FLAGS[name] || "🏳️"; }
function tname(name) { return NAMES_ES[name] || name; }

function formatDate(dateStr) {
  const [y, m, d] = dateStr.split("-").map(Number);
  const dt = new Date(y, m - 1, d);
  return `${DAYS_ES[dt.getDay()]} ${d} de ${MONTHS_ES[m-1]}`;
}

function formatTime(timeUtc) {
  return new Date(timeUtc).toLocaleTimeString("es", { hour: "2-digit", minute: "2-digit" });
}

function matchStatus(timeUtc, phase) {
  const diff = Date.now() - new Date(timeUtc).getTime();
  if (diff < 0) return "upcoming";
  const window = (phase === "knockout" ? RESULT_DELAY_KNOCKOUT : RESULT_DELAY_GROUP) * 60 * 1000;
  return diff < window ? "live" : "past-pending";
}

function quinielaPoints(predH, predA, actH, actA, phase) {
  if (phase === "group") {
    if (predH === actH && predA === actA) return 5;
    const pw = Math.sign(predH - predA), aw = Math.sign(actH - actA);
    if (pw === aw && (predH === actH || predA === actA)) return 3;
    if (pw === aw) return 2;
    if (predH === actH || predA === actA) return 1;
    return 0;
  } else {
    if (predH === actH && predA === actA) return 3;
    if (Math.sign(predH - predA) === Math.sign(actH - actA)) return 1;
    return 0;
  }
}

function eptsBadgeClass(ep) {
  if (!ep) return "low";
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

function renderCard(m, filter, locked, result) {
  const status = matchStatus(m.time_utc, m.phase);
  const hasResult = result && result.status === "final";

  if (filter === "upcoming"  && status !== "upcoming" && status !== "live") return "";
  if (filter === "finished"  && !hasResult)             return "";
  if (filter === "group"     && m.phase !== "group")    return "";
  if (filter === "knockout"  && m.phase !== "knockout") return "";

  const isTbd = m.tbd || !m.prediction;
  const pred  = m.prediction || {};

  // Choose card class
  let cardClass = "match-card";
  if      (hasResult)             cardClass += " completed";
  else if (status === "live")     cardClass += " live";
  else if (status === "past-pending") cardClass += " past-pending";

  // Build score block
  let scoreHtml;
  if (isTbd) {
    scoreHtml = `
      <div class="score-main"><span>?</span><span class="score-sep">-</span><span>?</span></div>
      <span class="status-pill tbd">Por definir</span>`;

  } else if (hasResult) {
    // Completed match: show result + locked prediction + points
    const lp  = locked;
    const pts = lp ? quinielaPoints(lp.home, lp.away, result.home_score, result.away_score, m.phase) : null;
    const ptsHtml = pts !== null ? `<span class="pts-earned p${pts}">+${pts} pts</span>` : "";
    const predHtml = lp
      ? `<div class="score-pred-row">Pred: <strong>${lp.home}-${lp.away}</strong></div>`
      : "";
    scoreHtml = `
      <div class="score-main">${result.home_score}<span class="score-sep">-</span>${result.away_score}</div>
      <div class="score-result-row">resultado</div>
      ${predHtml}
      <div class="score-inline-row">
        <span class="score-status-label">Finalizado</span>
        ${ptsHtml}
      </div>`;

  } else {
    // Upcoming or live: show prediction (locked if past kickoff, current if not)
    const displayPred = (status !== "upcoming" && locked) ? locked : pred;
    const ph = displayPred.home !== undefined ? displayPred.home : pred.home;
    const pa = displayPred.away !== undefined ? displayPred.away : pred.away;
    const ep = displayPred.expected_pts ?? pred.expected_pts;
    const statusPill = status === "live"
      ? `<span class="status-pill live">En curso</span>`
      : status === "past-pending"
      ? `<span class="status-pill past">Finalizado</span>`
      : "";
    const epRow = !statusPill && ep != null
      ? `<div class="score-inline-row">
          <span class="score-status-label">Esperado</span>
          <span class="score-epts ${eptsBadgeClass(ep)}">~${ep.toFixed(2)} pts</span>
        </div>`
      : statusPill;
    scoreHtml = `
      <div class="score-main">${ph}<span class="score-sep">-</span>${pa}</div>
      <span class="score-label">predicción</span>
      ${epRow}`;
  }

  const groupLabel  = m.group
    ? `<span class="match-group">${m.group}</span>`
    : `<span class="match-group">${m.round}</span>`;
  const phaseBadge  = m.phase === "knockout"
    ? `<span class="match-phase-badge knockout">Eliminatoria</span>`
    : `<span class="match-phase-badge">Grupos</span>`;
  const srcBadge = !isTbd
    ? `<span class="src-badge ${m.source === 'historical_fallback' ? 'fallback' : ''}">${m.source === 'kalshi' ? 'Kalshi' : 'Histórico'}</span>`
    : "";

  return `
  <div class="${cardClass}">
    <div class="match-meta">
      <span class="match-time">${formatTime(m.time_utc)}</span>
      ${groupLabel}
      ${phaseBadge}
    </div>
    <div class="team-home">
      <span class="team-name ${!FLAGS[m.home] ? 'tbd' : ''}">${tname(m.home)}</span>
      <span class="team-flag">${flag(m.home)}</span>
    </div>
    <div class="score-block">
      ${scoreHtml}
    </div>
    <div class="team-away">
      <span class="team-flag">${flag(m.away)}</span>
      <span class="team-name ${!FLAGS[m.away] ? 'tbd' : ''}">${tname(m.away)}</span>
    </div>
    <div class="prob-col">
      ${probBars(m.probabilities)}
      ${srcBadge}
    </div>
  </div>`;
}

function renderAll(matches, filter, lockedMap, resultsMap) {
  const container = document.getElementById("matches");
  if (!container) return;

  const sorted = [...matches].sort((a, b) => a.time_utc.localeCompare(b.time_utc));

  const byDate = {};
  for (const m of sorted) {
    if (!byDate[m.date]) byDate[m.date] = [];
    byDate[m.date].push(m);
  }

  let html = "";
  for (const [date, ms] of Object.entries(byDate).sort()) {
    const cards = ms.map(m =>
      renderCard(m, filter, lockedMap[String(m.id)], resultsMap[String(m.id)])
    ).join("");
    if (!cards.trim()) continue;
    html += `
    <div class="date-section">
      <div class="date-header"><h2>${formatDate(date)}</h2></div>
      ${cards}
    </div>`;
  }
  container.innerHTML = html ||
    "<p style='color:var(--text-muted);text-align:center;padding:40px'>No hay partidos para mostrar.</p>";
}

async function init() {
  const [predRes, lockedRes, resultsRes] = await Promise.allSettled([
    fetch("data/predictions.json").then(r => r.json()),
    fetch("data/locked_predictions.json").then(r => r.json()).catch(() => ({ matches: {} })),
    fetch("data/results.json").then(r => r.json()).catch(() => ({ matches: {} })),
  ]);

  if (predRes.status !== "fulfilled") {
    document.getElementById("matches").innerHTML =
      "<p style='color:red;padding:24px'>Error cargando predicciones.</p>";
    return;
  }

  const data      = predRes.value;
  const lockedMap = (lockedRes.status === "fulfilled" ? lockedRes.value.matches : {}) || {};
  const resultsMap= (resultsRes.status === "fulfilled" ? resultsRes.value.matches : {}) || {};

  // Update timestamp
  const updEl = document.getElementById("last-updated");
  if (updEl && data.generated_at) {
    updEl.textContent = `Actualizado: ${new Date(data.generated_at).toLocaleString("es")}`;
  }

  // Compute points
  let totalPts = 0, gamesPlayed = 0;
  for (const m of data.matches) {
    const lp = lockedMap[String(m.id)];
    const rs = resultsMap[String(m.id)];
    if (lp && rs && rs.status === "final") {
      totalPts += quinielaPoints(lp.home, lp.away, rs.home_score, rs.away_score, m.phase);
      gamesPlayed++;
    }
  }

  // Show points banner
  if (gamesPlayed > 0) {
    const banner = document.getElementById("points-banner");
    banner.style.display = "flex";
    document.getElementById("pts-total").textContent = `${totalPts} pts`;
    const avg = (totalPts / gamesPlayed).toFixed(2);
    document.getElementById("pts-meta").textContent =
      `· ${gamesPlayed} partido${gamesPlayed !== 1 ? "s" : ""} jugado${gamesPlayed !== 1 ? "s" : ""}`;
    const avgEl = document.getElementById("pts-avg");
    avgEl.textContent = `${avg} pts/partido`;
    avgEl.style.display = "";
  }

  let activeFilter = "all";

  function applyFilter(f) {
    activeFilter = f;
    document.querySelectorAll(".chip").forEach(c => {
      c.classList.toggle("active", c.dataset.filter === f);
    });
    renderAll(data.matches, f, lockedMap, resultsMap);
  }

  document.querySelectorAll(".chip").forEach(chip => {
    chip.addEventListener("click", () => applyFilter(chip.dataset.filter));
  });

  applyFilter("upcoming");
}

document.addEventListener("DOMContentLoaded", init);
