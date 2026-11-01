// Queen Board — 2D board client. No build step, no dependencies.
// The server is the source of truth for rules; this file only draws and sends clicks.

const SVGNS = "http://www.w3.org/2000/svg";
const S = 100; // square size in viewBox units
const GLYPH = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
const TEXT = "︎"; // force text (not emoji) presentation

const $ = (id) => document.getElementById(id);
const svg = $("board");

let st = null;           // last server state
let flipped = false;
let selected = null;     // square name, e.g. "e2"
let arrows = [];         // [{from, to, alpha}]
let busy = false;
let drag = null;         // {from, ghost, pieceEl}

// ---------- helpers ----------
const files = "abcdefgh";
const sqName = (f, r) => files[f] + (r + 1);
const sqFR = (n) => [files.indexOf(n[0]), +n[1] - 1];
function sqXY(n) {
  const [f, r] = sqFR(n);
  return flipped ? [(7 - f) * S, r * S] : [f * S, (7 - r) * S];
}
function el(tag, attrs = {}, parent = svg) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  parent && parent.appendChild(e);
  return e;
}
function piecesFromFen(fen) {
  const out = {};
  fen.split(" ")[0].split("/").forEach((row, i) => {
    let f = 0;
    for (const c of row) {
      if (/\d/.test(c)) f += +c;
      else { out[sqName(f, 7 - i)] = c; f++; }
    }
  });
  return out;
}
const isWhite = (p) => p === p.toUpperCase();
const humanCanMove = () =>
  st && !st.result && !busy && (st.mode === "hotseat" || (!st.engine_to_move && st.turn === st.human_color));

async function api(path, body) {
  const r = await fetch("/api/" + path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || r.statusText);
  return d;
}

// ---------- drawing ----------
function draw() {
  svg.innerHTML = "";
  const defs = el("defs");
  const g = el("radialGradient", { id: "checkGrad" }, defs);
  el("stop", { offset: "0%", "stop-color": "rgb(255,60,40)", "stop-opacity": ".95" }, g);
  el("stop", { offset: "100%", "stop-color": "rgb(255,60,40)", "stop-opacity": "0" }, g);
  const mk = el("marker", { id: "ah", viewBox: "0 0 10 10", refX: "5", refY: "5", markerWidth: "3.2",
    markerHeight: "3.2", orient: "auto-start-reverse" }, defs);
  el("path", { d: "M0,0 L10,5 L0,10 z", fill: "context-stroke" }, mk);

  // squares
  for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++) {
    const [x, y] = sqXY(sqName(f, r));
    el("rect", { x, y, width: S, height: S, class: (f + r) % 2 ? "sq-light" : "sq-dark" });
  }
  if (!st) return;

  // highlights
  if (st.last_move) for (const n of [st.last_move.slice(0, 2), st.last_move.slice(2, 4)]) {
    const [x, y] = sqXY(n); el("rect", { x, y, width: S, height: S, class: "hl-last" });
  }
  if (st.check) { const [x, y] = sqXY(st.check); el("rect", { x, y, width: S, height: S, class: "hl-check" }); }
  if (selected) { const [x, y] = sqXY(selected); el("rect", { x, y, width: S, height: S, class: "hl-sel" }); }

  // coordinates
  for (let i = 0; i < 8; i++) {
    const fileSq = sqName(i, flipped ? 7 : 0), [fx, fy] = sqXY(fileSq);
    el("text", { x: fx + S - 8, y: fy + S - 8, "text-anchor": "end", class: "coord",
      fill: (sqFR(fileSq)[0] + sqFR(fileSq)[1]) % 2 ? "#a7774a" : "#e9d8b4" }).textContent = files[i];
    const rankSq = sqName(flipped ? 7 : 0, i), [rx, ry] = sqXY(rankSq);
    el("text", { x: rx + 7, y: ry + 20, class: "coord",
      fill: (sqFR(rankSq)[0] + sqFR(rankSq)[1]) % 2 ? "#a7774a" : "#e9d8b4" }).textContent = i + 1;
  }

  // pieces
  const pcs = piecesFromFen(st.fen);
  for (const n in pcs) {
    const p = pcs[n], [x, y] = sqXY(n);
    const t = el("text", { x: x + S / 2, y: y + S / 2 + 4, class: "piece " + (isWhite(p) ? "w" : "b"), "data-sq": n });
    t.textContent = GLYPH[p.toLowerCase()] + TEXT;
    if (drag && drag.from === n) t.classList.add("dragging");
  }

  // move hints
  if (selected) for (const to of destsFrom(selected)) {
    const [x, y] = sqXY(to);
    if (pcs[to]) el("circle", { cx: x + S / 2, cy: y + S / 2, r: S / 2 - 6, class: "ring" });
    else el("circle", { cx: x + S / 2, cy: y + S / 2, r: 15, class: "dot" });
  }

  // arrows
  for (const a of arrows) {
    const [x1, y1] = sqXY(a.from), [x2, y2] = sqXY(a.to);
    const cx1 = x1 + S / 2, cy1 = y1 + S / 2, cx2 = x2 + S / 2, cy2 = y2 + S / 2;
    const len = Math.hypot(cx2 - cx1, cy2 - cy1), k = (len - 26) / len;
    el("line", { x1: cx1, y1: cy1, x2: cx1 + (cx2 - cx1) * k, y2: cy1 + (cy2 - cy1) * k,
      stroke: a.color || "rgb(230,160,40)", "stroke-opacity": a.alpha, "stroke-width": 16,
      "stroke-linecap": "round", "marker-end": "url(#ah)", "pointer-events": "none" });
  }
}

function destsFrom(from) {
  return [...new Set(st.legal.filter((m) => m.startsWith(from)).map((m) => m.slice(2, 4)))];
}

function renderPanel() {
  const status = $("status");
  status.classList.toggle("thinking", busy);
  if (st.result) status.textContent = st.result;
  else if (busy) status.textContent = `${st.engine} is thinking`;
  else {
    const side = st.turn === "white" ? "White" : "Black";
    status.textContent = `${side} to move` + (st.check ? " — check!" : "");
  }

  const opp = st.mode === "hotseat" ? "Human" : engineLabel(st);
  // Always say who you're playing, so a pending opponent change can't be mistaken for the real one.
  if (!st.result && !busy) {
    status.textContent += st.mode === "hotseat"
      ? " · you're playing both sides"
      : (st.turn === st.human_color ? " · your move" : "") + ` · vs ${opp}`;
  }
  const meTop = flipped ? "white" : "black";
  const name = (c) => st.mode === "hotseat" ? `${c} · human` : `${c} · ${c === st.human_color ? "you" : opp}`;
  $("tag-top").textContent = name(meTop);
  $("tag-bottom").textContent = name(meTop === "white" ? "black" : "white");

  const ol = $("moves"); ol.innerHTML = "";
  const startsBlack = st.fen && st.pgn.includes('[FEN "') && / b /.test(st.pgn.match(/\[FEN "([^"]+)"/)?.[1] || "");
  const h = startsBlack ? ["…", ...st.history] : st.history;
  for (let i = 0; i < h.length; i += 2) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="n">${i / 2 + 1}.</span><span class="m">${h[i]}</span><span class="m">${h[i + 1] || ""}</span>`;
    ol.appendChild(li);
  }
  const ms = ol.querySelectorAll(".m"); const lastIdx = h.length - 1;
  if (lastIdx >= 0 && ms[lastIdx]) ms[lastIdx].classList.add("last");
  ol.scrollTop = ol.scrollHeight;

  $("undo").disabled = busy || !st.history.length;
  $("analyze").disabled = busy || !!st.result;
  renderPending();
}

function engineLabel(s) {
  if (s.engine === "queen") return "Queen 4B";
  if (s.engine === "stockfish") return `Stockfish (level ${s.skill ?? "?"})`;
  if (s.engine === "random") return "Random mover";
  return s.engine;
}

// Opponent/colour in the controls vs. what the running game actually uses.
function pendingChange() {
  if (!st) return false;
  const opp = $("opp").value;
  if (opp === "hotseat") return st.mode !== "hotseat";
  return st.mode !== "engine" || st.engine !== opp || st.human_color !== $("color").value;
}
function renderPending() {
  const p = pendingChange();
  $("pending").classList.toggle("hidden", !p);
  $("new").classList.toggle("attention", p);
}

function update(newState) {
  st = newState; selected = null; arrows = []; draw(); renderPanel();
}

// ---------- input ----------
function eventSquare(ev) {
  const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
  const p = pt.matrixTransform(svg.getScreenCTM().inverse());
  let f = Math.floor(p.x / S), r = 7 - Math.floor(p.y / S);
  if (f < 0 || f > 7 || r < 0 || r > 7) return null;
  if (flipped) { f = 7 - f; r = 7 - r; }
  return sqName(f, r);
}
function svgPoint(ev) {
  const pt = svg.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
  return pt.matrixTransform(svg.getScreenCTM().inverse());
}
function ownPiece(sq) {
  const p = piecesFromFen(st.fen)[sq];
  return p && (isWhite(p) ? "white" : "black") === st.turn;
}

svg.addEventListener("pointerdown", (ev) => {
  if (!humanCanMove()) return;
  const sq = eventSquare(ev); if (!sq) return;
  if (selected && destsFrom(selected).includes(sq)) { tryMove(selected, sq); return; }
  if (ownPiece(sq)) {
    selected = sq;
    const p = piecesFromFen(st.fen)[sq], pos = svgPoint(ev);
    drag = { from: sq, moved: false };
    draw();
    drag.ghost = el("text", { x: pos.x, y: pos.y + 4, class: "piece ghost " + (isWhite(p) ? "w" : "b") });
    drag.ghost.textContent = GLYPH[p.toLowerCase()] + TEXT;
    svg.setPointerCapture(ev.pointerId);
  } else { selected = null; draw(); }
});
svg.addEventListener("pointermove", (ev) => {
  if (!drag || !drag.ghost) return;
  const pos = svgPoint(ev); drag.moved = true;
  drag.ghost.setAttribute("x", pos.x); drag.ghost.setAttribute("y", pos.y + 4);
});
svg.addEventListener("pointerup", (ev) => {
  if (!drag) return;
  const from = drag.from, to = eventSquare(ev);
  drag = null;
  if (to && to !== from && destsFrom(from).includes(to)) tryMove(from, to);
  else draw(); // dropped back / clicked: keep selection for click-to-move
});

function tryMove(from, to) {
  const opts = st.legal.filter((m) => m.slice(0, 4) === from + to);
  if (opts.length > 1) return askPromotion(opts);
  sendMove(opts[0]);
}

function askPromotion(opts) {
  const box = $("promo"); box.innerHTML = "";
  const white = st.turn === "white";
  for (const k of ["q", "r", "b", "n"]) {
    const m = opts.find((o) => o.endsWith(k)); if (!m) continue;
    const b = document.createElement("button");
    b.textContent = GLYPH[k] + TEXT; b.style.color = white ? "#fbf6ec" : "#231a12";
    b.style.background = white ? "#3a2c1d" : "#e9d8b4";
    b.onclick = () => { box.classList.add("hidden"); sendMove(m); };
    box.appendChild(b);
  }
  box.classList.remove("hidden");
}

async function sendMove(uci) {
  try { update(await api("move", { uci })); }
  catch (e) { alert(e.message); return; }
  await maybeEngineMove();
}

async function maybeEngineMove() {
  if (!st.engine_to_move) return;
  busy = true; renderPanel();
  try { const s = await api("engine_move", {}); busy = false; update(s); }
  catch (e) { busy = false; renderPanel(); $("status").textContent = "Engine error: " + e.message; }
}

// ---------- controls ----------
function syncSkillVisibility() {
  $("skill-wrap").style.visibility = $("opp").value === "stockfish" ? "visible" : "hidden";
}

// Changing opponent or colour before the first move starts the game right away;
// mid-game it waits for "New game" (and says so) instead of wiping the position.
async function onSettingsChange() {
  syncSkillVisibility();
  if (busy) return;
  if (!st.history.length || st.result) return startNewGame();
  renderPending();
}
$("opp").addEventListener("change", onSettingsChange);
$("color").addEventListener("change", onSettingsChange);

// Skill applies immediately, even mid-game.
$("skill").addEventListener("change", async () => {
  const level = Math.max(0, Math.min(20, Math.round(+$("skill").value || 0)));
  $("skill").value = level;
  try {
    await api("stockfish/skill", { level });
    update(await api("state"));
  } catch (e) { $("status").textContent = "Couldn't set level: " + e.message; }
});

$("new").onclick = () => startNewGame();

async function startNewGame() {
  const opp = $("opp").value;
  const body = opp === "hotseat"
    ? { mode: "hotseat" }
    : { mode: "engine", engine: opp, human_color: $("color").value };
  const fen = $("fen").value.trim(); if (fen) body.fen = fen;
  try {
    if (opp === "stockfish") await api("stockfish/skill", { level: +$("skill").value });
    flipped = opp !== "hotseat" && $("color").value === "black";
    update(await api("new", body));
    clearAnalysis();
    await maybeEngineMove();
  } catch (e) { $("status").textContent = "Couldn't start: " + e.message; }
}
$("undo").onclick = async () => { update(await api("undo", {})); };
$("flip").onclick = () => { flipped = !flipped; draw(); renderPanel(); };
$("copy-pgn").onclick = () => navigator.clipboard.writeText(st.pgn);
$("copy-fen").onclick = () => navigator.clipboard.writeText(st.fen);

function clearAnalysis() {
  const box = $("analysis"); box.className = "analysis empty";
  box.textContent = "Ask an engine about the current position. Its best move and line are drawn as arrows.";
}

$("analyze").onclick = async () => {
  const engine = $("an-engine").value, question = $("question").value.trim() || null;
  const box = $("analysis"); box.className = "analysis"; box.textContent = `Asking ${engine}…`;
  const fenAtAsk = st.fen;
  $("analyze").disabled = true;
  try {
    const d = await api("analyze", { engine, question });
    if (st.fen !== fenAtAsk) return; // position changed while waiting
    box.innerHTML = "";
    const best = document.createElement("div"); best.className = "best";
    let ev = "";
    if (d.mate_in != null) ev = ` · mate in ${Math.abs(d.mate_in)} for ${d.mate_in > 0 ? "White" : "Black"}`;
    else if (d.eval_cp != null) ev = ` · ${(d.eval_cp / 100).toFixed(2).replace(/^(?!-)/, "+")}`;
    best.textContent = (d.best_san || "no move") + ev;
    box.appendChild(best);
    if (d.line_san && d.line_san.length > 1) {
      const l = document.createElement("div"); l.textContent = "Line: " + d.line_san.join(" "); box.appendChild(l);
    }
    if (d.text) { const t = document.createElement("div"); t.style.marginTop = "8px"; t.textContent = d.text; box.appendChild(t); }
    if (d.raw && Object.keys(d.raw).length) {
      const det = document.createElement("details");
      det.innerHTML = "<summary>raw engine output</summary>";
      const pre = document.createElement("pre"); pre.textContent = JSON.stringify(d.raw, null, 1);
      det.appendChild(pre); box.appendChild(det);
    }
    const line = d.line && d.line.length ? d.line : (d.best_move ? [d.best_move] : []);
    arrows = line.slice(0, 3).map((m, i) => ({
      from: m.slice(0, 2), to: m.slice(2, 4), alpha: [0.9, 0.5, 0.35][i],
      color: i % 2 ? "rgb(70,140,220)" : "rgb(230,160,40)",
    }));
    draw();
  } catch (e) {
    box.innerHTML = `<span class="err">${e.message.replace(/</g, "&lt;")}</span>`;
  } finally { $("analyze").disabled = busy || !!st.result; }
};

// ---------- boot ----------
(async () => {
  try {
    const engines = await api("engines");
    for (const e of engines) {
      for (const sel of [$("opp"), $("an-engine")]) {
        const o = [...sel.options].find((x) => x.value === e.name);
        if (o && !e.available) { o.textContent += " (not set up)"; o.title = e.reason; }
      }
    }
  } catch {}
  // Make the controls match the game the server is actually running.
  const s = await api("state");
  $("opp").value = s.mode === "hotseat" ? "hotseat" : s.engine;
  $("color").value = s.human_color;
  if (s.skill != null) $("skill").value = s.skill;
  flipped = s.mode === "engine" && s.human_color === "black";
  syncSkillVisibility();
  update(s);
})();
