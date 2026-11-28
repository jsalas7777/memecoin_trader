const ROW_H = 48;
const $ = (id) => document.getElementById(id);
const rowsEl = $("rows");
const tip = $("tip");

let windowSec = 1;
let paused = false;
let last = null;
const rowEls = new Map(); // mint -> element
const metaCache = {};

// live trade list
const FEED_BUF = 4000; // trades kept in the browser
const FEED_ROWS = 250; // trades shown
let tradeBuf = []; // newest first
const tradeIds = new Set();
let pendingFeed = []; // received but not drawn yet (newest first)
let filterMint = null;

// ---------------------------------------------------------------- controls
$("windows").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-w]");
  if (!b) return;
  windowSec = Number(b.dataset.w);
  for (const x of $("windows").children) x.setAttribute("aria-checked", String(x === b));
  if (last) render(last);
});
const togglePause = () => {
  paused = !paused;
  $("pause").setAttribute("aria-pressed", String(paused));
  $("pause").textContent = paused ? "Paused — resume" : "Pause";
  if (!paused && last) { render(last); renderFeed(false); }
};
$("pause").addEventListener("click", togglePause);
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && e.target === document.body) { e.preventDefault(); togglePause(); }
  if (e.key === "Escape" && filterMint) setFilter(null);
});
$("feedClear").addEventListener("click", () => setFilter(null));

// ---------------------------------------------------------------- stream
function connect() {
  const es = new EventSource("/stream");
  es.addEventListener("backfill", (m) => {
    const d = JSON.parse(m.data);
    Object.assign(metaCache, d.meta);
    addTrades(d.trades);
    renderFeed(true);
  });
  es.onmessage = (m) => {
    const snap = JSON.parse(m.data);
    Object.assign(metaCache, snap.meta);
    addTrades(snap.trades || []);
    last = snap;
    renderStatus(snap.status);
    if (!paused) { render(snap); renderFeed(false); }
  };
  es.onerror = () => renderStatus({ connected: false, error: "lost connection to local server" });
}
connect();

function renderStatus(s) {
  const el = $("status");
  el.className = "status " + (s.connected ? "ok" : s.error ? "bad" : "");
  $("statusText").textContent = s.connected
    ? `live · ${/helius/.test(s.upstream) ? "Helius" : new URL(s.upstream).host} · ${s.msgRate} msgs/s`
    : s.error ? `disconnected: ${s.error}` : "connecting…";
}

// ---------------------------------------------------------------- render
function render(snap) {
  const w = snap.windows[windowSec];
  if (!w) return;
  const label = windowSec === 1 ? "last 1s" : `last ${windowSec}s`;
  $("kTxLabel").textContent = `Transactions · ${label}`;
  $("kTx").textContent = fmtInt(w.tx);
  $("kCoins").textContent = fmtInt(w.coins);
  const top = w.rows[0];
  const kTop = $("kTop");
  kTop.textContent = top ? `${name(top.mint)} · ${top.tx} tx` : "–";
  if (top) kTop.href = gmgnUrl(top.mint); else kTop.removeAttribute("href");

  renderTimeline(snap.totals);

  $("empty").hidden = w.rows.length > 0;
  rowsEl.style.height = `${w.rows.length * ROW_H}px`;
  const max = Math.max(1, ...w.rows.map((r) => r.tx));
  const live = new Set();

  w.rows.forEach((r, i) => {
    live.add(r.mint);
    let el = rowEls.get(r.mint);
    if (!el) {
      el = makeRow(r.mint);
      el.classList.add("enter");
      el.style.transform = `translateY(${i * ROW_H}px)`;
      rowsEl.appendChild(el);
      rowEls.set(r.mint, el);
      requestAnimationFrame(() => el.classList.remove("enter"));
    }
    el._data = r;
    el.classList.toggle("selected", r.mint === filterMint);
    el.style.transform = `translateY(${i * ROW_H}px)`;
    const m = metaCache[r.mint];
    el.querySelector(".rank").textContent = i + 1;
    const sym = el.querySelector(".sym");
    sym.firstChild.textContent = name(r.mint);
    sym.title = m?.name || r.mint;
    el.querySelector(".venue").textContent = r.venue === "pumpswap" ? "PumpSwap" : "pump.fun";
    const img = el.querySelector("img");
    if (m?.image && img.dataset.src !== m.image) { img.dataset.src = m.image; img.src = m.image; img.hidden = false; el.querySelector(".ph").hidden = true; }

    const bar = el.querySelector(".bar");
    bar.style.width = `${(r.tx / max) * 100}%`;
    const [bi, si] = bar.children;
    bi.style.flexGrow = r.buys;
    si.style.flexGrow = r.sells;
    bi.style.display = r.buys ? "" : "none";
    si.style.display = r.sells ? "" : "none";
    bi.classList.toggle("solo", !r.sells);
    si.classList.toggle("solo", !r.buys);

    el.querySelector(".tx").textContent = fmtInt(r.tx);
    el.querySelector(".tr").textContent = fmtInt(r.traders);
    el.querySelector(".sol").textContent = fmtSol(r.sol);
    drawSpark(el.querySelector(".spark"), snap.spark[r.mint] || []);
  });

  for (const [mint, el] of rowEls) {
    if (!live.has(mint)) { el.remove(); rowEls.delete(mint); }
  }
}

function makeRow(mint) {
  const el = document.createElement("div");
  el.className = "row";
  el.innerHTML = `
    <span class="rank"></span>
    <span class="coin">
      <img alt="" hidden referrerpolicy="no-referrer"><span class="ph"></span>
      <span class="txt">
        <a class="sym" target="_blank" rel="noopener" href="${gmgnUrl(mint)}"><span></span><span class="venue"></span></a>
        <span class="mint">${mint.slice(0, 4)}…${mint.slice(-5)}</span>
      </span>
    </span>
    <span class="bar"><i class="buy"></i><i class="sell"></i></span>
    <span class="v tx"></span>
    <span class="v tr"></span>
    <span class="v sol"></span>
    <svg class="spark" viewBox="0 0 128 28" preserveAspectRatio="none" aria-hidden="true"><path></path><circle r="2"></circle></svg>
    <button class="tbtn" title="List this coin's transactions">Trades</button>`;
  el.querySelector(".tbtn").onclick = (e) => {
    e.stopPropagation();
    setFilter(filterMint === mint ? null : mint);
  };
  el.querySelector("img").onerror = function () { this.hidden = true; el.querySelector(".ph").hidden = false; };
  // Clicking anywhere on the row opens the coin on GMGN (the name link handles its own clicks,
  // so cmd/ctrl/middle-click on it still work as a normal link).
  el.addEventListener("click", (e) => {
    if (e.target.closest("a, button")) return;
    window.open(gmgnUrl(mint), "_blank", "noopener");
  });
  el.addEventListener("mousemove", (e) => showRowTip(e, el._data));
  el.addEventListener("mouseleave", hideTip);
  return el;
}

function drawSpark(svg, series) {
  const n = series.length;
  if (!n) return;
  const max = Math.max(1, ...series);
  const W = 128, H = 28, pad = 3;
  const x = (i) => (i / (n - 1)) * (W - pad * 2) + pad;
  const y = (v) => H - pad - (v / max) * (H - pad * 2);
  svg.querySelector("path").setAttribute("d", series.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(""));
  const c = svg.querySelector("circle");
  c.setAttribute("cx", x(n - 1));
  c.setAttribute("cy", y(series[n - 1]));
}

let tlBars = [];
function renderTimeline(totals) {
  const el = $("timeline");
  if (tlBars.length !== totals.length) {
    el.innerHTML = "";
    tlBars = totals.map((_, i) => {
      const b = document.createElement("div");
      b.className = "b";
      b.addEventListener("mousemove", (e) => {
        const v = b._v, ago = totals.length - 1 - i;
        showTip(e, `<b>${fmtInt(v)}</b> transactions<br><span class="k">${ago === 0 ? "current second (still filling)" : `${ago}s ago`}</span>`);
      });
      b.addEventListener("mouseleave", hideTip);
      el.appendChild(b);
      return b;
    });
  }
  const max = Math.max(1, ...totals);
  totals.forEach((v, i) => { tlBars[i]._v = v; tlBars[i].style.height = `${Math.max(2, (v / max) * 100)}%`; });
}

// ---------------------------------------------------------------- live trade list
function addTrades(list) {
  const fresh = [];
  for (const t of list) {
    if (tradeIds.has(t.id)) continue;
    tradeIds.add(t.id);
    fresh.push(t);
  }
  if (!fresh.length) return;
  const byTime = (a, b) => b.t - a.t || b.id - a.id;
  fresh.sort(byTime);
  tradeBuf = fresh.concat(tradeBuf).sort(byTime);
  if (tradeBuf.length > FEED_BUF) for (const t of tradeBuf.splice(FEED_BUF)) tradeIds.delete(t.id);
  pendingFeed = fresh.concat(pendingFeed).slice(0, FEED_BUF);
}

function renderFeed(rebuild) {
  const list = $("flist");
  const match = (t) => !filterMint || t.mint === filterMint;
  if (rebuild) {
    pendingFeed = [];
    list.replaceChildren(...tradeBuf.filter(match).slice(0, FEED_ROWS).map((t) => feedRow(t, false)));
  } else {
    const items = pendingFeed.filter(match).slice(0, FEED_ROWS);
    pendingFeed = [];
    if (items.length) {
      const keepScroll = list.scrollTop > 0; // reading older trades: don't let new ones push the view
      const before = list.scrollHeight;
      list.prepend(...items.map((t) => feedRow(t, true)));
      while (list.children.length > FEED_ROWS) list.lastChild.remove();
      if (keepScroll) list.scrollTop += list.scrollHeight - before;
    }
  }
  $("fempty").hidden = list.children.length > 0;
  $("fempty").textContent = filterMint ? "No trades for this coin in the last 60s yet…" : "Waiting for trades…";
  updateFeedHead();
}

function updateFeedHead() {
  const cutoff = (last?.now || Date.now()) - 60_000;
  if (!filterMint) {
    const n = tradeBuf.filter((t) => t.t >= cutoff).length;
    $("feedTitle").textContent = "Live trades";
    $("feedSub").textContent = `All coins · newest first · ${fmtInt(Math.round(n / 60))} trades/s`;
  } else {
    const recent = tradeBuf.filter((t) => t.mint === filterMint && t.t >= cutoff);
    const buys = recent.filter((t) => t.buy);
    const sol = (arr) => arr.reduce((s, t) => s + t.sol, 0);
    const m = metaCache[filterMint];
    $("feedTitle").textContent = `Trades · ${m?.symbol || name(filterMint)}${m?.name ? " — " + m.name : ""}`;
    $("feedSub").textContent = `Last 60s: ${fmtInt(recent.length)} trades · ${fmtInt(buys.length)} buys (${fmtSol(sol(buys))} SOL) · ${fmtInt(recent.length - buys.length)} sells (${fmtSol(sol(recent) - sol(buys))} SOL)`;
  }
  $("feedClear").hidden = !filterMint;
}

async function setFilter(mint) {
  filterMint = mint;
  for (const [m, el] of rowEls) el.classList.toggle("selected", m === mint);
  $("flist").scrollTop = 0;
  renderFeed(true);
  if (!mint) return;
  try {
    // the server still holds the last 60s of this coin's trades, including ones before the page loaded
    const j = await (await fetch(`/trades?mint=${encodeURIComponent(mint)}`)).json();
    Object.assign(metaCache, j.meta);
    addTrades(j.trades);
    if (filterMint === mint) renderFeed(true);
  } catch {}
}

function feedRow(t, isNew) {
  const el = document.createElement("div");
  el.className = `ftr${isNew ? " new" : ""}${t.sol >= 5 ? " big" : ""}`;
  const txUrl = `https://solscan.io/tx/${t.sig}`;
  el.innerHTML = `
    <span class="time">${fmtTime(t.t)}</span>
    <span class="side ${t.buy ? "buy" : "sell"}">${t.buy ? "Buy" : "Sell"}</span>
    <button class="fcoin" title="Show only this coin's trades">${esc(name(t.mint))}</button>
    <span class="fsol">${fmtSol(t.sol)}</span>
    <a class="fwallet" target="_blank" rel="noopener" href="https://gmgn.ai/sol/address/${t.user}">${t.user.slice(0, 4)}…${t.user.slice(-4)}</a>
    <a class="ext" target="_blank" rel="noopener" href="${txUrl}" aria-label="View transaction on Solscan">↗</a>`;
  el.querySelector(".fcoin").onclick = (e) => { e.stopPropagation(); setFilter(t.mint); };
  el.addEventListener("click", (e) => {
    if (e.target.closest("a, button")) return;
    window.open(txUrl, "_blank", "noopener");
  });
  el.addEventListener("mousemove", (e) => {
    const m = metaCache[t.mint];
    showTip(e, `
      <div><span class="side ${t.buy ? "buy" : "sell"}">${t.buy ? "Buy" : "Sell"}</span> <b>${fmtSol(t.sol)} SOL</b> of <b>${esc(m?.symbol || name(t.mint))}</b> ${esc(m?.name || "")}</div>
      <div class="line"><span class="k">Time</span><span>${new Date(t.t).toLocaleString()}</span></div>
      <div class="line"><span class="k">Venue</span><span>${t.venue === "pumpswap" ? "PumpSwap" : "pump.fun"}</span></div>
      <div class="k" style="margin-top:6px">Coin</div><div class="mono">${t.mint}</div>
      <div class="k">Wallet</div><div class="mono">${t.user}</div>
      <div class="k">Transaction</div><div class="mono">${t.sig}</div>
      <div class="k" style="margin-top:6px">Click: Solscan · wallet: GMGN · coin: filter</div>`);
  });
  el.addEventListener("mouseleave", hideTip);
  return el;
}

// ---------------------------------------------------------------- tooltip
function showRowTip(e, r) {
  if (!r) return;
  const m = metaCache[r.mint];
  const pct = r.tx ? Math.round((r.buys / (r.buys + r.sells || 1)) * 100) : 0;
  showTip(e, `
    <div><b>${esc(m?.symbol || "")}</b> ${esc(m?.name || "")}</div>
    <div class="k" style="font-family:ui-monospace,monospace;font-size:11px;word-break:break-all;margin-bottom:6px">${r.mint}</div>
    <div class="line"><span class="k">Transactions</span><b>${fmtInt(r.tx)}</b></div>
    <div class="line"><span class="k"><i class="sw buy"></i> Buys</span><span>${fmtInt(r.buys)} (${pct}%)</span></div>
    <div class="line"><span class="k"><i class="sw sell"></i> Sells</span><span>${fmtInt(r.sells)}</span></div>
    <div class="line"><span class="k">Unique traders</span><span>${fmtInt(r.traders)}</span></div>
    <div class="line"><span class="k">SOL volume</span><span>${fmtSol(r.sol)}</span></div>
    <div class="k" style="margin-top:6px">Window: last ${windowSec}s · click to open on GMGN · “Trades” lists its transactions</div>`);
}
function showTip(e, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
function hideTip() { tip.hidden = true; }

// ---------------------------------------------------------------- format
function gmgnUrl(mint) { return `https://gmgn.ai/sol/token/${mint}`; }
function name(mint) {
  const m = metaCache[mint];
  return m?.symbol ? m.symbol : `${mint.slice(0, 4)}…${mint.slice(-4)}`;
}
function fmtTime(t) {
  const d = new Date(t);
  return `${d.toLocaleTimeString([], { hour12: false })}.${String(d.getMilliseconds()).padStart(3, "0")}`;
}
function fmtInt(n) { return Number(n).toLocaleString(); }
function fmtSol(s) { return s >= 100 ? s.toFixed(0) : s >= 1 ? s.toFixed(2) : s >= 0.001 ? s.toFixed(3) : s > 0 ? "<0.001" : "0"; }
function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]); }
