// Live "most-traded coins right now" server.
//
// One websocket to Helius (standard Solana `logsSubscribe`, available on the free plan)
// listens to every transaction that touches pump.fun (bonding curve) or PumpSwap (the AMM
// pump.fun coins graduate to). Each trade emits an Anchor event in the logs, which we decode
// to get the coin's mint without fetching the transaction. Trades are aggregated into rolling
// windows (1s / 5s / 30s / 60s) and pushed to browsers over Server-Sent Events.
//
// No dependencies: Node >= 22 has a global WebSocket client and fetch.

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
loadDotEnv(path.join(ROOT, ".env"));

const PORT = Number(process.env.PORT || 3000);
const KEY = process.env.HELIUS_API_KEY || "";
const WS_URL = process.env.RPC_WS_URL || (KEY && `wss://mainnet.helius-rpc.com/?api-key=${KEY}`);
const HTTP_URL = process.env.RPC_HTTP_URL || (KEY && `https://mainnet.helius-rpc.com/?api-key=${KEY}`);
const COMMITMENT = process.env.COMMITMENT || "processed"; // processed = fastest
if (!WS_URL || !HTTP_URL) {
  console.error("Missing HELIUS_API_KEY. Copy .env.example to .env and paste your free key from https://dashboard.helius.dev");
  process.exit(1);
}

const PUMP_FUN = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P";
const PUMP_SWAP = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA";
const WSOL = "So11111111111111111111111111111111111111112";

// Anchor event discriminators (first 8 bytes of the "Program data:" payload).
const D_PUMP_TRADE = "bddb7fd34ee661ee"; // pump.fun TradeEvent
const D_AMM_BUY = "67f4521f2cf57777"; // PumpSwap BuyEvent
const D_AMM_SELL = "3e2f370aa503dc2a"; // PumpSwap SellEvent

const WINDOWS = [1, 5, 30, 60]; // seconds
const KEEP_MS = 60_000;
const TOP_N = 25;
const SPARK_SECONDS = 30;
const TICK_MS = 250;

// ---------------------------------------------------------------- state

/** @type {{t:number, sig:string, mint:string, buy:boolean, sol:number, user:string, venue:string}[]} */
let trades = [];
const seenSigs = new Map(); // signature -> time, to dedupe a tx reported by both subscriptions
const poolMint = new Map(); // PumpSwap pool -> {mint, solIsBase, solIsQuote}
const pendingPools = new Map(); // pool -> [events waiting for the pool's mint]
const meta = new Map(); // mint -> {symbol, name, image} | null (lookup failed)
const metaQueue = new Set();
const stats = { connected: false, msgs: 0, msgRate: 0, lastMsgAt: 0, error: "", since: Date.now() };

// ---------------------------------------------------------------- upstream websocket

let ws;
let reconnectDelay = 1000;
function connect() {
  console.log(`[ws] connecting to ${mask(WS_URL)}`);
  ws = new WebSocket(WS_URL);
  const subIds = new Map();
  ws.onopen = () => {
    stats.connected = true;
    stats.error = "";
    reconnectDelay = 1000;
    console.log("[ws] connected, subscribing to pump.fun + PumpSwap logs");
    [PUMP_FUN, PUMP_SWAP].forEach((program, i) =>
      ws.send(JSON.stringify({ jsonrpc: "2.0", id: i + 1, method: "logsSubscribe", params: [{ mentions: [program] }, { commitment: COMMITMENT }] })),
    );
  };
  ws.onmessage = (m) => {
    stats.msgs++;
    stats.lastMsgAt = Date.now();
    let d;
    try { d = JSON.parse(m.data); } catch { return; }
    if (d.error) { stats.error = d.error.message || JSON.stringify(d.error); console.error("[ws] error:", stats.error); return; }
    if (d.id) { subIds.set(d.result, d.id); return; }
    const v = d.params?.result?.value;
    if (v) onLogs(v);
  };
  ws.onerror = (e) => { stats.error = e.message || "websocket error"; };
  ws.onclose = (e) => {
    stats.connected = false;
    if (e.code !== 1000) stats.error = stats.error || `closed (${e.code}${e.reason ? " " + e.reason : ""})`;
    console.log(`[ws] closed (${e.code}); reconnecting in ${reconnectDelay}ms`);
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 30_000);
  };
}
// pump.fun never goes quiet for long, so silence means a dead socket.
setInterval(() => {
  if (stats.connected && Date.now() - stats.lastMsgAt > 30_000) { stats.error = "no data for 30s"; ws.close(); }
}, 5000);

function onLogs(v) {
  if (v.err) return; // failed tx: no trade happened
  const now = Date.now();
  if (seenSigs.has(v.signature)) return;
  seenSigs.set(v.signature, now);

  for (const line of v.logs) {
    if (!line.startsWith("Program data: ")) continue;
    let b;
    try { b = Buffer.from(line.slice(14), "base64"); } catch { continue; }
    if (b.length < 8) continue;
    const disc = b.toString("hex", 0, 8);
    if (disc === D_PUMP_TRADE && b.length >= 97) {
      // mint[8..40] sol_amount u64[40] token_amount u64[48] is_buy u8[56] user[57..89]
      record({ t: now, sig: v.signature, mint: b58(b.subarray(8, 40)), buy: b[56] === 1, sol: Number(b.readBigUInt64LE(40)) / 1e9, user: b58(b.subarray(57, 89)), venue: "pump" });
    } else if ((disc === D_AMM_BUY || disc === D_AMM_SELL) && b.length >= 184) {
      // timestamp[8] base_amount[16] ... quote_amount[64] ... pool[120..152] user[152..184]
      const ev = { t: now, sig: v.signature, poolBuy: disc === D_AMM_BUY, base: Number(b.readBigUInt64LE(16)), quote: Number(b.readBigUInt64LE(64)), user: b58(b.subarray(152, 184)) };
      const pool = b58(b.subarray(120, 152));
      const info = poolMint.get(pool);
      if (info) recordAmm(ev, info);
      else if (info === undefined) {
        if (!pendingPools.has(pool)) pendingPools.set(pool, []);
        const q = pendingPools.get(pool);
        if (q.length < 500) q.push(ev);
      }
    }
  }
}

function recordAmm(ev, info) {
  // A pool "buy" buys the base token. When SOL is the base, that is a sell of the memecoin.
  const buy = info.solIsBase ? !ev.poolBuy : ev.poolBuy;
  const sol = info.solIsBase ? ev.base / 1e9 : info.solIsQuote ? ev.quote / 1e9 : 0;
  record({ t: ev.t, sig: ev.sig, mint: info.mint, buy, sol, user: ev.user, venue: "pumpswap" });
}

const outbox = []; // trades not yet sent to browsers (the live trade list)
let nextId = 1;
function record(tr) {
  if (tr.mint === WSOL) return;
  tr.id = nextId++;
  tr.sol = +tr.sol.toFixed(6);
  trades.push(tr);
  if (outbox.length < 5000) outbox.push(tr);
}

function metaFor(mints) {
  const out = {};
  for (const m of mints) if (meta.get(m)) out[m] = meta.get(m);
  return out;
}

// ---------------------------------------------------------------- HTTP RPC lookups (batched, rate-friendly)

async function rpc(method, params) {
  const r = await fetch(HTTP_URL, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ jsonrpc: "2.0", id: 1, method, params }) });
  if (!r.ok) throw new Error(`${method} HTTP ${r.status}`);
  const j = await r.json();
  if (j.error) throw new Error(`${method}: ${j.error.message}`);
  return j.result;
}

// Resolve PumpSwap pools to their coin. Pool account: base_mint at byte 43, quote_mint at 75.
let poolBusy = false;
setInterval(async () => {
  if (poolBusy || pendingPools.size === 0) return;
  poolBusy = true;
  const pools = [...pendingPools.keys()].slice(0, 100);
  try {
    const res = await rpc("getMultipleAccounts", [pools, { encoding: "base64", commitment: "confirmed" }]);
    res.value.forEach((acc, i) => {
      const pool = pools[i];
      const evs = pendingPools.get(pool) || [];
      pendingPools.delete(pool);
      if (!acc || acc.owner !== PUMP_SWAP) { poolMint.set(pool, null); return; }
      const d = Buffer.from(acc.data[0], "base64");
      const base = b58(d.subarray(43, 75));
      const quote = b58(d.subarray(75, 107));
      const info = { mint: base === WSOL ? quote : base, solIsBase: base === WSOL, solIsQuote: quote === WSOL };
      poolMint.set(pool, info);
      for (const ev of evs) recordAmm(ev, info);
    });
  } catch (e) {
    console.error("[rpc] pool lookup failed:", e.message, e.cause?.code || e.cause?.message || "");
  } finally {
    poolBusy = false;
  }
}, 300);

// Names, symbols and images via Helius DAS (getAssetBatch), only for coins that make a top list.
let metaBusy = false;
setInterval(async () => {
  if (metaBusy || metaQueue.size === 0) return;
  metaBusy = true;
  const ids = [...metaQueue].slice(0, 100);
  ids.forEach((id) => metaQueue.delete(id));
  try {
    const assets = await rpc("getAssetBatch", { ids });
    ids.forEach((id, i) => {
      const a = assets?.[i];
      const md = a?.content?.metadata || {};
      meta.set(id, {
        symbol: md.symbol || a?.token_info?.symbol || "",
        name: md.name || "",
        image: a?.content?.files?.[0]?.cdn_uri || a?.content?.links?.image || "",
      });
    });
  } catch (e) {
    console.error("[rpc] metadata lookup failed:", e.message, e.cause?.code || e.cause?.message || "");
    ids.forEach((id) => meta.set(id, null)); // don't hammer; retried below
    setTimeout(() => ids.forEach((id) => meta.delete(id)), 60_000);
  } finally {
    metaBusy = false;
  }
}, 1500);

// ---------------------------------------------------------------- aggregation

function snapshot() {
  const now = Date.now();
  // prune
  let cut = 0;
  while (cut < trades.length && now - trades[cut].t > KEEP_MS) cut++;
  if (cut) trades = trades.slice(cut);
  for (const [sig, t] of seenSigs) { if (now - t > KEEP_MS) seenSigs.delete(sig); else break; }

  const perWindow = WINDOWS.map(() => new Map());
  const totalSigs = Array.from({ length: 60 }, () => new Set());
  const spark = new Map(); // mint -> per-second unique tx counts (oldest first)
  const sparkSigs = new Map();

  for (let i = trades.length - 1; i >= 0; i--) {
    const tr = trades[i];
    const age = now - tr.t;
    const sec = Math.floor(age / 1000);
    if (sec < 60) totalSigs[59 - sec].add(tr.sig);
    if (sec < SPARK_SECONDS) {
      if (!sparkSigs.has(tr.mint)) sparkSigs.set(tr.mint, Array.from({ length: SPARK_SECONDS }, () => new Set()));
      sparkSigs.get(tr.mint)[SPARK_SECONDS - 1 - sec].add(tr.sig);
    }
    WINDOWS.forEach((w, wi) => {
      if (age >= w * 1000) return;
      const m = perWindow[wi];
      let c = m.get(tr.mint);
      if (!c) m.set(tr.mint, (c = { mint: tr.mint, sigs: new Set(), buySigs: new Set(), sellSigs: new Set(), traders: new Set(), sol: 0, venues: new Set() }));
      c.sigs.add(tr.sig);
      (tr.buy ? c.buySigs : c.sellSigs).add(tr.sig);
      c.traders.add(tr.user);
      c.sol += tr.sol;
      c.venues.add(tr.venue);
    });
  }

  const windows = {};
  const shown = new Set();
  WINDOWS.forEach((w, wi) => {
    const rows = [...perWindow[wi].values()]
      .map((c) => ({ mint: c.mint, tx: c.sigs.size, buys: c.buySigs.size, sells: c.sellSigs.size, traders: c.traders.size, sol: +c.sol.toFixed(6), venue: c.venues.has("pumpswap") ? "pumpswap" : "pump" }))
      .sort((a, b) => b.tx - a.tx || b.sol - a.sol)
      .slice(0, TOP_N);
    rows.forEach((r) => shown.add(r.mint));
    windows[w] = { coins: perWindow[wi].size, tx: new Set(trades.filter((t) => now - t.t < w * 1000).map((t) => t.sig)).size, rows };
  });

  const metaOut = {};
  for (const mint of shown) {
    if (meta.has(mint)) { if (meta.get(mint)) metaOut[mint] = meta.get(mint); }
    else metaQueue.add(mint);
    const s = sparkSigs.get(mint);
    if (s) spark.set(mint, s.map((x) => x.size));
  }

  return {
    now,
    status: { connected: stats.connected, error: stats.error, msgRate: stats.msgRate, pendingPools: pendingPools.size, upstream: mask(WS_URL), commitment: COMMITMENT },
    totals: totalSigs.map((s) => s.size),
    windows,
    spark: Object.fromEntries(spark),
    meta: metaOut,
  };
}

let lastMsgs = 0;
setInterval(() => { stats.msgRate = stats.msgs - lastMsgs; lastMsgs = stats.msgs; }, 1000);

// ---------------------------------------------------------------- HTTP server + SSE

const clients = new Set();
setInterval(() => {
  const fresh = outbox.splice(0);
  if (clients.size === 0) return;
  const snap = snapshot();
  snap.trades = fresh;
  Object.assign(snap.meta, metaFor(new Set(fresh.map((t) => t.mint))));
  const payload = `data: ${JSON.stringify(snap)}\n\n`;
  for (const res of clients) res.write(payload);
}, TICK_MS);
// keep pruning even with no browser open
setInterval(() => { if (clients.size === 0) snapshot(); }, 5000);

const TYPES = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml" };
http
  .createServer((req, res) => {
    const url = new URL(req.url, "http://x");
    if (url.pathname === "/stream") {
      res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
      res.write("retry: 2000\n\n");
      // Seed the trade list with the most recent trades so it isn't empty on page load.
      const recent = trades.slice(-300);
      res.write(`event: backfill\ndata: ${JSON.stringify({ trades: recent, meta: metaFor(new Set(recent.map((t) => t.mint))) })}\n\n`);
      clients.add(res);
      req.on("close", () => clients.delete(res));
      return;
    }
    if (url.pathname === "/trades") {
      // Every trade of one coin still in memory (last 60s), newest last.
      const mint = url.searchParams.get("mint") || "";
      const list = trades.filter((t) => t.mint === mint).slice(-2000);
      if (mint && !meta.has(mint)) metaQueue.add(mint);
      res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ trades: list, meta: metaFor([mint]) }));
      return;
    }
    const file = path.join(ROOT, "public", url.pathname === "/" ? "index.html" : path.normalize(url.pathname));
    if (!file.startsWith(path.join(ROOT, "public"))) { res.writeHead(403).end(); return; }
    fs.readFile(file, (err, buf) => {
      if (err) { res.writeHead(404).end("not found"); return; }
      res.writeHead(200, { "content-type": TYPES[path.extname(file)] || "application/octet-stream" }).end(buf);
    });
  })
  .on("error", (e) => {
    if (e.code === "EADDRINUSE") {
      console.error(`Port ${PORT} is already in use (another copy of this server?). Stop it with:\n  lsof -ti tcp:${PORT} | xargs kill\nor run on another port:  PORT=3001 npm start`);
      process.exit(1);
    }
    throw e;
  })
  .listen(PORT, () => console.log(`[http] open http://localhost:${PORT}`));

connect();

// ---------------------------------------------------------------- helpers

function b58(buf) {
  const A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
  const digits = [];
  for (const byte of buf) {
    let carry = byte;
    for (let j = 0; j < digits.length; j++) { carry += digits[j] << 8; digits[j] = carry % 58; carry = (carry / 58) | 0; }
    while (carry) { digits.push(carry % 58); carry = (carry / 58) | 0; }
  }
  let s = "";
  for (const byte of buf) { if (byte === 0) s += "1"; else break; }
  for (let i = digits.length - 1; i >= 0; i--) s += A[digits[i]];
  return s;
}

function mask(u) {
  return String(u).replace(/(api-key=)[^&]+/, "$1••••");
}

function loadDotEnv(file) {
  if (!fs.existsSync(file)) return;
  for (const line of fs.readFileSync(file, "utf8").split("\n")) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/);
    if (m && !(m[1] in process.env)) process.env[m[1]] = m[2].replace(/^["']|["']$/g, "");
  }
}
