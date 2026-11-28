# Live memecoin transaction board

Shows which Solana memecoins are getting the most transactions right now, with rolling
windows of 1s, 5s, 30s and 60s. Data comes from the Helius **free** plan.

## Run

```bash
cd webapp
cp .env.example .env        # paste your free key from https://dashboard.helius.dev
npm start                   # or: node server.js   (Node 22+, no npm install needed)
open http://localhost:3000
```

## How it works

```
Helius websocket (logsSubscribe)  ──>  server.js  ──SSE, 4×/s──>  browser (public/)
  pump.fun + PumpSwap logs             decode + aggregate
```

- **One upstream websocket.** It uses the standard Solana `logsSubscribe` method, which the
  free plan includes, subscribed to the pump.fun bonding-curve program (`6EF8…F6P`) and the
  PumpSwap AMM (`pAMM…EXEA`), where pump.fun coins trade after they graduate.
- **No per-transaction fetches.** Every trade writes an Anchor event to the transaction logs
  (`Program data: …`). The server decodes it directly:
  - pump.fun `TradeEvent`: mint, SOL amount, buy/sell flag, trader.
  - PumpSwap `BuyEvent`/`SellEvent`: pool, amounts, trader. Each pool is resolved to its
    coin once, with a batched `getMultipleAccounts` call, then cached.
- **Names and images** come from Helius DAS `getAssetBatch`. Only coins that reach a top list
  are looked up, and the results are cached.
- Failed transactions are skipped. A transaction seen by both subscriptions is counted once.
- **Tx** = unique transactions that traded the coin. **Traders** = unique signer wallets.
  **SOL vol** = SOL moved in those trades (0 for non-SOL pairs).
- The key stays on the server. The browser only talks to `localhost`.

## UI

- Window toggle: 1s / 5s / 30s / 60s. "Last 1s" is a rolling 1000 ms window, so bars don't
  reset to zero at every second boundary.
- **Pause** (or press space) freezes the list so you can read it. Click any row to open the coin
  on GMGN (`gmgn.ai/sol/token/<mint>`), in a new tab. The "Top coin" card links the same way.
- **Live trades** panel (right): every trade as it lands — time, buy/sell, coin, SOL, wallet.
  Click a trade to open the transaction on Solscan, the wallet to open it on GMGN, or the coin
  to show only that coin's trades. The **Trades** button on a ranking row does the same; the
  list then also loads that coin's trades from the last 60s and shows buy/sell totals.
  Esc or "All coins ✕" goes back to every coin. Comes from the same stream, so no extra credits.
- Hover a row, a trade or a timeline bar for details.

## Config (`.env`)

| var | default | |
|---|---|---|
| `HELIUS_API_KEY` | — | required |
| `PORT` | 3000 | |
| `COMMITMENT` | processed | `processed` is fastest. `confirmed` drops the rare forked tx and adds ~0.5–1s. |
| `RPC_WS_URL` / `RPC_HTTP_URL` | Helius URLs | override to use any other Solana RPC |

## Limits

- It covers pump.fun and PumpSwap, which is where most memecoin transactions on Solana
  happen. Raydium, Meteora and other launchpads are not included. Each needs its own
  event decoder.
- Free plan credits: the websocket streams continuously, so check usage in the Helius
  dashboard if you leave it running for long.
