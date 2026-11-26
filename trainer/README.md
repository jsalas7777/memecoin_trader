# trainer

Machine-learning trainer for the Solana memecoin trader.

Implemented so far:
1. `download_data.py`: **download memecoin price, volume and holder history directly from the
   Solana blockchain** (no Birdeye, DexScreener or other indexer APIs) and save it as CSV.
2. `build_dataset.py`: turn it into an ML dataset of **30-second candles** with holder counts,
   engineered features and future-return targets. Every input is described in [DATASET.md](DATASET.md).
3. `build_sequences.py`: tokenized model inputs. For each moment, the last **120 candles** become
   120 tokens each of candle, volume, holders and bonding curve progress (4 inputs; top-10, trades
   per minute, liquidity, bundle and whale buy/sell exist but are switched off, see MODEL.md).
   The spec is in [MODEL.md](MODEL.md).

## Data sources

| Source | Script | Gives | Cost |
|---|---|---|---|
| **GMGN API** (official) | `download_gmgn.py` | 30s candles from launch (dead coins included), holder history for coins with < 100 wallets, a snapshot per coin (holders, top-10, dev, bundlers, snipers, ...) | GMGN key (free tier ~2 candle requests/s) |
| **Solana RPC** (on-chain) | `download_data.py` | every swap: exact trade counts, buy/sell, liquidity, holders, top-10, whale buys/sells, launch bundle, exact curve progress per candle | RPC plan |

`build_dataset.py --source auto` (default) uses GMGN candles and fills in on-chain columns wherever on-chain data
exists. Inputs nobody has data for become `UNKNOWN` tokens, never guessed values.

### No keys: one command

```bash
cd trainer
.venv/bin/python download_free.py                 # 500 coins from the last 3 days, then builds the training data
.venv/bin/python download_free.py --coins 2000 --days 7
```

Finds pump.fun launches on the free public Solana RPC (sampled evenly, dead coins included), downloads each
from GMGN with its public test key, then builds `dataset_30s.csv` + `sequences_30s.npz`. Progress bars for
every step; rate limits are announced with the pause length. Ctrl+C is safe: rerun to continue. Each run adds
`--coins` new coins. Uses `GMGN_API_KEY` / `SOLANA_RPC_URL` instead when they're set (much faster).

### From a trade archive (fastest: tens of thousands of coins in minutes)

```bash
.venv/bin/python download_archive.py      # 1 day of every pump.fun trade (~1.5 GB) -> archives/ (not in git)
.venv/bin/python import_archive.py        # -> data/dataset/pumpdata-2026-10-15_30s/ (all inputs, verified)
.venv/bin/python train.py --dataset data/dataset/pumpdata-2026-10-15_30s
```

`import_archive.py` takes every coin launched in the archive (so each starts at its first trade), builds
30s candles, holders, top-10, liquidity, curve progress, launch bundle and whale buys/sells exactly like
`build_dataset.py`, in parallel, and checks a sample with the input verifier. `--limit 2000` for a quick run.
`download_archive.py corpus` gets a 39-day archive (~6.7 GB). Mayhem-mode coins (pump.fun's agent trades
their curve) get a 2B supply and `UNKNOWN` curve progress.

### Train the model

```bash
.venv/bin/python train.py          # causal transformer, best checkpoint -> data/models/transformer_30s.pt
.venv/bin/python predict.py        # 24-candle forecast for a test coin vs what actually happened
```

The model reads each coin from its first candle (no waiting for 120 candles) and, at every candle,
forecasts the next 24 candles: candle shape, volume and the price path. Details in [MODEL.md](MODEL.md).

### GMGN quick start

```bash
export GMGN_API_KEY=...                                   # create at https://gmgn.ai/ai
# 1. coins: unbiased sample of past launches (cheap on-chain discovery, works on the free public RPC)
python download_data.py discover --mode new --limit 1000 --min-age 6h --max-age 3d
#    ...and/or record every launch going forward from GMGN's live list
python download_gmgn.py collect --duration 24h
# 2. 30s candles from launch + snapshot + holder history for every coin in data/tokens.csv
python download_gmgn.py history
# 3. build
python build_dataset.py && python build_sequences.py
```

GMGN limits: klines have **no trade count and no per-wallet trades**, and there's **no history of top-10 or
liquidity** (only a current snapshot, saved as `gmgn_*_now` columns in `tokens.csv`). So trades/min, top-10,
liquidity, whale buy/sell and the launch bundle need on-chain data for the same coins (`download_data.py history`;
the bundle needs the history back to the coin's creation). Bonding curve progress works from GMGN alone
(derived from the price). `python download_gmgn.py --demo-key ...` uses GMGN's public test key (testing only).

## Setup

```bash
cd trainer
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Strongly recommended: a dedicated RPC (Helius, QuickNode, Triton, ...).
# The public endpoint works but is heavily rate limited (~1 tx/s).
export SOLANA_RPC_URL="https://mainnet.helius-rpc.com/?api-key=YOUR_KEY"
```

## Usage

```bash
# 1. Discover memecoins on-chain -> data/tokens.csv
# RECOMMENDED for training: launches sampled evenly over a past window. This keeps the real
# mix (most coins die), so the model learns what to avoid, not only what pumps.
python download_data.py discover --mode new --limit 500 --min-age 6h --max-age 2d
python download_data.py discover --mode new --limit 50        # latest launches that are >= 2h old
python download_data.py discover --mode active --limit 25     # hot right now (survivors only: biased)

# 2. Download swap history + OHLCV for every mint in data/tokens.csv
python download_data.py history --max-signatures 20000          # 30s candles by default

#    ...or for specific mints / a time window
python download_data.py history --mint <MINT> --mint <MINT2> --since 7d

# Discover + download in one step
python download_data.py run --mode new --limit 20

# Rebuild candles at other resolutions from saved trades (no RPC calls)
python download_data.py ohlcv --intervals 15s,1min,15min

# 3. Build the ML dataset -> data/dataset/dataset_30s.csv + DATASET.md
python build_dataset.py --interval 30s --horizons 1min,5min,10min --up-threshold 0.10

# 4. Tokenize into 4 x 120-token model inputs -> data/dataset/sequences_30s.npz + MODEL.md
#    (runs the input verification automatically and fails if any input is wrong)
python build_sequences.py

# Re-check the model inputs at any time
python check_inputs.py
```

Re-running `history` is **incremental**. It fetches transactions that are newer than the last run
and keeps backfilling older history until it reaches the start of each pool. Raise
`--max-signatures` (the number of transactions per pool per run) to go deeper per run.

Global flags: `--rpc URL`, `--rps N`, `--concurrency N`, `--data-dir DIR`, `-v`.

## Output

```
data/
  tokens.csv                    catalog: mint, symbol, name, supply, pools, n_trades, holders_now, ...
  trades/<mint>.csv             every swap
  balances/<mint>.csv           post-trade token balance of every wallet seen (for holder counts)
  ohlcv/<interval>/<mint>.csv   candles
  state/<mint>.json             pagination cursors + token metadata
  dataset/dataset_30s.csv       ML dataset (features + targets), see DATASET.md
  dataset/sequences_30s.npz     tokenized model inputs (4 x 120) + targets, see MODEL.md
```

**trades/<mint>.csv**: `signature, slot, tx_index, block_time, datetime, dex, pool_vault, side (buy|sell),
token_amount, quote_amount, quote (SOL|USDC|USDT), price, trader, token_reserve, quote_reserve`

**ohlcv/<interval>/<mint>.csv**: `timestamp, open, high, low, close, volume_sol, volume_token,
buy_volume_sol, sell_volume_sol, trades, buys, sells, unique_traders, sol_reserve, token_reserve, market_cap_sol, holders`

Prices are in **SOL per token**. Candles use SOL-quoted trades only. Empty intervals are
filled with volume 0 and the previous close, so the series is evenly spaced for ML; pass `--no-fill` to disable.

## How it works

1. **Discovery** (`solana_data/discovery.py`)
   - `new`: reads the history of the pump.fun mint authority
     (`TSLvdd1p…`), which signs every pump.fun token creation.
   - `active`: samples recent transactions of DEX programs and ranks mints by unique traders.
2. **Pool finding** (`solana_data/pools.py`)
   - The pump.fun bonding curve and its PumpSwap graduation pool are derived as PDAs from the mint.
   - Other DEX pools (Raydium, Meteora, Orca, …) are found among the largest holders: token accounts owned by a PDA are pool vaults.
3. **Swap extraction** (`solana_data/swaps.py`)
   - For every transaction that touches a pool vault, it reads the vault's token balance change and the opposite change in the pool's SOL reserve. That reserve is lamports on the bonding curve, or a WSOL vault for AMMs.
   - `price = Δsol / Δtoken`. This works the same way on every DEX, including aggregator routes (Jupiter, bots).
   - Liquidity adds/removes, launches, migrations and failed transactions are excluded.
4. **Holders** (`solana_data/holders.py`)
   - Over time: the exact post-transaction balance of every wallet in each pool transaction is
     replayed in order. Holders = wallets with balance > 0 at candle close.
   - Right now: the exact count from `getProgramAccounts`, saved as `holders_now`.
5. **OHLCV** (`solana_data/ohlcv.py`): trades are resampled into candles with pandas.
6. **Dataset** (`dataset/features.py`, `build_dataset.py`): features use only past data, targets use
   only future data, and train/val/test are split by token.

## Dead coins

The dataset deliberately includes coins that die. Most pump.fun launches do.
- `--mode new --min-age 6h --max-age 2d` samples launches evenly in time, whatever their outcome.
- `build_dataset.py` keeps any coin with at least 3 trades. After a coin's last trade, its series
  continues with empty candles, so the minutes before it died get real labels.
- `target_dump_*` (price -30% within the horizon) is the avoid/exit label. `outcome_dead`,
  `outcome_graduated`, `outcome_ath_mcap_sol`, etc. record each coin's fate. They are for
  evaluation only, never inputs.

## Notes / limitations

- Memecoin pools are full of bot transactions that touch the pool without trading. Each one still
  costs a `getTransaction` call, so download speed is set by your RPC plan.
- The dev's initial buy inside the token-creation transaction isn't recorded as a trade.
- Prices are in SOL. Convert to USD by joining a SOL/USD series if needed.
