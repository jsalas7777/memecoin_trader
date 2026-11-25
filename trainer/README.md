# trainer

Machine-learning trainer for the Solana memecoin trader.

Implemented so far:
1. `download_data.py`: **download memecoin price, volume and holder history directly from the
   Solana blockchain** (no Birdeye, DexScreener or other indexer APIs) and save it as CSV.
2. `build_dataset.py`: turn it into an ML dataset of **30-second candles** with holder counts,
   engineered features and future-return targets. Every input is described in [DATASET.md](DATASET.md).
3. `build_sequences.py`: tokenized model inputs. For each moment, the last **120 candles** become
   120 candle tokens, 120 volume tokens and 120 holder tokens. The spec is in [MODEL.md](MODEL.md).

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
python download_data.py discover --mode new --limit 50        # newest pump.fun launches
python download_data.py discover --mode active --limit 25     # most-traded on pump.fun / PumpSwap now

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

# 4. Tokenize into 3 x 120-token model inputs -> data/dataset/sequences_30s.npz + MODEL.md
python build_sequences.py
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
  dataset/sequences_30s.npz     tokenized 3 x 120 model inputs + targets, see MODEL.md
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

## Notes / limitations

- Memecoin pools are full of bot transactions that touch the pool without trading. Each one still
  costs a `getTransaction` call, so download speed is set by your RPC plan.
- The dev's initial buy inside the token-creation transaction isn't recorded as a trade.
- Prices are in SOL. Convert to USD by joining a SOL/USD series if needed.
