# Memecoin dataset

Generated 2026-11-25 00:26 UTC by `build_dataset.py` from on-chain Solana data downloaded with `download_data.py`.

## Build summary

| | |
|---|---|
| File | `data/dataset/dataset_30s.csv` |
| Candle timeframe | **30s** |
| Tokens | 2 |
| Rows (candles) | 3,480 |
| Time range | 2026-11-23 21:11:00+00:00 to 2026-11-25 00:08:00+00:00 |
| test | 1 tokens, 3,235 rows |
| train | 1 tokens, 245 rows |
| Input features | 53 |
| Targets | 15 |
| Coins dead at download | **0 of 2 (0%)** |
| Coins that graduated | 2 of 2 |
| `target_up_1m` positive rate | 0.0% |
| `target_dump_1m` positive rate | 0.0% |
| `target_up_5m` positive rate | 0.0% |
| `target_dump_5m` positive rate | 0.0% |
| `target_up_10m` positive rate | 0.5% |
| `target_dump_10m` positive rate | 0.0% |

One row = one token x one candle. Rows are sorted by token, then time.

## Identifier columns (not model inputs)

| Column | Description |
|---|---|
| `mint` | Token mint address |
| `symbol` | Token symbol (from on-chain metadata) |
| `timestamp` | Candle open time, UTC |
| `split` | train / val / test, assigned per token (all rows of a token share a split) |

## Inputs (features)

All features at row *t* use only information available at the close of candle *t*.

| # | Feature | Description | Unit | Non-null | Mean | Min | Max |
|---|---|---|---|---|---|---|---|
| 1 | `open` | First trade price in the candle | SOL/token | 100% | 4.25e-07 | 3.518e-07 | 4.262e-07 |
| 2 | `high` | Highest trade price in the candle | SOL/token | 100% | 4.25e-07 | 4.088e-07 | 4.262e-07 |
| 3 | `low` | Lowest trade price in the candle | SOL/token | 100% | 4.25e-07 | 3.476e-07 | 4.262e-07 |
| 4 | `close` | Last trade price in the candle (previous close if no trades) | SOL/token | 100% | 4.25e-07 | 4.088e-07 | 4.262e-07 |
| 5 | `volume_sol` | Total traded value in the candle | SOL | 100% | 0.009312 | 0 | 16.99 |
| 6 | `volume_token` | Total traded token amount in the candle | tokens | 100% | 2.416e+04 | 0 | 4.327e+07 |
| 7 | `buy_volume_sol` | SOL spent buying the token | SOL | 100% | 0.007019 | 0 | 12.74 |
| 8 | `sell_volume_sol` | SOL received selling the token | SOL | 100% | 0.002293 | 0 | 4.249 |
| 9 | `trades` | Number of swaps | count | 100% | 0.01293 | 0 | 26 |
| 10 | `buys` | Number of buy swaps | count | 100% | 0.008621 | 0 | 19 |
| 11 | `sells` | Number of sell swaps | count | 100% | 0.00431 | 0 | 8 |
| 12 | `unique_traders` | Distinct wallets that traded in the candle | count | 100% | 0.01121 | 0 | 21 |
| 13 | `token_reserve` | Tokens in the most recently traded pool at candle close | tokens | 100% | 2e+08 | 1.994e+08 | 206,900,000 |
| 14 | `market_cap_sol` | close x total supply | SOL | 100% | 397.3 | 388 | 398 |
| 15 | `holders` | Wallets holding > 0 tokens at candle close (rebuilt from on-chain balances) | count | 100% | 18.65 | 14 | 19 |
| 16 | `liquidity_sol` | Total SOL in all of the coin's pools at candle close (bonding curve + AMMs) | SOL | 100% | 91.31 | 87.59 | 91.59 |
| 17 | `liquidity_change_1` | liquidity_sol / liquidity_sol 1 candle ago - 1 | ratio | 100% | 0 | 0 | 0 |
| 18 | `liquidity_change_10` | liquidity_sol / liquidity_sol 10 candles ago - 1 (a sharp drop = liquidity pulled) | ratio | 99% | 0 | 0 | 0 |
| 19 | `liquidity_to_mcap` | liquidity_sol / market_cap_sol (how thin the market is) | ratio | 100% | 0.2298 | 0.2257 | 0.2301 |
| 20 | `trades_per_min` | Swaps in the last 60 seconds | trades/min | 100% | 0.02586 | 0 | 26 |
| 21 | `trades_per_min_5m` | Average swaps per minute over the last 5 minutes | trades/min | 100% | 0.02586 | 0 | 5.2 |
| 22 | `top10_pct` | % of total supply held by the 10 largest wallets (pools and program accounts excluded) | % | 100% | 10.06 | 5.108 | 10.43 |
| 23 | `top10_change_10` | top10_pct - top10_pct 10 candles ago (negative = big holders selling) | % points | 99% | 0 | 0 | 0 |
| 24 | `age_s` | Seconds since the token's first downloaded trade | s | 100% | 4.535e+04 | 0 | 97,020 |
| 25 | `candles_since_trade` | Consecutive candles with no trades up to now (0 = traded this candle) | count | 100% | 1512 | 0 | 3,234 |
| 26 | `ret_1` | Log return of close over 1 candle | log | 100% | 0 | 0 | 0 |
| 27 | `ret_2` | Log return of close over 2 candles | log | 100% | 0 | 0 | 0 |
| 28 | `ret_5` | Log return of close over 5 candles | log | 100% | 0 | 0 | 0 |
| 29 | `ret_10` | Log return of close over 10 candles | log | 99% | 0 | 0 | 0 |
| 30 | `ret_20` | Log return of close over 20 candles | log | 99% | 0 | 0 | 0 |
| 31 | `ema_ratio_10` | close / EMA(close, 10) - 1 | ratio | 100% | 0 | 0 | 0 |
| 32 | `ema_ratio_30` | close / EMA(close, 30) - 1 | ratio | 100% | 0 | 0 | 0 |
| 33 | `vwap_ratio_20` | close / VWAP over the last 20 candles - 1 | ratio | 1% | 0.08356 | 0.08137 | 0.08576 |
| 34 | `rsi_14` | Relative Strength Index over 14 candles | 0-100 | 99% | 100 | 100 | 100 |
| 35 | `drawdown_from_ath` | close / all-time-high so far - 1 | ratio | 100% | 0 | 0 | 0 |
| 36 | `body_pct` | close / open - 1 | ratio | 100% | 8.356e-05 | 0 | 0.162 |
| 37 | `range_pct` | high / low - 1 | ratio | 100% | 9.49e-05 | 0 | 0.1759 |
| 38 | `upper_wick_pct` | (high - max(open, close)) / close | ratio | 100% | 0 | 0 | 0 |
| 39 | `lower_wick_pct` | (min(open, close) - low) / close | ratio | 100% | 8.562e-06 | 0 | 0.01967 |
| 40 | `volatility_10` | Std of 1-candle log returns over 10 candles | log | 99% | 0 | 0 | 0 |
| 41 | `volatility_20` | Std of 1-candle log returns over 20 candles | log | 99% | 0 | 0 | 0 |
| 42 | `volume_sol_sum_10` | SOL volume over the last 10 candles | SOL | 100% | 0.09312 | 0 | 16.99 |
| 43 | `volume_ratio_10` | volume_sol / mean volume_sol of the last 10 candles | ratio | 1% | 0.1 | 0 | 1 |
| 44 | `trades_ratio_10` | trades / mean trades of the last 10 candles | ratio | 1% | 0.1 | 0 | 1 |
| 45 | `avg_trade_sol` | volume_sol / trades (0 if no trades) | SOL | 100% | 0.0004209 | 0 | 0.8116 |
| 46 | `buy_ratio` | buy_volume_sol / volume_sol (0.5 if no volume) | 0-1 | 100% | 0.5001 | 0.5 | 0.7581 |
| 47 | `buy_count_ratio` | buys / trades (0.5 if no trades) | 0-1 | 100% | 0.5001 | 0.5 | 0.7308 |
| 48 | `net_flow_sol` | buy_volume_sol - sell_volume_sol | SOL | 100% | 0.004727 | 0 | 8.488 |
| 49 | `net_flow_sol_10` | net_flow_sol summed over the last 10 candles | SOL | 100% | 0.04727 | 0 | 8.488 |
| 50 | `holders_change_1` | holders - holders 1 candle ago | count | 100% | 0 | 0 | 0 |
| 51 | `holders_change_10` | holders - holders 10 candles ago | count | 99% | 0 | 0 | 0 |
| 52 | `holders_growth_10` | holders / holders 10 candles ago - 1 | ratio | 99% | 0 | 0 | 0 |
| 53 | `log_market_cap` | log(1 + market_cap_sol) | log SOL | 100% | 5.987 | 5.964 | 5.989 |

## Targets (what to predict)

Targets look strictly into the future. Never feed them to the model as inputs.

| Target | Description | Non-null | Mean |
|---|---|---|---|
| `target_ret_1m` | Log return from close[t] to close[t+2] (1m ahead) | 100% | 0 |
| `target_max_ret_1m` | Best case: max(high[t+1..t+2]) / close[t] - 1 | 100% | 0 |
| `target_min_ret_1m` | Worst case: min(low[t+1..t+2]) / close[t] - 1 | 100% | 0 |
| `target_up_1m` | 1 if target_ret_1m > log(1 + 0.1) (price up more than 10%), else 0 | 100% | 0 |
| `target_dump_1m` | 1 if the price drops more than 30% below close[t] at any point in the next 1m (low[t+1..t+2]), else 0 | 100% | 0 |
| `target_ret_5m` | Log return from close[t] to close[t+10] (5m ahead) | 100% | 0 |
| `target_max_ret_5m` | Best case: max(high[t+1..t+10]) / close[t] - 1 | 100% | 0 |
| `target_min_ret_5m` | Worst case: min(low[t+1..t+10]) / close[t] - 1 | 100% | 0 |
| `target_up_5m` | 1 if target_ret_5m > log(1 + 0.1) (price up more than 10%), else 0 | 100% | 0 |
| `target_dump_5m` | 1 if the price drops more than 30% below close[t] at any point in the next 5m (low[t+1..t+10]), else 0 | 100% | 0 |
| `target_ret_10m` | Log return from close[t] to close[t+20] (10m ahead) | 100% | 0.01597 |
| `target_max_ret_10m` | Best case: max(high[t+1..t+20]) / close[t] - 1 | 100% | 0.1429 |
| `target_min_ret_10m` | Worst case: min(low[t+1..t+20]) / close[t] - 1 | 100% | 0 |
| `target_up_10m` | 1 if target_ret_10m > log(1 + 0.1) (price up more than 10%), else 0 | 100% | 0.00546 |
| `target_dump_10m` | 1 if the price drops more than 30% below close[t] at any point in the next 10m (low[t+1..t+20]), else 0 | 100% | 0 |

## Coin outcomes (labels, not inputs)

One value per coin, repeated on every row of that coin. These describe the coin's whole future, so **never use them as model inputs**. Use them to weight, filter or evaluate (for example, how often the model buys coins that end up dead).

| Column | Description |
|---|---|
| `outcome_dead` | 1 if the coin is dead at download time (no trade for 1h, or price 90%+ below its all-time high) |
| `outcome_graduated` | 1 if the coin left the pump.fun bonding curve and traded on an AMM (e.g. PumpSwap) |
| `outcome_ath_mcap_sol` | Highest market cap the coin ever reached |
| `outcome_final_drawdown` | Last price / all-time-high price - 1 (-0.99 = lost 99% from its peak) |
| `outcome_lifetime_s` | Seconds between the first and last trade |

<details><summary>Per-coin outcomes in this build</summary>

| symbol | mint | trades | dead | graduated | ATH mcap (SOL) | final drawdown | lifetime |
|---|---|---|---|---|---|---|---|
| JIMMY | `38M2z4UC…` | 37 | 0 | 1 | 1.803e+04 | -1% | 1623 min |
| 4CHAN | `DVdWuNK8…` | 29 | 0 | 1 | 2727 | -0% | 128.1 min |

</details>

## How the inputs are produced

- **Prices and volume** come from every swap that touched the token's pools (pump.fun bonding
  curve, PumpSwap, Raydium, Meteora, ...). Price = SOL moved / tokens moved in each swap,
  priced in SOL. Swaps under 0.00001 SOL are ignored as dust.
- **Candles** are 30s buckets. Candles with no trades are kept, with zero volume and
  OHLC equal to the previous close, so every token is an evenly spaced series.
- **Holders** are rebuilt by replaying the exact post-transaction token balance of every wallet
  that appears in a pool transaction. A wallet counts as a holder while its balance is > 0.
  Pool vaults count as holders, the same as on block explorers. Wallets that only received
  tokens by plain transfer and never traded through a pool are not seen, so this is a slight
  undercount. The exact current count from chain is in `data/tokens.csv` (`holders_now`).
- **Market cap** = close x current total supply.
- **Dead coins are kept.** Coins are not filtered by success, and every coin with at least
  3 trades is included. After a coin's last trade, the series continues with empty
  `NO_TRADE` candles up to the download time (capped at the longest horizon). So the
  minutes before a coin dies get real targets: the model sees "this pattern -> nothing / dump".
- **Split**: each token goes entirely into train, val or test (by a hash of its mint), so the
  model is always tested on coins it has never seen.

## Regenerate

```bash
python download_data.py run --mode new --limit 500 --min-age 6h --max-age 2d --max-signatures 20000
python build_dataset.py --interval 30s --horizons 1min,5min,10min --up-threshold 0.1
```

## Skipped tokens

Tokens with fewer than 3 trades, or too few candles for the longest horizon:

- `7fmD1p3jzva28EPP9bPqbDoxZGj5j398yJoE3EVypump`: only 18 candles
