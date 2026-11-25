# Model inputs

> [!IMPORTANT]
> ## ⭐ MODEL INPUTS: 6 sequences × 120 tokens
> Each sample = the last **120 candles of 30 seconds** (**60 minutes**) of one coin. Every candle becomes **one token in each of the 6 inputs**.
>
> | # | Input | Shape | Vocab | One token = | Built from (dataset column) |
> |---|---|---|---|---|---|
> | **1** | **`candle_tokens`** | **[batch, 120]** | 86 | **Candle**: price move vs previous close x candle range | `open`, `high`, `low`, `close`, `trades` |
> | **2** | **`volume_tokens`** | **[batch, 120]** | 28 | **Volume**: SOL volume traded in the candle (log scale) | `volume_sol` |
> | **3** | **`holder_tokens`** | **[batch, 120]** | 263 | **Holders**: number of holders at candle close (log scale) | `holders` |
> | **4** | **`top10_tokens`** | **[batch, 120]** | 42 | **Top-10 concentration**: % of supply held by the 10 largest wallets | `top10_pct` |
> | **5** | **`trades_tokens`** | **[batch, 120]** | 48 | **Trades per minute**: swaps in the last 60 seconds | `trades_per_min` |
> | **6** | **`liquidity_tokens`** | **[batch, 120]** | 74 | **Liquidity**: total SOL in the coin's pools (log scale) | `liquidity_sol` |
>
> **Input tensor:** `x = [batch, 120, 6]` int16 token ids (`0` = PAD). Position 119 = the current candle *t*.
>
> **NOT inputs:** `target_*` (what to predict) and `outcome_*` (each coin's fate). Both contain the future. Feeding them to the model is look-ahead leakage.

Generated 2026-11-25 00:28 UTC by `build_sequences.py`. Tokenization rules live in `dataset/tokenizer.py`.

## Overview

- Position 0 is the oldest candle and position 119 is the current candle *t*.
- Token id **0 = PAD** in every stream. A coin younger than 120 candles is
  **left-padded** with PAD, so the current candle is always at position 119. Mask with `tokens != 0`.
- All streams are aligned: position *i* in every stream describes the same candle.
- Candles are evenly spaced. A 30s period with no swaps is still a step, with the tokens
  `NO_TRADE` (candle) and `ZERO` (volume).
- Only data up to the close of candle *t* is used, so there is no look-ahead.

## ⭐ Input 1: `candle_tokens` [batch, 120]

Each candle with at least one swap becomes one of 21 x 4 = 84 tokens:

- **move** = close / previous close - 1, split into 21 signed bins with edges at
  ±0.1%, ±0.5%, ±1%, ±2%, ±3%, ±5%, ±8%, ±12%, ±20%, ±35%. The bins are wider in the tails, because memecoins move a lot.
- **range** = high / low - 1, split into 4 bins with edges at 1%, 3%, 8%
  (how volatile the bar was inside the 30s).

`token = 2 + move_bin x 4 + range_bin`. A candle with no swaps is `NO_TRADE`.
For the first candle of a coin, "previous close" is its own open.

## ⭐ Input 2: `volume_tokens` [batch, 120]

SOL volume of the candle (buys + sells) on a log10 scale, in 0.25-decade steps
(each bin is ~1.78x the previous one) from 0.001 SOL to 1000 SOL.
Volume is absolute, not normalized per coin, because 0.1 SOL and 100 SOL of volume mean
very different things for a memecoin. A candle with no volume is `ZERO`.

## ⭐ Input 3: `holder_tokens` [batch, 120]

Number of wallets holding > 0 tokens at the close of each candle. 1-20 holders each have
their own token. Above that, each bin covers 5% more holders than the previous one, so growth
from 1,000 to 1,050 holders is still visible. `UNKNOWN` means no balance data was available.
The holder history is rebuilt from on-chain balances (see [DATASET.md](DATASET.md)).

## ⭐ Input 4: `top10_tokens` [batch, 120]

% of the total supply held by the 10 largest wallets at candle close, in 2.5% steps
(40 bins from 0% to 100%). Pools, bonding curves and other program-owned (PDA) accounts are
excluded, because they're liquidity, not people who can dump. A high or rising value means a few
wallets can crash the price. A falling value means big holders are selling or distributing.
`UNKNOWN` means no balance data was available.

## ⭐ Input 5: `trades_tokens` [batch, 120]

Number of swaps in the last 60 seconds (the current 30s candle plus the previous one).
1-10 trades/min each have their own token. Above that, each bin covers 25% more
trades/min than the previous one. `ZERO` = no trades in the last minute (activity is dying).

## ⭐ Input 6: `liquidity_tokens` [batch, 120]

Total SOL in all of the coin's pools at candle close (pump.fun bonding curve + PumpSwap /
Raydium / Meteora pools), on a log10 scale in 0.1-decade steps (each bin ~1.26x the
previous) from 0.01 SOL to 100,000 SOL. When a coin graduates, the emptied
bonding curve stops counting, so liquidity is not double-counted. A sudden drop in this
stream is liquidity being pulled.

## Targets

Each sample carries the targets of its current candle *t*, from `build_dataset.py`:

| Target | Meaning |
|---|---|
| `target_up_5m` | **1 if the price is more than 10% higher 5 minutes later** (suggested main target) |
| `target_up_1m`, `target_up_10m` | same, 1 and 10 minutes ahead |
| `target_ret_{1m,5m,10m}` | log return to that horizon (regression) |
| `target_dump_{1m,5m,10m}` | **1 if the price falls more than 30% at any point within the horizon** (avoid / exit signal) |
| `target_max_ret_*`, `target_min_ret_*` | best and worst price within the horizon (take-profit / stop-loss) |
| `outcome_*` | per-coin fate: dead, graduated, ATH market cap, final drawdown, lifetime. Labels only, never inputs (see DATASET.md) |

Other arrays in the file: `mint`, `segment`, `timestamp` (unix seconds of candle *t*), `split` (train/val/test, per token) and `n_history` (real, non-PAD candles in the window). Windows never span a gap in the downloaded data: a new `segment` starts with fresh padding.

## Suggested model

```
candle_tokens    ─ Embedding(86, d) ─┐
volume_tokens    ─ Embedding(28, d) ─┤
holder_tokens    ─ Embedding(263, d) ┤
top10_tokens     ─ Embedding(42, d) ─┼─ sum ─ + positional embedding(120, d)
trades_tokens    ─ Embedding(48, d) ─┤
liquidity_tokens ─ Embedding(74, d) ─┘
      └─> Transformer encoder (causal or bidirectional, PAD masked)
            └─> representation at position 119 (current candle)
                  └─> heads: P(up 5m) [BCE], P(dump 5m) [BCE], return 5m [MSE]
```

Train on `split == "train"`, tune on `val` and report on `test`. Splits are by coin, so the test
set only has coins the model has never seen.

**Dead coins are in the data on purpose.** Most pump.fun launches die within minutes, so most
samples are negative (`target_up_* = 0`, often `target_dump_* = 1`). That is the real
distribution the model will trade in. Keep it, and handle the imbalance with class weights
or a focal loss rather than by removing dead coins. When evaluating, check
`outcome_dead` for the coins the model would have bought.

```python
import numpy as np
d = np.load("data/dataset/sequences_30s.npz")
streams = ["candle_tokens", "volume_tokens", "holder_tokens",
           "top10_tokens", "trades_tokens", "liquidity_tokens"]
x = np.stack([d[k] for k in streams], axis=-1)  # [N, 120, 6]
y = d["target_up_5m"]
train = d["split"] == "train"
```

## Vocabularies



<details><summary><b>Candle vocabulary (86 ids)</b></summary>

| id | meaning |
|---|---|
| 0 | PAD |
| 1 | NO_TRADE (no swaps in this candle) |
| 2 | down > 35%, range < 1% |
| 3 | down > 35%, range 1%-3% |
| 4 | down > 35%, range 3%-8% |
| 5 | down > 35%, range > 8% |
| 6 | down 20% to 35%, range < 1% |
| 7 | down 20% to 35%, range 1%-3% |
| 8 | down 20% to 35%, range 3%-8% |
| 9 | down 20% to 35%, range > 8% |
| 10 | down 12% to 20%, range < 1% |
| 11 | down 12% to 20%, range 1%-3% |
| 12 | down 12% to 20%, range 3%-8% |
| 13 | down 12% to 20%, range > 8% |
| 14 | down 8% to 12%, range < 1% |
| 15 | down 8% to 12%, range 1%-3% |
| 16 | down 8% to 12%, range 3%-8% |
| 17 | down 8% to 12%, range > 8% |
| 18 | down 5% to 8%, range < 1% |
| 19 | down 5% to 8%, range 1%-3% |
| 20 | down 5% to 8%, range 3%-8% |
| 21 | down 5% to 8%, range > 8% |
| 22 | down 3% to 5%, range < 1% |
| 23 | down 3% to 5%, range 1%-3% |
| 24 | down 3% to 5%, range 3%-8% |
| 25 | down 3% to 5%, range > 8% |
| 26 | down 2% to 3%, range < 1% |
| 27 | down 2% to 3%, range 1%-3% |
| 28 | down 2% to 3%, range 3%-8% |
| 29 | down 2% to 3%, range > 8% |
| 30 | down 1% to 2%, range < 1% |
| 31 | down 1% to 2%, range 1%-3% |
| 32 | down 1% to 2%, range 3%-8% |
| 33 | down 1% to 2%, range > 8% |
| 34 | down 0.5% to 1%, range < 1% |
| 35 | down 0.5% to 1%, range 1%-3% |
| 36 | down 0.5% to 1%, range 3%-8% |
| 37 | down 0.5% to 1%, range > 8% |
| 38 | down 0.1% to 0.5%, range < 1% |
| 39 | down 0.1% to 0.5%, range 1%-3% |
| 40 | down 0.1% to 0.5%, range 3%-8% |
| 41 | down 0.1% to 0.5%, range > 8% |
| 42 | flat (|move| < 0.1%), range < 1% |
| 43 | flat (|move| < 0.1%), range 1%-3% |
| 44 | flat (|move| < 0.1%), range 3%-8% |
| 45 | flat (|move| < 0.1%), range > 8% |
| 46 | up 0.1% to 0.5%, range < 1% |
| 47 | up 0.1% to 0.5%, range 1%-3% |
| 48 | up 0.1% to 0.5%, range 3%-8% |
| 49 | up 0.1% to 0.5%, range > 8% |
| 50 | up 0.5% to 1%, range < 1% |
| 51 | up 0.5% to 1%, range 1%-3% |
| 52 | up 0.5% to 1%, range 3%-8% |
| 53 | up 0.5% to 1%, range > 8% |
| 54 | up 1% to 2%, range < 1% |
| 55 | up 1% to 2%, range 1%-3% |
| 56 | up 1% to 2%, range 3%-8% |
| 57 | up 1% to 2%, range > 8% |
| 58 | up 2% to 3%, range < 1% |
| 59 | up 2% to 3%, range 1%-3% |
| 60 | up 2% to 3%, range 3%-8% |
| 61 | up 2% to 3%, range > 8% |
| 62 | up 3% to 5%, range < 1% |
| 63 | up 3% to 5%, range 1%-3% |
| 64 | up 3% to 5%, range 3%-8% |
| 65 | up 3% to 5%, range > 8% |
| 66 | up 5% to 8%, range < 1% |
| 67 | up 5% to 8%, range 1%-3% |
| 68 | up 5% to 8%, range 3%-8% |
| 69 | up 5% to 8%, range > 8% |
| 70 | up 8% to 12%, range < 1% |
| 71 | up 8% to 12%, range 1%-3% |
| 72 | up 8% to 12%, range 3%-8% |
| 73 | up 8% to 12%, range > 8% |
| 74 | up 12% to 20%, range < 1% |
| 75 | up 12% to 20%, range 1%-3% |
| 76 | up 12% to 20%, range 3%-8% |
| 77 | up 12% to 20%, range > 8% |
| 78 | up 20% to 35%, range < 1% |
| 79 | up 20% to 35%, range 1%-3% |
| 80 | up 20% to 35%, range 3%-8% |
| 81 | up 20% to 35%, range > 8% |
| 82 | up > 35%, range < 1% |
| 83 | up > 35%, range 1%-3% |
| 84 | up > 35%, range 3%-8% |
| 85 | up > 35%, range > 8% |

</details>

<details><summary><b>Volume vocabulary (28 ids)</b></summary>

| id | meaning |
|---|---|
| 0 | PAD |
| 1 | ZERO (no volume) |
| 2 | < 0.001 SOL |
| 3 | 0.001 - 0.001778 SOL |
| 4 | 0.001778 - 0.003162 SOL |
| 5 | 0.003162 - 0.005623 SOL |
| 6 | 0.005623 - 0.01 SOL |
| 7 | 0.01 - 0.01778 SOL |
| 8 | 0.01778 - 0.03162 SOL |
| 9 | 0.03162 - 0.05623 SOL |
| 10 | 0.05623 - 0.1 SOL |
| 11 | 0.1 - 0.1778 SOL |
| 12 | 0.1778 - 0.3162 SOL |
| 13 | 0.3162 - 0.5623 SOL |
| 14 | 0.5623 - 1 SOL |
| 15 | 1 - 1.778 SOL |
| 16 | 1.778 - 3.162 SOL |
| 17 | 3.162 - 5.623 SOL |
| 18 | 5.623 - 10 SOL |
| 19 | 10 - 17.78 SOL |
| 20 | 17.78 - 31.62 SOL |
| 21 | 31.62 - 56.23 SOL |
| 22 | 56.23 - 100 SOL |
| 23 | 100 - 177.8 SOL |
| 24 | 177.8 - 316.2 SOL |
| 25 | 316.2 - 562.3 SOL |
| 26 | 562.3 - 1000 SOL |
| 27 | >= 1000 SOL |

</details>

<details><summary><b>Holders vocabulary (263 ids)</b></summary>

| id | meaning |
|---|---|
| 0 | PAD |
| 1 | UNKNOWN (no balance data) |
| 2 | 0 holders |
| 3 | 1 holder |
| 4 | 2 holders |
| 5 | 3 holders |
| 6 | 4 holders |
| 7 | 5 holders |
| 8 | 6 holders |
| 9 | 7 holders |
| 10 | 8 holders |
| 11 | 9 holders |
| 12 | 10 holders |
| 13 | 11 holders |
| 14 | 12 holders |
| 15 | 13 holders |
| 16 | 14 holders |
| 17 | 15 holders |
| 18 | 16 holders |
| 19 | 17 holders |
| 20 | 18 holders |
| 21 | 19 holders |
| 22 | 20 holders |
| 23 | 21 - 22 holders |
| 24 | 23 holders |
| 25 | 24 holders |
| 26 | 25 holders |
| 27 | 26 holders |
| 28 | 27 - 28 holders |
| 29 | 29 holders |
| 30 | 30 - 31 holders |
| 31 | 32 holders |
| 32 | 33 - 34 holders |
| 33 | 35 holders |
| 34 | 36 - 37 holders |
| 35 | 38 - 39 holders |
| 36 | 40 - 41 holders |
| 37 | 42 - 43 holders |
| 38 | 44 - 45 holders |
| 39 | 46 - 48 holders |
| 40 | 49 - 50 holders |
| 41 | 51 - 53 holders |
| 42 | 54 - 55 holders |
| 43 | 56 - 58 holders |
| 44 | 59 - 61 holders |
| 45 | 62 - 64 holders |
| 46 | 65 - 67 holders |
| 47 | 68 - 71 holders |
| 48 | 72 - 74 holders |
| 49 | 75 - 78 holders |
| 50 | 79 - 82 holders |
| 51 | 83 - 86 holders |
| 52 | 87 - 90 holders |
| 53 | 91 - 95 holders |
| 54 | 96 - 100 holders |
| 55 | 101 - 105 holders |
| 56 | 106 - 110 holders |
| 57 | 111 - 115 holders |
| 58 | 116 - 121 holders |
| 59 | 122 - 127 holders |
| 60 | 128 - 134 holders |
| 61 | 135 - 140 holders |
| 62 | 141 - 147 holders |
| 63 | 148 - 155 holders |
| 64 | 156 - 162 holders |
| 65 | 163 - 171 holders |
| 66 | 172 - 179 holders |
| 67 | 180 - 188 holders |
| 68 | 189 - 198 holders |
| 69 | 199 - 208 holders |
| 70 | 209 - 218 holders |
| 71 | 219 - 229 holders |
| 72 | 230 - 240 holders |
| 73 | 241 - 252 holders |
| 74 | 253 - 265 holders |
| 75 | 266 - 278 holders |
| 76 | 279 - 292 holders |
| 77 | 293 - 307 holders |
| 78 | 308 - 322 holders |
| 79 | 323 - 338 holders |
| 80 | 339 - 355 holders |
| 81 | 356 - 373 holders |
| 82 | 374 - 392 holders |
| 83 | 393 - 411 holders |
| 84 | 412 - 432 holders |
| 85 | 433 - 454 holders |
| 86 | 455 - 476 holders |
| 87 | 477 - 500 holders |
| 88 | 501 - 525 holders |
| 89 | 526 - 551 holders |
| 90 | 552 - 579 holders |
| 91 | 580 - 608 holders |
| 92 | 609 - 638 holders |
| 93 | 639 - 670 holders |
| 94 | 671 - 704 holders |
| 95 | 705 - 739 holders |
| 96 | 740 - 776 holders |
| 97 | 777 - 815 holders |
| 98 | 816 - 856 holders |
| 99 | 857 - 899 holders |
| 100 | 900 - 944 holders |
| 101 | 945 - 991 holders |
| 102 | 992 - 1,040 holders |
| 103 | 1,041 - 1,092 holders |
| 104 | 1,093 - 1,147 holders |
| 105 | 1,148 - 1,204 holders |
| 106 | 1,205 - 1,265 holders |
| 107 | 1,266 - 1,328 holders |
| 108 | 1,329 - 1,394 holders |
| 109 | 1,395 - 1,464 holders |
| 110 | 1,465 - 1,537 holders |
| 111 | 1,538 - 1,614 holders |
| 112 | 1,615 - 1,695 holders |
| 113 | 1,696 - 1,780 holders |
| 114 | 1,781 - 1,869 holders |
| 115 | 1,870 - 1,962 holders |
| 116 | 1,963 - 2,060 holders |
| 117 | 2,061 - 2,163 holders |
| 118 | 2,164 - 2,271 holders |
| 119 | 2,272 - 2,385 holders |
| 120 | 2,386 - 2,504 holders |
| 121 | 2,505 - 2,630 holders |
| 122 | 2,631 - 2,761 holders |
| 123 | 2,762 - 2,899 holders |
| 124 | 2,900 - 3,044 holders |
| 125 | 3,045 - 3,196 holders |
| 126 | 3,197 - 3,356 holders |
| 127 | 3,357 - 3,524 holders |
| 128 | 3,525 - 3,700 holders |
| 129 | 3,701 - 3,885 holders |
| 130 | 3,886 - 4,080 holders |
| 131 | 4,081 - 4,284 holders |
| 132 | 4,285 - 4,498 holders |
| 133 | 4,499 - 4,723 holders |
| 134 | 4,724 - 4,959 holders |
| 135 | 4,960 - 5,207 holders |
| 136 | 5,208 - 5,467 holders |
| 137 | 5,468 - 5,741 holders |
| 138 | 5,742 - 6,028 holders |
| 139 | 6,029 - 6,329 holders |
| 140 | 6,330 - 6,645 holders |
| 141 | 6,646 - 6,978 holders |
| 142 | 6,979 - 7,327 holders |
| 143 | 7,328 - 7,693 holders |
| 144 | 7,694 - 8,078 holders |
| 145 | 8,079 - 8,482 holders |
| 146 | 8,483 - 8,906 holders |
| 147 | 8,907 - 9,351 holders |
| 148 | 9,352 - 9,819 holders |
| 149 | 9,820 - 10,310 holders |
| 150 | 10,311 - 10,825 holders |
| 151 | 10,826 - 11,366 holders |
| 152 | 11,367 - 11,935 holders |
| 153 | 11,936 - 12,531 holders |
| 154 | 12,532 - 13,158 holders |
| 155 | 13,159 - 13,816 holders |
| 156 | 13,817 - 14,507 holders |
| 157 | 14,508 - 15,232 holders |
| 158 | 15,233 - 15,994 holders |
| 159 | 15,995 - 16,793 holders |
| 160 | 16,794 - 17,633 holders |
| 161 | 17,634 - 18,515 holders |
| 162 | 18,516 - 19,441 holders |
| 163 | 19,442 - 20,413 holders |
| 164 | 20,414 - 21,433 holders |
| 165 | 21,434 - 22,505 holders |
| 166 | 22,506 - 23,630 holders |
| 167 | 23,631 - 24,812 holders |
| 168 | 24,813 - 26,052 holders |
| 169 | 26,053 - 27,355 holders |
| 170 | 27,356 - 28,723 holders |
| 171 | 28,724 - 30,159 holders |
| 172 | 30,160 - 31,667 holders |
| 173 | 31,668 - 33,250 holders |
| 174 | 33,251 - 34,913 holders |
| 175 | 34,914 - 36,659 holders |
| 176 | 36,660 - 38,492 holders |
| 177 | 38,493 - 40,416 holders |
| 178 | 40,417 - 42,437 holders |
| 179 | 42,438 - 44,559 holders |
| 180 | 44,560 - 46,787 holders |
| 181 | 46,788 - 49,126 holders |
| 182 | 49,127 - 51,583 holders |
| 183 | 51,584 - 54,162 holders |
| 184 | 54,163 - 56,870 holders |
| 185 | 56,871 - 59,713 holders |
| 186 | 59,714 - 62,699 holders |
| 187 | 62,700 - 65,834 holders |
| 188 | 65,835 - 69,126 holders |
| 189 | 69,127 - 72,582 holders |
| 190 | 72,583 - 76,211 holders |
| 191 | 76,212 - 80,022 holders |
| 192 | 80,023 - 84,023 holders |
| 193 | 84,024 - 88,224 holders |
| 194 | 88,225 - 92,635 holders |
| 195 | 92,636 - 97,267 holders |
| 196 | 97,268 - 102,130 holders |
| 197 | 102,131 - 107,237 holders |
| 198 | 107,238 - 112,599 holders |
| 199 | 112,600 - 118,229 holders |
| 200 | 118,230 - 124,140 holders |
| 201 | 124,141 - 130,347 holders |
| 202 | 130,348 - 136,865 holders |
| 203 | 136,866 - 143,708 holders |
| 204 | 143,709 - 150,893 holders |
| 205 | 150,894 - 158,438 holders |
| 206 | 158,439 - 166,360 holders |
| 207 | 166,361 - 174,678 holders |
| 208 | 174,679 - 183,412 holders |
| 209 | 183,413 - 192,583 holders |
| 210 | 192,584 - 202,212 holders |
| 211 | 202,213 - 212,322 holders |
| 212 | 212,323 - 222,939 holders |
| 213 | 222,940 - 234,085 holders |
| 214 | 234,086 - 245,790 holders |
| 215 | 245,791 - 258,079 holders |
| 216 | 258,080 - 270,983 holders |
| 217 | 270,984 - 284,532 holders |
| 218 | 284,533 - 298,759 holders |
| 219 | 298,760 - 313,697 holders |
| 220 | 313,698 - 329,382 holders |
| 221 | 329,383 - 345,851 holders |
| 222 | 345,852 - 363,144 holders |
| 223 | 363,145 - 381,301 holders |
| 224 | 381,302 - 400,366 holders |
| 225 | 400,367 - 420,384 holders |
| 226 | 420,385 - 441,404 holders |
| 227 | 441,405 - 463,474 holders |
| 228 | 463,475 - 486,647 holders |
| 229 | 486,648 - 510,980 holders |
| 230 | 510,981 - 536,529 holders |
| 231 | 536,530 - 563,355 holders |
| 232 | 563,356 - 591,523 holders |
| 233 | 591,524 - 621,099 holders |
| 234 | 621,100 - 652,154 holders |
| 235 | 652,155 - 684,762 holders |
| 236 | 684,763 - 719,000 holders |
| 237 | 719,001 - 754,950 holders |
| 238 | 754,951 - 792,698 holders |
| 239 | 792,699 - 832,333 holders |
| 240 | 832,334 - 873,949 holders |
| 241 | 873,950 - 917,647 holders |
| 242 | 917,648 - 963,529 holders |
| 243 | 963,530 - 1,011,706 holders |
| 244 | 1,011,707 - 1,062,291 holders |
| 245 | 1,062,292 - 1,115,406 holders |
| 246 | 1,115,407 - 1,171,176 holders |
| 247 | 1,171,177 - 1,229,735 holders |
| 248 | 1,229,736 - 1,291,221 holders |
| 249 | 1,291,222 - 1,355,782 holders |
| 250 | 1,355,783 - 1,423,572 holders |
| 251 | 1,423,573 - 1,494,750 holders |
| 252 | 1,494,751 - 1,569,488 holders |
| 253 | 1,569,489 - 1,647,962 holders |
| 254 | 1,647,963 - 1,730,360 holders |
| 255 | 1,730,361 - 1,816,878 holders |
| 256 | 1,816,879 - 1,907,722 holders |
| 257 | 1,907,723 - 2,003,108 holders |
| 258 | 2,003,109 - 2,103,264 holders |
| 259 | 2,103,265 - 2,208,427 holders |
| 260 | 2,208,428 - 2,318,849 holders |
| 261 | 2,318,850 - 2,434,791 holders |
| 262 | 2,434,792 or more holders |

</details>

<details><summary><b>Top-10 concentration vocabulary (42 ids)</b></summary>

| id | meaning |
|---|---|
| 0 | PAD |
| 1 | UNKNOWN (no balance data) |
| 2 | top 10 hold 0% - 2.5% |
| 3 | top 10 hold 2.5% - 5% |
| 4 | top 10 hold 5% - 7.5% |
| 5 | top 10 hold 7.5% - 10% |
| 6 | top 10 hold 10% - 12.5% |
| 7 | top 10 hold 12.5% - 15% |
| 8 | top 10 hold 15% - 17.5% |
| 9 | top 10 hold 17.5% - 20% |
| 10 | top 10 hold 20% - 22.5% |
| 11 | top 10 hold 22.5% - 25% |
| 12 | top 10 hold 25% - 27.5% |
| 13 | top 10 hold 27.5% - 30% |
| 14 | top 10 hold 30% - 32.5% |
| 15 | top 10 hold 32.5% - 35% |
| 16 | top 10 hold 35% - 37.5% |
| 17 | top 10 hold 37.5% - 40% |
| 18 | top 10 hold 40% - 42.5% |
| 19 | top 10 hold 42.5% - 45% |
| 20 | top 10 hold 45% - 47.5% |
| 21 | top 10 hold 47.5% - 50% |
| 22 | top 10 hold 50% - 52.5% |
| 23 | top 10 hold 52.5% - 55% |
| 24 | top 10 hold 55% - 57.5% |
| 25 | top 10 hold 57.5% - 60% |
| 26 | top 10 hold 60% - 62.5% |
| 27 | top 10 hold 62.5% - 65% |
| 28 | top 10 hold 65% - 67.5% |
| 29 | top 10 hold 67.5% - 70% |
| 30 | top 10 hold 70% - 72.5% |
| 31 | top 10 hold 72.5% - 75% |
| 32 | top 10 hold 75% - 77.5% |
| 33 | top 10 hold 77.5% - 80% |
| 34 | top 10 hold 80% - 82.5% |
| 35 | top 10 hold 82.5% - 85% |
| 36 | top 10 hold 85% - 87.5% |
| 37 | top 10 hold 87.5% - 90% |
| 38 | top 10 hold 90% - 92.5% |
| 39 | top 10 hold 92.5% - 95% |
| 40 | top 10 hold 95% - 97.5% |
| 41 | top 10 hold 97.5% - 100% (or more) |

</details>

<details><summary><b>Trades per minute vocabulary (48 ids)</b></summary>

| id | meaning |
|---|---|
| 0 | PAD |
| 1 | ZERO (no trades in the last minute) |
| 2 | 1 trade/min |
| 3 | 2 trades/min |
| 4 | 3 trades/min |
| 5 | 4 trades/min |
| 6 | 5 trades/min |
| 7 | 6 trades/min |
| 8 | 7 trades/min |
| 9 | 8 trades/min |
| 10 | 9 trades/min |
| 11 | 10 trades/min |
| 12 | 11 - 14 trades/min |
| 13 | 14 - 17 trades/min |
| 14 | 17 - 21 trades/min |
| 15 | 21 - 27 trades/min |
| 16 | 27 - 34 trades/min |
| 17 | 34 - 42 trades/min |
| 18 | 42 - 52 trades/min |
| 19 | 52 - 66 trades/min |
| 20 | 66 - 82 trades/min |
| 21 | 82 - 102 trades/min |
| 22 | 102 - 128 trades/min |
| 23 | 128 - 160 trades/min |
| 24 | 160 - 200 trades/min |
| 25 | 200 - 250 trades/min |
| 26 | 250 - 313 trades/min |
| 27 | 313 - 391 trades/min |
| 28 | 391 - 488 trades/min |
| 29 | 488 - 611 trades/min |
| 30 | 611 - 763 trades/min |
| 31 | 763 - 954 trades/min |
| 32 | 954 - 1,193 trades/min |
| 33 | 1,193 - 1,491 trades/min |
| 34 | 1,491 - 1,863 trades/min |
| 35 | 1,863 - 2,329 trades/min |
| 36 | 2,329 - 2,912 trades/min |
| 37 | 2,912 - 3,640 trades/min |
| 38 | 3,640 - 4,549 trades/min |
| 39 | 4,549 - 5,687 trades/min |
| 40 | 5,687 - 7,109 trades/min |
| 41 | 7,109 - 8,886 trades/min |
| 42 | 8,886 - 11,107 trades/min |
| 43 | 11,107 - 13,884 trades/min |
| 44 | 13,884 - 17,355 trades/min |
| 45 | 17,355 - 21,694 trades/min |
| 46 | 21,694 - 27,117 trades/min |
| 47 | 27,117+ trades/min |

</details>

<details><summary><b>Liquidity vocabulary (74 ids)</b></summary>

| id | meaning |
|---|---|
| 0 | PAD |
| 1 | UNKNOWN (no pool data) |
| 2 | < 0.01 SOL |
| 3 | 0.01 - 0.0126 SOL |
| 4 | 0.0126 - 0.0158 SOL |
| 5 | 0.0158 - 0.02 SOL |
| 6 | 0.02 - 0.0251 SOL |
| 7 | 0.0251 - 0.0316 SOL |
| 8 | 0.0316 - 0.0398 SOL |
| 9 | 0.0398 - 0.0501 SOL |
| 10 | 0.0501 - 0.0631 SOL |
| 11 | 0.0631 - 0.0794 SOL |
| 12 | 0.0794 - 0.1 SOL |
| 13 | 0.1 - 0.126 SOL |
| 14 | 0.126 - 0.158 SOL |
| 15 | 0.158 - 0.2 SOL |
| 16 | 0.2 - 0.251 SOL |
| 17 | 0.251 - 0.316 SOL |
| 18 | 0.316 - 0.398 SOL |
| 19 | 0.398 - 0.501 SOL |
| 20 | 0.501 - 0.631 SOL |
| 21 | 0.631 - 0.794 SOL |
| 22 | 0.794 - 1 SOL |
| 23 | 1 - 1.26 SOL |
| 24 | 1.26 - 1.58 SOL |
| 25 | 1.58 - 2 SOL |
| 26 | 2 - 2.51 SOL |
| 27 | 2.51 - 3.16 SOL |
| 28 | 3.16 - 3.98 SOL |
| 29 | 3.98 - 5.01 SOL |
| 30 | 5.01 - 6.31 SOL |
| 31 | 6.31 - 7.94 SOL |
| 32 | 7.94 - 10 SOL |
| 33 | 10 - 12.6 SOL |
| 34 | 12.6 - 15.8 SOL |
| 35 | 15.8 - 20 SOL |
| 36 | 20 - 25.1 SOL |
| 37 | 25.1 - 31.6 SOL |
| 38 | 31.6 - 39.8 SOL |
| 39 | 39.8 - 50.1 SOL |
| 40 | 50.1 - 63.1 SOL |
| 41 | 63.1 - 79.4 SOL |
| 42 | 79.4 - 100 SOL |
| 43 | 100 - 126 SOL |
| 44 | 126 - 158 SOL |
| 45 | 158 - 200 SOL |
| 46 | 200 - 251 SOL |
| 47 | 251 - 316 SOL |
| 48 | 316 - 398 SOL |
| 49 | 398 - 501 SOL |
| 50 | 501 - 631 SOL |
| 51 | 631 - 794 SOL |
| 52 | 794 - 1,000 SOL |
| 53 | 1,000 - 1,259 SOL |
| 54 | 1,259 - 1,585 SOL |
| 55 | 1,585 - 1,995 SOL |
| 56 | 1,995 - 2,512 SOL |
| 57 | 2,512 - 3,162 SOL |
| 58 | 3,162 - 3,981 SOL |
| 59 | 3,981 - 5,012 SOL |
| 60 | 5,012 - 6,310 SOL |
| 61 | 6,310 - 7,943 SOL |
| 62 | 7,943 - 10,000 SOL |
| 63 | 10,000 - 12,589 SOL |
| 64 | 12,589 - 15,849 SOL |
| 65 | 15,849 - 19,953 SOL |
| 66 | 19,953 - 25,119 SOL |
| 67 | 25,119 - 31,623 SOL |
| 68 | 31,623 - 39,811 SOL |
| 69 | 39,811 - 50,119 SOL |
| 70 | 50,119 - 63,096 SOL |
| 71 | 63,096 - 79,433 SOL |
| 72 | 79,433 - 100,000 SOL |
| 73 | >= 100,000 SOL |

</details>
