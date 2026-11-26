# Model inputs

> [!IMPORTANT]
> ## ⭐ MODEL INPUTS: 4 sequences × 120 tokens
> Each sample = the last **120 candles of 30 seconds** (**60 minutes**) of one coin. Every candle becomes **one token in each of the 4 inputs**.
>
> | # | Input | Shape | Vocab | One token = | Built from (dataset column) |
> |---|---|---|---|---|---|
> | **1** | **`candle_tokens`** | **[batch, 120]** | 86 | **Candle**: price move vs previous close x candle range | `open`, `high`, `low`, `close`, `volume_sol` |
> | **2** | **`volume_tokens`** | **[batch, 120]** | 28 | **Volume**: SOL volume traded in the candle (log scale) | `volume_sol` |
> | **3** | **`holder_tokens`** | **[batch, 120]** | 263 | **Holders**: number of holders at candle close (log scale) | `holders` |
> | **4** | **`curve_tokens`** | **[batch, 120]** | 43 | **Bonding curve progress**: % of the pump.fun curve sold, or GRADUATED | `curve_progress_pct`, `graduated` |
>
> **Input tensor:** `x = [batch, 120, 4]` int16 token ids (`0` = PAD). Position 119 = the current candle *t*.
>
> **NOT inputs:** `target_*` (what to predict) and `outcome_*` (each coin's fate). Both contain the future. Feeding them to the model is look-ahead leakage.

Generated 2026-11-26 21:14 UTC by `build_sequences.py`. Tokenization rules live in `dataset/tokenizer.py`.

## Overview

- Position 0 is the oldest candle and position 119 is the current candle *t*.
- Token id **0 = PAD** in every stream. A coin younger than 120 candles is
  **left-padded** with PAD, so the current candle is always at position 119. Mask with `tokens != 0`.
- All streams are aligned: position *i* in every stream describes the same candle.
- Candles are evenly spaced. A 30s period with no swaps is still a step, with the tokens
  `NO_TRADE` (candle) and `ZERO` (volume).
- Only data up to the close of candle *t* is used, so there is no look-ahead.
- Candles come from GMGN (30s klines from launch) and/or on-chain swaps; see the `source` column.
  A value no source has data for is the stream's `UNKNOWN` token, never a guess.

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
from 1,000 to 1,050 holders is still visible. `UNKNOWN` means no holder history is available.
Holder history comes from exact on-chain balances, or else from GMGN wallet holding periods
(exact for coins with fewer than 100 wallets, which covers most dead coins). See [DATASET.md](DATASET.md).

## ⭐ Input 4: `curve_tokens` [batch, 120]

How far the coin is along the pump.fun bonding curve at candle close, in 2.5% steps
(40 bins). 0% = just launched; 100% = all 793.1M sellable tokens bought and the curve is complete.
`GRADUATED` = the coin has moved to an AMM (PumpSwap, Raydium, ...). Many coins stall and die
just before completing the curve, and graduation itself often triggers a dump.
On-chain: exact, from the tokens left in the curve's vault (1B at launch, 206.9M at completion).
GMGN: from the price, using the curve's constant-product formula (within ~1 point), and
`GRADUATED` from GMGN's migration time. `UNKNOWN` = not a pump.fun coin.

## Inputs switched off

These streams exist in the code but are **not given to the model**: `top10_tokens` (Top-10 concentration), `trades_tokens` (Trades per minute), `liquidity_tokens` (Liquidity), `bundle_tokens` (Launch bundle), `whale_buy_tokens` (Whale buy), `whale_sell_tokens` (Whale sell). They need on-chain swap data, which the keyless GMGN download (`download_free.py`) doesn't have, so they would be almost all `UNKNOWN`. To use them again, download on-chain data (`download_data.py history`) and add their names to `MODEL_STREAMS` in `dataset/tokenizer.py`.

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

## ⭐ The model: causal transformer, 24-candle forecast (`train.py`)

> [!IMPORTANT]
> **Sequential training from the very first candle.** The model is causal (GPT-style): at candle *t* it
> only sees candles 0..*t* of the coin. Every candle is a training example: candle 0 trains with
> 1 candle of history, candle 1 with 2, and so on. **Nothing waits for 120 candles.** Empty
> or `UNKNOWN` inputs are ordinary tokens, so a coin trains from its first 30 seconds on.
>
> **Output: the next 24 candles (12 minutes) at every candle.**

```
per candle t, one token from each of the 4 inputs ── Embedding per input ──┐
coin age since launch (64 buckets) ──────────────────── Embedding ─────────────┼─ sum
position in the window (0..119) ─────────────────────── Embedding ─────────────┘
   └─> causal Transformer (default: d_model 128, 4 layers, 4 heads, ~1.3M parameters)
         └─> at EVERY position t: 3 heads, each forecasting candles t+1 .. t+24
               candle [24 x 86]  which candle token comes next (move x range, or NO_TRADE)
               volume [24 x 28]  its volume token
               ret    [24]       log(close[t+k] / close[t]): the price path from now
```

| Output | Shape | Loss | Meaning |
|---|---|---|---|
| `candle` | [batch, T, 24, 86] | cross-entropy | probabilities of each candle token for each of the next 24 candles |
| `volume` | [batch, T, 24, 28] | cross-entropy (x0.5) | volume token of each of the next 24 candles |
| `ret` | [batch, T, 24] | Huber | predicted price change from now to each of the next 24 candles (log ratio) |

**How coins are fed.** Each coin (each `segment`) is one sequence starting at its first candle.
Coins longer than 120 candles are cut into overlapping windows (120 long, step 60); the first
window trains every position, later windows only the positions the previous one didn't reach,
so each candle is trained exactly once per epoch, with at least 60 candles of history once
it is past the first window. Future candles beyond the downloaded data are masked out of the loss.
Dead coins keep 24 empty candles after their last trade, so "it dies" is a forecast the model learns.

```bash
python train.py                  # trains on data/dataset/dataset_30s.csv, best model -> data/models/transformer_30s.pt
python train.py --quick          # 2-epoch smoke test
python train.py --epochs 50 --d-model 256 --layers 6
python predict.py                # print a 24-candle forecast for a test coin vs what actually happened
```

Train on `split == "train"`, early-stop on `val`, report on `test`. Splits are by coin, so the test
set only has coins the model has never seen. Each epoch prints next-candle accuracy, the 5-minute
return error next to the "price doesn't move" baseline, and direction accuracy on moves > 5%.
The model is only useful where it beats that baseline on `test`.

**Dead coins are in the data on purpose.** Most pump.fun launches die within minutes, so most
forecasts are "flat / NO_TRADE / down". That is the real distribution the bot will trade in; keep
it. When evaluating, check `outcome_dead` for the coins the model would have bought.

`sequences_30s.npz` (fixed 120-candle windows, verified above) is still written for other
model types:

```python
import numpy as np
d = np.load("data/dataset/sequences_30s.npz")
x = np.stack([d[k] for k in ['candle_tokens', 'volume_tokens', 'holder_tokens', 'curve_tokens']], axis=-1)  # [N, 120, 4]
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

<details><summary><b>Bonding curve progress vocabulary (43 ids)</b></summary>

| id | meaning |
|---|---|
| 0 | PAD |
| 1 | UNKNOWN (not a pump.fun coin, or no data) |
| 2 | GRADUATED (left the bonding curve, trades on an AMM) |
| 3 | curve 0% - 2.5% complete |
| 4 | curve 2.5% - 5% complete |
| 5 | curve 5% - 7.5% complete |
| 6 | curve 7.5% - 10% complete |
| 7 | curve 10% - 12.5% complete |
| 8 | curve 12.5% - 15% complete |
| 9 | curve 15% - 17.5% complete |
| 10 | curve 17.5% - 20% complete |
| 11 | curve 20% - 22.5% complete |
| 12 | curve 22.5% - 25% complete |
| 13 | curve 25% - 27.5% complete |
| 14 | curve 27.5% - 30% complete |
| 15 | curve 30% - 32.5% complete |
| 16 | curve 32.5% - 35% complete |
| 17 | curve 35% - 37.5% complete |
| 18 | curve 37.5% - 40% complete |
| 19 | curve 40% - 42.5% complete |
| 20 | curve 42.5% - 45% complete |
| 21 | curve 45% - 47.5% complete |
| 22 | curve 47.5% - 50% complete |
| 23 | curve 50% - 52.5% complete |
| 24 | curve 52.5% - 55% complete |
| 25 | curve 55% - 57.5% complete |
| 26 | curve 57.5% - 60% complete |
| 27 | curve 60% - 62.5% complete |
| 28 | curve 62.5% - 65% complete |
| 29 | curve 65% - 67.5% complete |
| 30 | curve 67.5% - 70% complete |
| 31 | curve 70% - 72.5% complete |
| 32 | curve 72.5% - 75% complete |
| 33 | curve 75% - 77.5% complete |
| 34 | curve 77.5% - 80% complete |
| 35 | curve 80% - 82.5% complete |
| 36 | curve 82.5% - 85% complete |
| 37 | curve 85% - 87.5% complete |
| 38 | curve 87.5% - 90% complete |
| 39 | curve 90% - 92.5% complete |
| 40 | curve 92.5% - 95% complete |
| 41 | curve 95% - 97.5% complete |
| 42 | curve 97.5% - 100% complete |

</details>
