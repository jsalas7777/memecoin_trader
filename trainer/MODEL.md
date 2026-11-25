# Model inputs

Generated 2026-11-25 00:14 UTC by `build_sequences.py`. Tokenization rules live in `dataset/tokenizer.py`.

## Overview

Every training sample is one moment *t* in one token's life. The model sees the last **120 candles of 30 seconds** (= 60 minutes of history, candles *t-119* to *t*), as three parallel token sequences:

| Input | Shape | dtype | Vocabulary | What one token encodes |
|---|---|---|---|---|
| `candle_tokens` | [batch, 120] | int16 | 86 | price move vs previous close x candle range |
| `volume_tokens` | [batch, 120] | int16 | 28 | SOL volume traded in the candle (log scale) |
| `holder_tokens` | [batch, 120] | int16 | 263 | number of holders at candle close (log scale) |

- Position 0 is the oldest candle and position 119 is the current candle *t*.
- Token id **0 = PAD** in all three streams. A coin younger than 120 candles is
  **left-padded** with PAD, so the current candle is always at position 119. Mask with `tokens != 0`.
- The three streams are aligned: position *i* in each stream describes the same candle.
- Candles are evenly spaced. A 30s period with no swaps is still a step, with the tokens
  `NO_TRADE` (candle) and `ZERO` (volume).
- Only data up to the close of candle *t* is used, so there is no look-ahead.

## Stream 1: candle tokens (120)

Each candle with at least one swap becomes one of 21 x 4 = 84 tokens:

- **move** = close / previous close - 1, split into 21 signed bins with edges at
  ±0.1%, ±0.5%, ±1%, ±2%, ±3%, ±5%, ±8%, ±12%, ±20%, ±35%. The bins are wider in the tails, because memecoins move a lot.
- **range** = high / low - 1, split into 4 bins with edges at 1%, 3%, 8%
  (how volatile the bar was inside the 30s).

`token = 2 + move_bin x 4 + range_bin`. A candle with no swaps is `NO_TRADE`.
For the first candle of a coin, "previous close" is its own open.

## Stream 2: volume tokens (120)

SOL volume of the candle (buys + sells) on a log10 scale, in 0.25-decade steps
(each bin is ~1.78x the previous one) from 0.001 SOL to 1000 SOL.
Volume is absolute, not normalized per coin, because 0.1 SOL and 100 SOL of volume mean
very different things for a memecoin. A candle with no volume is `ZERO`.

## Stream 3: holder tokens (120)

Number of wallets holding > 0 tokens at the close of each candle. 1-20 holders each have
their own token. Above that, each bin covers 5% more holders than the previous one, so growth
from 1,000 to 1,050 holders is still visible. `UNKNOWN` means no balance data was available.
The holder history is rebuilt from on-chain balances (see [DATASET.md](DATASET.md)).

## Targets

Each sample carries the targets of its current candle *t*, from `build_dataset.py`:

| Target | Meaning |
|---|---|
| `target_up_5m` | **1 if the price is more than 10% higher 5 minutes later** (suggested main target) |
| `target_up_1m`, `target_up_10m` | same, 1 and 10 minutes ahead |
| `target_ret_{1m,5m,10m}` | log return to that horizon (regression) |
| `target_max_ret_*`, `target_min_ret_*` | best and worst price within the horizon (take-profit / stop-loss) |

Other arrays in the file: `mint`, `timestamp` (unix seconds of candle *t*), `split` (train/val/test, per token) and `n_history` (real, non-PAD candles in the window).

## Suggested model

```
candle_tokens ─ Embedding(86, d) ─┐
volume_tokens ─ Embedding(28, d) ─┼─ sum ─ + positional embedding(120, d)
holder_tokens ─ Embedding(263, d) ┘
      └─> Transformer encoder (causal or bidirectional, PAD masked)
            └─> representation at position 119 (current candle)
                  └─> heads: P(up 5m) [BCE], return 5m [MSE], max/min return [MSE]
```

Train on `split == "train"`, tune on `val` and report on `test`. Splits are by coin, so the test
set only has coins the model has never seen.

```python
import numpy as np
d = np.load("data/dataset/sequences_30s.npz")
x = np.stack([d["candle_tokens"], d["volume_tokens"], d["holder_tokens"]], axis=-1)  # [N, 120, 3]
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
