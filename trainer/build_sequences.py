#!/usr/bin/env python3
"""Build tokenized model inputs: 120 candle tokens, 120 volume tokens, 120 holder tokens.

Reads data/dataset/dataset_30s.csv (from build_dataset.py). For every candle t of every
token, it takes the window of the last 120 candles (t-119 .. t), converts each candle into
three tokens and stores them with that row's targets.

Writes:
  data/dataset/sequences_30s.npz   arrays ready for a PyTorch / JAX dataloader
  MODEL.md                         the model input specification + vocabularies

Examples:
  python build_sequences.py
  python build_sequences.py --min-history 30 --stride 2
  python build_sequences.py --doc-only        # just (re)write MODEL.md
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import tokenizer as tk

log = logging.getLogger("sequences")
HERE = Path(__file__).resolve().parent
STREAMS = {
    "candle_tokens": (tk.candle_tokens, tk.CANDLE_VOCAB, tk.candle_vocab),
    "volume_tokens": (tk.volume_tokens, tk.VOLUME_VOCAB, tk.volume_vocab),
    "holder_tokens": (tk.holder_tokens, tk.HOLDERS_VOCAB, tk.holders_vocab),
}


def build(args) -> dict[str, np.ndarray]:
    df = pd.read_csv(args.dataset, parse_dates=["timestamp"])
    target_cols = [c for c in df.columns if c.startswith("target_")]
    out: dict[str, list] = {k: [] for k in [*STREAMS, *target_cols, "mint", "timestamp", "split", "n_history"]}
    for mint, g in df.groupby("mint", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        ends = np.arange(args.min_history - 1, len(g), args.stride)
        if len(ends) == 0:
            continue
        for name, (fn, _, _) in STREAMS.items():
            out[name].append(tk.windows(fn(g), ends, args.seq_len))
        for c in target_cols:
            out[c].append(g[c].to_numpy(np.float32)[ends])
        out["mint"].append(np.full(len(ends), mint))
        unix = ((g["timestamp"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy(np.int64)
        out["timestamp"].append(unix[ends])
        out["split"].append(g["split"].to_numpy()[ends])
        out["n_history"].append(np.minimum(ends + 1, args.seq_len).astype(np.int16))
    if not out["mint"]:
        raise SystemExit("no windows produced; download more data or lower --min-history")
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    arrays["mint"] = arrays["mint"].astype(str)
    arrays["split"] = arrays["split"].astype(str)
    return arrays


# ---------------------------------------------------------------------------------
def vocab_table(vocab: list[tuple[int, str]], counts: np.ndarray | None) -> list[str]:
    head = "| id | meaning |" + (" share |" if counts is not None else "")
    rows = [head, "|---|---|" + ("---|" if counts is not None else "")]
    total = counts.sum() if counts is not None else 0
    for i, label in vocab:
        share = f" {counts[i] / total:.2%} |" if counts is not None and total else (" - |" if counts is not None else "")
        rows.append(f"| {i} | {label} |{share}")
    return rows


def decode_example(arrays: dict, idx: int, n: int = 12) -> list[str]:
    vocabs = {k: dict(v[2]()) for k, v in STREAMS.items()}
    L = ["| step | candle token | volume token | holder token |", "|---|---|---|---|"]
    for j in range(args_seq_len - n, args_seq_len):
        cells = []
        for k in STREAMS:
            t = int(arrays[k][idx, j])
            cells.append(f"`{t}` {vocabs[k][t]}")
        L.append(f"| t{j - args_seq_len + 1 if j < args_seq_len - 1 else ''} | " + " | ".join(cells) + " |")
    return L


args_seq_len = tk.SEQ_LEN


def write_markdown(args, arrays: dict | None) -> None:
    L = ["# Model inputs\n"]
    L.append(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `build_sequences.py`. "
             "Tokenization rules live in `dataset/tokenizer.py`.\n")
    L.append("## Overview\n")
    L.append(f"Every training sample is one moment *t* in one token's life. The model sees the last "
             f"**{args.seq_len} candles of 30 seconds** (= {args.seq_len * 30 // 60} minutes of history, "
             f"candles *t-{args.seq_len - 1}* to *t*), as three parallel token sequences:\n")
    L.append("| Input | Shape | dtype | Vocabulary | What one token encodes |")
    L.append("|---|---|---|---|---|")
    L.append(f"| `candle_tokens` | [batch, {args.seq_len}] | int16 | {tk.CANDLE_VOCAB} | price move vs previous close x candle range |")
    L.append(f"| `volume_tokens` | [batch, {args.seq_len}] | int16 | {tk.VOLUME_VOCAB} | SOL volume traded in the candle (log scale) |")
    L.append(f"| `holder_tokens` | [batch, {args.seq_len}] | int16 | {tk.HOLDERS_VOCAB} | number of holders at candle close (log scale) |")
    L.append("")
    L.append("""- Position 0 is the oldest candle and position 119 is the current candle *t*.
- Token id **0 = PAD** in all three streams. A coin younger than 120 candles is
  **left-padded** with PAD, so the current candle is always at position 119. Mask with `tokens != 0`.
- The three streams are aligned: position *i* in each stream describes the same candle.
- Candles are evenly spaced. A 30s period with no swaps is still a step, with the tokens
  `NO_TRADE` (candle) and `ZERO` (volume).
- Only data up to the close of candle *t* is used, so there is no look-ahead.
""")
    L.append("## Stream 1: candle tokens (120)\n")
    L.append(f"""Each candle with at least one swap becomes one of {tk.N_RET} x {tk.N_RANGE} = {tk.N_RET * tk.N_RANGE} tokens:

- **move** = close / previous close - 1, split into {tk.N_RET} signed bins with edges at
  ±{", ±".join(f"{e:g}%" for e in tk.RET_EDGES_PCT)}. The bins are wider in the tails, because memecoins move a lot.
- **range** = high / low - 1, split into {tk.N_RANGE} bins with edges at {", ".join(f"{e:g}%" for e in tk.RANGE_EDGES_PCT)}
  (how volatile the bar was inside the 30s).

`token = {tk.CANDLE_OFFSET} + move_bin x {tk.N_RANGE} + range_bin`. A candle with no swaps is `NO_TRADE`.
For the first candle of a coin, "previous close" is its own open.
""")
    L.append("## Stream 2: volume tokens (120)\n")
    L.append(f"""SOL volume of the candle (buys + sells) on a log10 scale, in {tk.VOL_STEP:g}-decade steps
(each bin is ~{10 ** tk.VOL_STEP:.2f}x the previous one) from {10 ** tk.VOL_LOG10_MIN:g} SOL to {10 ** tk.VOL_LOG10_MAX:g} SOL.
Volume is absolute, not normalized per coin, because 0.1 SOL and 100 SOL of volume mean
very different things for a memecoin. A candle with no volume is `ZERO`.
""")
    L.append("## Stream 3: holder tokens (120)\n")
    L.append(f"""Number of wallets holding > 0 tokens at the close of each candle. 1-{tk.HOLDERS_EXACT} holders each have
their own token. Above that, each bin covers {round((tk.HOLDERS_STEP - 1) * 100)}% more holders than the previous one, so growth
from 1,000 to 1,050 holders is still visible. `UNKNOWN` means no balance data was available.
The holder history is rebuilt from on-chain balances (see [DATASET.md](DATASET.md)).
""")
    L.append("## Targets\n")
    L.append("Each sample carries the targets of its current candle *t*, from `build_dataset.py`:\n")
    L.append("| Target | Meaning |\n|---|---|")
    L.append("| `target_up_5m` | **1 if the price is more than 10% higher 5 minutes later** (suggested main target) |")
    L.append("| `target_up_1m`, `target_up_10m` | same, 1 and 10 minutes ahead |")
    L.append("| `target_ret_{1m,5m,10m}` | log return to that horizon (regression) |")
    L.append("| `target_max_ret_*`, `target_min_ret_*` | best and worst price within the horizon (take-profit / stop-loss) |")
    L.append("")
    L.append("Other arrays in the file: `mint`, `timestamp` (unix seconds of candle *t*), `split` "
             "(train/val/test, per token) and `n_history` (real, non-PAD candles in the window).\n")
    L.append("## Suggested model\n")
    L.append(f"""```
candle_tokens ─ Embedding({tk.CANDLE_VOCAB}, d) ─┐
volume_tokens ─ Embedding({tk.VOLUME_VOCAB}, d) ─┼─ sum ─ + positional embedding({args.seq_len}, d)
holder_tokens ─ Embedding({tk.HOLDERS_VOCAB}, d) ┘
      └─> Transformer encoder (causal or bidirectional, PAD masked)
            └─> representation at position {args.seq_len - 1} (current candle)
                  └─> heads: P(up 5m) [BCE], return 5m [MSE], max/min return [MSE]
```

Train on `split == "train"`, tune on `val` and report on `test`. Splits are by coin, so the test
set only has coins the model has never seen.

```python
import numpy as np
d = np.load("data/dataset/sequences_30s.npz")
x = np.stack([d["candle_tokens"], d["volume_tokens"], d["holder_tokens"]], axis=-1)  # [N, {args.seq_len}, 3]
y = d["target_up_5m"]
train = d["split"] == "train"
```
""")
    counts = {}
    if arrays is not None:
        n = len(arrays["mint"])
        L.append("## This build\n")
        L.append("| | |\n|---|---|")
        L.append(f"| File | `{Path(args.out).relative_to(HERE) if Path(args.out).is_relative_to(HERE) else args.out}` |")
        L.append(f"| Samples | {n:,} |")
        L.append(f"| Coins | {len(set(arrays['mint']))} |")
        for s in ("train", "val", "test"):
            m = arrays["split"] == s
            L.append(f"| {s} | {m.sum():,} samples, {len(set(arrays['mint'][m]))} coins |")
        L.append(f"| Min history per sample | {args.min_history} candles |")
        L.append(f"| Samples with full {args.seq_len}-candle history | {(arrays['n_history'] == args.seq_len).mean():.0%} |")
        if "target_up_5m" in arrays:
            y = arrays["target_up_5m"]
            L.append(f"| `target_up_5m` positive rate | {np.nanmean(y):.1%} |")
        L.append("")
        for k, (_, size, _) in STREAMS.items():
            counts[k] = np.bincount(arrays[k].ravel(), minlength=size)
        # example: most recent full-history sample
        full = np.where(arrays["n_history"] == args.seq_len)[0]
        ex = int(full[-1]) if len(full) else n - 1
        L.append("### Example sample (last 12 of 120 steps)\n")
        L.append(f"Coin `{arrays['mint'][ex]}` at "
                 f"{datetime.fromtimestamp(int(arrays['timestamp'][ex]), timezone.utc):%Y-%m-%d %H:%M:%S} UTC, "
                 f"target_up_5m = {arrays.get('target_up_5m', [None] * n)[ex]}\n")
        L.extend(decode_example(arrays, ex))
        L.append("")

    L.append("## Vocabularies\n")
    L.append("`share` = how often each token appears in this build (PAD included)." if counts else "")
    for k, title in (("candle_tokens", "Candle"), ("volume_tokens", "Volume"), ("holder_tokens", "Holders")):
        vocab = STREAMS[k][2]()
        L.append(f"\n<details><summary><b>{title} vocabulary ({STREAMS[k][1]} ids)</b></summary>\n")
        L.extend(vocab_table(vocab, counts.get(k)))
        L.append("\n</details>")
    L.append("")
    Path(args.md).write_text("\n".join(L))
    log.info("wrote %s", args.md)


def main() -> None:
    global args_seq_len
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(HERE / "data" / "dataset" / "dataset_30s.csv"))
    p.add_argument("--out", default=str(HERE / "data" / "dataset" / "sequences_30s.npz"))
    p.add_argument("--md", default=str(HERE / "MODEL.md"))
    p.add_argument("--seq-len", type=int, default=tk.SEQ_LEN, help="candles per window (default 120)")
    p.add_argument("--min-history", type=int, default=10,
                   help="skip windows with fewer real candles than this (default 10 = 5 minutes)")
    p.add_argument("--stride", type=int, default=1, help="take every n-th candle as a sample")
    p.add_argument("--doc-only", action="store_true", help="only write MODEL.md (no data needed)")
    args = p.parse_args()
    args_seq_len = args.seq_len
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    if args.doc_only:
        return write_markdown(args, None)
    arrays = build(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    log.info("%d samples x 3 x %d tokens -> %s", len(arrays["mint"]), args.seq_len, args.out)
    write_markdown(args, arrays)


if __name__ == "__main__":
    main()
