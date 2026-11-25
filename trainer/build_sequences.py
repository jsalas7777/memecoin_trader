#!/usr/bin/env python3
"""Build tokenized model inputs: 10 streams x 120 tokens.

Streams: candle, volume, holders, top-10 concentration, trades per minute, liquidity,
bonding curve progress, launch bundle, whale buy, whale sell.

Reads data/dataset/dataset_30s.csv (from build_dataset.py). For every candle t of every
token, it takes the window of the last 120 candles (t-119 .. t), converts each candle into
one token per stream and stores them with that row's targets.

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
from dataset.verify import Report, verify

log = logging.getLogger("sequences")
HERE = Path(__file__).resolve().parent
STREAMS = {
    "candle_tokens": (tk.candle_tokens, tk.CANDLE_VOCAB, tk.candle_vocab),
    "volume_tokens": (tk.volume_tokens, tk.VOLUME_VOCAB, tk.volume_vocab),
    "holder_tokens": (tk.holder_tokens, tk.HOLDERS_VOCAB, tk.holders_vocab),
    "top10_tokens": (tk.top10_tokens, tk.TOP10_VOCAB, tk.top10_vocab),
    "trades_tokens": (tk.trades_tokens, tk.TRADES_VOCAB, tk.trades_vocab),
    "liquidity_tokens": (tk.liquidity_tokens, tk.LIQ_VOCAB, tk.liquidity_vocab),
    "curve_tokens": (tk.curve_tokens, tk.CURVE_VOCAB, tk.curve_vocab),
    "bundle_tokens": (tk.bundle_tokens, tk.BUNDLE_VOCAB, tk.bundle_vocab),
    "whale_buy_tokens": (tk.whale_buy_tokens, tk.WHALE_VOCAB, tk.whale_buy_vocab),
    "whale_sell_tokens": (tk.whale_sell_tokens, tk.WHALE_VOCAB, tk.whale_sell_vocab),
}
# stream -> source columns in dataset_30s.csv
STREAM_SOURCES = {
    "candle_tokens": "`open`, `high`, `low`, `close`, `volume_sol`",
    "volume_tokens": "`volume_sol`",
    "holder_tokens": "`holders`",
    "top10_tokens": "`top10_pct`",
    "trades_tokens": "`trades_per_min`",
    "liquidity_tokens": "`liquidity_sol`",
    "curve_tokens": "`curve_progress_pct`, `graduated`",
    "bundle_tokens": "`bundle_pct`, `bundle_launch_pct`",
    "whale_buy_tokens": "`max_buy_sol`",
    "whale_sell_tokens": "`max_sell_sol`",
}
# stream -> (title, what one token encodes)
STREAM_DOCS = {
    "candle_tokens": ("Candle", "price move vs previous close x candle range"),
    "volume_tokens": ("Volume", "SOL volume traded in the candle (log scale)"),
    "holder_tokens": ("Holders", "number of holders at candle close (log scale)"),
    "top10_tokens": ("Top-10 concentration", "% of supply held by the 10 largest wallets"),
    "trades_tokens": ("Trades per minute", "swaps in the last 60 seconds"),
    "liquidity_tokens": ("Liquidity", "total SOL in the coin's pools (log scale)"),
    "curve_tokens": ("Bonding curve progress", "% of the pump.fun curve sold, or GRADUATED"),
    "bundle_tokens": ("Launch bundle", "% of supply the launch-block wallets still hold"),
    "whale_buy_tokens": ("Whale buy", "biggest buy by one wallet in the candle (SOL, log scale)"),
    "whale_sell_tokens": ("Whale sell", "biggest sell by one wallet in the candle (SOL, log scale)"),
}


def build(args) -> dict[str, np.ndarray]:
    df = pd.read_csv(args.dataset, parse_dates=["timestamp"])
    target_cols = [c for c in df.columns if c.startswith(("target_", "outcome_"))]
    out: dict[str, list] = {k: [] for k in [*STREAMS, *target_cols, "mint", "segment", "timestamp", "split", "n_history"]}
    if "segment" not in df.columns:
        df["segment"] = 0
    for (mint, segment), g in df.groupby(["mint", "segment"], sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        ends = np.arange(args.min_history - 1, len(g), args.stride)
        if len(ends) == 0:
            continue
        for name, (fn, _, _) in STREAMS.items():
            out[name].append(tk.windows(fn(g), ends, args.seq_len))
        for c in target_cols:
            out[c].append(g[c].to_numpy(np.float32)[ends])
        out["mint"].append(np.full(len(ends), mint))
        out["segment"].append(np.full(len(ends), segment, dtype=np.int16))
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
    L = ["| step | " + " | ".join(STREAM_DOCS[k][0] for k in STREAMS) + " |",
         "|---|" + "---|" * len(STREAMS)]
    for j in range(args_seq_len - n, args_seq_len):
        cells = []
        for k in STREAMS:
            t = int(arrays[k][idx, j])
            cells.append(f"`{t}` {vocabs[k][t]}")
        L.append(f"| t{j - args_seq_len + 1 if j < args_seq_len - 1 else ''} | " + " | ".join(cells) + " |")
    return L


args_seq_len = tk.SEQ_LEN


def write_markdown(args, arrays: dict | None, report: Report | None = None) -> None:
    L = ["# Model inputs\n"]
    L.append("> [!IMPORTANT]")
    L.append(f"> ## ⭐ MODEL INPUTS: {len(STREAMS)} sequences × {args.seq_len} tokens")
    L.append(f"> Each sample = the last **{args.seq_len} candles of 30 seconds** (**{args.seq_len * 30 // 60} minutes**) "
             f"of one coin. Every candle becomes **one token in each of the {len(STREAMS)} inputs**.")
    L.append(">")
    L.append("> | # | Input | Shape | Vocab | One token = | Built from (dataset column) |")
    L.append("> |---|---|---|---|---|---|")
    for i, (k, (_, size, _)) in enumerate(STREAMS.items(), 1):
        L.append(f"> | **{i}** | **`{k}`** | **[batch, {args.seq_len}]** | {size} | "
                 f"**{STREAM_DOCS[k][0]}**: {STREAM_DOCS[k][1]} | {STREAM_SOURCES[k]} |")
    L.append(">")
    L.append(f"> **Input tensor:** `x = [batch, {args.seq_len}, {len(STREAMS)}]` int16 token ids "
             f"(`0` = PAD). Position {args.seq_len - 1} = the current candle *t*.")
    L.append(">")
    L.append("> **NOT inputs:** `target_*` (what to predict) and `outcome_*` (each coin's fate). "
             "Both contain the future. Feeding them to the model is look-ahead leakage.")
    if report is not None:
        status = "❌ FAILED" if report.failed else "✅ PASSED"
        L.append(">")
        L.append(f"> **Input verification: {status}.** `check_inputs.py` compared every input against the "
                 "dataset in this build (see [Input verification](#input-verification)).")
    L.append("")
    L.append(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `build_sequences.py`. "
             "Tokenization rules live in `dataset/tokenizer.py`.\n")
    L.append("## Overview\n")
    L.append("""- Position 0 is the oldest candle and position 119 is the current candle *t*.
- Token id **0 = PAD** in every stream. A coin younger than 120 candles is
  **left-padded** with PAD, so the current candle is always at position 119. Mask with `tokens != 0`.
- All streams are aligned: position *i* in every stream describes the same candle.
- Candles are evenly spaced. A 30s period with no swaps is still a step, with the tokens
  `NO_TRADE` (candle) and `ZERO` (volume).
- Only data up to the close of candle *t* is used, so there is no look-ahead.
- Candles come from GMGN (30s klines from launch) and/or on-chain swaps; see the `source` column.
  A value no source has data for is the stream's `UNKNOWN` token, never a guess.
""")
    L.append("## ⭐ Input 1: `candle_tokens` [batch, 120]\n")
    L.append(f"""Each candle with at least one swap becomes one of {tk.N_RET} x {tk.N_RANGE} = {tk.N_RET * tk.N_RANGE} tokens:

- **move** = close / previous close - 1, split into {tk.N_RET} signed bins with edges at
  ±{", ±".join(f"{e:g}%" for e in tk.RET_EDGES_PCT)}. The bins are wider in the tails, because memecoins move a lot.
- **range** = high / low - 1, split into {tk.N_RANGE} bins with edges at {", ".join(f"{e:g}%" for e in tk.RANGE_EDGES_PCT)}
  (how volatile the bar was inside the 30s).

`token = {tk.CANDLE_OFFSET} + move_bin x {tk.N_RANGE} + range_bin`. A candle with no swaps is `NO_TRADE`.
For the first candle of a coin, "previous close" is its own open.
""")
    L.append("## ⭐ Input 2: `volume_tokens` [batch, 120]\n")
    L.append(f"""SOL volume of the candle (buys + sells) on a log10 scale, in {tk.VOL_STEP:g}-decade steps
(each bin is ~{10 ** tk.VOL_STEP:.2f}x the previous one) from {10 ** tk.VOL_LOG10_MIN:g} SOL to {10 ** tk.VOL_LOG10_MAX:g} SOL.
Volume is absolute, not normalized per coin, because 0.1 SOL and 100 SOL of volume mean
very different things for a memecoin. A candle with no volume is `ZERO`.
""")
    L.append("## ⭐ Input 3: `holder_tokens` [batch, 120]\n")
    L.append(f"""Number of wallets holding > 0 tokens at the close of each candle. 1-{tk.HOLDERS_EXACT} holders each have
their own token. Above that, each bin covers {round((tk.HOLDERS_STEP - 1) * 100)}% more holders than the previous one, so growth
from 1,000 to 1,050 holders is still visible. `UNKNOWN` means no holder history is available.
Holder history comes from exact on-chain balances, or else from GMGN wallet holding periods
(exact for coins with fewer than 100 wallets, which covers most dead coins). See [DATASET.md](DATASET.md).
""")
    L.append("## ⭐ Input 4: `top10_tokens` [batch, 120]\n")
    L.append(f"""% of the total supply held by the 10 largest wallets at candle close, in {tk.TOP10_STEP_PCT:g}% steps
({tk.TOP10_BINS} bins from 0% to 100%). Pools, bonding curves and other program-owned (PDA) accounts are
excluded, because they're liquidity, not people who can dump. A high or rising value means a few
wallets can crash the price. A falling value means big holders are selling or distributing.
`UNKNOWN` means no balance data was available. This stream needs on-chain data, because GMGN
has no top-10 history.
""")
    L.append("## ⭐ Input 5: `trades_tokens` [batch, 120]\n")
    L.append(f"""Number of swaps in the last 60 seconds (the current 30s candle plus the previous one).
1-{tk.TRADES_EXACT} trades/min each have their own token. Above that, each bin covers {round((tk.TRADES_STEP - 1) * 100)}% more
trades/min than the previous one. `ZERO` = no trades in the last minute (activity is dying).
`UNKNOWN` = the trade count isn't available (GMGN candles have volume but no trade count; it is
filled from on-chain data where that exists).
""")
    L.append("## ⭐ Input 6: `liquidity_tokens` [batch, 120]\n")
    L.append(f"""Total SOL in all of the coin's pools at candle close (pump.fun bonding curve + PumpSwap /
Raydium / Meteora pools), on a log10 scale in {tk.LIQ_STEP:g}-decade steps (each bin ~{10 ** tk.LIQ_STEP:.2f}x the
previous) from {10 ** tk.LIQ_LOG10_MIN:g} SOL to {10 ** tk.LIQ_LOG10_MAX:,.0f} SOL. When a coin graduates, the emptied
bonding curve stops counting, so liquidity is not double-counted. A sudden drop in this
stream is liquidity being pulled. This stream needs on-chain data (GMGN has no liquidity
history); it is `UNKNOWN` otherwise.
""")
    L.append("## ⭐ Input 7: `curve_tokens` [batch, 120]\n")
    L.append(f"""How far the coin is along the pump.fun bonding curve at candle close, in {tk.CURVE_STEP_PCT:g}% steps
({tk.CURVE_BINS} bins). 0% = just launched; 100% = all 793.1M sellable tokens bought and the curve is complete.
`GRADUATED` = the coin has moved to an AMM (PumpSwap, Raydium, ...). Many coins stall and die
just before completing the curve, and graduation itself often triggers a dump.
On-chain: exact, from the tokens left in the curve's vault (1B at launch, 206.9M at completion).
GMGN: from the price, using the curve's constant-product formula (within ~1 point), and
`GRADUATED` from GMGN's migration time. `UNKNOWN` = not a pump.fun coin.
""")
    L.append("## ⭐ Input 8: `bundle_tokens` [batch, 120]\n")
    L.append(f"""% of the total supply still held by the wallets that bought in the **launch block** (the same
slot as the create transaction: the dev's own buy plus any wallets bundled with it), at candle close.
1% steps up to 20%, then 5% steps. A big bundle means a few insiders control the supply; the
stream falling means they are selling into buyers.
`NO_BUNDLE` = nobody bought in the launch block. `SOLD_OUT` = they bought and have sold
(less than {tk.BUNDLE_SOLD_OUT_PCT:g}% of supply left). `UNKNOWN` = the downloaded history doesn't reach back to the
launch, or the candles are GMGN-only (this stream needs on-chain data from creation).
Program-owned (PDA) accounts such as the curve itself are never counted.
""")
    L.append("## ⭐ Inputs 9 and 10: `whale_buy_tokens`, `whale_sell_tokens` [batch, 120]\n")
    L.append(f"""The biggest amount of SOL **one wallet** bought (input 9) and sold (input 10) in the candle.
A wallet's swaps in the same candle are added up, so splitting a buy into small swaps doesn't
hide it. Same log scale as volume ({10 ** tk.VOL_LOG10_MIN:g} SOL to {10 ** tk.VOL_LOG10_MAX:g} SOL, {tk.VOL_STEP:g}-decade steps).
Together with volume, it shows whether a move is one whale or many small traders; a whale
sell next to thin liquidity is a dump warning. `NONE` = no buys (sells) in the candle.
`UNKNOWN` = per-wallet amounts not available (GMGN candles have no trade list; on-chain fills it).
""")
    L.append("## Targets\n")
    L.append("Each sample carries the targets of its current candle *t*, from `build_dataset.py`:\n")
    L.append("| Target | Meaning |\n|---|---|")
    L.append("| `target_up_5m` | **1 if the price is more than 10% higher 5 minutes later** (suggested main target) |")
    L.append("| `target_up_1m`, `target_up_10m` | same, 1 and 10 minutes ahead |")
    L.append("| `target_ret_{1m,5m,10m}` | log return to that horizon (regression) |")
    L.append("| `target_dump_{1m,5m,10m}` | **1 if the price falls more than 30% at any point within the horizon** (avoid / exit signal) |")
    L.append("| `target_max_ret_*`, `target_min_ret_*` | best and worst price within the horizon (take-profit / stop-loss) |")
    L.append("| `outcome_*` | per-coin fate: dead, graduated, ATH market cap, final drawdown, lifetime. Labels only, never inputs (see DATASET.md) |")
    L.append("")
    L.append("Other arrays in the file: `mint`, `segment`, `timestamp` (unix seconds of candle *t*), `split` "
             "(train/val/test, per token) and `n_history` (real, non-PAD candles in the window). "
             "Windows never span a gap in the downloaded data: a new `segment` starts with fresh padding.\n")
    L.append("## Suggested model\n")
    L.append(f"""```
candle_tokens    ─ Embedding({tk.CANDLE_VOCAB}, d) ─┐
volume_tokens    ─ Embedding({tk.VOLUME_VOCAB}, d) ─┤
holder_tokens    ─ Embedding({tk.HOLDERS_VOCAB}, d) ┤
top10_tokens     ─ Embedding({tk.TOP10_VOCAB}, d) ─┼─ sum ─ + positional embedding({args.seq_len}, d)
trades_tokens    ─ Embedding({tk.TRADES_VOCAB}, d) ─┤
liquidity_tokens ─ Embedding({tk.LIQ_VOCAB}, d) ─┤
curve_tokens     ─ Embedding({tk.CURVE_VOCAB}, d) ─┤
bundle_tokens    ─ Embedding({tk.BUNDLE_VOCAB}, d) ─┤
whale_buy_tokens ─ Embedding({tk.WHALE_VOCAB}, d) ─┤
whale_sell_tokens─ Embedding({tk.WHALE_VOCAB}, d) ─┘
      └─> Transformer encoder (causal or bidirectional, PAD masked)
            └─> representation at position {args.seq_len - 1} (current candle)
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
           "top10_tokens", "trades_tokens", "liquidity_tokens",
           "curve_tokens", "bundle_tokens", "whale_buy_tokens", "whale_sell_tokens"]
x = np.stack([d[k] for k in streams], axis=-1)  # [N, {args.seq_len}, {len(STREAMS)}]
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

    if report is not None:
        L.append("## Input verification\n")
        L.append("Run automatically by `build_sequences.py` (and on demand with `python check_inputs.py`). "
                 "Each input is rebuilt independently from `dataset_30s.csv` and compared token-for-token.\n")
        L.append("| Status | Check | Detail |\n|---|---|---|")
        icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
        for st, name, detail in report.lines:
            L.append(f"| {icon[st]} {st} | {name} | {detail} |")
        L.append("")
    L.append("## Vocabularies\n")
    L.append("`share` = how often each token appears in this build (PAD included)." if counts else "")
    for k in STREAMS:
        title = STREAM_DOCS[k][0]
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
    p.add_argument("--min-history", type=int, default=1,
                   help="skip windows with fewer real candles than this (default 1: include the first "
                        "minutes of every coin, where most dead coins die)")
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
    log.info("%d samples x %d streams x %d tokens -> %s", len(arrays["mint"]), len(STREAMS), args.seq_len, args.out)
    log.info("verifying model inputs ...")
    report = verify(args.out, args.dataset, seq_len=args.seq_len)
    report.print()
    write_markdown(args, arrays, report)
    if report.failed:
        raise SystemExit("input verification FAILED, see MODEL.md / output above")


if __name__ == "__main__":
    main()
