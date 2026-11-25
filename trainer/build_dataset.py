#!/usr/bin/env python3
"""Build the ML dataset from downloaded on-chain data.

Reads data/trades + data/balances (from download_data.py), builds 30-second
candles with holder counts, engineers features and forward-looking targets,
and writes:
  data/dataset/dataset_<interval>.csv
  DATASET.md   (description of every input / target + stats of this build)

Examples:
  python build_dataset.py
  python build_dataset.py --interval 30s --horizons 1min,5min,10min --up-threshold 0.10
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dataset.features import FEATURES, ID_COLUMNS, make_features, make_targets, target_docs
from solana_data.ohlcv import build_ohlcv

log = logging.getLogger("dataset")
HERE = Path(__file__).resolve().parent


def assign_split(mint: str, val: float, test: float) -> str:
    """Deterministic per-token split so no token leaks between train and test."""
    x = int(hashlib.md5(mint.encode()).hexdigest(), 16) % 10_000 / 10_000
    return "test" if x < test else "val" if x < test + val else "train"


def parse_horizons(spec: str, interval: str) -> dict[str, int]:
    step = pd.Timedelta(interval)
    out = {}
    for h in spec.split(","):
        n = pd.Timedelta(h) / step
        if n < 1 or n != int(n):
            raise SystemExit(f"horizon {h} must be a whole multiple of the interval {interval}")
        out[h.replace("min", "m")] = int(n)
    return out


def build(args) -> tuple[pd.DataFrame, dict]:
    data = Path(args.data_dir)
    horizons = parse_horizons(args.horizons, args.interval)
    frames, skipped = [], []
    for trades_path in sorted((data / "trades").glob("*.csv")):
        mint = trades_path.stem
        trades = pd.read_csv(trades_path)
        if len(trades) < args.min_trades:
            skipped.append((mint, f"{len(trades)} trades"))
            continue
        state_path = data / "state" / f"{mint}.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        bal_path = data / "balances" / f"{mint}.csv"
        balances = pd.read_csv(bal_path) if bal_path.exists() else None

        candles = build_ohlcv(trades, args.interval, supply=state.get("supply"), balances=balances)
        if len(candles) <= max(horizons.values()):
            skipped.append((mint, f"only {len(candles)} candles"))
            continue
        f = make_targets(make_features(candles), horizons, args.up_threshold)
        f.insert(0, "mint", mint)
        f.insert(1, "symbol", state.get("symbol") or "")
        f["split"] = assign_split(mint, args.val, args.test)
        frames.append(f)
        log.info("%-10s %6d candles  %6d trades  holders=%s", (state.get("symbol") or mint[:8])[:10],
                 len(f), len(trades), int(f["holders"].iloc[-1]) if f["holders"].notna().any() else "n/a")

    for mint, why in skipped:
        log.info("skipped %s (%s)", mint, why)
    if not frames:
        raise SystemExit("no tokens with enough data; download more with download_data.py")

    df = pd.concat(frames, ignore_index=True)
    target_cols = [c for c in df.columns if c.startswith("target_")]
    if not args.keep_incomplete:
        # rows at the end of each series have no future yet -> unusable for training
        df = df.dropna(subset=[f"target_ret_{max(horizons, key=horizons.get)}"])
    cols = list(ID_COLUMNS) + list(FEATURES) + target_cols
    df = df[cols]
    meta = {"horizons": horizons, "skipped": skipped, "target_cols": target_cols}
    return df, meta


def fmt(x) -> str:
    if pd.isna(x):
        return "-"
    if isinstance(x, (int,)) or float(x).is_integer() and abs(x) < 1e12:
        return f"{int(x):,}"
    return f"{x:.4g}"


def write_markdown(args, df: pd.DataFrame, meta: dict, out_csv: Path) -> None:
    tdocs = target_docs(meta["horizons"], args.up_threshold)
    L = []
    L.append("# Memecoin dataset\n")
    L.append(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} by `build_dataset.py` "
             f"from on-chain Solana data downloaded with `download_data.py`.\n")
    L.append("## Build summary\n")
    split_counts = df.groupby("split")["mint"].agg(["nunique", "size"])
    L.append("| | |\n|---|---|")
    L.append(f"| File | `{out_csv.relative_to(HERE) if out_csv.is_relative_to(HERE) else out_csv}` |")
    L.append(f"| Candle timeframe | **{args.interval}** |")
    L.append(f"| Tokens | {df['mint'].nunique()} |")
    L.append(f"| Rows (candles) | {len(df):,} |")
    L.append(f"| Time range | {df['timestamp'].min()} to {df['timestamp'].max()} |")
    for split, row in split_counts.iterrows():
        L.append(f"| {split} | {row['nunique']} tokens, {row['size']:,} rows |")
    L.append(f"| Input features | {len(FEATURES)} |")
    L.append(f"| Targets | {len(meta['target_cols'])} |")
    if meta["target_cols"]:
        up = [c for c in meta["target_cols"] if c.startswith("target_up_")]
        for c in up:
            L.append(f"| `{c}` positive rate | {df[c].mean():.1%} |")
    L.append("")
    L.append("One row = one token x one candle. Rows are sorted by token, then time.\n")

    L.append("## Identifier columns (not model inputs)\n")
    L.append("| Column | Description |\n|---|---|")
    for k, v in ID_COLUMNS.items():
        L.append(f"| `{k}` | {v} |")
    L.append("")

    L.append("## Inputs (features)\n")
    L.append("All features at row *t* use only information available at the close of candle *t*.\n")
    L.append("| # | Feature | Description | Unit | Non-null | Mean | Min | Max |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, (k, (desc, unit)) in enumerate(FEATURES.items(), 1):
        s = pd.to_numeric(df[k], errors="coerce")
        L.append(f"| {i} | `{k}` | {desc} | {unit} | {s.notna().mean():.0%} | {fmt(s.mean())} | "
                 f"{fmt(s.min())} | {fmt(s.max())} |")
    L.append("")

    L.append("## Targets (what to predict)\n")
    L.append("Targets look strictly into the future. Never feed them to the model as inputs.\n")
    L.append("| Target | Description | Non-null | Mean |\n|---|---|---|---|")
    for k in meta["target_cols"]:
        s = pd.to_numeric(df[k], errors="coerce")
        L.append(f"| `{k}` | {tdocs.get(k, '')} | {s.notna().mean():.0%} | {fmt(s.mean())} |")
    L.append("")

    L.append("## How the inputs are produced\n")
    L.append("""- **Prices and volume** come from every swap that touched the token's pools (pump.fun bonding
  curve, PumpSwap, Raydium, Meteora, ...). Price = SOL moved / tokens moved in each swap,
  priced in SOL. Swaps under 0.00001 SOL are ignored as dust.
- **Candles** are """ + args.interval + """ buckets. Candles with no trades are kept, with zero volume and
  OHLC equal to the previous close, so every token is an evenly spaced series.
- **Holders** are rebuilt by replaying the exact post-transaction token balance of every wallet
  that appears in a pool transaction. A wallet counts as a holder while its balance is > 0.
  Pool vaults count as holders, the same as on block explorers. Wallets that only received
  tokens by plain transfer and never traded through a pool are not seen, so this is a slight
  undercount. The exact current count from chain is in `data/tokens.csv` (`holders_now`).
- **Market cap** = close x current total supply.
- **Split**: each token goes entirely into train, val or test (by a hash of its mint), so the
  model is always tested on coins it has never seen.
""")
    L.append("## Regenerate\n")
    L.append("```bash\npython download_data.py run --mode active --limit 50 --max-signatures 20000\n"
             f"python build_dataset.py --interval {args.interval} --horizons {args.horizons} "
             f"--up-threshold {args.up_threshold:g}\n```\n")
    if meta["skipped"]:
        L.append("## Skipped tokens\n")
        L.append(f"Tokens with fewer than {args.min_trades} trades, or too few candles for the longest horizon:\n")
        for mint, why in meta["skipped"]:
            L.append(f"- `{mint}`: {why}")
        L.append("")
    Path(args.md).write_text("\n".join(L))
    log.info("wrote %s", args.md)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(HERE / "data"))
    p.add_argument("--interval", default="30s", help="candle timeframe (default 30s)")
    p.add_argument("--horizons", default="1min,5min,10min", help="target horizons, multiples of interval")
    p.add_argument("--up-threshold", type=float, default=0.10,
                   help="price rise that counts as 'up' for target_up_* (default 0.10 = +10%%)")
    p.add_argument("--min-trades", type=int, default=20, help="skip tokens with fewer trades")
    p.add_argument("--val", type=float, default=0.15, help="fraction of tokens for validation")
    p.add_argument("--test", type=float, default=0.15, help="fraction of tokens for test")
    p.add_argument("--keep-incomplete", action="store_true",
                   help="keep the last rows of each token whose future targets are unknown")
    p.add_argument("--out", help="output CSV (default data/dataset/dataset_<interval>.csv)")
    p.add_argument("--md", default=str(HERE / "DATASET.md"), help="markdown description to write")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    df, meta = build(args)
    out = Path(args.out) if args.out else Path(args.data_dir) / "dataset" / f"dataset_{args.interval}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("dataset: %d rows, %d tokens, %d features, %d targets -> %s", len(df), df["mint"].nunique(),
             len(FEATURES), len(meta["target_cols"]), out)
    write_markdown(args, df, meta, out)


if __name__ == "__main__":
    main()
