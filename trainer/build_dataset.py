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

from dataset.features import FEATURES, ID_COLUMNS, OUTCOME_COLUMNS, make_features, make_targets, target_docs
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


def uncovered_intervals(trades: pd.DataFrame, state: dict) -> list[tuple[int, int]]:
    """Time ranges (unix s) where a pool existed but its transactions were NOT downloaded.

    A capped download fetches the newest N transactions of each pool. For a pool whose history
    is incomplete, the time between when it started existing and its earliest fetched trade is
    unknown. For an AMM pool of a pump.fun coin, "started existing" = graduation = the last
    bonding-curve trade. Filling such a gap with empty candles would look like a dead coin, so
    those candles are dropped instead.
    """
    t = trades.dropna(subset=["block_time"])
    if t.empty:
        return []
    first_by_pool = t.groupby("pool_vault")["block_time"].min()
    curve = t[t["dex"] == "pumpfun"]
    curve_last = curve["block_time"].max() if len(curve) else None
    out = []
    for vault, ps in (state.get("pools") or {}).items():
        if ps.get("complete") or vault not in first_by_pool.index:
            continue
        start = curve_last if ps.get("dex") != "pumpfun" and curve_last is not None else t["block_time"].min()
        end = first_by_pool[vault]
        if end > start:
            out.append((int(start), int(end)))
    return out


def split_segments(candles: pd.DataFrame, gaps: list[tuple[int, int]], interval: str) -> pd.DataFrame:
    """Drop candles inside coverage gaps and number the remaining contiguous stretches."""
    c = candles.copy()
    step = pd.Timedelta(interval)
    drop = pd.Series(False, index=c.index)
    for a, b in gaps:
        lo = pd.to_datetime(a, unit="s", utc=True).floor(interval)
        hi = pd.to_datetime(b, unit="s", utc=True).floor(interval)
        drop |= (c["timestamp"] > lo) & (c["timestamp"] < hi)
    c = c[~drop].reset_index(drop=True)
    c["segment"] = (c["timestamp"].diff() != step).cumsum() - 1
    return c


def coin_outcomes(trades: pd.DataFrame, supply: float | None, observed_until: int | None) -> dict:
    sol = trades[trades["quote"] == "SOL"].sort_values(["slot", "tx_index"], na_position="last")
    if sol.empty:
        return {k: float("nan") for k in OUTCOME_COLUMNS}
    ath, last = sol["price"].max(), sol["price"].iloc[-1]
    first_t, last_t = sol["block_time"].min(), sol["block_time"].max()
    idle = (observed_until or last_t) - last_t
    drawdown = last / ath - 1
    return {
        "outcome_dead": int(idle > 3600 or drawdown < -0.9),
        "outcome_graduated": int((sol["dex"] != "pumpfun").any()),
        "outcome_ath_mcap_sol": ath * supply if supply else float("nan"),
        "outcome_final_drawdown": drawdown,
        "outcome_lifetime_s": last_t - first_t,
    }


def build(args) -> tuple[pd.DataFrame, dict]:
    data = Path(args.data_dir)
    horizons = parse_horizons(args.horizons, args.interval)
    max_h = pd.Timedelta(args.interval) * max(horizons.values())
    frames, skipped, coins = [], [], []
    catalog_path = data / "tokens.csv"
    created = {}
    if catalog_path.exists():
        cat = pd.read_csv(catalog_path)
        if "created_at" in cat.columns:
            created = dict(zip(cat["mint"], cat["created_at"]))
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

        # The download fetched every trade up to `updated_at`. If the coin stopped trading
        # before that, the empty candles after its last trade are real ("it died"), so the
        # series is extended far enough that every real candle gets complete targets.
        observed = state.get("updated_at")
        end_time = None
        if observed:
            last_trade = pd.to_datetime(trades["block_time"].max(), unit="s", utc=True)
            end_time = min(pd.to_datetime(observed, unit="s", utc=True), last_trade + max_h)
        candles = build_ohlcv(trades, args.interval, supply=state.get("supply"), balances=balances,
                              end_time=end_time)
        gaps = uncovered_intervals(trades, state)
        candles = split_segments(candles, gaps, args.interval)
        if candles.empty:
            skipped.append((mint, "no candles"))
            continue
        t0 = pd.to_datetime(created[mint], unit="s", utc=True) if pd.notna(created.get(mint)) else \
            pd.to_datetime(trades["block_time"].min(), unit="s", utc=True)
        parts = []
        for _, seg in candles.groupby("segment", sort=True):
            seg = seg.reset_index(drop=True)
            parts.append(make_targets(make_features(seg, t0), horizons, args.up_threshold, args.dump_threshold))
        f = pd.concat(parts, ignore_index=True)
        if gaps:
            log.info("  %s: %d coverage gap(s) removed (history not fully downloaded), %d segment(s)",
                     state.get("symbol") or mint[:8], len(gaps), f["segment"].nunique())
        f.insert(0, "mint", mint)
        f.insert(1, "symbol", state.get("symbol") or "")
        f["split"] = assign_split(mint, args.val, args.test)
        outcome = coin_outcomes(trades, state.get("supply"), observed)
        for k, v in outcome.items():
            f[k] = v
        coins.append({"mint": mint, "symbol": state.get("symbol") or "", "trades": len(trades), **outcome})
        frames.append(f)
        log.info("%-10s %6d candles  %6d trades  holders=%s", (state.get("symbol") or mint[:8])[:10],
                 len(f), len(trades), int(f["holders"].iloc[-1]) if f["holders"].notna().any() else "n/a")

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    target_cols = [c for c in df.columns if c.startswith("target_")]
    if not df.empty and not args.keep_incomplete:
        # rows whose future is not (yet) known -> unusable for training
        before = df.groupby("mint").size()
        df = df.dropna(subset=[f"target_ret_{max(horizons, key=horizons.get)}"])
        for mint in sorted(set(before.index) - set(df["mint"])):
            skipped.append((mint, f"{before[mint]} candles, none with a known {max(horizons, key=horizons.get)} future"))
    if df.empty:
        if args.out_path.exists():
            args.out_path.unlink()  # never leave a stale dataset behind for build_sequences
        for mint, why in skipped:
            log.info("skipped %s (%s)", mint, why)
        raise SystemExit("no coin has enough data; download more (deeper --max-signatures or older coins)")
    for mint, why in skipped:
        log.info("skipped %s (%s)", mint, why)
    cols = list(ID_COLUMNS) + list(FEATURES) + target_cols + list(OUTCOME_COLUMNS)
    df = df[cols]
    meta = {"horizons": horizons, "skipped": skipped, "target_cols": target_cols, "coins": pd.DataFrame(coins)}
    return df, meta


def fmt(x) -> str:
    if pd.isna(x):
        return "-"
    if isinstance(x, (int,)) or float(x).is_integer() and abs(x) < 1e12:
        return f"{int(x):,}"
    return f"{x:.4g}"


def write_markdown(args, df: pd.DataFrame, meta: dict, out_csv: Path) -> None:
    tdocs = target_docs(meta["horizons"], args.up_threshold, args.dump_threshold)
    coins = meta["coins"]
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
    if len(coins):
        dead = int(coins["outcome_dead"].sum())
        L.append(f"| Coins dead at download | **{dead} of {len(coins)} ({dead / len(coins):.0%})** |")
        L.append(f"| Coins that graduated | {int(coins['outcome_graduated'].sum())} of {len(coins)} |")
    for c in meta["target_cols"]:
        if c.startswith(("target_up_", "target_dump_")):
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

    L.append("## Coin outcomes (labels, not inputs)\n")
    L.append("One value per coin, repeated on every row of that coin. These describe the coin's whole "
             "future, so **never use them as model inputs**. Use them to weight, filter or evaluate "
             "(for example, how often the model buys coins that end up dead).\n")
    L.append("| Column | Description |\n|---|---|")
    for k, v in OUTCOME_COLUMNS.items():
        L.append(f"| `{k}` | {v} |")
    L.append("")
    if len(coins):
        L.append("<details><summary>Per-coin outcomes in this build</summary>\n")
        L.append("| symbol | mint | trades | dead | graduated | ATH mcap (SOL) | final drawdown | lifetime |")
        L.append("|---|---|---|---|---|---|---|---|")
        for _, r in coins.sort_values("outcome_ath_mcap_sol", ascending=False).iterrows():
            L.append(f"| {r['symbol']} | `{r['mint'][:8]}…` | {r['trades']} | {fmt(r['outcome_dead'])} | "
                     f"{fmt(r['outcome_graduated'])} | {fmt(r['outcome_ath_mcap_sol'])} | "
                     f"{r['outcome_final_drawdown']:.0%} | {fmt(r['outcome_lifetime_s'] / 60)} min |")
        L.append("\n</details>\n")

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
- **Gaps in the download are not faked.** If a pool's history wasn't fully downloaded (capped by
  `--max-signatures`), the unknown time between the pool appearing and its earliest fetched
  trade is removed, instead of being filled with empty candles that would look like a dead coin.
  The rest of the coin is kept as separate `segment`s, and features, targets and model windows
  never cross a segment boundary. Download deeper (`--max-signatures`) to close the gaps.
- **Dead coins are kept.** Coins are not filtered by success, and every coin with at least
  """ + str(args.min_trades) + """ trades is included. After a coin's last trade, the series continues with empty
  `NO_TRADE` candles up to the download time (capped at the longest horizon). So the
  minutes before a coin dies get real targets: the model sees "this pattern -> nothing / dump".
- **Split**: each token goes entirely into train, val or test (by a hash of its mint), so the
  model is always tested on coins it has never seen.
""")
    L.append("## Regenerate\n")
    L.append("```bash\npython download_data.py run --mode new --limit 500 --min-age 6h --max-age 2d --max-signatures 20000\n"
             f"python build_dataset.py --interval {args.interval} --horizons {args.horizons} "
             f"--up-threshold {args.up_threshold:g}\n```\n")
    if meta["skipped"]:
        L.append("## Skipped tokens\n")
        L.append(f"Coins with fewer than {args.min_trades} trades, or with no candle whose future is known yet:\n")
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
    p.add_argument("--dump-threshold", type=float, default=0.30,
                   help="price drop that counts as a dump for target_dump_* (default 0.30 = -30%%)")
    p.add_argument("--min-trades", type=int, default=3,
                   help="skip coins with fewer trades (default 3; keep low so dead coins stay in)")
    p.add_argument("--val", type=float, default=0.15, help="fraction of tokens for validation")
    p.add_argument("--test", type=float, default=0.15, help="fraction of tokens for test")
    p.add_argument("--keep-incomplete", action="store_true",
                   help="keep the last rows of each token whose future targets are unknown")
    p.add_argument("--out", help="output CSV (default data/dataset/dataset_<interval>.csv)")
    p.add_argument("--md", default=str(HERE / "DATASET.md"), help="markdown description to write")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    out = Path(args.out) if args.out else Path(args.data_dir) / "dataset" / f"dataset_{args.interval}.csv"
    args.out_path = out
    df, meta = build(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("dataset: %d rows, %d tokens, %d features, %d targets -> %s", len(df), df["mint"].nunique(),
             len(FEATURES), len(meta["target_cols"]), out)
    write_markdown(args, df, meta, out)


if __name__ == "__main__":
    main()
