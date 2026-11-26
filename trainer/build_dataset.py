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

import numpy as np
import pandas as pd

from dataset.features import FEATURES, ID_COLUMNS, OUTCOME_COLUMNS, make_features, make_targets, target_docs
from solana_data.constants import PUMP_CURVE_SELLABLE_TOKENS, PUMP_VIRTUAL_SOL, PUMP_VIRTUAL_TOKENS
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


FLOW_COLUMNS = ["trades", "buys", "sells", "unique_traders", "buy_volume_sol", "sell_volume_sol",
                "max_buy_sol", "max_sell_sol", "unique_buyers", "unique_sellers", "new_buyers", "full_exits"]
STATE_COLUMNS = ["sol_reserve", "token_reserve", "liquidity_sol", "holders", "top10_pct",
                 "curve_progress_pct", "graduated", "bundle_pct"]
# known once at launch, the same on every candle of the coin
COIN_COLUMNS = ["bundle_launch_pct", "bundle_wallets"]
# on-chain value first, GMGN's own estimate where on-chain data doesn't cover the candle
GMGN_FALLBACK = {"holders", "curve_progress_pct", "graduated"}


def is_pump_coin(mint: str, state: dict) -> bool:
    pools = (state.get("pools") or {}).values()
    return any(p.get("dex") == "pumpfun" for p in pools) or mint.endswith("pump")


def launch_slot(balances: pd.DataFrame | None, state: dict) -> int | None:
    """Slot of the create tx, known only when the bonding curve's history was downloaded
    back to creation (the create tx is the curve vault's first transaction)."""
    pools = (state.get("pools") or {}).values()
    if balances is None or balances.empty or not any(p.get("dex") == "pumpfun" and p.get("complete") for p in pools):
        return None
    return int(balances["slot"].min())


def curve_progress_from_price(price_sol: pd.Series) -> pd.Series:
    """pump.fun curve progress in % from the price (SOL per token), using the curve's
    constant product of virtual reserves: price = k / virtual_tokens^2. Used for GMGN
    candles, where the curve's reserves aren't known. Accurate to ~1 point (fees and the
    5-minute SOL/USD conversion blur the price slightly)."""
    k = PUMP_VIRTUAL_SOL * PUMP_VIRTUAL_TOKENS
    virtual_tokens = np.sqrt(k / price_sol.where(price_sol > 0))
    return ((PUMP_VIRTUAL_TOKENS - virtual_tokens) / PUMP_CURVE_SELLABLE_TOKENS * 100).clip(0, 100)


def onchain_candles(data: Path, mint: str, args, max_h) -> tuple[pd.DataFrame | None, dict]:
    trades = pd.read_csv(data / "trades" / f"{mint}.csv")
    if len(trades) < args.min_trades:
        return None, {"skip": f"{len(trades)} on-chain trades"}
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
    candles = build_ohlcv(trades, args.interval, supply=state.get("supply"), balances=balances, end_time=end_time,
                          pump_curve=is_pump_coin(mint, state), launch_slot=launch_slot(balances, state))
    gaps = uncovered_intervals(trades, state)
    candles = split_segments(candles, gaps, args.interval)
    if candles.empty:
        return None, {"skip": "no on-chain candles"}
    return candles, {"trades": trades, "supply": state.get("supply"), "observed": observed,
                     "symbol": state.get("symbol"), "gaps": len(gaps),
                     "graduated": int((trades["dex"] != "pumpfun").any())}


def gmgn_candles(data: Path, mint: str, args, max_h) -> tuple[pd.DataFrame | None, dict]:
    """GMGN 30s klines -> evenly spaced candles. GMGN omits candles without trades, so missing
    candles are known-empty (volume 0, 0 trades). Columns GMGN can't provide (trade counts on
    traded candles, buy/sell split, liquidity, holders, top-10) are NaN = unknown."""
    try:
        k = pd.read_csv(data / "gmgn" / "kline_30s" / f"{mint}.csv")
    except pd.errors.EmptyDataError:
        k = pd.DataFrame()
    if "close" in k:
        k = k.dropna(subset=["close"])
    if k.empty or "close" not in k:
        return None, {"skip": "no GMGN candles"}
    sp = data / "gmgn" / "state" / f"{mint}.json"
    st = json.loads(sp.read_text()) if sp.exists() else {}
    k["ts"] = pd.to_datetime(k["time"], unit="s", utc=True)
    k = k.set_index("ts").sort_index()
    r = k.resample(args.interval, label="left", closed="left")
    out = pd.DataFrame({"open": r["open"].first(), "high": r["high"].max(), "low": r["low"].min(),
                        "close": r["close"].last(), "volume_sol": r["volume_sol"].sum(),
                        "volume_token": r["volume_token"].sum()})
    out = out[out["close"].notna()]
    end = out.index[-1]
    if st.get("fetched_at"):
        end = max(end, min(pd.to_datetime(st["fetched_at"], unit="s", utc=True), out.index[-1] + max_h)
                  .floor(args.interval))
    traded = out.index
    out = out.reindex(pd.date_range(out.index[0], end, freq=args.interval, tz="UTC"))
    empty = ~out.index.isin(traded)
    out["close"] = out["close"].ffill()
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out[["volume_sol", "volume_token"]] = out[["volume_sol", "volume_token"]].fillna(0.0)
    for c in FLOW_COLUMNS:
        out[c] = np.where(empty, 0.0, np.nan)
    for c in STATE_COLUMNS:
        out[c] = np.nan
    out.index.name = "timestamp"
    # Holder history from GMGN holding periods: exact when GMGN returned the full wallet list (< 100).
    hp = data / "gmgn" / "holders" / f"{mint}.json"
    if hp.exists():
        h = json.loads(hp.read_text())
        if h.get("complete"):
            close_t = ((out.index + pd.Timedelta(args.interval) - pd.Timestamp(0, tz="UTC"))
                       // pd.Timedelta(seconds=1)).to_numpy(np.int64)  # candle close, unix s
            starts = np.array([w["start"] for w in h["wallets"] if w.get("start")], dtype=float)
            ends = np.array([w["end"] if w.get("end") else np.inf for w in h["wallets"] if w.get("start")], dtype=float)
            held = (starts[None, :] <= close_t[:, None]) & (ends[None, :] > close_t[:, None])
            out["holders"] = held.sum(axis=1).astype(float)
    # Bonding curve progress from the price; graduated from GMGN's migration time.
    info_path = data / "gmgn" / "info" / f"{mint}.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    pump = mint.endswith("pump") or str(info.get("launchpad") or st.get("launchpad") or "").lower().startswith("pump")
    mayhem = "mayhem" in str(info.get("launchpad_platform") or st.get("launchpad") or "").lower()
    for c in COIN_COLUMNS:
        out[c] = np.nan
    if pump:
        close_s = ((out.index + pd.Timedelta(args.interval) - pd.Timestamp(0, tz="UTC"))
                   // pd.Timedelta(seconds=1)).to_numpy(np.int64)
        migrated = st.get("migrated_at") or info.get("migrated_timestamp") or 0
        progress = curve_progress_from_price(out["close"])
        if migrated:
            grad = pd.Series(close_s >= int(migrated), index=out.index)
        elif st.get("graduated"):
            grad = (progress >= 100).cummax()  # migration time unknown: from the curve's end price on
        else:
            grad = pd.Series(False, index=out.index)
        out["graduated"] = grad.astype(float)
        # Mayhem-mode coins: the price doesn't follow the curve formula, so progress is unknown
        out["curve_progress_pct"] = np.nan if mayhem else progress.where(~grad, 100.0)
    supply = st.get("supply")
    out["market_cap_sol"] = out["close"] * supply if supply else np.nan
    out = out.reset_index()
    out["segment"] = 0
    return out, {"supply": supply, "observed": st.get("fetched_at"), "symbol": st.get("symbol"),
                 "graduated": st.get("graduated")}


def merge_onchain(gm: pd.DataFrame, oc: pd.DataFrame | None) -> pd.DataFrame:
    """GMGN candles as the base; on-chain columns filled in where on-chain data covers the candle."""
    if oc is None:
        return gm
    base = gm.copy()
    o = oc.set_index("timestamp")
    for c in FLOW_COLUMNS + STATE_COLUMNS:
        vals = base["timestamp"].map(o[c]) if c in o else pd.Series(np.nan, index=base.index)
        if c in FLOW_COLUMNS:
            # on an empty candle GMGN already knows the flow is 0
            vals = vals.where(vals.notna() | (base["volume_sol"] > 0), 0.0)
        if c in GMGN_FALLBACK:
            vals = vals.where(vals.notna(), base[c])  # on-chain exact data first, GMGN's estimate otherwise
        base[c] = vals.to_numpy()
    for c in COIN_COLUMNS:  # per-coin launch facts apply to every candle
        known = oc[c].dropna() if c in oc else pd.Series(dtype=float)
        base[c] = known.iloc[0] if len(known) else np.nan
    return base


def candle_outcomes(c: pd.DataFrame, supply: float | None, observed: int | None, graduated) -> dict:
    real = c[c["volume_sol"] > 0]
    if real.empty:
        return {k: float("nan") for k in OUTCOME_COLUMNS}
    ath, last = real["high"].max(), real["close"].iloc[-1]
    first_t = real["timestamp"].iloc[0].timestamp()
    last_t = real["timestamp"].iloc[-1].timestamp()
    idle = (observed or last_t) - last_t
    drawdown = last / ath - 1
    return {
        "outcome_dead": int(idle > 3600 or drawdown < -0.9),
        "outcome_graduated": int(graduated) if graduated is not None and pd.notna(graduated) else float("nan"),
        "outcome_ath_mcap_sol": ath * supply if supply else float("nan"),
        "outcome_final_drawdown": drawdown,
        "outcome_lifetime_s": last_t - first_t,
    }


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


def featurize(candles: pd.DataFrame, t0, horizons: dict[str, int], args) -> pd.DataFrame:
    """Features + targets, computed per contiguous segment (never across a gap)."""
    parts = []
    for _, seg in candles.groupby("segment", sort=True):
        seg = seg.reset_index(drop=True)
        parts.append(make_targets(make_features(seg, t0), horizons, args.up_threshold, args.dump_threshold))
    return pd.concat(parts, ignore_index=True)


def finalize(f: pd.DataFrame, horizons: dict[str, int], keep_incomplete: bool = False) -> pd.DataFrame:
    """Drop rows whose longest-horizon future is unknown, and order the columns."""
    if not keep_incomplete:
        f = f.dropna(subset=[f"target_ret_{max(horizons, key=horizons.get)}"])
    target_cols = [c for c in f.columns if c.startswith("target_")]
    return f[list(ID_COLUMNS) + list(FEATURES) + target_cols + list(OUTCOME_COLUMNS)]


def build(args) -> tuple[pd.DataFrame, dict]:
    data = Path(args.data_dir)
    horizons = parse_horizons(args.horizons, args.interval)
    # Dead coins: empty candles after the last trade. Rows without a known future for the longest
    # target horizon are dropped at the end, so the tail is that horizon PLUS --tail-candles long:
    # after the drop, --tail-candles empty candles remain, and a forecast made at the last trade
    # sees all 24 future candles ("nothing happens", the coin is dead).
    max_h = pd.Timedelta(args.interval) * (max(horizons.values()) + args.tail_candles)
    frames, skipped, coins = [], [], []
    catalog_path = data / "tokens.csv"
    created = {}
    if catalog_path.exists():
        cat = pd.read_csv(catalog_path)
        if "created_at" in cat.columns:
            created = dict(zip(cat["mint"], cat["created_at"]))
    onchain_mints = {p.stem for p in (data / "trades").glob("*.csv")} if args.source != "gmgn" else set()
    gmgn_mints = {p.stem for p in (data / "gmgn" / "kline_30s").glob("*.csv")} if args.source != "onchain" else set()
    for mint in sorted(onchain_mints | gmgn_mints):
        oc, oc_info = onchain_candles(data, mint, args, max_h) if mint in onchain_mints else (None, {})
        gm, gm_info = gmgn_candles(data, mint, args, max_h) if mint in gmgn_mints else (None, {})
        if oc is None and gm is None:
            skipped.append((mint, oc_info.get("skip") or gm_info.get("skip") or "no data"))
            continue
        if gm is not None:
            candles, source = merge_onchain(gm, oc), ("gmgn+onchain" if oc is not None else "gmgn")
        else:
            candles, source = oc, "onchain"
        info = {**gm_info, **{k: v for k, v in oc_info.items() if v is not None}} if gm is not None else oc_info
        symbol = info.get("symbol") or ""
        first = candles.loc[candles["volume_sol"] > 0, "timestamp"].min()
        t0 = pd.to_datetime(created[mint], unit="s", utc=True) if pd.notna(created.get(mint)) else first
        f = featurize(candles, t0, horizons, args)
        if oc_info.get("gaps") and source == "onchain":
            log.info("  %s: %d coverage gap(s) removed (history not fully downloaded), %d segment(s)",
                     symbol or mint[:8], oc_info["gaps"], f["segment"].nunique())
        f.insert(0, "mint", mint)
        f.insert(1, "symbol", symbol)
        f["split"] = assign_split(mint, args.val, args.test)
        f["source"] = source
        if source == "onchain":
            outcome = coin_outcomes(oc_info["trades"], info.get("supply"), info.get("observed"))
        else:
            outcome = candle_outcomes(candles, info.get("supply"), info.get("observed"), info.get("graduated"))
        for k, v in outcome.items():
            f[k] = v
        n_trades = len(oc_info["trades"]) if "trades" in oc_info else None
        coins.append({"mint": mint, "symbol": symbol, "source": source,
                      "trades": n_trades if n_trades is not None else "-", **outcome})
        frames.append(f)
        log.info("%-10s %6d candles  %-13s holders=%s", symbol[:10] or mint[:8], len(f), source,
                 int(f["holders"].dropna().iloc[-1]) if f["holders"].notna().any() else "unknown")

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
    L.append("""- **Sources** (`source` column): `gmgn` = GMGN 30s klines from launch; `onchain` = our own swap
  parsing; `gmgn+onchain` = GMGN candles, with on-chain trade counts, buy/sell, liquidity,
  holders and top-10 wherever on-chain data covers the candle. Values no source has are NaN
  in this file and `UNKNOWN` tokens in the model inputs. GMGN prices and volumes (USD) are
  converted to SOL with GMGN's SOL/USD 5-minute candles.
- **On-chain prices and volume** come from every swap that touched the token's pools (pump.fun bonding
  curve, PumpSwap, Raydium, Meteora, ...). Price = SOL moved / tokens moved in each swap,
  priced in SOL. Swaps under 0.00001 SOL are ignored as dust.
- **Candles** are """ + args.interval + """ buckets. Candles with no trades are kept, with zero volume and
  OHLC equal to the previous close, so every token is an evenly spaced series.
- **Holders (GMGN)**: from GMGN wallet holding periods (start / end of holding), exact when the coin
  has fewer than 100 wallets; `UNKNOWN` above that unless on-chain data exists.
- **Holders (on-chain)** are rebuilt by replaying the exact post-transaction token balance of every wallet
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
    p.add_argument("--source", choices=["auto", "gmgn", "onchain"], default="auto",
                   help="auto (default): GMGN candles + on-chain columns where available; "
                        "gmgn: GMGN only; onchain: on-chain only")
    p.add_argument("--tail-candles", type=int, default=24,
                   help="after a coin's last trade, keep this many empty candles at least (default 24 = "
                        "the model's forecast length), so the model learns what dying looks like")
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
