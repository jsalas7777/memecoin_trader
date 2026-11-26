#!/usr/bin/env python3
"""Convert a downloaded pump.fun trade archive into the model's training dataset.

  python import_archive.py                  # archives/pumpdata-2026-10-15 -> data/dataset/pumpdata-2026-10-15_30s/
  python import_archive.py --limit 2000     # quick run: 2,000 coins sampled evenly over the day
  python import_archive.py --archive archives/<folder>

Then train on it:
  python train.py --dataset data/dataset/pumpdata-2026-10-15_30s

What it does (progress bar for each step):
  1. Reads the raw trades with DuckDB: only coins LAUNCHED in the archive (so every coin's
     history starts at its first trade), only landed trades against SOL. Maps each trade to
     the pipeline's trade format, and rebuilds every wallet's token balance after each trade.
  2. Builds every coin exactly like build_dataset.py, in parallel on all CPU cores: 30s
     candles, holders, top-10, liquidity, bonding curve progress, launch bundle, whale
     buys/sells, features, targets and outcomes. Dead coins included, with 24 empty
     candles after their last trade.
  3. Verifies a sample of coins with the same checks as build_sequences.py.
  4. Writes the dataset (Parquet), a per-coin table and a short report.

Output folder data/dataset/<archive>_30s/:
  candles/         one row per coin x 30s candle, Parquet parts (same columns as dataset_30s.csv)
  coins.parquet    one row per coin: symbol, launch time, trades, candles, outcome
  REPORT.md        what's in it, and the verification result
  verify/          the verification sample, its tokenized inputs and MODEL.md

Approximations (archive has trades, not balances): a wallet's balance = its buys minus its
sells (tokens received by plain transfer are not seen); the bonding curve itself is not
counted as a holder. pumpdata archive data is MIT licensed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm

from solana_data.constants import MAYHEM_AGENT, MAYHEM_SUPPLY

HERE = Path(__file__).resolve().parent
WSOL = "So11111111111111111111111111111111111111112"
log = logging.getLogger("import")


def say(msg: str) -> None:
    tqdm.write(msg)


# ---- step 1: raw trades -> pipeline trades + balances ---------------------------------
def stage(archive: Path, out: Path, limit: int | None) -> tuple[Path, pd.DataFrame, int]:
    import duckdb
    from solana_data.constants import PUMP_CURVE_START_TOKENS, PUMP_VIRTUAL_TOKENS

    trades_glob, launches_glob = archive / "data" / "*.parquet", archive / "launches" / "*.parquet"
    if not list(archive.glob("data/*.parquet")) or not list(archive.glob("launches/*.parquet")):
        raise SystemExit(f"{archive} has no data/ or launches/ parquet files. Download it first:\n"
                         "  python download_archive.py")
    con = duckdb.connect()
    con.execute(f"SET threads = {os.cpu_count() or 4}")
    # every launch in the archive whose bonding curve trades against SOL
    launches = con.execute(f"""
        SELECT mint, slot AS launch_slot, ts AS launch_ts, symbol, name,
               supply / pow(10, decimals) AS supply, decimals
        FROM read_parquet('{launches_glob}') WHERE quote_mint = '{WSOL}' ORDER BY ts, slot""").df()
    observed = int(con.execute(f"SELECT max(ts) FROM read_parquet('{trades_glob}')").fetchall()[0][0])
    if limit and len(launches) > limit:  # even sample over the day: same mix of dead/alive coins
        idx = np.unique(np.linspace(0, len(launches) - 1, limit).round().astype(int))
        launches = launches.iloc[idx].reset_index(drop=True)
    con.register("launch_df", launches)
    offset = PUMP_VIRTUAL_TOKENS - PUMP_CURVE_START_TOKENS  # curve vault = virtual token reserve - 73M
    staging = out / "_staging.parquet"
    say(f"1/4 Reading trades of {len(launches):,} coins with DuckDB …")
    t0 = time.time()
    con.execute(f"""
        COPY (
          WITH t AS (
            SELECT t.*, l.decimals FROM read_parquet('{trades_glob}') t JOIN launch_df l USING (mint)
            WHERE NOT t.failed AND t.quote_mint = '{WSOL}' AND t.token_amount > 0 AND t.sol_lamports > 0)
          SELECT mint,
            slot::VARCHAR || '-' || idx::VARCHAR || '-' || seq::VARCHAR AS signature,
            slot, idx::BIGINT * 1000 + seq AS tx_index, ts AS block_time,
            CASE venue WHEN 'curve' THEN 'pumpfun' ELSE 'pumpswap' END AS dex,
            CASE venue WHEN 'curve' THEN 'curve:' || mint ELSE pool END AS pool_vault,
            side,
            token_amount / pow(10, decimals) AS token_amount,
            sol_lamports / 1e9 AS quote_amount,
            'SOL' AS quote,
            (sol_lamports / 1e9) / (token_amount / pow(10, decimals)) AS price,
            wallet AS trader,
            -- reserves AFTER the trade: curve rows report them after; PumpSwap rows before
            CASE venue WHEN 'curve' THEN r2 / pow(10, decimals) - {offset}
                 ELSE (r1 + CASE side WHEN 'buy' THEN -token_amount::DOUBLE ELSE token_amount::DOUBLE END)
                      / pow(10, decimals) END AS token_reserve,
            CASE venue WHEN 'curve' THEN r3 / 1e9
                 ELSE (r2 + CASE side WHEN 'buy' THEN sol_lamports ELSE -sol_lamports END) / 1e9 END AS quote_reserve,
            -- the wallet's token balance after this trade = its buys minus its sells so far
            greatest(sum(CASE side WHEN 'buy' THEN token_amount::DOUBLE ELSE -token_amount::DOUBLE END)
                     OVER (PARTITION BY mint, wallet ORDER BY slot, idx, seq ROWS UNBOUNDED PRECEDING), 0)
                     / pow(10, decimals) AS balance
          FROM t ORDER BY mint, slot, idx, seq
        ) TO '{staging}' (FORMAT parquet, ROW_GROUP_SIZE 100000)""")
    counts = con.execute(f"""SELECT mint, count(*) AS n, bool_or(trader = '{MAYHEM_AGENT}') AS mayhem
                             FROM read_parquet('{staging}') GROUP BY mint ORDER BY mint""").df()
    launches = launches.merge(counts, on="mint", how="left").fillna({"n": 0, "mayhem": False})
    # Mayhem-mode coins have 2B tokens (1B extra for pump.fun's Mayhem agent), not the 1B listed
    launches.loc[launches["mayhem"].astype(bool), "supply"] = float(MAYHEM_SUPPLY)
    say(f"  ✓ {int(launches['n'].sum()):,} trades of {int((launches['n'] > 0).sum()):,} coins "
        f"({len(launches) - int((launches['n'] > 0).sum()):,} launches never traded against SOL; "
        f"{int(launches['mayhem'].sum()):,} Mayhem-mode) in {time.time() - t0:.0f}s")
    return staging, launches, observed


# ---- step 2: per-coin candles, features, targets ------------------------------------
TRADE_COLS = ["signature", "slot", "tx_index", "block_time", "dex", "pool_vault", "side", "token_amount",
              "quote_amount", "quote", "price", "trader", "token_reserve", "quote_reserve"]


def build_coin(mint: str, df: pd.DataFrame, info: dict, cfg: dict) -> tuple[str, pd.DataFrame | None, dict]:
    from build_dataset import assign_split, coin_outcomes, featurize, finalize, split_segments
    from solana_data.ohlcv import build_ohlcv

    args = SimpleNamespace(**cfg)
    if len(df) < args.min_trades:
        return mint, None, {"skip": f"{len(df)} trades"}
    trades = df[TRADE_COLS]
    balances = pd.DataFrame({"signature": df["signature"], "slot": df["slot"], "tx_index": df["tx_index"],
                             "block_time": df["block_time"], "account": df["trader"], "owner": df["trader"],
                             "balance": df["balance"]})
    max_h = pd.Timedelta(args.interval) * (max(args.horizons.values()) + args.tail_candles)
    last = pd.to_datetime(trades["block_time"].max(), unit="s", utc=True)
    end_time = min(pd.to_datetime(args.observed, unit="s", utc=True), last + max_h)
    candles = build_ohlcv(trades, args.interval, supply=info["supply"], balances=balances, end_time=end_time,
                          pump_curve=True, launch_slot=int(info["launch_slot"]))
    if candles.empty:
        return mint, None, {"skip": "no candles"}
    candles = split_segments(candles, [], args.interval)
    t0 = pd.to_datetime(int(info["launch_ts"]), unit="s", utc=True)
    f = featurize(candles, t0, args.horizons, args)
    f.insert(0, "mint", mint)
    f.insert(1, "symbol", str(info.get("symbol") or "")[:32])
    f["split"] = assign_split(mint, args.val, args.test)
    f["source"] = "archive"
    outcome = coin_outcomes(trades, info["supply"], args.observed)
    for k, v in outcome.items():
        f[k] = v
    f = finalize(f, args.horizons)
    meta = {"symbol": info.get("symbol"), "launch_ts": int(info["launch_ts"]), "trades": len(trades),
            "mayhem": int(bool(info.get("mayhem"))),
            "candles": len(f), "graduated_on_day": int((trades["dex"] != "pumpfun").any()), **outcome}
    if f.empty:
        return mint, None, {"skip": "no candle with a known future", **meta}
    return mint, f, meta


ID_STR = ("mint", "symbol", "split", "source")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Same dtypes in every part file."""
    for c in df.columns:
        if c in ID_STR:
            df[c] = df[c].astype(str)
        elif c == "timestamp":
            df[c] = pd.to_datetime(df[c], utc=True).astype("datetime64[ns, UTC]")
        elif c == "segment":
            df[c] = df[c].astype("int64")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def build_all(staging: Path, launches: pd.DataFrame, observed: int, out: Path, args) -> pd.DataFrame:
    import duckdb
    from build_dataset import parse_horizons

    cfg = {"interval": "30s", "horizons": parse_horizons(args.horizons, "30s"), "tail_candles": 24,
           "up_threshold": args.up_threshold, "dump_threshold": args.dump_threshold, "min_trades": args.min_trades,
           "val": 0.15, "test": 0.15, "observed": observed}
    traded = launches[launches["n"] > 0].sort_values("mint").reset_index(drop=True)
    info = traded.set_index("mint").to_dict("index")
    # batches of ~300k trades, in mint order (the staging file is sorted by mint)
    batches, cur, size = [], [], 0
    for m, n in zip(traded["mint"], traded["n"]):
        cur.append(m); size += int(n)
        if size >= 300_000:
            batches.append(cur); cur, size = [], 0
    if cur:
        batches.append(cur)

    con = duckdb.connect()
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    rows = part = 0
    coins_meta, skipped = [], 0
    bar = tqdm(total=len(traded), desc="2/4 Building coins", unit="coin", dynamic_ncols=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for batch in batches:
            df = con.execute(f"SELECT * FROM read_parquet('{staging}') WHERE mint BETWEEN ? AND ?",
                             [batch[0], batch[-1]]).df()
            futs = [pool.submit(build_coin, m, g.reset_index(drop=True), info[m], cfg)
                    for m, g in df.groupby("mint", sort=False)]
            frames = []
            for fut in as_completed(futs):
                mint, f, meta = fut.result()
                coins_meta.append({"mint": mint, **{k: v for k, v in meta.items() if k != "skip"},
                                   "skipped": meta.get("skip")})
                if f is None:
                    skipped += 1
                else:
                    frames.append(f)
                bar.update(1)
            if frames:
                chunk = normalize(pd.concat(frames, ignore_index=True).sort_values(["mint", "timestamp"]))
                chunk.to_parquet(out / "candles" / f"part-{part:05d}.parquet", index=False)
                part += 1
                rows += len(chunk)
            bar.set_postfix_str(f"{rows:,} candles, {skipped:,} skipped")
    bar.close()
    coins = pd.DataFrame(coins_meta)
    coins.to_parquet(out / "coins.parquet", index=False)
    say(f"  ✓ {rows:,} candle rows from {len(coins) - skipped:,} coins in {part} part files "
        f"({skipped:,} coins skipped: fewer than {args.min_trades} trades or no known future)")
    return coins


# ---- step 3: verify a sample ---------------------------------------------------------
def verify_sample(out: Path, n: int = 300) -> dict:
    vdir = out / "verify"
    vdir.mkdir(exist_ok=True)
    parts = sorted((out / "candles").glob("part-*.parquet"))
    mints = pd.concat([pd.read_parquet(p, columns=["mint"]).drop_duplicates() for p in parts])["mint"].tolist()
    rng = np.random.default_rng(0)
    pick = set(rng.choice(mints, size=min(n, len(mints)), replace=False).tolist())
    sample = pd.concat([d[d["mint"].isin(pick)] for d in (pd.read_parquet(p) for p in parts)], ignore_index=True)
    sample.to_csv(vdir / "dataset_30s.csv", index=False)
    say(f"3/4 Verifying a sample of {len(pick)} coins ({len(sample):,} candles) …")
    r = subprocess.run([sys.executable, str(HERE / "build_sequences.py"), "--dataset", str(vdir / "dataset_30s.csv"),
                        "--out", str(vdir / "sequences_30s.npz"), "--md", str(vdir / "MODEL.md")],
                       cwd=HERE, capture_output=True, text=True)
    text = r.stdout + r.stderr
    checks = {k: text.count(f"[{k}]") for k in ("PASS", "WARN", "FAIL")}
    for line in text.splitlines():
        if "[FAIL]" in line:
            say("  " + line.strip())
    status = "passed" if r.returncode == 0 and checks["FAIL"] == 0 else "FAILED"
    say(f"  {'✓' if status == 'passed' else '✗'} input checks {status}: {checks['PASS']} passed, "
        f"{checks['WARN']} warnings, {checks['FAIL']} failed (details: {vdir / 'MODEL.md'})")
    return {"status": status, **checks}


# ---- step 4: report --------------------------------------------------------------------
def report(out: Path, archive: Path, coins: pd.DataFrame, check: dict, observed: int, t_start: float) -> None:
    kept = coins[coins["skipped"].isna()]
    parts = sorted((out / "candles").glob("part-*.parquet"))
    tcols = ["target_up_5m", "target_dump_5m"]
    sums = {c: [0.0, 0] for c in tcols}
    rows = 0
    for p in parts:
        d = pd.read_parquet(p, columns=tcols)
        rows += len(d)
        for c in tcols:
            sums[c][0] += d[c].sum(); sums[c][1] += d[c].notna().sum()
    life = kept["outcome_lifetime_s"] if len(kept) else pd.Series(dtype=float)
    dead = kept["outcome_dead"].mean() if len(kept) else float("nan")
    grad = kept["graduated_on_day"].mean() if len(kept) else float("nan")
    lines = [
        f"# Dataset from `{archive.name}`\n",
        f"Built {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M UTC} by `import_archive.py` from the pump.fun trade "
        "archive (pumpdata.fun sample, MIT license).\n",
        "| | |\n|---|---|",
        f"| Coins | {len(kept):,} (of {len(coins):,} with trades) |",
        f"| Candle rows (30s) | {rows:,} |",
        f"| Dead at the end of the archive | {dead:.0%} |",
        f"| Graduated (left the bonding curve) | {grad:.1%} |",
        f"| Median coin lifetime (first to last trade) | {life.median() / 60:.1f} min |" if len(life) else "",
        f"| `target_up_5m` positive rate | {sums['target_up_5m'][0] / max(sums['target_up_5m'][1], 1):.1%} |",
        f"| `target_dump_5m` positive rate | {sums['target_dump_5m'][0] / max(sums['target_dump_5m'][1], 1):.1%} |",
        f"| Archive ends | {pd.to_datetime(observed, unit='s', utc=True):%Y-%m-%d %H:%M UTC} |",
        f"| Input verification (sample) | {check['status']}: {check['PASS']} passed, {check['WARN']} warnings, "
        f"{check['FAIL']} failed |",
        "",
        "Columns are the same as `dataset_30s.csv` (see DATASET.md); `source` = `archive`. All inputs have real "
        "data here, including the ones switched off for GMGN data (top-10, trades per minute, liquidity, "
        "launch bundle, whale buy/sell).",
        "",
        "Approximations: a wallet's balance is its buys minus its sells (plain transfers are not in the "
        "archive), and the bonding curve itself is not counted as a holder.",
        "",
        f"Mayhem-mode coins (pump.fun's Mayhem agent trades on their curve): {int(kept['mayhem'].sum()) if len(kept) else 0:,} "
        "of the coins. They use a 2B supply, and their curve progress is `UNKNOWN`.",
        "",
        "```bash\npython train.py --dataset " + str(out.relative_to(HERE) if out.is_relative_to(HERE) else out) + "\n```",
    ]
    (out / "REPORT.md").write_text("\n".join(l for l in lines if l is not None) + "\n")
    say("\n── Summary " + "─" * 40)
    say(f"Coins        : {len(kept):,}   (dead {dead:.0%}, graduated {grad:.1%})")
    say(f"Candle rows  : {rows:,}")
    say(f"Verification : {check['status']}")
    say(f"Time         : {(time.time() - t_start) / 60:.1f} min")
    say(f"Output       : {out}")
    say(f"\nTrain on it  : python train.py --dataset {out.relative_to(HERE) if out.is_relative_to(HERE) else out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive", default=str(HERE / "archives" / "pumpdata-2026-10-15"),
                   help="folder from download_archive.py (default archives/pumpdata-2026-10-15)")
    p.add_argument("--out", help="output folder (default data/dataset/<archive>_30s)")
    p.add_argument("--limit", type=int, help="only this many coins, sampled evenly over the archive (quick runs)")
    p.add_argument("--workers", type=int, help="parallel processes (default: CPU cores - 1)")
    p.add_argument("--min-trades", type=int, default=3, help="skip coins with fewer trades (default 3)")
    p.add_argument("--horizons", default="1min,5min,10min")
    p.add_argument("--up-threshold", type=float, default=0.10)
    p.add_argument("--dump-threshold", type=float, default=0.30)
    p.add_argument("--keep-staging", action="store_true", help="keep the intermediate trades file")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    archive = Path(args.archive).resolve()
    out = Path(args.out) if args.out else HERE / "data" / "dataset" / f"{archive.name}_30s"
    if out.exists():
        shutil.rmtree(out)  # always rebuild from scratch: no stale parts mixed in
    (out / "candles").mkdir(parents=True)
    t_start = time.time()
    staging, launches, observed = stage(archive, out, args.limit)
    coins = build_all(staging, launches, observed, out, args)
    if not args.keep_staging:
        staging.unlink(missing_ok=True)
    if not list((out / "candles").glob("part-*.parquet")):
        raise SystemExit("no coin produced any candles")
    check = verify_sample(out)
    report(out, archive, coins, check, observed, t_start)
    (out / "build.json").write_text(json.dumps({"archive": str(archive), "args": vars(args), "observed": observed,
                                                 "verification": check}, indent=1))
    if check["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
