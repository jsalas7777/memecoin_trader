#!/usr/bin/env python3
"""Download memecoin data from the official GMGN OpenAPI.

  export GMGN_API_KEY=...        # create at https://gmgn.ai/ai

  # coins to download (any of these; all write to data/tokens.csv)
  python download_data.py discover --mode new --limit 1000 --min-age 6h --max-age 3d   # on-chain, unbiased, incl. dead coins
  python download_gmgn.py discover                      # GMGN live lists (new / near completion / graduated)
  python download_gmgn.py collect --duration 24h        # poll GMGN all day: records every new launch

  # 30s candles from launch + a snapshot per coin, for every coin in data/tokens.csv
  python download_gmgn.py history

Outputs:
  data/gmgn/kline_30s/<mint>.csv   30s candles (USD as given by GMGN, plus SOL-converted)
  data/gmgn/info/<mint>.json       full GMGN token snapshot (holders, top-10, dev, wallet tags, ...)
  data/gmgn/holders/<mint>.json    holding periods of up to 100 wallets (exact holder history if < 100)
  data/gmgn/state/<mint>.json      fetch time / coverage (used by build_dataset.py)
  data/gmgn/sol_usd_5m.csv         SOL price, used to convert USD to SOL
  data/tokens.csv                  catalog, gmgn_* columns added
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from solana_data.constants import WSOL_MINT
from solana_data.gmgn import DEMO_KEY, GMGNClient, GMGNError

log = logging.getLogger("gmgn")
HERE = Path(__file__).resolve().parent


def parse_duration(v: str) -> int:
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if not v or v[-1] not in units:
        raise SystemExit(f"bad duration {v!r}; use e.g. 30m, 6h, 2d")
    return int(float(v[:-1]) * units[v[-1]])


# ---- catalog ---------------------------------------------------------------------
def merge_catalog(data: Path, rows: list[dict]) -> int:
    path = data / "tokens.csv"
    new = pd.DataFrame(rows)
    if new.empty:
        return 0
    if path.exists():
        old = pd.read_csv(path)
        added = (~new["mint"].isin(old["mint"])).sum()
        new = pd.concat([old, new[~new["mint"].isin(old["mint"])]], ignore_index=True)
    else:
        added = len(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    return int(added)


def trench_rows(lists: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for kind, items in lists.items():
        for t in items:
            if not t.get("address"):
                continue
            rows.append({
                "mint": t["address"],
                "symbol": t.get("symbol"),
                "created_at": t.get("created_timestamp"),
                "creator": t.get("creator"),
                "source": f"gmgn_{kind}",
            })
    return rows


def cmd_discover(args, gm: GMGNClient) -> None:
    lists = gm.trenches(args.types.split(","), args.platforms.split(",") if args.platforms else None, args.limit)
    for k, v in lists.items():
        log.info("%s: %d coins", k, len(v))
    log.info("added %d new coins to data/tokens.csv", merge_catalog(Path(args.data_dir), trench_rows(lists)))


def cmd_collect(args, gm: GMGNClient) -> None:
    """GMGN only lists the ~60 newest launches at any moment. Polling it regularly records
    every launch, dead or alive, so they can be downloaded later once their fate is known."""
    end = time.time() + parse_duration(args.duration)
    total = 0
    while time.time() < end:
        try:
            lists = gm.trenches(args.types.split(","), args.platforms.split(",") if args.platforms else None, 80)
            added = merge_catalog(Path(args.data_dir), trench_rows(lists))
            total += added
            log.info("+%d new coins (total this run %d)", added, total)
        except GMGNError as e:
            log.warning("poll failed: %s", e)
        time.sleep(args.every)


# ---- SOL price for USD -> SOL conversion -----------------------------------------
def sol_prices(gm: GMGNClient, data: Path, frm: int, to: int) -> pd.DataFrame:
    path = data / "gmgn" / "sol_usd_5m.csv"
    cache = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["time", "sol_usd"])
    ranges = []
    if cache.empty:
        ranges.append((frm - 600, to))
    else:
        if frm < cache["time"].min() - 600:
            ranges.append((frm - 600, int(cache["time"].min())))
        if to > cache["time"].max() + 600:
            ranges.append((int(cache["time"].max()), to))
    for a, b in ranges:
        rows = gm.kline_range(WSOL_MINT, a, b, "5m")
        add = pd.DataFrame({"time": [int(r["time"]) // 1000 for r in rows],
                            "sol_usd": [float(r["close"]) for r in rows]})
        cache = pd.concat([cache, add], ignore_index=True)
    if ranges:
        cache = cache.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_csv(path, index=False)
    return cache


# ---- per-coin history ------------------------------------------------------------
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def download_coin(gm: GMGNClient, data: Path, mint: str, catalog_created: float | None,
                  holders: bool = True) -> dict:
    kdir, idir, sdir, hdir = (data / "gmgn" / d for d in ("kline_30s", "info", "state", "holders"))
    for d in (kdir, idir, sdir, hdir):
        d.mkdir(parents=True, exist_ok=True)
    now = int(time.time())
    info = gm.token_info(mint)
    (idir / f"{mint}.json").write_text(json.dumps(info, indent=1))
    created = info.get("creation_timestamp") or info.get("open_timestamp") or catalog_created
    kpath = kdir / f"{mint}.csv"
    old = pd.read_csv(kpath) if kpath.exists() else None
    frm = int(old["time"].max()) + 1 if old is not None and len(old) else int((created or now - 7 * 86400)) - 60

    rows = gm.kline_range(mint, frm, now, "30s")
    new = pd.DataFrame({
        "time": [int(r["time"]) // 1000 for r in rows],
        "open_usd": [float(r["open"]) for r in rows],
        "high_usd": [float(r["high"]) for r in rows],
        "low_usd": [float(r["low"]) for r in rows],
        "close_usd": [float(r["close"]) for r in rows],
        # GMGN naming: `volume` is USD, `amount` is token units
        "volume_usd": [float(r["volume"]) for r in rows],
        "volume_token": [float(r["amount"]) for r in rows],
    })
    k = pd.concat([old, new], ignore_index=True) if old is not None and not new.empty else (old if old is not None else new)
    if not k.empty:
        k["time"] = k["time"].astype("int64")
        k = k.drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True)
        sol = sol_prices(gm, data, int(k["time"].min()), int(k["time"].max()))
        k = pd.merge_asof(k.drop(columns=[c for c in ("sol_usd",) if c in k]), sol.astype({"time": "int64"}),
                          on="time", direction="nearest")
        for c in ("open", "high", "low", "close"):
            k[c] = k[f"{c}_usd"] / k["sol_usd"]
        k["volume_sol"] = k["volume_usd"] / k["sol_usd"]
    k.to_csv(kpath, index=False)

    n_wallets = None
    if holders:
        wallets = gm.top_holders(mint, 100)
        n_wallets = len(wallets)
        (hdir / f"{mint}.json").write_text(json.dumps({
            "fetched_at": now,
            # GMGN returns at most 100 wallets; with 100 the list may be cut off
            "complete": len(wallets) < 100,
            "wallets": [{"address": w.get("address"), "start": w.get("start_holding_at"),
                         "end": w.get("end_holding_at"), "balance": _f(w.get("balance")),
                         "addr_type": w.get("addr_type")} for w in wallets],
        }))

    stat = info.get("stat") or {}
    tags = info.get("wallet_tags_stat") or {}
    dev = info.get("dev") or {}
    state = {
        "mint": mint, "fetched_at": now, "created_at": created,
        "candles": len(k), "first_time": int(k["time"].min()) if len(k) else None,
        "last_time": int(k["time"].max()) if len(k) else None,
        "symbol": info.get("symbol"), "name": info.get("name"),
        "supply": _f(info.get("total_supply")),
        "launchpad": info.get("launchpad_platform") or info.get("launchpad"),
        "launchpad_status": info.get("launchpad_status"),  # 0 not open, 1 live (bonding curve), 2 migrated
        "graduated": int(info.get("launchpad_status") == 2 or bool(info.get("migrated_pool"))
                         or bool(info.get("migrated_timestamp"))),
        "migrated_at": info.get("migrated_timestamp") or None,  # unix s of the move to the AMM
    }
    (sdir / f"{mint}.json").write_text(json.dumps(state, indent=1))
    return {
        "mint": mint, "symbol": info.get("symbol"), "name": info.get("name"),
        "gmgn_candles": len(k), "gmgn_fetched_at": now, "gmgn_wallets": n_wallets,
        "gmgn_holders_now": stat.get("holder_count", info.get("holder_count")),
        "gmgn_top10_now": _f(stat.get("top_10_holder_rate")),
        "gmgn_liquidity_usd_now": _f(info.get("liquidity")),
        "gmgn_launchpad_status": info.get("launchpad_status"),
        "gmgn_graduated": state["graduated"],
        "gmgn_creator_created_count": stat.get("creator_created_count"),
        "gmgn_bundler_rate": _f(stat.get("top_bundler_trader_percentage")),
        "gmgn_bot_degen_rate": _f(stat.get("bot_degen_rate")),
        "gmgn_fresh_wallet_rate": _f(stat.get("fresh_wallet_rate")),
        "gmgn_sniper_wallets": tags.get("sniper_wallets"),
        "gmgn_smart_wallets": tags.get("smart_wallets"),
        "gmgn_creator_status": dev.get("creator_token_status"),
    }


def update_catalog(data: Path, row: dict) -> None:
    path = data / "tokens.csv"
    cat = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["mint"])
    if row["mint"] not in set(cat["mint"]):
        cat = pd.concat([cat, pd.DataFrame([row])], ignore_index=True)
    else:
        for key, v in row.items():
            if key == "mint":
                continue
            if key not in cat.columns:
                cat[key] = None
            if key in ("symbol", "name") and pd.notna(cat.loc[cat["mint"] == row["mint"], key]).all():
                continue
            cat[key] = cat[key].astype(object)
            cat.loc[cat["mint"] == row["mint"], key] = v
    cat.to_csv(path, index=False)


def cmd_history(args, gm: GMGNClient) -> None:
    data = Path(args.data_dir)
    cat_path = data / "tokens.csv"
    if args.mint:
        mints, created = args.mint, {}
    else:
        if not cat_path.exists():
            raise SystemExit(f"{cat_path} not found; run a discover command first")
        cat = pd.read_csv(cat_path)
        created = dict(zip(cat["mint"], cat.get("created_at", pd.Series([None] * len(cat)))))
        mints = cat["mint"].tolist()
        if args.min_age:
            cutoff = time.time() - parse_duration(args.min_age)
            mints = [m for m in mints if pd.isna(created.get(m)) or created[m] <= cutoff]
        if args.skip_fresh:
            sdir = data / "gmgn" / "state"
            def fresh(m):
                p = sdir / f"{m}.json"
                return p.exists() and time.time() - json.loads(p.read_text())["fetched_at"] < parse_duration(args.skip_fresh)
            mints = [m for m in mints if not fresh(m)]
    log.info("%d coins to download from GMGN", len(mints))
    t0 = time.time()
    for i, mint in enumerate(mints, 1):
        try:
            row = download_coin(gm, data, mint, created.get(mint), holders=not args.no_holders)
            update_catalog(data, row)
            log.info("[%d/%d] %-10s %5d candles  holders=%s  graduated=%s",
                     i, len(mints), (row["symbol"] or mint[:8])[:10], row["gmgn_candles"],
                     row["gmgn_holders_now"], row["gmgn_graduated"])
        except GMGNError as e:
            log.error("[%d/%d] %s failed: %s", i, len(mints), mint, e)
    log.info("done in %.0fs, %d API requests", time.time() - t0, gm.requests_made)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(HERE / "data"))
    p.add_argument("--points", type=float, default=5.0,
                   help="GMGN rate-limit points per second of your plan (Free 5 (default), Plus 20, Pro 50)")
    p.add_argument("--demo-key", action="store_true", help="use GMGN's public TEST key (read-only, testing only)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("discover", help="add GMGN's live launch lists to data/tokens.csv")
    c = sub.add_parser("collect", help="poll GMGN's new-launch list to record every launch over time")
    for sp, types in ((d, "new_creation,near_completion,completed"), (c, "new_creation")):
        sp.add_argument("--types", default=types, help=f"trench lists (default {types})")
        sp.add_argument("--platforms", default="Pump.fun", help="launchpads, comma separated (empty = GMGN default)")
    d.add_argument("--limit", type=int, default=80)
    c.add_argument("--every", type=int, default=60, help="seconds between polls (default 60)")
    c.add_argument("--duration", default="24h", help="how long to keep collecting (e.g. 6h, 3d)")
    h = sub.add_parser("history", help="download 30s candles from launch + snapshot for each coin")
    h.add_argument("--mint", action="append", help="mint address (repeatable); default: data/tokens.csv")
    h.add_argument("--min-age", default="2h", help="skip coins younger than this, their fate is unknown (default 2h)")
    h.add_argument("--no-holders", action="store_true",
                   help="skip the wallet list used to rebuild holder history (saves 1 heavier call per coin)")
    h.add_argument("--skip-fresh", help="skip coins already fetched within this time (e.g. 12h)")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    with GMGNClient(DEMO_KEY if args.demo_key else None, points_per_s=args.points) as gm:
        {"discover": cmd_discover, "collect": cmd_collect, "history": cmd_history}[args.cmd](args, gm)


if __name__ == "__main__":
    main()
