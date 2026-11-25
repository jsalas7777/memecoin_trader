#!/usr/bin/env python3
"""Download Solana memecoin price / volume history directly from the chain.

Examples:
  # find the 30 newest pump.fun launches and save them to data/tokens.csv
  python download_data.py discover --mode new --limit 30

  # find currently hot tokens trading on pump.fun / PumpSwap
  python download_data.py discover --mode active --limit 20

  # download swap history + OHLCV for every token in data/tokens.csv
  python download_data.py history --intervals 30s,1min,5min

  # one specific mint, capped at the 20k most recent txs per pool
  python download_data.py history --mint <MINT> --max-signatures 20000

  # discover + download in one go
  python download_data.py run --mode new --limit 20

  # rebuild candles from already-downloaded trades (no RPC calls)
  python download_data.py ohlcv --intervals 15s,1min
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from solana_data.discovery import discover_active, discover_new_pumpfun
from solana_data.history import HistoryDownloader
from solana_data.ohlcv import build_ohlcv
from solana_data.rpc import DEFAULT_RPC_URL, SolanaRPC

log = logging.getLogger("download")
HERE = Path(__file__).resolve().parent


def parse_since(value: str | None) -> int | None:
    if not value:
        return None
    if value[-1] in "hd" and value[:-1].replace(".", "").isdigit():
        hours = float(value[:-1]) * (24 if value[-1] == "d" else 1)
        return int(time.time() - hours * 3600)
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def tokens_path(args) -> Path:
    return Path(args.data_dir) / "tokens.csv"


def save_tokens(args, tokens: list[dict]) -> pd.DataFrame:
    path = tokens_path(args)
    new = pd.DataFrame(tokens)
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new = new.drop_duplicates("mint", keep="first")
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)
    log.info("tokens catalog: %d mints -> %s", len(new), path)
    return new


async def cmd_discover(args, rpc: SolanaRPC) -> list[str]:
    if args.mode == "new":
        tokens = await discover_new_pumpfun(rpc, args.limit)
    else:
        tokens = await discover_active(rpc, args.programs.split(","), args.sample_txs, args.limit)
    log.info("discovered %d tokens", len(tokens))
    save_tokens(args, tokens)
    return [t["mint"] for t in tokens]


def load_balances(args, mint: str) -> pd.DataFrame | None:
    path = Path(args.data_dir) / "balances" / f"{mint}.csv"
    return pd.read_csv(path) if path.exists() else None


def write_candles(args, mint: str, trades: pd.DataFrame, supply: float | None) -> None:
    balances = load_balances(args, mint)
    for interval in args.intervals.split(","):
        candles = build_ohlcv(trades, interval, supply=supply, fill_gaps=not args.no_fill, balances=balances)
        out = Path(args.data_dir) / "ohlcv" / interval / f"{mint}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        candles.to_csv(out, index=False)
        log.info("  %s candles: %d rows -> %s", interval, len(candles), out)


def update_catalog(args, mint: str, state: dict, trades: pd.DataFrame) -> None:
    path = tokens_path(args)
    cat = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["mint"])
    row = {
        "mint": mint, "symbol": state.get("symbol"), "name": state.get("name"),
        "decimals": state.get("decimals"), "supply": state.get("supply"),
        "pools": ";".join(f"{v['dex']}:{k}" for k, v in state["pools"].items()),
        "history_complete": all(p.get("complete") for p in state["pools"].values()),
        "n_trades": len(trades),
        "holders_now": state.get("holders_now"),
        "first_trade": trades["datetime"].min() if len(trades) else None,
        "last_trade": trades["datetime"].max() if len(trades) else None,
    }
    if mint not in set(cat["mint"]):
        cat = pd.concat([cat, pd.DataFrame([row])], ignore_index=True)
    else:
        for k, v in row.items():
            if k not in cat.columns:
                cat[k] = None
            cat[k] = cat[k].astype(object)
            cat.loc[cat["mint"] == mint, k] = v
    cat.to_csv(path, index=False)


async def cmd_history(args, rpc: SolanaRPC, mints: list[str] | None = None) -> None:
    if not mints:
        mints = args.mint or []
    if not mints:
        path = tokens_path(args)
        if not path.exists():
            sys.exit(f"no --mint given and {path} not found; run `discover` first")
        mints = pd.read_csv(path)["mint"].tolist()
    dl = HistoryDownloader(rpc, args.data_dir, max_signatures=args.max_signatures,
                           since=parse_since(args.since), count_holders=not args.no_holders)
    for i, mint in enumerate(mints, 1):
        log.info("[%d/%d] %s", i, len(mints), mint)
        try:
            trades = await dl.download(mint)
        except Exception as e:
            log.error("  failed: %s", e)
            continue
        state = json.loads((Path(args.data_dir) / "state" / f"{mint}.json").read_text())
        write_candles(args, mint, trades, state.get("supply"))
        update_catalog(args, mint, state, trades)


def cmd_ohlcv(args) -> None:
    data = Path(args.data_dir)
    for f in sorted((data / "trades").glob("*.csv")):
        mint = f.stem
        st = data / "state" / f"{mint}.json"
        supply = json.loads(st.read_text()).get("supply") if st.exists() else None
        log.info("%s", mint)
        write_candles(args, mint, pd.read_csv(f), supply)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rpc", help="Solana RPC URL (default: $SOLANA_RPC_URL or public mainnet)")
    p.add_argument("--rps", type=float, help="max RPC requests/second (default: 2 on public RPC, 20 otherwise)")
    p.add_argument("--concurrency", type=int, help="parallel requests (default: 2 on public RPC, 16 otherwise)")
    p.add_argument("--data-dir", default=str(HERE / "data"))
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_discover(sp):
        sp.add_argument("--mode", choices=["new", "active"], default="new",
                        help="new = latest pump.fun launches; active = most traded right now")
        sp.add_argument("--limit", type=int, default=25)
        sp.add_argument("--programs", default="pumpfun,pumpswap",
                        help="active mode: DEX programs to scan (pumpfun,pumpswap,raydium_v4,"
                             "raydium_cpmm,raydium_launchlab,meteora_dlmm or a program id)")
        sp.add_argument("--sample-txs", type=int, default=300, help="active mode: txs sampled per program")

    def add_history(sp):
        sp.add_argument("--max-signatures", type=int, default=5000,
                        help="max transactions fetched per pool per run (re-run to backfill further)")
        sp.add_argument("--since", help="stop at this age: ISO date (2026-11-01) or relative (12h, 7d)")
        sp.add_argument("--intervals", default="30s", help="candle sizes, comma separated (30s,1min,5min,...)")
        sp.add_argument("--no-holders", action="store_true", help="skip the exact current holder count query")
        sp.add_argument("--no-fill", action="store_true", help="don't emit empty candles for gaps")

    add_discover(sub.add_parser("discover", help="find memecoin mints on-chain"))
    h = sub.add_parser("history", help="download swap history + OHLCV CSVs")
    h.add_argument("--mint", action="append", help="mint address (repeatable); default: data/tokens.csv")
    add_history(h)
    r = sub.add_parser("run", help="discover then download history")
    add_discover(r)
    add_history(r)
    o = sub.add_parser("ohlcv", help="rebuild candles from saved trades")
    o.add_argument("--intervals", default="30s")
    o.add_argument("--no-fill", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if args.cmd == "ohlcv":
        return cmd_ohlcv(args)

    url = args.rpc or os.environ.get("SOLANA_RPC_URL") or DEFAULT_RPC_URL
    public = "api.mainnet-beta.solana.com" in url
    if public:
        log.warning("using the public Solana RPC (heavily rate limited); set SOLANA_RPC_URL for real downloads")
    rps = args.rps or (2 if public else 20)
    concurrency = args.concurrency or (2 if public else 16)

    async def runner():
        async with SolanaRPC(url, rps=rps, max_concurrency=concurrency) as rpc:
            if args.cmd == "discover":
                await cmd_discover(args, rpc)
            elif args.cmd == "history":
                await cmd_history(args, rpc)
            elif args.cmd == "run":
                await cmd_history(args, rpc, await cmd_discover(args, rpc))

    asyncio.run(runner())


if __name__ == "__main__":
    main()
