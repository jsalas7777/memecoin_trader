#!/usr/bin/env python3
"""Download memecoin history for training. No API key needed.

  python download_free.py                       # 500 coins launched in the last 3 days
  python download_free.py --coins 2000 --days 7
  python download_free.py --coins all --days 1  # every pump.fun launch of the last day (slow)

Steps (each with a progress bar):
  1. Find coins: pump.fun launches, sampled evenly over the time window, dead ones
     included. Read from the free public Solana RPC (GMGN itself only lists the newest ~60).
  2. Download each coin from GMGN with GMGN's public test key: 30s candles from launch,
     holder history and a snapshot.
  3. Build the training data (dataset_30s.csv + sequences_30s.npz, MODEL.md, DATASET.md).

Rate limits are announced as they happen, with how long the script will pause, and the
script slows itself down after each one. Stop any time with Ctrl+C; running the same
command again continues where it stopped (finished coins are never downloaded twice).

Uses your own keys instead if they are set (much faster): GMGN_API_KEY, SOLANA_RPC_URL.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from download_gmgn import download_coin, merge_catalog, parse_duration, update_catalog
from solana_data.constants import PUMP_FUN_MINT_AUTHORITY
from solana_data.discovery import launches_in_tx
from solana_data.gmgn import DEMO_KEY, PLAN_POINTS, GMGNClient, GMGNError
from solana_data.rpc import DEFAULT_RPC_URL, SolanaRPC

HERE = Path(__file__).resolve().parent


def clock(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time()).strftime("%H:%M:%S")


def fmt_duration(s: float) -> str:
    s = int(s)
    return f"{s // 3600}h {s % 3600 // 60}m" if s >= 3600 else f"{s // 60}m {s % 60}s" if s >= 60 else f"{s}s"


def say(msg: str) -> None:
    tqdm.write(msg)


class TqdmLogHandler(logging.Handler):
    def emit(self, record):
        say(self.format(record))


# ---- step 1: find coins ------------------------------------------------------------
class RPCReporter:
    """Reports public-RPC rate limits without flooding the screen."""

    def __init__(self):
        self.limits = 0
        self._last = 0.0

    def __call__(self, method, err, sleep_s):
        code = getattr(err, "code", None)
        if code not in (429, -32005):
            return
        self.limits += 1
        if time.time() - self._last > 30:
            self._last = time.time()
            say(f"⚠ {clock()} Solana RPC rate limit (HTTP 429 on {method}): retrying automatically "
                f"({self.limits} so far). The free public RPC is shared, this is normal.")


async def find_coins(n: int | None, days: float, min_age_s: int, known_sigs: set[str],
                     reporter: RPCReporter) -> list[dict]:
    public = not os.environ.get("SOLANA_RPC_URL")
    rpc = SolanaRPC(rps=2 if public else 20, max_concurrency=2 if public else 16, on_retry=reporter)
    now = int(time.time())
    newest, oldest = now - min_age_s, now - int(days * 86400)
    try:
        # a) list every launch in the window (1,000 launches per request)
        sigs: list[dict] = []
        bar = tqdm(total=newest - oldest, desc="1/3 Listing launches", unit="s", unit_scale=False,
                   bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}", dynamic_ncols=True)
        before, done = None, False
        while not done:
            page = await rpc.get_signatures_for_address(PUMP_FUN_MINT_AUTHORITY, before=before, limit=1000)
            if not page:
                break
            for s in page:
                t = s.get("blockTime") or 0
                if t > newest or s.get("err") is not None:
                    continue
                if t < oldest:
                    done = True
                    break
                sigs.append(s)
            before = page[-1]["signature"]
            t_last = max(oldest, min(newest, page[-1].get("blockTime") or oldest))
            bar.n = newest - t_last
            bar.set_postfix_str(f"{len(sigs):,} launches, back to {datetime.fromtimestamp(t_last):%b %d %H:%M}")
            bar.refresh()
        bar.n = bar.total
        bar.close()

        sigs = [s for s in sigs if s["signature"] not in known_sigs]
        if n is not None and len(sigs) > n:  # even sample over the window = no survivorship bias
            idx = sorted({round(i * (len(sigs) - 1) / max(n - 1, 1)) for i in range(n)})
            sigs = [sigs[i] for i in idx]
        if not sigs:
            return []

        # b) read each creation tx to get the coin's address
        coins: list[dict] = []
        bar = tqdm(total=len(sigs), desc="1/3 Reading launches", unit="coin", dynamic_ncols=True)

        async def one(sig):
            try:
                tx = await rpc.get_transaction(sig)
                return launches_in_tx(tx) if tx else []
            except Exception as e:  # noqa: BLE001 - one bad tx must not stop discovery
                say(f"  skipped launch {sig[:12]}…: {e}")
                return []

        for fut in asyncio.as_completed([one(s["signature"]) for s in sigs]):
            coins.extend(await fut)
            bar.update(1)
            bar.set_postfix_str(f"{len(coins):,} coins")
        bar.close()
        return coins
    finally:
        await rpc.close()


# ---- step 2: download from GMGN ------------------------------------------------------
def download_all(args, mints: list[str], created: dict, using_demo: bool) -> dict:
    data = Path(args.data_dir)
    stats = {"ok": 0, "failed": 0, "candles": 0, "failed_mints": []}
    bar = tqdm(total=len(mints), desc="2/3 Downloading", unit="coin", dynamic_ncols=True)
    state = {"kind": ""}

    def postfix(extra: str = "") -> None:
        bar.set_postfix_str(f"ok {stats['ok']} | failed {stats['failed']} | rate limits {gm.rate_limits}"
                            + (f" | {extra}" if extra else ""))

    def on_event(kind: str, message: str, wait: float) -> None:
        state["kind"] = kind
        if kind == "rate_limit":
            say(f"⚠ {clock()} GMGN rate limit ({message}): pausing {fmt_duration(wait)} until "
                f"{clock(time.time() + wait)}, then continuing slower ({gm.points:.1f} points/s)."
                + (" The public test key is shared by everyone." if using_demo and gm.rate_limits == 1 else ""))
        else:
            say(f"⚠ {clock()} GMGN {kind} error ({message}): retrying in {fmt_duration(wait)}.")

    def countdown(wait: float) -> None:
        end = time.time() + wait
        label = "⏸ rate limited" if state["kind"] == "rate_limit" else f"⏸ {state['kind']} error"
        while (left := end - time.time()) > 0:
            postfix(f"{label}, resuming in {fmt_duration(left)}")
            bar.refresh()
            time.sleep(min(1.0, left))
        postfix()

    gm = GMGNClient(DEMO_KEY if using_demo else None, points_per_s=args.points, on_event=on_event, sleep=countdown)
    streak = 0
    try:
        for mint in mints:
            postfix(f"{mint[:6]}…")
            try:
                row = download_coin(gm, data, mint, created.get(mint), holders=not args.no_holders)
                update_catalog(data, row)
                stats["ok"] += 1
                stats["candles"] += row["gmgn_candles"]
                streak = 0
            except GMGNError as e:
                stats["failed"] += 1
                stats["failed_mints"].append(mint)
                streak += 1
                say(f"✗ {clock()} {mint}: {e}")
                if streak >= 10:
                    say(f"⚠ {clock()} 10 coins failed in a row: GMGN is refusing requests. Pausing 5 minutes.")
                    countdown(300)
                    streak = 0
            bar.update(1)
            postfix()
    finally:
        bar.close()
        stats["requests"] = gm.requests_made
        stats["rate_limits"] = gm.rate_limits
        stats["rate_limit_wait_s"] = gm.rate_limit_wait_s
        gm.close()
    return stats


# ---- step 3: build ---------------------------------------------------------------------
def build(args) -> None:
    data = Path(args.data_dir)
    ds = data / "dataset"
    # docs go next to the code for the default data folder, else into the custom folder
    docs = HERE if data.resolve() == (HERE / "data").resolve() else data
    steps = (("build_dataset.py", ["--data-dir", str(data), "--md", str(docs / "DATASET.md")]),
             ("build_sequences.py", ["--dataset", str(ds / "dataset_30s.csv"), "--out", str(ds / "sequences_30s.npz"),
                                     "--md", str(docs / "MODEL.md")]))
    for i, (script, extra) in enumerate(steps, 1):
        say(f"3/3 Building ({i}/2): {script} …")
        r = subprocess.run([sys.executable, str(HERE / script), *extra], cwd=HERE, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip().splitlines()
        if r.returncode != 0:
            say(f"✗ {script} stopped: {out[-1] if out else 'unknown error'}")
            say("  (usually: not enough coins yet; download more and run again)")
            return
        for line in out:
            msg = line.split(" INFO ")[-1]
            if msg.startswith(("dataset:", "skipped")) or " samples x " in msg or "[FAIL]" in msg:
                say("  " + msg.strip())
        checks = {k: sum(f"[{k}]" in line for line in out) for k in ("PASS", "WARN", "FAIL")}
        if any(checks.values()):
            say(f"  input checks: {checks['PASS']} passed, {checks['WARN']} warnings, {checks['FAIL']} failed")
    say(f"✓ Training data ready: {ds / 'sequences_30s.npz'} (inputs described in {docs / 'MODEL.md'})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coins", default="500", help="how many coins to download, or 'all' (default 500)")
    p.add_argument("--days", type=float, default=3, help="sample coins launched in the last N days (default 3)")
    p.add_argument("--min-age", default="2h",
                   help="skip coins younger than this; their fate isn't known yet (default 2h)")
    p.add_argument("--points", type=float, default=PLAN_POINTS["free"],
                   help="GMGN points per second to use (default 5 = free plan; Plus 20, Pro 50)")
    p.add_argument("--no-holders", action="store_true", help="skip holder history (faster)")
    p.add_argument("--no-build", action="store_true", help="only download, don't build the training data")
    p.add_argument("--data-dir", default=str(HERE / "data"))
    args = p.parse_args()
    n = None if args.coins.lower() == "all" else int(args.coins)

    logging.basicConfig(level=logging.WARNING, handlers=[TqdmLogHandler()], format="⚠ %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    data = Path(args.data_dir)
    using_demo = not os.environ.get("GMGN_API_KEY")
    rpc_url = os.environ.get("SOLANA_RPC_URL") or DEFAULT_RPC_URL
    say(f"GMGN: {'public test key (no key needed, shared by everyone: slow, expect rate limits)' if using_demo else 'your GMGN_API_KEY'}")
    say(f"Solana RPC: {'free public endpoint' if rpc_url == DEFAULT_RPC_URL else 'your SOLANA_RPC_URL'}")
    say(f"Target: {'every' if n is None else f'{n:,}'} coin(s) launched in the last {args.days:g} days "
        f"(at least {args.min_age} old), saved to {data}\n")

    started = time.time()
    reporter = RPCReporter()
    stats: dict = {}
    try:
        # coins already listed but not downloaded yet (e.g. a previous run was stopped)
        cat_path = data / "tokens.csv"
        cat = pd.read_csv(cat_path) if cat_path.exists() else pd.DataFrame(columns=["mint", "created_at"])
        cutoff = time.time() - parse_duration(args.min_age)
        oldest = time.time() - args.days * 86400
        sdir = data / "gmgn" / "state"
        in_window = cat["created_at"].between(oldest, cutoff) if "created_at" in cat else pd.Series(False, index=cat.index)
        pending = [m for m in cat.loc[in_window, "mint"] if not (sdir / f"{m}.json").exists()]
        need = None if n is None else max(0, n - len(pending))
        if need is None or need > 0:
            known = set(cat["creation_signature"].dropna()) if "creation_signature" in cat else set()
            coins = asyncio.run(find_coins(need, args.days, parse_duration(args.min_age), known, reporter))
            added = merge_catalog(data, coins)
            say(f"✓ Found {added:,} new coins" + (f" (+{len(pending):,} left from a previous run)" if pending else ""))
            cat = pd.read_csv(cat_path) if cat_path.exists() else cat
            in_window = cat["created_at"].between(oldest, cutoff)
            pending = [m for m in cat.loc[in_window, "mint"] if not (sdir / f"{m}.json").exists()]
        else:
            say(f"✓ {len(pending):,} coins already listed from a previous run, continuing with those")
        mints = pending if n is None else pending[:n]
        created = dict(zip(cat["mint"], cat["created_at"]))
        if mints:
            stats = download_all(args, mints, created, using_demo)
        else:
            say("Nothing new to download (all coins in this window are done). Try --days 7 or more --coins.")
    except KeyboardInterrupt:
        say("\nStopped. Run the same command again to continue where it left off.")
        args.no_build = True

    say("\n── Summary " + "─" * 40)
    if stats:
        say(f"Coins downloaded : {stats['ok']:,}" + (f"  ({stats['failed']:,} failed, they are retried next run)"
                                                     if stats["failed"] else ""))
        say(f"Candles          : {stats['candles']:,} (30s)")
        say(f"GMGN requests    : {stats['requests']:,}")
        say(f"GMGN rate limits : {stats['rate_limits']:,}" + (
            f" (paused {fmt_duration(stats['rate_limit_wait_s'])} in total)" if stats["rate_limits"] else " ✓"))
    say(f"Solana RPC limits: {reporter.limits:,}" + (" (retried automatically)" if reporter.limits else " ✓"))
    say(f"Time             : {fmt_duration(time.time() - started)}")
    if stats and stats.get("rate_limits", 0) > 5 and using_demo:
        say("Tip: a free GMGN key (https://gmgn.ai/ai, then export GMGN_API_KEY=...) avoids most rate limits.")
    if not args.no_build and stats.get("ok"):
        say("")
        build(args)


if __name__ == "__main__":
    main()
