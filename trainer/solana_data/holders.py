"""Holder counts: exact current count from chain, and a historical series from balances."""
from __future__ import annotations

import base64
import heapq
import logging
from functools import lru_cache

import pandas as pd

from .pubkey import b58encode, is_pda
from .rpc import SolanaRPC

log = logging.getLogger(__name__)
_EPS = 1e-9  # ignore float residue when balances net to zero


async def current_holders(rpc: SolanaRPC, mint: str, token_program: str) -> int | None:
    """Exact number of wallets holding > 0 of `mint` right now (getProgramAccounts).

    Returns None if the RPC refuses the query (some providers disable it or it times out
    for tokens with very many holders).
    """
    filters = [{"memcmp": {"offset": 0, "bytes": mint}}]
    try:
        res = await rpc.call(
            "getProgramAccounts",
            [token_program, {"encoding": "base64", "filters": filters,
                             "dataSlice": {"offset": 32, "length": 40}}],  # owner(32) + amount(8)
            max_retries=3,
        )
    except Exception as e:
        log.warning("holder count failed for %s: %s", mint, e)
        return None
    owners = set()
    for acc in res:
        raw = base64.b64decode(acc["account"]["data"][0])
        if len(raw) == 40 and int.from_bytes(raw[32:40], "little") > 0:
            owners.add(b58encode(raw[:32]))
    return len(owners)


@lru_cache(maxsize=200_000)
def _is_program_owned(owner: str) -> bool:
    try:
        return is_pda(owner)
    except Exception:
        return False


def holder_stats(balances: pd.DataFrame | None, interval: str, supply: float | None = None) -> pd.DataFrame:
    """Per-candle holder statistics, indexed by candle start (UTC).

    holders    wallets holding > 0 at candle close (pools included, like block explorers)
    top10_pct  % of total supply held by the 10 largest holders at candle close,
               excluding pools and other program-owned (PDA) accounts. Those are
               liquidity or locks, not people who can dump.

    Built by replaying the absolute post-trade balances of every token account seen in
    pool transactions. Wallets that only ever received tokens by plain transfer (never
    traded through a tracked pool) are not seen, so `holders` is a slight undercount.
    """
    cols = ["holders", "top10_pct"]
    if balances is None or balances.empty:
        return pd.DataFrame(columns=cols, dtype="float64")
    b = balances.dropna(subset=["block_time"]).sort_values(["slot", "tx_index"], na_position="last")
    buckets = pd.to_datetime(b["block_time"], unit="s", utc=True).dt.floor(interval)
    acct_bal: dict[str, float] = {}
    owner_bal: dict[str, float] = {}
    people: dict[str, float] = {}  # wallet owners only (no PDAs)
    count = 0
    rows, index = [], []

    def snapshot(bucket):
        top = heapq.nlargest(10, people.values())
        index.append(bucket)
        rows.append((count, 100 * sum(top) / supply if supply else float("nan")))

    current = None
    for acct, owner, bal, bucket in zip(b["account"], b["owner"], b["balance"], buckets):
        if current is not None and bucket != current:
            snapshot(current)
        current = bucket
        prev = acct_bal.get(acct, 0.0)
        acct_bal[acct] = bal
        before = owner_bal.get(owner, 0.0)
        after = before - prev + bal
        owner_bal[owner] = after
        was, now = before > _EPS, after > _EPS
        if was != now:
            count += 1 if now else -1
        if not _is_program_owned(owner):
            if now:
                people[owner] = after
            else:
                people.pop(owner, None)
    if current is not None:
        snapshot(current)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index), columns=cols)


def bundle_stats(balances: pd.DataFrame | None, interval: str, supply: float | None,
                 launch_slot: int | None) -> tuple[pd.DataFrame, dict]:
    """Launch-bundle detection and what the bundle still holds over time.

    Bundle wallets = every wallet (program-owned/PDA accounts excluded) whose balance went
    up in the coin's launch slot, the same block as the create transaction. That includes
    the dev's own buy in the create tx and any wallets that bought in the same block
    (usually a Jito bundle controlled by the dev or a sniper group).

    Returns (per-candle frame with `bundle_pct` = % of supply the bundle wallets hold at
    candle close, {"bundle_launch_pct": % of supply bought in the launch slot,
    "bundle_wallets": number of bundle wallets}).
    Needs the coin's history from creation (launch_slot); otherwise everything is unknown.
    """
    unknown = (pd.DataFrame(columns=["bundle_pct"], dtype="float64"),
               {"bundle_launch_pct": float("nan"), "bundle_wallets": float("nan")})
    if balances is None or balances.empty or launch_slot is None or not supply:
        return unknown
    b = balances.dropna(subset=["block_time"]).sort_values(["slot", "tx_index"], na_position="last")
    launch = b[b["slot"] == launch_slot]
    start: dict[str, float] = {}
    for acct, owner, bal in zip(launch["account"], launch["owner"], launch["balance"]):
        if not _is_program_owned(owner):
            start[acct] = bal  # last balance in the launch slot
    owners = {o for a, o in zip(launch["account"], launch["owner"]) if a in start and start[a] > _EPS}
    accounts = {a for a in start if start[a] > _EPS}
    launch_pct = 100 * sum(start[a] for a in accounts) / supply
    info = {"bundle_launch_pct": launch_pct, "bundle_wallets": float(len(owners))}
    if not owners:
        return pd.DataFrame(columns=["bundle_pct"], dtype="float64"), info

    # replay every account owned by a bundle wallet (they may buy more / sell later)
    mine = b[b["owner"].isin(owners)]
    buckets = pd.to_datetime(mine["block_time"], unit="s", utc=True).dt.floor(interval)
    held: dict[str, float] = {}
    rows, index, current = [], [], None
    for acct, bal, bucket in zip(mine["account"], mine["balance"], buckets):
        if current is not None and bucket != current:
            index.append(current)
            rows.append(100 * sum(held.values()) / supply)
        current = bucket
        held[acct] = bal
    index.append(current)
    rows.append(100 * sum(held.values()) / supply)
    return pd.DataFrame({"bundle_pct": rows}, index=pd.DatetimeIndex(index)), info
