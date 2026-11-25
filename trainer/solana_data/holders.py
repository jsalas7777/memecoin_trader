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
