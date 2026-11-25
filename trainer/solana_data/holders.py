"""Holder counts: exact current count from chain, and a historical series from balances."""
from __future__ import annotations

import base64
import logging

import pandas as pd

from .pubkey import b58encode
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


def holder_series(balances: pd.DataFrame, exclude_owners: set[str] | None = None) -> pd.Series:
    """Running count of wallets with a positive balance, indexed by time (UTC).

    Built by replaying the absolute post-trade balances of every token account seen
    in pool transactions. Wallets that only ever received tokens by plain transfer
    (never traded through a tracked pool) are not seen, so this is a slight undercount.
    """
    if balances is None or balances.empty:
        return pd.Series(dtype="float64")
    exclude_owners = exclude_owners or set()
    b = balances.dropna(subset=["block_time"]).sort_values(["slot", "tx_index"], na_position="last")
    acct_bal: dict[str, float] = {}
    owner_bal: dict[str, float] = {}
    count = 0
    times, counts = [], []
    for acct, owner, bal, t in zip(b["account"], b["owner"], b["balance"], b["block_time"]):
        prev = acct_bal.get(acct, 0.0)
        acct_bal[acct] = bal
        if owner in exclude_owners:
            continue
        before = owner_bal.get(owner, 0.0)
        after = before - prev + bal
        owner_bal[owner] = after
        was, now = before > _EPS, after > _EPS
        if was != now:
            count += 1 if now else -1
        times.append(t)
        counts.append(count)
    s = pd.Series(counts, index=pd.to_datetime(times, unit="s", utc=True))
    return s.groupby(level=0).last()
