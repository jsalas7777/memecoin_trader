"""Discover memecoin mints straight from on-chain activity."""
from __future__ import annotations

import asyncio
import logging
from collections import Counter

from .constants import DEX_PROGRAMS, NON_MEME_MINTS, PUMP_FUN_MINT_AUTHORITY
from .rpc import SolanaRPC
from .swaps import account_keys

log = logging.getLogger(__name__)


async def _signatures(rpc: SolanaRPC, address: str, count: int) -> list[dict]:
    sigs: list[dict] = []
    before = None
    while len(sigs) < count:
        page = await rpc.get_signatures_for_address(address, before=before, limit=min(1000, count - len(sigs)))
        if not page:
            break
        sigs.extend(s for s in page if s.get("err") is None)
        before = page[-1]["signature"]
    return sigs[:count]


async def _fetch_all(rpc: SolanaRPC, signatures: list[str]) -> list[dict]:
    txs = await asyncio.gather(*(rpc.get_transaction(s) for s in signatures), return_exceptions=True)
    out = []
    for s, t in zip(signatures, txs):
        if isinstance(t, Exception):
            log.warning("failed to fetch %s: %s", s, t)
        elif t:
            out.append(t)
    return out


async def discover_new_pumpfun(rpc: SolanaRPC, limit: int = 50) -> list[dict]:
    """Newest pump.fun launches, found via the pump.fun mint authority's history."""
    sigs = await _signatures(rpc, PUMP_FUN_MINT_AUTHORITY, limit)
    log.info("scanning %d pump.fun creation txs", len(sigs))
    tokens = []
    for tx in await _fetch_all(rpc, [s["signature"] for s in sigs]):
        meta = tx["meta"]
        pre = {b["mint"] for b in meta.get("preTokenBalances") or []}
        new = {b["mint"] for b in meta.get("postTokenBalances") or []} - pre - NON_MEME_MINTS
        keys = account_keys(tx)
        for mint in new:
            if mint not in keys:
                continue
            tokens.append({
                "mint": mint,
                "created_at": tx.get("blockTime"),
                "creator": keys[0],
                "creation_signature": tx["transaction"]["signatures"][0],
                "source": "pumpfun_new",
            })
    return tokens


async def discover_active(rpc: SolanaRPC, programs: list[str] | None = None,
                          sample_txs: int = 300, top: int = 25) -> list[dict]:
    """Most-traded mints in the latest transactions of memecoin DEX programs."""
    programs = programs or ["pumpfun", "pumpswap"]
    counts: Counter[str] = Counter()
    traders: dict[str, set] = {}
    for name in programs:
        program = DEX_PROGRAMS.get(name, name)
        sigs = await _signatures(rpc, program, sample_txs)
        log.info("scanning %d recent %s txs", len(sigs), name)
        for tx in await _fetch_all(rpc, [s["signature"] for s in sigs]):
            mints = {b["mint"] for b in (tx["meta"].get("postTokenBalances") or [])} - NON_MEME_MINTS
            signer = account_keys(tx)[0]
            for m in mints:
                counts[m] += 1
                traders.setdefault(m, set()).add(signer)
    ranked = sorted(counts, key=lambda m: (len(traders[m]), counts[m]), reverse=True)[:top]
    return [
        {"mint": m, "recent_txs": counts[m], "recent_traders": len(traders[m]), "source": "active"}
        for m in ranked
    ]
