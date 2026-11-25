"""Turn raw Solana transactions into swap trades for a given pool.

The approach is DEX-agnostic: every AMM / bonding curve keeps the memecoin in
a token account ("vault") owned by a PDA (the pool or its authority). For each
transaction touching that vault we read the vault's token balance change and
the opposite change in the pool's quote reserve:

* pump.fun bonding curve -> SOL held as lamports on the bonding-curve account
* PumpSwap / Raydium / Meteora / Orca -> a WSOL (or USDC) vault with the same owner

Liquidity adds/removes, pool creation and migrations move both sides in the
same direction (or create the vault) and are ignored, so only real swaps remain.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .constants import LAMPORTS_PER_SOL, QUOTE_MINTS, WSOL_MINT


@dataclass
class Pool:
    mint: str
    vault: str          # token account holding the memecoin
    owner: str          # PDA that owns the vault (bonding curve / pool / authority)
    dex: str
    quote_vault: str | None = None  # learned from transactions when token-quoted
    extra: dict = field(default_factory=dict)


TRADE_COLUMNS = [
    "signature", "slot", "tx_index", "block_time", "dex", "pool_vault", "side",
    "token_amount", "quote_amount", "quote", "price", "trader",
    "token_reserve", "quote_reserve",
]


def _balances(meta: dict, key: str) -> dict[int, tuple[str, str | None, int, int]]:
    out = {}
    for b in meta.get(key) or []:
        amt = b["uiTokenAmount"]
        out[b["accountIndex"]] = (b["mint"], b.get("owner"), int(amt["amount"]), amt["decimals"])
    return out


def account_keys(tx: dict) -> list[str]:
    return [k["pubkey"] if isinstance(k, dict) else k for k in tx["transaction"]["message"]["accountKeys"]]


def extract_trade(tx: dict, pool: Pool, tx_index: int | None = None) -> dict | None:
    meta = tx.get("meta") or {}
    if meta.get("err") is not None:
        return None
    keys = account_keys(tx)
    try:
        vault_idx = keys.index(pool.vault)
    except ValueError:
        return None

    pre, post = _balances(meta, "preTokenBalances"), _balances(meta, "postTokenBalances")
    if vault_idx not in pre:
        return None  # vault created in this tx (launch / pool init), not a swap
    _, _, pre_amt, decimals = pre[vault_idx]
    post_amt = post[vault_idx][2] if vault_idx in post else 0
    token_delta = post_amt - pre_amt
    if token_delta == 0:
        return None
    sign = 1 if token_delta > 0 else -1

    quote_raw = quote_decimals = quote_reserve_raw = None
    quote_mint = None

    # 1) token-denominated quote vault (WSOL/USDC) owned by the same PDA
    candidates = []
    for idx in set(pre) | set(post):
        if idx == vault_idx:
            continue
        rec = post.get(idx) or pre.get(idx)
        mint, owner, _, dec = rec
        if mint not in QUOTE_MINTS or owner != pool.owner:
            continue
        d = (post[idx][2] if idx in post else 0) - (pre[idx][2] if idx in pre else 0)
        if d != 0 and (d > 0) != (sign > 0):
            candidates.append((keys[idx], idx, d, dec, mint))

    chosen = None
    if pool.quote_vault:
        chosen = next((c for c in candidates if c[0] == pool.quote_vault), None)
    if chosen is None and len(candidates) == 1:
        chosen = candidates[0]
        if pool.quote_vault is None:
            pool.quote_vault = chosen[0]
    if chosen is not None:
        _, idx, d, quote_decimals, quote_mint = chosen
        quote_raw = abs(d)
        quote_reserve_raw = post[idx][2] if idx in post else 0
    else:
        # 2) native SOL held as lamports by the owner (pump.fun bonding curve)
        if pool.owner in keys:
            oidx = keys.index(pool.owner)
            d = meta["postBalances"][oidx] - meta["preBalances"][oidx]
            if d != 0 and (d > 0) != (sign > 0):
                quote_raw, quote_decimals, quote_mint = abs(d), 9, WSOL_MINT
                quote_reserve_raw = meta["postBalances"][oidx]
    if quote_raw is None:
        return None

    token_amount = abs(token_delta) / 10**decimals
    quote_amount = quote_raw / 10**quote_decimals
    signer = next(
        (k["pubkey"] for k in tx["transaction"]["message"]["accountKeys"] if isinstance(k, dict) and k.get("signer")),
        keys[0],
    )
    return {
        "signature": tx["transaction"]["signatures"][0],
        "slot": tx["slot"],
        "tx_index": tx_index,
        "block_time": tx.get("blockTime"),
        "dex": pool.dex,
        "pool_vault": pool.vault,
        # tokens leaving the pool = someone bought
        "side": "sell" if sign > 0 else "buy",
        "token_amount": token_amount,
        "quote_amount": quote_amount,
        "quote": QUOTE_MINTS[quote_mint],
        "price": quote_amount / token_amount,
        "trader": signer,
        "token_reserve": post_amt / 10**decimals,
        "quote_reserve": quote_reserve_raw / 10**quote_decimals,
    }


BALANCE_COLUMNS = ["signature", "slot", "tx_index", "block_time", "account", "owner", "balance"]


def extract_balances(tx: dict, mint: str, tx_index: int | None = None) -> list[dict]:
    """Post-transaction balances of every token account of `mint` whose balance changed.

    These are absolute balances (not deltas), so replaying them in order gives the
    exact balance of every wallet seen trading, which is used to count holders over time.
    """
    meta = tx.get("meta") or {}
    if meta.get("err") is not None:
        return []
    keys = account_keys(tx)
    pre, post = _balances(meta, "preTokenBalances"), _balances(meta, "postTokenBalances")
    rows = []
    for idx in set(pre) | set(post):
        rec = post.get(idx) or pre.get(idx)
        if rec[0] != mint:
            continue
        before = pre[idx][2] if idx in pre else 0
        after = post[idx][2] if idx in post else 0  # missing from post = account closed
        if before == after:
            continue
        rows.append({
            "signature": tx["transaction"]["signatures"][0],
            "slot": tx["slot"],
            "tx_index": tx_index,
            "block_time": tx.get("blockTime"),
            "account": keys[idx],
            "owner": rec[1],
            "balance": after / 10 ** rec[3],
        })
    return rows
