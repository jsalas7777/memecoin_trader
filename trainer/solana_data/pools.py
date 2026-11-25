"""Find the on-chain liquidity pools (vaults) that trade a given mint."""
from __future__ import annotations

import logging

from .constants import (
    DEX_PROGRAMS, PUMP_FUN_PROGRAM, PUMP_SWAP_PROGRAM, TOKEN_2022_PROGRAM, TOKEN_PROGRAM, WSOL_MINT,
)
from .pubkey import associated_token_address, b58decode, find_program_address, is_pda
from .rpc import SolanaRPC
from .swaps import Pool

log = logging.getLogger(__name__)

# Shared authority PDAs that own vaults of many pools (have no program-owned account data).
KNOWN_AUTHORITIES = {
    "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1": "raydium_v4",
    "GpMZbSM2GgvTKHJirzeGfMFoaZ8UR2X7F4v8vHTvxFbL": "raydium_cpmm",
    "WLHv2UAZm6z4KyaaELi5pjdbJh6RESMva1Rnn8pJVVh": "raydium_launchlab",
}
_PROGRAM_LABELS = {v: k for k, v in DEX_PROGRAMS.items()}


def pumpfun_bonding_curve(mint: str) -> str:
    return find_program_address([b"bonding-curve", b58decode(mint)], PUMP_FUN_PROGRAM)[0]


def pumpswap_canonical_pool(mint: str) -> str:
    """PumpSwap pool that pump.fun migrates a graduated token into."""
    authority = find_program_address([b"pool-authority", b58decode(mint)], PUMP_FUN_PROGRAM)[0]
    return find_program_address(
        [b"pool", (0).to_bytes(2, "little"), b58decode(authority), b58decode(mint), b58decode(WSOL_MINT)],
        PUMP_SWAP_PROGRAM,
    )[0]


async def get_token_info(rpc: SolanaRPC, mint: str) -> dict:
    acc = (await rpc.get_multiple_accounts([mint]))[0]
    if acc is None:
        raise ValueError(f"mint {mint} not found")
    program = acc["owner"]
    if program not in (TOKEN_PROGRAM, TOKEN_2022_PROGRAM):
        raise ValueError(f"{mint} is not an SPL token mint (owner {program})")
    info = acc["data"]["parsed"]["info"]
    return {
        "mint": mint,
        "token_program": program,
        "decimals": info["decimals"],
        "supply": int(info["supply"]) / 10 ** info["decimals"],
        "extensions": info.get("extensions") or [],
    }


async def discover_pools(rpc: SolanaRPC, mint: str, token: dict, max_pools: int = 4,
                         min_share: float = 0.005, scan_holders: bool = True) -> list[Pool]:
    """Find pools for `mint`.

    pump.fun / PumpSwap pools are derived from the mint. Other DEX pools (Raydium,
    Meteora, Orca, ...) are found by scanning the largest holders for PDA-owned vaults.
    """
    pools: list[Pool] = []

    # pump.fun bonding curve (exists for every pump.fun launch, even after graduation)
    # and the PumpSwap pool it graduates into. Derived directly, no indexer needed.
    bc, amm = pumpfun_bonding_curve(mint), pumpswap_canonical_pool(mint)
    bc_acc, amm_acc = await rpc.get_multiple_accounts([bc, amm], encoding="base64")
    tp = token["token_program"]
    if bc_acc is not None:
        pools.append(Pool(mint, associated_token_address(bc, mint, tp), bc, "pumpfun"))
    if amm_acc is not None and amm_acc["owner"] == PUMP_SWAP_PROGRAM:
        pools.append(Pool(mint, associated_token_address(amm, mint, tp), amm, "pumpswap"))

    if not scan_holders:
        return pools
    # Largest holders that are owned by a PDA are AMM / launchpad vaults.
    try:
        largest = await rpc.get_token_largest_accounts(mint)
    except Exception as e:  # can fail for tokens with huge holder counts
        log.warning("getTokenLargestAccounts failed for %s: %s", mint, e)
        largest = []
    supply_raw = token["supply"] * 10 ** token["decimals"]
    largest = [a for a in largest if supply_raw and int(a["amount"]) / supply_raw >= min_share]
    if largest:
        accs = await rpc.get_multiple_accounts([a["address"] for a in largest])
        vaults = []
        for a, acc in zip(largest, accs):
            if acc is None:
                continue
            owner = acc["data"]["parsed"]["info"]["owner"]
            if owner in (bc, amm) or not is_pda(owner):
                continue
            vaults.append((a["address"], owner, int(a["amount"])))
        owner_accs = await rpc.get_multiple_accounts([v[1] for v in vaults], encoding="base64") if vaults else []
        known = {p.vault for p in pools}
        for (vault, owner, _), oacc in zip(vaults, owner_accs):
            if vault in known:
                continue
            if owner in KNOWN_AUTHORITIES:
                dex = KNOWN_AUTHORITIES[owner]
            elif oacc is not None:
                dex = _PROGRAM_LABELS.get(oacc["owner"], f"program:{oacc['owner'][:8]}")
            else:
                dex = "unknown"
            pools.append(Pool(mint, vault, owner, dex))
    return pools[:max_pools]
