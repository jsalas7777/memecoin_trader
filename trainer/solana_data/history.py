"""Download the full swap history of a memecoin from Solana and store it as CSV.

Layout under the data dir:
  trades/<mint>.csv          every swap (one row per tx per pool)
  ohlcv/<interval>/<mint>.csv candles built from the trades
  state/<mint>.json          pagination cursors so re-runs are incremental
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import pandas as pd

from .holders import current_holders
from .metadata import get_metadata
from .pools import discover_pools, get_token_info
from .rpc import SolanaRPC
from .swaps import BALANCE_COLUMNS, TRADE_COLUMNS, Pool, extract_balances, extract_trade

log = logging.getLogger(__name__)


class HistoryDownloader:
    def __init__(self, rpc: SolanaRPC, data_dir: str | Path, max_signatures: int = 5000,
                 since: int | None = None, batch_size: int = 200,
                 count_holders: bool = True):
        self.rpc = rpc
        self.data_dir = Path(data_dir)
        self.max_signatures = max_signatures  # per pool, per run
        self.since = since  # unix seconds; stop paginating older than this
        self.batch_size = batch_size
        self.count_holders = count_holders
        for sub in ("trades", "balances", "state", "ohlcv"):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)

    # ---- state ---------------------------------------------------------------
    def _state_path(self, mint: str) -> Path:
        return self.data_dir / "state" / f"{mint}.json"

    def _load_state(self, mint: str) -> dict:
        p = self._state_path(mint)
        return json.loads(p.read_text()) if p.exists() else {"pools": {}}

    def _save_state(self, mint: str, state: dict) -> None:
        self._state_path(mint).write_text(json.dumps(state, indent=2))

    # ---- signatures ------------------------------------------------------------
    async def _paginate(self, address: str, budget: int, before: str | None = None,
                        until: str | None = None) -> tuple[list[dict], bool]:
        """Return (signatures newest->oldest, reached_beginning)."""
        out: list[dict] = []
        while len(out) < budget:
            limit = min(1000, budget - len(out))
            page = await self.rpc.get_signatures_for_address(address, before=before, until=until, limit=limit)
            for s in page:
                if self.since and s.get("blockTime") and s["blockTime"] < self.since:
                    return out, True
                out.append(s)
            if len(page) < limit:
                return out, True  # no more history
            before = page[-1]["signature"]
        return out, False

    # ---- main ------------------------------------------------------------------
    async def download(self, mint: str) -> pd.DataFrame:
        t0 = time.time()
        token = await get_token_info(self.rpc, mint)
        meta = await get_metadata(self.rpc, mint, token)
        state = self._load_state(mint)
        pools = await discover_pools(self.rpc, mint, token)
        known = state["pools"]
        # keep pools seen in earlier runs even if they dropped out of the largest holders
        for vault, ps in known.items():
            if vault not in {p.vault for p in pools}:
                pools.append(Pool(mint, vault, ps["owner"], ps["dex"], ps.get("quote_vault")))
        log.info("%s (%s): %d pool(s): %s", meta["symbol"] or mint[:8], mint, len(pools),
                 ", ".join(f"{p.dex}:{p.vault[:6]}" for p in pools))

        state.update({
            "mint": mint, "symbol": meta["symbol"], "name": meta["name"], "uri": meta["uri"],
            "decimals": token["decimals"], "supply": token["supply"], "token_program": token["token_program"],
        })
        df = None
        for pool in pools:
            trades: list[dict] = []
            balances: list[dict] = []
            ps = known.setdefault(pool.vault, {"owner": pool.owner, "dex": pool.dex})
            pool.quote_vault = pool.quote_vault or ps.get("quote_vault")
            sigs = await self._pool_signatures(pool, ps)
            tx_index = {s["signature"]: s.get("transactionIndex") for s in sigs}
            for i in range(0, len(sigs), self.batch_size):
                chunk = [s["signature"] for s in sigs[i : i + self.batch_size]]
                txs = await asyncio.gather(*(self.rpc.get_transaction(s) for s in chunk), return_exceptions=True)
                for sig, tx in zip(chunk, txs):
                    if isinstance(tx, Exception):
                        log.warning("    tx %s failed: %s", sig[:10], tx)
                        continue
                    if not tx:
                        continue
                    idx = tx_index.get(sig, tx.get("transactionIndex"))
                    if tr := extract_trade(tx, pool, idx):
                        trades.append(tr)
                    balances.extend(extract_balances(tx, mint, idx))
                log.info("    fetched %d/%d txs, %d trades so far", min(i + self.batch_size, len(sigs)),
                         len(sigs), len(trades))
            ps["quote_vault"] = pool.quote_vault
            # checkpoint after every pool so long backfills can be resumed
            df = self._merge_trades(mint, trades)
            self._merge_csv(self.data_dir / "balances" / f"{mint}.csv", balances, BALANCE_COLUMNS,
                            ["signature", "account"])
            state["updated_at"] = int(time.time())
            self._save_state(mint, state)
        if df is None:
            df = self._merge_trades(mint, [])
        if self.count_holders:
            holders = await current_holders(self.rpc, mint, token["token_program"])
            if holders is not None:
                state["holders_now"] = holders
                state["holders_now_at"] = int(time.time())
                log.info("  current on-chain holders: %d", holders)
        log.info("  saved %d trades total for %s in %.1fs (%d RPC calls so far)",
                 len(df), meta["symbol"] or mint[:8], time.time() - t0, self.rpc.requests_made)
        return df

    async def _pool_signatures(self, pool: Pool, ps: dict) -> list[dict]:
        """Fetch signatures newer than the last run, then continue backfilling older ones."""
        budget = self.max_signatures
        sigs: list[dict] = []
        if ps.get("newest"):
            newer, caught_up = await self._paginate(pool.vault, budget, until=ps["newest"])
            if not caught_up:
                log.warning("  %s: more than %d new txs since last run; raise --max-signatures to avoid a gap",
                            pool.vault[:6], budget)
            if newer:
                ps["newest"] = newer[0]["signature"]
            sigs.extend(newer)
            budget -= len(newer)
        if not ps.get("complete") and budget > 0:
            older, done = await self._paginate(pool.vault, budget, before=ps.get("oldest"))
            if older:
                ps.setdefault("newest", older[0]["signature"])
                ps["oldest"] = older[-1]["signature"]
            ps["complete"] = done
            sigs.extend(older)
        ok = [s for s in sigs if s.get("err") is None]
        log.info("  %s %s: %d new signatures, %d successful (failed txs are skipped)",
                 pool.dex, pool.vault[:6], len(sigs), len(ok))
        return ok

    @staticmethod
    def _merge_csv(path: Path, new: list[dict], columns: list[str], key: list[str]) -> pd.DataFrame:
        df = pd.DataFrame(new, columns=columns)
        if path.exists():
            old = pd.read_csv(path)
            df = old if df.empty else pd.concat([old, df], ignore_index=True)
        if not df.empty:
            df = df.drop_duplicates(key, keep="last")
            df = df.sort_values(["slot", "tx_index"], na_position="last").reset_index(drop=True)
        df.to_csv(path, index=False)
        return df

    def _merge_trades(self, mint: str, new: list[dict]) -> pd.DataFrame:
        path = self.data_dir / "trades" / f"{mint}.csv"
        frames = [pd.DataFrame(new, columns=TRADE_COLUMNS)]
        if path.exists():
            frames.insert(0, pd.read_csv(path))
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
            not f.empty for f in frames) else frames[-1]
        if not df.empty:
            df = df.drop_duplicates(["signature", "pool_vault"], keep="last")
            df = df.sort_values(["slot", "tx_index"], na_position="last").reset_index(drop=True)
            df["datetime"] = pd.to_datetime(df["block_time"], unit="s", utc=True)
        df.to_csv(path, index=False)
        return df
