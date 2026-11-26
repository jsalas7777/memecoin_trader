"""Async Solana JSON-RPC client with rate limiting and retries."""
from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_RPC_URL = "https://api.mainnet-beta.solana.com"
# Highest transaction version this client understands (v1 txs are live on mainnet).
MAX_TX_VERSION = 1
RETRYABLE_RPC_CODES = {-32005, -32014, -32016, -32603, 429}


class RPCError(Exception):
    def __init__(self, code: int | None, message: str):
        super().__init__(f"RPC error {code}: {message}")
        self.code = code


class SolanaRPC:
    """Thin JSON-RPC wrapper.

    The public mainnet endpoint is heavily rate limited (roughly 10 req/s and
    stricter on getTransaction). For real downloads set SOLANA_RPC_URL to a
    dedicated provider (Helius, QuickNode, Triton, ...) and raise --rps.
    """

    def __init__(
        self,
        url: str | None = None,
        rps: float = 8.0,
        max_concurrency: int = 8,
        max_retries: int = 8,
        timeout: float = 30.0,
        commitment: str = "confirmed",
        on_retry=None,
    ):
        self.url = url or os.environ.get("SOLANA_RPC_URL") or DEFAULT_RPC_URL
        self.commitment = commitment
        self.max_retries = max_retries
        self._min_interval = 1.0 / rps if rps > 0 else 0.0
        self._next_slot = 0.0
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(max_concurrency)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Content-Type": "application/json"},
            limits=httpx.Limits(max_connections=max_concurrency * 2),
        )
        self._id = 0
        self.requests_made = 0
        self.retries = 0
        self.on_retry = on_retry  # callable(method, error, sleep_s), e.g. to report rate limits

    async def __aenter__(self) -> "SolanaRPC":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def _throttle(self) -> None:
        if not self._min_interval:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self._min_interval
        if wait > 0:
            await asyncio.sleep(wait)

    async def call(self, method: str, params: list | None = None, max_retries: int | None = None) -> Any:
        retries = self.max_retries if max_retries is None else max_retries
        self._id += 1
        payload = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or []}
        delay = 1.0
        for attempt in range(retries + 1):
            async with self._sem:
                await self._throttle()
                try:
                    resp = await self._client.post(self.url, json=payload)
                    self.requests_made += 1
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    err: Exception = e
                    retry_after = None
                else:
                    retry_after = resp.headers.get("retry-after")
                    if resp.status_code == 200:
                        body = resp.json()
                        if "error" not in body:
                            return body.get("result")
                        code = body["error"].get("code")
                        err = RPCError(code, body["error"].get("message", ""))
                        if code not in RETRYABLE_RPC_CODES:
                            raise err
                    elif resp.status_code in (429, 500, 502, 503, 504):
                        err = RPCError(resp.status_code, resp.text[:200])
                    else:
                        raise RPCError(resp.status_code, resp.text[:200])
            if attempt == retries:
                raise err
            sleep_for = float(retry_after) if retry_after and retry_after.isdigit() else delay
            sleep_for += random.uniform(0, 0.5)
            log.debug("%s failed (%s); retrying in %.1fs", method, err, sleep_for)
            self.retries += 1
            if self.on_retry:
                self.on_retry(method, err, sleep_for)
            await asyncio.sleep(sleep_for)
            delay = min(delay * 2, 30.0)
        raise RuntimeError("unreachable")

    # ---- convenience wrappers ------------------------------------------------
    async def get_signatures_for_address(
        self, address: str, before: str | None = None, until: str | None = None, limit: int = 1000
    ) -> list[dict]:
        cfg: dict = {"limit": limit, "commitment": self.commitment}
        if before:
            cfg["before"] = before
        if until:
            cfg["until"] = until
        return await self.call("getSignaturesForAddress", [address, cfg]) or []

    async def get_transaction(self, signature: str) -> dict | None:
        return await self.call(
            "getTransaction",
            [
                signature,
                {
                    "encoding": "jsonParsed",
                    "maxSupportedTransactionVersion": MAX_TX_VERSION,
                    "commitment": self.commitment,
                },
            ],
        )

    async def get_multiple_accounts(self, addresses: list[str], encoding: str = "jsonParsed") -> list[dict | None]:
        out: list[dict | None] = []
        for i in range(0, len(addresses), 100):
            chunk = addresses[i : i + 100]
            res = await self.call(
                "getMultipleAccounts", [chunk, {"encoding": encoding, "commitment": self.commitment}]
            )
            out.extend(res["value"])
        return out

    async def get_token_largest_accounts(self, mint: str, max_retries: int | None = 2) -> list[dict]:
        res = await self.call("getTokenLargestAccounts", [mint, {"commitment": self.commitment}], max_retries)
        return res["value"]

    async def get_token_supply(self, mint: str) -> dict:
        res = await self.call("getTokenSupply", [mint, {"commitment": self.commitment}])
        return res["value"]
