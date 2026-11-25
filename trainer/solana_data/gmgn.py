"""Client for the official GMGN OpenAPI (https://openapi.gmgn.ai), read-only market data.

Auth (from GMGN's official gmgn-cli source): header `X-APIKEY`, plus `timestamp`
(unix seconds) and a fresh `client_id` UUID as query params on every request.
Read-only endpoints need no signature. Create a key at https://gmgn.ai/ai and
export it as GMGN_API_KEY.

What GMGN provides here:
  * 30s (and 1m, 5m, ...) OHLCV candles from a coin's launch, including dead coins
  * a current snapshot per coin (holders, top-10 rate, liquidity, launchpad status, dev info)
  * live lists of new / about-to-graduate / graduated pump.fun coins ("trenches")
  * holding periods (start/end) of up to 100 wallets -> exact holder history for small coins
What it does NOT provide: trade counts per candle, or top-10 / liquidity history.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid

import httpx

log = logging.getLogger(__name__)

HOST = "https://openapi.gmgn.ai"
# GMGN's documented public key: read-only commands, "for testing only".
DEMO_KEY = "gmgn_solbscbaseethmonadtron"
KLINE_PAGE = 100  # GMGN returns at most 100 candles per request (the latest ones in [from, to])
RATE_LIMIT_ERRORS = {"RATE_LIMIT_EXCEEDED", "RATE_LIMIT_BANNED", "RATE_LIMIT_BANNED_MAX", "ERROR_RATE_LIMIT_BLOCKED"}


class GMGNError(Exception):
    pass


class GMGNClient:
    """Synchronous, rate-limited GMGN client.

    rps: request budget per second. GMGN uses a leaky bucket per key (Free 5, Plus 20,
    Pro 50 weight/s) and kline calls weigh 2, so the free tier sustains ~2.5 klines/s.
    """

    def __init__(self, api_key: str | None = None, rps: float = 2.0, max_ban_wait: float = 300.0,
                 timeout: float = 30.0):
        self.api_key = api_key or os.environ.get("GMGN_API_KEY")
        if not self.api_key:
            raise SystemExit("GMGN_API_KEY is not set. Create a key at https://gmgn.ai/ai, then\n"
                             "  export GMGN_API_KEY=...\n(or pass --demo-key to test with GMGN's public test key)")
        self._min_interval = 1.0 / rps if rps > 0 else 0.0
        self._next = 0.0
        self.max_ban_wait = max_ban_wait
        self.requests_made = 0
        # GMGN's API only works over IPv4 (documented), so bind the socket to IPv4.
        self._http = httpx.Client(
            transport=httpx.HTTPTransport(local_address="0.0.0.0", retries=2),
            timeout=timeout,
            headers={"X-APIKEY": self.api_key, "User-Agent": "memecoin-trainer/1.0",
                     "Content-Type": "application/json"},
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _throttle(self) -> None:
        now = time.monotonic()
        if self._next > now:
            time.sleep(self._next - now)
        self._next = max(now, self._next) + self._min_interval

    def _request(self, method: str, path: str, params: dict, body: dict | None = None):
        for attempt in range(6):
            self._throttle()
            q = {k: v for k, v in params.items() if v is not None}
            q.update(timestamp=int(time.time()), client_id=str(uuid.uuid4()))
            try:
                r = self._http.request(method, HOST + path, params=q,
                                       content=json.dumps(body) if body is not None else None)
                self.requests_made += 1
            except httpx.TransportError as e:
                log.debug("%s %s transport error %s", method, path, e)
                time.sleep(2 ** attempt)
                continue
            try:
                j = r.json()
            except ValueError:
                if r.status_code >= 500:
                    time.sleep(2 ** attempt)
                    continue
                raise GMGNError(f"{path}: HTTP {r.status_code} (non-JSON response)")
            if j.get("code") == 0:
                return j.get("data")
            err = str(j.get("error") or j.get("message") or j.get("code"))
            if r.status_code == 429 or err in RATE_LIMIT_ERRORS:
                # Retrying during a ban extends it, so wait until the reset time GMGN gives.
                reset = r.headers.get("x-ratelimit-reset") or j.get("reset_at")
                wait = max(float(reset) - time.time(), 0) + 1.0 if reset else 5.0 * (attempt + 1)
                if wait > self.max_ban_wait:
                    raise GMGNError(f"{path}: rate limited for {wait:.0f}s ({err})")
                log.warning("GMGN rate limit (%s); waiting %.0fs", err, wait)
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            raise GMGNError(f"{path}: {err} {j.get('upgrade_message') or ''}".strip())
        raise GMGNError(f"{path}: gave up after retries")

    # ---- endpoints -------------------------------------------------------------
    def token_info(self, mint: str, chain: str = "sol") -> dict:
        return self._request("GET", "/v1/token/info", {"chain": chain, "address": mint}) or {}

    def top_holders(self, mint: str, limit: int = 100, chain: str = "sol") -> list[dict]:
        """Up to 100 wallets, including ones that already sold, with start_holding_at /
        end_holding_at (unix s; end is None while still holding)."""
        data = self._request("GET", "/v1/market/token_top_holders",
                             {"chain": chain, "address": mint, "limit": limit})
        return (data or {}).get("list", []) if isinstance(data, dict) else (data or [])

    def kline(self, mint: str, resolution: str = "30s", frm_ms: int | None = None, to_ms: int | None = None,
              chain: str = "sol") -> list[dict]:
        """Latest <= 100 candles in [frm_ms, to_ms]. Times are unix milliseconds.
        Candles without trades are omitted by GMGN. Fields: time, open, high, low, close
        (USD strings), volume (USD!), amount (token units!)."""
        data = self._request("GET", "/v1/market/token_kline", {
            "chain": chain, "address": mint, "resolution": resolution, "from": frm_ms, "to": to_ms})
        return (data or {}).get("list", []) if isinstance(data, dict) else (data or [])

    def kline_range(self, mint: str, frm_s: int, to_s: int, resolution: str = "30s",
                    chain: str = "sol") -> list[dict]:
        """All candles in [frm_s, to_s], paging backwards 100 candles at a time."""
        out: dict[int, dict] = {}
        to_ms, frm_ms = to_s * 1000, frm_s * 1000
        while to_ms >= frm_ms:
            page = self.kline(mint, resolution, frm_ms, to_ms, chain)
            if not page:
                break
            for c in page:
                out[int(c["time"])] = c
            oldest = min(int(c["time"]) for c in page)
            if len(page) < KLINE_PAGE or oldest <= frm_ms:
                break
            to_ms = oldest - 1
        return [out[t] for t in sorted(out)]

    def trenches(self, types: list[str], platforms: list[str] | None = None, limit: int = 80,
                 chain: str = "sol") -> dict[str, list[dict]]:
        """Live launchpad lists: new_creation (~60 newest), near_completion, completed."""
        section = {"filters": ["offchain", "onchain"], "launchpad_platform_v2": True, "limit": limit}
        if platforms:
            section["launchpad_platform"] = platforms
        body = {"version": "v2", **{t: dict(section) for t in types}}
        data = self._request("POST", "/v1/trenches", {"chain": chain}, body) or {}
        # near_completion is returned under the key "pump"
        if "pump" in data and "near_completion" not in data:
            data["near_completion"] = data.pop("pump")
        return {k: v for k, v in data.items() if isinstance(v, list)}
