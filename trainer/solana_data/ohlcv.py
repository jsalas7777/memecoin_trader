"""Aggregate swap trades into OHLCV candles."""
from __future__ import annotations

import pandas as pd

from .holders import holder_series

CANDLE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume_sol", "volume_token",
    "buy_volume_sol", "sell_volume_sol", "trades", "buys", "sells", "unique_traders",
    "sol_reserve", "token_reserve", "market_cap_sol", "holders",
]


def build_ohlcv(trades: pd.DataFrame, interval: str = "30s", supply: float | None = None,
                fill_gaps: bool = True, min_quote: float = 1e-5,
                balances: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build SOL-denominated candles.

    interval: any pandas offset alias ("1s", "15s", "1min", "5min", "1h", "1D").
    fill_gaps: emit empty candles (volume 0, OHLC = previous close) so the series
               is regularly spaced, which most ML models expect.
    min_quote: drop dust trades below this many SOL (their implied price is noisy).
    balances: per-account balance updates (data/balances/<mint>.csv); adds a `holders`
              column = wallets holding > 0 at candle close.
    """
    df = trades[(trades["quote"] == "SOL") & (trades["quote_amount"] >= min_quote)].copy()
    df = df.dropna(subset=["block_time"])
    if df.empty:
        return pd.DataFrame(columns=CANDLE_COLUMNS)
    df["ts"] = pd.to_datetime(df["block_time"], unit="s", utc=True)
    df = df.sort_values(["slot", "tx_index"], na_position="last").set_index("ts")
    is_buy = df["side"] == "buy"
    df["buy_sol"] = df["quote_amount"].where(is_buy, 0.0)
    df["sell_sol"] = df["quote_amount"].where(~is_buy, 0.0)
    df["is_buy"] = is_buy.astype(int)

    r = df.resample(interval, label="left", closed="left")
    out = r["price"].ohlc()
    out["volume_sol"] = r["quote_amount"].sum()
    out["volume_token"] = r["token_amount"].sum()
    out["buy_volume_sol"] = r["buy_sol"].sum()
    out["sell_volume_sol"] = r["sell_sol"].sum()
    out["trades"] = r["price"].count()
    out["buys"] = r["is_buy"].sum()
    out["sells"] = out["trades"] - out["buys"]
    out["unique_traders"] = r["trader"].nunique()
    # reserves of the most active pool at candle close (liquidity proxy)
    out["sol_reserve"] = r["quote_reserve"].last()
    out["token_reserve"] = r["token_reserve"].last()

    hs = holder_series(balances)
    if not hs.empty:
        hc = hs.resample(interval, label="left", closed="left").last()
        out["holders"] = hc.reindex(hc.index.union(out.index)).ffill().reindex(out.index)
    else:
        out["holders"] = float("nan")

    if fill_gaps:
        out["close"] = out["close"].ffill()
        for col in ("open", "high", "low"):
            out[col] = out[col].fillna(out["close"])
        out[["sol_reserve", "token_reserve"]] = out[["sol_reserve", "token_reserve"]].ffill()
    else:
        out = out[out["trades"] > 0]
    out["market_cap_sol"] = out["close"] * supply if supply else float("nan")
    out = out.reset_index().rename(columns={"ts": "timestamp"})
    return out[CANDLE_COLUMNS]
