"""Aggregate swap trades into OHLCV candles."""
from __future__ import annotations

import pandas as pd

from .holders import holder_stats

CANDLE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume_sol", "volume_token",
    "buy_volume_sol", "sell_volume_sol", "trades", "buys", "sells", "unique_traders",
    "sol_reserve", "token_reserve", "liquidity_sol", "market_cap_sol", "holders", "top10_pct",
]


def _liquidity(df: pd.DataFrame, interval: str) -> pd.Series:
    """Total SOL in all of the coin's pools at candle close.

    Each pool's SOL reserve is known after every trade on it. The latest value per pool is
    carried forward and the pools are summed. When a pump.fun coin graduates, its bonding
    curve is emptied into the AMM pool by a migration tx (not a trade), so the curve is
    counted as 0 from the first trade on any other pool.
    """
    per_pool = (df.pivot_table(index=df.index, columns="pool_vault", values="quote_reserve", aggfunc="last")
                .resample(interval, label="left", closed="left").last().ffill())
    dex = df.groupby("pool_vault")["dex"].first()
    curve = [p for p in per_pool.columns if dex[p] == "pumpfun"]
    amm = [p for p in per_pool.columns if dex[p] != "pumpfun"]
    if curve and amm:
        graduated = per_pool[amm].notna().any(axis=1).cummax()
        per_pool.loc[graduated, curve] = 0.0
    return per_pool.sum(axis=1, min_count=1)


def build_ohlcv(trades: pd.DataFrame, interval: str = "30s", supply: float | None = None,
                fill_gaps: bool = True, min_quote: float = 1e-5,
                balances: pd.DataFrame | None = None,
                end_time: pd.Timestamp | None = None) -> pd.DataFrame:
    """Build SOL-denominated candles.

    interval: any pandas offset alias ("1s", "15s", "1min", "5min", "1h", "1D").
    fill_gaps: emit empty candles (volume 0, OHLC = previous close) so the series
               is regularly spaced, which most ML models expect.
    min_quote: drop dust trades below this many SOL (their implied price is noisy).
    balances: per-account balance updates (data/balances/<mint>.csv); adds `holders`
              (wallets holding > 0) and `top10_pct` (% of supply held by the top 10 wallets).
    end_time: extend the series with empty candles up to this time (used for coins that
              stopped trading: "nothing happened" after the last trade is real information).
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
    out["liquidity_sol"] = _liquidity(df, interval)

    if end_time is not None and fill_gaps:
        end = pd.Timestamp(end_time).floor(interval)
        if end > out.index[-1]:
            out = out.reindex(pd.date_range(out.index[0], end, freq=interval, tz="UTC"))
            for col in ("volume_sol", "volume_token", "buy_volume_sol", "sell_volume_sol",
                        "trades", "buys", "sells", "unique_traders"):
                out[col] = out[col].fillna(0)
            out[["trades", "buys", "sells", "unique_traders"]] = out[["trades", "buys", "sells", "unique_traders"]].astype(int)
            out.index.name = "ts"

    hs = holder_stats(balances, interval, supply)
    for col in ("holders", "top10_pct"):
        if hs.empty:
            out[col] = float("nan")
        else:
            out[col] = hs[col].reindex(hs.index.union(out.index)).ffill().reindex(out.index)

    if fill_gaps:
        out["close"] = out["close"].ffill()
        for col in ("open", "high", "low"):
            out[col] = out[col].fillna(out["close"])
        out[["sol_reserve", "token_reserve", "liquidity_sol"]] = (
            out[["sol_reserve", "token_reserve", "liquidity_sol"]].ffill())
    else:
        out = out[out["trades"] > 0]
    out["market_cap_sol"] = out["close"] * supply if supply else float("nan")
    out = out.reset_index().rename(columns={"ts": "timestamp"})
    return out[CANDLE_COLUMNS]
