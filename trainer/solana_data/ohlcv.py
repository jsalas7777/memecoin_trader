"""Aggregate swap trades into OHLCV candles."""
from __future__ import annotations

import pandas as pd

from .constants import PUMP_CURVE_RESERVED_TOKENS, PUMP_CURVE_SELLABLE_TOKENS, PUMP_CURVE_START_TOKENS
from .holders import bundle_stats, holder_stats

CANDLE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume_sol", "volume_token",
    "buy_volume_sol", "sell_volume_sol", "trades", "buys", "sells", "unique_traders",
    "sol_reserve", "token_reserve", "liquidity_sol", "market_cap_sol", "holders", "top10_pct",
    "max_buy_sol", "max_sell_sol", "curve_progress_pct", "graduated",
    "bundle_pct", "bundle_launch_pct", "bundle_wallets",
]


def curve_progress_from_vault(vault_tokens):
    """pump.fun bonding curve progress in % from the tokens left in the curve's vault."""
    return ((PUMP_CURVE_START_TOKENS - vault_tokens) / PUMP_CURVE_SELLABLE_TOKENS * 100).clip(0, 100)


def _curve(df: pd.DataFrame, index: pd.DatetimeIndex, interval: str) -> pd.DataFrame:
    """Per-candle `curve_progress_pct` and `graduated` for a pump.fun coin.

    graduated = 1 from the first trade on any AMM pool, or once the curve vault is down to
    the 206.9M reserved tokens (curve complete). Before that, progress comes from the
    curve vault's balance after its latest trade. Unknown (NaN) before the first trade seen.
    """
    r = lambda s, how: getattr(s.resample(interval, label="left", closed="left"), how)().reindex(index)
    curve = df[df["dex"] == "pumpfun"]
    vault = r(curve["token_reserve"], "last").ffill() if len(curve) else pd.Series(float("nan"), index=index)
    amm = r((df["dex"] != "pumpfun").astype(float), "max").fillna(0).cummax() > 0
    done = (vault <= PUMP_CURVE_RESERVED_TOKENS + 1).fillna(False).cummax()
    grad = (amm | done).astype(float).where(amm | done | vault.notna())
    progress = curve_progress_from_vault(vault).where(grad == 0, 100.0).where(grad.notna())
    return pd.DataFrame({"curve_progress_pct": progress, "graduated": grad}, index=index)


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
                end_time: pd.Timestamp | None = None,
                pump_curve: bool = False, launch_slot: int | None = None) -> pd.DataFrame:
    """Build SOL-denominated candles.

    interval: any pandas offset alias ("1s", "15s", "1min", "5min", "1h", "1D").
    fill_gaps: emit empty candles (volume 0, OHLC = previous close) so the series
               is regularly spaced, which most ML models expect.
    min_quote: drop dust trades below this many SOL (their implied price is noisy).
    balances: per-account balance updates (data/balances/<mint>.csv); adds `holders`
              (wallets holding > 0) and `top10_pct` (% of supply held by the top 10 wallets).
    end_time: extend the series with empty candles up to this time (used for coins that
              stopped trading: "nothing happened" after the last trade is real information).
    pump_curve: the coin launched on the pump.fun bonding curve (adds curve progress / graduated).
    launch_slot: slot of the create tx, if the history is complete back to creation
                 (adds the launch-bundle columns).
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
    # whales: the biggest amount ONE wallet bought / sold in the candle (its swaps summed,
    # so splitting a buy into several small swaps doesn't hide it)
    per_wallet = df.groupby([pd.Grouper(freq=interval, label="left", closed="left"), "trader"])
    out["max_buy_sol"] = per_wallet["buy_sol"].sum().groupby(level=0).max().reindex(out.index).fillna(0.0)
    out["max_sell_sol"] = per_wallet["sell_sol"].sum().groupby(level=0).max().reindex(out.index).fillna(0.0)

    if end_time is not None and fill_gaps:
        end = pd.Timestamp(end_time).floor(interval)
        if end > out.index[-1]:
            out = out.reindex(pd.date_range(out.index[0], end, freq=interval, tz="UTC"))
            for col in ("volume_sol", "volume_token", "buy_volume_sol", "sell_volume_sol",
                        "trades", "buys", "sells", "unique_traders", "max_buy_sol", "max_sell_sol"):
                out[col] = out[col].fillna(0)
            out[["trades", "buys", "sells", "unique_traders"]] = out[["trades", "buys", "sells", "unique_traders"]].astype(int)
            out.index.name = "ts"

    hs = holder_stats(balances, interval, supply)
    for col in ("holders", "top10_pct"):
        if hs.empty:
            out[col] = float("nan")
        else:
            out[col] = hs[col].reindex(hs.index.union(out.index)).ffill().reindex(out.index)

    if pump_curve:
        out[["curve_progress_pct", "graduated"]] = _curve(df, out.index, interval)
    else:
        out["curve_progress_pct"] = out["graduated"] = float("nan")
    bs, binfo = bundle_stats(balances, interval, supply, launch_slot)
    out["bundle_launch_pct"] = binfo["bundle_launch_pct"]
    out["bundle_wallets"] = binfo["bundle_wallets"]
    if bs.empty:
        out["bundle_pct"] = 0.0 if binfo["bundle_wallets"] == 0 else float("nan")
    else:
        out["bundle_pct"] = bs["bundle_pct"].reindex(bs.index.union(out.index)).ffill().reindex(out.index)

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
