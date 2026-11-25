"""Feature and target engineering for one token's candle series.

Every feature at row t only uses data up to the close of candle t (no look-ahead).
Every target looks strictly forward (candles t+1 ... t+h).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# name -> (description, unit). This dict is the single source of truth for DATASET.md.
FEATURES: dict[str, tuple[str, str]] = {
    # --- raw candle values ---
    "open": ("First trade price in the candle", "SOL/token"),
    "high": ("Highest trade price in the candle", "SOL/token"),
    "low": ("Lowest trade price in the candle", "SOL/token"),
    "close": ("Last trade price in the candle (previous close if no trades)", "SOL/token"),
    "volume_sol": ("Total traded value in the candle", "SOL"),
    "volume_token": ("Total traded token amount in the candle", "tokens"),
    "buy_volume_sol": ("SOL spent buying the token", "SOL"),
    "sell_volume_sol": ("SOL received selling the token", "SOL"),
    "trades": ("Number of swaps", "count"),
    "buys": ("Number of buy swaps", "count"),
    "sells": ("Number of sell swaps", "count"),
    "unique_traders": ("Distinct wallets that traded in the candle", "count"),
    "sol_reserve": ("SOL liquidity in the pool at candle close", "SOL"),
    "token_reserve": ("Token liquidity in the pool at candle close", "tokens"),
    "market_cap_sol": ("close x total supply", "SOL"),
    "holders": ("Wallets holding > 0 tokens at candle close (rebuilt from on-chain balances)", "count"),
    # --- lifecycle ---
    "age_s": ("Seconds since the token's first downloaded trade", "s"),
    "candles_since_trade": ("Consecutive candles with no trades up to now (0 = traded this candle)", "count"),
    # --- returns / momentum ---
    "ret_1": ("Log return of close over 1 candle", "log"),
    "ret_2": ("Log return of close over 2 candles", "log"),
    "ret_5": ("Log return of close over 5 candles", "log"),
    "ret_10": ("Log return of close over 10 candles", "log"),
    "ret_20": ("Log return of close over 20 candles", "log"),
    "ema_ratio_10": ("close / EMA(close, 10) - 1", "ratio"),
    "ema_ratio_30": ("close / EMA(close, 30) - 1", "ratio"),
    "vwap_ratio_20": ("close / VWAP over the last 20 candles - 1", "ratio"),
    "rsi_14": ("Relative Strength Index over 14 candles", "0-100"),
    "drawdown_from_ath": ("close / all-time-high so far - 1", "ratio"),
    # --- candle shape / volatility ---
    "body_pct": ("close / open - 1", "ratio"),
    "range_pct": ("high / low - 1", "ratio"),
    "upper_wick_pct": ("(high - max(open, close)) / close", "ratio"),
    "lower_wick_pct": ("(min(open, close) - low) / close", "ratio"),
    "volatility_10": ("Std of 1-candle log returns over 10 candles", "log"),
    "volatility_20": ("Std of 1-candle log returns over 20 candles", "log"),
    # --- volume / order flow ---
    "volume_sol_sum_10": ("SOL volume over the last 10 candles", "SOL"),
    "volume_ratio_10": ("volume_sol / mean volume_sol of the last 10 candles", "ratio"),
    "trades_ratio_10": ("trades / mean trades of the last 10 candles", "ratio"),
    "avg_trade_sol": ("volume_sol / trades (0 if no trades)", "SOL"),
    "buy_ratio": ("buy_volume_sol / volume_sol (0.5 if no volume)", "0-1"),
    "buy_count_ratio": ("buys / trades (0.5 if no trades)", "0-1"),
    "net_flow_sol": ("buy_volume_sol - sell_volume_sol", "SOL"),
    "net_flow_sol_10": ("net_flow_sol summed over the last 10 candles", "SOL"),
    # --- holders / liquidity dynamics ---
    "holders_change_1": ("holders - holders 1 candle ago", "count"),
    "holders_change_10": ("holders - holders 10 candles ago", "count"),
    "holders_growth_10": ("holders / holders 10 candles ago - 1", "ratio"),
    "sol_reserve_change_10": ("sol_reserve / sol_reserve 10 candles ago - 1", "ratio"),
    "log_market_cap": ("log(1 + market_cap_sol)", "log SOL"),
}

ID_COLUMNS: dict[str, str] = {
    "mint": "Token mint address",
    "symbol": "Token symbol (from on-chain metadata)",
    "timestamp": "Candle open time, UTC",
    "split": "train / val / test, assigned per token (all rows of a token share a split)",
}


def target_docs(horizons: dict[str, int], up_threshold: float) -> dict[str, str]:
    docs = {}
    for name, h in horizons.items():
        docs[f"target_ret_{name}"] = f"Log return from close[t] to close[t+{h}] ({name} ahead)"
        docs[f"target_max_ret_{name}"] = f"Best case: max(high[t+1..t+{h}]) / close[t] - 1"
        docs[f"target_min_ret_{name}"] = f"Worst case: min(low[t+1..t+{h}]) / close[t] - 1"
        docs[f"target_up_{name}"] = (f"1 if target_ret_{name} > log(1 + {up_threshold:g}) "
                                     f"(price up more than {up_threshold:.0%}), else 0")
    return docs


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).where(loss > 0, 100.0).where(gain.notna())


def make_features(c: pd.DataFrame) -> pd.DataFrame:
    """c: candles of ONE token from build_ohlcv (gap-filled, sorted by time)."""
    f = c.copy()
    close, high, low, open_ = f["close"], f["high"], f["low"], f["open"]
    lr = np.log(close).diff()

    f["age_s"] = (f["timestamp"] - f["timestamp"].iloc[0]).dt.total_seconds()
    idle = (f["trades"] == 0).astype(int)
    f["candles_since_trade"] = idle.groupby((idle == 0).cumsum()).cumsum()

    for k in (1, 2, 5, 10, 20):
        f[f"ret_{k}"] = np.log(close / close.shift(k))
    f["ema_ratio_10"] = close / close.ewm(span=10, adjust=False).mean() - 1
    f["ema_ratio_30"] = close / close.ewm(span=30, adjust=False).mean() - 1
    vwap = _safe_div(f["volume_sol"].rolling(20, min_periods=1).sum(),
                     f["volume_token"].rolling(20, min_periods=1).sum())
    f["vwap_ratio_20"] = close / vwap - 1
    f["rsi_14"] = _rsi(close)
    f["drawdown_from_ath"] = close / high.cummax() - 1

    f["body_pct"] = close / open_ - 1
    f["range_pct"] = high / low - 1
    f["upper_wick_pct"] = (high - np.maximum(open_, close)) / close
    f["lower_wick_pct"] = (np.minimum(open_, close) - low) / close
    f["volatility_10"] = lr.rolling(10).std()
    f["volatility_20"] = lr.rolling(20).std()

    f["volume_sol_sum_10"] = f["volume_sol"].rolling(10, min_periods=1).sum()
    f["volume_ratio_10"] = _safe_div(f["volume_sol"], f["volume_sol"].rolling(10, min_periods=1).mean())
    f["trades_ratio_10"] = _safe_div(f["trades"], f["trades"].rolling(10, min_periods=1).mean())
    f["avg_trade_sol"] = _safe_div(f["volume_sol"], f["trades"]).fillna(0.0)
    f["buy_ratio"] = _safe_div(f["buy_volume_sol"], f["volume_sol"]).fillna(0.5)
    f["buy_count_ratio"] = _safe_div(f["buys"], f["trades"]).fillna(0.5)
    f["net_flow_sol"] = f["buy_volume_sol"] - f["sell_volume_sol"]
    f["net_flow_sol_10"] = f["net_flow_sol"].rolling(10, min_periods=1).sum()

    h = f["holders"]
    f["holders_change_1"] = h.diff(1)
    f["holders_change_10"] = h.diff(10)
    f["holders_growth_10"] = _safe_div(h, h.shift(10)) - 1
    f["sol_reserve_change_10"] = _safe_div(f["sol_reserve"], f["sol_reserve"].shift(10)) - 1
    f["log_market_cap"] = np.log1p(f["market_cap_sol"])
    return f.replace([np.inf, -np.inf], np.nan)


def make_targets(f: pd.DataFrame, horizons: dict[str, int], up_threshold: float) -> pd.DataFrame:
    close = f["close"]
    for name, h in horizons.items():
        fut_close = close.shift(-h)
        # max/min over the NEXT h candles (t+1 .. t+h)
        fut_high = f["high"][::-1].rolling(h, min_periods=h).max()[::-1].shift(-1)
        fut_low = f["low"][::-1].rolling(h, min_periods=h).min()[::-1].shift(-1)
        ret = np.log(fut_close / close)
        f[f"target_ret_{name}"] = ret
        f[f"target_max_ret_{name}"] = fut_high / close - 1
        f[f"target_min_ret_{name}"] = fut_low / close - 1
        f[f"target_up_{name}"] = (ret > np.log1p(up_threshold)).astype("Int8").where(ret.notna())
    return f
