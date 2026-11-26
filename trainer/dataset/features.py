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
    "token_reserve": ("Tokens in the most recently traded pool at candle close", "tokens"),
    "market_cap_sol": ("close x total supply", "SOL"),
    "holders": ("Wallets holding > 0 tokens at candle close (rebuilt from on-chain balances)", "count"),
    # --- liquidity ---
    "liquidity_sol": ("Total SOL in all of the coin's pools at candle close (bonding curve + AMMs)", "SOL"),
    "liquidity_change_1": ("liquidity_sol / liquidity_sol 1 candle ago - 1", "ratio"),
    "liquidity_change_10": ("liquidity_sol / liquidity_sol 10 candles ago - 1 (a sharp drop = liquidity pulled)", "ratio"),
    "liquidity_to_mcap": ("liquidity_sol / market_cap_sol (how thin the market is)", "ratio"),
    # --- activity ---
    "trades_per_min": ("Swaps in the last 60 seconds", "trades/min"),
    "trades_per_min_5m": ("Average swaps per minute over the last 5 minutes", "trades/min"),
    # --- concentration ---
    "top10_pct": ("% of total supply held by the 10 largest wallets (pools and program accounts excluded)", "%"),
    "top10_change_10": ("top10_pct - top10_pct 10 candles ago (negative = big holders selling)", "% points"),
    # --- pump.fun bonding curve ---
    "curve_progress_pct": ("pump.fun bonding curve progress at candle close (100 = curve complete / graduated). "
                           "On-chain: from the curve vault's tokens. GMGN: from the price. Unknown for Mayhem-mode "
                           "coins (pump.fun's agent moves their curve)", "%"),
    "graduated": ("1 once the bonding curve is complete or the coin trades on an AMM (PumpSwap, Raydium, ...), else 0", "0/1"),
    "curve_progress_change_10": ("curve_progress_pct - curve_progress_pct 10 candles ago", "% points"),
    # --- launch bundle ---
    "bundle_launch_pct": ("% of supply bought in the launch block (same slot as the create tx, dev's buy included)", "%"),
    "bundle_wallets": ("Wallets that bought in the launch block (1 = only the dev)", "count"),
    "bundle_pct": ("% of supply the launch-block wallets still hold at candle close", "%"),
    "bundle_change_10": ("bundle_pct - bundle_pct 10 candles ago (negative = the bundle is selling)", "% points"),
    # --- whales ---
    "max_buy_sol": ("Biggest buy by ONE wallet in the candle (its swaps summed), 0 if no buys", "SOL"),
    "max_sell_sol": ("Biggest sell by ONE wallet in the candle (its swaps summed), 0 if no sells", "SOL"),
    "max_buy_to_liquidity": ("max_buy_sol / liquidity_sol (how much one wallet moved the pool)", "ratio"),
    # --- order flow by wallet ---
    "unique_buyers": ("Distinct wallets that bought in the candle", "count"),
    "unique_sellers": ("Distinct wallets that sold in the candle", "count"),
    "new_buyers": ("Wallets buying this coin for the first time in the candle (new money)", "count"),
    "full_exits": ("Wallets that sold (almost) their whole position in the candle: under 1% of their peak left", "count"),
    "max_sell_to_liquidity": ("max_sell_sol / liquidity_sol", "ratio"),
    # --- lifecycle ---
    "age_s": ("Seconds since the coin was launched (or since its first downloaded trade if the launch time is unknown)", "s"),
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
    "log_market_cap": ("log(1 + market_cap_sol)", "log SOL"),
}

ID_COLUMNS: dict[str, str] = {
    "mint": "Token mint address",
    "symbol": "Token symbol (from on-chain metadata)",
    "timestamp": "Candle open time, UTC",
    "segment": "Contiguous stretch of downloaded data within the coin (0, 1, ...). Windows never cross segments",
    "split": "train / val / test, assigned per token (all rows of a token share a split)",
    "source": "Where the candles came from: gmgn, onchain, gmgn+onchain (GMGN candles + on-chain holders/liquidity/trades), "
              "or archive (a downloaded pump.fun trade archive, import_archive.py)",
}


OUTCOME_COLUMNS: dict[str, str] = {
    "outcome_dead": "1 if the coin is dead at download time (no trade for 1h, or price 90%+ below its all-time high)",
    "outcome_graduated": "1 if the coin left the pump.fun bonding curve and traded on an AMM (e.g. PumpSwap)",
    "outcome_ath_mcap_sol": "Highest market cap the coin ever reached",
    "outcome_final_drawdown": "Last price / all-time-high price - 1 (-0.99 = lost 99% from its peak)",
    "outcome_lifetime_s": "Seconds between the first and last trade",
}


def target_docs(horizons: dict[str, int], up_threshold: float, dump_threshold: float = 0.3) -> dict[str, str]:
    docs = {}
    for name, h in horizons.items():
        docs[f"target_ret_{name}"] = f"Log return from close[t] to close[t+{h}] ({name} ahead)"
        docs[f"target_max_ret_{name}"] = f"Best case: max(high[t+1..t+{h}]) / close[t] - 1"
        docs[f"target_min_ret_{name}"] = f"Worst case: min(low[t+1..t+{h}]) / close[t] - 1"
        docs[f"target_up_{name}"] = (f"1 if target_ret_{name} > log(1 + {up_threshold:g}) "
                                     f"(price up more than {up_threshold:.0%}), else 0")
        docs[f"target_dump_{name}"] = (f"1 if the price drops more than {dump_threshold:.0%} below close[t] at any "
                                       f"point in the next {name} (low[t+1..t+{h}]), else 0")
    return docs


def _safe_div(a, b):
    return a / b.replace(0, np.nan)


def _rolling(x: pd.Series, n: int, how: str) -> pd.Series:
    """Rolling sum/mean that stays NaN if any value in the window is unknown (NaN),
    instead of silently treating unknown as zero."""
    r = getattr(x.rolling(n, min_periods=1), how)()
    unknown = x.isna().astype(int).rolling(n, min_periods=1).sum() > 0
    return r.mask(unknown)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).where(loss > 0, 100.0).where(gain.notna())


def make_features(c: pd.DataFrame, t0: pd.Timestamp | None = None) -> pd.DataFrame:
    """c: ONE contiguous segment of one coin's candles (gap-filled, sorted by time).
    t0: the coin's launch time (for age_s); defaults to the first candle."""
    f = c.copy()
    close, high, low, open_ = f["close"], f["high"], f["low"], f["open"]
    lr = np.log(close).diff()

    f["age_s"] = (f["timestamp"] - (t0 if t0 is not None else f["timestamp"].iloc[0])).dt.total_seconds()
    # a candle traded if it has volume (GMGN gives volume but no trade count)
    idle = (f["volume_sol"] <= 0).astype(int)
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
    no_vol = f["volume_sol"] <= 0
    f["trades_ratio_10"] = _safe_div(f["trades"], _rolling(f["trades"], 10, "mean"))
    f["avg_trade_sol"] = _safe_div(f["volume_sol"], f["trades"]).mask(no_vol, 0.0)
    # neutral 0.5 only when there was no trading; NaN when the split is unknown
    f["buy_ratio"] = _safe_div(f["buy_volume_sol"], f["volume_sol"]).mask(no_vol, 0.5)
    f["buy_count_ratio"] = _safe_div(f["buys"], f["trades"]).mask(f["trades"] == 0, 0.5)
    f["net_flow_sol"] = f["buy_volume_sol"] - f["sell_volume_sol"]
    f["net_flow_sol_10"] = _rolling(f["net_flow_sol"], 10, "sum")

    h = f["holders"]
    f["holders_change_1"] = h.diff(1)
    f["holders_change_10"] = h.diff(10)
    f["holders_growth_10"] = _safe_div(h, h.shift(10)) - 1
    liq = f["liquidity_sol"]
    f["liquidity_change_1"] = _safe_div(liq, liq.shift(1)) - 1
    f["liquidity_change_10"] = _safe_div(liq, liq.shift(10)) - 1
    f["liquidity_to_mcap"] = _safe_div(liq, f["market_cap_sol"])

    step_s = (f["timestamp"].iloc[1] - f["timestamp"].iloc[0]).total_seconds() if len(f) > 1 else 60.0
    n1 = max(1, round(60 / step_s))
    n5 = max(1, round(300 / step_s))
    f["trades_per_min"] = _rolling(f["trades"], n1, "sum") * 60 / (n1 * step_s)
    f["trades_per_min_5m"] = _rolling(f["trades"], n5, "sum") * 60 / (n5 * step_s)

    f["top10_change_10"] = f["top10_pct"].diff(10)
    f["curve_progress_change_10"] = f["curve_progress_pct"].diff(10)
    f["bundle_change_10"] = f["bundle_pct"].diff(10)
    f["max_buy_to_liquidity"] = _safe_div(f["max_buy_sol"], liq)
    f["max_sell_to_liquidity"] = _safe_div(f["max_sell_sol"], liq)
    f["log_market_cap"] = np.log1p(f["market_cap_sol"])
    return f.replace([np.inf, -np.inf], np.nan)


def make_targets(f: pd.DataFrame, horizons: dict[str, int], up_threshold: float,
                 dump_threshold: float = 0.3) -> pd.DataFrame:
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
        worst = f[f"target_min_ret_{name}"]
        f[f"target_dump_{name}"] = (worst < -dump_threshold).astype("Int8").where(worst.notna())
    return f
