"""Convert 30s candles into discrete tokens for a sequence model.

Each candle produces one token per input stream:
  candle    : price move vs the previous close x candle range (shape of the bar)
  volume    : SOL volume on a log scale
  holders   : number of holders on a log scale (5% steps)
  top10     : % of supply held by the 10 largest wallets (2.5% steps)
  trades    : trades per minute (exact up to 10, then 25% steps)
  liquidity : total SOL in the coin's pools on a log scale (~26% steps)
  curve     : pump.fun bonding curve progress (2.5% steps), or GRADUATED
  bundle    : % of supply the launch-block (bundle) wallets still hold
  whale_buy : biggest buy by one wallet in the candle (SOL, log scale)
  whale_sell: biggest sell by one wallet in the candle (SOL, log scale)

Id 0 is always PAD (left padding for tokens with < 120 candles of history).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SEQ_LEN = 120  # 120 x 30s = 60 minutes of history

PAD = 0

# ---- candle stream -------------------------------------------------------------
CANDLE_NO_TRADE = 1
# absolute % move of close vs previous close; symmetric, wider bins in the fat tails
RET_EDGES_PCT = [0.1, 0.5, 1, 2, 3, 5, 8, 12, 20, 35]
# candle range = high / low - 1, in %
RANGE_EDGES_PCT = [1, 3, 8]
N_RET = 2 * len(RET_EDGES_PCT) + 1   # 21
N_RANGE = len(RANGE_EDGES_PCT) + 1   # 4
CANDLE_OFFSET = 2
CANDLE_VOCAB = CANDLE_OFFSET + N_RET * N_RANGE  # 86

# ---- volume stream -------------------------------------------------------------
VOLUME_ZERO = 1
VOL_LOG10_MIN, VOL_LOG10_MAX, VOL_STEP = -3.0, 3.0, 0.25   # 0.001 SOL .. 1000 SOL
VOL_EDGES = np.arange(VOL_LOG10_MIN, VOL_LOG10_MAX + 1e-9, VOL_STEP)  # 25 edges -> 26 bins
VOLUME_OFFSET = 2
VOLUME_VOCAB = VOLUME_OFFSET + len(VOL_EDGES) + 1  # 28

# ---- holders stream ------------------------------------------------------------
HOLDERS_UNKNOWN = 1
HOLDERS_ZERO = 2
HOLDERS_EXACT = 20    # 1..20 holders each get their own token
HOLDERS_STEP = 1.05   # above 20, each bin covers 5% more holders than the previous
HOLDERS_LOG_BINS = 240  # 20 * 1.05^240 ~ 2.4M holders
HOLDERS_OFFSET = 3
HOLDERS_VOCAB = HOLDERS_OFFSET + HOLDERS_EXACT + HOLDERS_LOG_BINS  # 263


# ---- top-10 concentration stream ------------------------------------------------
TOP10_UNKNOWN = 1
TOP10_STEP_PCT = 2.5
TOP10_BINS = int(100 / TOP10_STEP_PCT)  # 40
TOP10_OFFSET = 2
TOP10_VOCAB = TOP10_OFFSET + TOP10_BINS  # 42

# ---- trades-per-minute stream --------------------------------------------------
TRADES_ZERO = 1
TRADES_UNKNOWN = 2       # trade count not available (e.g. GMGN-only candles)
TRADES_EXACT = 10        # 1..10 trades/min each get their own token
TRADES_STEP = 1.25       # above 10, each bin is 25% more trades/min
TRADES_LOG_BINS = 36     # 10 * 1.25^36 ~ 30k trades/min
TRADES_OFFSET = 3
TRADES_VOCAB = TRADES_OFFSET + TRADES_EXACT + TRADES_LOG_BINS  # 49

# ---- liquidity stream ----------------------------------------------------------
LIQ_UNKNOWN = 1
LIQ_LOG10_MIN, LIQ_LOG10_MAX, LIQ_STEP = -2.0, 5.0, 0.1   # 0.01 SOL .. 100k SOL
LIQ_EDGES = np.round(np.arange(LIQ_LOG10_MIN, LIQ_LOG10_MAX + 1e-9, LIQ_STEP), 6)  # 71 edges -> 72 bins
LIQ_OFFSET = 2
LIQ_VOCAB = LIQ_OFFSET + len(LIQ_EDGES) + 1  # 74


# ---- bonding curve stream -----------------------------------------------------
CURVE_UNKNOWN = 1        # not a pump.fun coin, or no data
CURVE_GRADUATED = 2      # left the curve for an AMM
CURVE_STEP_PCT = 2.5
CURVE_BINS = int(100 / CURVE_STEP_PCT)  # 40
CURVE_OFFSET = 3
CURVE_VOCAB = CURVE_OFFSET + CURVE_BINS  # 43

# ---- launch bundle stream ------------------------------------------------------
BUNDLE_UNKNOWN = 1       # history doesn't reach back to the launch block
BUNDLE_NONE = 2          # nobody bought in the launch block
BUNDLE_SOLD_OUT = 3      # the launch-block wallets bought, and have sold (almost) everything
BUNDLE_SOLD_OUT_PCT = 0.05  # below this % of supply counts as sold out
# % of supply still held: 1% steps up to 20%, then 5% steps
BUNDLE_EDGES = [float(x) for x in range(1, 21)] + [float(x) for x in range(25, 100, 5)]  # 35 edges -> 36 bins
BUNDLE_OFFSET = 4
BUNDLE_VOCAB = BUNDLE_OFFSET + len(BUNDLE_EDGES) + 1  # 40

# ---- whale buy / sell streams --------------------------------------------------
WHALE_NONE = 1           # no buys (sells) in the candle
WHALE_UNKNOWN = 2        # per-wallet amounts not available (GMGN-only candles)
WHALE_OFFSET = 3
WHALE_EDGES = VOL_EDGES  # same log10 scale as volume: 0.001 .. 1000 SOL, 0.25-decade steps
WHALE_VOCAB = WHALE_OFFSET + len(WHALE_EDGES) + 1  # 29


def _ret_bin(ret_pct: np.ndarray) -> np.ndarray:
    """Signed % move -> 0..20 (10 = flat)."""
    mag = np.searchsorted(RET_EDGES_PCT, np.abs(ret_pct), side="right")  # 0..10
    return np.where(ret_pct >= 0, len(RET_EDGES_PCT) + mag, len(RET_EDGES_PCT) - mag)


def candle_tokens(c: pd.DataFrame) -> np.ndarray:
    close = c["close"].to_numpy(float)
    prev = np.concatenate([[c["open"].iloc[0]], close[:-1]])
    ret_pct = (close / prev - 1) * 100
    range_pct = (c["high"].to_numpy(float) / c["low"].to_numpy(float) - 1) * 100
    rb = _ret_bin(np.nan_to_num(ret_pct))
    gb = np.searchsorted(RANGE_EDGES_PCT, np.nan_to_num(range_pct), side="right")
    tok = CANDLE_OFFSET + rb * N_RANGE + gb
    traded = (np.nan_to_num(c["trades"].to_numpy(float)) > 0) | (c["volume_sol"].to_numpy(float) > 0)
    return np.where(traded, tok, CANDLE_NO_TRADE).astype(np.int16)


def volume_tokens(c: pd.DataFrame) -> np.ndarray:
    v = c["volume_sol"].to_numpy(float)
    with np.errstate(divide="ignore"):
        lv = np.log10(np.where(v > 0, v, np.nan))
    b = np.searchsorted(VOL_EDGES, np.nan_to_num(lv, nan=-99), side="right")
    return np.where(v > 0, VOLUME_OFFSET + b, VOLUME_ZERO).astype(np.int16)


def _holders_bin(h: np.ndarray) -> np.ndarray:
    """holders >= 1 -> 0..HOLDERS_EXACT+HOLDERS_LOG_BINS-1."""
    with np.errstate(divide="ignore", invalid="ignore"):
        geo = HOLDERS_EXACT + np.floor(np.log(np.maximum(h, 1) / (HOLDERS_EXACT + 1)) / math.log(HOLDERS_STEP))
    b = np.where(h <= HOLDERS_EXACT, h - 1, geo)
    return np.clip(np.nan_to_num(b), 0, HOLDERS_EXACT + HOLDERS_LOG_BINS - 1).astype(int)


def holder_tokens(c: pd.DataFrame) -> np.ndarray:
    h = np.round(c["holders"].to_numpy(float))
    tok = HOLDERS_OFFSET + _holders_bin(h)
    tok = np.where(h == 0, HOLDERS_ZERO, tok)
    return np.where(np.isnan(h), HOLDERS_UNKNOWN, tok).astype(np.int16)


def top10_tokens(c: pd.DataFrame) -> np.ndarray:
    pct = c["top10_pct"].to_numpy(float)
    b = np.clip(np.floor(np.nan_to_num(pct) / TOP10_STEP_PCT), 0, TOP10_BINS - 1).astype(int)
    return np.where(np.isnan(pct), TOP10_UNKNOWN, TOP10_OFFSET + b).astype(np.int16)


def trades_tokens(c: pd.DataFrame) -> np.ndarray:
    tpm = np.round(c["trades_per_min"].to_numpy(float), 6)
    with np.errstate(divide="ignore", invalid="ignore"):
        geo = TRADES_EXACT + np.floor(np.log(np.maximum(tpm, 1) / (TRADES_EXACT + 1)) / math.log(TRADES_STEP))
    exact = np.ceil(tpm) - 1  # (0,1] -> 0, (1,2] -> 1, ... (9,10] -> 9
    b = np.where(tpm <= TRADES_EXACT, exact, np.maximum(geo, TRADES_EXACT))
    b = np.clip(np.nan_to_num(b), 0, TRADES_EXACT + TRADES_LOG_BINS - 1).astype(int)
    tok = np.where(np.nan_to_num(tpm) > 0, TRADES_OFFSET + b, TRADES_ZERO)
    return np.where(np.isnan(tpm), TRADES_UNKNOWN, tok).astype(np.int16)


def liquidity_tokens(c: pd.DataFrame) -> np.ndarray:
    liq = c["liquidity_sol"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.log10(np.where(liq > 0, liq, np.nan))
    b = np.searchsorted(LIQ_EDGES, np.nan_to_num(lv, nan=-99), side="right")
    return np.where(np.isnan(liq), LIQ_UNKNOWN, LIQ_OFFSET + b).astype(np.int16)


def curve_tokens(c: pd.DataFrame) -> np.ndarray:
    pct = c["curve_progress_pct"].to_numpy(float)
    grad = c["graduated"].to_numpy(float)
    b = np.clip(np.floor(np.nan_to_num(pct) / CURVE_STEP_PCT), 0, CURVE_BINS - 1).astype(int)
    tok = np.where(grad == 1, CURVE_GRADUATED, CURVE_OFFSET + b)
    return np.where(np.isnan(pct) | np.isnan(grad), CURVE_UNKNOWN, tok).astype(np.int16)


def bundle_tokens(c: pd.DataFrame) -> np.ndarray:
    held = c["bundle_pct"].to_numpy(float)
    launch = c["bundle_launch_pct"].to_numpy(float)
    b = np.searchsorted(BUNDLE_EDGES, np.nan_to_num(held), side="left")
    tok = np.where(held < BUNDLE_SOLD_OUT_PCT, BUNDLE_SOLD_OUT, BUNDLE_OFFSET + b)
    tok = np.where(launch <= 0, BUNDLE_NONE, tok)
    return np.where(np.isnan(launch) | (np.isnan(held) & (launch > 0)), BUNDLE_UNKNOWN, tok).astype(np.int16)


def _whale_tokens(v: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.log10(np.where(v > 0, v, np.nan))
    b = np.searchsorted(WHALE_EDGES, np.nan_to_num(lv, nan=-99), side="right")
    tok = np.where(np.nan_to_num(v) > 0, WHALE_OFFSET + b, WHALE_NONE)
    return np.where(np.isnan(v), WHALE_UNKNOWN, tok).astype(np.int16)


def whale_buy_tokens(c: pd.DataFrame) -> np.ndarray:
    return _whale_tokens(c["max_buy_sol"].to_numpy(float))


def whale_sell_tokens(c: pd.DataFrame) -> np.ndarray:
    return _whale_tokens(c["max_sell_sol"].to_numpy(float))


def windows(tokens: np.ndarray, ends: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """Left-padded windows: row i = tokens[ends[i]-seq_len+1 .. ends[i]]."""
    padded = np.concatenate([np.full(seq_len - 1, PAD, dtype=tokens.dtype), tokens])
    idx = ends[:, None] + np.arange(seq_len)[None, :]
    return padded[idx]


# ---- human-readable vocab (used to write MODEL.md) -----------------------------
def _fmt_pct(x: float) -> str:
    return f"{x:g}%"


def candle_vocab() -> list[tuple[int, str]]:
    ret_labels = []
    e = RET_EDGES_PCT
    for i in range(N_RET):
        k = i - len(e)
        if k == 0:
            ret_labels.append(f"flat (|move| < {_fmt_pct(e[0])})")
        elif k > 0:
            lo, hi = e[k - 1], (e[k] if k < len(e) else None)
            ret_labels.append(f"up {_fmt_pct(lo)} to {_fmt_pct(hi)}" if hi else f"up > {_fmt_pct(lo)}")
        else:
            lo, hi = e[-k - 1], (e[-k] if -k < len(e) else None)
            ret_labels.append(f"down {_fmt_pct(lo)} to {_fmt_pct(hi)}" if hi else f"down > {_fmt_pct(lo)}")
    rg = RANGE_EDGES_PCT
    range_labels = [f"range < {_fmt_pct(rg[0])}"] + [
        f"range {_fmt_pct(rg[i - 1])}-{_fmt_pct(rg[i])}" for i in range(1, len(rg))
    ] + [f"range > {_fmt_pct(rg[-1])}"]
    out = [(PAD, "PAD"), (CANDLE_NO_TRADE, "NO_TRADE (no swaps in this candle)")]
    for r, rl in enumerate(ret_labels):
        for g, gl in enumerate(range_labels):
            out.append((CANDLE_OFFSET + r * N_RANGE + g, f"{rl}, {gl}"))
    return out


def volume_vocab() -> list[tuple[int, str]]:
    out = [(PAD, "PAD"), (VOLUME_ZERO, "ZERO (no volume)")]
    edges = [10 ** x for x in VOL_EDGES]
    out.append((VOLUME_OFFSET, f"< {edges[0]:.4g} SOL"))
    for i in range(1, len(edges)):
        out.append((VOLUME_OFFSET + i, f"{edges[i - 1]:.4g} - {edges[i]:.4g} SOL"))
    out.append((VOLUME_OFFSET + len(edges), f">= {edges[-1]:.4g} SOL"))
    return out


def holders_vocab() -> list[tuple[int, str]]:
    out = [(PAD, "PAD"), (HOLDERS_UNKNOWN, "UNKNOWN (no balance data)"), (HOLDERS_ZERO, "0 holders")]
    for h in range(1, HOLDERS_EXACT + 1):
        out.append((HOLDERS_OFFSET + h - 1, f"{h} holder" + ("s" if h > 1 else "")))
    n_bins = HOLDERS_EXACT + HOLDERS_LOG_BINS
    for k in range(HOLDERS_LOG_BINS):
        lo = math.ceil((HOLDERS_EXACT + 1) * HOLDERS_STEP ** k - 1e-9)
        hi = math.ceil((HOLDERS_EXACT + 1) * HOLDERS_STEP ** (k + 1) - 1e-9) - 1
        b = HOLDERS_EXACT + k
        if b == n_bins - 1:
            out.append((HOLDERS_OFFSET + b, f"{lo:,} or more holders"))
        elif hi >= lo:
            out.append((HOLDERS_OFFSET + b, f"{lo:,} holders" if lo == hi else f"{lo:,} - {hi:,} holders"))
    return out


def top10_vocab() -> list[tuple[int, str]]:
    out = [(PAD, "PAD"), (TOP10_UNKNOWN, "UNKNOWN (no balance data)")]
    for b in range(TOP10_BINS):
        lo, hi = b * TOP10_STEP_PCT, (b + 1) * TOP10_STEP_PCT
        out.append((TOP10_OFFSET + b, f"top 10 hold {lo:g}% - {hi:g}%" + (" (or more)" if b == TOP10_BINS - 1 else "")))
    return out


def trades_vocab() -> list[tuple[int, str]]:
    out = [(PAD, "PAD"), (TRADES_ZERO, "ZERO (no trades in the last minute)"),
           (TRADES_UNKNOWN, "UNKNOWN (trade count not available for this candle)")]
    for k in range(1, TRADES_EXACT + 1):
        out.append((TRADES_OFFSET + k - 1, f"{k} trade{'s' if k > 1 else ''}/min"))
    n = TRADES_EXACT + TRADES_LOG_BINS
    for k in range(TRADES_LOG_BINS):
        lo = (TRADES_EXACT + 1) * TRADES_STEP ** k
        hi = (TRADES_EXACT + 1) * TRADES_STEP ** (k + 1)
        b = TRADES_EXACT + k
        label = f"{lo:,.0f}+ trades/min" if b == n - 1 else f"{lo:,.0f} - {hi:,.0f} trades/min"
        out.append((TRADES_OFFSET + b, label))
    return out


def _sol(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else f"{x:.3g}"


def liquidity_vocab() -> list[tuple[int, str]]:
    out = [(PAD, "PAD"), (LIQ_UNKNOWN, "UNKNOWN (no pool data)")]
    edges = [10 ** x for x in LIQ_EDGES]
    out.append((LIQ_OFFSET, f"< {_sol(edges[0])} SOL"))
    for i in range(1, len(edges)):
        out.append((LIQ_OFFSET + i, f"{_sol(edges[i - 1])} - {_sol(edges[i])} SOL"))
    out.append((LIQ_OFFSET + len(edges), f">= {_sol(edges[-1])} SOL"))
    return out


def curve_vocab() -> list[tuple[int, str]]:
    out = [(PAD, "PAD"), (CURVE_UNKNOWN, "UNKNOWN (not a pump.fun coin, or no data)"),
           (CURVE_GRADUATED, "GRADUATED (left the bonding curve, trades on an AMM)")]
    for b in range(CURVE_BINS):
        lo, hi = b * CURVE_STEP_PCT, (b + 1) * CURVE_STEP_PCT
        out.append((CURVE_OFFSET + b, f"curve {lo:g}% - {hi:g}% complete"))
    return out


def bundle_vocab() -> list[tuple[int, str]]:
    out = [(PAD, "PAD"), (BUNDLE_UNKNOWN, "UNKNOWN (history doesn't reach the launch block)"),
           (BUNDLE_NONE, "NO_BUNDLE (nobody bought in the launch block)"),
           (BUNDLE_SOLD_OUT, f"SOLD_OUT (launch-block wallets hold < {BUNDLE_SOLD_OUT_PCT:g}% of supply now)")]
    e = BUNDLE_EDGES
    out.append((BUNDLE_OFFSET, f"bundle holds {BUNDLE_SOLD_OUT_PCT:g}% - {e[0]:g}%"))
    for i in range(1, len(e)):
        out.append((BUNDLE_OFFSET + i, f"bundle holds {e[i - 1]:g}% - {e[i]:g}%"))
    out.append((BUNDLE_OFFSET + len(e), f"bundle holds > {e[-1]:g}%"))
    return out


def _whale_vocab(side: str) -> list[tuple[int, str]]:
    out = [(PAD, "PAD"), (WHALE_NONE, f"NONE (no {side}s in this candle)"),
           (WHALE_UNKNOWN, "UNKNOWN (per-wallet amounts not available)")]
    edges = [10 ** x for x in WHALE_EDGES]
    out.append((WHALE_OFFSET, f"biggest {side} < {edges[0]:.4g} SOL"))
    for i in range(1, len(edges)):
        out.append((WHALE_OFFSET + i, f"biggest {side} {edges[i - 1]:.4g} - {edges[i]:.4g} SOL"))
    out.append((WHALE_OFFSET + len(edges), f"biggest {side} >= {edges[-1]:.4g} SOL"))
    return out


def whale_buy_vocab() -> list[tuple[int, str]]:
    return _whale_vocab("buy")


def whale_sell_vocab() -> list[tuple[int, str]]:
    return _whale_vocab("sell")
