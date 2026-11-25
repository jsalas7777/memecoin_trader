"""Convert 30s candles into discrete tokens for a sequence model.

Each candle produces three tokens, one per input stream:
  candle  : price move vs the previous close x candle range (shape of the bar)
  volume  : SOL volume on a log scale
  holders : number of holders on a log scale (5% steps)

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
    return np.where(c["trades"].to_numpy() > 0, tok, CANDLE_NO_TRADE).astype(np.int16)


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
