"""Coin sequences for the causal transformer.

Each coin (each contiguous `segment` of it) is one sequence that starts at its FIRST candle.
The model is causal, so in one pass every position t learns from candles 0..t only:
candle 0 is trained with 1 candle of history, candle 1 with 2, ... No position waits for a
full 120-candle window, and UNKNOWN / empty inputs are simply tokens like any other.

Coins longer than the context (120 candles) are cut into overlapping windows. The first
window trains every position; each later window only trains the positions the previous
window didn't reach, so every candle is trained exactly once per epoch, with between
(context - stride) and context candles of history.

Targets for position t, k = 1..24:
  candle[t, k] = candle token of candle t+k   (shape of the future bar, 86 classes)
  volume[t, k] = volume token of candle t+k   (28 classes)
  ret[t, k]    = log(close[t+k] / close[t])   (price path relative to now)
Future candles past the end of the downloaded data are unknown and masked out of the loss.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from build_sequences import STREAMS
from dataset import tokenizer as tk

log = logging.getLogger(__name__)

STREAM_NAMES = list(STREAMS)
VOCAB_SIZES = [STREAMS[k][1] for k in STREAM_NAMES]
CANDLE = STREAM_NAMES.index("candle_tokens")
VOLUME = STREAM_NAMES.index("volume_tokens")
CONTEXT = tk.SEQ_LEN   # 120 candles of history at most
PRED_LEN = 24          # forecast 24 candles (12 minutes) ahead
IGNORE = -100          # target id ignored by the loss
RET_CLIP = 5.0         # |log return| cap for the regression target (e^5 = 148x)

# coin age (candles since launch) -> embedding id: exact for the first 16 candles, then
# 4 buckets per doubling, up to ~19 days
AGE_EXACT, AGE_PER_DOUBLING, N_AGE = 16, 4, 64


def age_bucket(age_s: np.ndarray, interval_s: float = 30.0) -> np.ndarray:
    c = np.nan_to_num(np.asarray(age_s, float) / interval_s, nan=0.0).clip(min=0)
    with np.errstate(divide="ignore"):
        geo = AGE_EXACT + np.floor(AGE_PER_DOUBLING * np.log2(np.maximum(c, 1) / AGE_EXACT))
    return np.clip(np.where(c < AGE_EXACT, np.floor(c), geo), 0, N_AGE - 1).astype(np.int16)


NEEDED = ["mint", "segment", "split", "timestamp", "age_s", "open", "high", "low", "close", "trades",
          "volume_sol", "holders", "top10_pct", "trades_per_min", "liquidity_sol", "curve_progress_pct",
          "graduated", "bundle_pct", "bundle_launch_pct", "max_buy_sol", "max_sell_sol"]


@dataclass
class Coins:
    """All sequences concatenated; coin i is rows offsets[i]:offsets[i+1]."""
    tokens: np.ndarray      # [N, n_streams] int16
    age: np.ndarray         # [N] int16
    close: np.ndarray       # [N] float64
    timestamp: np.ndarray   # [N] int64 unix s
    offsets: np.ndarray     # [n_coins + 1]
    mint: np.ndarray        # [n_coins] str
    segment: np.ndarray     # [n_coins]
    split: np.ndarray       # [n_coins] str

    def __len__(self) -> int:
        return len(self.mint)

    def rows(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def indices(self, split: str) -> list[int]:
        return [i for i in range(len(self)) if self.split[i] == split]


def load_coins(dataset: str | Path, cache: bool = True) -> Coins:
    """Tokenize the dataset per coin (same tokenizers as build_sequences.py), cached.
    dataset: dataset_30s.csv, a .parquet file, or a folder of part-*.parquet (import_archive.py)."""
    dataset = Path(dataset)
    if dataset.is_dir() and (dataset / "candles").is_dir():
        dataset = dataset / "candles"   # import_archive.py output folder
    files = sorted(dataset.glob("part-*.parquet")) if dataset.is_dir() else [dataset]
    if not files:
        raise SystemExit(f"no data in {dataset}")
    stats = [f.stat() for f in files]
    key = f"{sum(s.st_size for s in stats)}-{max(int(s.st_mtime) for s in stats)}-{len(files)}-{','.join(STREAM_NAMES)}"
    cache_path = dataset / "_coins_cache.npz" if dataset.is_dir() else dataset.with_name(dataset.stem + "_coins.npz")
    if cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if str(z["key"]) == key:
            return Coins(**{f: z[f] for f in Coins.__dataclass_fields__})
    if dataset.suffix == ".csv":
        df = pd.read_csv(dataset, usecols=lambda c: c in NEEDED, parse_dates=["timestamp"])
    else:
        df = pd.concat([pd.read_parquet(f, columns=[c for c in NEEDED if c != "segment"] + ["segment"]) for f in files],
                       ignore_index=True)
    if "segment" not in df:
        df["segment"] = 0
    parts = {k: [] for k in ("tokens", "age", "close", "timestamp")}
    offsets, mints, segs, splits = [0], [], [], []
    groups = df.groupby(["mint", "segment"], sort=False)
    for (mint, seg), g in (tqdm(groups, total=groups.ngroups, desc="tokenizing coins", unit="coin",
                                dynamic_ncols=True, leave=False) if groups.ngroups > 2000 else groups):
        g = g.sort_values("timestamp").reset_index(drop=True)
        parts["tokens"].append(np.stack([STREAMS[k][0](g) for k in STREAM_NAMES], axis=1).astype(np.int16))
        parts["age"].append(age_bucket(g["age_s"].to_numpy()))
        parts["close"].append(g["close"].to_numpy(float))
        parts["timestamp"].append(((g["timestamp"] - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(seconds=1))
                                  .to_numpy(np.int64))
        offsets.append(offsets[-1] + len(g))
        mints.append(mint)
        segs.append(seg)
        splits.append(g["split"].iloc[0])
    coins = Coins(tokens=np.concatenate(parts["tokens"]), age=np.concatenate(parts["age"]),
                  close=np.concatenate(parts["close"]), timestamp=np.concatenate(parts["timestamp"]),
                  offsets=np.array(offsets, np.int64), mint=np.array(mints, str),
                  segment=np.array(segs, np.int64), split=np.array(splits, str))
    if cache:
        tmp = cache_path.with_suffix(".tmp.npz")
        np.savez(tmp, key=np.array(key), **{f: getattr(coins, f) for f in Coins.__dataclass_fields__})
        os.replace(tmp, cache_path)
    return coins


def future_targets(tokens: np.ndarray, close: np.ndarray, pred_len: int = PRED_LEN):
    """Targets for every position of one coin: [L, pred_len] each."""
    L = len(close)
    q = np.arange(L)[:, None] + np.arange(1, pred_len + 1)[None, :]   # future row index
    ok = q < L
    qc = np.minimum(q, L - 1)
    candle = np.where(ok, tokens[qc, CANDLE], IGNORE).astype(np.int64)
    volume = np.where(ok, tokens[qc, VOLUME], IGNORE).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.log(close[qc] / close[:, None])
    ret = np.where(ok & np.isfinite(ret), np.clip(ret, -RET_CLIP, RET_CLIP), np.nan).astype(np.float32)
    return candle, volume, ret


def windows_for(length: int, context: int = CONTEXT, stride: int = CONTEXT // 2) -> list[tuple[int, int, int]]:
    """(start, train_from, end) per window; positions train_from..end-1 get a loss."""
    out, start, trained_to = [], 0, 0
    while True:
        end = min(start + context, length)
        out.append((start, trained_to, end))
        trained_to = end
        if end >= length:
            return out
        start += stride


class CoinWindows(Dataset):
    """One item = one window of one coin, right-padded to `context`."""

    def __init__(self, coins: Coins, idx: list[int], context: int = CONTEXT, stride: int = CONTEXT // 2,
                 pred_len: int = PRED_LEN):
        self.coins, self.context, self.pred_len = coins, context, pred_len
        self.items: list[tuple[int, int, int, int]] = []
        for i in idx:
            r = coins.rows(i)
            for start, train_from, end in windows_for(r.stop - r.start, context, stride):
                self.items.append((i, start, train_from, end))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, n: int):
        i, start, train_from, end = self.items[n]
        r = self.coins.rows(i)
        T, n_real = self.context, end - start
        tokens = np.zeros((T, self.coins.tokens.shape[1]), np.int64)
        tokens[:n_real] = self.coins.tokens[r][start:end]
        age = np.zeros(T, np.int64)
        age[:n_real] = self.coins.age[r][start:end]
        candle = np.full((T, self.pred_len), IGNORE, np.int64)
        volume = np.full((T, self.pred_len), IGNORE, np.int64)
        ret = np.full((T, self.pred_len), np.nan, np.float32)
        # targets for this window only: its candles plus the next pred_len after it
        hi = min(end + self.pred_len, r.stop - r.start)
        c, v, rt = future_targets(self.coins.tokens[r][start:hi], self.coins.close[r][start:hi], self.pred_len)
        candle[:n_real], volume[:n_real], ret[:n_real] = c[:n_real], v[:n_real], rt[:n_real]
        # train only the positions this window is responsible for
        train = np.zeros(T, bool)
        train[train_from - start:n_real] = True
        candle[~train], volume[~train], ret[~train] = IGNORE, IGNORE, np.nan
        return {"tokens": torch.from_numpy(tokens), "age": torch.from_numpy(age),
                "candle": torch.from_numpy(candle), "volume": torch.from_numpy(volume),
                "ret": torch.from_numpy(ret), "n_real": n_real}


def context_for(coins: Coins, i: int, position: int, context: int = CONTEXT):
    """Model input for forecasting from `position` of coin i: the last <= context candles."""
    r = coins.rows(i)
    lo = max(0, position + 1 - context)
    tokens = torch.from_numpy(coins.tokens[r][lo:position + 1].astype(np.int64))[None]
    age = torch.from_numpy(coins.age[r][lo:position + 1].astype(np.int64))[None]
    return tokens, age


def describe(coins: Coins) -> str:
    lengths = np.diff(coins.offsets)
    by = {s: int((coins.split == s).sum()) for s in ("train", "val", "test")}
    return (f"{len(coins):,} coin sequences, {int(lengths.sum()):,} candles "
            f"(median {int(np.median(lengths)) if len(lengths) else 0}, max {int(lengths.max()) if len(lengths) else 0}); "
            f"train {by['train']}, val {by['val']}, test {by['test']}")


def n_params(model: torch.nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def log_ratio_to_pct(x: float) -> float:
    return (math.exp(x) - 1) * 100
