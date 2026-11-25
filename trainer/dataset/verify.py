"""Verify that the tokenized model inputs are complete, aligned and leak-free.

Checks sequences_30s.npz against dataset_30s.csv (independent recomputation, not just
re-running the same code path).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import tokenizer as tk

# stream -> (tokenize fn, vocab size, vocab fn, source columns, monotonic source col or None,
#            special ids that are not part of the ordered scale)
SPEC = {
    "candle_tokens": (tk.candle_tokens, tk.CANDLE_VOCAB, tk.candle_vocab,
                      ["open", "high", "low", "close", "trades"], None, ()),
    "volume_tokens": (tk.volume_tokens, tk.VOLUME_VOCAB, tk.volume_vocab,
                      ["volume_sol"], "volume_sol", (tk.PAD, tk.VOLUME_ZERO)),
    "holder_tokens": (tk.holder_tokens, tk.HOLDERS_VOCAB, tk.holders_vocab,
                      ["holders"], "holders", (tk.PAD, tk.HOLDERS_UNKNOWN, tk.HOLDERS_ZERO)),
    "top10_tokens": (tk.top10_tokens, tk.TOP10_VOCAB, tk.top10_vocab,
                     ["top10_pct"], "top10_pct", (tk.PAD, tk.TOP10_UNKNOWN)),
    "trades_tokens": (tk.trades_tokens, tk.TRADES_VOCAB, tk.trades_vocab,
                      ["trades_per_min"], "trades_per_min", (tk.PAD, tk.TRADES_ZERO)),
    "liquidity_tokens": (tk.liquidity_tokens, tk.LIQ_VOCAB, tk.liquidity_vocab,
                         ["liquidity_sol"], "liquidity_sol", (tk.PAD, tk.LIQ_UNKNOWN)),
}
UNKNOWN_IDS = {"holder_tokens": tk.HOLDERS_UNKNOWN, "top10_tokens": tk.TOP10_UNKNOWN,
               "liquidity_tokens": tk.LIQ_UNKNOWN}


class Report:
    def __init__(self):
        self.lines: list[tuple[str, str, str]] = []

    def ok(self, name, detail=""):
        self.lines.append(("PASS", name, detail))

    def fail(self, name, detail=""):
        self.lines.append(("FAIL", name, detail))

    def warn(self, name, detail=""):
        self.lines.append(("WARN", name, detail))

    def check(self, cond, name, detail=""):
        (self.ok if cond else self.fail)(name, detail)

    @property
    def failed(self) -> bool:
        return any(s == "FAIL" for s, _, _ in self.lines)

    def print(self) -> None:
        for status, name, detail in self.lines:
            print(f"  [{status}] {name}" + (f": {detail}" if detail else ""))


def verify(seq_path: str, dataset_path: str, seq_len: int = tk.SEQ_LEN, n_samples: int = 300,
           seed: int = 0) -> Report:
    r = Report()
    d = np.load(seq_path)
    df = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    n = len(d["mint"])
    r.check(n > 0, "samples present", f"{n:,} samples, {len(set(d['mint']))} coins")

    # 1. every stream present with the right shape / dtype / id range / known ids
    for k, (_, vocab, vocab_fn, src, _, _) in SPEC.items():
        if k not in d.files:
            r.fail(f"{k} present", "missing from the npz")
            continue
        a = d[k]
        r.check(a.shape == (n, seq_len) and a.dtype == np.int16, f"{k} shape",
                f"{a.shape} {a.dtype}")
        known = {i for i, _ in vocab_fn()}
        used = set(np.unique(a).tolist())
        r.check(used <= known and a.min() >= 0 and a.max() < vocab, f"{k} ids valid",
                f"{len(used)} distinct ids used of {vocab}" + (f", unknown ids {sorted(used - known)[:5]}" if used - known else ""))
        missing_src = [c for c in src if c not in df.columns]
        r.check(not missing_src, f"{k} source columns in dataset", ", ".join(src) if not missing_src else f"missing {missing_src}")
    if r.failed:
        return r

    # 2. padding: left-only, consistent with n_history, current candle never PAD
    nh = d["n_history"].astype(int)
    for k in SPEC:
        a = d[k]
        real = a != tk.PAD
        prefix_ok = np.all(real == (np.arange(seq_len)[None, :] >= seq_len - nh[:, None]))
        r.check(prefix_ok, f"{k} padding", "left-padded, PAD count == 120 - n_history, position 119 is the current candle")

    # 3. alignment: rebuild windows by timestamp from the dataset and compare every token
    rng = np.random.default_rng(seed)
    pick = rng.choice(n, size=min(n_samples, n), replace=False)
    if "segment" not in df.columns:
        df["segment"] = 0
    seg_arr = d["segment"] if "segment" in d.files else np.zeros(n, dtype=int)
    groups = {key: g.sort_values("timestamp").reset_index(drop=True) for key, g in df.groupby(["mint", "segment"])}
    step = None
    mism = {k: 0 for k in SPEC}
    for i in pick:
        g = groups[(d["mint"][i], int(seg_arr[i]))]
        if step is None and len(g) > 1:
            step = g["timestamp"].iloc[1] - g["timestamp"].iloc[0]
        t = pd.Timestamp(int(d["timestamp"][i]), unit="s", tz="UTC")
        pos = int(g["timestamp"].searchsorted(t))
        if pos >= len(g) or g["timestamp"].iloc[pos] != t:
            mism = {k: v + 1 for k, v in mism.items()}
            continue
        lo = max(0, pos - seq_len + 1)
        win = g.iloc[lo: pos + 1]
        # the candle token needs the previous close: tokenize with one extra row in front
        ctx = g.iloc[max(0, lo - 1): pos + 1]
        for k, (fn, *_rest) in SPEC.items():
            toks = fn(ctx)[-len(win):] if k == "candle_tokens" else fn(win)
            expect = np.concatenate([np.zeros(seq_len - len(win), np.int16), toks])
            mism[k] += int(not np.array_equal(expect, d[k][i]))
    for k, bad in mism.items():
        r.check(bad == 0, f"{k} aligned with dataset candles",
                f"{len(pick) - bad}/{len(pick)} random samples match token-for-token")

    # 4. meaning: tokens are ordered like the raw values (bigger value -> bigger-or-equal token)
    for k, (fn, _, _, _, col, special) in SPEC.items():
        if col is None:
            continue
        v = df[col].to_numpy(float)
        tok = fn(df).astype(int)
        m = ~np.isin(tok, special) & ~np.isnan(v)
        if m.sum() < 2:
            r.warn(f"{k} ordering", "not enough non-special values to check")
            continue
        order = np.argsort(v[m], kind="stable")
        r.check(np.all(np.diff(tok[m][order]) >= 0), f"{k} ordering",
                f"token rises monotonically with {col} over {m.sum():,} candles")

    # 5. no look-ahead: targets recomputed from future candles only
    hmap = {"1m": 2, "5m": 10, "10m": 20} if step == pd.Timedelta(seconds=30) else {}
    bad_t = checked = 0
    for i in pick[:100]:
        g = groups[(d["mint"][i], int(seg_arr[i]))]
        t = pd.Timestamp(int(d["timestamp"][i]), unit="s", tz="UTC")
        pos = int(g["timestamp"].searchsorted(t))
        for name, h in hmap.items():
            key = f"target_ret_{name}"
            if key not in d.files or pos + h >= len(g) or np.isnan(d[key][i]):
                continue
            checked += 1
            expect = np.log(g["close"].iloc[pos + h] / g["close"].iloc[pos])
            bad_t += int(not np.isclose(expect, d[key][i], rtol=1e-4, atol=1e-6))
    if checked:
        r.check(bad_t == 0, "targets look only forward", f"{checked - bad_t}/{checked} target values recomputed from close[t+h]")
    targets = [k for k in d.files if k.startswith("target_")]
    r.check("target_up_5m" in targets and "target_dump_5m" in targets, "targets present", ", ".join(targets))

    # 6. data quality
    for k, uid in UNKNOWN_IDS.items():
        real = d[k][d[k] != tk.PAD]
        share = (real == uid).mean() if real.size else 1.0
        (r.warn if share > 0.05 else r.ok)(f"{k} data coverage", f"{share:.1%} UNKNOWN")
    for k in SPEC:
        real = d[k][d[k] != tk.PAD]
        distinct = len(np.unique(real))
        (r.ok if distinct > 1 else r.warn)(f"{k} varies", f"{distinct} distinct tokens in use")
    for k in ("split",):
        splits = dict(zip(*np.unique(d[k], return_counts=True)))
        (r.ok if len(splits) == 3 else r.warn)("train/val/test splits", ", ".join(f"{s}={c:,}" for s, c in splits.items()))
    return r
