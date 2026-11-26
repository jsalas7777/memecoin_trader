#!/usr/bin/env python3
"""Show the model's 24-candle forecast for one coin of the dataset.

  python predict.py                          # a random test coin, at a random candle
  python predict.py --mint <address>         # that coin, at its last candle with a known future
  python predict.py --mint <address> --at 15 # forecast made at candle 15 (16 candles of history)

Prints, for each of the next 24 candles (30s each): the most likely candle, its
probability, the predicted price change since now, and what actually happened.
"""
from __future__ import annotations

import argparse
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from build_sequences import STREAMS
from model.data import CANDLE, VOLUME, context_for, load_coins, log_ratio_to_pct
from model.transformer import load

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(HERE / "data" / "models" / "transformer_30s.pt"))
    p.add_argument("--dataset", default=str(HERE / "data" / "dataset" / "dataset_30s.csv"))
    p.add_argument("--mint")
    p.add_argument("--at", type=int, help="candle index to forecast from (0 = first candle)")
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    rng = random.Random(args.seed)

    model, ck = load(args.model)
    P = model.cfg.pred_len
    coins = load_coins(args.dataset)
    if args.mint:
        idx = [i for i in range(len(coins)) if coins.mint[i] == args.mint]
        if not idx:
            raise SystemExit(f"{args.mint} is not in {args.dataset}")
    else:
        idx = coins.indices("test") or list(range(len(coins)))
    i = rng.choice(idx)
    r = coins.rows(i)
    L = r.stop - r.start
    at = args.at if args.at is not None else (rng.randrange(L) if not args.mint else max(0, L - 1 - P))
    at = min(max(at, 0), L - 1)

    tokens, age = context_for(coins, i, at, model.cfg.context)
    with torch.no_grad():
        out = model(tokens, age)
    # PAD (id 0) only marks "no candle" in the input; it is never a real outcome
    probs = torch.softmax(out["candle"][0, -1].index_fill(-1, torch.tensor([0]), -1e9), -1)   # [P, 86]
    vol_probs = torch.softmax(out["volume"][0, -1].index_fill(-1, torch.tensor([0]), -1e9), -1)
    ret = out["ret"][0, -1].numpy()
    candle_label = dict(STREAMS["candle_tokens"][2]())
    volume_label = dict(STREAMS["volume_tokens"][2]())

    close, ts = coins.close[r], coins.timestamp[r]
    hist = tokens.shape[1]
    print(f"Coin {coins.mint[i]} ({coins.split[i]} split), forecast made at candle {at} "
          f"({datetime.fromtimestamp(int(ts[at]), timezone.utc):%Y-%m-%d %H:%M:%S} UTC), "
          f"using {hist} candle(s) of history. Model from epoch {ck.get('epoch')}.\n")
    print(f"{'k':>3} {'time':>6}  {'most likely candle':<40} {'p':>5}  {'volume':<22} {'pred Δ':>8}  {'actual Δ':>8}")
    for k in range(P):
        q = at + k + 1
        c = int(probs[k].argmax())
        v = int(vol_probs[k].argmax())
        actual = f"{(close[q] / close[at] - 1) * 100:+7.1f}%" if q < L else "     ?"
        print(f"{k + 1:>3} {f'+{(k + 1) * 30}s':>6}  {candle_label[c][:40]:<40} {probs[k, c].item():>5.0%}  "
              f"{volume_label[v][:22]:<22} {log_ratio_to_pct(float(ret[k])):+7.1f}%  {actual:>8}")
    best_k = int(np.argmax(ret))
    worst_k = int(np.argmin(ret))
    print(f"\nPredicted best point:  {log_ratio_to_pct(float(ret[best_k])):+.1f}% at +{(best_k + 1) * 30}s;"
          f"  worst: {log_ratio_to_pct(float(ret[worst_k])):+.1f}% at +{(worst_k + 1) * 30}s.")
    p_dead = float(probs[:, 1].mean())   # token 1 = NO_TRADE
    print(f"Average chance a future candle has no trades at all: {p_dead:.0%}")


if __name__ == "__main__":
    main()
