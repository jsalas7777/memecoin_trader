#!/usr/bin/env python3
"""Train the causal memecoin transformer: reads each coin from its first candle, and at
every candle forecasts the next 24 candles (candle shape, volume and price path).

  python train.py                       # defaults: 30 epochs, d_model 128, 4 layers, on the Mac GPU
  python train.py --epochs 50 --d-model 256 --layers 6
  python train.py --quick               # 2 epochs, tiny model: check the pipeline works

Reads   data/dataset/dataset_30s.csv  (from build_dataset.py / download_free.py)
Writes  data/models/transformer_30s.pt        best checkpoint (lowest validation loss)
        data/models/transformer_30s.json      config + per-epoch metrics

Training is causal and sequential: position t of a coin only sees candles 0..t, so every
candle is a training example from the very first one (1 candle of history), without
waiting for 120 candles. See MODEL.md, "The model".
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.data import (CONTEXT, PRED_LEN, CoinWindows, describe, load_coins, log_ratio_to_pct,
                        n_params)
from model.transformer import LossWeights, MemecoinTransformer, ModelConfig, compute_loss, save

log = logging.getLogger("train")
HERE = Path(__file__).resolve().parent
H5M = 10  # 10 candles = 5 minutes


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to(batch: dict, device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, w: LossWeights, device) -> dict:
    model.eval()
    sums = {"loss": 0.0, "candle": 0.0, "volume": 0.0, "ret": 0.0}
    n_batches = 0
    acc1 = accN = n1 = nN = 0
    mae5 = base5 = mae24 = base24 = n5 = n24 = 0.0
    dir_ok = dir_n = 0
    for batch in loader:
        batch = to(batch, device)
        out = model(batch["tokens"], batch["age"])
        loss, parts = compute_loss(out, batch, w)
        sums["loss"] += loss.item()
        for k in parts:
            sums[k] += parts[k]
        n_batches += 1
        pred_c = out["candle"].argmax(-1)
        for k, (acc_name, cnt_name) in ((0, ("acc1", "n1")), (PRED_LEN - 1, ("accN", "nN"))):
            tgt = batch["candle"][..., k]
            m = tgt >= 0
            if acc_name == "acc1":
                acc1 += (pred_c[..., k][m] == tgt[m]).sum().item(); n1 += m.sum().item()
            else:
                accN += (pred_c[..., k][m] == tgt[m]).sum().item(); nN += m.sum().item()
        r, pr = batch["ret"], out["ret"]
        for k, acc in ((H5M - 1, "5"), (PRED_LEN - 1, "24")):
            m = ~torch.isnan(r[..., k])
            if not m.any():
                continue
            err = (pr[..., k][m] - r[..., k][m]).abs().sum().item()
            base = r[..., k][m].abs().sum().item()   # baseline: "price stays the same"
            if acc == "5":
                mae5 += err; base5 += base; n5 += m.sum().item()
                big = m & (r[..., k].abs() > math.log(1.05))   # moves of more than 5%
                dir_ok += (torch.sign(pr[..., k][big]) == torch.sign(r[..., k][big])).sum().item()
                dir_n += big.sum().item()
            else:
                mae24 += err; base24 += base; n24 += m.sum().item()
    model.train()
    nb = max(n_batches, 1)
    return {
        "loss": sums["loss"] / nb, "loss_candle": sums["candle"] / nb, "loss_volume": sums["volume"] / nb,
        "loss_ret": sums["ret"] / nb,
        "candle_acc_next": acc1 / max(n1, 1), f"candle_acc_{PRED_LEN}": accN / max(nN, 1),
        "ret_mae_5m": mae5 / max(n5, 1), "ret_mae_5m_baseline": base5 / max(n5, 1),
        "ret_mae_12m": mae24 / max(n24, 1), "ret_mae_12m_baseline": base24 / max(n24, 1),
        "direction_acc_5m": dir_ok / dir_n if dir_n else float("nan"), "direction_n_5m": dir_n,
    }


def fmt_metrics(m: dict) -> str:
    return (f"loss {m['loss']:.3f} | next-candle acc {m['candle_acc_next']:.1%} | "
            f"5m return MAE {m['ret_mae_5m']:.3f} (no-change baseline {m['ret_mae_5m_baseline']:.3f}) | "
            f"5m direction on >5% moves {m['direction_acc_5m']:.1%} (n={m['direction_n_5m']:,})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(HERE / "data" / "dataset" / "dataset_30s.csv"))
    p.add_argument("--out", default=str(HERE / "data" / "models" / "transformer_30s.pt"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--stride", type=int, default=CONTEXT // 2,
                   help=f"window step for coins longer than {CONTEXT} candles (default {CONTEXT // 2})")
    p.add_argument("--patience", type=int, default=5, help="stop after N epochs without a better val loss")
    p.add_argument("--device", default="auto", help="auto (default: cuda > mps > cpu), cpu, mps, cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="2 epochs, tiny model: smoke test")
    args = p.parse_args()
    if args.quick:
        args.epochs, args.d_model, args.layers, args.heads = 2, 32, 2, 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    device = pick_device(args.device)
    coins = load_coins(args.dataset)
    log.info("data: %s", describe(coins))
    tr_idx, va_idx, te_idx = coins.indices("train"), coins.indices("val"), coins.indices("test")
    if not tr_idx:
        raise SystemExit("no training coins; download more data (python download_free.py)")
    if not va_idx:
        k = max(1, len(tr_idx) // 10)
        va_idx, tr_idx = tr_idx[:k], tr_idx[k:] or tr_idx
        log.warning("no validation coins in the dataset; holding out %d training coin(s) instead", k)
    train_ds = CoinWindows(coins, tr_idx, stride=args.stride)
    val_ds = CoinWindows(coins, va_idx, stride=args.stride)
    test_ds = CoinWindows(coins, te_idx, stride=args.stride) if te_idx else None
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    cfg = ModelConfig(d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, dropout=args.dropout)
    model = MemecoinTransformer(cfg).to(device)
    w = LossWeights()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    steps = args.epochs * max(1, len(train_dl))
    warmup = max(1, min(500, steps // 20))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    log.info("model: %s parameters, d_model %d, %d layers, %d heads, context %d candles, forecast %d candles; device %s",
             n_params(model), cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.context, cfg.pred_len, device)
    log.info("windows: train %d, val %d%s", len(train_ds), len(val_ds), f", test {len(test_ds)}" if test_ds else "")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    history, best, bad = [], float("inf"), 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        bar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", unit="batch", dynamic_ncols=True)
        run = None
        for batch in bar:
            batch = to(batch, device)
            loss, parts = compute_loss(model(batch["tokens"], batch["age"]), batch, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
            bar.set_postfix_str(f"loss {run:.3f} (candle {parts['candle']:.2f}, ret {parts['ret']:.3f})")
        m = evaluate(model, val_dl, w, device)
        m["epoch"], m["train_loss"] = epoch, run
        history.append(m)
        improved = m["loss"] < best
        tqdm.write(f"  val {fmt_metrics(m)}" + ("  ✓ saved" if improved else ""))
        if improved:
            best, bad = m["loss"], 0
            save(out, model, {"epoch": epoch, "val": m, "args": vars(args)})
        else:
            bad += 1
            if bad >= args.patience:
                tqdm.write(f"  no improvement for {args.patience} epochs, stopping")
                break

    result = {"config": cfg.to_dict(), "args": vars(args), "history": history,
              "best_val_loss": best, "train_time_s": time.time() - t0}
    if test_ds and len(test_ds):
        from model.transformer import load
        best_model, _ = load(out, device)
        result["test"] = evaluate(best_model, DataLoader(test_ds, batch_size=args.batch_size), w, device)
        log.info("test (coins never seen in training): %s", fmt_metrics(result["test"]))
    out.with_suffix(".json").write_text(json.dumps(result, indent=1, default=float))
    log.info("best model: %s (val loss %.3f), metrics: %s", out, best, out.with_suffix(".json"))
    log.info("forecast a coin: python predict.py")


if __name__ == "__main__":
    main()
