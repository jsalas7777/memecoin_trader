"""Causal (GPT-style) transformer over the 10 token streams, with a 24-candle forecast head."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from dataset import tokenizer as tk

from .data import CANDLE, CONTEXT, IGNORE, N_AGE, PRED_LEN, STREAM_NAMES, VOCAB_SIZES, VOLUME


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    context: int = CONTEXT
    pred_len: int = PRED_LEN
    streams: list[str] = field(default_factory=lambda: list(STREAM_NAMES))
    vocab_sizes: list[int] = field(default_factory=lambda: list(VOCAB_SIZES))
    n_age: int = N_AGE

    def to_dict(self) -> dict:
        return asdict(self)


class MemecoinTransformer(nn.Module):
    """Input:  tokens [B, T, 10] (one token per stream per candle), age [B, T] (coin-age bucket).
    Output, for EVERY position t (causal: only candles <= t are seen):
      candle [B, T, 24, 86]  logits of the candle token of each of the next 24 candles
      volume [B, T, 24, 28]  logits of their volume token
      ret    [B, T, 24]      predicted log(close[t+k] / close[t]), k = 1..24
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.stream_emb = nn.ModuleList(nn.Embedding(v, d, padding_idx=tk.PAD) for v in cfg.vocab_sizes)
        self.age_emb = nn.Embedding(cfg.n_age, d)
        self.pos_emb = nn.Embedding(cfg.context, d)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.candle_vocab = cfg.vocab_sizes[CANDLE]
        self.volume_vocab = cfg.vocab_sizes[VOLUME]
        self.candle_head = nn.Linear(d, cfg.pred_len * self.candle_vocab)
        self.volume_head = nn.Linear(d, cfg.pred_len * self.volume_vocab)
        self.ret_head = nn.Linear(d, cfg.pred_len)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()

    def forward(self, tokens: torch.Tensor, age: torch.Tensor) -> dict[str, torch.Tensor]:
        B, T, S = tokens.shape
        x = sum(emb(tokens[..., s]) for s, emb in enumerate(self.stream_emb))
        x = x + self.age_emb(age) + self.pos_emb(torch.arange(T, device=tokens.device))[None]
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), diagonal=1)
        h = self.norm(self.encoder(self.drop(x), mask=causal, is_causal=True))
        P = self.cfg.pred_len
        return {"candle": self.candle_head(h).view(B, T, P, self.candle_vocab),
                "volume": self.volume_head(h).view(B, T, P, self.volume_vocab),
                "ret": self.ret_head(h)}


@dataclass
class LossWeights:
    candle: float = 1.0
    volume: float = 0.5
    ret: float = 1.0
    huber_beta: float = 0.1   # log-return error where the loss switches from squared to linear


def compute_loss(out: dict, batch: dict, w: LossWeights) -> tuple[torch.Tensor, dict]:
    c_loss = F.cross_entropy(out["candle"].flatten(0, 2), batch["candle"].flatten(), ignore_index=IGNORE)
    v_loss = F.cross_entropy(out["volume"].flatten(0, 2), batch["volume"].flatten(), ignore_index=IGNORE)
    known = ~torch.isnan(batch["ret"])
    if known.any():
        r_loss = F.smooth_l1_loss(out["ret"][known], batch["ret"][known], beta=w.huber_beta)
    else:
        r_loss = out["ret"].sum() * 0.0
    total = w.candle * c_loss + w.volume * v_loss + w.ret * r_loss
    return total, {"candle": c_loss.item(), "volume": v_loss.item(), "ret": r_loss.item()}


def save(path, model: MemecoinTransformer, extra: dict) -> None:
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict(), **extra}, path)


def load(path, device="cpu") -> tuple[MemecoinTransformer, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = MemecoinTransformer(ModelConfig(**ck["config"])).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
