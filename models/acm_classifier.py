# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1,max_len,d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,d]
        L = x.size(1)
        return x + self.pe[:, :L, :]


class TinyTransformerClassifier(nn.Module):
    """
    Input: [B, L, F]
    Output: logits [B, num_classes]
    """
    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.1,
        num_classes: int = 8,
        max_len: int = 2048,
        pooling: str = "mean",  # "mean" or "last"
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pooling = pooling

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,F]
        h = self.proj(x)
        h = self.pos(h)
        h = self.enc(h)  # [B,L,d]

        if self.pooling == "last":
            z = h[:, -1, :]
        else:
            z = h.mean(dim=1)

        logits = self.head(z)
        return logits
