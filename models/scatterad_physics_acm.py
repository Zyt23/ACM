# -*- coding: utf-8 -*-
# models/scatterad_physics_acm.py
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.scatterad_acm import build_temporal_adj, GraphAttentionLayer


class PhysicsEdgeBias(nn.Module):
    """
    Physics-guided edge bias for time-graph attention:
    bias(i,j) = MLP([x_i_raw, x_j_raw, dt_emb]) + prior(dt)
    where dt = i - j in [0..tau], only valid on edges.
    """
    def __init__(self, raw_dim: int, tau: int, emb_dim: int = 8, hid: int = 32, prior_decay: float = 0.2):
        super().__init__()
        self.tau = int(tau)
        self.dt_emb = nn.Embedding(self.tau + 1, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim * 2 + emb_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, 1),
        )
        self.prior_decay = float(prior_decay)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        x_raw: [B, L, Draw]
        return edge_bias: [B, L, L]
        """
        B, L, D = x_raw.shape
        device = x_raw.device

        # dt matrix: dt[i,j] = clamp(i-j, 0..tau) for valid edges
        idx = torch.arange(L, device=device)
        dt = (idx.view(L, 1) - idx.view(1, L)).clamp(min=0, max=self.tau)  # [L,L]

        # broadcast raw features
        xi = x_raw.unsqueeze(2).expand(B, L, L, D)  # [B,L,L,D]
        xj = x_raw.unsqueeze(1).expand(B, L, L, D)  # [B,L,L,D]
        dt_e = self.dt_emb(dt).unsqueeze(0).expand(B, L, L, -1)            # [B,L,L,E]

        feat = torch.cat([xi, xj, dt_e], dim=-1)                           # [B,L,L,2D+E]
        bias = self.mlp(feat).squeeze(-1)                                   # [B,L,L]

        # simple physical prior: prefer shorter lag (bigger bias for small dt)
        prior = (-self.prior_decay * dt.float()).unsqueeze(0)              # [1,L,L]
        return bias + prior


class ScatterADPhysics(nn.Module):
    """
    ScatterAD on hidden representation, with physics-guided edge bias inside GAT.

    Inputs:
      - h:     [B, L, H_in]  (TimerXL hidden/features per step)
      - x_raw: [B, L, D_raw] (raw variables, used only to build physics edge bias)

    Encoders:
      - z_t = Linear(h)
      - z_g = GAT(z_t, adj, edge_bias=PhysicsEdgeBias(x_raw))
      - fuse -> z
    Loss/score same as baseline ScatterAD.
    """
    def __init__(
        self,
        h_in: int,
        raw_dim: int,
        hid_dim: int = 64,
        tau: int = 3,
        gat_layers: int = 1,
        temp: float = 0.1,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        center_momentum: float = 0.9,
        dropout: float = 0.1,
        prior_decay: float = 0.2,
    ):
        super().__init__()
        self.tau = int(tau)
        self.temp = float(temp)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.center_momentum = float(center_momentum)

        self.in_proj = nn.Sequential(
            nn.Linear(h_in, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.edge_bias = PhysicsEdgeBias(raw_dim=raw_dim, tau=self.tau, prior_decay=prior_decay)

        self.gat = nn.ModuleList([GraphAttentionLayer(hid_dim, hid_dim, dropout=dropout) for _ in range(gat_layers)])
        self.fuse = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.register_buffer("center", torch.zeros(hid_dim))
        self._center_inited = False

    @staticmethod
    def _cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        a = F.normalize(a, dim=-1, eps=eps)
        b = F.normalize(b, dim=-1, eps=eps)
        return (a * b).sum(dim=-1)

    def _update_center(self, z_norm: torch.Tensor) -> None:
        batch_mean = z_norm.mean(dim=(0, 1))
        if not self._center_inited:
            self.center = batch_mean.detach()
            self._center_inited = True
        else:
            self.center = self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_mean.detach()

    def encode(self, h: torch.Tensor, x_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        # h: [B,L,H_in], x_raw: [B,L,Draw]
        B, L, _ = h.shape
        z_t = self.in_proj(h)  # [B,L,H]
        adj = build_temporal_adj(L, tau=self.tau, device=h.device)
        eb = self.edge_bias(x_raw)  # [B,L,L]
        z_g = z_t
        for layer in self.gat:
            z_g = layer(z_g, adj_mask=adj, edge_bias=eb)
        z = self.fuse(torch.cat([z_t, z_g], dim=-1))
        return {"z_t": z_t, "z_g": z_g, "z": z}

    def forward(self, h: torch.Tensor, x_raw: torch.Tensor) -> Dict[str, torch.Tensor]:
        pack = self.encode(h, x_raw)
        z_t, z_g, z = pack["z_t"], pack["z_g"], pack["z"]

        z_norm = F.normalize(z, dim=-1)
        self._update_center(z_norm)
        c = F.normalize(self.center, dim=-1).view(1, 1, -1)

        cos_adj = self._cos(z_norm[:, 1:, :], z_norm[:, :-1, :])
        l_time = ((1.0 - cos_adj) ** 2).mean()

        l_scatter = -self._cos(z_norm, c).mean()

        B, L, H = z_t.shape
        q = F.normalize(z_t, dim=-1).reshape(B * L, H)
        k = F.normalize(z_g, dim=-1).reshape(B * L, H)
        logits = (q @ k.t()) / self.temp
        labels = torch.arange(B * L, device=h.device)
        l_contrast = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

        loss = self.alpha * l_contrast + self.beta * l_scatter + self.gamma * l_time
        return {
            "loss": loss,
            "l_contrast": l_contrast.detach(),
            "l_scatter": l_scatter.detach(),
            "l_time": l_time.detach(),
        }

    @torch.no_grad()
    def anomaly_score(self, h: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        pack = self.encode(h, x_raw)
        z = pack["z"]
        z_norm = F.normalize(z, dim=-1)
        c = F.normalize(self.center, dim=-1).view(1, 1, -1)

        cos_tc = self._cos(z_norm, c)
        scat_dev = 1.0 / cos_tc.min(dim=1).values.clamp(min=1e-4)
        time_inc = (z_norm[:, 1:, :] - z_norm[:, :-1, :]).pow(2).mean(dim=(1, 2))
        return scat_dev + time_inc
