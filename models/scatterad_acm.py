# -*- coding: utf-8 -*-
# models/scatterad_acm.py
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_temporal_adj(L: int, tau: int = 3, self_loop: bool = True, device=None) -> torch.Tensor:
    """
    Directed temporal adjacency on time-nodes:
    for node i, connect edges from i-dt -> i, dt in [1..tau].
    Return: adj_mask [L, L] bool, where adj[i, j]=True means j -> i is an edge.
    """
    adj = torch.zeros((L, L), dtype=torch.bool, device=device)
    for i in range(L):
        if self_loop:
            adj[i, i] = True
        for dt in range(1, tau + 1):
            j = i - dt
            if j >= 0:
                adj[i, j] = True
    return adj


class TemporalEncoder(nn.Module):
    """
    Multi-scale causal conv encoder (paper style):
    multiple branches with different kernel_size & dilation.
    Input:  x [B, L, D]
    Output: z [B, L, H]
    """
    def __init__(
        self,
        in_dim: int,
        hid_dim: int = 64,
        kernel_sizes=(3, 5, 7),
        dilations=(1, 2, 4),
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        self.branches = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            pad = (k - 1) * d  # causal padding, later cut to last L
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(in_dim, hid_dim, kernel_size=k, dilation=d, padding=pad),
                    nn.BatchNorm1d(hid_dim),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.proj = nn.Conv1d(hid_dim * len(self.branches), hid_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] -> [B, D, L]
        x1 = x.transpose(1, 2)
        L = x1.shape[-1]
        outs = []
        for br in self.branches:
            y = br(x1)          # [B, H, L + pad]
            y = y[..., -L:]     # causal cut: keep last L
            outs.append(y)
        y = torch.cat(outs, dim=1)  # [B, H*n_branch, L]
        y = self.proj(y)            # [B, H, L]
        return y.transpose(1, 2)    # [B, L, H]


class GraphAttentionLayer(nn.Module):
    """
    Dense GAT over time nodes (L<=~128 recommended).
    h: [B, L, Fin], adj_mask: [L, L] bool (j->i)
    optional edge_bias: [B, L, L] float, added to attention logits
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        self.concat = concat

    def forward(self, h: torch.Tensor, adj_mask: torch.Tensor, edge_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = h.shape
        Wh = self.W(h)  # [B, L, Fout]
        Fout = Wh.shape[-1]

        Wh_i = Wh.unsqueeze(2).expand(B, L, L, Fout)
        Wh_j = Wh.unsqueeze(1).expand(B, L, L, Fout)
        a_in = torch.cat([Wh_i, Wh_j], dim=-1)           # [B, L, L, 2*Fout]
        e = self.leakyrelu(self.a(a_in).squeeze(-1))     # [B, L, L]

        if edge_bias is not None:
            e = e + edge_bias

        mask = adj_mask.unsqueeze(0).expand(B, L, L)
        e = e.masked_fill(~mask, float("-inf"))

        att = F.softmax(e, dim=-1)   # over neighbors j
        att = self.dropout(att)
        h_prime = torch.matmul(att, Wh)  # [B, L, Fout]
        return F.elu(h_prime) if self.concat else h_prime


class ScatterAD(nn.Module):
    """
    Baseline ScatterAD:
    - Temporal encoder: multi-scale causal conv
    - Topological encoder: GAT on directed temporal graph
    - Fusion: concat(z_t, z_g) -> MLP -> z
    Loss:
    - L_time  : temporal consistency (cos similarity between adjacent z)
    - L_scatter: align to center c on hypersphere (maximize cos(z,c))
    - L_contrast: InfoNCE between temporal view and topological view
    Score:
    - scatter deviation: 1 / min_t cos(z_t, c)
    - time inconsistency: mean ||z_t - z_{t-1}||^2
    """
    def __init__(
        self,
        in_dim: int,
        hid_dim: int = 64,
        tau: int = 3,
        gat_layers: int = 1,
        temp: float = 0.1,
        alpha: float = 1.0,  # contrast
        beta: float = 1.0,   # scatter
        gamma: float = 1.0,  # time
        center_momentum: float = 0.9,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.temporal = TemporalEncoder(in_dim, hid_dim=hid_dim, dropout=dropout)
        self.gat = nn.ModuleList([GraphAttentionLayer(hid_dim, hid_dim, dropout=dropout) for _ in range(gat_layers)])
        self.fuse = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.tau = int(tau)
        self.temp = float(temp)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.center_momentum = float(center_momentum)

        self.register_buffer("center", torch.zeros(hid_dim))
        self._center_inited = False

    @staticmethod
    def _cos(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        a = F.normalize(a, dim=-1, eps=eps)
        b = F.normalize(b, dim=-1, eps=eps)
        return (a * b).sum(dim=-1)

    def _update_center(self, z_norm: torch.Tensor) -> None:
        # z_norm: [B, L, H], already normalized
        batch_mean = z_norm.mean(dim=(0, 1))  # [H]
        if not self._center_inited:
            self.center = batch_mean.detach()
            self._center_inited = True
        else:
            self.center = self.center_momentum * self.center + (1.0 - self.center_momentum) * batch_mean.detach()

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, L, D]
        B, L, _ = x.shape
        z_t = self.temporal(x)  # [B, L, H]
        adj = build_temporal_adj(L, tau=self.tau, device=x.device)
        z_g = z_t
        for layer in self.gat:
            z_g = layer(z_g, adj_mask=adj)  # [B, L, H]
        z = self.fuse(torch.cat([z_t, z_g], dim=-1))  # [B, L, H]
        return {"z_t": z_t, "z_g": z_g, "z": z}

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pack = self.encode(x)
        z_t, z_g, z = pack["z_t"], pack["z_g"], pack["z"]

        z_norm = F.normalize(z, dim=-1)
        self._update_center(z_norm)

        c = F.normalize(self.center, dim=-1).view(1, 1, -1)

        # L_time = mean (1 - cos(z_t, z_{t-1}))^2
        cos_adj = self._cos(z_norm[:, 1:, :], z_norm[:, :-1, :])
        l_time = ((1.0 - cos_adj) ** 2).mean()

        # L_scatter = -mean cos(z, c)  (maximize similarity to center)
        l_scatter = -self._cos(z_norm, c).mean()

        # L_contrast = symmetric InfoNCE between z_t and z_g
        B, L, H = z_t.shape
        q = F.normalize(z_t, dim=-1).reshape(B * L, H)
        k = F.normalize(z_g, dim=-1).reshape(B * L, H)
        logits = (q @ k.t()) / self.temp
        labels = torch.arange(B * L, device=x.device)
        l_contrast = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

        loss = self.alpha * l_contrast + self.beta * l_scatter + self.gamma * l_time
        return {
            "loss": loss,
            "l_contrast": l_contrast.detach(),
            "l_scatter": l_scatter.detach(),
            "l_time": l_time.detach(),
        }

    @torch.no_grad()
    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        # return score per sample: [B]
        pack = self.encode(x)
        z = pack["z"]
        z_norm = F.normalize(z, dim=-1)

        c = F.normalize(self.center, dim=-1).view(1, 1, -1)
        cos_tc = self._cos(z_norm, c)  # [B, L]

        # scatter deviation: 1 / min_t cos(z_t, c)
        scat_dev = 1.0 / cos_tc.min(dim=1).values.clamp(min=1e-4)

        # time inconsistency: mean squared diff
        time_inc = (z_norm[:, 1:, :] - z_norm[:, :-1, :]).pow(2).mean(dim=(1, 2))
        return scat_dev + time_inc
