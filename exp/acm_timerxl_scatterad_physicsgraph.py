# -*- coding: utf-8 -*-
# exp/acm_timerxl_scatterad_physicsgraph.py
import os
import sys
import time
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_loader_acm_320 import FlightDataset_acm
from models.timer_xl import Model as TimerXL
from models.scatterad_physics_acm import ScatterADPhysics


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class TimerXLFeatureExtractor(nn.Module):
    """
    尽量不改你原 timer_xl.py：
    - forward(x) 输出可能是 [B, ?, L] 或 [B, L, ?] 或 tuple
    - 我们统一抽一个 per-step feature: [B, L, H]
    如果抽不到，就 fallback 用模型输出 reshape；再不行用输入 x 作为 feature（保证代码能跑）。
    """
    def __init__(self, timerxl: TimerXL):
        super().__init__()
        self.m = timerxl

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.m(x)
        feat = None
        pred = out

        if isinstance(out, (tuple, list)):
            pred = out[0]
            for t in out[1:]:
                if torch.is_tensor(t) and t.dim() == 3:
                    feat = t
                    break

        if feat is None:
            if torch.is_tensor(pred) and pred.dim() == 3:
                # could be [B,L,H] or [B,H,L]
                if pred.shape[1] == x.shape[1]:
                    feat = pred
                elif pred.shape[2] == x.shape[1]:
                    feat = pred.transpose(1, 2)
                else:
                    feat = pred
            else:
                # last resort
                feat = x

        # ensure [B,L,H]
        if feat.dim() == 2:
            feat = feat.unsqueeze(-1)
        if feat.shape[1] != x.shape[1] and feat.shape[2] == x.shape[1]:
            feat = feat.transpose(1, 2)

        return pred, feat


class WindowForTimerXL(Dataset):
    """
    从 FlightDataset_acm 的 seghead 里切窗口，然后按你“回归任务”的输入/输出方式构造：
      - 输入：5个协变量（mask掉 target）
      - 这里 target 用 PACKx_COMPR_T（与你当前代码一致）
    同时保留 raw 6维（给 PhysicsEdgeBias 用）
    returns:
      x_in:  [L,5]
      x_raw: [L,6]
      meta:  [base_idx, sub_id]
    """
    def __init__(self, base: FlightDataset_acm, win_len: int = 96, stride: int = 96):
        super().__init__()
        self.base = base
        self.win_len = int(win_len)
        self.stride = int(stride)

        names = getattr(base, "feature_names", [])
        n2i = {n: i for i, n in enumerate(names)}

        if base.side == "PACK1":
            all_names = ["PACK1_BYPASS_V","PACK1_DISCH_T","PACK1_RAM_I_DR","PACK1_RAM_O_DR","PACK_FLOW_R1","PACK1_COMPR_T"]
            target = "PACK1_COMPR_T"
        else:
            all_names = ["PACK2_BYPASS_V","PACK2_DISCH_T","PACK2_RAM_I_DR","PACK2_RAM_O_DR","PACK_FLOW_R2","PACK2_COMPR_T"]
            target = "PACK2_COMPR_T"

        miss = [c for c in all_names if c not in n2i]
        if miss:
            raise ValueError(f"Missing columns in base.feature_names: {miss}")

        self.idx_raw = [n2i[c] for c in all_names]  # 6
        self.idx_target = n2i[target]
        self.idx_in = [n2i[c] for c in all_names if c != target]  # 5

        if len(base) > 0:
            self.base_len = int(base.data.shape[1])
        else:
            self.base_len = self.win_len
        if self.base_len < self.win_len:
            raise ValueError(f"base_len={self.base_len} < win_len={self.win_len}")
        self.n_sub = 1 + (self.base_len - self.win_len) // self.stride

    def __len__(self):
        return len(self.base) * self.n_sub

    def __getitem__(self, idx):
        base_idx = idx // self.n_sub
        sub_id = idx % self.n_sub
        st = sub_id * self.stride
        seg = self.base.data[base_idx]  # numpy [keep_len, D]

        win_raw = seg[st:st + self.win_len, :][:, self.idx_raw].astype(np.float32)  # [L,6]
        win_in = seg[st:st + self.win_len, :][:, self.idx_in].astype(np.float32)    # [L,5]

        return (
            torch.from_numpy(win_in),
            torch.from_numpy(win_raw),
            torch.tensor([base_idx, sub_id], dtype=torch.long),
        )


@torch.no_grad()
def eval_scores_per_flight_scatterphysics(
    model_sc: ScatterADPhysics,
    feat_extractor: TimerXLFeatureExtractor,
    base_ds: FlightDataset_acm,
    win_len: int,
    stride: int,
    device,
    agg: str = "max",
) -> pd.DataFrame:
    model_sc.eval()
    feat_extractor.eval()

    ds = WindowForTimerXL(base_ds, win_len=win_len, stride=stride)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    flight_scores: Dict[int, List[float]] = {}

    for x_in, x_raw, meta in tqdm(loader, desc="Scoring windows (TimerXL+Scatter)"):
        x_in = x_in.to(device)    # [B,L,5]
        x_raw = x_raw.to(device)  # [B,L,6]
        _, h = feat_extractor(x_in)  # [B,L,H]
        s = model_sc.anomaly_score(h, x_raw).detach().cpu().numpy().tolist()
        meta = meta.numpy()
        for i in range(len(s)):
            base_idx = int(meta[i, 0])
            flight_scores.setdefault(base_idx, []).append(float(s[i]))

    rows = []
    for base_idx, scores in flight_scores.items():
        if not scores:
            continue
        fs = float(np.mean(scores)) if agg == "mean" else float(np.max(scores))
        tail = base_ds.window_tails[base_idx] if hasattr(base_ds, "window_tails") else "UNKNOWN"
        stime = base_ds.window_start_times[base_idx] if hasattr(base_ds, "window_start_times") else "UNKNOWN"
        rows.append(
            {
                "base_idx": int(base_idx),
                "tail": str(tail),
                "seg_start_time": str(stime),
                "n_windows": int(len(scores)),
                "flight_score": fs,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["tail", "seg_start_time"], ascending=True)
    return df


def alarm_by_tail_trend(df_flight: pd.DataFrame, thr: float, k: int = 3) -> pd.DataFrame:
    if df_flight.empty:
        return df_flight
    out = []
    for tail, g in df_flight.groupby("tail"):
        g = g.sort_values("seg_start_time", ascending=True).reset_index(drop=True)
        scores = g["flight_score"].values.astype(float)
        alarms = []
        for i in range(len(scores)):
            s = scores[i]
            up = False
            if i >= k - 1:
                window = scores[i - (k - 1): i + 1]
                up = bool(np.all(np.diff(window) > 0.0) and (window[-1] >= 0.8 * thr))
            alarms.append(int((s >= thr) or up))
        g["alarm"] = alarms
        g["thr"] = float(thr)
        out.append(g)
    return pd.concat(out, axis=0).reset_index(drop=True)


def train_timerxl_scatter_physics(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Stage A: load pretrained TimerXL =====
    timerxl = TimerXL(args.timerxl_cfg).to(device)
    state = safe_torch_load(args.timerxl_ckpt, device)
    timerxl.load_state_dict(state)
    timerxl.eval()
    for p in timerxl.parameters():
        p.requires_grad_(False)

    feat_extractor = TimerXLFeatureExtractor(timerxl).to(device)

    # ===== Stage B: train ScatterADPhysics on normal =====
    train_base = FlightDataset_acm(args, Tag="train_normal", side=args.side)
    full_ds = WindowForTimerXL(train_base, win_len=args.win_len, stride=args.stride)
    if len(full_ds) == 0:
        raise RuntimeError("WindowForTimerXL empty, check data range / filters.")

    n_val = max(1, int(len(full_ds) * args.val_ratio))
    n_train = len(full_ds) - n_val
    g = torch.Generator().manual_seed(args.split_seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # infer hidden dim
    with torch.no_grad():
        x_in, x_raw, _ = next(iter(train_loader))
        x_in = x_in.to(device)
        x_raw = x_raw.to(device)
        _, h = feat_extractor(x_in)
        h_in = int(h.shape[-1])
        raw_dim = int(x_raw.shape[-1])

    model_sc = ScatterADPhysics(
        h_in=h_in,
        raw_dim=raw_dim,
        hid_dim=args.hid_dim,
        tau=args.tau,
        gat_layers=args.gat_layers,
        temp=args.temp,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        center_momentum=args.center_momentum,
        dropout=args.dropout,
        prior_decay=args.prior_decay,
    ).to(device)

    optim = torch.optim.AdamW(model_sc.parameters(), lr=args.lr, weight_decay=args.wd)

    save_dir = os.path.join(args.checkpoints, args.setting)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best_timerxl_scatter_physics.pth")
    final_path = os.path.join(save_dir, "final_timerxl_scatter_physics.pth")

    best_val = 1e18
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model_sc.train()
        tr_sum, tr_n = 0.0, 0
        for x_in, x_raw, _ in train_loader:
            x_in = x_in.to(device)
            x_raw = x_raw.to(device)
            with torch.no_grad():
                _, h = feat_extractor(x_in)

            out = model_sc(h, x_raw)
            loss = out["loss"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_sc.parameters(), args.grad_clip)
            optim.step()

            bsz = x_in.size(0)
            tr_sum += float(loss.item()) * bsz
            tr_n += bsz

        train_loss = tr_sum / max(1, tr_n)

        model_sc.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for x_in, x_raw, _ in val_loader:
                x_in = x_in.to(device)
                x_raw = x_raw.to(device)
                _, h = feat_extractor(x_in)
                out = model_sc(h, x_raw)
                loss = out["loss"]
                bsz = x_in.size(0)
                va_sum += float(loss.item()) * bsz
                va_n += bsz
        val_loss = va_sum / max(1, va_n)

        dt = time.time() - t0
        print(f"[Ep {ep}/{args.epochs}] train={train_loss:.6f} val={val_loss:.6f} | {dt:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model_sc.state_dict(), best_path)
            print(f"  -> new best: {best_path} (val={best_val:.6f})")

    torch.save(model_sc.state_dict(), final_path)
    print("Saved final:", final_path)

    # ===== threshold from train_normal flight scores =====
    model_sc.load_state_dict(torch.load(best_path, map_location=device))
    model_sc.eval()

    df_train = eval_scores_per_flight_scatterphysics(model_sc, feat_extractor, train_base, args.win_len, args.stride, device, agg=args.agg)
    if df_train.empty:
        raise RuntimeError("train flight score empty; cannot compute threshold.")
    thr = float(np.quantile(df_train["flight_score"].values, args.thr_q))

    for tag in ["test_normal_recent", "test_abnormal"]:
        base = FlightDataset_acm(args, Tag=tag, side=args.side)
        if len(base) == 0:
            print(f"[WARN] {tag} empty, skip.")
            continue

        df = eval_scores_per_flight_scatterphysics(model_sc, feat_extractor, base, args.win_len, args.stride, device, agg=args.agg)
        df = alarm_by_tail_trend(df, thr=thr, k=args.trend_k)

        out_dir = os.path.join(save_dir, "results", tag)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"per_flight_{tag}.csv"), index=False)
        print(f"[{tag}] saved:", os.path.join(out_dir, f"per_flight_{tag}.csv"))

    print("Done.")


@dataclass
class TimerXLConfigsLike:
    # 你 TimerXL 的 config（保持你原来的字段名）
    input_token_len: int = 96
    d_model: int = 128
    n_heads: int = 4
    e_layers: int = 3
    d_ff: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    output_attention: bool = False
    covariate: bool = False
    flash_attention: bool = False
    use_norm: bool = False


@dataclass
class Args:
    # ==== your dataset args ====
    seq_len: int = 96
    max_windows_per_flight: int = 5
    normal_months: int = 10
    test_normal_months: int = 1
    fault_gap_months: int = 6
    normal_anchor_end: str = "2025-08-01"
    raw_months: int = 12
    raw_end_use_gap: bool = False
    flight_gap_threshold_sec: float = 3600.0
    dataset_scale: bool = True
    verbose_raw: bool = False
    verbose_every_n_param: int = 1
    verbose_flush: bool = True
    verbose_ds: bool = False

    # ==== run ====
    side: str = "PACK2"
    win_len: int = 96
    stride: int = 96
    batch_size: int = 256
    num_workers: int = 4
    val_ratio: float = 0.1
    split_seed: int = 42

    # ==== ScatterADPhysics ====
    hid_dim: int = 64
    tau: int = 3
    gat_layers: int = 1
    temp: float = 0.1
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    center_momentum: float = 0.9
    dropout: float = 0.1
    prior_decay: float = 0.2

    lr: float = 3e-4
    wd: float = 1e-4
    epochs: int = 30
    grad_clip: float = 5.0

    agg: str = "max"
    thr_q: float = 0.99
    trend_k: int = 3

    checkpoints: str = "./checkpoints"
    setting: str = "timerxl_scatter_physics_acm"

    # ==== TimerXL checkpoint ====
    timerxl_ckpt: str = "./checkpoints/your_timerxl_path/best_timerxl_regress_win96.pth"

    # TimerXL cfg object (you can swap to your TimerXLConfigs(args) too)
    timerxl_cfg: TimerXLConfigsLike = TimerXLConfigsLike(input_token_len=96)


if __name__ == "__main__":
    args = Args()
    train_timerxl_scatter_physics(args)
