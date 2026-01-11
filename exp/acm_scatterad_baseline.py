# -*- coding: utf-8 -*-
# exp/acm_scatterad_baseline.py
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_loader_acm_320 import FlightDataset_acm
from models.scatterad_acm import ScatterAD


class WindowFromSegHead(Dataset):
    """
    base_dataset: FlightDataset_acm, each item is [keep_len, D]
    slice windows: [win_len, D] with stride
    returns:
      x: [win_len, D] (float32)
      meta: (base_idx, sub_id)
    """
    def __init__(self, base_dataset: FlightDataset_acm, win_len: int = 96, stride: int = 96):
        super().__init__()
        self.base = base_dataset
        self.win_len = int(win_len)
        self.stride = int(stride)
        if len(self.base) > 0:
            self.base_len = int(self.base.data.shape[1])
            self.D = int(self.base.data.shape[2])
        else:
            self.base_len = self.win_len
            self.D = 0
        if self.base_len < self.win_len:
            raise ValueError(f"base_len={self.base_len} < win_len={self.win_len}")
        self.n_sub = 1 + (self.base_len - self.win_len) // self.stride

    def __len__(self):
        return len(self.base) * self.n_sub

    def __getitem__(self, idx):
        base_idx = idx // self.n_sub
        sub_id = idx % self.n_sub
        st = sub_id * self.stride
        seg = self.base[base_idx]  # numpy [keep_len, D]
        x = seg[st:st + self.win_len, :].astype(np.float32)
        return torch.from_numpy(x), torch.tensor([base_idx, sub_id], dtype=torch.long)


@torch.no_grad()
def eval_scores_per_flight(
    model: ScatterAD,
    base_ds: FlightDataset_acm,
    win_len: int,
    stride: int,
    device,
    agg: str = "max",
) -> pd.DataFrame:
    """
    Compute per-flight (per seghead) anomaly score.
    agg: max / mean
    """
    model.eval()
    ds = WindowFromSegHead(base_ds, win_len=win_len, stride=stride)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # collect window scores per base_idx
    flight_scores: Dict[int, List[float]] = {}

    for x, meta in tqdm(loader, desc="Scoring windows"):
        x = x.to(device)  # [B, L, D]
        s = model.anomaly_score(x).detach().cpu().numpy().tolist()
        meta = meta.numpy()
        for i in range(len(s)):
            base_idx = int(meta[i, 0])
            flight_scores.setdefault(base_idx, []).append(float(s[i]))

    rows = []
    for base_idx, scores in flight_scores.items():
        if len(scores) == 0:
            continue
        if agg == "mean":
            fs = float(np.mean(scores))
        else:
            fs = float(np.max(scores))

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
    """
    你的逻辑：持续上升 or 短期高于阈值报警
    这里用 flight_score 做：
      - 如果 flight_score >= thr -> alarm
      - 或最近 k 个 flight_score 严格递增且最后一个 >= 0.8*thr -> alarm
    """
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


def train_scatterad_baseline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) datasets (reuse your cache)
    train_base = FlightDataset_acm(args, Tag="train_normal", side=args.side)
    full_ds = WindowFromSegHead(train_base, win_len=args.win_len, stride=args.stride)
    n_total = len(full_ds)
    if n_total == 0:
        raise RuntimeError("Window dataset is empty. Check IoTDB / time range / filters.")

    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(args.split_seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    D = int(train_base.data.shape[2])
    model = ScatterAD(
        in_dim=D,
        hid_dim=args.hid_dim,
        tau=args.tau,
        gat_layers=args.gat_layers,
        temp=args.temp,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        center_momentum=args.center_momentum,
        dropout=args.dropout,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    save_dir = os.path.join(args.checkpoints, args.setting)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best_scatterad_baseline.pth")
    final_path = os.path.join(save_dir, "final_scatterad_baseline.pth")

    best_val = 1e18
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_sum, tr_n = 0.0, 0
        for x, _ in train_loader:
            x = x.to(device)
            out = model(x)
            loss = out["loss"]
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            bsz = x.size(0)
            tr_sum += float(loss.item()) * bsz
            tr_n += bsz

        train_loss = tr_sum / max(1, tr_n)

        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                out = model(x)
                loss = out["loss"]
                bsz = x.size(0)
                va_sum += float(loss.item()) * bsz
                va_n += bsz
        val_loss = va_sum / max(1, va_n)
        dt = time.time() - t0

        print(f"[Ep {ep}/{args.epochs}] train={train_loss:.6f} val={val_loss:.6f} | {dt:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best: {best_path} (val={best_val:.6f})")

    torch.save(model.state_dict(), final_path)
    print("Saved final:", final_path)

    # 2) threshold from train_normal flight scores (quantile)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    df_train_score = eval_scores_per_flight(model, train_base, win_len=args.win_len, stride=args.stride, device=device, agg=args.agg)
    if df_train_score.empty:
        raise RuntimeError("df_train_score empty, cannot compute threshold.")
    thr = float(np.quantile(df_train_score["flight_score"].values, args.thr_q))

    # 3) eval on recent normal & abnormal
    for tag in ["test_normal_recent", "test_abnormal"]:
        base = FlightDataset_acm(args, Tag=tag, side=args.side)
        if len(base) == 0:
            print(f"[WARN] {tag} dataset empty, skip.")
            continue
        df = eval_scores_per_flight(model, base, win_len=args.win_len, stride=args.stride, device=device, agg=args.agg)
        df = alarm_by_tail_trend(df, thr=thr, k=args.trend_k)

        out_dir = os.path.join(save_dir, "results", tag)
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, f"per_flight_{tag}.csv"), index=False)
        print(f"[{tag}] saved:", os.path.join(out_dir, f"per_flight_{tag}.csv"))

    print("Done.")


@dataclass
class Args:
    # ==== your dataset args (keep compatible) ====
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

    # ==== scatterad training ====
    side: str = "PACK2"
    win_len: int = 96
    stride: int = 96
    batch_size: int = 256
    num_workers: int = 4
    val_ratio: float = 0.1
    split_seed: int = 42

    hid_dim: int = 64
    tau: int = 3
    gat_layers: int = 1
    temp: float = 0.1
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    center_momentum: float = 0.9
    dropout: float = 0.1

    lr: float = 3e-4
    wd: float = 1e-4
    epochs: int = 30
    grad_clip: float = 5.0

    agg: str = "max"        # per-flight aggregation: max / mean
    thr_q: float = 0.99     # threshold quantile from train_normal
    trend_k: int = 3

    checkpoints: str = "./checkpoints"
    setting: str = "scatterad_baseline_acm"


if __name__ == "__main__":
    args = Args()
    train_scatterad_baseline(args)
