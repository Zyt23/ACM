# -*- coding: utf-8 -*-
import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_loader_acm_320 import FlightDataset_acm
from data_provider.acm_window_dataset import SegHeadWindowDataset
from models.timer_xl import Model as TimerXL
from models.acm_classifier import TinyTransformerClassifier

def collate_keep_meta(batch):
    wins, metas = zip(*batch)

    # wins: may be np.ndarray or torch.Tensor
    win0 = wins[0]
    if isinstance(win0, np.ndarray):
        wins = torch.from_numpy(np.stack(wins, axis=0)).float()   # [B,L,6]
    else:
        wins = torch.stack(wins, dim=0).float()

    return wins, list(metas)
# ---------------- safe torch.load ----------------
def safe_torch_load(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# ---------------- TimerXL configs ----------------
class TimerXLConfigs:
    def __init__(self, args):
        self.input_token_len = args.input_token_len
        self.d_model = args.d_model
        self.n_heads = args.nhead
        self.e_layers = args.num_layers
        self.d_ff = args.dim_ff
        self.dropout = args.dropout
        self.activation = "gelu"
        self.output_attention = False
        self.covariate = False
        self.flash_attention = False
        self.use_norm = False


def build_timerxl_regressor(args, device):
    cfg = TimerXLConfigs(args)
    return TimerXL(cfg).to(device)


@torch.no_grad()
def timerxl_predict_sequence(reg_model: nn.Module, x_cov: torch.Tensor) -> torch.Tensor:
    out_all = reg_model(x_cov)
    pred = out_all[:, -1, :].unsqueeze(-1)  # [B,L,1]
    return pred


def build_classifier(args, in_dim, num_classes, device):
    return TinyTransformerClassifier(
        in_dim=in_dim,
        d_model=args.cls_d_model,
        nhead=args.cls_nhead,
        num_layers=args.cls_layers,
        dim_ff=args.cls_ff,
        dropout=args.cls_dropout,
        num_classes=num_classes,
        pooling=args.cls_pooling,
        max_len=max(2048, args.win_len + 64),
    ).to(device)


def hist_intersection_threshold(a: np.ndarray, b: np.ndarray, bins: int = 80) -> float:
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float(np.median(a))

    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    diff = ha - hb
    for i in range(1, len(diff)):
        if diff[i - 1] == 0:
            return float(centers[i - 1])
        if diff[i - 1] * diff[i] < 0:
            return float(centers[i])
    j = int(np.argmin(np.abs(diff)))
    return float(centers[j])


def run_eval(tag: str, base_args, reg, clf, device, out_dir: str, p_thr: float):
    base_ds = FlightDataset_acm(base_args, Tag=tag, side=base_args.side)
    if len(base_ds) == 0:
        print(f"[{tag}] dataset empty, skip.")
        return None

    win_ds = SegHeadWindowDataset(base_ds, win_len=base_args.win_len, stride=base_args.stride)
    loader = DataLoader(
        win_ds,
        batch_size=base_args.batch_size,
        shuffle=False,
        num_workers=base_args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(base_args.num_workers > 0),
        collate_fn=collate_keep_meta,  # ✅ 必须
    )

    idx_x = [0, 1, 2, 3, 4]
    idx_y = 5

    rows = []
    clf.eval()
    reg.eval()

    with torch.no_grad():
        for win, meta in tqdm(loader, desc=f"[{tag}] eval"):
            win = win.to(device)
            cov = win[:, :, idx_x]
            y_true = win[:, :, idx_y:idx_y+1]
            y_pred = timerxl_predict_sequence(reg, cov)
            resid = y_true - y_pred

            mse = (resid ** 2).mean(dim=(1, 2)).detach().cpu().numpy()

            x_clf = torch.cat([cov, y_true, y_pred, resid], dim=-1)  # [B,L,8]
            logits = clf(x_clf)  # [B,2]
            prob = torch.softmax(logits, dim=1).detach().cpu().numpy()
            p_abn = prob[:, 1]   # ✅ 二分类：异常概率
            pred_cls = prob.argmax(axis=1)

            for i in range(win.size(0)):
                # meta 兼容两种结构：dataclass 或 dict
                m = meta[i]
                tail = m.tail if hasattr(m, "tail") else m["tail"]
                seg_start_time = m.seg_start_time if hasattr(m, "seg_start_time") else m["seg_start_time"]
                seg_index = int(m.seg_index) if hasattr(m, "seg_index") else int(m["seg_index"])
                step_offset = int(m.step_offset) if hasattr(m, "step_offset") else int(m["step_offset"])

                rows.append({
                    "tag": tag,
                    "tail": str(tail),
                    "seg_start_time": str(seg_start_time),
                    "seg_index": seg_index,
                    "step_offset": step_offset,
                    "mse": float(mse[i]),
                    "p_abn": float(p_abn[i]),
                    "pred_class": int(pred_cls[i]),
                    "is_alarm": int(p_abn[i] >= p_thr),
                })

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"{tag}_per_window.csv"), index=False)

    # per-flight (seghead) aggregation
    g = df.groupby(["tail", "seg_index", "seg_start_time"], as_index=False).agg(
        n_windows=("p_abn", "count"),
        alarm_any=("is_alarm", "max"),
        alarm_rate=("is_alarm", "mean"),
        p_abn_mean=("p_abn", "mean"),
        p_abn_max=("p_abn", "max"),
        mse_mean=("mse", "mean"),
        mse_max=("mse", "max"),
    )
    g.to_csv(os.path.join(out_dir, f"{tag}_per_flight.csv"), index=False)

    # per-tail aggregation
    t = g.groupby(["tail"], as_index=False).agg(
        n_flights=("p_abn_mean", "count"),
        alarm_any_rate=("alarm_any", "mean"),  # 有多少航段至少报过一次
        alarm_rate=("alarm_rate", "mean"),
        p_abn_mean=("p_abn_mean", "mean"),
        p_abn_max=("p_abn_max", "max"),
        mse_mean=("mse_mean", "mean"),
        mse_max=("mse_max", "max"),
    )
    t.to_csv(os.path.join(out_dir, f"{tag}_per_tail.csv"), index=False)

    # quick print
    fpr_or_tpr = df["is_alarm"].mean()
    print(f"[{tag}] windows={len(df)} | alarm_rate={fpr_or_tpr:.6f} (using p_thr={p_thr:.6f})")

    return df


def main():
    class Args: pass
    args = Args()

    # ----- must match training -----
    args.side = "PACK2"
    args.win_len = 96
    args.stride = 96
    args.input_token_len = 96

    # data split config (same as your loader)
    args.seq_len = 96
    args.max_windows_per_flight = 5
    args.normal_months = 10
    args.test_normal_months = 1
    args.fault_gap_months = 6
    args.normal_anchor_end = "2025-08-01"
    args.raw_months = 12
    args.raw_end_use_gap = False

    # runtime
    args.batch_size = 512
    args.num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TimerXL regressor config
    args.d_model = 128
    args.nhead = 4
    args.num_layers = 3
    args.dim_ff = 256
    args.dropout = 0.1
    args.reg_ckpt = "./checkpoints/timerxl_aligned_regress_keep5x96_raw12m_train10m_test1m_gap6m_2025-08-01end_noALTSTD_win96_stride96_PACK2/best_timerxl_regress_win96.pth"

    # classifier config (BINARY!)
    args.cls_d_model = 128
    args.cls_nhead = 4
    args.cls_layers = 3
    args.cls_ff = 256
    args.cls_dropout = 0.1
    args.cls_pooling = "mean"
    args.cls_ckpt = "./checkpoints/mech_cls_BIN_PACK2_win96_stride96_panom0.5/best_mech_cls_bin.pth"
    args.thr_json = "./checkpoints/mech_cls_BIN_PACK2_win96_stride96_panom0.5/thresholds_bin.json"

    # load threshold
    if not os.path.exists(args.thr_json):
        raise FileNotFoundError(f"thr_json not found: {args.thr_json}")
    thr = json.load(open(args.thr_json, "r", encoding="utf-8"))
    p_thr = float(thr.get("p_abn_thr", 0.5))
    print(f"[Loaded threshold] p_abn_thr={p_thr:.6f}")

    # build models
    reg = build_timerxl_regressor(args, device)
    reg.load_state_dict(safe_torch_load(args.reg_ckpt, device))
    reg.eval()

    clf = build_classifier(args, in_dim=8, num_classes=2, device=device)  # ✅ 二分类
    clf.load_state_dict(safe_torch_load(args.cls_ckpt, device))
    clf.eval()

    out_dir = os.path.join("./checkpoints", "eval_mech_cls_BIN", args.side)
    os.makedirs(out_dir, exist_ok=True)

    df_n = run_eval("test_normal_recent", args, reg, clf, device, out_dir, p_thr=p_thr)
    df_a = run_eval("test_abnormal", args, reg, clf, device, out_dir, p_thr=p_thr)

    if df_n is not None and df_a is not None and len(df_n) > 0 and len(df_a) > 0:
        thr_p = hist_intersection_threshold(df_n["p_abn"].values, df_a["p_abn"].values)
        thr_m = hist_intersection_threshold(df_n["mse"].values, df_a["mse"].values)
        print(f"[Intersection] p_abn_thr~{thr_p:.6f} | mse_thr~{thr_m:.6f}")
        print(f"[Calibrated ] p_abn_thr={p_thr:.6f}")

        # also print basic TPR/FPR under calibrated threshold
        fpr = float((df_n["p_abn"].values >= p_thr).mean())
        tpr = float((df_a["p_abn"].values >= p_thr).mean())
        print(f"[Under calibrated thr] FPR(normal)={fpr:.6f} | TPR(abnormal)={tpr:.6f}")


if __name__ == "__main__":
    main()
