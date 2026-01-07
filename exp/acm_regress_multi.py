# exp/acm_regress_partialD_96_48to48.py
# -*- coding: utf-8 -*-
"""
Partial target history regression (96-step window):
  - covariates(5) for all 96 steps
  - target D = PACKx_COMPR_T for first 48 steps
  - target D masked for last 48 steps
Predict:
  - target D (PACKx_COMPR_T) for last 48 steps

Wrapper output:
  x: [B, 1, 96, 6(or7)]   (6 = 5 cov + 1 target; 7 if include_mask_channel)
  y: [B, 1, 48, 1]
Loss computed only on future 48 steps.

TimerXL usage kept consistent with your previous scripts:
  out_all = model(x_seq)
  pred_all = out_all[:, -1, :].unsqueeze(-1)
"""

import os
import time
import sys
import logging
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

# project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_loader_acm_320 import (
    FlightDataset_acm,
    Dataset_PartialTargetHistory_Regress_FromSegHead,
)

from models.timer_xl import Model as TimerXL


# =========================================================
# 0) TimerXL configs & logger
# =========================================================
class TimerXLConfigs:
    def __init__(self, args):
        self.input_token_len = args.input_token_len  # must == win_len (96)
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


def setup_logger(setting: str):
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/{setting}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(setting)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _build_model_and_optim(args, device):
    cfg = TimerXLConfigs(args)
    model = TimerXL(cfg).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.tmax, eta_min=1e-8)
        if args.cosine else None
    )
    criterion = nn.MSELoss()
    return model, optim, sched, criterion


def _safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _save_train_curves(save_dir: str, df: pd.DataFrame, title: str):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "train_curve.csv")
    df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(df["epoch"].values, df["train_loss"].values, label="train_loss")
    plt.plot(df["epoch"].values, df["val_loss"].values, label="val_loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    png_path = os.path.join(save_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


# =========================================================
# 1) loaders
# =========================================================
def _build_loaders(args, logger):
    base_all = FlightDataset_acm(args, Tag="train_normal", side=args.side)

    full_ds = Dataset_PartialTargetHistory_Regress_FromSegHead(
        base_dataset=base_all,
        win_len=args.win_len,       # 96
        hist_len=args.hist_len,     # 48
        fut_len=args.fut_len,       # 48
        stride=args.stride,         # usually 96
        mask_value=args.mask_value,
        include_mask_channel=args.include_mask_channel,
    )

    n_total = len(full_ds)
    if n_total == 0:
        raise RuntimeError("full_ds is empty, please check IoTDB query or filters.")

    val_ratio = float(getattr(args, "val_ratio", 0.1))
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    if n_train <= 0:
        raise RuntimeError(f"Not enough samples to split, total={n_total}, val_ratio={val_ratio}")

    split_seed = int(getattr(args, "split_seed", 42))
    g = torch.Generator().manual_seed(split_seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    logger.info("[PartialDReg] win_len=%d hist_len=%d fut_len=%d stride=%d input_token_len=%d",
                int(args.win_len), int(args.hist_len), int(args.fut_len), int(args.stride), int(args.input_token_len))
    logger.info("mask_value=%.3f include_mask_channel=%s", float(args.mask_value), str(args.include_mask_channel))
    logger.info("Total train_normal segheads=%d", len(base_all))
    logger.info("Total windows=%d | Train=%d | Val=%d", len(full_ds), len(train_ds), len(val_ds))
    logger.info("Feature names=%s", getattr(base_all, "feature_names", None))

    return base_all, full_ds, train_ds, val_ds, train_loader, val_loader


# =========================================================
# 2) training
# =========================================================
def train_timerxl_partialD_regress(args):
    logger = setup_logger(args.setting)
    device = args.gpu

    _, _, train_ds, val_ds, train_loader, val_loader = _build_loaders(args, logger)
    model, optim, sched, criterion = _build_model_and_optim(args, device)

    save_dir = os.path.join(args.checkpoints, args.setting)
    os.makedirs(save_dir, exist_ok=True)

    best_val = float("inf")
    best_path = os.path.join(save_dir, "best_timerxl_partialD_96_48to48.pth")
    final_path = os.path.join(save_dir, "final_timerxl_partialD_96_48to48.pth")

    logger.info("========== TRAIN START ==========")
    logger.info("device=%s | batch=%d | epochs=%d", str(device), int(args.batch_size), int(args.train_epochs))
    logger.info("train_ds=%d | val_ds=%d", len(train_ds), len(val_ds))

    curve_rows = []
    printed_shape = False

    for ep in range(1, int(args.train_epochs) + 1):
        model.train()
        t0 = time.time()
        tr_sum, tr_n = 0.0, 0

        for batch_x, batch_y, _ in train_loader:
            # batch_x: [B,1,96,6(or7)]  batch_y: [B,1,48,1]
            _, _, L, C = batch_x.shape
            x = batch_x.to(device).reshape(-1, L, C)              # [B,96,C]
            y = batch_y.to(device).reshape(-1, args.fut_len, 1)   # [B,48,1]

            out_all = model(x)
            pred_all = out_all[:, -1, :].unsqueeze(-1)            # expect [B,96,1]
            assert pred_all.shape[1] == args.win_len, f"pred_all time dim != win_len: {pred_all.shape}"

            pred_fut = pred_all[:, args.hist_len: args.hist_len + args.fut_len, :]  # [B,48,1]

            if (not printed_shape) and bool(getattr(args, "debug_print_shapes", True)):
                logger.info("[Shape] x=%s y=%s out_all=%s pred_all=%s pred_fut=%s",
                            tuple(x.shape), tuple(y.shape), tuple(out_all.shape),
                            tuple(pred_all.shape), tuple(pred_fut.shape))
                printed_shape = True

            loss = criterion(pred_fut, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            bsz = x.size(0)
            tr_sum += float(loss.item()) * bsz
            tr_n += bsz

        train_loss = tr_sum / max(1, tr_n)

        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                _, _, L, C = batch_x.shape
                x = batch_x.to(device).reshape(-1, L, C)
                y = batch_y.to(device).reshape(-1, args.fut_len, 1)

                out_all = model(x)
                pred_all = out_all[:, -1, :].unsqueeze(-1)
                assert pred_all.shape[1] == args.win_len, f"pred_all time dim != win_len: {pred_all.shape}"

                pred_fut = pred_all[:, args.hist_len: args.hist_len + args.fut_len, :]
                loss = criterion(pred_fut, y)

                bsz = x.size(0)
                va_sum += float(loss.item()) * bsz
                va_n += bsz

        val_loss = va_sum / max(1, va_n)

        if sched is not None:
            sched.step()

        dt = time.time() - t0
        lr_now = float(optim.param_groups[0]["lr"])
        logger.info("Epoch %d/%d | train=%.6f | val=%.6f | lr=%.2e | %.1fs",
                    ep, int(args.train_epochs), train_loss, val_loss, lr_now, dt)

        curve_rows.append({
            "epoch": int(ep),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(lr_now),
            "sec": float(dt),
            "n_train": int(len(train_ds)),
            "n_val": int(len(val_ds)),
        })
        _save_train_curves(save_dir, pd.DataFrame(curve_rows), args.setting)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info("New best -> %s (val=%.6f)", best_path, best_val)

    torch.save(model.state_dict(), final_path)
    logger.info("Training finished. best_val=%.6f", best_val)
    logger.info("Best saved:  %s", best_path)
    logger.info("Final saved: %s", final_path)
    return best_path


# =========================================================
# 3) evaluation: per-flight / per-tail (future-48 MSE)
# =========================================================
@torch.no_grad()
def evaluate_by_flight_and_tail_partialD(args, ckpt_path, tag: str):
    logger = setup_logger(args.setting + f"_{tag}_eval")
    device = args.gpu

    logger.info("========== PartialD future-48 eval (%s) ==========", tag)

    base_ds = FlightDataset_acm(args, Tag=tag, side=args.side)
    if len(base_ds) == 0:
        logger.warning("[%s] dataset empty, skip.", tag)
        return None, None

    model, _, _, _ = _build_model_and_optim(args, device)
    state = _safe_torch_load(ckpt_path, device)
    model.load_state_dict(state)
    model.eval()

    eval_ds = Dataset_PartialTargetHistory_Regress_FromSegHead(
        base_dataset=base_ds,
        win_len=args.win_len,
        hist_len=args.hist_len,
        fut_len=args.fut_len,
        stride=args.stride,
        mask_value=args.mask_value,
        include_mask_channel=args.include_mask_channel,
    )

    flight2losses = {}
    tail2losses = {}

    loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    for batch_x, batch_y, packed in tqdm(loader, desc=f"[{tag}] eval batches"):
        _, _, L, C = batch_x.shape
        x = batch_x.to(device).reshape(-1, L, C)              # [B,96,C]
        y = batch_y.to(device).reshape(-1, args.fut_len, 1)   # [B,48,1]

        out_all = model(x)
        pred_all = out_all[:, -1, :].unsqueeze(-1)            # [B,96,1]
        assert pred_all.shape[1] == args.win_len, f"pred_all time dim != win_len: {pred_all.shape}"

        pred_fut = pred_all[:, args.hist_len: args.hist_len + args.fut_len, :]  # [B,48,1]

        mse_each = ((pred_fut - y) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()
        packed = packed.detach().cpu().numpy().astype(int)

        for i in range(len(mse_each)):
            base_idx = int(packed[i] // 100)
            loss_i = float(mse_each[i])

            flight2losses.setdefault(base_idx, []).append(loss_i)

            tail = base_ds.window_tails[base_idx] if hasattr(base_ds, "window_tails") else "UNKNOWN"
            tail2losses.setdefault(str(tail), []).append(loss_i)

    flight_rows = []
    for base_idx, losses in flight2losses.items():
        tail = base_ds.window_tails[base_idx] if hasattr(base_ds, "window_tails") else "UNKNOWN"
        seg_start = base_ds.window_start_times[base_idx] if hasattr(base_ds, "window_start_times") else "UNKNOWN"
        flight_rows.append({
            "tail": str(tail),
            "seg_start_time": str(seg_start),
            "flight_index": int(base_idx),
            "n_windows": int(len(losses)),
            "flight_mse_future48": float(np.mean(losses)),
        })

    df_flight = pd.DataFrame(flight_rows)
    if not df_flight.empty:
        df_flight = df_flight.sort_values(["tail", "seg_start_time"], ascending=True)

    df_tail = pd.DataFrame([
        {"tail": t, "n_windows": int(len(v)), "avg_mse_future48": float(np.mean(v))}
        for t, v in tail2losses.items()
    ])
    if not df_tail.empty:
        df_tail = df_tail.sort_values(["avg_mse_future48"], ascending=True)

    out_dir = os.path.join(
        args.checkpoints,
        args.setting,
        "test_results_partialD_96_48to48",
        tag
    )
    os.makedirs(out_dir, exist_ok=True)

    flight_csv = os.path.join(out_dir, f"{tag}_per_flight.csv")
    tail_csv = os.path.join(out_dir, f"{tag}_per_tail.csv")
    df_flight.to_csv(flight_csv, index=False)
    df_tail.to_csv(tail_csv, index=False)

    logger.info("[%s] saved per-flight -> %s", tag, flight_csv)
    logger.info("[%s] saved per-tail   -> %s", tag, tail_csv)

    if not df_flight.empty:
        logger.info("[%s] overall mean flight_mse_future48=%.6f", tag, float(df_flight["flight_mse_future48"].mean()))
    if not df_tail.empty:
        logger.info("[%s] overall mean tail avg_mse_future48=%.6f", tag, float(df_tail["avg_mse_future48"].mean()))

    return df_flight, df_tail


# =========================================================
# 4) main
# =========================================================
if __name__ == "__main__":

    class Args:
        pass

    base_args = Args()

    # seghead base (keep_len = max_windows_per_flight * seq_len)
    base_args.seq_len = 96
    base_args.max_windows_per_flight = 10

    # task window
    base_args.win_len = 96
    base_args.hist_len = 48
    base_args.fut_len = 48
    base_args.stride = 96

    # masking
    base_args.mask_value = 0.0
    base_args.include_mask_channel = True  # 建议 True

    # device
    base_args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TimerXL
    base_args.input_token_len = 96
    base_args.d_model = 128
    base_args.nhead = 4
    base_args.num_layers = 3
    base_args.dim_ff = 256
    base_args.dropout = 0.1

    # Train
    base_args.batch_size = 256
    base_args.eval_batch_size = 256
    base_args.num_workers = 4
    base_args.learning_rate = 3e-4
    base_args.weight_decay = 1e-4
    base_args.tmax = 20
    base_args.train_epochs = 30
    base_args.cosine = True

    # Split
    base_args.val_ratio = 0.1
    base_args.split_seed = 42

    # Paths
    base_args.checkpoints = "./checkpoints"

    # time split (your rules)
    base_args.normal_months = 10
    base_args.test_normal_months = 1
    base_args.fault_gap_months = 6
    base_args.normal_anchor_end = "2025-08-01"

    # raw cache
    base_args.raw_months = 12
    base_args.raw_end_use_gap = False

    # verbose
    base_args.verbose_raw = True
    base_args.verbose_every_n_param = 1
    base_args.verbose_flush = True

    # debug prints
    base_args.debug_print_shapes = True

    mask_flag = 1 if base_args.include_mask_channel else 0
    base_setting = (
        f"timerxl_partialD96_hist48_fut48_"
        f"keep{base_args.max_windows_per_flight}x{base_args.seq_len}_"
        f"stride{base_args.stride}_mask{mask_flag}_"
        f"raw{base_args.raw_months}m_train{base_args.normal_months}m_test{base_args.test_normal_months}m_"
        f"gap{base_args.fault_gap_months}m_{base_args.normal_anchor_end}end_noALTSTD"
    )

    for side in ["PACK2", "PACK1"]:
        args = Args()
        args.__dict__.update(base_args.__dict__)
        args.side = side
        args.setting = f"{base_setting}_{side}"

        save_dir = os.path.join(args.checkpoints, args.setting)
        os.makedirs(save_dir, exist_ok=True)
        best_ckpt = os.path.join(save_dir, "best_timerxl_partialD_96_48to48.pth")

        print("\n==============================")
        print(
            f"Running side={side} | setting={args.setting}\n"
            f"  keep={args.max_windows_per_flight}x{args.seq_len}\n"
            f"  task: win_len={args.win_len}, hist_len={args.hist_len}, fut_len={args.fut_len}, stride={args.stride}\n"
            f"  include_mask_channel={args.include_mask_channel}, mask_value={args.mask_value}\n"
            f"  rawM={args.raw_months} trainM={args.normal_months} testM={args.test_normal_months} "
            f"gapM={args.fault_gap_months} anchor_end={args.normal_anchor_end}\n"
            f"  ckpt={best_ckpt}"
        )
        print("==============================")

        if not os.path.exists(best_ckpt):
            best_ckpt = train_timerxl_partialD_regress(args)

        if os.path.exists(best_ckpt):
            evaluate_by_flight_and_tail_partialD(args, ckpt_path=best_ckpt, tag="test_normal_recent")
            evaluate_by_flight_and_tail_partialD(args, ckpt_path=best_ckpt, tag="test_abnormal")
        else:
            print("[ERROR] best checkpoint not found:", best_ckpt)
