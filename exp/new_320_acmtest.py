# exp/new_320_acm_regress24to24.py
# -*- coding: utf-8 -*-
import os, time, sys, logging
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

# base dataset + aligned regression wrapper (must exist in data_loader_acm_320.py)
from data_provider.data_loader_acm_320 import (
    FlightDataset_acm,
    Dataset_Aligned24to24_Regress_FromSegHead,
)

from models.timer_xl import Model as TimerXL


# =========================================================
# 0) TimerXL configs & logger
# =========================================================
class TimerXLConfigs:
    def __init__(self, args):
        self.input_token_len = args.input_token_len  # should be 24
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
    # torch>=2.0 supports weights_only; use it when possible to avoid pickle warning
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# =========================================================
# 1) loaders: build aligned regression dataset
# =========================================================
def _build_loaders(args, logger):
    base_all = FlightDataset_acm(args, Tag="train_normal", side=args.side)

    full_ds = Dataset_Aligned24to24_Regress_FromSegHead(
        base_all,
        win_len=args.win_len,     # 24
        stride=args.stride,       # 24 or 1
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
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    base_len = int(base_all.data.shape[1]) if len(base_all) > 0 else -1
    logger.info("[AlignedReg24to24] win_len=%d | stride=%d | input_token_len=%d",
                int(args.win_len), int(args.stride), int(args.input_token_len))
    logger.info("Total train_normal segheads=%d | base_len=%d", len(base_all), base_len)
    logger.info("Total windows=%d | Train=%d | Val=%d", len(full_ds), len(train_ds), len(val_ds))
    logger.info("Feature names=%s", getattr(base_all, "feature_names", None))

    return base_all, full_ds, train_ds, val_ds, train_loader, val_loader


# =========================================================
# 2) training
# =========================================================
def train_timerxl_regress24to24(args):
    logger = setup_logger(args.setting)
    device = args.gpu

    base_all, full_ds, train_ds, val_ds, train_loader, val_loader = _build_loaders(args, logger)
    model, optim, sched, criterion = _build_model_and_optim(args, device)

    save_dir = os.path.join(args.checkpoints, args.setting)
    os.makedirs(save_dir, exist_ok=True)

    best_val = float("inf")
    best_path = os.path.join(save_dir, "best_timerxl_regress24to24.pth")
    final_path = os.path.join(save_dir, "final_timerxl_regress24to24.pth")

    logger.info("========== TRAIN START ==========")
    logger.info("device=%s | batch=%d | epochs=%d", str(device), int(args.batch_size), int(args.train_epochs))
    logger.info("train_ds=%d | val_ds=%d", len(train_ds), len(val_ds))

    # optional: a single shape print
    printed_shape = False

    for ep in range(1, int(args.train_epochs) + 1):
        model.train()
        t0 = time.time()
        tr_sum, tr_n = 0.0, 0

        for batch_x, batch_y, _ in train_loader:
            # batch_x: [B,1,24,5]  batch_y: [B,1,24,1]
            _, _, L, C = batch_x.shape
            x = batch_x.to(device).reshape(-1, L, C)     # [B,24,5]
            y = batch_y.to(device).reshape(-1, L, 1)     # [B,24,1]

            out_all = model(x)
            pred = out_all[:, -1, :].unsqueeze(-1)       # [B,24,1] (沿用你现有 TimerXL 的取法)

            if (not printed_shape) and getattr(args, "debug_print_shapes", True):
                logger.info("[Shape] x=%s y=%s out_all=%s pred=%s",
                            tuple(x.shape), tuple(y.shape), tuple(out_all.shape), tuple(pred.shape))
                printed_shape = True

            loss = criterion(pred, y)

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
                y = batch_y.to(device).reshape(-1, L, 1)

                out_all = model(x)
                pred = out_all[:, -1, :].unsqueeze(-1)
                loss = criterion(pred, y)

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
# 3) test: per-flight / per-tail evaluation (aligned regression)
# =========================================================
@torch.no_grad()
def evaluate_by_flight_and_tail_regress24to24(args, ckpt_path, tag: str):
    """
    aligned regression:
      X: [ABC(t..t+23)] -> Y: [D(t..t+23)]
    输出每航班、每飞机的 MSE
    """
    logger = setup_logger(args.setting + f"_{tag}_regress24to24_eval")
    device = args.gpu

    logger.info("========== Regress24to24 flight eval (%s) ==========", tag)

    base_ds = FlightDataset_acm(args, Tag=tag, side=args.side)
    if len(base_ds) == 0:
        logger.warning("[%s] dataset empty, skip.", tag)
        return None, None

    # build model
    model, _, _, _ = _build_model_and_optim(args, device)
    state = _safe_torch_load(ckpt_path, device)
    model.load_state_dict(state)
    model.eval()

    win_len = int(args.win_len)
    stride = int(args.stride)

    feature_names = getattr(base_ds, "feature_names", [])
    if not feature_names:
        raise RuntimeError("base_ds.feature_names is empty.")
    n2i = {n: i for i, n in enumerate(feature_names)}

    # D = PACKx_DISCH_T (target)
    if args.side == "PACK1":
        target_name = "PACK1_DISCH_T"
        all_names = [
            "PACK1_BYPASS_V", "PACK1_DISCH_T", "PACK1_RAM_I_DR",
            "PACK1_RAM_O_DR", "PACK_FLOW_R1", "PACK1_COMPR_T",
        ]
    else:
        target_name = "PACK2_DISCH_T"
        all_names = [
            "PACK2_BYPASS_V", "PACK2_DISCH_T", "PACK2_RAM_I_DR",
            "PACK2_RAM_O_DR", "PACK_FLOW_R2", "PACK2_COMPR_T",
        ]

    miss = [c for c in all_names if c not in n2i]
    if miss:
        raise RuntimeError(f"[{tag}] missing columns: {miss} | feature_names={feature_names}")

    idx_y = n2i[target_name]
    input_names = [c for c in all_names if c != target_name]  # mask D
    idx_x = [n2i[c] for c in input_names]

    flight_rows = []
    tail2losses = {}

    for i in tqdm(range(len(base_ds)), desc=f"[{tag}] regress24to24 flight eval"):
        seg = base_ds.data[i]  # [keep_len, D]
        tail = base_ds.window_tails[i] if hasattr(base_ds, "window_tails") else "UNKNOWN"
        seg_start = base_ds.window_start_times[i] if hasattr(base_ds, "window_start_times") else "UNKNOWN"

        keep_len = int(seg.shape[0])
        if keep_len < win_len:
            continue

        xs, ys = [], []
        for st in range(0, keep_len - win_len + 1, stride):
            xs.append(seg[st:st + win_len, idx_x])      # [24,5]
            ys.append(seg[st:st + win_len, idx_y])      # [24]

        if not xs:
            continue

        x_t = torch.from_numpy(np.stack(xs)).float().to(device)                 # [N,24,5]
        y_t = torch.from_numpy(np.stack(ys)).float().unsqueeze(-1).to(device)  # [N,24,1]

        out_all = model(x_t)
        pred = out_all[:, -1, :].unsqueeze(-1)  # [N,24,1]

        losses = ((pred - y_t) ** 2).mean(dim=(1, 2)).detach().cpu().numpy()  # [N]
        flight_mse = float(losses.mean())

        flight_rows.append({
            "tail": str(tail),
            "seg_start_time": str(seg_start),
            "flight_index": int(i),
            "n_windows": int(len(losses)),
            "flight_mse": flight_mse,
        })
        tail2losses.setdefault(str(tail), []).append(flight_mse)

        if i % 200 == 0:
            logger.info("[%s] processed flights %d / %d", tag, i, len(base_ds))

    # save
    out_dir = os.path.join(args.checkpoints, args.setting, "test_results_regress24to24", tag)
    os.makedirs(out_dir, exist_ok=True)

    df_flight = pd.DataFrame(flight_rows)
    if not df_flight.empty:
        df_flight = df_flight.sort_values(["tail", "seg_start_time"], ascending=True)

    df_tail = pd.DataFrame([
        {"tail": t, "n_flights": int(len(v)), "avg_flight_mse": float(np.mean(v))}
        for t, v in tail2losses.items()
    ])
    if not df_tail.empty:
        df_tail = df_tail.sort_values(["avg_flight_mse"], ascending=True)

    flight_csv = os.path.join(out_dir, f"{tag}_per_flight.csv")
    tail_csv = os.path.join(out_dir, f"{tag}_per_tail.csv")
    df_flight.to_csv(flight_csv, index=False)
    df_tail.to_csv(tail_csv, index=False)

    logger.info("[%s] saved per-flight -> %s", tag, flight_csv)
    logger.info("[%s] saved per-tail   -> %s", tag, tail_csv)
    logger.info("[%s] flights=%d | tails=%d", tag, len(df_flight), len(df_tail))

    if not df_flight.empty:
        logger.info("[%s] overall mean flight_mse=%.6f", tag, float(df_flight["flight_mse"].mean()))
    if not df_tail.empty:
        logger.info("[%s] overall mean tail avg_flight_mse=%.6f", tag, float(df_tail["avg_flight_mse"].mean()))

    logger.info("========== DONE (%s) ==========", tag)
    return df_flight, df_tail


# =========================================================
# 4) main
# =========================================================
if __name__ == "__main__":

    class Args:
        pass

    base_args = Args()

    # seghead base window (keep_len = max_windows_per_flight * seq_len)
    base_args.seq_len = 96
    base_args.max_windows_per_flight = 5

    # aligned regression window
    base_args.win_len = 24
    base_args.stride = 24  # 想更密可改 1，但样本数会爆炸

    # device
    base_args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TimerXL
    base_args.input_token_len = 24  # 必须=win_len
    base_args.d_model = 128
    base_args.nhead = 4
    base_args.num_layers = 3
    base_args.dim_ff = 256
    base_args.dropout = 0.1

    # Train
    base_args.batch_size = 256
    base_args.num_workers = 4
    base_args.learning_rate = 3e-4
    base_args.weight_decay = 1e-4
    base_args.tmax = 20
    base_args.train_epochs = 30
    base_args.cosine = True

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

    base_setting = (
        f"timerxl_regress_win{base_args.win_len}_stride{base_args.stride}_"
        f"keep{base_args.max_windows_per_flight}x{base_args.seq_len}_"
        f"raw{base_args.raw_months}m_train{base_args.normal_months}m_test{base_args.test_normal_months}m_"
        f"gap{base_args.fault_gap_months}m_{base_args.normal_anchor_end}end_noALTSTD"
    )

    for side in ["PACK2"]:
        args = Args()
        args.__dict__.update(base_args.__dict__)
        args.side = side
        args.setting = f"{base_setting}_{side}"

        save_dir = os.path.join(args.checkpoints, args.setting)
        os.makedirs(save_dir, exist_ok=True)

        best_ckpt = os.path.join(save_dir, "best_timerxl_regress24to24.pth")

        print("\n==============================")
        print(
            f"Running side={side} | setting={args.setting} | "
            f"keep={args.max_windows_per_flight}x{args.seq_len} | "
            f"regress: win_len={args.win_len} stride={args.stride} | "
            f"rawM={args.raw_months} trainM={args.normal_months} testM={args.test_normal_months} gapM={args.fault_gap_months} "
            f"anchor_end={args.normal_anchor_end}"
        )
        print("==============================")

        # train if needed
        if not os.path.exists(best_ckpt):
            best_ckpt = train_timerxl_regress24to24(args)

        # evaluate
        if os.path.exists(best_ckpt):
            evaluate_by_flight_and_tail_regress24to24(args, ckpt_path=best_ckpt, tag="test_normal_recent")
            evaluate_by_flight_and_tail_regress24to24(args, ckpt_path=best_ckpt, tag="test_abnormal")
        else:
            print("[ERROR] best checkpoint not found:", best_ckpt)
