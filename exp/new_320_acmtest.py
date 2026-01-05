# exp/new_320_acmtest.py
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 这里要改：用 24->24 的 wrapper
from data_provider.data_loader_acm_320 import FlightDataset_acm, Dataset_Forecast24to24_From96
from models.timer_xl import Model as TimerXL


# =========================================================
# 0) TimerXL 配置 & logger
# =========================================================
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
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=args.tmax, eta_min=1e-8
    ) if args.cosine else None
    criterion = nn.MSELoss()
    return model, optim, sched, criterion


# =========================================================
# 1) train/val split：只在 train_normal 的窗口池里 split
#    注意：base_all 的 seq_len=96，但 full_ds 是 24->24 的样本集合
# =========================================================
def _build_loaders(args, logger):
    base_all = FlightDataset_acm(args, Tag="train_normal", side=args.side)

    # 关键：24->24 切片 wrapper
    full_ds = Dataset_Forecast24to24_From96(
        base_all,
        in_len=args.in_len,
        out_len=args.out_len,
        stride=args.stride
    )

    n_total = len(full_ds)
    if n_total == 0:
        raise RuntimeError("full_ds is empty, please check IoTDB query or filters.")

    val_ratio = getattr(args, "val_ratio", 0.1)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    if n_train <= 0:
        raise RuntimeError(f"Not enough samples to split, total={n_total}, val_ratio={val_ratio}")

    split_seed = getattr(args, "split_seed", 42)
    g = torch.Generator().manual_seed(split_seed)

    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 打印你到底是不是 24->24
    logger.info("[Config] base_seq_len=%d | in_len=%d -> out_len=%d | stride=%d",
                int(args.seq_len), int(args.in_len), int(args.out_len), int(args.stride))
    logger.info("Total train_normal base windows = %d", len(base_all))
    logger.info("Total 24->24 samples (after slicing) = %d", len(full_ds))
    logger.info("Train samples = %d | Val samples = %d", len(train_ds), len(val_ds))
    logger.info("Feature names = %s", getattr(base_all, "feature_names", None))

    return base_all, full_ds, train_ds, val_ds, train_loader, val_loader


# =========================================================
# 2) eval：按 24->24 计算 MSE/MAE
# =========================================================
@torch.no_grad()
def _eval_on_subset_24to24(args, logger, ckpt_path, subset, tag: str):
    device = args.gpu
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    model, _, _, _ = _build_model_and_optim(args, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    mse_list, mae_list = [], []

    for batch_x, batch_y, _ in loader:
        # batch_x: [B,1,24,6]  batch_y:[B,1,24,1]
        _, _, L, C = batch_x.shape
        x = batch_x.to(device).reshape(-1, L, C)      # [B,24,6]
        y = batch_y.to(device).reshape(-1, L, 1)      # [B,24,1]

        out_all = model(x)                            # 预期输出含时间维 L
        pred = out_all[:, -1, :].unsqueeze(-1)        # [B,24,1] 兼容你原来的写法

        p = pred.detach().cpu().numpy().astype(np.float32)  # [B,24,1]
        t = y.detach().cpu().numpy().astype(np.float32)     # [B,24,1]

        mse_list.extend(((p - t) ** 2).mean(axis=(1, 2)).tolist())
        mae_list.extend(np.abs(p - t).mean(axis=(1, 2)).tolist())

    if len(mse_list) == 0:
        logger.warning("%s has no samples, skip.", tag)
        return None

    avg_mse = float(np.mean(mse_list))
    avg_mae = float(np.mean(mae_list))
    logger.info("[%s] Avg MSE=%.6f | Avg MAE=%.6f | n=%d", tag, avg_mse, avg_mae, len(mse_list))
    return {"tag": tag, "mse": avg_mse, "mae": avg_mae, "n": int(len(mse_list))}


# =========================================================
# 3) train：保存 epoch 级记录 + 曲线图
# =========================================================
def _save_train_curves(save_dir: str, df: pd.DataFrame, setting: str):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "train_curve.csv")
    df.to_csv(csv_path, index=False)

    # loss curve png
    plt.figure()
    plt.plot(df["epoch"].values, df["train_loss"].values, label="train_loss")
    plt.plot(df["epoch"].values, df["val_loss"].values, label="val_loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(setting)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    png_path = os.path.join(save_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def train_timerxl_forecast24to24(args):
    logger = setup_logger(args.setting)
    device = args.gpu

    base_all, full_ds, train_ds, val_ds, train_loader, val_loader = _build_loaders(args, logger)

    model, optim, sched, criterion = _build_model_and_optim(args, device)

    save_dir = os.path.join(args.checkpoints, args.setting)
    os.makedirs(save_dir, exist_ok=True)

    best_val = float("inf")
    best_path = os.path.join(save_dir, "best_timerxl_24to24.pth")
    final_path = os.path.join(save_dir, "final_timerxl_24to24.pth")

    curve_rows = []

    for ep in range(1, args.train_epochs + 1):
        model.train()
        t0 = time.time()
        tr_sum, nsamp = 0.0, 0

        for batch_x, batch_y, _ in train_loader:
            _, _, L, C = batch_x.shape  # L=24 C=6
            x = batch_x.to(device).reshape(-1, L, C)  # [B,24,6]
            y = batch_y.to(device).reshape(-1, L, 1)  # [B,24,1]

            out_all = model(x)
            pred = out_all[:, -1, :].unsqueeze(-1)     # [B,24,1]
            loss = criterion(pred, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            tr_sum += loss.item() * x.size(0)
            nsamp += x.size(0)

        train_loss = tr_sum / max(1, nsamp)

        model.eval()
        val_sum, vsamp = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                _, _, L, C = batch_x.shape
                x = batch_x.to(device).reshape(-1, L, C)
                y = batch_y.to(device).reshape(-1, L, 1)

                out_all = model(x)
                pred = out_all[:, -1, :].unsqueeze(-1)
                loss = criterion(pred, y)

                val_sum += loss.item() * x.size(0)
                vsamp += x.size(0)

        val_loss = val_sum / max(1, vsamp)
        dt = time.time() - t0
        lr_now = float(optim.param_groups[0]["lr"])

        logger.info(
            "Epoch %d/%d | train=%.6f | val=%.6f | lr=%.2e | %.1fs",
            ep, args.train_epochs, train_loss, val_loss, lr_now, dt
        )

        curve_rows.append({
            "epoch": ep,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": lr_now,
            "sec": float(dt),
            "n_train": int(len(train_ds)),
            "n_val": int(len(val_ds)),
        })

        # save curve each epoch (方便中途挂了也有记录)
        curve_df = pd.DataFrame(curve_rows)
        _save_train_curves(save_dir, curve_df, args.setting)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info("New best model saved -> %s", best_path)

        if sched is not None:
            sched.step()

    torch.save(model.state_dict(), final_path)
    logger.info("Training finished. Best val=%.6f | best_path=%s", best_val, best_path)
    logger.info("Final model saved -> %s", final_path)


# =========================================================
# 4) evaluate：test_normal_recent / test_abnormal
# =========================================================
def evaluate_timerxl_forecast24to24(args, ckpt_path):
    logger = setup_logger(args.setting + "_eval")

    logger.info("Evaluating on test_normal_recent ...")
    base_nr = FlightDataset_acm(args, Tag="test_normal_recent", side=args.side)
    if len(base_nr) > 0:
        ds_nr = Dataset_Forecast24to24_From96(base_nr, in_len=args.in_len, out_len=args.out_len, stride=args.stride)
        _eval_on_subset_24to24(args, logger, ckpt_path, ds_nr, tag="test_normal_recent")
    else:
        logger.warning("No test_normal_recent data found, skip.")

    logger.info("Evaluating on test_abnormal ...")
    base_abn = FlightDataset_acm(args, Tag="test_abnormal", side=args.side)
    if len(base_abn) > 0:
        ds_abn = Dataset_Forecast24to24_From96(base_abn, in_len=args.in_len, out_len=args.out_len, stride=args.stride)
        _eval_on_subset_24to24(args, logger, ckpt_path, ds_abn, tag="test_abnormal")
    else:
        logger.warning("No abnormal data found, skip.")


# =========================================================
# 5) Main：自动跑 PACK1 / PACK2
# =========================================================
if __name__ == "__main__":

    class Args:
        pass

    base_args = Args()

    # base window（用于缓存）：仍然用 96
    base_args.seq_len = 96

    # 24->24 的训练切片参数（用于确认是否用了 24->24）
    base_args.in_len = 24
    base_args.out_len = 24
    base_args.stride = 24

    base_args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TimerXL
    base_args.input_token_len = 24
    base_args.d_model = 128
    base_args.nhead = 4
    base_args.num_layers = 3
    base_args.dim_ff = 256
    base_args.dropout = 0.1

    # Train
    base_args.batch_size = 64
    base_args.num_workers = 4
    base_args.learning_rate = 3e-4
    base_args.weight_decay = 1e-4
    base_args.tmax = 20
    base_args.train_epochs = 30
    base_args.cosine = True

    # Split
    base_args.val_ratio = 0.1
    base_args.split_seed = 42

    # 每个航段取多少个 base window（
    base_args.max_windows_per_flight = 5

    # Paths
    base_args.checkpoints = "./checkpoints"

    # 时间切分（你的规则）
    base_args.normal_months = 10
    base_args.test_normal_months = 1
    base_args.fault_gap_months = 6
    base_args.normal_anchor_end = "2025-08-01"

    # raw 两年缓存参数
    base_args.raw_months = 12
    base_args.raw_end_use_gap = False  # True: raw_end = fd-gap_months；False: raw_end=fd

    # raw verbose（想看更细 print 就打开）
    base_args.verbose_raw =  True  
    base_args.verbose_every_n_param = 1
    base_args.verbose_flush = True

    # debug 航段可视化（可选）
    base_args.debug_plot_tail = ""
    base_args.debug_plot_mode = "train_normal"
    base_args.debug_plot_n_segments = 3
    base_args.debug_plot_steps = 96 * 5

    base_setting = (
        f"timerxl_forecast_in{base_args.in_len}_out{base_args.out_len}_stride{base_args.stride}_"
        f"raw{base_args.raw_months}m_train{base_args.normal_months}m_test{base_args.test_normal_months}m_"
        f"gap{base_args.fault_gap_months}m_{base_args.normal_anchor_end}end_noALTSTD"
    )

    for side in ["PACK2", "PACK1"]:
        args = Args()
        args.__dict__.update(base_args.__dict__)
        args.side = side
        args.setting = f"{base_setting}_{side}"

        best_ckpt = os.path.join(args.checkpoints, args.setting, "best_timerxl_24to24.pth")

        print("\n==============================")
        print(
            f"Running side={side} | setting={args.setting} | "
            f"base_seq_len={args.seq_len} | in={args.in_len} out={args.out_len} stride={args.stride} | "
            f"rawM={args.raw_months} trainM={args.normal_months} testM={args.test_normal_months} gapM={args.fault_gap_months} "
            f"anchor_end(no-fault)={args.normal_anchor_end} raw_end_use_gap={args.raw_end_use_gap} | "
            f"max_windows_per_flight={args.max_windows_per_flight}"
        )
        print("==============================")

        if os.path.exists(best_ckpt):
            print("Found existing checkpoint -> evaluating")
            evaluate_timerxl_forecast24to24(args, ckpt_path=best_ckpt)
        else:
            print("No checkpoint found -> training")
            train_timerxl_forecast24to24(args)
            if os.path.exists(best_ckpt):
                evaluate_timerxl_forecast24to24(args, ckpt_path=best_ckpt)
