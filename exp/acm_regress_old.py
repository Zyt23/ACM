# exp/new_320_acmtest.py
# -*- coding: utf-8 -*-
"""
训练/验证：2个月（train_normal 时间段）
test_normal：紧接着的1个月（Tag=test_normal_recent）
abnormal：故障前1个月（Tag=test_abnormal）

对有故障飞机（按侧首次故障 fd）：
  train/val:   [fd-9M, fd-7M)  (trainM=2,testM=1,gapM=6)
  test_normal: [fd-7M, fd-6M)
  abnormal:    [fd-1M, fd)

对无故障飞机：
  anchor_end = args.normal_anchor_end (默认 2024-01-01)
  train/val:   [end-3M, end-1M)
  test_normal: [end-1M, end)
"""

import os, time, sys, logging
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ===== 可视化（保存 png）=====
import matplotlib
matplotlib.use("Agg")  # 服务器/无GUI环境也能保存图
import matplotlib.pyplot as plt

# ★★★ 先把项目根目录加进 sys.path ★★★
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.acm320_old import FlightDataset_acm, Dataset_RegRight_TimerXL
from models.timer_xl import Model as TimerXL


# =========================================================
# 0) 报警逻辑（滑窗均值 -> 均值差分 -> 连续上升段 -> mean_up）
# =========================================================
configs = {
    "length": 6,
    "end_point_idx": 100,
    "window_size": 35,
}

def calculate_mean_up(df: pd.DataFrame) -> pd.DataFrame:
    length_threshold = configs["length"]
    end_point_idx = configs["end_point_idx"]

    df["non_negative"] = df["mean_diff"] >= 0
    df["sequence_group"] = (df["non_negative"] != df["non_negative"].shift(1)).cumsum()
    sequence_lengths = df.groupby("sequence_group").size()

    for group_id, length in sequence_lengths.items():
        is_nonneg_segment = bool(df[df["sequence_group"] == group_id]["non_negative"].iloc[0])
        if length >= length_threshold and is_nonneg_segment:
            sequence_indices = df[df["sequence_group"] == group_id].index
            if len(sequence_indices) == 0:
                continue

            first_point_idx = sequence_indices[0]
            start_idx = max(0, first_point_idx - end_point_idx)
            search_means = df.loc[list(range(start_idx, first_point_idx + 1)), "mean"]
            if search_means.empty:
                continue

            min_mean = float(search_means.min())
            for idx in sequence_indices:
                current_mean = float(df.loc[idx, "mean"])
                df.loc[idx, "mean_up"] = (current_mean / min_mean) if min_mean != 0 else 1.0

    df.drop(columns=["non_negative", "sequence_group", "mean_diff"], inplace=True)
    return df


def find_warning_condition(df_max: pd.DataFrame) -> pd.DataFrame:
    df_max = df_max.copy()
    if "loss" not in df_max.columns:
        df_max["mean"] = np.nan
        df_max["mean_up"] = np.nan
        return df_max

    window_size = configs["window_size"]
    df_max["mean"] = 0.0
    df_max["mean_diff"] = np.nan
    df_max["mean_up"] = 0.0

    if len(df_max) >= window_size:
        df_max.loc[window_size - 1 :, "mean"] = np.nan
    df_max.loc[: min(window_size - 2, len(df_max) - 1), "mean"] = float(
        np.mean(df_max["loss"].iloc[: window_size - 1])
    )

    for i in range(len(df_max)):
        if i >= window_size - 1:
            window_data = df_max.iloc[i - window_size + 1 : i + 1]
            df_max.loc[df_max.index[i], "mean"] = float(np.mean(window_data["loss"]))

    df_max["mean_diff"] = df_max["mean"].diff()
    df_max = calculate_mean_up(df_max)
    return df_max


def alarm_from_loss_series(
    tail: str,
    loss_df: pd.DataFrame,
    save_dir: str,
    tag: str,
    threshold: float = 0.5,
    keep_last_n: int = 300,
):
    os.makedirs(save_dir, exist_ok=True)

    df = loss_df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    hist_path = os.path.join(save_dir, f"loss_{tag}_{tail}.csv")
    if os.path.exists(hist_path):
        old = pd.read_csv(hist_path)
        if len(old) > 0:
            old["time"] = pd.to_datetime(old["time"], errors="coerce")
            old = old.dropna(subset=["time"])
    else:
        old = pd.DataFrame(columns=["time", "loss"])

    new_loss = pd.concat([old, df[["time", "loss"]]], ignore_index=True)
    new_loss = new_loss.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    new_loss = new_loss.tail(keep_last_n).reset_index(drop=True)
    new_loss.to_csv(hist_path, index=False)

    if len(new_loss) <= 40:
        return None, False, None, new_loss

    out = find_warning_condition(new_loss.rename(columns={"loss": "loss"}))
    if "mean_up" not in out.columns:
        return None, False, None, out

    final_mean_up = float(out["mean_up"].iloc[-1])
    is_alarm = final_mean_up > threshold

    first_alarm_time = None
    if is_alarm:
        hits = out[out["mean_up"] > threshold]
        if len(hits) > 0:
            first_alarm_time = pd.to_datetime(hits["time"].iloc[0], errors="coerce")

    return final_mean_up, is_alarm, first_alarm_time, out


# =========================================================
# 0.5) 可视化：每架飞机选前 N 个 window，把 96 点 pred/true 画出来
# =========================================================
def _plot_first_k_windows_96pt(df_tail_windows: pd.DataFrame,
                              seq_npz_path: str,
                              out_dir: str,
                              tag: str,
                              tail: str,
                              k: int = 5):
    """
    df_tail_windows 必须包含: time, window_idx
    seq_npz_path: 该 tag 对应的 npz（里面有 window_idx, pred, true）
    """
    try:
        npz = np.load(seq_npz_path)
    except Exception:
        return

    widx_all = npz["window_idx"].astype(np.int64)  # [N]
    pred_all = npz["pred"]  # [N, 96]
    true_all = npz["true"]  # [N, 96]

    idx_map = {int(w): i for i, w in enumerate(widx_all)}

    d = df_tail_windows.dropna(subset=["window_idx"]).sort_values("time").reset_index(drop=True)
    if len(d) == 0:
        return

    os.makedirs(out_dir, exist_ok=True)

    take = d.head(k)
    for j, row in take.iterrows():
        wid = int(row["window_idx"])
        if wid not in idx_map:
            continue
        rid = idx_map[wid]
        p = pred_all[rid]
        t = true_all[rid]

        plt.figure(figsize=(10, 4))
        plt.plot(t, label="true")
        plt.plot(p, label="pred")
        plt.grid(True, alpha=0.3)
        t0 = row.get("time", "")
        plt.title(f"{tag} | {tail} | window#{j+1} | start={t0} | window_idx={wid}")
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"{tag}_{tail}_win{j+1}_idx{wid}.png")
        plt.savefig(out_png, dpi=150)
        plt.close()


def _plot_residual_hist(df_all: pd.DataFrame, out_png: str, title: str = ""):
    """
    df_all 必须包含: pred_last, true_last
    """
    d = df_all.dropna(subset=["pred_last", "true_last"])
    if len(d) == 0:
        return

    resid = (d["pred_last"] - d["true_last"]).astype(float)

    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=60)
    plt.grid(True, alpha=0.3)
    plt.xlabel("pred_last - true_last")
    plt.ylabel("count")
    plt.title(title or "Residual histogram (last point)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# =========================================================
# 1) fault side csv：用于给每个 tail 关联“本侧故障日期”
# =========================================================
def load_fault_map_for_side(side_pack: str) -> dict:
    csv_path = os.path.join("data_provider", "320_ACM_faults_side.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] fault_side csv not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    need = {"机号", "首发日期", "pack"}
    if not need.issubset(set(df.columns)):
        print(f"[WARN] fault_side csv columns mismatch. Need={need}, Got={set(df.columns)}")
        return {}

    df["首发日期"] = pd.to_datetime(df["首发日期"], errors="coerce")
    df = df.dropna(subset=["机号", "首发日期", "pack"])
    df["pack"] = pd.to_numeric(df["pack"], errors="coerce").astype("Int64")

    pack_id = 1 if side_pack == "PACK1" else 2
    df = df[df["pack"] == pack_id].copy()
    if len(df) == 0:
        return {}

    return df.groupby("机号")["首发日期"].min().to_dict()


# =========================================================
# 2) TimerXL 配置 & logger
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
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.tmax, eta_min=1e-8) if args.cosine else None
    criterion = nn.MSELoss()
    return model, optim, sched, criterion


# =========================================================
# 3) train/val split：只在 train_normal 这2个月窗口里 split
# =========================================================
def _build_loaders(args):
    base_all = FlightDataset_acm(args, Tag="train_normal", side=args.side)
    full_ds = Dataset_RegRight_TimerXL(base_all)

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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    return base_all, full_ds, train_ds, val_ds, train_loader, val_loader


# =========================================================
# 4) 核心 eval：
#    - 保存 window-level metrics（mse/mae 等）
#    - 保存每个 window 的 96 点 pred/true 序列到 npz（不聚合）
#    - 每架飞机仅画前 5 个 window 的 96 点曲线图
# =========================================================
@torch.no_grad()
def _eval_on_subset(args, logger, ckpt_path, base_dataset, subset, tag: str):
    device = args.gpu
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model, _, _, _ = _build_model_and_optim(args, device)
    # 安全建议：如果你完全信任 ckpt，可以保持默认；否则建议 weights_only=True（新版本pytorch支持）
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    mse_list, mae_list, idx_list = [], [], []
    pred_last_list, true_last_list = [], []

    pred_seq_chunks, true_seq_chunks = [], []

    for batch_x, batch_y, batch_idx in loader:
        _, _, L, C = batch_x.shape
        x = batch_x.to(device).reshape(-1, L, C)
        y = batch_y.to(device).reshape(-1, L, 1)

        out_all = model(x)                       # [B, C_out, L]
        pred = out_all[:, -1, :].unsqueeze(-1)   # [B, L, 1]

        p = pred.squeeze(-1).detach().cpu().numpy().astype(np.float32)  # [B, 96]
        t = y.squeeze(-1).detach().cpu().numpy().astype(np.float32)     # [B, 96]

        idx_list.append(batch_idx.view(-1).cpu().numpy())

        # window-level metrics（基于 96 点序列）
        mse_list.extend(((p - t) ** 2).mean(axis=1).tolist())
        mae_list.extend(np.abs(p - t).mean(axis=1).tolist())

        # 快速标量：最后一个时间点（可改成均值等）
        pred_last_list.extend(p[:, -1].tolist())
        true_last_list.extend(t[:, -1].tolist())

        # 关键：保存每个 window 的 96 点原始序列
        pred_seq_chunks.append(p)
        true_seq_chunks.append(t)

    if len(idx_list) == 0:
        logger.warning("%s has no samples, skip.", tag)
        return

    widx = np.concatenate(idx_list, axis=0).astype(np.int64)

    pred_seq = np.concatenate(pred_seq_chunks, axis=0)  # [N, 96]
    true_seq = np.concatenate(true_seq_chunks, axis=0)  # [N, 96]

    window_times = getattr(base_dataset, "window_start_times", None)
    window_tails = getattr(base_dataset, "window_tails", None)

    metrics_df = pd.DataFrame({
        "window_idx": widx,
        "mse": np.array(mse_list, dtype=float),
        "mae": np.array(mae_list, dtype=float),
        "pred_last": np.array(pred_last_list, dtype=float),
        "true_last": np.array(true_last_list, dtype=float),
        "start_time": [window_times[i] for i in widx] if window_times is not None else None,
        "tail": [window_tails[i] for i in widx] if window_tails is not None else None,
    })
    metrics_df["start_time"] = pd.to_datetime(metrics_df["start_time"], errors="coerce")

    fault_map = load_fault_map_for_side(args.side)
    if fault_map:
        metrics_df["fault_date"] = metrics_df["tail"].map(fault_map)
        metrics_df["fault_date"] = pd.to_datetime(metrics_df["fault_date"], errors="coerce")
        metrics_df["days_to_fault"] = (
            metrics_df["fault_date"] - metrics_df["start_time"]
        ).dt.total_seconds() / 86400.0
    else:
        metrics_df["fault_date"] = pd.NaT
        metrics_df["days_to_fault"] = np.nan

    save_dir = os.path.join(args.checkpoints, args.setting)
    os.makedirs(save_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---- 1) 总体 metrics csv（window-level）----
    metrics_path = os.path.join(save_dir, f"{tag}_metrics_{stamp}.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("%s metrics saved: %s | Avg MSE=%.6f MAE=%.6f",
                tag, metrics_path, float(np.mean(mse_list)), float(np.mean(mae_list)))

    # ---- 2) 保存每个 window 的 96点 pred/true 原始序列到 npz（不聚合）----
    seq_path = os.path.join(save_dir, f"{tag}_pred_true_seq_{stamp}.npz")
    np.savez_compressed(
        seq_path,
        window_idx=widx.astype(np.int64),
        pred=pred_seq.astype(np.float32),   # [N, 96]
        true=true_seq.astype(np.float32),   # [N, 96]
    )
    logger.info("%s pred/true seq saved: %s | shape=%s", tag, seq_path, tuple(pred_seq.shape))

    # ---- 3) 总体残差直方图（用 last 点，防止太大）----
    plots_dir = os.path.join(save_dir, "plots", tag)
    os.makedirs(plots_dir, exist_ok=True)
    _plot_residual_hist(
        metrics_df,
        out_png=os.path.join(plots_dir, f"residual_hist_last_{tag}_{stamp}.png"),
        title=f"{args.setting} | {tag} | residual(last) pred_last-true_last"
    )

    # ---- 4) per-tail 明细 + 每架飞机画前5个window的96点曲线 ----
    per_tail_dir = os.path.join(save_dir, "per_tail")
    os.makedirs(per_tail_dir, exist_ok=True)

    alarm_rows = []
    for tail, g in metrics_df.groupby("tail"):
        if pd.isna(tail):
            continue
        g = g.sort_values("start_time").reset_index(drop=True)

        # per-tail csv：不写 96 点（避免 csv 巨大），只写 window_idx 作为索引去 npz 取序列
        tail_df = pd.DataFrame({
            "time": g["start_time"],
            "loss": g["mse"].astype(float),
            "mae": g["mae"].astype(float),
            "pred_last": g["pred_last"].astype(float),
            "true_last": g["true_last"].astype(float),
            "window_idx": g["window_idx"].astype(int),
            "days_to_fault": g["days_to_fault"],
            "fault_date": g["fault_date"],
        })

        per_tail_path = os.path.join(per_tail_dir, f"{tag}_{tail}_windows_{stamp}.csv")
        tail_df.to_csv(per_tail_path, index=False)

        # 每架飞机只画前5个 window 的 96点曲线
        tail_plot_dir = os.path.join(save_dir, "plots", tag, str(tail))
        _plot_first_k_windows_96pt(
            df_tail_windows=tail_df[["time", "window_idx"]],
            seq_npz_path=seq_path,
            out_dir=tail_plot_dir,
            tag=tag,
            tail=str(tail),
            k=getattr(args, "plot_first_k_windows", 5),
        )

        # 报警逻辑仍然基于 loss（mse）
        mean_up, is_alarm, first_alarm_time, _ = alarm_from_loss_series(
            tail=tail,
            loss_df=tail_df[["time", "loss"]],
            save_dir=per_tail_dir,
            tag=tag,
            threshold=getattr(args, "alarm_threshold", 0.5),
            keep_last_n=300,
        )

        alarm_rows.append({
            "tag": tag,
            "tail": tail,
            "n_windows": int(len(g)),
            "final_mean_up": mean_up,
            "alarm": bool(is_alarm) if mean_up is not None else False,
            "first_alarm_time": first_alarm_time,
            "fault_date": g["fault_date"].iloc[0] if "fault_date" in g.columns else pd.NaT,
        })

    alarm_df_sum = pd.DataFrame(alarm_rows)
    if len(alarm_df_sum) > 0:
        alarm_df_sum = alarm_df_sum.sort_values(["alarm", "final_mean_up"], ascending=[False, False])

    alarm_path = os.path.join(save_dir, f"alarm_summary_{tag}_{stamp}.csv")
    alarm_df_sum.to_csv(alarm_path, index=False)
    logger.info("%s alarm summary saved: %s", tag, alarm_path)

    logger.info("Plots saved under: %s", os.path.join(save_dir, "plots", tag))
    logger.info("Per-tail CSV saved under: %s", per_tail_dir)


# =========================================================
# 5) train / eval
# =========================================================
def train_right_regression_timerxl(args):
    logger = setup_logger(args.setting)
    device = args.gpu

    base_all, full_ds, train_ds, val_ds, train_loader, val_loader = _build_loaders(args)

    logger.info("Total train_normal windows (2 months window pool) = %d", len(full_ds))
    logger.info("Train windows = %d", len(train_ds))
    logger.info("Val windows   = %d", len(val_ds))
    logger.info("Feature names = %s", getattr(base_all, "feature_names", None))

    model, optim, sched, criterion = _build_model_and_optim(args, device)

    save_dir = os.path.join(args.checkpoints, args.setting)
    os.makedirs(save_dir, exist_ok=True)

    best_val = float("inf")
    best_path = os.path.join(save_dir, "best_timerxl_right_reg.pth")

    for ep in range(1, args.train_epochs + 1):
        model.train()
        t0 = time.time()
        tr_sum, nsamp = 0.0, 0

        for batch_x, batch_y, _ in train_loader:
            _, _, L, C = batch_x.shape
            x = batch_x.to(device).reshape(-1, L, C)
            y = batch_y.to(device).reshape(-1, L, 1)

            pred_all = model(x)
            pred = pred_all[:, -1, :].unsqueeze(-1)

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
        lr_now = optim.param_groups[0]["lr"]

        logger.info(
            f"Epoch {ep}/{args.train_epochs} | train={train_loss:.6f} | val={val_loss:.6f} | lr={lr_now:.2e} | {dt:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model saved → {best_path}")

        if sched is not None:
            sched.step()

    final_path = os.path.join(save_dir, "final_timerxl_right_reg.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training finished. Best val={best_val:.6f}")


def evaluate_regression_timerxl(args, ckpt_path):
    """
    评估只跑：
      1) test_normal_recent （你定义的 test_normal：紧接 train 的 1 个月）
      2) test_abnormal      （故障前 1 个月）
    """
    logger = setup_logger(args.setting + "_eval")

    logger.info("Evaluating on test_normal_recent (continuous 1 month) ...")
    base_nr = FlightDataset_acm(args, Tag="test_normal_recent", side=args.side)
    if len(base_nr) > 0:
        ds_nr = Dataset_RegRight_TimerXL(base_nr)
        _eval_on_subset(args, logger, ckpt_path, base_nr, ds_nr, tag="test_normal_recent")
    else:
        logger.warning("No test_normal_recent data found, skip.")

    logger.info("Evaluating on test_abnormal (continuous 1 month before fault) ...")
    base_abn = FlightDataset_acm(args, Tag="test_abnormal", side=args.side)
    if len(base_abn) > 0:
        ds_abn = Dataset_RegRight_TimerXL(base_abn)
        _eval_on_subset(args, logger, ckpt_path, base_abn, ds_abn, tag="test_abnormal")
    else:
        logger.warning("No abnormal data found, skip test_abnormal evaluation.")


# =========================================================
# 6) Main：自动跑 PACK1/PACK2
# =========================================================
if __name__ == "__main__":

    class Args:
        pass

    base_args = Args()
    base_args.seq_len = 96
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

    # Split (train/val only)
    base_args.val_ratio = 0.1
    base_args.split_seed = 42

    # Alarm
    base_args.alarm_threshold = 0.5

    # Plot controls
    base_args.plot_first_k_windows = 5  # 每架飞机只画前5个window的96点曲线

    # Paths
    base_args.checkpoints = "./checkpoints"

    # ====== 关键：时间切分参数 ======
    base_args.normal_months = 12           # train/val 2个月
    base_args.test_normal_months = 1      # test_normal 1个月
    base_args.fault_gap_months = 6        # 离故障 6 个月那个月作为 anchor_end
    base_args.normal_anchor_end = "2025-08-01"  # 无故障飞机的 anchor_end

    base_setting = (
        f"timerxl_reg_train{base_args.normal_months}m_test{base_args.test_normal_months}m_"
        f"gap{base_args.fault_gap_months}m_{base_args.normal_anchor_end}endmonth_PACK_DISCH_T"
    )

    for side in ["PACK2", "PACK1"]:
        args = Args()
        args.__dict__.update(base_args.__dict__)
        args.side = side
        args.setting = f"{base_setting}_{side}"

        best_ckpt = os.path.join(args.checkpoints, args.setting, "best_timerxl_right_reg.pth")

        print("\n==============================")
        print(
            f"Running side={side} | setting={args.setting} | "
            f"trainM={args.normal_months} testM={args.test_normal_months} gapM={args.fault_gap_months} "
            f"anchor_end(no-fault)={args.normal_anchor_end}"
        )
        print("==============================")

        if os.path.exists(best_ckpt):
            print("Found existing checkpoint → evaluating")
            evaluate_regression_timerxl(args, ckpt_path=best_ckpt)
        else:
            print("No checkpoint found → training")
            train_right_regression_timerxl(args)
            if os.path.exists(best_ckpt):
                evaluate_regression_timerxl(args, ckpt_path=best_ckpt)
