# -*- coding: utf-8 -*-
import os, time, torch, logging, numpy as np, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from datetime import datetime

# ★★★ 先把项目根目录加进 sys.path ★★★
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 再去 import 自己项目里的模块
from data_provider.data_loader_acm_320_old import FlightDataset_acm, Dataset_RegRight_TimerXL
from models.timer_xl import Model as TimerXL


# ------------------------------------------------------------
# TimerXL 配置转换
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------
def setup_logger(setting: str):
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/{setting}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(setting)
    logger.setLevel(logging.INFO)

    # 清理旧 handler
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


# ------------------------------------------------------------
# 构建模型和优化器
# ------------------------------------------------------------
def _build_model_and_optim(args, device):
    cfg = TimerXLConfigs(args)
    model = TimerXL(cfg).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=args.tmax, eta_min=1e-8
        )
        if args.cosine
        else None
    )
    criterion = nn.MSELoss()
    return model, optim, sched, criterion


# ------------------------------------------------------------
# 数据加载器构建：一次性构建 normal 全集，并且划分 train / val / test
# ------------------------------------------------------------
def _build_loaders(args):
    """
    整体逻辑：

      1. 只构建一次 FlightDataset_acm(Tag='train_normal')：
            - 第一次运行会去 IoTDB 查 normal 那一个月并写 cache；
            - 之后再运行会直接从 cache 里读 npy，不再查 IoTDB。

      2. 基于同一个 base_all 构建 Dataset_RegRight_TimerXL(full_ds)，
         然后用 random_split 一次性切成 train / val / test 三个 subset，
         三者在“窗口”维度上互不重叠。

      3. 把三份 subset 的 base 索引写入 normal_split_indices.npz，
         后续 evaluate 的时候可复现同样的划分。
    """

    # 1) 只构建一次 normal 数据集（IoTDB 在这里被触发）
    base_all = FlightDataset_acm(args, Tag="train_normal", side=args.side)
    full_ds = Dataset_RegRight_TimerXL(base_all)  # 每个元素 = 一个窗口

    n_total = len(full_ds)
    if n_total == 0:
        raise RuntimeError("full_ds is empty, please check IoTDB query or filters.")

    # 划分比例：train / val / test
    val_ratio = getattr(args, "val_ratio", 0.1)
    test_ratio = getattr(args, "test_ratio", 0.1)

    n_val = max(1, int(n_total * val_ratio))
    n_test = max(1, int(n_total * test_ratio))
    n_train = n_total - n_val - n_test
    if n_train <= 0:
        raise RuntimeError(
            f"Not enough samples to split, "
            f"total={n_total}, val_ratio={val_ratio}, test_ratio={test_ratio}"
        )

    # 为了可复现，用一个固定的 seed
    split_seed = getattr(args, "split_seed", 42)
    g = torch.Generator().manual_seed(split_seed)

    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=g
    )

    # 2) 保存划分的 base 索引（注意：这些索引是相对于 full_ds / base_all 的窗口索引）
    split_dir = os.path.join(args.checkpoints, args.setting)
    os.makedirs(split_dir, exist_ok=True)
    np.savez(
        os.path.join(split_dir, "normal_split_indices.npz"),
        train_idx=np.array(train_ds.indices, dtype=np.int64),
        val_idx=np.array(val_ds.indices, dtype=np.int64),
        test_idx=np.array(test_ds.indices, dtype=np.int64),
    )

    # 3) 构建 DataLoader（训练阶段只用 train / val）
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 返回所有对象，方便后续用
    return base_all, full_ds, train_ds, val_ds, test_ds, train_loader, val_loader


# ------------------------------------------------------------
# 在任意一个 subset 上做推理并保存结果
# ------------------------------------------------------------
@torch.no_grad()
def _eval_on_subset(args, logger, ckpt_path, base_dataset, subset, tag: str):
    """
    base_dataset : FlightDataset_acm 对象（内部有 window_start_times）
    subset       : Dataset_RegRight_TimerXL(base_dataset) 或它的 Subset
    tag          : 'val_normal' / 'test_normal' / 'test_abnormal' 等
    """
    device = args.gpu

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model, _, _, _ = _build_model_and_optim(args, device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    preds_list, targs_list, idx_list = [], [], []
    mse_list, mae_list = [], []

    for batch_x, batch_y, batch_idx in loader:
        _, _, L, C = batch_x.shape  # [B,1,L,C_in]
        x = batch_x.to(device).reshape(-1, L, C)  # [B,L,C_in]
        y = batch_y.to(device).reshape(-1, L, 1)  # [B,L,1]

        out_all = model(x)  # [B, C_out, L]
        pred = out_all[:, -1, :].unsqueeze(-1)  # 取最后一通道作为目标 [B,L,1]

        p = pred.squeeze(-1).cpu().numpy()  # [B,L]
        t = y.squeeze(-1).cpu().numpy()     # [B,L]

        preds_list.append(p)
        targs_list.append(t)
        idx_list.append(batch_idx.view(-1).cpu().numpy())  # 这里的 idx 是 base 的窗口索引

        mse_list.extend(((p - t) ** 2).mean(axis=1))
        mae_list.extend(np.abs(p - t).mean(axis=1))

    preds = np.concatenate(preds_list, axis=0)
    targs = np.concatenate(targs_list, axis=0)
    widx = np.concatenate(idx_list, axis=0)

    save_dir = os.path.join(args.checkpoints, args.setting)
    os.makedirs(save_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez_compressed(
        os.path.join(save_dir, f"{tag}_preds_{stamp}.npz"),
        pred=preds,
        target=targs,
        window_idx=widx,
    )

    metrics_df = pd.DataFrame(
        {
            "window_idx": widx,
            "mse": mse_list,
            "mae": mae_list,
        }
    )

    window_times = getattr(base_dataset, "window_start_times", None)
    if window_times is not None:
        metrics_df["start_time"] = [window_times[i] for i in widx]

    metrics_df.to_csv(
        os.path.join(save_dir, f"{tag}_metrics_{stamp}.csv"),
        index=False,
    )

    logger.info(
        f"{tag} metrics saved.  Avg MSE={np.mean(mse_list):.6f}  MAE={np.mean(mae_list):.6f}"
    )


# ------------------------------------------------------------
# 训练
# ------------------------------------------------------------
def train_right_regression_timerxl(args):
    logger = setup_logger(args.setting)
    device = args.gpu

    (
        base_all,
        full_ds,
        train_ds,
        val_ds,
        test_ds,
        train_loader,
        val_loader,
    ) = _build_loaders(args)

    logger.info("Total normal windows = %d", len(full_ds))
    logger.info("Train windows = %d", len(train_ds))
    logger.info("Val windows   = %d", len(val_ds))
    logger.info("Test windows  = %d (normal)", len(test_ds))
    logger.info("Feature names = %s", getattr(base_all, "feature_names", None))

    model, optim, sched, criterion = _build_model_and_optim(args, device)

    save_dir = os.path.join(args.checkpoints, args.setting)
    os.makedirs(save_dir, exist_ok=True)

    best_val = float("inf")
    best_path = os.path.join(save_dir, "best_timerxl_right_reg.pth")
    history = []

    for ep in range(1, args.train_epochs + 1):
        # ------------------- Train -------------------
        model.train()
        t0 = time.time()
        tr_sum, nsamp = 0.0, 0

        for batch_x, batch_y, _ in train_loader:
            N, S, L, C = batch_x.shape  # [B,1,L,C_in]
            x = batch_x.to(device).reshape(-1, L, C)  # [B,L,C_in]
            y = batch_y.to(device).reshape(-1, L, 1)  # [B,L,1]

            pred_all = model(x)  # [B, C_out, L]
            pred = pred_all[:, -1, :].unsqueeze(-1)  # 取最后一通道 [B,L,1]

            loss = criterion(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            tr_sum += loss.item() * x.size(0)
            nsamp += x.size(0)

        train_loss = tr_sum / max(1, nsamp)

        # ------------------- Val -------------------
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
            f"Epoch {ep}/{args.train_epochs}  |  train={train_loss:.6f} "
            f"|  val={val_loss:.6f}  |  lr={lr_now:.2e}  |  {dt:.1f}s"
        )

        history.append(
            {
                "epoch": ep,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr_now,
                "seconds": dt,
            }
        )

        # 保存最佳模型（按 val_loss）
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info(f"New best model saved → {best_path}")

        if sched is not None:
            sched.step()

    # 保存最终模型
    final_path = os.path.join(save_dir, "final_timerxl_right_reg.pth")
    torch.save(model.state_dict(), final_path)
    logger.info(f"Training finished. Best val={best_val:.6f}")

    # 保存训练曲线
    pd.DataFrame(history).to_csv(
        os.path.join(save_dir, "training_history.csv"), index=False
    )

    # ★ 这里不再自动重新构建数据集做 val_normal 的评估，避免再查 IoTDB。
    #   真正需要保存详细预测结果时，用 evaluate_regression_timerxl 再跑一遍即可。


# ------------------------------------------------------------
# 评估接口：使用 normal_split_indices.npz 保证 test_normal 不与 train 重叠
# ------------------------------------------------------------
def evaluate_regression_timerxl(args, ckpt_path):
    logger = setup_logger(args.setting + "_eval")

    # ------------------- 1) normal test -------------------
    logger.info("Evaluating on normal test split ...")
    # 读取 train_normal 的数据（此时会直接从 cache 读，不再查 IoTDB）
    base_all = FlightDataset_acm(args, Tag="train_normal", side=args.side)
    full_ds = Dataset_RegRight_TimerXL(base_all)

    split_file = os.path.join(args.checkpoints, args.setting, "normal_split_indices.npz")
    if os.path.exists(split_file):
        idxs = np.load(split_file)
        test_idx = idxs["test_idx"]

        test_subset = Subset(full_ds, test_idx)

        _eval_on_subset(
            args,
            logger,
            ckpt_path=ckpt_path,
            base_dataset=base_all,
            subset=test_subset,
            tag="test_normal",
        )
    else:
        logger.warning(
            "normal_split_indices.npz not found, "
            "fallback: use full normal dataset as test_normal."
        )
        _eval_on_subset(
            args,
            logger,
            ckpt_path=ckpt_path,
            base_dataset=base_all,
            subset=full_ds,
            tag="test_normal",
        )

    # ------------------- 2) abnormal test -------------------
    logger.info("Evaluating on abnormal test set ...")
    base_abn = FlightDataset_acm(args, Tag="test_abnormal", side=args.side)
    if len(base_abn) > 0:
        ds_abn = Dataset_RegRight_TimerXL(base_abn)
        _eval_on_subset(
            args,
            logger,
            ckpt_path=ckpt_path,
            base_dataset=base_abn,
            subset=ds_abn,  # 全部 abnormal 窗口作为 test_abnormal
            tag="test_abnormal",
        )
    else:
        logger.warning("No abnormal data found, skip test_abnormal evaluation.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":

    class Args:
        pass

    args = Args()

    # device / 序列长度
    args.seq_len = 96
    args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.ddp = False
    args.local_rank = 0

    # Timer-XL patch length
    args.input_token_len = 24

    # Model
    args.d_model = 128
    args.nhead = 4
    args.num_layers = 3
    args.dim_ff = 256
    args.dropout = 0.1

    # Training
    args.batch_size = 64
    args.num_workers = 4
    args.learning_rate = 3e-4
    args.weight_decay = 1e-4
    args.tmax = 20
    args.train_epochs = 30
    args.cosine = True

    # 数据划分参数
    args.val_ratio = 0.1   # normal 中 10% 做 val
    args.test_ratio = 0.1  # normal 中 10% 做 test_normal
    args.split_seed = 42   # 划分随机种子，可复现
    args.side = "PACK2"    # 或 "PACK1"

    # 路径设置
    args.checkpoints = "./checkpoints"
    args.setting = "timerxl_right_reg_target_1y_PACK_DISCH_T"

    best_ckpt = os.path.join(
        args.checkpoints, args.setting, "best_timerxl_right_reg.pth"
    )

    if os.path.exists(best_ckpt):
        print("Found existing checkpoint → evaluating")
        evaluate_regression_timerxl(args, ckpt_path=best_ckpt)
    else:
        print("No checkpoint found → training")
        train_right_regression_timerxl(args)
        if os.path.exists(best_ckpt):
            evaluate_regression_timerxl(args, ckpt_path=best_ckpt)
