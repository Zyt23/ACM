# -*- coding: utf-8 -*-
import os, time, torch, logging, numpy as np, pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime

# 你现有的数据集（已做地面段筛选&派生量&归一化）
from data_provider.data_loader_acm import Dataset_RegRight_TimerXL, Dataset_RegLeft_TimerXL,FlightDataset_acm, Args
# 你已有的 Timer-XL 模型
from models.timer_xl import Model as TimerXL


class TimerXLConfigs:
    """把训练超参映射到 timer_xl.py 需要的 configs 字段"""
    def __init__(self, args):
        self.input_token_len = args.input_token_len   # 必须整除 seq_len
        self.d_model = args.d_model
        self.n_heads = args.nhead
        self.e_layers = args.num_layers
        self.d_ff = args.dim_ff
        self.dropout = args.dropout
        self.activation = 'gelu'
        self.output_attention = False
        self.covariate = False
        self.flash_attention = False
        self.use_norm = False


def setup_logger(setting: str):
    """创建 logger，日志同时写文件和终端；避免重复 handler"""
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/{setting}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger(setting)
    logger.setLevel(logging.INFO)
    # 清理旧 handler，避免重复打印
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


def _build_loaders(args, side='R'):
    """构建右侧训练/验证加载器"""
    base_train = FlightDataset_acm(args, Tag='train_normal', side=side)
    base_val   = FlightDataset_acm(args, Tag='val_normal',   side=side)

    train_ds = Dataset_RegRight_TimerXL(base_train)
    val_ds   = Dataset_RegRight_TimerXL(base_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    return base_train, base_val, train_loader, val_loader


def _build_model_and_optim(args, device):
    cfg = TimerXLConfigs(args)
    model = TimerXL(cfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.tmax, eta_min=1e-8) if args.cosine else None
    criterion = nn.MSELoss()
    return model, optim, sched, criterion


def train_right_regression_timerxl(args):
    logger = setup_logger(args.setting)
    device = torch.device(f'cuda:{args.local_rank}') if args.ddp else args.gpu

    base_train, base_val, train_loader, val_loader = _build_loaders(args, side='R')

    logger.info("R-side features: %s", getattr(base_train, 'feature_names', None))
    #todo
    for k in ['TDIFFTURB2L', 'POSTBVL', 'POSRAIL', 'TDIFFCPRSRL']:
        logger.info("%s in features: %s", k, (k in base_train.feature_names))

    model, optim, sched, criterion = _build_model_and_optim(args, device)

    save_dir = os.path.join(args.checkpoints, args.setting)
    os.makedirs(save_dir, exist_ok=True)

    history = []  # 记录每个 epoch 的指标
    best_val = float('inf')
    best_path = os.path.join(save_dir, 'best_timerxl_right_reg.pth')

    for ep in range(1, args.train_epochs + 1):
        # -------- Train --------
        model.train()
        t0 = time.time()
        tr_sum, nsamp = 0.0, 0

        for batch_x, batch_y, _ in train_loader:
            # batch_x: [N,1,L,4]
            N, S, L, C = batch_x.shape
            x = batch_x.to(device).reshape(-1, L, C)      # [B,L,4]
            y = batch_y.to(device).reshape(-1, L, 1)      # [B,L,1]

            pred_all = model(x)                           # [B, 4, L]
            pred = pred_all[:, 3, :].unsqueeze(-1)        # 第4通道作为目标

            loss = criterion(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            tr_sum += loss.item() * x.size(0)
            nsamp  += x.size(0)

        train_loss = tr_sum / max(1, nsamp)

        # -------- Val --------
        model.eval()
        val_sum, vsamp = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                N, S, L, C = batch_x.shape
                x = batch_x.to(device).reshape(-1, L, C)
                y = batch_y.to(device).reshape(-1, L, 1)
                out_all = model(x)
                pred = out_all[:, 3, :].unsqueeze(-1)
                loss = criterion(pred, y)
                val_sum += loss.item() * x.size(0)
                vsamp   += x.size(0)
        val_loss = val_sum / max(1, vsamp)

        # 日志
        lr_now = optim.param_groups[0]['lr']
        dt = time.time()-t0
        logger.info(f"Epoch {ep}/{args.train_epochs} | "
                    f"train {train_loss:.6f} | val {val_loss:.6f} | "
                    f"lr {lr_now:.2e} | {dt:.1f}s")

        # 记录历史
        history.append({
            "epoch": ep,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(lr_now),
            "seconds": float(dt),
        })

        # 保存最佳
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info("New best model saved with val_loss=%.6f → %s", best_val, best_path)

        if sched is not None:
            sched.step()

    # 保存最终模型
    final_path = os.path.join(save_dir, 'final_timerxl_right_reg.pth')
    torch.save(model.state_dict(), final_path)
    logger.info("Done training. Best val MSE: %.6f | final model: %s", best_val, final_path)

    # 保存训练曲线
    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(save_dir, "training_history.csv")
    hist_df.to_csv(hist_csv, index=False)
    logger.info("Training history saved to %s", hist_csv)

    # 训练完成后，可选：用最佳权重在验证集保存一份预测（示例）
    try:
        _save_eval_outputs(args, logger, ckpt_path=best_path, tag='val_normal', side='R')
    except Exception as e:
        logger.warning("Post-train eval on val_normal failed: %s", e)



@torch.no_grad()
def _save_eval_outputs(args, logger, ckpt_path, tag='val_normal', side='R'):
    """在指定 split 上做推理并保存预测/真值、指标，并包含窗口起始时间"""
    device = torch.device(f'cuda:{args.local_rank}') if args.ddp else args.gpu

    # ====== 加载数据 ======
    base = FlightDataset_acm(args, Tag=tag, side=side)

    # 若要保存时间戳信息，则确保在构建 FlightDataset_acm._flight_data() 时返回这些信息
    # 这里新增存放窗口起始时间（按窗口顺序对应）
    window_start_times = getattr(base, "window_start_times", None)
    if window_start_times is None:
        logger.warning("当前数据集未包含窗口起始时间（window_start_times），请稍后在 FlightDataset_acm._flight_data() 添加。")
        window_start_times = [None] * len(base)

    ds = Dataset_RegRight_TimerXL(base) if side == 'R' else Dataset_RegLeft_TimerXL(base)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # ====== 模型加载 ======
    model, _, _, _ = _build_model_and_optim(args, device)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info("Loaded checkpoint for eval: %s", ckpt_path)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.eval()

    # ====== 推理 ======
    preds_list, targs_list, idx_list = [], [], []
    mse_list, mae_list = [], []

    for i, (batch_x, batch_y, batch_idx) in enumerate(loader):
        N, S, L, C = batch_x.shape
        x = batch_x.to(device).reshape(-1, L, C)
        y = batch_y.to(device).reshape(-1, L, 1)
        out_all = model(x)
        pred = out_all[:, 3, :].unsqueeze(-1)

        p = pred.squeeze(-1).cpu().numpy()
        t = y.squeeze(-1).cpu().numpy()
        preds_list.append(p)
        targs_list.append(t)
        idx_list.append(batch_idx.view(-1).cpu().numpy())

        se = (p - t) ** 2
        ae = np.abs(p - t)
        mse_list.extend(se.mean(axis=1).tolist())
        mae_list.extend(ae.mean(axis=1).tolist())

    if len(preds_list) == 0:
        logger.warning("No data in split=%s to evaluate.", tag)
        return

    preds = np.concatenate(preds_list, axis=0)
    targs = np.concatenate(targs_list, axis=0)
    widx  = np.concatenate(idx_list, axis=0)

    # ====== 保存结果 ======
    save_dir = os.path.join(args.checkpoints, args.setting)
    os.makedirs(save_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    npz_path = os.path.join(save_dir, f"{tag}_preds_{stamp}.npz")
    np.savez_compressed(npz_path, pred=preds, target=targs, window_idx=widx)
    logger.info("Saved %s predictions to %s  (shape: %s)", tag, npz_path, preds.shape)

    # ====== 窗口级指标 & 时间 ======
    metrics_df = pd.DataFrame({
        "window_idx": widx,
        "mse": mse_list,
        "mae": mae_list,
    })

    # 加入时间戳列（若数据集中存在）
    if window_start_times is not None and len(window_start_times) >= len(metrics_df):
        metrics_df["start_time"] = [window_start_times[i] for i in widx]
    else:
        metrics_df["start_time"] = None

    csv_path = os.path.join(save_dir, f"{tag}_metrics_{stamp}.csv")
    metrics_df.to_csv(csv_path, index=False)
    logger.info("%s metrics saved to %s | avg MSE=%.6f MAE=%.6f",
                tag, csv_path, metrics_df['mse'].mean(), metrics_df['mae'].mean())

def evaluate_regression_timerxl(args, ckpt_path):
    logger = setup_logger(args.setting + "_eval")
    try:
        if args.side == 'L':
            _save_eval_outputs(args, logger, ckpt_path=ckpt_path, tag='test_normal', side='L')
            _save_eval_outputs(args, logger, ckpt_path, tag='test_abnormal', side='L')
        else:
            _save_eval_outputs(args, logger, ckpt_path=ckpt_path, tag='test_normal', side='R')
            _save_eval_outputs(args, logger, ckpt_path, tag='test_abnormal', side='R')
            
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise



if __name__ == '__main__':
    # 你工程里的 Args 也可直接复用，这里先给默认值
    args = Args()
    # 设备
    args.gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.ddp = False
    args.local_rank = 0

    # Timer-XL 需要 patch 长度能整除 seq_len (默认 96 → 可用 12/16/24/32/48)
    args.input_token_len = 24

    # 模型结构
    args.d_model = 128
    args.nhead = 4
    args.num_layers = 3
    args.dim_ff = 256
    args.dropout = 0.1

    # 训练
    args.batch_size = 64
    args.num_workers = 4
    args.learning_rate = 3e-4
    args.weight_decay = 1e-4
    args.tmax = 20
    args.train_epochs = 30
    args.cosine = True
    args.patience = 5
    args.side = 'R'  # 'R' or 'L'

    # 目录
    args.checkpoints = './checkpoints'
    args.setting = 'timerxl_right_reg_v2'

    # === 如果已经有最优权重，则直接评测；否则先训练再评测 ===
    best_ckpt = os.path.join(args.checkpoints, args.setting, 'best_timerxl_right_reg.pth')
    if os.path.exists(best_ckpt):
        print("Found existing best checkpoint, skip training and directly evaluate.")
        evaluate_regression_timerxl(args, ckpt_path=best_ckpt)
    else:
        print("No existing best checkpoint, start training first.")
        train_right_regression_timerxl(args)
        # 训练后再评测
        if os.path.exists(best_ckpt):
            evaluate_regression_timerxl(args, ckpt_path=best_ckpt)
