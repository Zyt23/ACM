# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_loader_acm_320 import FlightDataset_acm
from data_provider.acm_window_dataset import SegHeadWindowDataset, AnomalyMixDataset, split_indices_by_tail
from data_provider.acm_anomaly_synth import AnomType
from models.timer_xl import Model as TimerXL
from models.acm_classifier import TinyTransformerClassifier


# ---------------- logger ----------------
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


def collate_keep_info(batch):
    # batch: List[(x:[L,6], y:long, info:any)]
    xs, ys, infos = zip(*batch)
    xs = torch.stack(xs, dim=0)      # [B,L,6]
    ys = torch.stack(ys, dim=0)      # [B]
    return xs, ys, list(infos)


# ---------------- TimerXL configs (match your regress script) ----------------
class TimerXLConfigs:
    def __init__(self, args):
        self.input_token_len = args.input_token_len  # must == win_len
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
    model = TimerXL(cfg).to(device)
    return model


@torch.no_grad()
def timerxl_predict_sequence(reg_model: nn.Module, x_cov: torch.Tensor) -> torch.Tensor:
    """
    x_cov: [B,L,5]
    return y_pred: [B,L,1]
    NOTE: adjust here if your TimerXL output shape differs.
    """
    out_all = reg_model(x_cov)
    pred = out_all[:, -1, :].unsqueeze(-1)      # -> [B,L,1] (match your regress script usage)
    return pred


def build_classifier(args, in_dim: int, num_classes: int, device):
    clf = TinyTransformerClassifier(
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
    return clf


def safe_torch_load(path: str, device):
    # 优先使用 weights_only=True（新版 torch 更安全），旧版不支持则回退
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def to_binary_label(y_raw: torch.Tensor) -> torch.Tensor:
    """
    y_raw: 多类标签（AnomType.*）
    return y_bin: 0(normal) / 1(abnormal)
    """
    return (y_raw != int(AnomType.NORMAL)).long()


def compute_metrics_binary(logits: torch.Tensor, y_bin: torch.Tensor):
    """
    logits: [B,2], y_bin:[B]
    """
    prob = torch.softmax(logits, dim=1)
    p_abn = prob[:, 1]
    pred_bin = (p_abn > 0.5).long()
    acc_bin = (pred_bin == y_bin).float().mean().item()
    return {"acc_bin@0.5": acc_bin}


def main():
    class Args: pass
    args = Args()

    # ---------------- data/time split (use your existing) ----------------
    args.seq_len = 96
    args.max_windows_per_flight = 5

    args.normal_months = 10
    args.test_normal_months = 1
    args.fault_gap_months = 6
    args.normal_anchor_end = "2025-08-01"

    args.raw_months = 12
    args.raw_end_use_gap = False

    # ---------------- side & window ----------------
    args.side = "PACK2"
    args.win_len = 96
    args.stride = 96
    args.input_token_len = args.win_len  # MUST match TimerXL

    # ---------------- device ----------------
    args.gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- TimerXL regressor (must match your trained ckpt config) ----------------
    args.d_model = 128
    args.nhead = 4
    args.num_layers = 3
    args.dim_ff = 256
    args.dropout = 0.1

    # path to pretrained regressor ckpt (from your aligned regress training)
    args.reg_ckpt = "./checkpoints/timerxl_aligned_regress_keep5x96_raw12m_train10m_test1m_gap6m_2025-08-01end_noALTSTD_win96_stride96_PACK2/best_timerxl_regress_win96.pth"

    # ---------------- classifier hyperparams ----------------
    args.cls_d_model = 128
    args.cls_nhead = 4
    args.cls_layers = 3
    args.cls_ff = 256
    args.cls_dropout = 0.1
    args.cls_pooling = "mean"

    # training
    args.batch_size = 256
    args.num_workers = 4
    args.epochs = 50
    args.lr = 3e-4
    args.wd = 1e-4
    args.cosine = True
    args.tmax = 50

    # anomaly mixing
    args.p_anom = 0.5
    args.val_ratio_tail = 0.1
    args.seed = 42

    # threshold calibration target FPR on NORMAL (val)
    args.fpr_target = 0.005

    # save
    args.checkpoints = "./checkpoints"
    args.setting = f"mech_cls_BIN_{args.side}_win{args.win_len}_stride{args.stride}_panom{args.p_anom}"

    logger = setup_logger(args.setting)
    device = args.gpu

    # ---------------- build base dataset (loads from your seghead cache) ----------------
    base = FlightDataset_acm(args, Tag="train_normal", side=args.side)
    if len(base) == 0:
        raise RuntimeError("train_normal base dataset is empty.")

    win_ds = SegHeadWindowDataset(base, win_len=args.win_len, stride=args.stride)

    # anomaly dataset (原本是 multi-class 标签，但我们训练时映射为二分类)
    # 为了不让 train/val 相互影响，分别实例化两个 AnomalyMixDataset
    anom_ds_train = AnomalyMixDataset(
        win_ds,
        p_anom=args.p_anom,
        seed=args.seed,
        anom_kinds=[
            AnomType.ENERGY_RATIO,
            AnomType.MISALIGN_TEMP,
            AnomType.MISALIGN_VALVE,
            AnomType.DRIFT,
            AnomType.OSCILLATION,
            AnomType.STUCK_VALVE,
            AnomType.NOISE,
        ],
        synth_params={}
    )
    anom_ds_val = AnomalyMixDataset(
        win_ds,
        p_anom=args.p_anom,
        seed=args.seed + 999,  # val 用不同 seed（稳定且避免和 train 完全同模板）
        anom_kinds=[
            AnomType.ENERGY_RATIO,
            AnomType.MISALIGN_TEMP,
            AnomType.MISALIGN_VALVE,
            AnomType.DRIFT,
            AnomType.OSCILLATION,
            AnomType.STUCK_VALVE,
            AnomType.NOISE,
        ],
        synth_params={}
    )

    # tail split (BEST PRACTICE)
    tr_idx, va_idx = split_indices_by_tail(win_ds.sample_tails, val_ratio=args.val_ratio_tail, seed=args.seed)
    train_set = Subset(anom_ds_train, tr_idx)
    val_set = Subset(anom_ds_val, va_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_keep_info,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_keep_info,
    )

    # ---------------- load regressor and freeze ----------------
    reg = build_timerxl_regressor(args, device)
    if not os.path.exists(args.reg_ckpt):
        raise FileNotFoundError(f"reg_ckpt not found: {args.reg_ckpt}")
    state = safe_torch_load(args.reg_ckpt, device)
    reg.load_state_dict(state)
    reg.eval()
    for p in reg.parameters():
        p.requires_grad = False

    # indices in window: [BYPASS, DISCH, RAM_I, RAM_O, FLOW, COMPR]
    idx_x = [0, 1, 2, 3, 4]  # covariates (mask target)
    idx_y = 5               # COMPR_T

    # classifier input dim = cov(5) + y_true(1) + y_pred(1) + residual(1) = 8
    in_dim = 8
    num_classes = 2  # ✅ 二分类
    clf = build_classifier(args, in_dim=in_dim, num_classes=num_classes, device=device)

    # label smoothing 对“合成标签不完美”更稳
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    optim = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.tmax, eta_min=1e-8) if args.cosine else None

    save_dir = os.path.join(args.checkpoints, args.setting)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best_mech_cls_bin.pth")
    thr_path = os.path.join(save_dir, "thresholds_bin.json")

    best_val = 1e18

    logger.info("========== TRAIN mech-cls (BINARY) ==========")
    logger.info("base segheads=%d | windows=%d | train=%d | val=%d", len(base), len(win_ds), len(train_set), len(val_set))
    logger.info("num_classes=%d | in_dim=%d | device=%s", num_classes, in_dim, str(device))

    # ---------------- training loop ----------------
    for ep in range(1, int(args.epochs) + 1):
        t0 = time.time()
        clf.train()
        tr_loss, tr_n = 0.0, 0

        for win, y_raw, _info in train_loader:
            # win: [B,L,6]
            win = win.to(device)
            y_raw = y_raw.to(device)
            y = to_binary_label(y_raw)  # ✅ 二分类标签

            cov = win[:, :, idx_x]              # [B,L,5]
            y_true = win[:, :, idx_y:idx_y+1]   # [B,L,1]

            with torch.no_grad():
                y_pred = timerxl_predict_sequence(reg, cov)  # [B,L,1]
            resid = y_true - y_pred

            x_clf = torch.cat([cov, y_true, y_pred, resid], dim=-1)  # [B,L,8]

            logits = clf(x_clf)   # [B,2]
            loss = crit(logits, y)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            bsz = win.size(0)
            tr_loss += float(loss.item()) * bsz
            tr_n += bsz

        tr_loss = tr_loss / max(1, tr_n)

        # val
        clf.eval()
        va_loss, va_n = 0.0, 0
        m_accb, m_cnt = 0.0, 0
        with torch.no_grad():
            for win, y_raw, _info in val_loader:
                win = win.to(device)
                y_raw = y_raw.to(device)
                y = to_binary_label(y_raw)

                cov = win[:, :, idx_x]
                y_true = win[:, :, idx_y:idx_y+1]
                y_pred = timerxl_predict_sequence(reg, cov)
                resid = y_true - y_pred
                x_clf = torch.cat([cov, y_true, y_pred, resid], dim=-1)

                logits = clf(x_clf)
                loss = crit(logits, y)

                bsz = win.size(0)
                va_loss += float(loss.item()) * bsz
                va_n += bsz

                mets = compute_metrics_binary(logits, y)
                m_accb += mets["acc_bin@0.5"] * bsz
                m_cnt += bsz

        va_loss = va_loss / max(1, va_n)
        accb = m_accb / max(1, m_cnt)

        if sched is not None:
            sched.step()

        lr_now = float(optim.param_groups[0]["lr"])
        dt = time.time() - t0
        logger.info("Epoch %d/%d | tr=%.6f | va=%.6f | acc_bin=%.3f | lr=%.2e | %.1fs",
                    ep, int(args.epochs), tr_loss, va_loss, accb, lr_now, dt)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(clf.state_dict(), best_path)
            logger.info("New best -> %s (va=%.6f)", best_path, best_val)

    # ---------------- threshold calibration on NORMAL windows in val split ----------------
    logger.info("========== Calibrate threshold on val NORMAL (BINARY) ==========")
    clf.load_state_dict(safe_torch_load(best_path, device))
    clf.eval()

    p_abn_list = []
    with torch.no_grad():
        for win, y_raw, _info in val_loader:
            # 只取“原始标签=normal”的样本做FPR校准
            mask = (y_raw == int(AnomType.NORMAL))
            if mask.sum().item() == 0:
                continue

            win = win[mask].to(device)

            cov = win[:, :, idx_x]
            y_true = win[:, :, idx_y:idx_y+1]
            y_pred = timerxl_predict_sequence(reg, cov)
            resid = y_true - y_pred
            x_clf = torch.cat([cov, y_true, y_pred, resid], dim=-1)

            logits = clf(x_clf)                  # [B,2]
            prob = torch.softmax(logits, dim=1)
            p_abn = prob[:, 1]                   # ✅ 二分类：abnormal 概率
            p_abn_list.append(p_abn.detach().cpu().numpy())

    if p_abn_list:
        p = np.concatenate(p_abn_list)
        thr = float(np.quantile(p, 1.0 - float(args.fpr_target)))
    else:
        thr = 0.5

    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump({"p_abn_thr": thr, "fpr_target": float(args.fpr_target)}, f, ensure_ascii=False, indent=2)

    logger.info("Saved best cls: %s", best_path)
    logger.info("Saved threshold: %s | p_abn_thr=%.6f", thr_path, thr)
    logger.info("DONE.")


if __name__ == "__main__":
    main()
