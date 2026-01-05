# alarm_eval_utils.py
# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Callable, Any

from alarm import (
    load_metrics_csv, find_latest_metrics_csv,
    alarm_high_value, alarm_mean_up
)

def load_latest_pair(checkpoints_dir: str,
                     normal_pattern: str = "test_normal*_metrics_*.csv",
                     abnormal_pattern: str = "test_abnormal*_metrics_*.csv"):
    normal_csv = find_latest_metrics_csv(checkpoints_dir, normal_pattern)
    abnormal_csv = find_latest_metrics_csv(checkpoints_dir, abnormal_pattern)
    df_n = load_metrics_csv(normal_csv)
    df_a = load_metrics_csv(abnormal_csv)
    return normal_csv, abnormal_csv, df_n, df_a

def per_tail_decisions(df: pd.DataFrame,
                       method: str,
                       params: Dict[str, Any],
                       loss_col_high: str = "mae",
                       loss_col_meanup: str = "mse") -> Dict[str, bool]:
    """
    返回 tail -> alarm(bool)
    method: "high_value" or "mean_up"
    """
    decisions = {}
    for tail, g in df.groupby("tail"):
        if pd.isna(tail):
            continue
        if method == "high_value":
            r = alarm_high_value(g, loss_col=loss_col_high, **params)
            decisions[str(tail)] = bool(r.alarm)
        elif method == "mean_up":
            r = alarm_mean_up(g, loss_col=loss_col_meanup, **params)
            decisions[str(tail)] = bool(r.alarm)
        else:
            raise ValueError(f"Unknown method={method}")
    return decisions

def confusion_from_two_sets(normal_dec: Dict[str, bool],
                            abnormal_dec: Dict[str, bool]) -> Dict[str, int]:
    """
    normal是负样本、abnormal是正样本
    """
    fp = sum(1 for t, a in normal_dec.items() if a)
    tn = sum(1 for t, a in normal_dec.items() if not a)
    tp = sum(1 for t, a in abnormal_dec.items() if a)
    fn = sum(1 for t, a in abnormal_dec.items() if not a)
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn}

def score_from_conf(conf: Dict[str, int], mode: str = "f1",
                    fp_weight: float = 1.0) -> float:
    tp, fn, fp, tn = conf["tp"], conf["fn"], conf["fp"], conf["tn"]

    # precision / recall
    prec = tp / max(1, (tp + fp))
    rec  = tp / max(1, (tp + fn))

    if mode == "f1":
        return 2 * prec * rec / max(1e-12, (prec + rec))

    if mode == "recall":   # 只追求召回
        return rec

    if mode == "weighted": # 加权：惩罚 FP
        # 你可以按业务改：比如 fp_weight=2.0 表示 FP 更痛
        return (rec) - fp_weight * (fp / max(1, (fp + tn)))

    raise ValueError(f"Unknown score mode: {mode}")

def eval_params(checkpoints_dir: str,
                method: str,
                params: Dict[str, Any],
                score_mode: str = "f1",
                fp_weight: float = 2.0,
                normal_pattern: str = "test_normal*_metrics_*.csv",
                abnormal_pattern: str = "test_abnormal*_metrics_*.csv"):
    _, _, df_n, df_a = load_latest_pair(checkpoints_dir, normal_pattern, abnormal_pattern)
    ndec = per_tail_decisions(df_n, method=method, params=params)
    adec = per_tail_decisions(df_a, method=method, params=params)
    conf = confusion_from_two_sets(ndec, adec)
    sc = score_from_conf(conf, mode=score_mode, fp_weight=fp_weight)
    return sc, conf
