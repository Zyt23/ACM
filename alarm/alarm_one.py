# alarm_hyperopt_allinone.py
# -*- coding: utf-8 -*-

import os
import glob
import random
import math
import numpy as np
import pandas as pd
from typing import Dict, Any
from types import SimpleNamespace

# =========================================================
# 数据加载
# =========================================================

def find_latest_metrics_csv(root: str, pattern: str) -> str:
    files = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    if not files:
        raise FileNotFoundError(pattern)
    return max(files, key=os.path.getmtime)

def load_metrics_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values("ts") if "ts" in df.columns else df

def load_latest_pair(checkpoints_dir: str):
    n_csv = find_latest_metrics_csv(checkpoints_dir, "test_normal*_metrics_*.csv")
    a_csv = find_latest_metrics_csv(checkpoints_dir, "test_abnormal*_metrics_*.csv")
    return load_metrics_csv(n_csv), load_metrics_csv(a_csv)

# =========================================================
# 核心报警方法：mean_up + zscore + ratio + hits
# =========================================================

def alarm_mean_up_zscore(
    df: pd.DataFrame,
    loss_col: str = "mse",
    baseline_window: int = 120,
    recent_window: int = 30,
    z_thresh: float = 3.0,
    ratio_thresh: float = 1.3,
    min_hits: int = 3,
):
    x = df[loss_col].dropna().values
    if len(x) < baseline_window + recent_window:
        return SimpleNamespace(alarm=False)

    base = x[-(baseline_window + recent_window):-recent_window]
    recent = x[-recent_window:]

    mu = base.mean()
    std = base.std() + 1e-6

    z = (recent - mu) / std
    ratio = recent / max(mu, 1e-6)

    hits = np.sum((z >= z_thresh) & (ratio >= ratio_thresh))

    return SimpleNamespace(
        alarm=hits >= min_hits,
        hits=int(hits),
        z_max=float(z.max()),
        ratio_max=float(ratio.max()),
    )

# =========================================================
# tail 级别决策
# =========================================================

def per_tail_decisions(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, bool]:
    out = {}
    for tail, g in df.groupby("tail"):
        if pd.isna(tail):
            continue
        r = alarm_mean_up_zscore(g, **params)
        out[str(tail)] = bool(r.alarm)
    return out

# =========================================================
# 评估指标
# =========================================================

def confusion(normal_dec: Dict[str, bool], abnormal_dec: Dict[str, bool]):
    fp = sum(v for v in normal_dec.values())
    tn = sum(not v for v in normal_dec.values())
    tp = sum(v for v in abnormal_dec.values())
    fn = sum(not v for v in abnormal_dec.values())
    return dict(tp=tp, fn=fn, fp=fp, tn=tn)

def score_from_conf(conf, fp_weight=2.0):
    tp, fn, fp, tn = conf["tp"], conf["fn"], conf["fp"], conf["tn"]
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    return recall - fp_weight * fpr

# =========================================================
# 工具：log-uniform 采样（非常关键）
# =========================================================

def log_uniform(rnd: random.Random, low: float, high: float) -> float:
    return math.exp(rnd.uniform(math.log(low), math.log(high)))

# =========================================================
# random search（真正“广 + 细”）
# =========================================================

def random_search_mean_up_zscore(
    checkpoints_dir: str,
    n_trials: int = 20000,
    seed: int = 42,
    fp_weight: float = 2.0,
    topk: int = 30,
    print_every: int = 200,
    save_every: int = 200,
):
    rnd = random.Random(seed)
    df_n, df_a = load_latest_pair(checkpoints_dir)

    rows = []
    best = -1e9
    out_path = os.path.join(checkpoints_dir, "hyperopt_mean_up_zscore_partial.csv")

    print(f"[SEARCH] mean_up_zscore trials={n_trials}")

    for i in range(1, n_trials + 1):

        # ===============================
        # ⭐ 广 + 细 的参数采样（核心）
        # ===============================
        baseline_window = int(log_uniform(rnd, 40, 400))
        recent_window   = int(log_uniform(rnd, 6, min(120, baseline_window // 2)))

        params = dict(
            baseline_window = baseline_window,
            recent_window   = recent_window,

            z_thresh        = rnd.uniform(0.5, 8.0),     # 连续
            ratio_thresh    = rnd.uniform(1.01, 3.0),    # 连续
            min_hits        = rnd.randint(1, max(2, recent_window // 4)),
        )

        nd = per_tail_decisions(df_n, params)
        ad = per_tail_decisions(df_a, params)

        conf = confusion(nd, ad)
        score = score_from_conf(conf, fp_weight)

        rows.append(dict(score=score, **conf, **params))

        if score > best:
            best = score

        # ---------- 中间写盘 ----------
        if i % save_every == 0:
            pd.DataFrame(rows).to_csv(out_path, index=False)
            print(f"[SAVE] checkpoint saved @ trial {i}")

        if i == 1 or i % print_every == 0:
            print(
                f"[{i:5d}/{n_trials}] "
                f"score={score:.4f} best={best:.4f} "
                f"(tp={conf['tp']} fp={conf['fp']} fn={conf['fn']}) "
                f"bw={baseline_window} rw={recent_window}"
            )

    # 最终结果
    final_path = os.path.join(checkpoints_dir, "hyperopt_mean_up_zscore.csv")
    pd.DataFrame(rows).to_csv(final_path, index=False)

    return (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .head(topk)
    )

# =========================================================
# main
# =========================================================

if __name__ == "__main__":
    CHECKPOINT_DIR = "./checkpoints/timerxl_reg_train2m_test1m_gap6m_PACK_DISCH_T_PACK2"

    df = random_search_mean_up_zscore(
        CHECKPOINT_DIR,
        n_trials=30000,     # ⭐ 一晚上量级
        save_every=200,
        print_every=200,
    )

    out = os.path.join(CHECKPOINT_DIR, "hyperopt_mean_up_zscore.csv")
    df.to_csv(out, index=False)

    print("\n[TOP RESULTS]")
    print(df.head(10))
    print("\nSaved to:", out)
