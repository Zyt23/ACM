# alarm_hyperopt_allinone.py
# -*- coding: utf-8 -*-

import os
import random
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from types import SimpleNamespace


# =========================================================
# 读取 per_flight CSV
# =========================================================

def load_per_flight_csv(path: str) -> pd.DataFrame:
    """
    期望至少有列：tail, flight_mse
    推荐有列：start_time（用于排序）
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    # 兼容列名：flight_mse
    if "flight_mse" not in df.columns:
        for alt in ["mean_mse", "mse_loss", "MSE", "mse", "flight_mse"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "flight_mse"})
                break
    if "flight_mse" not in df.columns:
        raise ValueError(f"'flight_mse' not found in {path}. columns={list(df.columns)}")

    # 兼容列名：tail
    if "tail" not in df.columns:
        for alt in ["tail_id", "aircraft", "plane", "ac"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "tail"})
                break
    if "tail" not in df.columns:
        raise ValueError(f"'tail' not found in {path}. columns={list(df.columns)}")

    df["flight_mse"] = pd.to_numeric(df["flight_mse"], errors="coerce")

    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

    return df


def load_normal_abnormal_pair(checkpoints_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    兼容以下位置：
    1) checkpoints_dir/test_results_aligned_regress/win96/*.csv
    2) checkpoints_dir/test_results_aligned_regress/*.csv
    3) checkpoints_dir/*.csv
    """
    checkpoints_dir = checkpoints_dir.strip()

    candidates = [
        os.path.join(checkpoints_dir, "test_results_aligned_regress", "win96"),
        os.path.join(checkpoints_dir, "test_results_aligned_regress"),
        checkpoints_dir,
    ]

    base = None
    for c in candidates:
        n_path = os.path.join(c, "test_normal_recent_per_flight.csv")
        a_path = os.path.join(c, "test_abnormal_per_flight.csv")
        if os.path.isfile(n_path) and os.path.isfile(a_path):
            base = c
            break

    if base is None:
        raise FileNotFoundError(
            "Cannot locate per_flight csv pair. Expected files:\n"
            "  test_normal_recent_per_flight.csv\n"
            "  test_abnormal_per_flight.csv\n"
            "Searched in:\n" + "\n".join(["  " + x for x in candidates])
        )

    df_n = load_per_flight_csv(os.path.join(base, "test_normal_recent_per_flight.csv"))
    df_a = load_per_flight_csv(os.path.join(base, "test_abnormal_per_flight.csv"))
    return df_n, df_a


# =========================================================
# 4种报警模式
# =========================================================

def alarm_zscore_ratio_hits(
    df: pd.DataFrame,
    loss_col: str = "flight_mse",
    baseline_window: int = 120,
    recent_window: int = 30,
    z_thresh: float = 3.0,
    ratio_thresh: float = 1.3,
    min_hits: int = 5,
):
    """
    点级别：z + ratio + hits
    """
    x = df[loss_col].dropna().values
    if len(x) < baseline_window + recent_window:
        return SimpleNamespace(alarm=False, hits=0, z_max=np.nan, ratio_max=np.nan)

    base = x[-(baseline_window + recent_window):-recent_window]
    recent = x[-recent_window:]

    mu = float(base.mean())
    std = float(base.std()) + 1e-6

    z = (recent - mu) / std
    ratio = recent / max(mu, 1e-6)

    hits = int(np.sum((z >= z_thresh) & (ratio >= ratio_thresh)))

    return SimpleNamespace(
        alarm=hits >= min_hits,
        hits=hits,
        z_max=float(np.max(z)) if len(z) else np.nan,
        ratio_max=float(np.max(ratio)) if len(ratio) else np.nan,
        base_mean=mu,
        recent_mean=float(np.mean(recent)) if len(recent) else np.nan,
    )


def alarm_mean_up_only(
    df: pd.DataFrame,
    loss_col: str = "flight_mse",
    baseline_window: int = 120,
    recent_window: int = 30,
    ratio_thresh: float = 1.3,
):
    """
    只用 mean_up（均值抬升）：recent_mean / baseline_mean >= ratio_thresh
    """
    x = df[loss_col].dropna().values
    if len(x) < baseline_window + recent_window:
        return SimpleNamespace(alarm=False)

    base = x[-(baseline_window + recent_window):-recent_window]
    recent = x[-recent_window:]

    mu = float(base.mean())
    rmu = float(recent.mean())
    ratio_mean = rmu / max(mu, 1e-6)

    return SimpleNamespace(
        alarm=ratio_mean >= ratio_thresh,
        ratio_mean=float(ratio_mean),
        base_mean=mu,
        recent_mean=rmu,
    )


def alarm_mean_baseline_only(
    df: pd.DataFrame,
    loss_col: str = "flight_mse",
    baseline_window: int = 120,
    recent_window: int = 30,
    mean_thresh: float = 0.1,
):
    """
    只用“均值 baseline + 均值差”：
      recent_mean - baseline_mean >= mean_thresh
    """
    x = df[loss_col].dropna().values
    if len(x) < baseline_window + recent_window:
        return SimpleNamespace(alarm=False)

    base = x[-(baseline_window + recent_window):-recent_window]
    recent = x[-recent_window:]

    mu = float(base.mean())
    rmu = float(recent.mean())
    delta = rmu - mu

    return SimpleNamespace(
        alarm=delta >= mean_thresh,
        delta=float(delta),
        base_mean=mu,
        recent_mean=rmu,
    )


def alarm_window_hits_only(
    df: pd.DataFrame,
    loss_col: str = "flight_mse",
    recent_window: int = 30,
    mse_thresh: float = 1.0,
    min_hits: int = 5,
):
    """
    最近 recent_window 个点中：
      hits = count(mse >= mse_thresh)
      alarm = hits >= min_hits
    """
    x = df[loss_col].dropna().values
    if len(x) < recent_window:
        return SimpleNamespace(alarm=False, hits=0)

    recent = x[-recent_window:]
    hits = int(np.sum(recent >= mse_thresh))

    return SimpleNamespace(
        alarm=hits >= min_hits,
        hits=hits,
        mse_thresh=float(mse_thresh),
    )


def alarm_dispatch(df: pd.DataFrame, params: Dict[str, Any]):
    mode = params["mode"]
    if mode == "zscore_ratio_hits":
        return alarm_zscore_ratio_hits(
            df,
            loss_col=params["loss_col"],
            baseline_window=params["baseline_window"],
            recent_window=params["recent_window"],
            z_thresh=params["z_thresh"],
            ratio_thresh=params["ratio_thresh"],
            min_hits=params["min_hits"],
        )
    elif mode == "mean_up_only":
        return alarm_mean_up_only(
            df,
            loss_col=params["loss_col"],
            baseline_window=params["baseline_window"],
            recent_window=params["recent_window"],
            ratio_thresh=params["ratio_thresh"],
        )
    elif mode == "mean_baseline_only":
        return alarm_mean_baseline_only(
            df,
            loss_col=params["loss_col"],
            baseline_window=params["baseline_window"],
            recent_window=params["recent_window"],
            mean_thresh=params["mean_thresh"],
        )
    elif mode == "window_hits_only":
        return alarm_window_hits_only(
            df,
            loss_col=params["loss_col"],
            recent_window=params["recent_window"],
            mse_thresh=params["mse_thresh"],
            min_hits=params["min_hits"],
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =========================================================
# tail 级别决策
# =========================================================

def per_flight_decisions(df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, bool]:
    out = {}

    if "start_time" in df.columns:
        df = df.sort_values(["tail", "start_time"])
    else:
        df = df.copy()

    for tail, g in df.groupby("tail"):
        if pd.isna(tail):
            continue
        r = alarm_dispatch(g, params)
        out[str(tail)] = bool(r.alarm)

    return out


# =========================================================
# 评估指标
# =========================================================

def confusion(normal_dec: Dict[str, bool], abnormal_dec: Dict[str, bool]):
    fp = sum(bool(v) for v in normal_dec.values())
    tn = sum(not bool(v) for v in normal_dec.values())
    tp = sum(bool(v) for v in abnormal_dec.values())
    fn = sum(not bool(v) for v in abnormal_dec.values())
    return dict(tp=tp, fn=fn, fp=fp, tn=tn)


def score_from_conf(conf, fp_weight=2.0):
    tp, fn, fp, tn = conf["tp"], conf["fn"], conf["fp"], conf["tn"]
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    return recall - fp_weight * fpr


# =========================================================
# 工具：log-uniform 采样
# =========================================================

def log_uniform(rnd: random.Random, low: float, high: float) -> float:
    return math.exp(rnd.uniform(math.log(low), math.log(high)))


# =========================================================
# random search（支持 4 模式）
# =========================================================

def random_search(
    checkpoints_dir: str,
    mode: str,
    n_trials: int = 20000,
    seed: int = 42,
    fp_weight: float = 2.0,
    topk: int = 30,
    print_every: int = 200,
    save_every: int = 200,
):
    rnd = random.Random(seed)
    df_n, df_a = load_normal_abnormal_pair(checkpoints_dir)

    rows = []
    best = -1e9

    out_partial = os.path.join(checkpoints_dir, f"hyperopt_{mode}_partial.csv")
    out_final   = os.path.join(checkpoints_dir, f"hyperopt_{mode}_all.csv")

    print(f"[SEARCH] mode={mode} trials={n_trials}")
    print(f"[DATA] normal rows={len(df_n)} abnormal rows={len(df_a)}")

    for i in range(1, n_trials + 1):
        baseline_window = int(log_uniform(rnd, 40, 400))
        recent_window   = int(log_uniform(rnd, 6, min(120, max(6, baseline_window // 2))))

        params = {
            "mode": mode,
            "loss_col": "flight_mse",  # ✅ 固定正确列名
            "baseline_window": baseline_window,
            "recent_window": recent_window,
        }

        if mode == "zscore_ratio_hits":
            params.update({
                "z_thresh": rnd.uniform(0.5, 8.0),
                "ratio_thresh": rnd.uniform(1.01, 3.0),
                "min_hits": rnd.randint(5, max(5, recent_window // 4)),  # ✅ >=5
            })

        elif mode == "mean_up_only":
            params.update({
                "ratio_thresh": rnd.uniform(1.01, 3.0),
            })

        elif mode == "mean_baseline_only":
            # 均值差阈值：对数采样，范围可按你 mse 量级调整
            params.update({
                "mean_thresh": log_uniform(rnd, 1e-4, 5.0),
            })

        elif mode == "window_hits_only":
            # 绝对阈值：对数采样（长尾更稳）
            params.update({
                "mse_thresh": log_uniform(rnd, 1e-4, 10.0),
                "min_hits": rnd.randint(5, max(5, recent_window // 3)),  # ✅ >=5
            })

        else:
            raise ValueError(f"Unknown mode: {mode}")

        nd = per_flight_decisions(df_n, params)
        ad = per_flight_decisions(df_a, params)

        conf = confusion(nd, ad)
        score = score_from_conf(conf, fp_weight)

        rows.append(dict(score=score, **conf, **params))

        if score > best:
            best = score

        if i % save_every == 0:
            pd.DataFrame(rows).to_csv(out_partial, index=False)
            print(f"[SAVE] partial saved @ trial {i}")

        if i == 1 or i % print_every == 0:
            print(
                f"[{i:5d}/{n_trials}] score={score:.6f} best={best:.6f} "
                f"(tp={conf['tp']} fp={conf['fp']} fn={conf['fn']} tn={conf['tn']}) "
                f"bw={baseline_window} rw={recent_window}"
            )

    df_all = pd.DataFrame(rows)
    df_all.to_csv(out_final, index=False)

    df_top = df_all.sort_values("score", ascending=False).head(topk)
    out_top = os.path.join(checkpoints_dir, f"hyperopt_{mode}_topk.csv")
    df_top.to_csv(out_top, index=False)

    return df_top


# =========================================================
# main
# =========================================================

if __name__ == "__main__":
    CHECKPOINT_DIR = "./checkpoints/timerxl_aligned_regress_keep5x96_raw12m_train10m_test1m_gap6m_2025-08-01end_noALTSTD_win96_stride96_PACK2"

    # 你的初始值（用于 zscore_ratio_hits）
    init_params = dict(
        mode="zscore_ratio_hits",
        loss_col="flight_mse",
        baseline_window=245,
        recent_window=93,
        z_thresh=0.854325,
        ratio_thresh=1.433580,
        min_hits=22,
    )

    df_n, df_a = load_normal_abnormal_pair(CHECKPOINT_DIR)
    nd0 = per_flight_decisions(df_n, init_params)
    ad0 = per_flight_decisions(df_a, init_params)
    conf0 = confusion(nd0, ad0)

    print("\n[INIT PARAMS EVAL]")
    print(init_params)
    print("[CONFUSION]", conf0)

    # 四种模式都跑（按需删）
    for mode in [ "window_hits_only"]:
        df_top = random_search(
            CHECKPOINT_DIR,
            mode=mode,
            n_trials=30000,
            save_every=200,
            print_every=200,
            fp_weight=2.0,
            topk=30,
        )

        print(f"\n[TOP RESULTS] mode={mode}")
        print(df_top.head(10))
        print(f"[SAVED] hyperopt_{mode}_topk.csv (in {CHECKPOINT_DIR})")
