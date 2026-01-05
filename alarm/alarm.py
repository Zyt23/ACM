# alarm_from_latest_csv.py
# -*- coding: utf-8 -*-

import os
import glob
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from datetime import datetime


# =============================
# 0) 基础工具：读最新 metrics
# =============================
def find_latest_metrics_csv(root_dir: str, pattern: str = "*_metrics_*.csv") -> str:
    """
    在 root_dir 及其子目录中搜索符合 pattern 的 csv，返回 mtime 最新的那个。
    例：pattern="test_abnormal_metrics_*.csv" 或 "*_metrics_*.csv"
    """
    files = glob.glob(os.path.join(root_dir, "**", pattern), recursive=True)
    if not files:
        raise FileNotFoundError(f"No csv found under {root_dir} with pattern={pattern}")
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def load_metrics_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    if "tail" not in df.columns:
        df["tail"] = None
    return df


# =============================
# 1) 统一结果结构
# =============================
@dataclass
class AlarmResult:
    alarm: bool
    score: Optional[float]
    first_alarm_time: Optional[pd.Timestamp]
    detail: Dict[str, Any]


# =============================
# 2) 报警方法1：高值倍数法（推荐）
# =============================
def alarm_high_value(
    df: pd.DataFrame,
    loss_col: str = "mae",
    time_col: str = "start_time",
    baseline_window: int = 100,
    lookback: int = 30,
    high_ratio: float = 2.0,
    min_hits: int = 5,
) -> AlarmResult:
    """
    用“基线*倍数”判定高值报警：
      - 基线 = 过去 baseline_window 个点的中位数（更抗尖峰）
      - 在最近 lookback 点里，超过 baseline*high_ratio 的次数 >= min_hits -> 报警
    """
    x = df[[time_col, loss_col]].copy()
    x = x.dropna(subset=[loss_col]).sort_values(time_col).reset_index(drop=True)

    if len(x) < max(20, min_hits + 5):
        return AlarmResult(False, None, None, {"reason": "too_few_points", "n": len(x)})

    # 基线：用“除去最后 lookback 点”后的历史（避免把异常段也算进基线）
    hist = x.iloc[: max(0, len(x) - lookback)]
    if len(hist) < 10:
        hist = x.iloc[: max(1, len(x) // 2)]

    base_series = hist[loss_col].tail(baseline_window)
    baseline = float(np.nanmedian(base_series.values))
    baseline = max(baseline, 1e-8)

    recent = x.tail(lookback).copy()
    thr = baseline * high_ratio
    hits = recent[recent[loss_col] > thr]

    alarm = len(hits) >= min_hits
    first_time = pd.to_datetime(hits[time_col].iloc[0], errors="coerce") if alarm else None

    # score：最近 lookback 点里，超过阈值的最大倍数
    score = float((recent[loss_col].max() / baseline)) if len(recent) else None

    return AlarmResult(
        alarm=alarm,
        score=score,
        first_alarm_time=first_time,
        detail={
            "baseline_median": baseline,
            "threshold": thr,
            "lookback": lookback,
            "min_hits": min_hits,
            "hits": int(len(hits)),
            "max_recent": float(recent[loss_col].max()),
            "high_ratio": high_ratio,
        },
    )


# =============================
# 3) 报警方法2：mean_up（放宽版）
# =============================
def alarm_mean_up(
    df: pd.DataFrame,
    loss_col: str = "mse",
    time_col: str = "start_time",
    smooth_window: int = 25,
    diff_stride: int = 5,
    diff_eps: float = 0.0,
    length: int = 6,
    end_point_idx: int = 100,
    meanup_threshold: float = 1.3,
) -> AlarmResult:
    """
    mean_up 放宽版：
      1) mean = 滑窗均值(smooth_window)
      2) mean_diff = mean - mean.shift(diff_stride)
      3) 找连续 length 个点满足 mean_diff >= -diff_eps 的段
      4) 对这些段计算 mean_up = mean / min_mean(回看 end_point_idx 范围)
      5) final_mean_up = 最近一次非空 mean_up
      6) final_mean_up > meanup_threshold -> 报警
    """
    x = df[[time_col, loss_col]].copy()
    x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
    x = x.dropna(subset=[time_col, loss_col]).sort_values(time_col).reset_index(drop=True)

    if len(x) < max(smooth_window + diff_stride + length + 5, 50):
        return AlarmResult(False, None, None, {"reason": "too_few_points", "n": len(x)})

    x["mean"] = x[loss_col].rolling(window=smooth_window, min_periods=smooth_window).mean()
    x["mean_diff"] = x["mean"] - x["mean"].shift(diff_stride)

    x["ok"] = x["mean_diff"] >= (-diff_eps)
    grp = (x["ok"] != x["ok"].shift(1)).cumsum()

    x["mean_up"] = np.nan

    for _, g in x.groupby(grp):
        if not bool(g["ok"].iloc[0]):
            continue
        if len(g) < length:
            continue

        idxs = g.index.tolist()
        first_idx = idxs[0]
        start_idx = max(0, first_idx - end_point_idx)

        hist_means = x.loc[start_idx:first_idx, "mean"].dropna()
        if hist_means.empty:
            continue

        min_mean = float(hist_means.min())
        if min_mean <= 0:
            min_mean = 1e-8

        for i in idxs:
            cur = x.at[i, "mean"]
            if pd.isna(cur):
                continue
            x.at[i, "mean_up"] = float(cur) / min_mean

    valid = x["mean_up"].dropna()
    if valid.empty:
        return AlarmResult(False, None, None, {"reason": "no_trend_segment_triggered"})

    final_mean_up = float(valid.iloc[-1])
    alarm = final_mean_up > meanup_threshold

    first_time = None
    if alarm:
        first_hit_idx = x.index[x["mean_up"] > meanup_threshold]
        if len(first_hit_idx) > 0:
            first_time = x.loc[first_hit_idx[0], time_col]

    return AlarmResult(
        alarm=alarm,
        score=final_mean_up,
        first_alarm_time=first_time,
        detail={
            "smooth_window": smooth_window,
            "diff_stride": diff_stride,
            "diff_eps": diff_eps,
            "length": length,
            "end_point_idx": end_point_idx,
            "meanup_threshold": meanup_threshold,
            "final_mean_up": final_mean_up,
        },
    )


# =============================
# 4) 报警方法3：平均MSE阈值
# =============================
def alarm_mean_mse_threshold(
    df: pd.DataFrame,
    loss_col: str = "mse",
    time_col: str = "start_time",
    mse_threshold: float = 0.6,
    min_points: int = 10,
) -> AlarmResult:
    """
    极简报警：
      - 对该 tail 的 mse 取均值
      - mean(mse) > 0.6 -> 报警
    """
    x = df[[time_col, loss_col]].copy()
    if time_col in x.columns:
        x[time_col] = pd.to_datetime(x[time_col], errors="coerce")
        x = x.dropna(subset=[loss_col]).sort_values(time_col).reset_index(drop=True)
    else:
        x = x.dropna(subset=[loss_col]).reset_index(drop=True)

    if len(x) < min_points:
        return AlarmResult(False, None, None, {"reason": "too_few_points", "n": len(x)})

    mean_mse = float(np.nanmean(x[loss_col].values))
    alarm = mean_mse > float(mse_threshold)

    first_time = None
    if alarm and time_col in x.columns:
        hits = x[x[loss_col] > mse_threshold]
        if len(hits) > 0:
            first_time = pd.to_datetime(hits[time_col].iloc[0], errors="coerce")

    return AlarmResult(
        alarm=alarm,
        score=mean_mse,
        first_alarm_time=first_time,
        detail={
            "mean_mse": mean_mse,
            "threshold": float(mse_threshold),
            "min_points": int(min_points),
            "n": int(len(x)),
        },
    )


# =============================
# 5) 跑最新 normal/abnormal，按 tail 产出三种方法结果
# =============================
def run_alarm_on_latest(
    checkpoints_dir: str,
    normal_pattern: str = "test_normal*_metrics_*.csv",
    abnormal_pattern: str = "test_abnormal*_metrics_*.csv",
    loss_col_high: str = "mae",
    loss_col_meanup: str = "mse",
) -> Dict[str, Dict[str, Dict[str, AlarmResult]]]:
    normal_csv = find_latest_metrics_csv(checkpoints_dir, normal_pattern)
    abnormal_csv = find_latest_metrics_csv(checkpoints_dir, abnormal_pattern)

    df_n = load_metrics_csv(normal_csv)
    df_a = load_metrics_csv(abnormal_csv)

    results = {"normal": {}, "abnormal": {}}

    for name, df_ in [("normal", df_n), ("abnormal", df_a)]:
        for tail, g in df_.groupby("tail"):
            if pd.isna(tail):
                continue

            # 1) high_value（你原来的参数）
            r1 = alarm_high_value(
                g, loss_col=loss_col_high,
                baseline_window=200, lookback=60, high_ratio=2.2, min_hits=3
            )

            # 2) mean_up（你原来的参数）
            r2 = alarm_mean_up(
                g, loss_col=loss_col_meanup,
                smooth_window=35, diff_stride=5, diff_eps=0.0,
                length=6, end_point_idx=100, meanup_threshold=1.3
            )

            # 3) mean mse threshold（新加）
            r3 = alarm_mean_mse_threshold(
                g, loss_col=loss_col_meanup,
                mse_threshold=0.6,
                min_points=10,
            )

            results[name][str(tail)] = {
                "high_value": r1,
                "mean_up": r2,
                "mean_mse": r3,
            }

    print("Normal latest csv:", normal_csv)
    print("Abnormal latest csv:", abnormal_csv)
    return results


# =============================
# 6) 统计 TP/FP/TN/FN + Precision/Recall/F1
# =============================
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def compute_metrics_from_results(results: dict, method_key: str) -> Dict[str, Any]:
    """
    normal=负样本, abnormal=正样本
    method_key: 'high_value' / 'mean_up' / 'mean_mse'
    """
    normal = results.get("normal", {})
    abnormal = results.get("abnormal", {})

    # 对齐 tail 集合（允许某边没有）
    tails = sorted(set(normal.keys()) | set(abnormal.keys()))

    tp = fp = tn = fn = 0
    for t in tails:
        pred_alarm = False

        # 预测：如果在任一 split 里有该 tail，就取对应 split 的方法报警（优先用 abnormal 里的，因为那是正样本评估）
        # 实际上一个 tail 可能同时出现在 normal/abnormal 两边（通常不会），这里做个稳妥选择：
        if t in abnormal:
            pred_alarm = bool(abnormal[t][method_key].alarm)
        elif t in normal:
            pred_alarm = bool(normal[t][method_key].alarm)

        # 真值：tail 在 abnormal 视为正样本，否则在 normal 视为负样本
        is_pos = t in abnormal

        if is_pos and pred_alarm:
            tp += 1
        elif is_pos and (not pred_alarm):
            fn += 1
        elif (not is_pos) and pred_alarm:
            fp += 1
        else:
            tn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return {
        "method": method_key,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pos": tp + fn,
        "n_neg": fp + tn,
    }


def print_metrics_table(metrics_list):
    df = pd.DataFrame(metrics_list).copy()
    show_cols = ["method", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "n_pos", "n_neg"]
    df = df[show_cols]
    # 控制小数显示
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n========== Alarm Metrics Summary (per method) ==========")
    print(df.to_string(index=False))
    print("=======================================================\n")
    return df


# =============================
# 7) 保存：tail明细 + summary行
# =============================
def save_alarm_results_to_csv(
    results: dict,
    save_dir: str,
    prefix: str = "alarm_compare",
):
    os.makedirs(save_dir, exist_ok=True)

    # ---- 1) tail明细 ----
    rows = []
    for tag, tails in results.items():
        for tail, methods in tails.items():
            r_high = methods.get("high_value")
            r_mean = methods.get("mean_up")
            r_mse = methods.get("mean_mse")

            rows.append({
                "tag": tag,
                "tail": tail,

                # high_value
                "alarm_high": bool(r_high.alarm) if r_high else False,
                "score_high": r_high.score if r_high else None,
                "first_alarm_high": r_high.first_alarm_time if r_high else None,

                # mean_up
                "alarm_meanup": bool(r_mean.alarm) if r_mean else False,
                "score_meanup": r_mean.score if r_mean else None,
                "first_alarm_meanup": r_mean.first_alarm_time if r_mean else None,

                # mean_mse threshold
                "alarm_meanmse": bool(r_mse.alarm) if r_mse else False,
                "score_meanmse": r_mse.score if r_mse else None,
                "first_alarm_meanmse": r_mse.first_alarm_time if r_mse else None,
            })

    df_tail = pd.DataFrame(rows)
    if not df_tail.empty:
        df_tail = df_tail.sort_values(
            ["tag", "alarm_high", "alarm_meanup", "alarm_meanmse", "score_high"],
            ascending=[True, False, False, False, False],
        )

    # ---- 2) summary（TP/FP/TN/FN + F1等）----
    metrics_list = []
    for mk in ["high_value", "mean_up", "mean_mse"]:
        metrics_list.append(compute_metrics_from_results(results, method_key=mk))
    df_sum = pd.DataFrame(metrics_list)

    # 把 summary 也写进同一个 CSV：用 tag="__summary__"
    # 这样你打开一个文件就能看 tail 明细 + 总体指标
    sum_rows = []
    for _, r in df_sum.iterrows():
        sum_rows.append({
            "tag": "__summary__",
            "tail": r["method"],

            "alarm_high": None,
            "score_high": None,
            "first_alarm_high": None,

            "alarm_meanup": None,
            "score_meanup": None,
            "first_alarm_meanup": None,

            "alarm_meanmse": None,
            "score_meanmse": None,
            "first_alarm_meanmse": None,

            # 额外字段（summary专用）
            "tp": int(r["tp"]),
            "fp": int(r["fp"]),
            "tn": int(r["tn"]),
            "fn": int(r["fn"]),
            "precision": float(r["precision"]),
            "recall": float(r["recall"]),
            "f1": float(r["f1"]),
            "n_pos": int(r["n_pos"]),
            "n_neg": int(r["n_neg"]),
        })

    df_sum_rows = pd.DataFrame(sum_rows)

    # 对齐列：把 df_tail 也补上 summary 列（否则 concat 会出现 NaN 列混乱）
    for col in ["tp", "fp", "tn", "fn", "precision", "recall", "f1", "n_pos", "n_neg"]:
        if col not in df_tail.columns:
            df_tail[col] = np.nan

    df_out = pd.concat([df_tail, df_sum_rows], ignore_index=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_dir, f"{prefix}_{ts}.csv")
    df_out.to_csv(out_path, index=False)

    print(f"[OK] Alarm comparison saved to: {out_path}")

    # print summary table
    print_metrics_table(metrics_list)

    return out_path


# =============================
# 8) main
# =============================
if __name__ == "__main__":
    setting_dir = "./checkpoints/timerxl_reg_train2m_test1m_gap6m_PACK_DISCH_T_PACK2"

    results = run_alarm_on_latest(setting_dir)

    save_alarm_results_to_csv(
        results,
        save_dir=setting_dir,
        prefix="alarm_compare",
    )
