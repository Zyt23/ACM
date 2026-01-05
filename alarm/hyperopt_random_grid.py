# hyperopt_random_grid.py
# -*- coding: utf-8 -*-
import os, json, random
import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, Any, List

from alarm_eval_utils import eval_params

def grid_search_high_value(checkpoints_dir: str,
                           score_mode: str = "f1",
                           fp_weight: float = 2.0,
                           topk: int = 20,
                           print_every: int = 20):
    baseline_window_list = [80, 120, 160, 200, 260]
    lookback_list = [20, 30, 40, 60]
    high_ratio_list = [1.6, 1.8, 2.0, 2.2, 2.5, 3.0]
    min_hits_list = [2, 3, 4, 5]

    combos = list(product(
        baseline_window_list,
        lookback_list,
        high_ratio_list,
        min_hits_list
    ))
    total = len(combos)
    print(f"[GRID] high_value total combinations = {total}")

    rows = []
    best_score = -1e9

    for i, (bw, lb, hr, mh) in enumerate(combos, 1):
        params = dict(
            baseline_window=bw,
            lookback=lb,
            high_ratio=hr,
            min_hits=mh,
        )

        score, conf = eval_params(
            checkpoints_dir,
            method="high_value",
            params=params,
            score_mode=score_mode,
            fp_weight=fp_weight,
        )

        rows.append({
            "method": "high_value",
            "score": score,
            **conf,
            **params,
        })

        # -------- 进度打印 --------
        if score > best_score:
            best_score = score

        if i == 1 or i % print_every == 0 or i == total:
            print(
                f"[GRID][{i:4d}/{total}] "
                f"score={score:.4f} best={best_score:.4f} "
                f"bw={bw} lb={lb} hr={hr} mh={mh} "
                f"(tp={conf['tp']} fp={conf['fp']} fn={conf['fn']})"
            )

    df = (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .head(topk)
    )
    print("[GRID] done.")
    return df

def random_search_mean_up(checkpoints_dir: str,
                          n_trials: int = 300,
                          seed: int = 42,
                          score_mode: str = "f1",
                          fp_weight: float = 2.0,
                          topk: int = 20,
                          print_every: int = 20):
    rnd = random.Random(seed)
    rows = []
    best_score = -1e9

    print(f"[RANDOM] mean_up n_trials = {n_trials}")

    for i in range(1, n_trials + 1):
        params = dict(
            smooth_window=rnd.choice([15, 20, 25, 30, 35, 45, 60]),
            diff_stride=rnd.choice([3, 5, 7, 10]),
            diff_eps=rnd.choice([0.0, 0.01, 0.02, 0.05]),
            length=rnd.choice([4, 5, 6, 8, 10]),
            end_point_idx=rnd.choice([60, 80, 100, 150, 200]),
            meanup_threshold=rnd.choice([1.1, 1.2, 1.3, 1.4, 1.5, 1.6]),
        )

        score, conf = eval_params(
            checkpoints_dir,
            method="mean_up",
            params=params,
            score_mode=score_mode,
            fp_weight=fp_weight,
        )

        rows.append({
            "method": "mean_up",
            "score": score,
            **conf,
            **params,
        })

        if score > best_score:
            best_score = score

        if i == 1 or i % print_every == 0 or i == n_trials:
            print(
                f"[RANDOM][{i:4d}/{n_trials}] "
                f"score={score:.4f} best={best_score:.4f} "
                f"(tp={conf['tp']} fp={conf['fp']} fn={conf['fn']}) "
                f"{params}"
            )

    df = (
        pd.DataFrame(rows)
        .sort_values("score", ascending=False)
        .head(topk)
    )
    print("[RANDOM] done.")
    return df


if __name__ == "__main__":
    setting_dir = "./checkpoints/timerxl_reg_train2m_test1m_gap6m_PACK_DISCH_T_PACK2"

    # 你可以换成 "weighted"（更惩罚 FP）或 "recall"
    score_mode = "weighted"
    fp_weight = 2.0

    df1 = grid_search_high_value(setting_dir, score_mode=score_mode, fp_weight=fp_weight, topk=30)
    df2 = random_search_mean_up(setting_dir, n_trials=500, score_mode=score_mode, fp_weight=fp_weight, topk=30)

    out1 = os.path.join(setting_dir, "hyperopt_high_value_grid.csv")
    out2 = os.path.join(setting_dir, "hyperopt_mean_up_random.csv")
    df1.to_csv(out1, index=False)
    df2.to_csv(out2, index=False)

    print("[OK] saved:", out1)
    print("[OK] saved:", out2)
    print("\nTop high_value:\n", df1.head(10))
    print("\nTop mean_up:\n", df2.head(10))
