# -*- coding: utf-8 -*-
"""
示例：只取 B-2080 左侧作为 normal，右侧作为 abnormal
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from iotdb.table_session import TableSession, TableSessionConfig

# -----------------------------
# 参数设置
# -----------------------------
para_l = [
    "POSRAIL","TOUTPHXL","TOUTSHXL","TINPHXL","TOUTCPRSRL","POSRAEL",
    "TINTURB2L","POSTBVL","TINCONDL","ECVCLOSED_L","POSLVLVL","TOUTPACKL",
]
para_r = [
    "POSRAIR","TOUTPHXR","TOUTSHXR","TINPHXR","TOUTCPRSRR","POSRAER",
    "TINTURB2R","POSTBVR","TINCONDR","ECVCLOSED_R","POSLVLVR","TOUTPACKR",
]

time_start = '2025-04-01'
time_end   = '2025-04-30'
DB_NAME    = "b777"
TIMEZONE   = "UTC+8"
OUT_DIR    = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# IoTDB 连接
# -----------------------------
def connect_session():
    try:
        cfg = TableSessionConfig(
            node_urls=["127.0.0.1:6667"],
            username="root",
            password="root",
            time_zone=TIMEZONE
        )
        session = TableSession(cfg)
    except Exception:
        cfg = TableSessionConfig(
            node_urls=["10.254.43.34:6667"],
            username="root",
            password="root",
            time_zone=TIMEZONE
        )
        session = TableSession(cfg)

    session.execute_non_query_statement(f"USE {DB_NAME}")
    return session

# -----------------------------
# 抓取单侧数据（原样复用）
# -----------------------------
def fetch_side_dataframe(session, tail_list, para_list, side_tag, time_start, time_end):
    all_flights = []

    for tail in tqdm(tail_list, desc=f"Fetching {side_tag} side"):
        side_df = pd.DataFrame()
        for param in para_list:
            query = f"""
            SELECT value
            FROM {param}
            WHERE "aircraft/tail" = '{tail}'
            AND TIME >= {time_start} AND TIME < {time_end}
            """
            try:
                df = session.execute_query_statement(query).todf()
            except Exception as e:
                print(f"[WARN] Query failed: {e}")
                df = pd.DataFrame()

            if df.empty:
                continue

            df.columns = [param]

            if side_df.empty:
                side_df = df.copy()
            else:
                if len(side_df) >= len(df):
                    original_index = np.arange(len(df))
                    new_index      = np.linspace(0, len(df) - 1, len(side_df))
                    interp_vals    = np.interp(new_index, original_index, df[param].values)
                    side_df[param] = interp_vals
                else:
                    original_index = np.arange(len(side_df))
                    new_index      = np.linspace(0, len(side_df) - 1, len(df))
                    interp_df      = pd.DataFrame(index=df.index)
                    for col in side_df.columns:
                        interp_vals = np.interp(new_index, original_index, side_df[col].values)
                        interp_df[col] = interp_vals
                    interp_df[param] = df[param].values
                    side_df = interp_df

        if side_df.empty:
            continue

        side_df = side_df.reset_index(drop=True)
        side_df["tail"] = tail
        side_df["side"] = side_tag
        side_df["t_idx"] = np.arange(len(side_df))

        all_flights.append(side_df)

    if not all_flights:
        return pd.DataFrame()

    return pd.concat(all_flights, ignore_index=True)

# -----------------------------
# 特殊：B-2080 左=normal, 右=abnormal
# -----------------------------
def build_dataset_b2080_LR(session):
    normal_left = fetch_side_dataframe(
        session, ["B-2080"], para_l, "L", time_start, time_end
    )
    if not normal_left.empty:
        normal_left["label"] = "normal"

    abnormal_right = fetch_side_dataframe(
        session, ["B-2080"], para_r, "R", time_start, time_end
    )
    if not abnormal_right.empty:
        abnormal_right["label"] = "abnormal"

    full_df = pd.concat([normal_left, abnormal_right], ignore_index=True)

    meta_cols = ["tail", "side", "t_idx", "label"]
    feat_cols = [c for c in full_df.columns if c not in meta_cols]
    return full_df[meta_cols + feat_cols]

# -----------------------------
# 直方图对比
# -----------------------------
def plot_histogram_compare(df, columns, bins=50, save_prefix="hist"):
    for col in columns:
        if col not in df.columns: continue
        sub = df[[col, "label"]].dropna()
        if sub.empty: continue

        plt.figure(figsize=(6,4))
        for lbl, alpha in [("normal", 0.6), ("abnormal", 0.6)]:
            if lbl in sub["label"].unique():
                vals = sub[sub["label"] == lbl][col].values
                plt.hist(vals, bins=bins, alpha=alpha, density=True, label=lbl)
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.title(f"Distribution: {col} (B-2080 L vs R)")
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"{save_prefix}_{col}.png")
        plt.savefig(out, dpi=150)
        plt.close()

def infer_lr_pairs(para_l, para_r):
    """
    根据列名末尾的 L/R 推断成对的 (lcol, rcol, base) 列名元组列表
    例：TOUTCPRSRL / TOUTCPRSRR -> ("TOUTCPRSRL", "TOUTCPRSRR", "TOUTCPRSR")
    """
    def base_name(s):
        # 去掉末尾 1 个字母（L/R）
        return s[:-1] if s and s[-1] in ("L","R") else s

    bases_l = {base_name(c): c for c in para_l if c and c[-1] == "L"}
    bases_r = {base_name(c): c for c in para_r if c and c[-1] == "R"}
    common  = sorted(set(bases_l.keys()) & set(bases_r.keys()))
    return [(bases_l[b], bases_r[b], b) for b in common]

def plot_histogram_LR_by_pairs(df, lr_pairs, bins=50, save_prefix="b2080_pairhist"):
    """
    对于每个 (lcol, rcol, base)：
      - 同一张图叠加 L 列与 R 列的直方图
      - 文件名含底名 base
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    for lcol, rcol, base in lr_pairs:
        if lcol not in df.columns and rcol not in df.columns:
            continue

        # 取出数据并去空
        lvals = df[lcol].dropna().values if lcol in df.columns else np.array([])
        rvals = df[rcol].dropna().values if rcol in df.columns else np.array([])
        if lvals.size == 0 and rvals.size == 0:
            continue

        # 统一 bin 边界，避免两侧尺度不一致
        all_vals = np.concatenate([lvals, rvals]) if rvals.size else lvals
        if all_vals.size == 0:
            continue
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        if vmin == vmax:  # 常数列保护
            vmax = vmin + 1e-6
        bin_edges = np.linspace(vmin, vmax, bins + 1)

        plt.figure(figsize=(6,4))
        if lvals.size:
            plt.hist(lvals, bins=bin_edges, alpha=0.55, density=True,
                     label="L (normal)", color="steelblue")
        if rvals.size:
            plt.hist(rvals, bins=bin_edges, alpha=0.55, density=True,
                     label="R (abnormal)", color="orangered")

        plt.xlabel(base)  # 用底名作横轴标题，更直观
        plt.ylabel("Density")
        plt.title(f"{base} : B-2080 L vs R")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f"{save_prefix}_{base}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"已输出：{out_path}")

# -----------------------------
# B-2080：每个变量单独一张时序图（原始数据、无平滑、无标准化）
# -----------------------------
def plot_timeseries_LR_separate(
    df,
    lr_pairs,
    save_prefix="b2080_timeseries",   # 文件名前缀
    out_dir=OUT_DIR
):
    os.makedirs(out_dir, exist_ok=True)

    # 基础校验
    if "t_idx" not in df.columns:
        raise ValueError("数据缺少 t_idx，请确认对齐流程已执行。")

    # 拆分 L / R
    df_L = df[df["side"] == "L"].reset_index(drop=True)
    df_R = df[df["side"] == "R"].reset_index(drop=True)
    if df_L.empty and df_R.empty:
        print("[WARN] 没有 L 或 R 侧数据可画")
        return

    # 使用共同的长度（以较短者为准），保持简单
    n = min(len(df_L), len(df_R)) if (not df_L.empty and not df_R.empty) else \
        (len(df_L) if not df_L.empty else len(df_R))
    if n == 0:
        print("[WARN] L/R 长度为 0，跳过绘图")
        return
    t = np.arange(n)  # 用对齐后的索引作横轴

    for lcol, rcol, base in lr_pairs:
        has_l = (lcol in df_L.columns) and (len(df_L) > 0)
        has_r = (rcol in df_R.columns) and (len(df_R) > 0)
        if not has_l and not has_r:
            continue

        plt.figure(figsize=(12, 3.2))
        if has_l:
            yL = df_L[lcol].values[:n].astype(float)
            plt.plot(t, yL, label="L (normal)", linewidth=1.0)
        if has_r:
            yR = df_R[rcol].values[:n].astype(float)
            plt.plot(t, yR, label="R (abnormal)", linewidth=1.0)

        plt.title(f"{base} — B-2080 L vs R")
        plt.xlabel("Aligned Time Index (t_idx)")
        plt.ylabel(base)  # 原始量纲，直接用变量底名
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{save_prefix}_{base}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"已输出：{out_path}")
# -----------------------------
# 主流程
# -----------------------------
if __name__ == "__main__":
    session = connect_session()
    df_b2080 = build_dataset_b2080_LR(session)

    # 可选：保存 CSV
    csv_path = os.path.join(OUT_DIR, "b2080_LR.csv")
    df_b2080.to_csv(csv_path, index=False)
    print(f"数据已保存: {csv_path}")

    # 推断 L/R 成对变量
    lr_pairs = infer_lr_pairs(para_l, para_r)

    #直方图
    lr_pairs = infer_lr_pairs(para_l, para_r)
    plot_histogram_LR_by_pairs(df_b2080, lr_pairs, bins=50, save_prefix="b2080_hist_pair")

    # 每个变量一张图（原始数据，简单 L/R 曲线）
    plot_timeseries_LR_separate(
        df_b2080,
        lr_pairs,
        save_prefix="b2080_ts_single"
    )