# -*- coding: utf-8 -*-
"""
示例：只取指定航班号，左侧标记为 L、右侧标记为 R（不再使用 normal/abnormal）
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
TAIL_NO   = "B-2080"   # <<< 只需改这里即可切换航班号
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
        side_df["side"] = side_tag   # 只保留 L/R
        side_df["t_idx"] = np.arange(len(side_df))

        all_flights.append(side_df)

    if not all_flights:
        return pd.DataFrame()

    return pd.concat(all_flights, ignore_index=True)

# -----------------------------
# 数据拼装：指定航班号，L 与 R
# -----------------------------
def build_dataset_LR(session):
    df_left = fetch_side_dataframe(
        session, [TAIL_NO], para_l, "L", time_start, time_end
    )

    df_right = fetch_side_dataframe(
        session, [TAIL_NO], para_r, "R", time_start, time_end
    )

    full_df = pd.concat([df_left, df_right], ignore_index=True)

    meta_cols = ["tail", "side", "t_idx"]
    feat_cols = [c for c in full_df.columns if c not in meta_cols]
    return full_df[meta_cols + feat_cols]

# -----------------------------
# 直方图对比（按 side=L/R）
# -----------------------------
def plot_histogram_compare(df, columns, bins=50, save_prefix="hist"):
    for col in columns:
        if col not in df.columns:
            continue
        sub = df[[col, "side"]].dropna()
        if sub.empty:
            continue

        plt.figure(figsize=(6,4))
        for lbl, alpha in [("L", 0.6), ("R", 0.6)]:
            if lbl in sub["side"].unique():
                vals = sub[sub["side"] == lbl][col].values
                plt.hist(vals, bins=bins, alpha=alpha, density=True, label=lbl)
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend(title="Side")
        plt.title(f"Distribution: {col} ({TAIL_NO} L vs R)")
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"{save_prefix}_{col}.png")
        plt.savefig(out, dpi=150)
        plt.close()

def infer_lr_pairs(para_l, para_r):
    """
    根据列名末尾的 L/R 推断成对的 (lcol, rcol, base) 列名元组列表
    """
    def base_name(s):
        return s[:-1] if s and s[-1] in ("L","R") else s

    bases_l = {base_name(c): c for c in para_l if c and c[-1] == "L"}
    bases_r = {base_name(c): c for c in para_r if c and c[-1] == "R"}
    common  = sorted(set(bases_l.keys()) & set(bases_r.keys()))
    return [(bases_l[b], bases_r[b], b) for b in common]

def plot_histogram_LR_by_pairs(df, lr_pairs, bins=50, save_prefix="pairhist"):
    os.makedirs(OUT_DIR, exist_ok=True)
    for lcol, rcol, base in lr_pairs:
        if lcol not in df.columns and rcol not in df.columns:
            continue

        lvals = df[lcol].dropna().values if lcol in df.columns else np.array([])
        rvals = df[rcol].dropna().values if rcol in df.columns else np.array([])
        if lvals.size == 0 and rvals.size == 0:
            continue

        all_vals = np.concatenate([lvals, rvals]) if rvals.size else lvals
        if all_vals.size == 0:
            continue
        vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
        if vmin == vmax:
            vmax = vmin + 1e-6
        bin_edges = np.linspace(vmin, vmax, bins + 1)

        plt.figure(figsize=(6,4))
        if lvals.size:
            plt.hist(lvals, bins=bin_edges, alpha=0.55, density=True,
                     label="L", color="steelblue")
        if rvals.size:
            plt.hist(rvals, bins=bin_edges, alpha=0.55, density=True,
                     label="R", color="orangered")

        plt.xlabel(base)
        plt.ylabel("Density")
        plt.title(f"{base} : {TAIL_NO} L vs R")
        plt.legend(title="Side")
        plt.tight_layout()

        out_path = os.path.join(OUT_DIR, f"{save_prefix}_{base}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"已输出：{out_path}")

def plot_timeseries_LR_separate(
    df,
    lr_pairs,
    save_prefix="timeseries",   # 文件名前缀
    out_dir=OUT_DIR
):
    os.makedirs(out_dir, exist_ok=True)

    if "t_idx" not in df.columns:
        raise ValueError("数据缺少 t_idx，请确认对齐流程已执行。")

    df_L = df[df["side"] == "L"].reset_index(drop=True)
    df_R = df[df["side"] == "R"].reset_index(drop=True)
    if df_L.empty and df_R.empty:
        print("[WARN] 没有 L 或 R 侧数据可画")
        return

    n = min(len(df_L), len(df_R)) if (not df_L.empty and not df_R.empty) else \
        (len(df_L) if not df_L.empty else len(df_R))
    if n == 0:
        print("[WARN] L/R 长度为 0，跳过绘图")
        return
    t = np.arange(n)

    for lcol, rcol, base in lr_pairs:
        has_l = (lcol in df_L.columns) and (len(df_L) > 0)
        has_r = (rcol in df_R.columns) and (len(df_R) > 0)
        if not has_l and not has_r:
            continue

        plt.figure(figsize=(12, 3.2))
        if has_l:
            yL = df_L[lcol].values[:n].astype(float)
            plt.plot(t, yL, label="L", linewidth=1.0)
        if has_r:
            yR = df_R[rcol].values[:n].astype(float)
            plt.plot(t, yR, label="R", linewidth=1.0)

        plt.title(f"{base} — {TAIL_NO} L vs R")
        plt.xlabel("Aligned Time Index (t_idx)")
        plt.ylabel(base)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Side")
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
    df_LR = build_dataset_LR(session)

    csv_path = os.path.join(OUT_DIR, f"{TAIL_NO}_LR.csv")
    df_LR.to_csv(csv_path, index=False)
    print(f"数据已保存: {csv_path}")

    lr_pairs = infer_lr_pairs(para_l, para_r)

    # 直方图（左右对比）
    plot_histogram_LR_by_pairs(df_LR, lr_pairs, bins=50, save_prefix=f"{TAIL_NO}_hist_pair")

    # 时序图（左右两条曲线）
    plot_timeseries_LR_separate(
        df_LR,
        lr_pairs,
        save_prefix=f"{TAIL_NO}_ts_single"
    )
