# -*- coding: utf-8 -*-
"""
可视化 B777 ACM（PACK）正常与故障数据
- 从 IoTDB 拉数（按给定机号清单、参数清单与时间窗）
- 生成可查看的合并数据集（CSV）
- 画分布对比 / 时序对比 / 相关性热力图 / PCA 散点图
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from iotdb.table_session import TableSession, TableSessionConfig

# -----------------------------
# 你的飞机清单与参数清单（原样沿用）
# -----------------------------

train_left_ac_normal = [
    "B-2008","B-2007","B-2029","B-2009","B-20DM","B-226D","B-226C","B-20C5","B-2080","B-2081",
    "B-2042","B-223N","B-2041","B-209Y","B-2048","B-2026","B-2028","B-2027","B-2049","B-20CK",
    "B-20EM","B-222W","B-20EN","B-223G","B-7185",
]
train_right_ac_normal = [
    "B-2008","B-2007","B-2029","B-2009","B-20DM","B-226D","B-226C","B-20C5","B-2081",
    "B-2042","B-223N","B-2041","B-209Y","B-2048","B-2026","B-2028","B-2027","B-2049","B-20CK",
    "B-20EM","B-222W","B-20EN","B-223G","B-7185",
]
val_left_ac_normal  = ["B-2099", "B-2010", "B-20AC", "B-7588"]
val_right_ac_normal = ["B-2099", "B-2010", "B-20AC", "B-7588"]
test_left_normal    = ["B-7183", "B-2073", "B-2072", "B-2075"]
test_right_normal   = ["B-7183", "B-2073", "B-2072", "B-2075"]
test_left_abnormal  = []
test_right_abnormal = ["B-2080"]

# 左/右参数
para_l = [
    "POSRAIL","TOUTPHXL","TOUTSHXL","TINPHXL","TOUTCPRSRL","POSRAEL",
    "TINTURB2L","POSTBVL","TINCONDL","ECVCLOSED_L","POSLVLVL","TOUTPACKL",
]
para_r = [
    "POSRAIR","TOUTPHXR","TOUTSHXR","TINPHXR","TOUTCPRSRR","POSRAER",
    "TINTURB2R","POSTBVR","TINCONDR","ECVCLOSED_R","POSLVLVR","TOUTPACKR",
]

# 时间窗（与你给的一致）
time_start = '2025-04-01'
time_end   = '2025-04-30'

# 数据库/时区（与你给的一致）
DB_NAME = "b777"
TIMEZONE = "UTC+8"

# 输出目录
OUT_DIR = "outputs"
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
# 核心：按机号清单+参数清单取一侧的数据（对齐插值）
# 返回：每个机号一段长表，列：index(时间序号)、tail、side、param...
# -----------------------------
def fetch_side_dataframe(session, tail_list, para_list, side_tag, time_start, time_end):
    all_flights = []

    for tail in tqdm(tail_list, desc=f"Fetching {side_tag} side"):
        # 逐个参数查，再在行维度对齐
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
                # 该参数在该机号此时间窗无数据，跳过
                continue

            # 只留一列并改名为参数名（去 IoTDB 默认列名 value）
            df.columns = [param]

            if side_df.empty:
                side_df = df.copy()
            else:
                # 与当前累积的 side_df 做长度对齐（线性插值）
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
                    # 拼上当前参数
                    interp_df[param] = df[param].values
                    side_df = interp_df

        if side_df.empty:
            continue

        # 加标签列
        side_df = side_df.reset_index(drop=True)
        side_df["tail"] = tail
        side_df["side"] = side_tag
        # 用递增序号代表时间步（若需要真实时间，可从 IoTDB 的时间列再取）
        side_df["t_idx"] = np.arange(len(side_df))

        all_flights.append(side_df)

    if not all_flights:
        return pd.DataFrame()

    return pd.concat(all_flights, ignore_index=True)


# -----------------------------
# 组装：正常/故障 两类数据 + 左/右两侧
# 返回：长表（含 columns：tail, side, t_idx, 各参数... , label）
# -----------------------------
def build_dataset(session):
    # 正常集合（这里示例用 test_normal，你也可换成 train/val）
    normal_left  = fetch_side_dataframe(session, test_left_normal,  para_l, "L", time_start, time_end)
    normal_right = fetch_side_dataframe(session, test_right_normal, para_r, "R", time_start, time_end)
    normal_df = pd.concat([normal_left, normal_right], ignore_index=True)
    if not normal_df.empty:
        normal_df["label"] = "normal"

    # 故障集合（仅示例右侧异常机号）
    ab_left  = fetch_side_dataframe(session, test_left_abnormal,  para_l, "L", time_start, time_end)
    ab_right = fetch_side_dataframe(session, test_right_abnormal, para_r, "R", time_start, time_end)
    ab_df = pd.concat([ab_left, ab_right], ignore_index=True)
    if not ab_df.empty:
        ab_df["label"] = "abnormal"

    full_df = pd.concat([normal_df, ab_df], ignore_index=True)

    # 保证列顺序（先标签列，后特征）
    meta_cols = ["tail", "side", "t_idx", "label"]
    feat_cols = [c for c in full_df.columns if c not in meta_cols]
    # 将 meta 放前面
    full_df = full_df[meta_cols + feat_cols]

    return full_df


# -----------------------------
# 可视化：分布对比（直方图）
# -----------------------------
def plot_histogram_compare(df, columns, bins=50, save_prefix="hist"):
    os.makedirs(OUT_DIR, exist_ok=True)
    for col in columns:
        if col not in df.columns:
            continue
        sub = df[[col, "label"]].dropna()
        if sub.empty:
            continue

        plt.figure(figsize=(6,4))
        # 正常/故障分布
        for lbl, alpha in [("normal", 0.6), ("abnormal", 0.6)]:
            if lbl in sub["label"].unique():
                vals = sub[sub["label"] == lbl][col].values
                plt.hist(vals, bins=bins, alpha=alpha, density=True, label=lbl)
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.title(f"Distribution: {col} (normal vs abnormal)")
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"{save_prefix}_{col}.png")
        plt.savefig(out, dpi=150)
        plt.close()


# -----------------------------
# 可视化：随机选一个机号/侧别，画时序对比（同一参数）
# -----------------------------
import random
def plot_timeseries_compare(df, param, save_prefix="timeseries", max_points=2000):
    os.makedirs(OUT_DIR, exist_ok=True)
    # 仅保留存在该参数的数据
    sub = df[["tail","side","t_idx","label",param]].dropna()
    if sub.empty:
        return

    # 选一个机号与侧别，使得正常/故障都存在（若没有，就各自随机选一个）
    # 先尝试同 tail/side
    pairs = sub.groupby(["tail","side","label"]).size().reset_index()[["tail","side","label"]]
    normal_pairs   = set(map(tuple, pairs[pairs["label"]=="normal"][["tail","side"]].values))
    abnormal_pairs = set(map(tuple, pairs[pairs["label"]=="abnormal"][["tail","side"]].values))
    common_pairs = list(normal_pairs & abnormal_pairs)

    if common_pairs:
        tail, side = random.choice(common_pairs)
        n_series = sub[(sub["tail"]==tail)&(sub["side"]==side)&(sub["label"]=="normal")].sort_values("t_idx")
        a_series = sub[(sub["tail"]==tail)&(sub["side"]==side)&(sub["label"]=="abnormal")].sort_values("t_idx")
        title_extra = f"tail={tail}, side={side}"
    else:
        # 退而求其次：各自随机一个样本
        n_grp = sub[sub["label"]=="normal"].groupby(["tail","side"]).size()
        a_grp = sub[sub["label"]=="abnormal"].groupby(["tail","side"]).size()
        if n_grp.empty or a_grp.empty:
            return
        tail_n, side_n = random.choice(list(n_grp.index))
        tail_a, side_a = random.choice(list(a_grp.index))
        n_series = sub[(sub["tail"]==tail_n)&(sub["side"]==side_n)&(sub["label"]=="normal")].sort_values("t_idx")
        a_series = sub[(sub["tail"]==tail_a)&(sub["side"]==side_a)&(sub["label"]=="abnormal")].sort_values("t_idx")
        title_extra = f"normal:({tail_n},{side_n}) vs abnormal:({tail_a},{side_a})"

    # 下采样防止点过多
    def downsample(df_ser):
        if len(df_ser) <= max_points:
            return df_ser
        idx = np.linspace(0, len(df_ser)-1, max_points).astype(int)
        return df_ser.iloc[idx]

    n_series = downsample(n_series)
    a_series = downsample(a_series)

    plt.figure(figsize=(8,4))
    plt.plot(n_series["t_idx"].values, n_series[param].values, label="normal")
    if not a_series.empty:
        plt.plot(a_series["t_idx"].values, a_series[param].values, label="abnormal")
    plt.xlabel("t_idx")
    plt.ylabel(param)
    plt.title(f"Time Series Compare: {param} | {title_extra}")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{save_prefix}_{param}.png")
    plt.savefig(out, dpi=150)
    plt.close()


# -----------------------------
# 可视化：相关性热力图（分别对正常、故障）
# -----------------------------
def plot_correlation_heatmap(df, side=None, save_prefix="corr_heatmap", vmax=1.0):
    os.makedirs(OUT_DIR, exist_ok=True)
    # 仅数值列
    meta_cols = ["tail","side","t_idx","label"]
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["t_idx"]]
    if not feat_cols:
        return

    for lbl in df["label"].dropna().unique():
        sub = df[df["label"]==lbl]
        if side is not None:
            sub = sub[sub["side"]==side]
            side_tag = f"_{side}"
        else:
            side_tag = ""

        if sub.empty:
            continue

        # 只用该侧存在的列
        used_cols = [c for c in feat_cols if c in sub.columns]
        sub_num = sub[used_cols].dropna(axis=1, how="all").fillna(method="ffill").fillna(method="bfill")
        if sub_num.shape[1] < 2:
            continue

        corr = np.corrcoef(sub_num.values, rowvar=False)
        labels = sub_num.columns.tolist()

        plt.figure(figsize=(max(6, 0.4*len(labels)), max(5, 0.4*len(labels))))
        im = plt.imshow(corr, vmin=-vmax, vmax=vmax, aspect='auto')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.title(f"Correlation ({lbl}{side_tag})")
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"{save_prefix}_{lbl}{side_tag}.png")
        plt.savefig(out, dpi=150)
        plt.close()


# -----------------------------
# 可视化：PCA 二维散点（正常 vs 故障）
# -----------------------------
def plot_pca_scatter(df, save_name="pca_scatter.png"):
    # 仅数值列（去掉 t_idx 影响）
    feat_cols = [c for c in df.columns if c not in ["tail","side","t_idx","label"]]
    X = df[feat_cols].copy()
    X = X.replace([np.inf,-np.inf], np.nan).dropna(axis=1, how="all")
    if X.empty or X.shape[1] < 2:
        return
    # 简单填补
    X = X.fillna(method="ffill").fillna(method="bfill")
    X = X.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)

    plt.figure(figsize=(6,5))
    labels = df["label"].values
    for lbl in ["normal","abnormal"]:
        mask = (labels == lbl)
        if mask.sum() == 0:
            continue
        plt.scatter(Xp[mask,0], Xp[mask,1], s=8, alpha=0.6, label=lbl)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA: normal vs abnormal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, save_name), dpi=150)
    plt.close()


# -----------------------------
# 主流程
# -----------------------------
def main():
    print("[1/5] Connecting IoTDB ...")
    session = connect_session()

    print("[2/5] Building dataset ...")
    df = build_dataset(session)
    if df.empty:
        print("No data fetched. Please check tails/params/time window.")
        return

    # 导出可查看的 CSV
    csv_path = os.path.join(OUT_DIR, "acm_samples.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] {csv_path}")

    # 简要统计
    print("[3/5] Summary:")
    print(df[["label","side","tail"]].drop_duplicates().groupby(["label","side"]).size())

    # 选择要可视化的关键参数（存在于数据中的才会画）
    key_params = [
        "TOUTPACKL","TOUTPACKR",
        "TOUTCPRSRL","TOUTCPRSRR",
        "TOUTPHXL","TOUTPHXR",
        "TOUTSHXL","TOUTSHXR",
        "TINPHXL","TINPHXR",
        "TINTURB2L","TINTURB2R",
        "POSRAIL","POSRAIR",
        "POSRAEL","POSRAER",
        "POSTBVL","POSTBVR",
        "TINCONDL","TINCONDR",
        "ECVCLOSED_L","ECVCLOSED_R",
        "POSLVLVL","POSLVLVR",
    ]
    existing_keys = [k for k in key_params if k in df.columns]

    print("[4/5] Plotting histograms / timeseries ...")
    # 分布对比
    plot_histogram_compare(df, existing_keys, bins=50, save_prefix="hist")

    # 时序对比（仅挑几项温度/阀位）
    ts_candidates = [k for k in existing_keys if k.startswith(("TOUT","TIN","POS"))]
    for param in ts_candidates[:8]:  # 限制数量，避免生成过多图
        plot_timeseries_compare(df, param, save_prefix="timeseries", max_points=2000)

    print("[5/5] Plotting correlations / PCA ...")
    # 相关性热力图（总体、以及分侧）
    plot_correlation_heatmap(df, side=None, save_prefix="corr_heatmap_all")
    for side in ["L","R"]:
        plot_correlation_heatmap(df, side=side, save_prefix=f"corr_heatmap_side")

    # PCA 散点
    plot_pca_scatter(df, save_name="pca_scatter.png")

    print("All done. Check the 'outputs/' folder for CSV and figures.")


if __name__ == "__main__":
    main()
