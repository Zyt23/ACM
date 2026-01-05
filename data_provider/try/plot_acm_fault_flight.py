#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_acm_fault_flight.py
---------------------------------
按时间戳对齐，分别可视化以下6条参数（整段航班）并输出一张图：
- 右侧：TINPHXR、TOUTPHXR、TOUTCPRSRR
- 左侧：TINPHXL、TOUTPHXL、TOUTCPRSRL

数据库连接方式参考给定示例（优先127.0.0.1:6667，失败则10.254.43.34:6667）。

用法示例：
python plot_acm_fault_flight.py   --tail B-2080   --start "2025-04-01 00:00:00"   --end   "2025-04-30 23:59:59"   --out "fault_B-2080_20250401-20250430.png"

依赖：pandas、numpy、matplotlib、iotdb
"""
import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IoTDB TableSession
from iotdb.table_session import TableSession, TableSessionConfig

RIGHT_PARAMS = ["TINPHXR", "TOUTPHXR", "TOUTCPRSRR"]
LEFT_PARAMS  = ["TINPHXL", "TOUTPHXL", "TOUTCPRSRL"]

ALL_PARAMS = RIGHT_PARAMS + LEFT_PARAMS

def get_session() -> TableSession:
    """Create IoTDB session. Try localhost first, fallback to 10.254.43.34."""
    try:
        config = TableSessionConfig(
            node_urls=["127.0.0.1:6667"],
            username="root",
            password="root",
            time_zone="UTC+8"
        )
        session = TableSession(config)
        # ensure DB
        session.execute_non_query_statement("USE b777")
        return session
    except Exception as e:
        # fallback
        config = TableSessionConfig(
            node_urls=["10.254.43.34:6667"],
            username="root",
            password="root",
            time_zone="UTC+8"
        )
        session = TableSession(config)
        session.execute_non_query_statement("USE b777")
        return session

def fetch_series(session: TableSession, measurement: str, tail: str, t_start: str, t_end: str) -> pd.DataFrame:
    """
    从IoTDB读取某个measurement的 time, value，限定飞机尾号与时间窗。
    返回包含两列：['time', measurement]
    """
    # IoTDB SQL：时间可以用字符串，列名为 time 和 value（value为测量值）
    query = f'''
    SELECT time, value
    FROM {measurement}
    WHERE "aircraft/tail" = '{tail}'
      AND TIME >= {{"{t_start}"}} AND TIME < {{"{t_end}"}}'''
    result = session.execute_query_statement(query)
    df = result.todf() if result is not None else pd.DataFrame(columns=["time", "value"])
    if df.empty:
        return pd.DataFrame(columns=["time", measurement])
    # 统一列名
    df = df.rename(columns={"value": measurement})
    # 保障时间戳为Datetime，并设置为索引
    if not np.issubdtype(df["time"].dtype, np.datetime64):
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.sort_values("time").set_index("time")
    return df[[measurement]]

def align_all(session: TableSession, tail: str, t_start: str, t_end: str, resample_rule: str = None) -> pd.DataFrame:
    """
    拉取6个序列并按时间外连接（对齐）。
    若提供resample_rule（例如 '1S'、'2S'），则会重采样到统一时间格并线性插值。
    """
    frames = []
    for m in ALL_PARAMS:
        df = fetch_series(session, m, tail, t_start, t_end)
        frames.append(df)
    if not any([not f.empty for f in frames]):
        return pd.DataFrame(columns=ALL_PARAMS)

    merged = pd.concat(frames, axis=1, join="outer").sort_index()

    if resample_rule:
        # 仅当采样间隔不一致时使用重采样，避免“锯齿”。
        merged = merged.resample(resample_rule).mean()
        merged = merged.interpolate(method="time", limit_direction="both")

    return merged

def plot_figure(df: pd.DataFrame, title: str, out_path: str):
    if df.empty:
        raise RuntimeError("查询结果为空：没有可绘制的数据。请检查尾号/时间窗/测点。")

    # 6行共享x轴
    nrows = 6
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 12), sharex=True)
    plt.subplots_adjust(hspace=0.25, top=0.92)

    # 右侧
    axes[0].plot(df.index, df["TINPHXR"], linewidth=1)
    axes[0].set_title("TINPHXR (主级热交换器进口温度 - 右)")
    axes[1].plot(df.index, df["TOUTPHXR"], linewidth=1)
    axes[1].set_title("TOUTPHXR (主级热交换器出口温度 - 右)")
    axes[2].plot(df.index, df["TOUTCPRSRR"], linewidth=1)
    axes[2].set_title("TOUTCPRSRR (压气机出口温度 - 右)")

    # 左侧
    axes[3].plot(df.index, df["TINPHXL"], linewidth=1)
    axes[3].set_title("TINPHXL (主级热交换器进口温度 - 左)")
    axes[4].plot(df.index, df["TOUTPHXL"], linewidth=1)
    axes[4].set_title("TOUTPHXL (主级热交换器出口温度 - 左)")
    axes[5].plot(df.index, df["TOUTCPRSRL"], linewidth=1)
    axes[5].set_title("TOUTCPRSRL (压气机出口温度 - 左)")

    axes[-1].set_xlabel("时间")

    fig.suptitle(title, y=0.995, fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {out_path}")

def main():
    ap = argparse.ArgumentParser(description="按时间戳对齐可视化ACM关键温度（整段航班）。")
    ap.add_argument("--tail", required=True, help="飞机尾号，例如 B-2080")
    ap.add_argument("--start", required=True, help="开始时间（本地UTC+8），例如 '2025-04-01 00:00:00'")
    ap.add_argument("--end", required=True, help="结束时间（本地UTC+8），例如 '2025-04-30 23:59:59'")
    ap.add_argument("--resample", default="1S", help="可选：重采样粒度（如'1S'、'2S'、'500ms'）。留空表示不重采样。")
    ap.add_argument("--out", default=None, help="输出图片路径（默认自动命名）。")
    args = ap.parse_args()

    # 统一输出名
    if args.out is None:
        t0 = args.start.replace(" ", "").replace(":", "").replace("-", "")
        t1 = args.end.replace(" ", "").replace(":", "").replace("-", "")
        args.out = f"fault_{args.tail}_{t0}-{t1}.png"

    session = get_session()
    try:
        df = align_all(session, args.tail, args.start, args.end, resample_rule=(args.resample or None))
    finally:
        try:
            session.close()
        except Exception:
            pass

    title = f"Tail: {args.tail} | {args.start} ~ {args.end} (UTC+8)"
    plot_figure(df, title, args.out)

if __name__ == "__main__":
    main()
