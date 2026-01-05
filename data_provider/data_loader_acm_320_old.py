# data_provider/data_loader_acm_320.py
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from iotdb.table_session import TableSession, TableSessionConfig
from tqdm import tqdm
import torch

# =========================================================
# 1. 飞机列表（ac_320_new）
# =========================================================
ac_320_new = [
    "B-301A", "B-301G", "B-301H", "B-301J", "B-301K", "B-301L", "B-301W", "B-301Y",
    "B-302J", "B-302K", "B-302L", "B-305C", "B-305D", "B-305E", "B-305M", "B-305N",
    "B-305P", "B-305Q", "B-306G", "B-306M", "B-307R", "B-307S", "B-309K", "B-309L",
    "B-309M", "B-309X", "B-309Y", "B-30A8", "B-30AA", "B-30AJ", "B-30AK", "B-30CC",
    "B-30CD", "B-30DG", "B-30EZ", "B-30F8", "B-30FX", "B-322F", "B-322V", "B-323E",
    "B-325D", "B-32ET", "B-32EU", "B-32EV", "B-32FU", "B-32FV", "B-32FW", "B-32FX",
    "B-32FY", "B-32GC", "B-32GD", "B-32GG", "B-32H3", "B-32HQ", "B-32J6", "B-32JL",
    "B-32L2", "B-32LL", "B-32LN", "B-32LP", "B-32LR", "B-32M8", "B-32ML", "B-32MQ",
]

# 调试可以只开一架：
# ac_320_new = ["B-302J"]

# =========================================================
# 2. 读取故障 CSV + 侧别 CSV，构建各侧故障日期
# =========================================================
FAULT_CSV_PATH = "data_provider/320_ACM_faults.csv"
FAULT_SIDE_CSV_PATH = "data_provider/320_ACM_faults_side.csv"

try:
    _fault_df = pd.read_csv(FAULT_CSV_PATH)
    _fault_df = _fault_df.dropna(subset=["机号", "首发日期"])
    _fault_df["首发日期"] = pd.to_datetime(_fault_df["首发日期"])

    # 读侧别映射（如果没提供，就全部 unknown）
    try:
        _side_df = pd.read_csv(FAULT_SIDE_CSV_PATH)
        _side_df["首发日期"] = pd.to_datetime(_side_df["首发日期"])
        _side_df["pack"] = _side_df["pack"].astype("Int64")
        _fault_df = _fault_df.merge(
            _side_df[["机号", "首发日期", "side", "pack"]],
            on=["机号", "首发日期"],
            how="left",
        )
    except FileNotFoundError:
        print("[WARN] 未找到 fault side CSV, 所有故障侧别视为未知。")
        _fault_df["side"] = pd.NA
        _fault_df["pack"] = pd.NA

    # 所有故障（不分侧）
    fault_dates_by_tail = (
        _fault_df.groupby("机号")["首发日期"]
        .apply(lambda s: sorted(s.unique()))
        .to_dict()
    )

    # PACK1（左侧）故障日期
    fault_dates_by_tail_pack1 = (
        _fault_df[_fault_df["pack"] == 1]
        .groupby("机号")["首发日期"]
        .apply(lambda s: sorted(s.unique()))
        .to_dict()
    )
    # PACK2（右侧）故障日期
    fault_dates_by_tail_pack2 = (
        _fault_df[_fault_df["pack"] == 2]
        .groupby("机号")["首发日期"]
        .apply(lambda s: sorted(s.unique()))
        .to_dict()
    )

    # time_start/time_end 保留（现在不直接用来查库）
    if len(_fault_df) > 0:
        _min_start = _fault_df["首发日期"].min() - pd.DateOffset(months=6)
        _max_end = _fault_df["首发日期"].max() + pd.DateOffset(months=6)
        time_start = _min_start.strftime("%Y-%m-%d")
        time_end = _max_end.strftime("%Y-%m-%d")
    else:
        time_start = "2024-01-01"
        time_end = "2024-02-01"

except FileNotFoundError:
    print("[WARN] 未找到 320_ACM_faults.csv, 视为无故障机号。")
    fault_dates_by_tail = {}
    fault_dates_by_tail_pack1 = {}
    fault_dates_by_tail_pack2 = {}
    time_start = "2024-01-01"
    time_end = "2024-02-01"


def _get_fault_dates_for_tail_side(tail_num: str, side_key: str):
    """给定机号和 side_key ('PACK1' / 'PACK2')，取该侧的故障日期列表"""
    if side_key == "PACK1":
        return fault_dates_by_tail_pack1.get(tail_num, [])
    elif side_key == "PACK2":
        return fault_dates_by_tail_pack2.get(tail_num, [])
    else:
        return []


# =========================================================
# 3. 参数列表：PACK1 / PACK2
# =========================================================
PACK1_PARA = [
    "PACK1_BYPASS_V",
    "PACK1_DISCH_T",
    "PACK1_RAM_I_DR",
    "PACK1_RAM_O_DR",
    "PACK_FLOW_R1",
    "PACK1_COMPR_T",
    "ALT_STD",
]

PACK2_PARA = [
    "PACK2_BYPASS_V",
    "PACK2_DISCH_T",
    "PACK2_RAM_I_DR",
    "PACK2_RAM_O_DR",
    "PACK_FLOW_R2",
    "PACK2_COMPR_T",
    "ALT_STD",
]

PARAMS_BY_SIDE = {
    "PACK1": PACK1_PARA,
    "PACK2": PACK2_PARA,
}

# =========================================================
# 4. 插值工具
# =========================================================
def interpolate_to_grid(df_src: pd.DataFrame, grid_time: np.ndarray, col_name: str):
    """
    把 df_src[col_name] 沿 df_src['time'] 线性插值到 grid_time（纳秒）
    """
    if len(df_src) == 0:
        return np.full(len(grid_time), np.nan, dtype=float)

    x = df_src["time"].values.astype(np.int64)
    y = df_src[col_name].values.astype(float)

    uniq, idx = np.unique(x, return_index=True)
    x = uniq
    y = y[idx]

    return np.interp(grid_time.astype(np.int64), x, y, left=y[0], right=y[-1])

# =========================================================
# 5. 按“侧别 + 模式”计算查询时间段
# =========================================================
def get_time_range_for_tail(tail_num: str, mode: str, side_key: str):
    """
    side_key: 'PACK1' / 'PACK2' （对应左/右）

    mode = 'normal':
        - 若该侧有故障：取【该侧首次故障前 13~1 个月，一整年】 => [fd-13M, fd-1M)
        - 若该侧无故障：固定用 1 年数据 [2023-01-01, 2024-01-01)

    mode = 'abnormal':
        - 仅对该侧有故障的飞机：仍然只取【该侧首次故障前一个月】 => [fd-1M, fd)
        - 若该侧无故障：返回 (None, None)
    """
    fdates = _get_fault_dates_for_tail_side(tail_num, side_key)

    if mode == "normal":
        if fdates:
            fd = min(fdates)
            # normal：故障前 13~1 个月，一整年数据
            end = (fd - pd.DateOffset(months=1)).normalize()           # fd-1M
            start = (end - pd.DateOffset(years=1)).normalize()         # (fd-1M)-1Y
            return start, end
        else:
            # 无故障飞机：给一个固定的一年区间
            start = pd.Timestamp("2023-01-01")
            end = pd.Timestamp("2024-01-01")
            return start, end

    elif mode == "abnormal":
        if not fdates:
            return None, None
        fd = min(fdates)
        # abnormal：仍然使用故障前 1 个月
        start = (fd - pd.DateOffset(months=1)).normalize()
        end = fd.normalize()
        return start, end

    else:
        raise ValueError(f"Unknown mode: {mode}")

# =========================================================
# 6. A320 PACK 数据集：FlightDataset_acm（支持 PACK1 / PACK2）
# =========================================================
class FlightDataset_acm(Dataset):
    """
    A320 ACM 数据集（左侧 PACK1 / 右侧 PACK2 通用）

    Tag:
      - 'train_normal' / 'val_normal' / 'test_normal':
            若某侧无故障：该侧只用 2024-01-01 ~ 2024-02-01 的数据；
            若该侧有故障：只用 “该侧首次故障 6 个月之前的那一个月” [fd-7M, fd-6M)。

      - 'test_abnormal' / 'train_abnormal':
            只用“在该侧有故障的机号”，
            且只取“该侧首次故障前一个月”的数据 [fd-1M, fd)。

    side:
      - 'PACK1'  → 左侧，变量名以 PACK1_ 开头，流量用 PACK_FLOW_R1
      - 'PACK2'  → 右侧，变量名以 PACK2_ 开头，流量用 PACK_FLOW_R2
    """

    def __init__(self, args, Tag, side="PACK1"):
        super().__init__()
        self.args = args
        print(f"[FlightDataset_acm] Tag: {Tag}, side: {side}")
        if side not in ("PACK1", "PACK2"):
            raise ValueError(f"side 必须是 'PACK1' 或 'PACK2'，当前: {side}")
        self.side = side
        self.Tag = Tag
        self.scaler = StandardScaler()

        # -------- IoTDB 连接 --------
        t_conn0 = time.time()
        try:
            config = TableSessionConfig(
                node_urls=["127.0.0.1:10901"],
                username="root",
                password="root",
                time_zone="Asia/Shanghai",
            )
            session = TableSession(config) 
        except Exception:
            config = TableSessionConfig(
                node_urls=["10.254.43.34:10901"],
                username="root",
                password="root",
                time_zone="Asia/Shanghai",
            )
            session = TableSession(config)
        t_conn1 = time.time()
        print(f"[IoTDB] 建立 TableSession 耗时: {t_conn1 - t_conn0:.3f} s")

        # === 调试：打印 database 列表 ===
        try:
            resp = session.execute_query_statement("SHOW DATABASES")
            print("=== IoTDB DATABASES ===")
            print(resp.todf())
        except Exception:
            pass

        db_name = "a320_ata21"
        session.execute_non_query_statement(f"USE {db_name}")
        print(f"=== USE {db_name} OK ===")

        # 调试：打印当前库中的表
        try:
            print("=== SHOW TABLES IN", db_name, "===")
            tables_resp = session.execute_query_statement("SHOW TABLES")
            print(tables_resp.todf())
        except Exception as e:
            print("SHOW TABLES error:", e)

        self.session = session

        # -------- 飞机 & 故障飞机列表（按侧别） --------
        self.all_planes = ac_320_new

        if self.side == "PACK1":
            fault_tails_side = set(fault_dates_by_tail_pack1.keys())
        else:
            fault_tails_side = set(fault_dates_by_tail_pack2.keys())

        self.fault_planes = sorted(set(self.all_planes) & fault_tails_side)

        self.train_list = self.all_planes
        self.val_list = self.all_planes
        self.test_normal_list = self.all_planes
        self.test_abnormal_list = self.fault_planes

        self.para = PARAMS_BY_SIDE[self.side]

        # -------- 缓存命名：带上 side，避免 PACK1 / PACK2 冲突 --------
        self.dataset_name = (
            f"A320_1ySlice_{self.side}_{len(self.train_list)}_planes_"
            + "_".join(self.para)
        )
        data_path = f"cache/{self.dataset_name}_{Tag}_data.npy"
        feat_path = f"cache/{self.dataset_name}_{Tag}_feature_names.npy"

        scaler_tag = "train_normal"
        scaler_mean_path = f"cache/{self.dataset_name}_{scaler_tag}_mean.npy"
        scaler_std_path = f"cache/{self.dataset_name}_{scaler_tag}_std.npy"
        use_dataset_scale = getattr(self.args, "dataset_scale", True)

        # -------- 如果已经有缓存就直接读 --------
        if os.path.exists(data_path):
            t_load0 = time.time()
            self.data = np.load(data_path, allow_pickle=False)
            if os.path.exists(feat_path):
                self.feature_names = list(np.load(feat_path, allow_pickle=True))
            else:
                self.feature_names = list(self.para)
            t_load1 = time.time()
            print(
                f"[Cache] 直接从 {data_path} 读取 (side={self.side}, Tag={Tag})，"
                f"耗时 {t_load1 - t_load0:.3f} s, data.shape={self.data.shape}"
            )
        else:
            # -------- 构建原始序列 --------
            t_build0 = time.time()
            if Tag in ["train_normal", "val_normal", "test_normal"]:
                tails = (
                    self.train_list
                    if Tag == "train_normal"
                    else (self.val_list if Tag == "val_normal" else self.test_normal_list)
                )
                self.data = self._flight_data(tails, self.para, mode="normal")
            elif Tag in ["test_abnormal", "train_abnormal"]:
                self.data = self._flight_data(
                    self.test_abnormal_list, self.para, mode="abnormal"
                )
            else:
                raise ValueError(f"Unknown Tag: {Tag}")
            t_build1 = time.time()
            print(
                f"[Build] _flight_data(Tag={Tag}, side={self.side}) 完成，"
                f"耗时 {t_build1 - t_build0:.3f} s, data.shape={self.data.shape}"
            )

            os.makedirs("cache", exist_ok=True)

            # ===== 写 npy 前打印一下窗口对应的机号和起始时间 =====
            if hasattr(self, "window_start_times") and hasattr(self, "window_tails"):
                print(
                    f"\n=== Debug: 将要写入 {data_path} 的窗口信息"
                    f" (Tag={Tag}, side={self.side}) ==="
                )
                print(f"  窗口总数: {len(self.window_start_times)}")
                for i, (t_, tail_) in enumerate(
                    zip(self.window_start_times, self.window_tails)
                ):
                    if i >= 50:
                        print(
                            f"  ... 共 {len(self.window_start_times)} 个窗口，只展示前 50 个"
                        )
                        break
                    print(f"  idx={i:04d}  tail={tail_}  start_time={t_}")
            else:
                print("\n=== Debug: 没有 window_start_times / window_tails，可能 data 为空 ===")

            # -------- 归一化 --------
            if self.data.shape[0] == 0:
                np.save(data_path, self.data)
                np.save(
                    feat_path,
                    np.array(getattr(self, "feature_names", []), dtype=object),
                )
            else:
                n, l, c = self.data.shape
                flat = self.data.reshape(-1, c)

                t_scale0 = time.time()
                if Tag == "train_normal":
                    self.scaler.fit(flat)
                    np.save(scaler_mean_path, self.scaler.mean_)
                    np.save(scaler_std_path, self.scaler.scale_)
                    flat = self.scaler.transform(flat).astype(np.float32)
                    self.data = flat.reshape(n, l, c)
                else:
                    if (
                        use_dataset_scale
                        and os.path.exists(scaler_mean_path)
                        and os.path.exists(scaler_std_path)
                    ):
                        mean = np.load(scaler_mean_path)
                        std = np.load(scaler_std_path)
                        self.scaler.mean_ = mean
                        self.scaler.scale_ = std
                        self.scaler.var_ = std ** 2
                        flat = self.scaler.transform(flat).astype(np.float32)
                        self.data = flat.reshape(n, l, c)
                    else:
                        self.scaler.fit(flat)
                        flat = self.scaler.transform(flat).astype(np.float32)
                        self.data = flat.reshape(n, l, c)
                t_scale1 = time.time()
                print(f"[Scale] 标准化(Tag={Tag}, side={self.side}) 耗时 {t_scale1 - t_scale0:.3f} s")

                print(f"\n=== Debug: 写入 npy 完成 ===")
                print(f"  data_path = {data_path}")
                print(f"  feat_path = {feat_path}")
                print(f"  data 形状 = {self.data.shape}")

                np.save(data_path, self.data)
                np.save(feat_path, np.array(self.feature_names, dtype=object))

    # -------- 内部：拉数 + 按航班段切窗口 --------
    def _flight_data(self, tail_list, para_list, mode):
        """
        每个机号 & 该侧只查一个月：

          - mode='normal':
              该侧有故障：该侧首次故障前 7~6 个月 [fd-7M, fd-6M)
              该侧无故障：2024-01-01 ~ 2024-02-01

          - mode='abnormal':
              该侧有故障：首次故障前一个月 [fd-1M, fd)
              该侧无故障：直接跳过

        然后：
          1. 以 PACKx_DISCH_T 为“主时间轴”（x=1/2）；
          2. 按时间间隔 > gap_threshold_sec 切航段；
          3. 每个航段只取前 max_windows_per_flight 个 [seq_len] 窗口。
        """
        all_seqs = []
        window_start_times = []
        window_tails = []
        feature_names_once = None

        max_windows_per_flight = getattr(self.args, "max_windows_per_flight", 5)
        gap_threshold_sec = getattr(self.args, "flight_gap_threshold_sec", 3600.0)

        master_param = (
            "PACK1_DISCH_T" if self.side == "PACK1" else "PACK2_DISCH_T"
        )

        for tail_num in tqdm(tail_list, desc=f"[A320-{mode}-{self.side}] Fetching..."):
            print(f"\n--- [{tail_num}] 开始处理 (mode={mode}, side={self.side}) ---")

            # 根据规则算出这个机号该侧的时间范围
            start_date, end_date = get_time_range_for_tail(
                tail_num, mode, self.side
            )
            if start_date is None or end_date is None or start_date >= end_date:
                print(f"  !!! 机号 {tail_num} 在 mode={mode}, side={self.side} 下没有有效时间段，跳过")
                continue

            print(f"  时间范围: {start_date} ~ {end_date} (左闭右开)")

            start_ts_ns = pd.Timestamp(start_date, tz="Asia/Shanghai").value
            end_ts_ns = pd.Timestamp(end_date, tz="Asia/Shanghai").value

            master_query = f"""
                SELECT time, value
                FROM {master_param}
                WHERE "aircraft/tail" = '{tail_num}'
                AND TIME >= {start_ts_ns} AND TIME < {end_ts_ns}
                ORDER BY TIME
            """

            t_q0 = time.time()
            try:
                master_df = self.session.execute_query_statement(master_query).todf()
            except Exception as e:
                print(f"[{tail_num}] {master_param} query error:", e)
                continue
            t_q1 = time.time()
            print(
                f"[Perf][{tail_num}] 主变量 {master_param} 查询耗时: {t_q1 - t_q0:.3f} s, "
                f"行数={len(master_df)}"
            )

            if len(master_df) == 0:
                print(f"  !!! {master_param} 结果为空, 跳过该机号 {tail_num}")
                continue

            # 看 IoTDB 实际返回的时间范围
            iot_start = (
                pd.to_datetime(master_df["time"].iloc[0], utc=True)
                .tz_convert("Asia/Shanghai")
                .tz_localize(None)
            )
            iot_end = (
                pd.to_datetime(master_df["time"].iloc[-1], utc=True)
                .tz_convert("Asia/Shanghai")
                .tz_localize(None)
            )
            print(f"    IoTDB 数据起始时间：{iot_start}")
            print(f"    IoTDB 数据结束时间：{iot_end}")

            fault_all = fault_dates_by_tail.get(tail_num, [])
            fault_side = _get_fault_dates_for_tail_side(tail_num, self.side)
            print(f"    该机号所有故障日期(不分侧)：{fault_all}")
            print(f"    当前侧 {self.side} 的故障日期：{fault_side}")

            # 统一成带 Asia/Shanghai 时区的时间列
            master_df.columns = ["time", master_param]
            master_df["time"] = pd.to_datetime(
                master_df["time"], utc=True
            ).dt.tz_convert("Asia/Shanghai")

            times = master_df["time"].values
            t_ns = times.view("i8")
            t_s = (t_ns.astype(np.int64) / 1e9).astype(float)

            # 1) 按时间间隔切“航段”
            t_seg0 = time.time()
            segments = []
            start_idx = 0
            for i in range(1, len(master_df)):
                dt = t_s[i] - t_s[i - 1]
                if dt > gap_threshold_sec:
                    segments.append((start_idx, i))
                    start_idx = i
            segments.append((start_idx, len(master_df)))
            segments = [(s, e) for (s, e) in segments if e - s >= self.args.seq_len]
            t_seg1 = time.time()
            print(
                f"[Perf][{tail_num}] 航段分割耗时: {t_seg1 - t_seg0:.3f} s, 航段数={len(segments)}"
            )
            if len(segments) == 0:
                continue

            # 2) 插值其它变量；ALT_STD 先用0占位
            grid_time_ns = t_ns
            raw = {
                "time": times,
                "ALT_STD": np.zeros(len(master_df), dtype=float),
            }

            t_interp0 = time.time()
            for param in [p for p in para_list if p not in ["ALT_STD"]]:
                q = f"""
                    SELECT time, value
                    FROM {param}
                    WHERE "aircraft/tail" = '{tail_num}'
                    AND TIME >= {start_ts_ns} AND TIME < {end_ts_ns}
                    ORDER BY TIME
                """
                t_p0 = time.time()
                try:
                    dfp = self.session.execute_query_statement(q).todf()
                    t_p1 = time.time()
                    print(
                        f"[Perf][{tail_num}] 变量 {param} 查询耗时: {t_p1 - t_p0:.3f} s, "
                        f"行数={len(dfp)}"
                    )
                    if len(dfp) == 0:
                        raw[param] = np.full(len(master_df), np.nan, dtype=float)
                        continue
                    dfp.columns = ["time", param]
                    dfp["time"] = pd.to_datetime(
                        dfp["time"], utc=True
                    ).dt.tz_convert("Asia/Shanghai")
                    raw[param] = interpolate_to_grid(dfp, grid_time_ns, param)
                except Exception as e:
                    print(f"[{tail_num}] param {param} error:", e)
                    raw[param] = np.full(len(master_df), np.nan, dtype=float)
            t_interp1 = time.time()
            print(f"[Perf][{tail_num}] 所有变量插值总耗时: {t_interp1 - t_interp0:.3f} s")

            full_df = pd.DataFrame(raw)
            feat_cols = [c for c in para_list if c in full_df.columns]
            if feature_names_once is None:
                feature_names_once = feat_cols
                self.feature_names = feature_names_once

            seq_len = self.args.seq_len

            # 3) 每个航段只取前 max_windows_per_flight 个窗口
            t_win0 = time.time()
            total_windows_this_tail = 0
            for (s, e) in segments:
                seg_len = e - s
                if seg_len < seq_len:
                    continue
                max_k = min(max_windows_per_flight, seg_len // seq_len)
                for k in range(max_k):
                    start_i = s + k * seq_len
                    end_i = start_i + seq_len
                    if end_i > e:
                        break
                    window = full_df.iloc[start_i:end_i]
                    all_seqs.append(window[feat_cols].to_numpy(dtype=float))
                    window_start_times.append(window["time"].iloc[0])
                    window_tails.append(tail_num)
                    total_windows_this_tail += 1
            t_win1 = time.time()
            print(
                f"[Perf][{tail_num}] 窗口切片耗时: {t_win1 - t_win0:.3f} s, "
                f"生成窗口数={total_windows_this_tail}"
            )

        self.window_start_times = window_start_times
        self.window_tails = window_tails
        print(
            f"[Perf] 所有机号总窗口数 (mode={mode}, side={self.side}) = {len(all_seqs)}"
        )

        if len(all_seqs) == 0:
            if not hasattr(self, "feature_names"):
                self.feature_names = list(self.para)
            return np.zeros(
                (0, self.args.seq_len, len(self.feature_names)), dtype=float
            )
        return np.stack(all_seqs, axis=0)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

# =========================================================
# 7. TimerXL 回归包装：Dataset_RegRight_TimerXL（支持 PACK1 / PACK2）
# =========================================================
class Dataset_RegRight_TimerXL(Dataset):
    """
    A320 PACK 回归包装：

      若 base.side == 'PACK1':
          输入 X 通道（6 个物理量 + 1 个 dummy 通道）:
            [PACK1_BYPASS_V,
             PACK1_DISCH_T,
             PACK1_RAM_I_DR,
             PACK1_RAM_O_DR,
             PACK_FLOW_R1,
             ALT_STD,
             dummy_zero]
          目标 y: PACK1_COMPR_T

      若 base.side == 'PACK2':
          输入 X 通道:
            [PACK2_BYPASS_V,
             PACK2_DISCH_T,
             PACK2_RAM_I_DR,
             PACK2_RAM_O_DR,
             PACK_FLOW_R2,
             ALT_STD,
             dummy_zero]
          目标 y: PACK2_COMPR_T

    返回:
      x: [1, L, 7]
      y: [1, L, 1]
      flight_index: [1]
    """

    def __init__(self, base_dataset: FlightDataset_acm):
        super().__init__()
        self.base = base_dataset
        names = getattr(self.base, "feature_names", [])
        if not names:
            raise ValueError("base_dataset.feature_names 为空，请确认已正确构建 FlightDataset_acm。")

        n2i = {n: i for i, n in enumerate(names)}

        if self.base.side == "PACK1":
            self.input_names = [
                "PACK1_BYPASS_V",
                "PACK1_DISCH_T",
                "PACK1_RAM_I_DR",
                "PACK1_RAM_O_DR",
                "PACK_FLOW_R1",
                "ALT_STD",
            ]
            target_name = "PACK1_COMPR_T"
        else:  # PACK2
            self.input_names = [
                "PACK2_BYPASS_V",
                "PACK2_DISCH_T",
                "PACK2_RAM_I_DR",
                "PACK2_RAM_O_DR",
                "PACK_FLOW_R2",
                "ALT_STD",
            ]
            target_name = "PACK2_COMPR_T"

        need = self.input_names + [target_name]
        miss = [c for c in need if c not in n2i]
        if miss:
            raise ValueError(
                f"A320 所需列缺失: {miss}\n当前列: {names}"
            )

        self.idx_x = [n2i[n] for n in self.input_names]
        self.idx_y = n2i[target_name]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        arr = self.base[idx]  # [L, D]
        L = arr.shape[0]
        x_feats = torch.from_numpy(arr[:, self.idx_x]).float()       # [L,6]
        y = torch.from_numpy(arr[:, self.idx_y : self.idx_y + 1]).float()  # [L,1]
        zero = torch.zeros((L, 1), dtype=torch.float32)
        x = torch.cat([x_feats, zero], dim=1)                        # [L,7]
        return x.unsqueeze(0), y.unsqueeze(0), torch.tensor([idx], dtype=torch.long)
