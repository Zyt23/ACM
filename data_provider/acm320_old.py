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
# 0. 统一基于本文件位置的路径（避免 cwd 不同导致找不到 CSV）
# =========================================================
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../project_root/data_provider
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)              # .../project_root

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

# =========================================================
# 2. 读取故障 CSV + 侧别 CSV，构建各侧故障日期
# =========================================================
FAULT_CSV_PATH = os.path.join(_THIS_DIR, "320_ACM_faults.csv")
FAULT_SIDE_CSV_PATH = os.path.join(_THIS_DIR, "320_ACM_faults_side.csv")

try:
    _fault_df = pd.read_csv(FAULT_CSV_PATH)
    _fault_df = _fault_df.dropna(subset=["机号", "首发日期"])
    _fault_df["首发日期"] = pd.to_datetime(_fault_df["首发日期"], errors="coerce")
    _fault_df = _fault_df.dropna(subset=["首发日期"])

    # 读侧别映射
    try:
        _side_df = pd.read_csv(FAULT_SIDE_CSV_PATH)
        _side_df["首发日期"] = pd.to_datetime(_side_df["首发日期"], errors="coerce")
        _side_df["pack"] = pd.to_numeric(_side_df["pack"], errors="coerce").astype("Int64")
        _side_df = _side_df.dropna(subset=["机号", "首发日期", "pack"])

        _fault_df = _fault_df.merge(
            _side_df[["机号", "首发日期", "side", "pack"]],
            on=["机号", "首发日期"],
            how="left",
        )
    except FileNotFoundError:
        print(f"[WARN] 未找到 fault side CSV: {FAULT_SIDE_CSV_PATH}，所有故障侧别视为未知。")
        _fault_df["side"] = pd.NA
        _fault_df["pack"] = pd.NA

    # 所有故障（不分侧）
    fault_dates_by_tail = (
        _fault_df.groupby("机号")["首发日期"]
        .apply(lambda s: sorted(pd.to_datetime(s, errors="coerce").dropna().unique()))
        .to_dict()
    )

    # PACK1 故障日期
    fault_dates_by_tail_pack1 = (
        _fault_df[_fault_df["pack"] == 1]
        .groupby("机号")["首发日期"]
        .apply(lambda s: sorted(pd.to_datetime(s, errors="coerce").dropna().unique()))
        .to_dict()
    )

    # PACK2 故障日期
    fault_dates_by_tail_pack2 = (
        _fault_df[_fault_df["pack"] == 2]
        .groupby("机号")["首发日期"]
        .apply(lambda s: sorted(pd.to_datetime(s, errors="coerce").dropna().unique()))
        .to_dict()
    )

except FileNotFoundError:
    print(f"[WARN] 未找到 faults CSV: {FAULT_CSV_PATH}，视为无故障机号。")
    fault_dates_by_tail = {}
    fault_dates_by_tail_pack1 = {}
    fault_dates_by_tail_pack2 = {}

def _get_fault_dates_for_tail_side(tail_num: str, side_key: str):
    """给定机号和 side_key ('PACK1' / 'PACK2')，取该侧的故障日期列表"""
    if side_key == "PACK1":
        return fault_dates_by_tail_pack1.get(tail_num, [])
    elif side_key == "PACK2":
        return fault_dates_by_tail_pack2.get(tail_num, [])
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
PARAMS_BY_SIDE = {"PACK1": PACK1_PARA, "PACK2": PACK2_PARA}

# =========================================================
# 4. 插值工具
# =========================================================
def interpolate_to_grid(df_src: pd.DataFrame, grid_time: np.ndarray, col_name: str):
    """把 df_src[col_name] 沿 df_src['time'] 线性插值到 grid_time（纳秒）"""
    if len(df_src) == 0:
        return np.full(len(grid_time), np.nan, dtype=float)

    x = df_src["time"].values.astype(np.int64)
    y = df_src[col_name].values.astype(float)

    uniq, idx = np.unique(x, return_index=True)
    x = uniq
    y = y[idx]

    return np.interp(grid_time.astype(np.int64), x, y, left=y[0], right=y[-1])

# =========================================================
# 5. 统一的时间切分规则（你要求的版本）
# =========================================================
def get_time_range_for_tail(
    tail_num: str,
    mode: str,
    side_key: str,
    train_months: int,
    test_months: int,
    gap_months: int,
    anchor_end_str: str = "2024-01-01",
):
    """
    你要求的规则（按 side 的首次故障 fd）：

    对有故障的飞机：
      abnormal:   [fd-1M, fd)
      anchor_end = fd-gap_months   (例如 gap_months=6)
      test_normal:  [anchor_end-test_months, anchor_end)
      train_normal: [anchor_end-(test_months+train_months), anchor_end-test_months)

    对无故障飞机：
      使用 anchor_end_str 作为 anchor_end
      test_normal:  [anchor_end-test_months, anchor_end)
      train_normal: [anchor_end-(test_months+train_months), anchor_end-test_months)
      abnormal: None
    """
    train_months = int(max(1, train_months))
    test_months = int(max(1, test_months))
    gap_months = int(max(0, gap_months))

    fdates = _get_fault_dates_for_tail_side(tail_num, side_key)
    has_fault = bool(fdates)

    if mode == "abnormal":
        if not has_fault:
            return None, None
        fd = pd.Timestamp(min(fdates)).normalize()
        start = (fd - pd.DateOffset(months=1)).normalize()
        end = fd
        return start, end

    # train_normal / test_normal 都走 anchor_end 逻辑
    if has_fault:
        fd = pd.Timestamp(min(fdates)).normalize()
        anchor_end = (fd - pd.DateOffset(months=gap_months)).normalize()
    else:
        anchor_end = pd.Timestamp(anchor_end_str).normalize()

    if mode == "test_normal":
        end = anchor_end
        start = (end - pd.DateOffset(months=test_months)).normalize()
        return start, end

    if mode == "train_normal":
        end = (anchor_end - pd.DateOffset(months=test_months)).normalize()
        start = (end - pd.DateOffset(months=train_months)).normalize()
        return start, end

    raise ValueError(f"Unknown mode: {mode}")

# =========================================================
# 6. A320 PACK 数据集：FlightDataset_acm（支持 PACK1 / PACK2）
# =========================================================
class FlightDataset_acm(Dataset):
    """
    A320 ACM 数据集（左侧 PACK1 / 右侧 PACK2 通用）

    Tag 支持：
      - train_normal / val_normal : 用 train_normal 时间段（2个月）
      - test_normal_recent        : 用 test_normal 时间段（1个月，紧挨 train 之后）
      - test_abnormal             : 用 abnormal 时间段（故障前1个月）
    """

    def __init__(self, args, Tag, side="PACK1"):
        super().__init__()
        self.args = args
        self.Tag = Tag
        if side not in ("PACK1", "PACK2"):
            raise ValueError(f"side 必须是 'PACK1' 或 'PACK2'，当前: {side}")
        self.side = side

        print(f"[FlightDataset_acm] Tag: {Tag}, side: {side}")
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

        db_name = "a320_ata21"
        session.execute_non_query_statement(f"USE {db_name}")
        print(f"=== USE {db_name} OK ===")
        self.session = session

        # -------- 飞机列表 --------
        self.all_planes = ac_320_new

        if self.side == "PACK1":
            fault_tails_side = set(fault_dates_by_tail_pack1.keys())
        else:
            fault_tails_side = set(fault_dates_by_tail_pack2.keys())
        self.fault_planes = sorted(set(self.all_planes) & fault_tails_side)

        self.para = PARAMS_BY_SIDE[self.side]

        # =========================================================
        # args 参数（按你要求）
        # =========================================================
        self.train_months = int(getattr(self.args, "normal_months", 2))               # 训练/验证 2个月
        self.test_normal_months = int(getattr(self.args, "test_normal_months", 1))   # test_normal 1个月
        self.fault_gap_months = int(getattr(self.args, "fault_gap_months", 6))       # 故障前 gap 个月作为 anchor_end
        self.normal_anchor_end = str(getattr(self.args, "normal_anchor_end", "2024-01-01"))

        # ========== Cache version（改逻辑必须升级，避免读旧缓存） ==========
        cache_version = "v3_train2_test1_gap6"

        cache_dir = os.path.join(_PROJECT_ROOT, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        # 把关键参数写进 dataset_name，避免不同配置误读缓存
        anchor_safe = self.normal_anchor_end.replace("-", "")
        self.dataset_name = (
            f"A320_train{self.train_months}M_test{self.test_normal_months}M_gap{self.fault_gap_months}M_"
            f"anchor{anchor_safe}_{cache_version}_{self.side}_{len(self.all_planes)}planes_"
            + "_".join(self.para)
        )

        data_path = os.path.join(cache_dir, f"{self.dataset_name}_{Tag}_data.npy")
        feat_path = os.path.join(cache_dir, f"{self.dataset_name}_{Tag}_feature_names.npy")
        meta_time_path = os.path.join(cache_dir, f"{self.dataset_name}_{Tag}_window_start_times.npy")
        meta_tail_path = os.path.join(cache_dir, f"{self.dataset_name}_{Tag}_window_tails.npy")

        scaler_tag = "train_normal"
        scaler_mean_path = os.path.join(cache_dir, f"{self.dataset_name}_{scaler_tag}_mean.npy")
        scaler_std_path = os.path.join(cache_dir, f"{self.dataset_name}_{scaler_tag}_std.npy")
        use_dataset_scale = getattr(self.args, "dataset_scale", True)

        has_data = os.path.exists(data_path)
        has_meta = os.path.exists(meta_time_path) and os.path.exists(meta_tail_path)

        if has_data and has_meta:
            t_load0 = time.time()
            self.data = np.load(data_path, allow_pickle=False)

            if os.path.exists(feat_path):
                self.feature_names = list(np.load(feat_path, allow_pickle=True))
            else:
                self.feature_names = list(self.para)

            self.window_start_times = list(np.load(meta_time_path, allow_pickle=True))
            self.window_tails = list(np.load(meta_tail_path, allow_pickle=True))
            t_load1 = time.time()
            print(
                f"[Cache] 使用完整 cache: {data_path} (Tag={Tag}, side={self.side}) | "
                f"{t_load1 - t_load0:.3f}s | data.shape={self.data.shape} | meta_len={len(self.window_start_times)}"
            )
            return

        if has_data and not has_meta:
            print(
                f"[Cache][WARN] 发现 data cache 但缺少 meta cache，强制重建。\n"
                f"  data:      {data_path}\n"
                f"  meta_time:  {meta_time_path}\n"
                f"  meta_tail:  {meta_tail_path}"
            )

        # -------- 构建数据 --------
        t_build0 = time.time()

        if Tag in ["train_normal", "val_normal"]:
            tails = self.all_planes
            self.data = self._flight_data(tails, self.para, mode="train_normal")
        elif Tag in ["test_normal_recent", "test_normal"]:
            tails = self.all_planes
            self.data = self._flight_data(tails, self.para, mode="test_normal")
        elif Tag in ["test_abnormal", "train_abnormal"]:
            tails = self.fault_planes
            self.data = self._flight_data(tails, self.para, mode="abnormal")
        else:
            raise ValueError(f"Unknown Tag: {Tag}")

        t_build1 = time.time()
        print(
            f"[Build] _flight_data(Tag={Tag}, side={self.side}) 完成 | "
            f"{t_build1 - t_build0:.3f}s | data.shape={self.data.shape}"
        )

        # -------- 归一化 + 保存 --------
        if self.data.shape[0] == 0:
            np.save(data_path, self.data)
            np.save(feat_path, np.array(getattr(self, "feature_names", []), dtype=object))
            np.save(meta_time_path, np.array([], dtype=object))
            np.save(meta_tail_path, np.array([], dtype=object))
            return

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
            if use_dataset_scale and os.path.exists(scaler_mean_path) and os.path.exists(scaler_std_path):
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
        print(f"[Scale] 标准化(Tag={Tag}, side={self.side}) 耗时 {t_scale1 - t_scale0:.3f}s")

        np.save(data_path, self.data)
        np.save(feat_path, np.array(self.feature_names, dtype=object))

        if hasattr(self, "window_start_times") and hasattr(self, "window_tails"):
            np.save(meta_time_path, np.array([str(t) for t in self.window_start_times], dtype=object))
            np.save(meta_tail_path, np.array(self.window_tails, dtype=object))
        else:
            np.save(meta_time_path, np.array([], dtype=object))
            np.save(meta_tail_path, np.array([], dtype=object))

    # -------- 内部：拉数 + 按航段切窗口 --------
    def _flight_data(self, tail_list, para_list, mode):
        """
        1) 以 PACKx_DISCH_T 为主时间轴；
        2) 按 gap_threshold_sec 分割航段；
        3) 每航段只取前 max_windows_per_flight 个 seq_len 窗口。
        """
        all_seqs = []
        window_start_times = []
        window_tails = []
        feature_names_once = None

        max_windows_per_flight = getattr(self.args, "max_windows_per_flight", 5)
        gap_threshold_sec = getattr(self.args, "flight_gap_threshold_sec", 3600.0)
        master_param = "PACK1_DISCH_T" if self.side == "PACK1" else "PACK2_DISCH_T"

        for tail_num in tqdm(tail_list, desc=f"[A320-{mode}-{self.side}] Fetching..."):
            print(f"\n--- [{tail_num}] 开始处理 (mode={mode}, side={self.side}) ---")

            start_date, end_date = get_time_range_for_tail(
                tail_num=tail_num,
                mode=mode,
                side_key=self.side,
                train_months=self.train_months,
                test_months=self.test_normal_months,
                gap_months=self.fault_gap_months,
                anchor_end_str=self.normal_anchor_end,
            )
            if start_date is None or end_date is None or start_date >= end_date:
                print(f"  !!! 机号 {tail_num} 在 mode={mode}, side={self.side} 下没有有效时间段，跳过")
                continue

            print(f"  时间范围: {start_date} ~ {end_date} (左闭右开) | "
                  f"trainM={self.train_months} testM={self.test_normal_months} gapM={self.fault_gap_months} "
                  f"anchor_end(no-fault)={self.normal_anchor_end}")

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
            print(f"[Perf][{tail_num}] 主变量 {master_param} 查询耗时: {t_q1 - t_q0:.3f}s, 行数={len(master_df)}")

            if len(master_df) == 0:
                print(f"  !!! {master_param} 结果为空, 跳过该机号 {tail_num}")
                continue

            fault_all = fault_dates_by_tail.get(tail_num, [])
            fault_side = _get_fault_dates_for_tail_side(tail_num, self.side)
            print(f"    该机号所有故障日期(不分侧)：{fault_all}")
            print(f"    当前侧 {self.side} 的故障日期：{fault_side}")

            master_df.columns = ["time", master_param]
            master_df["time"] = pd.to_datetime(master_df["time"], utc=True).dt.tz_convert("Asia/Shanghai")

            times = master_df["time"].values
            t_ns = times.view("i8")
            t_s = (t_ns.astype(np.int64) / 1e9).astype(float)

            # 1) 航段分割
            segments = []
            start_idx = 0
            for i in range(1, len(master_df)):
                dt = t_s[i] - t_s[i - 1]
                if dt > gap_threshold_sec:
                    segments.append((start_idx, i))
                    start_idx = i
            segments.append((start_idx, len(master_df)))
            segments = [(s, e) for (s, e) in segments if e - s >= self.args.seq_len]
            print(f"[Perf][{tail_num}] 航段数={len(segments)}")
            if len(segments) == 0:
                continue

            # 2) 插值其它变量；ALT_STD 先用0占位
            grid_time_ns = t_ns
            raw = {"time": times, "ALT_STD": np.zeros(len(master_df), dtype=float)}

            for param in [p for p in para_list if p not in ["ALT_STD"]]:
                q = f"""
                    SELECT time, value
                    FROM {param}
                    WHERE "aircraft/tail" = '{tail_num}'
                    AND TIME >= {start_ts_ns} AND TIME < {end_ts_ns}
                    ORDER BY TIME
                """
                try:
                    dfp = self.session.execute_query_statement(q).todf()
                    if len(dfp) == 0:
                        raw[param] = np.full(len(master_df), np.nan, dtype=float)
                        continue
                    dfp.columns = ["time", param]
                    dfp["time"] = pd.to_datetime(dfp["time"], utc=True).dt.tz_convert("Asia/Shanghai")
                    raw[param] = interpolate_to_grid(dfp, grid_time_ns, param)
                except Exception as e:
                    print(f"[{tail_num}] param {param} error:", e)
                    raw[param] = np.full(len(master_df), np.nan, dtype=float)

            full_df = pd.DataFrame(raw)
            feat_cols = [c for c in para_list if c in full_df.columns]
            if feature_names_once is None:
                feature_names_once = feat_cols
                self.feature_names = feature_names_once

            seq_len = self.args.seq_len

            # 3) 切窗口（每航段前 max_windows_per_flight 个）
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

            print(f"[Perf][{tail_num}] 生成窗口数={total_windows_this_tail}")

        self.window_start_times = window_start_times
        self.window_tails = window_tails
        print(f"[Perf] 所有机号总窗口数 (mode={mode}, side={self.side}) = {len(all_seqs)}")

        if len(all_seqs) == 0:
            if not hasattr(self, "feature_names"):
                self.feature_names = list(self.para)
            return np.zeros((0, self.args.seq_len, len(self.feature_names)), dtype=float)

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
    输入 x: [1, L, 7] （最后一维 dummy=0）
    目标 y: PACKx_COMPR_T
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
                "PACK1_BYPASS_V", "PACK1_DISCH_T", "PACK1_RAM_I_DR",
                "PACK1_RAM_O_DR", "PACK_FLOW_R1", "ALT_STD",
            ]
            target_name = "PACK1_COMPR_T"
        else:
            self.input_names = [
                "PACK2_BYPASS_V", "PACK2_DISCH_T", "PACK2_RAM_I_DR",
                "PACK2_RAM_O_DR", "PACK_FLOW_R2", "ALT_STD",
            ]
            target_name = "PACK2_COMPR_T"

        need = self.input_names + [target_name]
        miss = [c for c in need if c not in n2i]
        if miss:
            raise ValueError(f"A320 所需列缺失: {miss}\n当前列: {names}")

        self.idx_x = [n2i[n] for n in self.input_names]
        self.idx_y = n2i[target_name]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        arr = self.base[idx]  # [L, D]
        L = arr.shape[0]
        x_feats = torch.from_numpy(arr[:, self.idx_x]).float()                    # [L,6]
        y = torch.from_numpy(arr[:, self.idx_y: self.idx_y + 1]).float()         # [L,1]
        zero = torch.zeros((L, 1), dtype=torch.float32)
        x = torch.cat([x_feats, zero], dim=1)                                     # [L,7]
        return x.unsqueeze(0), y.unsqueeze(0), torch.tensor([idx], dtype=torch.long)
