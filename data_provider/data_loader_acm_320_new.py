# data_provider/data_loader_acm_320.py
# -*- coding: utf-8 -*-
import os
import time
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from iotdb.table_session import TableSession, TableSessionConfig
from tqdm import tqdm
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# =========================================================
# 0. 简单日志 & 计时器
# =========================================================
def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(f"[{_now()}] {msg}", flush=True)

class StepTimer:
    def __init__(self, name: str, timing_dict: dict = None):
        self.name = name
        self.timing_dict = timing_dict

    def __enter__(self):
        self.t0 = time.time()
        log(f"[START] {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        log(f"[END]   {self.name} | {dt:.3f}s")
        if self.timing_dict is not None:
            self.timing_dict[self.name] = self.timing_dict.get(self.name, 0.0) + dt

# =========================================================
# 1. 飞机列表
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
# 2. 故障日期（按侧）
# =========================================================
FAULT_CSV_PATH = os.path.join(_THIS_DIR, "320_ACM_faults.csv")
FAULT_SIDE_CSV_PATH = os.path.join(_THIS_DIR, "320_ACM_faults_side.csv")

try:
    _fault_df = pd.read_csv(FAULT_CSV_PATH)
    _fault_df = _fault_df.dropna(subset=["机号", "首发日期"])
    _fault_df["首发日期"] = pd.to_datetime(_fault_df["首发日期"], errors="coerce")
    _fault_df = _fault_df.dropna(subset=["首发日期"])

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
        log(f"[WARN] 未找到 fault side CSV: {FAULT_SIDE_CSV_PATH}")
        _fault_df["side"] = pd.NA
        _fault_df["pack"] = pd.NA

    fault_dates_by_tail = (
        _fault_df.groupby("机号")["首发日期"]
        .apply(lambda s: sorted(pd.to_datetime(s, errors="coerce").dropna().unique()))
        .to_dict()
    )

    fault_dates_by_tail_pack1 = (
        _fault_df[_fault_df["pack"] == 1]
        .groupby("机号")["首发日期"]
        .apply(lambda s: sorted(pd.to_datetime(s, errors="coerce").dropna().unique()))
        .to_dict()
    )

    fault_dates_by_tail_pack2 = (
        _fault_df[_fault_df["pack"] == 2]
        .groupby("机号")["首发日期"]
        .apply(lambda s: sorted(pd.to_datetime(s, errors="coerce").dropna().unique()))
        .to_dict()
    )

except FileNotFoundError:
    log(f"[WARN] 未找到 faults CSV: {FAULT_CSV_PATH}，视为无故障机号。")
    fault_dates_by_tail = {}
    fault_dates_by_tail_pack1 = {}
    fault_dates_by_tail_pack2 = {}

def _get_fault_dates_for_tail_side(tail_num: str, side_key: str):
    if side_key == "PACK1":
        return fault_dates_by_tail_pack1.get(tail_num, [])
    elif side_key == "PACK2":
        return fault_dates_by_tail_pack2.get(tail_num, [])
    return []

# =========================================================
# 3. 参数
# =========================================================
PACK1_PARA = [
    "PACK1_BYPASS_V", "PACK1_DISCH_T", "PACK1_RAM_I_DR", "PACK1_RAM_O_DR",
    "PACK_FLOW_R1", "PACK1_COMPR_T", "ALT_STD",
]
PACK2_PARA = [
    "PACK2_BYPASS_V", "PACK2_DISCH_T", "PACK2_RAM_I_DR", "PACK2_RAM_O_DR",
    "PACK_FLOW_R2", "PACK2_COMPR_T", "ALT_STD",
]
PARAMS_BY_SIDE = {"PACK1": PACK1_PARA, "PACK2": PACK2_PARA}

# =========================================================
# 4. 插值
# =========================================================
def interpolate_to_grid(df_src: pd.DataFrame, grid_time: np.ndarray, col_name: str):
    if len(df_src) == 0:
        return np.full(len(grid_time), np.nan, dtype=float)

    x = df_src["time"].values.astype(np.int64)
    y = df_src[col_name].values.astype(float)

    uniq, idx = np.unique(x, return_index=True)
    x = uniq
    y = y[idx]
    return np.interp(grid_time.astype(np.int64), x, y, left=y[0], right=y[-1])

# =========================================================
# 5. 时间切分（baseline 改为 train 前 1 个月）
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
    约定（左闭右开）：
      - abnormal: [fault_date-1M, fault_date)
      - test_normal:  [anchor_end-test_months, anchor_end)
      - train_normal: [anchor_end-test_months-train_months, anchor_end-test_months)
      - baseline_normal: train_normal 的前 1 个月
                       = [train_start-1M, train_start)

    anchor_end:
      - 有故障：anchor_end = fault_date - gap_months
      - 无故障：anchor_end = anchor_end_str
    """
    train_months = int(max(1, train_months))
    test_months = int(max(1, test_months))
    gap_months = int(max(0, gap_months))

    fdates = _get_fault_dates_for_tail_side(tail_num, side_key)
    has_fault = bool(fdates)

    def _anchor_end():
        if has_fault:
            fd = pd.Timestamp(min(fdates)).normalize()
            return (fd - pd.DateOffset(months=gap_months)).normalize()
        return pd.Timestamp(anchor_end_str).normalize()

    if mode == "abnormal":
        if not has_fault:
            return None, None
        fd = pd.Timestamp(min(fdates)).normalize()
        return (fd - pd.DateOffset(months=1)).normalize(), fd

    anchor_end = _anchor_end()

    if mode == "test_normal":
        end = anchor_end
        start = (end - pd.DateOffset(months=test_months)).normalize()
        return start, end

    if mode == "train_normal":
        end = (anchor_end - pd.DateOffset(months=test_months)).normalize()
        start = (end - pd.DateOffset(months=train_months)).normalize()
        return start, end

    if mode == "baseline_normal":
        # baseline = train_normal 前 1 个月
        train_end = (anchor_end - pd.DateOffset(months=test_months)).normalize()
        train_start = (train_end - pd.DateOffset(months=train_months)).normalize()
        end = train_start
        start = (end - pd.DateOffset(months=1)).normalize()
        return start, end

    raise ValueError(f"Unknown mode: {mode}")

# =========================================================
# 6. Dataset
# =========================================================
class FlightDataset_acm(Dataset):
    """
    Tag 支持：
      - train_normal / val_normal         -> mode="train_normal"
      - test_normal_recent / test_normal  -> mode="test_normal"
      - baseline_normal                   -> mode="baseline_normal" (train 前 1 个月)
      - test_abnormal                     -> mode="abnormal"
    """

    def __init__(self, args, Tag, side="PACK1"):
        super().__init__()
        self.args = args
        self.Tag = Tag
        if side not in ("PACK1", "PACK2"):
            raise ValueError(f"side 必须是 'PACK1' 或 'PACK2'，当前: {side}")
        self.side = side
        self.scaler = StandardScaler()

        # timing & stats
        self._timing = {}
        self._sql_stats = {
            "queries": 0,
            "rows_total": 0,
            "by_param_queries": defaultdict(int),
            "by_param_rows": defaultdict(int),
            "errors": 0,
        }

        log(f"[FlightDataset_acm] Tag={Tag}, side={side}")

        # IoTDB
        with StepTimer("[IoTDB] connect", self._timing):
            try:
                config = TableSessionConfig(
                    node_urls=["127.0.0.1:10901"],
                    username="root",
                    password="root",
                    time_zone="Asia/Shanghai",
                )
                session = TableSession(config)
                log("[IoTDB] connected to 127.0.0.1:10901")
            except Exception as e:
                log(f"[IoTDB] local connect failed: {repr(e)}")
                config = TableSessionConfig(
                    node_urls=["10.254.43.34:10901"],
                    username="root",
                    password="root",
                    time_zone="Asia/Shanghai",
                )
                session = TableSession(config)
                log("[IoTDB] connected to 10.254.43.34:10901")

        db_name = "a320_ata21"
        session.execute_non_query_statement(f"USE {db_name}")
        self.session = session
        log(f"[IoTDB] USE {db_name}")

        self.all_planes = ac_320_new
        if self.side == "PACK1":
            fault_tails_side = set(fault_dates_by_tail_pack1.keys())
        else:
            fault_tails_side = set(fault_dates_by_tail_pack2.keys())
        self.fault_planes = sorted(set(self.all_planes) & fault_tails_side)

        self.para = PARAMS_BY_SIDE[self.side]

        # args
        self.train_months = int(getattr(self.args, "normal_months", 2))
        self.test_normal_months = int(getattr(self.args, "test_normal_months", 1))
        self.fault_gap_months = int(getattr(self.args, "fault_gap_months", 6))
        self.normal_anchor_end = str(getattr(self.args, "normal_anchor_end", "2024-01-01"))

        # cache（升级版本避免旧缓存污染）
        cache_version = "v5_baseline_before_train"
        anchor_safe = self.normal_anchor_end.replace("-", "")
        cache_dir = os.path.join(_PROJECT_ROOT, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        self.dataset_name = (
            f"A320_train{self.train_months}M_test{self.test_normal_months}M_gap{self.fault_gap_months}M_"
            f"anchor{anchor_safe}_{cache_version}_{self.side}_"
            f"{len(self.all_planes)}planes_" + "_".join(self.para)
        )

        data_path = os.path.join(cache_dir, f"{self.dataset_name}_{Tag}_data.npy")
        feat_path = os.path.join(cache_dir, f"{self.dataset_name}_{Tag}_feature_names.npy")
        meta_time_path = os.path.join(cache_dir, f"{self.dataset_name}_{Tag}_window_start_times.npy")
        meta_tail_path = os.path.join(cache_dir, f"{self.dataset_name}_{Tag}_window_tails.npy")

        scaler_tag = "train_normal"
        scaler_mean_path = os.path.join(cache_dir, f"{self.dataset_name}_{scaler_tag}_mean.npy")
        scaler_std_path = os.path.join(cache_dir, f"{self.dataset_name}_{scaler_tag}_std.npy")
        use_dataset_scale = getattr(self.args, "dataset_scale", True)

        # cache hit
        if os.path.exists(data_path) and os.path.exists(meta_time_path) and os.path.exists(meta_tail_path):
            with StepTimer("[Cache] load", self._timing):
                self.data = np.load(data_path, allow_pickle=False)
                self.feature_names = list(np.load(feat_path, allow_pickle=True)) if os.path.exists(feat_path) else list(self.para)
                self.window_start_times = list(np.load(meta_time_path, allow_pickle=True))
                self.window_tails = list(np.load(meta_tail_path, allow_pickle=True))
            log(f"[Cache] hit: {data_path} | shape={self.data.shape}")
            log(f"[Timing summary] {dict(self._timing)}")
            return

        # build
        with StepTimer(f"[Build] Tag={Tag}", self._timing):
            if Tag in ["train_normal", "val_normal"]:
                tails = self.all_planes
                self.data = self._flight_data(tails, self.para, mode="train_normal")
            elif Tag in ["test_normal_recent", "test_normal"]:
                tails = self.all_planes
                self.data = self._flight_data(tails, self.para, mode="test_normal")
            elif Tag in ["baseline_normal"]:
                tails = self.all_planes
                self.data = self._flight_data(tails, self.para, mode="baseline_normal")
            elif Tag in ["test_abnormal", "train_abnormal"]:
                tails = self.fault_planes
                self.data = self._flight_data(tails, self.para, mode="abnormal")
            else:
                raise ValueError(f"Unknown Tag: {Tag}")

        # save empty
        if self.data.shape[0] == 0:
            np.save(data_path, self.data)
            np.save(feat_path, np.array(getattr(self, "feature_names", []), dtype=object))
            np.save(meta_time_path, np.array([], dtype=object))
            np.save(meta_tail_path, np.array([], dtype=object))
            log(f"[Cache] saved empty: {data_path} | shape={self.data.shape}")
            log(f"[Timing summary] {dict(self._timing)}")
            log(self._format_sql_summary())
            return

        # scale
        with StepTimer(f"[Scale] Tag={Tag}", self._timing):
            n, l, c = self.data.shape
            flat = self.data.reshape(-1, c)

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

        # save cache
        with StepTimer("[Cache] save", self._timing):
            np.save(data_path, self.data)
            np.save(feat_path, np.array(self.feature_names, dtype=object))
            np.save(meta_time_path, np.array([str(t) for t in self.window_start_times], dtype=object))
            np.save(meta_tail_path, np.array(self.window_tails, dtype=object))

        log(f"[Cache] saved: {data_path} | shape={self.data.shape}")
        log(f"[Timing summary] {dict(self._timing)}")
        log(self._format_sql_summary())

    # ---------- SQL helper (每次打印耗时&行数) ----------
    def _query_iotdb_df(self, sql: str, param: str, tail: str):
        t0 = time.time()
        try:
            df = self.session.execute_query_statement(sql).todf()
            dt = time.time() - t0
            rows = len(df)

            self._sql_stats["queries"] += 1
            self._sql_stats["rows_total"] += rows
            self._sql_stats["by_param_queries"][param] += 1
            self._sql_stats["by_param_rows"][param] += rows

            log(f"[SQL] tail={tail} param={param} rows={rows} time={dt:.3f}s")
            return df
        except Exception as e:
            self._sql_stats["errors"] += 1
            log(f"[SQL ERROR] tail={tail} param={param} err={repr(e)}")
            return None

    def _format_sql_summary(self):
        parts = []
        parts.append(
            f"[SQL summary] queries={self._sql_stats['queries']}, "
            f"rows_total={self._sql_stats['rows_total']}, errors={self._sql_stats['errors']}"
        )
        rows_items = sorted(self._sql_stats["by_param_rows"].items(), key=lambda x: x[1], reverse=True)
        top = rows_items[:20]
        if top:
            parts.append(
                "[SQL by_param] " + " | ".join(
                    [f"{k}: q={self._sql_stats['by_param_queries'][k]}, rows={v}" for k, v in top]
                )
            )
        return "\n".join(parts)

    def _flight_data(self, tail_list, para_list, mode: str):
        all_seqs = []
        window_start_times = []
        window_tails = []
        feature_names_once = None

        max_windows_per_flight = getattr(self.args, "max_windows_per_flight", 5)
        gap_threshold_sec = getattr(self.args, "flight_gap_threshold_sec", 3600.0)
        master_param = "PACK1_DISCH_T" if self.side == "PACK1" else "PACK2_DISCH_T"

        tail_windows = defaultdict(int)
        tail_segments = defaultdict(int)

        for tail_num in tqdm(tail_list, desc=f"[A320-{mode}-{self.side}] Fetching..."):
            t_tail0 = time.time()

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
                log(f"[TAIL] {tail_num} mode={mode} side={self.side} -> skip (no valid range)")
                continue

            log(f"[TAIL] {tail_num} mode={mode} side={self.side} range={start_date.date()} ~ {end_date.date()}")

            start_ts_ns = pd.Timestamp(start_date, tz="Asia/Shanghai").value
            end_ts_ns = pd.Timestamp(end_date, tz="Asia/Shanghai").value

            master_query = f"""
                SELECT time, value
                FROM {master_param}
                WHERE "aircraft/tail" = '{tail_num}'
                AND TIME >= {start_ts_ns} AND TIME < {end_ts_ns}
                ORDER BY TIME
            """
            master_df = self._query_iotdb_df(master_query, master_param, tail_num)
            if master_df is None or len(master_df) == 0:
                log(f"[TAIL] {tail_num} master rows=0 -> skip")
                continue

            master_df.columns = ["time", master_param]
            master_df["time"] = pd.to_datetime(master_df["time"], utc=True).dt.tz_convert("Asia/Shanghai")

            times = master_df["time"].values
            t_ns = times.view("i8")
            t_s = (t_ns.astype(np.int64) / 1e9).astype(float)

            segments = []
            start_idx = 0
            for i in range(1, len(master_df)):
                if (t_s[i] - t_s[i - 1]) > gap_threshold_sec:
                    segments.append((start_idx, i))
                    start_idx = i
            segments.append((start_idx, len(master_df)))
            segments = [(s, e) for (s, e) in segments if e - s >= self.args.seq_len]
            if not segments:
                log(f"[TAIL] {tail_num} segments=0 (after filter e-s>=seq_len) -> skip")
                continue

            tail_segments[tail_num] = len(segments)
            log(f"[TAIL] {tail_num} segments={len(segments)} gap_thr={gap_threshold_sec}s seq_len={self.args.seq_len}")

            grid_time_ns = t_ns
            raw = {"time": times, "ALT_STD": np.zeros(len(master_df), dtype=float)}

            with StepTimer(f"[TAIL {tail_num}] query+interp params", None):
                for param in [p for p in para_list if p not in ["ALT_STD"]]:
                    q = f"""
                        SELECT time, value
                        FROM {param}
                        WHERE "aircraft/tail" = '{tail_num}'
                        AND TIME >= {start_ts_ns} AND TIME < {end_ts_ns}
                        ORDER BY TIME
                    """
                    dfp = self._query_iotdb_df(q, param, tail_num)
                    if dfp is None:
                        raw[param] = np.full(len(master_df), np.nan, dtype=float)
                        continue
                    if len(dfp) == 0:
                        raw[param] = np.full(len(master_df), np.nan, dtype=float)
                        continue

                    dfp.columns = ["time", param]
                    dfp["time"] = pd.to_datetime(dfp["time"], utc=True).dt.tz_convert("Asia/Shanghai")

                    t_interp0 = time.time()
                    raw[param] = interpolate_to_grid(dfp, grid_time_ns, param)
                    log(f"[INTERP] tail={tail_num} param={param} grid={len(master_df)} src={len(dfp)} time={(time.time()-t_interp0):.3f}s")

            full_df = pd.DataFrame(raw)
            feat_cols = [c for c in para_list if c in full_df.columns]
            if feature_names_once is None:
                feature_names_once = feat_cols
                self.feature_names = feature_names_once
                log(f"[FEATURES] {self.feature_names}")

            seq_len = self.args.seq_len
            n_windows_tail = 0
            with StepTimer(f"[TAIL {tail_num}] windowing", None):
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
                        n_windows_tail += 1

            tail_windows[tail_num] += n_windows_tail
            log(f"[TAIL] {tail_num} windows={n_windows_tail} (max_windows_per_flight={max_windows_per_flight})")
            log(f"[TAIL] {tail_num} total_time={(time.time()-t_tail0):.3f}s")

        self.window_start_times = window_start_times
        self.window_tails = window_tails

        if len(tail_windows) > 0:
            total_w = sum(tail_windows.values())
            total_seg = sum(tail_segments.values())
            log(f"[SUMMARY] mode={mode} side={self.side} tails_used={len(tail_windows)} total_segments={total_seg} total_windows={total_w}")
            top10 = sorted(tail_windows.items(), key=lambda x: x[1], reverse=True)[:10]
            log("[SUMMARY] top10_windows: " + ", ".join([f"{t}:{w}" for t, w in top10]))
        else:
            log(f"[SUMMARY] mode={mode} side={self.side} tails_used=0")

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
# 7. TimerXL 回归包装
# =========================================================
class Dataset_RegRight_TimerXL(Dataset):
    def __init__(self, base_dataset: FlightDataset_acm):
        super().__init__()
        self.base = base_dataset
        names = getattr(self.base, "feature_names", [])
        if not names:
            raise ValueError("base_dataset.feature_names 为空")

        n2i = {n: i for i, n in enumerate(names)}
        if self.base.side == "PACK1":
            self.input_names = ["PACK1_BYPASS_V","PACK1_DISCH_T","PACK1_RAM_I_DR","PACK1_RAM_O_DR","PACK_FLOW_R1","ALT_STD"]
            target_name = "PACK1_COMPR_T"
        else:
            self.input_names = ["PACK2_BYPASS_V","PACK2_DISCH_T","PACK2_RAM_I_DR","PACK2_RAM_O_DR","PACK_FLOW_R2","ALT_STD"]
            target_name = "PACK2_COMPR_T"

        need = self.input_names + [target_name]
        miss = [c for c in need if c not in n2i]
        if miss:
            raise ValueError(f"缺列: {miss}\n当前列: {names}")

        self.idx_x = [n2i[n] for n in self.input_names]
        self.idx_y = n2i[target_name]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        arr = self.base[idx]  # [L, D]
        L = arr.shape[0]
        x_feats = torch.from_numpy(arr[:, self.idx_x]).float()          # [L,6]
        y = torch.from_numpy(arr[:, self.idx_y:self.idx_y+1]).float()   # [L,1]
        zero = torch.zeros((L, 1), dtype=torch.float32)
        x = torch.cat([x_feats, zero], dim=1)                           # [L,7]
        return x.unsqueeze(0), y.unsqueeze(0), torch.tensor([idx], dtype=torch.long)
