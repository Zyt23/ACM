# data_provider/data_loader_acm_320.py
# -*- coding: utf-8 -*-
import os
import time
import hashlib
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from iotdb.table_session import TableSession, TableSessionConfig
from tqdm import tqdm
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# 0. 统一基于本文件位置的路径
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
    print(f"[WARN] 未找到 faults CSV: {FAULT_CSV_PATH}，视为无故障机号。")
    fault_dates_by_tail_pack1 = {}
    fault_dates_by_tail_pack2 = {}


def _get_fault_dates_for_tail_side(tail_num: str, side_key: str):
    if side_key == "PACK1":
        return fault_dates_by_tail_pack1.get(tail_num, [])
    elif side_key == "PACK2":
        return fault_dates_by_tail_pack2.get(tail_num, [])
    return []


# =========================================================
# 3. 参数列表：PACK1 / PACK2（完全删除 ALT_STD）
# =========================================================
PACK1_PARA = [
    "PACK1_BYPASS_V",
    "PACK1_DISCH_T",
    "PACK1_RAM_I_DR",
    "PACK1_RAM_O_DR",
    "PACK_FLOW_R1",
    "PACK1_COMPR_T",
]
PACK2_PARA = [
    "PACK2_BYPASS_V",
    "PACK2_DISCH_T",
    "PACK2_RAM_I_DR",
    "PACK2_RAM_O_DR",
    "PACK_FLOW_R2",
    "PACK2_COMPR_T",
]
PARAMS_BY_SIDE = {"PACK1": PACK1_PARA, "PACK2": PACK2_PARA}


# =========================================================
# 4. 插值工具
# =========================================================
def interpolate_to_grid(df_src: pd.DataFrame, grid_time_ns: np.ndarray, col_name: str):
    if len(df_src) == 0:
        return np.full(len(grid_time_ns), np.nan, dtype=float)

    x = df_src["time"].values.astype(np.int64)
    y = df_src[col_name].values.astype(float)

    uniq, idx = np.unique(x, return_index=True)
    x = uniq
    y = y[idx]

    return np.interp(grid_time_ns.astype(np.int64), x, y, left=y[0], right=y[-1])


# =========================================================
# 5. 时间切分规则（你的版本）
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
# 6. verbose print（可控）
# =========================================================
def _vprint(args, *msg):
    if not bool(getattr(args, "verbose_raw", False)):
        return
    flush = bool(getattr(args, "verbose_flush", True))
    print(*msg, flush=flush)


def _dprint(args, *msg):
    if not bool(getattr(args, "verbose_ds", True)):
        return
    flush = bool(getattr(args, "verbose_flush", True))
    print(*msg, flush=flush)


def _fmt_dt_ns(ns: int):
    try:
        return str(pd.to_datetime(int(ns), unit="ns", utc=True).tz_convert("Asia/Shanghai"))
    except Exception:
        return "NA"


# =========================================================
# 7. raw npz 缓存（tail+side+raw_range）
# =========================================================
def _feat_hash(feat_cols):
    s = ",".join([str(c) for c in feat_cols]).encode("utf-8")
    return hashlib.md5(s).hexdigest()[:10]


def _raw_cache_dir():
    d = os.path.join(_PROJECT_ROOT, "cache", "acm_raw_2y_v2_no_object")
    os.makedirs(d, exist_ok=True)
    return d


def get_raw_2y_range_for_tail(
    tail_num: str,
    side: str,
    raw_months: int,
    fault_gap_months: int,
    anchor_end_str: str,
    raw_end_use_gap: bool = False,
):
    raw_months = int(max(1, raw_months))
    fdates = _get_fault_dates_for_tail_side(tail_num, side)
    if fdates:
        fd = pd.Timestamp(min(fdates)).normalize()
        raw_end = (fd - pd.DateOffset(months=int(fault_gap_months))).normalize() if raw_end_use_gap else fd
    else:
        raw_end = pd.Timestamp(anchor_end_str).normalize()
    raw_start = (raw_end - pd.DateOffset(months=raw_months)).normalize()
    return raw_start, raw_end


def load_or_build_raw_npz_2y(
    session,
    args,
    tail_num: str,
    side: str,
    feat_cols: list,
    raw_start: pd.Timestamp,
    raw_end: pd.Timestamp,
    gap_threshold_sec: float,
):
    cache_root = _raw_cache_dir()
    fhash = _feat_hash(feat_cols)

    raw_start = pd.Timestamp(raw_start).normalize()
    raw_end = pd.Timestamp(raw_end).normalize()

    start_ts_ns = pd.Timestamp(raw_start, tz="Asia/Shanghai").value
    end_ts_ns = pd.Timestamp(raw_end, tz="Asia/Shanghai").value

    npz_path = os.path.join(
        cache_root,
        f"{side}_{tail_num}_raw{raw_start.strftime('%Y%m%d')}_{raw_end.strftime('%Y%m%d')}_gap{int(gap_threshold_sec)}_{fhash}.npz"
    )

    if os.path.exists(npz_path):
        t0 = time.time()
        try:
            npz = np.load(npz_path, allow_pickle=False)
            time_ns = npz["time_ns"].astype(np.int64)
            X = npz["X"].astype(np.float32)

            fn = npz.get("feature_names", None)
            if fn is None:
                raise ValueError("missing feature_names")
            if fn.dtype == object:
                raise ValueError("legacy cache: feature_names is object dtype")
            feat_saved = [str(x) for x in fn.tolist()]

            sv = npz.get("schema_version", None)
            if sv is not None and int(sv) != 1:
                raise ValueError(f"schema_version mismatch: {sv}")

            if X.ndim != 2 or X.shape[1] != len(feat_saved):
                raise ValueError(f"shape mismatch: X={X.shape} vs len(feature_names)={len(feat_saved)}")

            t1 = time.time()
            _vprint(args, f"[RAW][{tail_num}][{side}] HIT cache | T={len(time_ns)} D={X.shape[1]} | load={t1-t0:.3f}s")
            return time_ns, X, feat_saved

        except Exception as e:
            _vprint(args, f"[RAW][{tail_num}][{side}] cache invalid -> remove & rebuild | {repr(e)}")
            try:
                os.remove(npz_path)
            except Exception:
                pass

    master_param = "PACK1_DISCH_T" if side == "PACK1" else "PACK2_DISCH_T"

    master_query = f"""
        SELECT time, value
        FROM {master_param}
        WHERE "aircraft/tail" = '{tail_num}'
        AND TIME >= {start_ts_ns} AND TIME < {end_ts_ns}
        ORDER BY TIME
    """
    _vprint(args, f"[RAW][{tail_num}][{side}] MISS -> query master={master_param} | {raw_start}~{raw_end}")
    t_master0 = time.time()
    master_df = session.execute_query_statement(master_query).todf()
    t_master1 = time.time()
    _vprint(args, f"[RAW][{tail_num}][{side}] master done | rows={len(master_df)} | {t_master1-t_master0:.3f}s")

    if len(master_df) == 0:
        np.savez(
            npz_path,
            time_ns=np.array([], dtype=np.int64),
            X=np.zeros((0, len(feat_cols)), dtype=np.float32),
            feature_names=np.array([str(c) for c in feat_cols], dtype=np.str_),
            schema_version=np.int32(1),
        )
        return np.array([], dtype=np.int64), np.zeros((0, len(feat_cols)), dtype=np.float32), [str(c) for c in feat_cols]

    master_df.columns = ["time", master_param]
    master_df["time"] = pd.to_datetime(master_df["time"], utc=True).dt.tz_convert("Asia/Shanghai")
    times = master_df["time"].values
    time_ns = times.view("i8").astype(np.int64)

    raw_cols = {}
    raw_cols[master_param] = master_df[master_param].astype(float).to_numpy()

    every_n = int(max(1, getattr(args, "verbose_every_n_param", 1)))

    for pi, param in enumerate(feat_cols):
        if param == master_param:
            continue

        if getattr(args, "verbose_raw", False) and (pi % every_n == 0):
            _vprint(args, f"[RAW][{tail_num}][{side}] param({pi+1}/{len(feat_cols)}): {param}")

        q = f"""
            SELECT time, value
            FROM {param}
            WHERE "aircraft/tail" = '{tail_num}'
            AND TIME >= {start_ts_ns} AND TIME < {end_ts_ns}
            ORDER BY TIME
        """
        t0 = time.time()
        try:
            dfp = session.execute_query_statement(q).todf()
            t1 = time.time()
            _vprint(args, f"[RAW][{tail_num}][{side}] {param} rows={len(dfp)} | query={t1-t0:.3f}s")

            if len(dfp) == 0:
                raw_cols[param] = np.full(len(time_ns), np.nan, dtype=float)
                continue

            dfp.columns = ["time", param]
            dfp["time"] = pd.to_datetime(dfp["time"], utc=True).dt.tz_convert("Asia/Shanghai")
            dfp["time"] = dfp["time"].values.view("i8")
            tmp = dfp[["time", param]].copy()

            t2 = time.time()
            raw_cols[param] = interpolate_to_grid(tmp, time_ns, param)
            t3 = time.time()
            _vprint(args, f"[RAW][{tail_num}][{side}] {param} interp={t3-t2:.3f}s")

        except Exception as e:
            t1 = time.time()
            _vprint(args, f"[RAW][{tail_num}][{side}] {param} ERROR after {t1-t0:.3f}s | {repr(e)}")
            raw_cols[param] = np.full(len(time_ns), np.nan, dtype=float)

    X = np.stack(
        [raw_cols.get(c, np.full(len(time_ns), np.nan, dtype=float)) for c in feat_cols],
        axis=1
    ).astype(np.float32)

    np.savez(
        npz_path,
        time_ns=time_ns.astype(np.int64),
        X=X.astype(np.float32),
        feature_names=np.array([str(c) for c in feat_cols], dtype=np.str_),
        schema_version=np.int32(1),
    )
    _vprint(args, f"[RAW][{tail_num}][{side}] saved npz: {npz_path} | X.shape={X.shape}")

    return time_ns, X, [str(c) for c in feat_cols]


def slice_raw_by_time_range(
    time_ns: np.ndarray,
    X: np.ndarray,
    start_ts_ns: int,
    end_ts_ns: int,
    args=None,
    tail="",
    side="",
    mode="",
):
    if len(time_ns) == 0:
        if args is not None:
            _dprint(args, f"[SLICE][{tail}][{side}][{mode}] raw empty -> slice empty")
        return time_ns, X

    l = int(np.searchsorted(time_ns, start_ts_ns, side="left"))
    r = int(np.searchsorted(time_ns, end_ts_ns, side="left"))
    l2 = max(0, min(l, len(time_ns)))
    r2 = max(0, min(r, len(time_ns)))

    if args is not None:
        _dprint(args, f"[SLICE][{tail}][{side}][{mode}] start={_fmt_dt_ns(start_ts_ns)} end={_fmt_dt_ns(end_ts_ns)}")
        _dprint(args, f"[SLICE][{tail}][{side}][{mode}] searchsorted l={l} r={r} | clipped l2={l2} r2={r2} | raw_T={len(time_ns)}")
        _dprint(args, f"[SLICE][{tail}][{side}][{mode}] raw_first={_fmt_dt_ns(time_ns[0])} raw_last={_fmt_dt_ns(time_ns[-1])}")

        if start_ts_ns >= time_ns[-1]:
            _dprint(args, f"[SLICE][{tail}][{side}][{mode}] !!! mode_start >= raw_last -> WILL BE EMPTY")
        if end_ts_ns <= time_ns[0]:
            _dprint(args, f"[SLICE][{tail}][{side}][{mode}] !!! mode_end <= raw_first -> WILL BE EMPTY")

    if r2 <= l2:
        if args is not None:
            _dprint(args, f"[SLICE][{tail}][{side}][{mode}] RESULT empty (r2<=l2)")
        return np.array([], dtype=np.int64), np.zeros((0, X.shape[1]), dtype=X.dtype)

    out_t = time_ns[l2:r2]
    out_x = X[l2:r2]
    if args is not None:
        _dprint(args, f"[SLICE][{tail}][{side}][{mode}] RESULT T={len(out_t)} | first={_fmt_dt_ns(out_t[0])} last={_fmt_dt_ns(out_t[-1])}")
    return out_t, out_x


def recompute_segments_from_time(time_ns: np.ndarray, gap_threshold_sec: float):
    if len(time_ns) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    t_s = (time_ns.astype(np.float64) / 1e9)
    segments = []
    start_idx = 0
    for i in range(1, len(time_ns)):
        if (t_s[i] - t_s[i - 1]) > gap_threshold_sec:
            segments.append((start_idx, i))
            start_idx = i
    segments.append((start_idx, len(time_ns)))
    return np.array(segments, dtype=np.int32)


# =========================================================
# 8. 航段开始可视化（多变量）
# =========================================================
def plot_segment_starts(
    time_ns: np.ndarray,
    X: np.ndarray,
    segments: np.ndarray,
    feature_names: list,
    out_png: str,
    n_segments: int = 3,
    n_steps: int = 96 * 5,
    tail: str = "",
    mode: str = "",
    side: str = "",
):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    if len(time_ns) == 0 or X.shape[0] == 0 or len(segments) == 0:
        return

    segs = segments[:n_segments]
    D = X.shape[1]
    S = len(segs)

    fig, axes = plt.subplots(D, S, figsize=(4.5 * S, 2.0 * D), sharex=False)
    if D == 1 and S == 1:
        axes = np.array([[axes]])
    elif D == 1:
        axes = axes.reshape(1, -1)
    elif S == 1:
        axes = axes.reshape(-1, 1)

    for j, (s, e) in enumerate(segs):
        s = int(s); e = int(e)
        take_end = min(e, s + n_steps)
        idx = np.arange(s, take_end)
        x = np.arange(len(idx))

        for i in range(D):
            ax = axes[i, j]
            ax.plot(x, X[idx, i])
            ax.grid(True, alpha=0.3)
            if i == 0:
                t0 = pd.to_datetime(time_ns[s], unit="ns", utc=True).tz_convert("Asia/Shanghai")
                ax.set_title(f"seg#{j+1} start={t0}")
            if j == 0:
                ax.set_ylabel(feature_names[i])
            if i == D - 1:
                ax.set_xlabel("step from seg start")

    fig.suptitle(f"{tail} | {side} | {mode} | seg starts ({n_steps} steps)", y=0.995)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)


def print_first_steps(
    time_ns: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    n_steps: int = 96 * 5,
    tail: str = "",
    side: str = "",
    mode: str = "",
):
    if len(time_ns) == 0 or X.shape[0] == 0:
        print(f"[Preview] empty series for {tail} {side} {mode}")
        return
    take = min(int(n_steps), len(time_ns))
    t = pd.to_datetime(time_ns[:take], unit="ns", utc=True).tz_convert("Asia/Shanghai")
    df = pd.DataFrame(X[:take, :], columns=feature_names)
    df.insert(0, "time", t.astype(str))
    print(f"\n[Preview] {tail} {side} {mode} first {take} rows:")
    print(df.head(take).to_string(index=False))


# =========================================================
# 9. A320 PACK 数据集：FlightDataset_acm（支持 PACK1 / PACK2）
#    只保留每个航段前 keep_len = max_windows_per_flight * seq_len 点
# =========================================================
class FlightDataset_acm(Dataset):
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

        # args 参数
        self.train_months = int(getattr(self.args, "normal_months", 2))
        self.test_normal_months = int(getattr(self.args, "test_normal_months", 1))
        self.fault_gap_months = int(getattr(self.args, "fault_gap_months", 6))
        self.normal_anchor_end = str(getattr(self.args, "normal_anchor_end", "2024-01-01"))

        # raw 缓存范围
        self.raw_months = int(getattr(self.args, "raw_months", 24))
        self.raw_end_use_gap = bool(getattr(self.args, "raw_end_use_gap", False))

        # keep_len（每航段只保存前 max_windows_per_flight*seq_len 点）
        base_seq_len = int(getattr(self.args, "seq_len", 96))  # 仍然是 96
        self.max_windows_per_flight = int(getattr(self.args, "max_windows_per_flight", 5))
        self.keep_len = base_seq_len * self.max_windows_per_flight  # 例如 480

        cache_dir = os.path.join(_PROJECT_ROOT, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        cache_version = "vA3_seghead_keepKx96_noALTSTD_no_object"
        anchor_safe = self.normal_anchor_end.replace("-", "")
        self.dataset_name = (
            f"A320_keep{self.max_windows_per_flight}x{base_seq_len}_L{self.keep_len}_"
            f"raw{self.raw_months}M_"
            f"train{self.train_months}M_test{self.test_normal_months}M_gap{self.fault_gap_months}M_"
            f"anchor{anchor_safe}_{cache_version}_{self.side}_{len(self.all_planes)}planes_"
            + "_".join(self.para)
        )

        # seghead cache（避免撞旧 cache）
        data_path = os.path.join(cache_dir, f"{self.dataset_name}_{Tag}_seghead_data.npy")
        feat_path = os.path.join(cache_dir, f"{self.dataset_name}_{Tag}_feature_names.npy")
        meta_time_path = os.path.join(cache_dir, f"{self.dataset_name}_{Tag}_seghead_start_times.npy")
        meta_tail_path = os.path.join(cache_dir, f"{self.dataset_name}_{Tag}_seghead_tails.npy")

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
                self.feature_names = [str(x) for x in np.load(feat_path, allow_pickle=False).tolist()]
            else:
                self.feature_names = list(self.para)

            self.window_start_times = [str(x) for x in np.load(meta_time_path, allow_pickle=False).tolist()]
            self.window_tails = [str(x) for x in np.load(meta_tail_path, allow_pickle=False).tolist()]
            t_load1 = time.time()
            print(
                f"[Cache] 使用 seghead cache: {data_path} (Tag={Tag}, side={self.side}) | "
                f"{t_load1 - t_load0:.3f}s | data.shape={self.data.shape} | meta_len={len(self.window_start_times)}"
            )
            return

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
            np.save(feat_path, np.array(getattr(self, "feature_names", []), dtype=np.str_))
            np.save(meta_time_path, np.array([], dtype=np.str_))
            np.save(meta_tail_path, np.array([], dtype=np.str_))
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
        np.save(feat_path, np.array([str(x) for x in self.feature_names], dtype=np.str_))

        if hasattr(self, "window_start_times") and hasattr(self, "window_tails"):
            np.save(meta_time_path, np.array([str(t) for t in self.window_start_times], dtype=np.str_))
            np.save(meta_tail_path, np.array([str(x) for x in self.window_tails], dtype=np.str_))
        else:
            np.save(meta_time_path, np.array([], dtype=np.str_))
            np.save(meta_tail_path, np.array([], dtype=np.str_))

    def _flight_data(self, tail_list, para_list, mode):
        """
        流程：
          1) 统一 raw 范围缓存（tail+side+raw_range）
          2) 按 mode 裁剪
          3) 裁剪后重切航段（gap_threshold_sec）
          4) 每航段只保留起始 keep_len = max_windows_per_flight*seq_len 点（1条/航段）
        """
        all_seqs = []
        seg_start_times = []
        seg_tails = []
        feature_names_once = None

        max_windows_per_flight = int(getattr(self.args, "max_windows_per_flight", 5))
        gap_threshold_sec = float(getattr(self.args, "flight_gap_threshold_sec", 3600.0))
        seq_len = int(getattr(self.args, "seq_len", 96))  # base window=96
        keep_len = int(seq_len * max_windows_per_flight)  # 例如 480

        debug_plot_tail = str(getattr(self.args, "debug_plot_tail", ""))
        debug_plot_mode = str(getattr(self.args, "debug_plot_mode", mode))
        debug_plot_nseg = int(getattr(self.args, "debug_plot_n_segments", 3))
        debug_plot_steps = int(getattr(self.args, "debug_plot_steps", keep_len))

        for tail_num in tqdm(tail_list, desc=f"[A320-{mode}-{self.side}] Fetching..."):
            print(f"\n--- [{tail_num}] 开始处理 (mode={mode}, side={self.side}) ---")
            _dprint(self.args, f"[DS][{tail_num}][{self.side}][{mode}] train_months={self.train_months} test_months={self.test_normal_months} gap_months={self.fault_gap_months} anchor_end={self.normal_anchor_end}")
            _dprint(self.args, f"[DS][{tail_num}][{self.side}][{mode}] raw_months={self.raw_months} raw_end_use_gap={self.raw_end_use_gap} gap_threshold_sec={gap_threshold_sec} seq_len={seq_len} keep_len={keep_len}")

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

            start_ts_ns_mode = pd.Timestamp(start_date, tz="Asia/Shanghai").value
            end_ts_ns_mode = pd.Timestamp(end_date, tz="Asia/Shanghai").value
            print(f"  mode范围: {start_date} ~ {end_date} | keep_len={keep_len}")

            raw_start, raw_end = get_raw_2y_range_for_tail(
                tail_num=tail_num,
                side=self.side,
                raw_months=self.raw_months,
                fault_gap_months=self.fault_gap_months,
                anchor_end_str=self.normal_anchor_end,
                raw_end_use_gap=self.raw_end_use_gap,
            )
            print(f"  raw范围:  {raw_start} ~ {raw_end} (raw_months={self.raw_months}, raw_end_use_gap={self.raw_end_use_gap})")

            t_raw0 = time.time()
            time_ns_raw, X_raw, feat_cols = load_or_build_raw_npz_2y(
                session=self.session,
                args=self.args,
                tail_num=tail_num,
                side=self.side,
                feat_cols=para_list,
                raw_start=raw_start,
                raw_end=raw_end,
                gap_threshold_sec=gap_threshold_sec,
            )
            t_raw1 = time.time()
            print(f"[Perf][{tail_num}] raw load/build: {t_raw1 - t_raw0:.3f}s | T_raw={len(time_ns_raw)}")

            if len(time_ns_raw) > 0:
                _dprint(self.args, f"[RAWSTAT][{tail_num}][{self.side}] raw_first={_fmt_dt_ns(time_ns_raw[0])} raw_last={_fmt_dt_ns(time_ns_raw[-1])}")

            if len(time_ns_raw) == 0 or X_raw.shape[0] == 0:
                print(f"  !!! raw为空, 跳过 {tail_num}")
                continue

            time_ns, X = slice_raw_by_time_range(
                time_ns_raw, X_raw,
                start_ts_ns_mode, end_ts_ns_mode,
                args=self.args, tail=tail_num, side=self.side, mode=mode
            )
            print(f"[Perf][{tail_num}] mode slice: T={len(time_ns)}")

            if len(time_ns) < keep_len:
                print(f"  !!! 裁剪后长度<{keep_len}, 跳过")
                continue

            segments = recompute_segments_from_time(time_ns, gap_threshold_sec=gap_threshold_sec)
            seg_lens = [int(e) - int(s) for (s, e) in segments.tolist()]
            if len(seg_lens) > 0:
                _dprint(self.args, f"[SEG][{tail_num}][{self.side}][{mode}] segments_total={len(seg_lens)} | len_min={min(seg_lens)} len_med={int(np.median(seg_lens))} len_max={max(seg_lens)}")
                for ii, (s, e) in enumerate(segments.tolist()[:5]):
                    s = int(s); e = int(e)
                    _dprint(self.args, f"[SEG][{tail_num}] seg#{ii} idx=[{s},{e}) len={e-s} t0={_fmt_dt_ns(time_ns[s])} t1={_fmt_dt_ns(time_ns[e-1])}")

            # 关键：只保留能提供 keep_len 的航段
            valid_segments = [(int(s), int(e)) for (s, e) in segments.tolist() if (int(e) - int(s)) >= keep_len]
            print(f"[Perf][{tail_num}] segments={len(valid_segments)} after filter(>={keep_len})")
            if len(valid_segments) == 0:
                continue

            if feature_names_once is None:
                feature_names_once = list(feat_cols)
                self.feature_names = feature_names_once

            if debug_plot_tail and str(tail_num) == debug_plot_tail and str(mode) == debug_plot_mode:
                out_png = os.path.join(_PROJECT_ROOT, "cache", "debug_plots",
                                       f"segstart_{tail_num}_{self.side}_{mode}_steps{debug_plot_steps}.png")
                plot_segment_starts(
                    time_ns=time_ns,
                    X=X,
                    segments=np.array(valid_segments, dtype=np.int32),
                    feature_names=feat_cols,
                    out_png=out_png,
                    n_segments=debug_plot_nseg,
                    n_steps=debug_plot_steps,
                    tail=str(tail_num),
                    mode=str(mode),
                    side=str(self.side),
                )
                print(f"[DebugPlot] saved: {out_png}")

            total_kept_this_tail = 0
            for (s, e) in valid_segments:
                head = X[s:s + keep_len, :]  # [keep_len, D] 例如 [480,6]
                all_seqs.append(head.astype(np.float32))

                t0 = pd.to_datetime(time_ns[s], unit="ns", utc=True).tz_convert("Asia/Shanghai")
                seg_start_times.append(t0)
                seg_tails.append(tail_num)
                total_kept_this_tail += 1

            print(f"[Perf][{tail_num}] segheads_kept={total_kept_this_tail}")

        # 兼容你原字段名（用于保存 meta）
        self.window_start_times = seg_start_times
        self.window_tails = seg_tails
        print(f"[Perf] 总seghead数 (mode={mode}, side={self.side}) = {len(all_seqs)}")

        if len(all_seqs) == 0:
            if not hasattr(self, "feature_names"):
                self.feature_names = list(self.para)
            return np.zeros((0, keep_len, len(self.feature_names)), dtype=np.float32)

        return np.stack(all_seqs, axis=0)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


# =========================================================
# 10) 24->24 Wrapper：从 seghead keep_len 切 24->24
# =========================================================
class Dataset_Forecast24to24_FromSegHead(Dataset):
    """
    base_dataset: FlightDataset_acm
      - 每条样本是 [keep_len, 6]（例如 keep_len=480）
    输出:
      x: [1, 24, 6]
      y: [1, 24, 1]  (未来24点的 PACKx_COMPR_T)
    """
    def __init__(self, base_dataset: FlightDataset_acm, in_len=24, out_len=24, stride=24):
        super().__init__()
        self.base = base_dataset
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.stride = int(stride)

        names = getattr(self.base, "feature_names", [])
        if not names:
            raise ValueError("base_dataset.feature_names 为空。")
        n2i = {n: i for i, n in enumerate(names)}

        if self.base.side == "PACK1":
            self.input_names = [
                "PACK1_BYPASS_V", "PACK1_DISCH_T", "PACK1_RAM_I_DR",
                "PACK1_RAM_O_DR", "PACK_FLOW_R1", "PACK1_COMPR_T",
            ]
            target_name = "PACK1_DISCH_T"
        else:
            self.input_names = [
                "PACK2_BYPASS_V", "PACK2_DISCH_T", "PACK2_RAM_I_DR",
                "PACK2_RAM_O_DR", "PACK_FLOW_R2", "PACK2_COMPR_T",
            ]
            target_name = "PACK2_DISCH_T"

        miss = [c for c in (self.input_names + [target_name]) if c not in n2i]
        if miss:
            raise ValueError(f"缺列: {miss} | 当前: {names}")

        self.idx_x = [n2i[n] for n in self.input_names]
        self.idx_y = n2i[target_name]

        if len(self.base) > 0:
            self.base_len = int(self.base.data.shape[1])
        else:
            self.base_len = self.in_len + self.out_len

        if self.base_len < self.in_len + self.out_len:
            raise ValueError(f"base_len={self.base_len} < {self.in_len + self.out_len}")

        # 一个 seghead 能切出多少个 24->24
        self.n_sub = 1 + (self.base_len - (self.in_len + self.out_len)) // self.stride

    def __len__(self):
        return len(self.base) * self.n_sub

    def __getitem__(self, idx):
        base_idx = idx // self.n_sub
        sub_id = idx % self.n_sub
        st = sub_id * self.stride

        arr = self.base[base_idx]  # [base_len, D]
        x = arr[st:st + self.in_len, self.idx_x]  # [24,6]
        y = arr[st + self.in_len:st + self.in_len + self.out_len, self.idx_y]  # [24]

        x = torch.from_numpy(x).float().unsqueeze(0)                # [1,24,6]
        y = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(-1)  # [1,24,1]

        packed = torch.tensor([base_idx * 100 + sub_id], dtype=torch.long)
        return x, y, packed


# =========================================================
# 11) 轻量预览 main（可选）
# =========================================================
def _make_session():
    t0 = time.time()
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
    t1 = time.time()
    print(f"[IoTDB] session connect {t1-t0:.3f}s")
    session.execute_non_query_statement("USE a320_ata21")
    return session


def preview_tails_without_building_dataset(
    tails,
    side="PACK2",
    mode="train_normal",
    seq_len=96,
    max_windows_per_flight=5,
    n_steps=96 * 5,
    raw_months=24,
    fault_gap_months=6,
    test_months=1,
    train_months=12,
    anchor_end="2024-08-01",
    gap_threshold_sec=3600.0,
    out_dir=None,
    verbose=True,
):
    class Args:
        pass

    args = Args()
    args.verbose_raw = bool(verbose)
    args.verbose_ds = True
    args.verbose_flush = True
    args.verbose_every_n_param = 1

    session = _make_session()

    para_list = PARAMS_BY_SIDE[side]
    out_dir = out_dir or os.path.join(_PROJECT_ROOT, "cache", "preview_plots")
    os.makedirs(out_dir, exist_ok=True)

    keep_len = int(seq_len * max_windows_per_flight)

    for tail in tails:
        print("\n==============================")
        print(f"[Preview] tail={tail} side={side} mode={mode}")
        print("==============================")

        start_date, end_date = get_time_range_for_tail(
            tail_num=tail,
            mode=mode if mode in ("train_normal", "test_normal", "abnormal") else "train_normal",
            side_key=side,
            train_months=int(train_months),
            test_months=int(test_months),
            gap_months=int(fault_gap_months),
            anchor_end_str=str(anchor_end),
        )
        if start_date is None:
            print("[Preview] no valid date range (likely no fault for abnormal). skip")
            continue

        start_ts_ns_mode = pd.Timestamp(start_date, tz="Asia/Shanghai").value
        end_ts_ns_mode = pd.Timestamp(end_date, tz="Asia/Shanghai").value
        print(f"[Preview] mode range: {start_date} ~ {end_date}")

        raw_start, raw_end = get_raw_2y_range_for_tail(
            tail_num=tail,
            side=side,
            raw_months=int(raw_months),
            fault_gap_months=int(fault_gap_months),
            anchor_end_str=str(anchor_end),
            raw_end_use_gap=False,
        )
        print(f"[Preview] raw range:  {raw_start} ~ {raw_end}")

        time_ns_raw, X_raw, feat_cols = load_or_build_raw_npz_2y(
            session=session,
            args=args,
            tail_num=tail,
            side=side,
            feat_cols=para_list,
            raw_start=raw_start,
            raw_end=raw_end,
            gap_threshold_sec=float(gap_threshold_sec),
        )
        print(f"[Preview] raw loaded: T={len(time_ns_raw)} X={X_raw.shape}")

        time_ns, X = slice_raw_by_time_range(
            time_ns_raw, X_raw,
            start_ts_ns_mode, end_ts_ns_mode,
            args=args, tail=tail, side=side, mode=mode
        )
        print(f"[Preview] sliced: T={len(time_ns)} X={X.shape}")
        if len(time_ns) == 0:
            continue

        print_first_steps(
            time_ns=time_ns,
            X=X,
            feature_names=feat_cols,
            n_steps=int(n_steps),
            tail=tail,
            side=side,
            mode=mode,
        )

        segments = recompute_segments_from_time(time_ns, gap_threshold_sec=float(gap_threshold_sec))
        valid = [(int(s), int(e)) for (s, e) in segments.tolist() if (int(e) - int(s)) >= int(keep_len)]
        print(f"[Preview] segments total={len(segments)} valid(>={keep_len})={len(valid)}")
        if len(valid) == 0:
            continue

        out_png = os.path.join(out_dir, f"segstart_{tail}_{side}_{mode}_steps{int(n_steps)}.png")
        plot_segment_starts(
            time_ns=time_ns,
            X=X,
            segments=np.array(valid, dtype=np.int32),
            feature_names=feat_cols,
            out_png=out_png,
            n_segments=3,
            n_steps=int(n_steps),
            tail=tail,
            mode=mode,
            side=side,
        )
        print(f"[Preview] plot saved: {out_png}")


if __name__ == "__main__":
    preview_tails_without_building_dataset(
        tails=["B-301A", "B-301G"],
        side="PACK2",
        mode="train_normal",
        seq_len=96,
        max_windows_per_flight=5,
        n_steps=96 * 5,
        raw_months=24,
        fault_gap_months=6,
        test_months=1,
        train_months=12,
        anchor_end="2025-08-01",
        gap_threshold_sec=3600.0,
        out_dir=os.path.join(_PROJECT_ROOT, "cache", "preview_plots"),
        verbose=True,
    )
