# "POSRAIL*POSRAIR"#冲压进气活门位置
# "TOUTPHXL*TOUTPHXR"#主级热交换器出口温度
# "TOUTSHXL*TOUTSHXR"#次级热交换器出口温度
# "TINPHXL*TINPHXR"#主级级热交换器进口温度
# "TOUTCPRSRL*TOUTCPRSRR"#压气机出口温度
# "POSRAEL*POSRAER"#冲压排气活门位置
# "TINTURB2L*TINTURB2R"#二级涡轮进口温度
# "POSTBVL*POSTBVR"#涡轮旁通活门
# "TOUTPACKL*TOUTPACKR#组件出口温度
# "TINCONDL*TINCONDR"#冷凝器进口温度
# "ECVCLOSED_L*ECVCLOSED_R"#经济冷却活门
# "POSLVLVL*POSLVLVR"#低限活门

import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import os
from iotdb.table_session import TableSession, TableSessionConfig

# 飞机尾号列表
train_left_ac_normal = [
        "B-2008", "B-2007", "B-2029", "B-2009", "B-20DM",
        "B-226D", "B-226C", "B-20C5",  "B-2081",
        "B-2042", "B-223N", "B-2041", "B-209Y", "B-2048",
        "B-2026", "B-2028", "B-2027", "B-2049", "B-20CK",
        "B-20EM", "B-222W", "B-20EN", "B-223G", "B-7185",
]
train_right_ac_normal = [
        "B-2008", "B-2007", "B-2029", "B-2009", "B-20DM",
        "B-226D", "B-226C", "B-20C5", "B-2081",
        "B-2042", "B-223N", "B-2041", "B-209Y", "B-2048",
        "B-2026", "B-2028", "B-2027", "B-2049", "B-20CK",
        "B-20EM", "B-222W", "B-20EN", "B-223G", "B-7185",
]

val_left_ac_normal = [
        "B-2099", "B-2010", "B-20AC", "B-7588"
]
val_right_ac_normal = [
        "B-2099", "B-2010", "B-20AC", "B-7588"
]

test_left_normal = [
        "B-7183", "B-2073", "B-2072", "B-2075",
]
test_right_normal = [
        "B-7183", "B-2073", "B-2072", "B-2075",
]

test_left_abnormal = [
    "B-2080"
]
test_right_abnormal = [

]

# 简化飞机列表
# 参数列表
para_l = [ # 左剖面
"POSRAIL", # 冲压进气活门位置 是否只用于位置筛选
"POSTBVL", # 涡轮旁通活门位置

"ALT_STD",#飞行高度，但有的会出现负值-200，有的会长时间出现1400，估计是跟海拔有关系，所以地面工况写的筛选条件是

"TINTURB2L", # 二级涡轮进口温度
"TOUTPACKL",  # 组件出口温度
#"TDIFFTURB2L", # 涡轮降温 人工后期做的，"TOUTPACKL"-"TINTURB2L"

"TOUTCPRSRL", # 压气机出口温度
"TOUTPHXL", # 主级热交换器出口温度
#"TDIFFCPRSRL",#压气机升温 人工后期做的，"TOUTPHXL"-"TOUTCPRSRL"

#"POSLVLVL", # 低限活门位置
# "TINPHXL", # 主级热交换器进口温度 主热集交换器进出口经历了什么？似乎只有风扇做工？冲压进气活门的开度应该是不是主要影响主机热交换器进出口的温度差？？
#"POSRAEL", # 冲压排气活门位置

]

para_r = [ # 左剖面
"POSRAIR", # 冲压进气活门位置 是否只用于位置筛选
"POSTBVR", # 涡轮旁通活门位置

"ALT_STD",#飞行高度，但有的会出现负值-200，有的会长时间出现1400，估计是跟海拔有关系，所以地面工况写的筛选条件是

"TINTURB2R", # 二级涡轮进口温度
"TOUTPACKR",  # 组件出口温度
#"TDIFFTURB2R", # 涡轮降温 人工后期做的，"TINTURB2R"-"TOUTPACKR"

"TOUTCPRSRR", # 压气机出口温度
"TOUTPHXR", # 主级热交换器出口温度
#"TDIFFCPRSRR",#压气机升温 人工后期做的，"TOUTCPRSRR-"TOUTPHXR""

#"POSLVLVR", # 低限活门位置 在北方的时候会不会也开启？考虑地面温度
# "TINPHXR", # 主级热交换器进口温度 主热集交换器进出口经历了什么？似乎只有风扇做工？冲压进气活门的开度应该是不是主要影响主机热交换器进出口的温度差？？
#"POSRAER", # 冲压排气活门位置

]
# 时间范围
time_start = '2025-04-01'
time_end = '2025-04-30'

import numpy as np
import pandas as pd


#确定是地面工况 考虑到海拔和波动 而且这个单位似乎不是米，应该是英尺？（飞行高度最高有30000）
def simple_ground_segments_with_time(
    t_ns: np.ndarray,
    alt: np.ndarray,
    alt_threshold: float = 3000.0,     # 地面高度阈值 ft
    max_climb_rate_fpm: float = 300.0, # 允许的最大爬升/下降速率 |dALT/dt| m/min
    min_segment_sec: int = 120         # 地面段最小持续时长（秒）
):
    """
    输入:
      t_ns: 时间戳(纳秒) 一维数组，单调递增
      alt:  ALT_STD 值（ft）
    返回:
      segments_all: 所有稳定地面段 [(s,e), ...]  时间用索引区间 [s, e) 表示
      segments_pre: 起飞前地面段（每个飞行块左侧相邻）
      segments_post: 落地后地面段（每个飞行块右侧相邻）
      mask_ground: 布尔掩码（与 alt 同长）
    """
    n = len(alt)
    if n == 0:
        return [], [], [], np.zeros(0, dtype=bool)

    # 采样间隔（秒）
    t_s = (t_ns.astype(np.int64) / 1e9).astype(float)
    dt = np.diff(t_s, prepend=t_s[0])
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0

    # 爬升/下降速率 ft/min
    dalt = np.diff(alt, prepend=alt[0])
    climb_rate_fpm = (dalt / dt) * 60.0

    mask_low = alt < alt_threshold
    mask_stable = np.abs(climb_rate_fpm) <= max_climb_rate_fpm
    mask = mask_low & mask_stable

    # 只保留连续时长 >= min_segment_sec 的 True 段
    segments_all = []
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        # 该段持续时长（秒）
        dur = t_s[j-1] - t_s[i]
        if dur >= min_segment_sec:
            segments_all.append((i, j))
        i = j

    # 重建最终 ground mask（只保留满足时长条件的段）
    mask_ground = np.zeros(n, dtype=bool)
    for s, e in segments_all:
        mask_ground[s:e] = True

    # 找飞行块（非地面）
    flight_blocks = []
    i = 0
    while i < n:
        if mask_ground[i]:
            i += 1
            continue
        j = i
        while j < n and not mask_ground[j]:
            j += 1
        # 过滤掉太短的“非地面抖动”
        if j - i >= 2:
            flight_blocks.append((i, j))
        i = j

    # 为每个飞行块找左/右相邻的地面段
    segments_pre, segments_post = [], []
    for f0, f1 in flight_blocks:
        left = next(((s, e) for (s, e) in segments_all if e == f0), None)
        right = next(((s, e) for (s, e) in segments_all if s == f1), None)
        if left:  segments_pre.append(left)
        if right: segments_post.append(right)

    return segments_all, segments_pre, segments_post, mask_ground


def interpolate_to_grid(df_src: pd.DataFrame, grid_time: np.ndarray, col_name: str):
    """
    把 df_src[col_name] 沿 df_src['time'] 线性插值到 grid_time（纳秒）
    返回 numpy 数组（与 grid_time 等长）
    """
    if len(df_src) == 0:
        return np.full(len(grid_time), np.nan, dtype=float)

    x = df_src['time'].values.astype(np.int64)
    y = df_src[col_name].values.astype(float)

    # 去重 & 保序
    uniq, idx = np.unique(x, return_index=True)
    x = uniq
    y = y[idx]

    # 对 grid 范围外做边缘延拓
    return np.interp(grid_time.astype(np.int64), x, y, left=y[0], right=y[-1])


    # 统一插值到 ALT 的时间轴# 派生变量安全相减
def _safe_diff(df: pd.DataFrame, name: str, a: str, b: str):
    if a in df.columns and b in df.columns:
        df[name] = df[a].astype(float) - df[b].astype(float)
    else:
        df[name] = np.nan


import torch
from torch.utils.data import Dataset

class Dataset_RegRight_TimerXL(Dataset):
    """
    右侧回归包装：
      X 通道0~2: [TDIFFTURB2R, POSTBVR, POSRAIR]
      X 通道3  : 常数0（防止目标泄露）
      y        : TDIFFCPRSRR
    返回:
      x: [1, L, 4]
      y: [1, L, 1]
      flight_index: [1]
    """
    def __init__(self, base_dataset):
        super().__init__()
        self.base = base_dataset
        names = getattr(self.base, 'feature_names', [])
        if not names:
            raise ValueError("base_dataset.feature_names 为空，请确认缓存或构建阶段已写入。")
        n2i = {n:i for i,n in enumerate(names)}
        need = ['TDIFFTURB2R','POSTBVR','POSRAIR','TDIFFCPRSRR']
        miss = [c for c in need if c not in n2i]
        if miss:
            raise ValueError(f"右侧所需列缺失: {miss}\n当前列: {names}")
        self.idx_x = [n2i['TDIFFTURB2R'], n2i['POSTBVR'], n2i['POSRAIR']]
        self.idx_y = n2i['TDIFFCPRSRR']

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        arr = self.base[idx]       # [L, D]
        L = arr.shape[0]
        x3 = torch.from_numpy(arr[:, self.idx_x]).float()     # [L,3]
        y  = torch.from_numpy(arr[:, self.idx_y:self.idx_y+1]).float()  # [L,1]
        zero = torch.zeros((L,1), dtype=torch.float32)
        x = torch.cat([x3, zero], dim=1)   # [L,4]
        return x.unsqueeze(0), y.unsqueeze(0), torch.tensor([idx], dtype=torch.long)


class Dataset_RegLeft_TimerXL(Dataset):
    """
    左侧回归包装：
      X 通道0~2: [TDIFFTURB2L, POSTBVL, POSRAIL]
      X 通道3  : 常数0
      y        : TDIFFCPRSRL
    """
    def __init__(self, base_dataset):
        super().__init__()
        self.base = base_dataset
        names = getattr(self.base, 'feature_names', [])
        if not names:
            raise ValueError("base_dataset.feature_names 为空，请确认缓存或构建阶段已写入。")
        n2i = {n:i for i,n in enumerate(names)}
        need = ['TDIFFTURB2L', 'POSTBVL', 'POSRAIL', 'TDIFFCPRSRL']
        miss = [c for c in need if c not in n2i]
        if miss:
            raise ValueError(f"左侧所需列缺失: {miss}\n当前列: {names}")
        self.idx_x = [n2i['TDIFFTURB2L'], n2i['POSTBVL'], n2i['POSRAIL']]
        self.idx_y = n2i['TDIFFCPRSRL']

    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        arr = self.base[idx]
        L = arr.shape[0]
        x3 = torch.from_numpy(arr[:, self.idx_x]).float()
        y  = torch.from_numpy(arr[:, self.idx_y:self.idx_y+1]).float()
        zero = torch.zeros((L,1), dtype=torch.float32)
        x = torch.cat([x3, zero], dim=1)
        return x.unsqueeze(0), y.unsqueeze(0), torch.tensor([idx], dtype=torch.long)

# ----------------- 数据集（分侧） -----------------
# ----------------- ACM 数据集（含归一化） -----------------
class FlightDataset_acm(Dataset):
    """
    side: 'R' or 'L' —— 只构建一侧的数据，不再 concat 左右
    返回: np.array [num_seq, seq_len, feat_dim]
    同时缓存 feature_names.npy，便于下游按名索引
    归一化策略：
      - 仅在 Tag=='train_normal' 上 fit 并保存 mean/std
      - 其他 Tag 优先复用 train_normal 的 mean/std；缺失则 fallback 到 self-normalization
    """
    def __init__(self, args, Tag, side='R'):
        super().__init__()
        self.args = args
        self.Tag = Tag
        self.side = side
        self.scaler = StandardScaler()

        # 连接 IoTDB
        try:
            config = TableSessionConfig(
                node_urls=["127.0.0.1:6667"],
                username="root",
                password="root",
                time_zone="Asia/Shanghai",
            )
            session = TableSession(config)
        except Exception:
            config = TableSessionConfig(
                node_urls=["10.254.43.34:6667"],
                username="root",
                password="root",
                time_zone="Asia/Shanghai",
            )
            session = TableSession(config)
        session.execute_non_query_statement("USE b777")
        self.session = session

        # 选择当前侧的飞机清单与参数表
        if self.side == "R":
            self.train_list = train_right_ac_normal
            self.val_list = val_right_ac_normal
            self.test_normal_list = test_right_normal
            self.test_abnormal_list = test_right_abnormal
            self.para = para_r
        else:
            self.train_list = train_left_ac_normal
            self.val_list = val_left_ac_normal
            self.test_normal_list = test_left_normal
            self.test_abnormal_list = test_left_abnormal
            self.para = para_l

        self.time_start = time_start
        self.time_end = time_end

        # 缓存名带 side
        self.dataset_name = (
            f"{self.side}_{len(self.train_list)}_planes_"
            + "_".join(self.para)
            + "_"
            + str(self.time_start)
            + "_"
            + str(self.time_end)
        )
        data_path = f"cache/{self.dataset_name}_{Tag}_data.npy"
        feat_path = f"cache/{self.dataset_name}_{Tag}_feature_names.npy"

        # 归一化统计缓存（只与 train_normal 绑定）
        scaler_tag = "train_normal"
        scaler_mean_path = f"cache/{self.dataset_name}_{scaler_tag}_mean.npy"
        scaler_std_path = f"cache/{self.dataset_name}_{scaler_tag}_std.npy"
        use_dataset_scale = getattr(self.args, "dataset_scale", True)

        if os.path.exists(data_path):
            # 直接加载已归一化后的数据
            self.data = np.load(data_path, allow_pickle=False)
            if os.path.exists(feat_path):
                self.feature_names = list(np.load(feat_path, allow_pickle=True))
            else:
                derived = (
                    ["TDIFFTURB2R", "TDIFFCPRSRR"]
                    if self.side == "R"
                    else ["TDIFFTURB2L", "TDIFFCPRSRL"]
                )
                self.feature_names = [c for c in self.para] + [
                    c for c in derived if c not in self.para
                ]
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print(f"Loading cached data from {data_path}")
        else:
            # 构建原始序列（未归一化）
            if Tag == "train_normal":
                self.data = self._flight_data(self.train_list, self.para)
            elif Tag == "val_normal":
                self.data = self._flight_data(self.val_list, self.para)
            elif Tag == "test_normal":
                self.data = self._flight_data(self.test_normal_list, self.para)
            elif Tag == "test_abnormal":
                self.data = self._flight_data(self.test_abnormal_list, self.para)
            else:
                raise ValueError(f"Unknown Tag: {Tag}")

            # ===== 归一化（按 prsov 策略，但考虑到 3D 张量结构） =====
            # self.data: [N_seq, L, C]  ——> 展平成 [N_seq*L, C] 拟合 per-feature 统计量
            os.makedirs("cache", exist_ok=True)

            if self.data.shape[0] == 0:
                # 空数据保护
                np.save(data_path, self.data)
                np.save(feat_path, np.array(getattr(self, "feature_names", []), dtype=object))
            else:
                n, l, c = self.data.shape
                flat = self.data.reshape(-1, c)  # [N*L, C]

                if Tag == "train_normal":
                    # 仅在 train 上拟合 & 保存
                    self.scaler.fit(flat)
                    np.save(scaler_mean_path, self.scaler.mean_)
                    np.save(scaler_std_path, self.scaler.scale_)
                    flat = self.scaler.transform(flat).astype(np.float32)
                    self.data = flat.reshape(n, l, c)
                    if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                        print("[ACM] Fitted scaler on train_normal and applied to data.")
                else:
                    if use_dataset_scale and os.path.exists(scaler_mean_path) and os.path.exists(scaler_std_path):
                        mean = np.load(scaler_mean_path)
                        std = np.load(scaler_std_path)
                        self.scaler.mean_ = mean
                        self.scaler.scale_ = std
                        self.scaler.var_ = std ** 2
                        flat = self.scaler.transform(flat).astype(np.float32)
                        self.data = flat.reshape(n, l, c)
                        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                            print("[ACM] Use dataset-level scaling from train_normal.")
                    else:
                        # fallback：当前 split 自拟合，避免没有 train_normal 缓存时报错
                        self.scaler.fit(flat)
                        flat = self.scaler.transform(flat).astype(np.float32)
                        self.data = flat.reshape(n, l, c)
                        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                            print(f"[ACM] Fallback to self-normalization on Tag={self.Tag}.")

                # 保存缓存（已归一化）
                np.save(data_path, self.data)
                np.save(feat_path, np.array(self.feature_names, dtype=object))
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print(f"Data saved to {data_path}")
                    print(f"Feature names saved to {feat_path}")

    # ----------------- 内部数据构建 -----------------
    def _flight_data(self, tail_list, para_list):
        all_seqs = []
        feature_names_once = None

        for tail_num in tqdm(tail_list, desc=f"[{self.side}] Fetching flight data..."):
            # ALT 时间轴
            start_ts_ns = pd.Timestamp(self.time_start + " 00:00:00", tz="Asia/Shanghai").value
            end_ts_ns = pd.Timestamp(self.time_end + " 23:59:59", tz="Asia/Shanghai").value
            alt_query = f"""
                SELECT time, value
                FROM ALT_STD
                WHERE "aircraft/tail" = '{tail_num}'
                AND TIME >= {start_ts_ns} AND TIME < {end_ts_ns}
                ORDER BY TIME
            """
            try:
                alt_df = self.session.execute_query_statement(alt_query).todf()
            except Exception as e:
                print(f"[{tail_num}] ALT_STD query error:", e)
                continue
            if len(alt_df) == 0:
                continue
            alt_df.columns = ["time", "ALT_STD"]
            alt_df["time"] = pd.to_datetime(alt_df["time"], utc=True).dt.tz_convert("Asia/Shanghai")

            # 地面段（起飞前+落地后）
            seg_all, seg_pre, seg_post, _ = simple_ground_segments_with_time(
                alt_df["time"].values.view("i8"),
                alt_df["ALT_STD"].values,
                alt_threshold=3000.0,
                max_climb_rate_fpm=300.0,
                min_segment_sec=120,
            )
            keep_segments = seg_pre + seg_post
            if len(keep_segments) == 0:
                continue

            # 查询其余参数并插值到 ALT 时间轴
            raw = {"time": alt_df["time"].values, "ALT_STD": alt_df["ALT_STD"].values}
            grid_time_ns = alt_df["time"].values.view("i8")
            for param in [p for p in para_list if p != "ALT_STD"]:
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
                        raw[param] = np.full(len(alt_df), np.nan, dtype=float)
                        continue
                    dfp.columns = ["time", param]
                    dfp["time"] = pd.to_datetime(dfp["time"], utc=True).dt.tz_convert("Asia/Shanghai")
                    raw[param] = interpolate_to_grid(dfp, grid_time_ns, param)
                except Exception as e:
                    print(f"[{tail_num}] param {param} error:", e)
                    raw[param] = np.full(len(alt_df), np.nan, dtype=float)

            full_df = pd.DataFrame(raw)

            # 派生量（保持你的定义方向）
            if self.side == "L":
                full_df["TDIFFTURB2L"] = full_df["TINTURB2L"] - full_df["TOUTPACKL"]
                full_df["TDIFFCPRSRL"] = full_df["TOUTCPRSRL"] - full_df["TOUTPHXL"]
                derived = ["TDIFFTURB2L", "TDIFFCPRSRL"]
            else:
                full_df["TDIFFTURB2R"] = full_df["TINTURB2R"] - full_df["TOUTPACKR"]
                full_df["TDIFFCPRSRR"] = full_df["TOUTCPRSRR"] - full_df["TOUTPHXR"]
                derived = ["TDIFFTURB2R", "TDIFFCPRSRR"]

            feat_cols = [c for c in para_list if c in full_df.columns] + derived
            if feature_names_once is None:
                feature_names_once = feat_cols
                self.feature_names = feature_names_once

            # 在每个地面段内切 seq_len（不跨段）
            seq_len = self.args.seq_len
            for (s, e) in keep_segments:
                if e - s < seq_len:
                    continue
                i = s
                while i + seq_len <= e:
                    window = full_df.iloc[i : i + seq_len]
                    all_seqs.append(window[feat_cols].to_numpy(dtype=float))
                    i += seq_len

        if len(all_seqs) == 0:
            # 空数据保护：feature_names 至少要初始化
            if not hasattr(self, "feature_names"):
                derived = (
                    ["TDIFFTURB2R", "TDIFFCPRSRR"] if self.side == "R" else ["TDIFFTURB2L", "TDIFFCPRSRL"]
                )
                self.feature_names = [c for c in self.para] + [c for c in derived if c not in self.para]
            return np.zeros((0, self.args.seq_len, len(self.feature_names)), dtype=float)
        return np.stack(all_seqs, axis=0)

    # ----------------- 访问接口 -----------------
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    # -----------------（可选）窗口级（反）归一化工具 -----------------
    def transform_windows(self, windows_np: np.ndarray) -> np.ndarray:
        """
        对 [num_win, L, C] 做列标准化（按 self.scaler 的 mean_/scale_）
        """
        if not hasattr(self.scaler, "mean_") or not hasattr(self.scaler, "scale_"):
            return windows_np
        mean = self.scaler.mean_.reshape(1, 1, -1)
        std = self.scaler.scale_.reshape(1, 1, -1)
        std = np.where(std == 0, 1.0, std)
        return (windows_np - mean) / std

    def inverse_transform_windows(self, windows_np: np.ndarray) -> np.ndarray:
        """
        把标准化后的 [num_win, L, C] 反归一化回原量纲
        """
        if not hasattr(self.scaler, "mean_") or not hasattr(self.scaler, "scale_"):
            return windows_np
        mean = self.scaler.mean_.reshape(1, 1, -1)
        std = self.scaler.scale_.reshape(1, 1, -1)
        return windows_np * std + mean


# ----------------- 简易 Args（可替换为你工程里的 Args） -----------------
class Args:
    def __init__(self):
        self.bins = 10
        self.seq_len = 96
        self.ddp = False
        self.local_rank = 0
        # 是否强制使用 train_normal 的统计量做 dataset-level scaling
        self.dataset_scale = True

        

if __name__ == "__main__":
    args = Args()
    dataset = FlightDataset_acm(args, Tag='train_normal')

    print("样本数:", len(dataset))
    if len(dataset) > 0:
        print("单样本 shape (seq_len, feat_dim):", dataset[0].shape)

    # 打印特征名与派生量校验
    feat_names = getattr(dataset, 'feature_names', [])
    print("特征列表({}):".format(len(feat_names)))
    print(feat_names)

    if len(dataset) > 0 and len(feat_names) > 0:
        sample0 = dataset[0]
        name_to_idx = {n:i for i,n in enumerate(feat_names)}
        def _peek(name, k=6):
            return sample0[:k, name_to_idx[name]] if name in name_to_idx else None

        # 左侧派生量核对
        tpL = _peek('TOUTPACKL'); tiL = _peek('TINTURB2L'); tdL = _peek('TDIFFTURB2L')
        if tpL is not None and tiL is not None and tdL is not None:
            print("\n校验 TDIFFTURB2L = TOUTPACKL - TINTURB2L（前6点）")
            print("TOUTPACKL   :", tpL)
            print("TINTURB2L   :", tiL)
            print("TDIFFTURB2L :", tdL)
            print("差值(再算一遍):", tiL-tpL)

        thL = _peek('TOUTPHXL'); tcL = _peek('TOUTCPRSRL'); tdCL = _peek('TDIFFCPRSRL')
        if thL is not None and tcL is not None and tdCL is not None:
            print("\n校验 TDIFFCPRSRL = TOUTPHXL - TOUTCPRSRL（前6点）")
            print("TOUTPHXL     :", thL)
            print("TOUTCPRSRL   :", tcL)
            print("TDIFFCPRSRL  :", tdCL)
            print("差值(再算一遍):", tcL-thL)

        # 右侧派生量核对
        tpR = _peek('TOUTPACKR'); tiR = _peek('TINTURB2R'); tdR = _peek('TDIFFTURB2R')
        if tpR is not None and tiR is not None and tdR is not None:
            print("\n校验 TDIFFTURB2R = TOUTPACKR - TINTURB2R（前6点）")
            print("TOUTPACKR   :", tpR)
            print("TINTURB2R   :", tiR)
            print("TDIFFTURB2R :", tdR)
            print("差值(再算一遍):", tiR-tpR)

        thR = _peek('TOUTPHXR'); tcR = _peek('TOUTCPRSRR'); tdCR = _peek('TDIFFCPRSRR')
        if thR is not None and tcR is not None and tdCR is not None:
            print("\n校验 TDIFFCPRSRR = TOUTPHXR - TOUTCPRSRR（前3点）")
            print("TOUTPHXR     :", thR)
            print("TOUTCPRSRR   :", tcR)
            print("TDIFFCPRSRR  :", tdCR)
            print("差值(再算一遍):", tcR-thR)
