# acm用于给给飞机空调系统提供合适的温度
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
        "B-226D", "B-226C", "B-20C5", "B-2080", "B-2081",
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
]
test_right_abnormal = [
"B-2080"
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
#"TDIFFTURB2R", # 涡轮降温 人工后期做的，"TOUTPACKR"-"TINTURB2R"

"TOUTCPRSRR", # 压气机出口温度
"TOUTPHXR", # 主级热交换器出口温度
#"TDIFFCPRSRR",#压气机升温 人工后期做的，"TOUTPHXR"-"TOUTCPRSRR"

#"POSLVLVR", # 低限活门位置
# "TINPHXR", # 主级热交换器进口温度 主热集交换器进出口经历了什么？似乎只有风扇做工？冲压进气活门的开度应该是不是主要影响主机热交换器进出口的温度差？？
#"POSRAER", # 冲压排气活门位置

]
time_start = '2025-04-01'
time_end   = '2025-04-30'

class FlightDataset_acm(Dataset):
    def __init__(self, args, Tag):
        try:
            config = TableSessionConfig(
                node_urls=["127.0.0.1:6667"],
                username="root",
                password="root",
                time_zone='Asia/Shanghai'
            )
            session = TableSession(config)
        except Exception as e:
            config = TableSessionConfig(
                node_urls=["10.254.43.34:6667"],
                username="root",
                password="root",
                time_zone='Asia/Shanghai'
            )
            session = TableSession(config)
        session.execute_non_query_statement("USE b777")
        self.session = session
        self.train_left = train_left_ac_normal
        self.train_right = train_right_ac_normal
        self.val_left = val_left_ac_normal
        self.val_right = val_right_ac_normal

        self.test_left_normal = test_left_normal
        self.test_right_normal = test_right_normal
        self.test_left_abnormal = test_left_abnormal
        self.test_right_abnormal = test_right_abnormal

        # 参数表
        self.para_l = para_l
        self.para_r = para_r

        # 时间范围
        self.time_start = time_start
        self.time_end = time_end
        self.args = args
        self.bins = args.bins
        self.scaler = StandardScaler()
        self.Tag = Tag
        self.dataset_name = '%d_planes' % len(self.train_left) + '_' + '_'.join(self.para_l) + '_'.join(self.para_r) + '_' + str(self.time_start) + '_' + str(self.time_end)

        """获取参数表"""
        data_path = f"cache/{self.dataset_name}_{Tag}_data_time.npy"

        #if os.path.exists(data_path):
        if False:
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print(f"Loading cached data from {data_path}")
            self.data = np.load(data_path)

        else:
            if Tag == 'train_normal':
                data1 = self._flight_data(self.train_left, self.para_l)
                data2 = self._flight_data(self.train_right, self.para_r)
                self.data = np.concatenate((data1, data2))

            elif Tag == 'val_normal':
                data1 = self._flight_data(self.val_left, self.para_l)
                data2 = self._flight_data(self.val_right, self.para_r)
                self.data = np.concatenate((data1, data2))

            elif Tag == 'test_normal':
                data1 = self._flight_data(self.test_left_normal, self.para_l)
                data2 = self._flight_data(self.test_right_normal, self.para_r)
                self.data = np.concatenate((data1, data2))

            elif Tag == 'test_abnormal':
                data1 = self._flight_data(self.test_left_abnormal, self.para_l)
                data2 = self._flight_data(self.test_right_abnormal, self.para_r)
                self.data = np.concatenate((data1, data2))
            
            os.makedirs('cache', exist_ok=True)
            np.save(data_path, self.data)
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print(f"Data saved to cache/{self.dataset_name}_{Tag}_data.npy")

    def _flight_data(self, tail_list, para_list):
        all_flight = []
        for tail_num in tqdm(tail_list, desc='Fetching flight data...'):
            single_flight = pd.DataFrame()
                        
            # 你的日期字符串
            start_str = self.time_start + ' 00:00:00'
            end_str   = self.time_end   + ' 23:59:59'

            # 直接指定 tz='Asia/Shanghai'
            start_ts_ns = pd.Timestamp(start_str, tz='Asia/Shanghai').value  # 纳秒级整数
            end_ts_ns   = pd.Timestamp(end_str,   tz='Asia/Shanghai').value
            for param in para_list:
                query = f"""
                SELECT time, value
                FROM {param}
                WHERE "aircraft/tail" = '{tail_num}'
                AND TIME >= {start_ts_ns} AND TIME < {end_ts_ns}
                """
                try:
                    records = self.session.execute_query_statement(query).todf()
                    if records is None or len(records) == 0:
                        continue
                except Exception as e:
                    print(e)
                    continue

                tcol = 'time' if 'time' in records.columns else ('Time' if 'Time' in records.columns else None)
                if tcol is None: 
                    continue
                rec = records[[tcol, 'value']].rename(columns={'value': param}).copy()
                rec[tcol] = pd.to_datetime(rec[tcol], errors='coerce')
                rec = rec.dropna(subset=[tcol]).drop_duplicates(subset=[tcol]).sort_values(tcol).reset_index(drop=True)

                # 把每个参数的时间单独一列存起来
                single_param_time = pd.DataFrame({param + "_time": rec[tcol].values})

                if single_flight.empty:
                    single_flight = pd.DataFrame(index=np.arange(len(rec)))
                    single_flight[param] = rec[param].values
                    single_time   = pd.DataFrame(index=np.arange(len(rec)))
                    single_time[param + "_time"] = single_param_time[param + "_time"].values
                else:
                    # 以 rec 的长度为准
                    original_index = np.arange(len(single_flight))
                    new_index = np.linspace(0, len(single_flight) - 1, len(rec))

                    # 1) 先新建一个 DataFrame，有 len(rec) 行
                    resized = pd.DataFrame(index=np.arange(len(rec)))

                    # 2) 对 single_flight 现有的每一列做插值并写入 resized
                    for col in single_flight.columns:
                        resized[col] = np.interp(new_index, original_index, single_flight[col].values)

                    # 3) single_flight 替换成新 DataFrame（已经是 len(rec) 行）
                    single_flight = resized

                    # 4) 同步处理时间列 single_time (做法同上)
                    resized_time = pd.DataFrame(index=np.arange(len(rec)))
                    for col in single_time.columns:
                        t_interp = np.interp(new_index,
                                            original_index,
                                            np.arange(len(single_time)))
                        idx = np.round(t_interp).astype(int)
                        idx = np.clip(idx, 0, len(single_time)-1)
                        resized_time[col] = single_time[col].values[idx]
                    single_time = resized_time

                    # 5) 现在 single_flight 和 single_time 都是 len(rec) 行，可以安全写入新参数
                    single_flight[param] = rec[param].values
                    single_time[param + "_time"] = single_param_time[param + "_time"].values


            if not single_flight.empty:
                # 把数值和时间分块拼接
                combined = pd.concat([single_time, single_flight], axis=1)
                all_flight.append(combined)

        if len(all_flight) == 0:
            return np.empty((0, len(para_list) * 2), dtype=object)

        all_flight = pd.concat(all_flight, ignore_index=True, axis=0)

        # -------- 抽样打印：随机 5 段，每段 10 行，所有变量各自时间戳和值 --------
        if len(all_flight) > 10:
            for k in range(5):
                start = np.random.randint(0, len(all_flight) - 10)
                print(f"\n[原版-多时间列] 段 {k+1}: 行 {start}~{start+10}")
                print(all_flight.iloc[start:start+10])

        # 返回时依然只返回数值部分（不含 *_time 列）保持接口兼容
        numeric_cols = [c for c in all_flight.columns if not c.endswith("_time")]
        return np.array(all_flight[numeric_cols])

        def __getitem__(self, idx):
            start = idx
            end = start + self.args.seq_len
            return self.data[start:end]

        def __len__(self):
            """返回总航班数量"""
            return len(self.data) - self.args.seq_len

# args_stub.py
class Args:
    def __init__(self):
        self.bins = 10         # 你自己需要的 bins
        self.seq_len = 50      # 序列长度
        self.ddp = False       # 如果没用分布式就 False
        self.local_rank = 0    # 单机单卡就 0

if __name__ == "__main__":
    args = Args()

    # 这里 Tag 可以是 'train_normal' / 'val_normal' / 'test_normal' / 'test_abnormal'
    dataset = FlightDataset_acm(args, Tag='train_normal')

    # 随便取一段样本看看
    print("数据集总长度:", len(dataset))
    print("第一段样本 shape:", dataset[0].shape)   # 一个序列的形状


