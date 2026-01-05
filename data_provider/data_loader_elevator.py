#升降舵主要功能是实现飞机俯仰功能，由每侧升降舵安装2个动力控制组件（PCU-powercontrolunit）来作动实现舵面偏转。
#主飞行计算机PFC（primaryflightcomputers）向ACE发送数字指令，ACE将其更改为模拟信号。ACE随后发送模拟舵面位置命令PCU作动升降舵。
#PCU上的位置传感器发送作动筒位置通过ACE向PFC反馈。升降舵PCU接收模拟和离散电气来自作动筒控制电子组件（ACE-actuatorcontrolelectronics）的信号，
#这些离散信号用于旁通、阻塞、和减压电磁阀；模拟信号用于电动液压伺服阀（EHSV）。
#四个升降舵PCU中的每一个都将这些模拟反馈信号发送到ACE来控制升降舵舵面偏转。
# "PITCH", # 俯仰姿态
# "PITCH_CPT", # 机长侧驾驶杆位置 选为输入
# "PITCH_FO", # 副驾驶驾驶杆位置 选为输入
# "ELEVL", # 左升降舵位置
# "LELEVIBPOS", # 左升降舵内侧PCU位置 选为输出
# "LELEVOBPOS", # 左升降舵外侧PCU位置 选为输出

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
"B-2009", "B-20DM", "B-2048",
"B-226D", "B-223N", "B-209Y",
"B-2026", "B-2028", "B-2027", "B-2049",
"B-20EM", "B-222W", "B-223G",
"B-2071", "B-7183",  "B-2075"
]
train_right_ac_normal = [
"B-2009", "B-20DM", "B-2048",
"B-226D", "B-223N", "B-209Y", "B-2049",
"B-2026", "B-2028", "B-2027",
"B-20EM", "B-222W", "B-223G",
"B-2071", "B-7183",  "B-2075"
]

val_left_ac_normal = [
"B-2099", "B-2010", "B-20AC"
]
val_right_ac_normal = [
"B-2099", "B-2010", "B-20AC"
]

test_left_normal = [
"B-2007",
]
test_right_normal = [
"B-2007",
]

test_left_abnormal = [
]
test_right_abnormal = [
"B-2008"
]

# 简化飞机列表
# 参数列表
para_l = [ # 左剖面
"PITCH_FO", # 副驾驶驾驶杆位置 选为输入
"LELEVIBPOS", # 左升降舵内侧PCU位置 输出
"LELEVOBPOS", # 左升降舵外侧PCU位置 输出
]

para_r = [ # 右剖面
"PITCH_FO", # 副驾驶驾驶杆位置
"RELEVIBPOS", # 右升降舵内侧PCU位置
"RELEVOBPOS", # 右升降舵外侧PCU位置
]
# 时间范围
time_start = '2025-04-01'
time_end = '2025-04-30'

class FlightDataset_elevator(Dataset):
    def __init__(self, args, Tag):
        try:
            config = TableSessionConfig(
                node_urls=["127.0.0.1:6667"],
                username="root",
                password="root",
                time_zone="UTC+8"
            )
            session = TableSession(config)
        except Exception as e:
            config = TableSessionConfig(
                node_urls=["10.254.43.34:6667"],
                username="root",
                password="root",
                time_zone="UTC+8"
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
        data_path = f"cache/{self.dataset_name}_{Tag}_data.npy"

        if os.path.exists(data_path):
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
            for param_index, param in enumerate(para_list):
                query = f"""
                SELECT value
                FROM {param}
                WHERE "aircraft/tail" = '{tail_num}'
                AND TIME >= {self.time_start} AND TIME < {self.time_end}
                """
                records = self.session.execute_query_statement(query).todf()
                single_flight = pd.concat([single_flight, records], axis=1)
            all_flight.append(single_flight)
        all_flight = pd.concat(all_flight, ignore_index=True, axis=0)
        return np.array(all_flight)

    def __getitem__(self, idx):
        start = idx
        end = start + self.args.seq_len
        return self.data[start:end]

    def __len__(self):
        """返回总航班数量"""
        return len(self.data) - self.args.seq_len
