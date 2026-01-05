"""
实现基于飞机的数据加载和推理
"""
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from iotdb.table_session import TableSession, TableSessionConfig
from data_provider.data_loader_prsov import FlightDataset_prsov
from tqdm import tqdm
import os
def data_pipline_prsov(df_data):
    '''
    找到prsov调节压力的最高点的索引
    '''
    df_data['close_prsov'] = 0
    if 'E60031500R' in df_data.columns and 'E60041500R' in df_data.columns:
        diff = (df_data['E60031500R'].diff() - df_data['E60041500R'].diff()).diff()
        df_data['close_prsov'] = diff
        # df_data['close_prsov'] = df_data['close_prsov'].shift(-1)

        """条件1：保留在相应范围内，满足趋势要求的特征值"""
        # 计算下降趋势条件：
        # 获取当前值及后三个值
        next_41 = df_data['E60041500R'].shift(-1)
        next_42 = df_data['E60041500R'].shift(-2)
        next_43 = df_data['E60041500R'].shift(-3)

        # 定义下降趋势的条件（当前值大于后一个值，后一个值大于再后一个值）
        # 也可添加斜率条件作为替代方案
        condition_trend = (
                (df_data['E60041500R'] > next_41) &  # 当前值 > 后一个值
                (next_41 > next_42) &  # 后一个值 > 后两个值
                (next_42 > next_43)  # 后两个值 > 后三个值
            # 替代方案：使用斜率判断下降趋势（需处理边界值）
            # (df_data['E60041500L'] - next_3) / 3 > 0.5
        )

        # 计算下降趋势条件：
        # 获取当前值及后三个值
        next_31 = df_data['E60031500R'].shift(-1)
        next_32 = df_data['E60031500R'].shift(-2)
        next_33 = df_data['E60031500R'].shift(-3)

        # 定义下降趋势的条件（当前值大于后一个值，后一个值大于再后一个值）
        # 也可添加斜率条件作为替代方案
        condition_trend2 = (
                (df_data['E60031500R'] <= next_31) &  # 当前值 > 后一个值
                (next_31 <= next_32) &  # 后一个值 > 后两个值
                (next_32 <= next_33)  # 后两个值 > 后三个值
            # 替代方案：使用斜率判断下降趋势（需处理边界值）
            # (df_data['E60041500L'] - next_3) / 3 > 0.5
        )
        # 填充NaN值为False（当没有足够的后值时，认为不满足下降趋势）
        condition_trend = condition_trend.fillna(False)
        condition_trend2 = condition_trend2.fillna(False)
        # 复合条件
        condition = (
            # df_data['close_prsov'].between(0, 2) &
                (df_data['E60041500R'] >= 55) &
                (df_data['E60031500R'] >= 50) &
                condition_trend &
                condition_trend2
        )
        # df_data['close_prsov2'] = 0
        # df_data.loc[condition, 'close_prsov2'] = 1
        # 对满足条件的行，close_prsov值加2
        df_data.loc[condition, 'close_prsov'] = 10

        """条件2：过滤发动机-中间级压力-总管压力不满足要求的特征值"""
        # df_data['close_prsov']中低于2的值全部设为0，引气管道压力和引气压力变化率太低
        df_data.loc[df_data['close_prsov'] <= 2.5, 'close_prsov'] = 0
        # df_data['E60031500L']中低于45的位置在df_data['close_prsov']中设为0
        # 对n21列小于80的值全部设为0
        df_data.loc[df_data['n22'] <= 80, 'close_prsov'] = 0
        # 发动机转速波动不明显的时候，prsov不调节
        n21_diff = df_data['n22'] - df_data['n22'].rolling(window=10).min()
        df_data.loc[(n21_diff <= 1 & ~condition), 'close_prsov'] = 0
        # 传输管道压力值过低 df_data['close_prsov']低于10，prsov不调节
        df_data.loc[(df_data['E60031500R'] < 50) & (df_data['close_prsov'] < 8), 'close_prsov'] = 0
        # 引气压力值低于50且 df_data['close_prsov']低于10，prsov不调节
        df_data.loc[(df_data['E60041500R'] <= 55) & (df_data['close_prsov'] < 8), 'close_prsov'] = 0
        # E60091515L中为1的位置在df_data['close_prsov']中设为0
        # 高压活门打开的时候，prsov不调节
        df_data.loc[df_data['E60091515R'] == 1, 'close_prsov'] = 0

        """条件3：过滤位置相近的特征值，保留最大值的索引"""
        # close_prsov列中大于零的索引位置在df_data['E60041500L']前后10个索引位置最大值的位置在df_data['close_prsov']中设为20
        # 获取close_prsov列中所有大于0的值的索引
        pos_mask = df_data['close_prsov'] > 0
        pos_idx = df_data.index[pos_mask]

        # 计算相邻大于0位置的间隔距离
        distances = pos_idx.to_series().diff().fillna(0)

        # 根据距离创建分组（间隔<10的归为同一组）
        group_ids = (distances > 20).cumsum().rename('group_ids')

        # 直接创建分组映射
        group_mapping = pd.Series(index=pos_idx, data=group_ids.values)


        def get_max_positions():
            for group_id in group_ids.unique():
                # 获取原始分组内的索引
                group_original = group_mapping[group_mapping == group_id].index

                if not group_original.empty:
                    # 确定分组边界范围（前扩10后扩10）
                    start_idx = max(group_original.min() - 5, df_data.index.min())
                    end_idx = min(group_original.max() + 5, df_data.index.max())

                    # 创建扩展后的索引范围
                    group_indices = df_data.index[df_data.index.slice_indexer(start_idx, end_idx)]

                    # 在扩展范围内找最大值位置
                    if not group_indices.empty:
                        max_pos = df_data.loc[group_indices, 'E60041500R'].idxmax()
                        yield max_pos


        max_positions = list(get_max_positions())

        # 创建掩码：标记需要设为20的位置
        mask_20 = df_data.index.isin(max_positions)

        # 创建掩码：标记需要归零的位置 (其他大于0的位置)
        mask_zero = pos_mask & ~mask_20

        # 应用修改：将最大值位置设为20，其他大于0位置归零
        df_data['close_prsov'] = np.where(
            mask_20,
            1,
            np.where(
                mask_zero,
                0,
                df_data['close_prsov']
            )
        )
        # 返回为1的索引位置
    else:
        '''
        找到prsov调节压力的最高点的索引
        '''
        df_data['close_prsov'] = 0
        diff = (df_data['E60031500L'].diff() - df_data['E60041500L'].diff()).diff()
        df_data['close_prsov'] = diff
        # df_data['close_prsov'] = df_data['close_prsov'].shift(-1)

        """条件1：保留在相应范围内，满足趋势要求的特征值"""
        # 计算下降趋势条件：
        # 获取当前值及后三个值
        next_41 = df_data['E60041500L'].shift(-1)
        next_42 = df_data['E60041500L'].shift(-2)
        next_43 = df_data['E60041500L'].shift(-3)

        # 定义下降趋势的条件（当前值大于后一个值，后一个值大于再后一个值）
        # 也可添加斜率条件作为替代方案
        condition_trend = (
                (df_data['E60041500L'] > next_41) &  # 当前值 > 后一个值
                (next_41 > next_42) &  # 后一个值 > 后两个值
                (next_42 > next_43)  # 后两个值 > 后三个值
            # 替代方案：使用斜率判断下降趋势（需处理边界值）
            # (df_data['E60041500L'] - next_3) / 3 > 0.5
        )

        # 计算下降趋势条件：
        # 获取当前值及后三个值
        next_31 = df_data['E60031500L'].shift(-1)
        next_32 = df_data['E60031500L'].shift(-2)
        next_33 = df_data['E60031500L'].shift(-3)

        # 定义下降趋势的条件（当前值大于后一个值，后一个值大于再后一个值）
        # 也可添加斜率条件作为替代方案
        condition_trend2 = (
                (df_data['E60031500L'] <= next_31) &  # 当前值 > 后一个值
                (next_31 <= next_32) &  # 后一个值 > 后两个值
                (next_32 <= next_33)  # 后两个值 > 后三个值
            # 替代方案：使用斜率判断下降趋势（需处理边界值）
            # (df_data['E60041500L'] - next_3) / 3 > 0.5
        )
        # 填充NaN值为False（当没有足够的后值时，认为不满足下降趋势）
        condition_trend = condition_trend.fillna(False)
        condition_trend2 = condition_trend2.fillna(False)
        # 复合条件
        condition = (
            # df_data['close_prsov'].between(0, 2) &
                (df_data['E60041500L'] >= 55) &
                (df_data['E60031500L'] >= 50) &
                condition_trend &
                condition_trend2
        )
        # df_data['close_prsov2'] = 0
        # df_data.loc[condition, 'close_prsov2'] = 1
        # 对满足条件的行，close_prsov值加2
        df_data.loc[condition, 'close_prsov'] = 10

        """条件2：过滤发动机-中间级压力-总管压力不满足要求的特征值"""
        # df_data['close_prsov']中低于2的值全部设为0，引气管道压力和引气压力变化率太低
        df_data.loc[df_data['close_prsov'] <= 2.5, 'close_prsov'] = 0
        # df_data['E60031500L']中低于45的位置在df_data['close_prsov']中设为0
        # 对n21列小于80的值全部设为0
        df_data.loc[df_data['n21'] <= 80, 'close_prsov'] = 0
        # 发动机转速波动不明显的时候，prsov不调节
        n21_diff = df_data['n21'] - df_data['n21'].rolling(window=10).min()
        df_data.loc[(n21_diff <= 1 & ~condition), 'close_prsov'] = 0
        # 传输管道压力值过低 df_data['close_prsov']低于10，prsov不调节
        df_data.loc[(df_data['E60031500L'] < 50) & (df_data['close_prsov'] < 8), 'close_prsov'] = 0
        # 引气压力值低于50且 df_data['close_prsov']低于10，prsov不调节
        df_data.loc[(df_data['E60041500L'] <= 55) & (df_data['close_prsov'] < 8), 'close_prsov'] = 0
        # E60091515L中为1的位置在df_data['close_prsov']中设为0
        # 高压活门打开的时候，prsov不调节
        df_data.loc[df_data['E60091515L'] == 1, 'close_prsov'] = 0

        """条件3：过滤位置相近的特征值，保留最大值的索引"""
        # close_prsov列中大于零的索引位置在df_data['E60041500L']前后10个索引位置最大值的位置在df_data['close_prsov']中设为20
        # 获取close_prsov列中所有大于0的值的索引
        pos_mask = df_data['close_prsov'] > 0
        pos_idx = df_data.index[pos_mask]

        # 计算相邻大于0位置的间隔距离
        distances = pos_idx.to_series().diff().fillna(0)

        # 根据距离创建分组（间隔<10的归为同一组）
        group_ids = (distances > 20).cumsum().rename('group_ids')

        # 直接创建分组映射
        group_mapping = pd.Series(index=pos_idx, data=group_ids.values)

        def get_max_positions():
            for group_id in group_ids.unique():
                # 获取原始分组内的索引
                group_original = group_mapping[group_mapping == group_id].index

                if not group_original.empty:
                    # 确定分组边界范围（前扩10后扩10）
                    start_idx = max(group_original.min() - 5, df_data.index.min())
                    end_idx = min(group_original.max() + 5, df_data.index.max())

                    # 创建扩展后的索引范围
                    group_indices = df_data.index[df_data.index.slice_indexer(start_idx, end_idx)]

                    # 在扩展范围内找最大值位置
                    if not group_indices.empty:
                        max_pos = df_data.loc[group_indices, 'E60041500L'].idxmax()
                        yield max_pos

        max_positions = list(get_max_positions())

        # 创建掩码：标记需要设为20的位置
        mask_20 = df_data.index.isin(max_positions)

        # 创建掩码：标记需要归零的位置 (其他大于0的位置)
        mask_zero = pos_mask & ~mask_20

        # 应用修改：将最大值位置设为20，其他大于0位置归零
        df_data['close_prsov'] = np.where(
            mask_20,
            1,
            np.where(
                mask_zero,
                0,
                df_data['close_prsov']
            )
        )
    return df_data.index[df_data['close_prsov'] == 1].to_numpy()


class FlightDataset_prsov_singleac(Dataset):
    def __init__(self, session, tail_list, para_list_l, para_list_r, time_start, time_end, args, tail_num):
        self.session = session
        self.tail_list = tail_list
        self.para_list_l = para_list_l
        self.para_list_r = para_list_r
        self.time_start = time_start
        self.time_end = time_end
        self.scaler = StandardScaler()
        self.args = args
        self.tail_num = tail_num

        """获取故障表，用于赋予飞机故障标签信息"""
        # 获取报文故障表，全部飞机系统报警信息，用于获取绝对正常的数据。
        # self.session.execute_non_query_statement("USE ata36abnormal")
        self.fault_table_cfd = self.session.execute_query_statement("SELECT * FROM ata36abnormal").todf()
        # 获取真实故障表，经过确认的真实故障信息，用于确认真实的故障信息，此信息用于测试集
        self.fault_table_turefault_l = self.session.execute_query_statement("SELECT * FROM prsovl").todf()
        self.fault_table_turefault_r = self.session.execute_query_statement("SELECT * FROM prsovr").todf()
        # self.session.execute_non_query_statement("USE b777")

        # 仅保留'aircraft/tail' = tail_num的行
        self.fault_table_cfd = self.fault_table_cfd[self.fault_table_cfd['aircraft/tail'] == self.tail_num]
        self.fault_table_turefault_l = self.fault_table_turefault_l[self.fault_table_turefault_l['aircraft/tail'] == self.tail_num]
        self.fault_table_turefault_r = self.fault_table_turefault_r[self.fault_table_turefault_r['aircraft/tail'] == self.tail_num]

        self.dataset_name = '%d_planes' % len(self.tail_list) + '_' + '_'.join(self.para_list_l) + '_'.join(self.para_list_r) + '_' + str(self.time_start) + '_' + str(self.time_end)
        # 使用train数据做归一化
        scaler_Tag = 'train_normal'
        if self.args.dataset_scale and os.path.exists(f"cache/{self.dataset_name}_{scaler_Tag}_mean.npy"):
            # * 使用训练集指标作为归一化依据（这里使用左右系统全量训练数据）
            train_mean = np.load(f"cache/{self.dataset_name}_{scaler_Tag}_mean.npy")
            train_std = np.load(f"cache/{self.dataset_name}_{scaler_Tag}_std.npy")
            print(f"Using dataset level scaling: {scaler_Tag}")
        else:
            print("loading train-normal data")
            tail_list = [
            "B-2008", "B-2007", "B-2029", "B-2009", "B-20DM",
            "B-226D", "B-226C", "B-20C5", "B-2080", "B-2081",
            "B-2042", "B-223N", "B-2041", "B-209Y", "B-2048",
            "B-2026", "B-2028", "B-2027", "B-2049", "B-20CK",
            "B-20EM", "B-222W", "B-20EN", "B-223G", "B-7185",
            "B-2071", "B-7183", "B-2073", "B-2072", "B-2075",
            "B-2099", "B-2010", "B-20AC", "B-7588"
            ]
            # 简化飞机列表
            # 参数列表
            para_list_l = [  # 第一个刨面
                "n21",  # query8 (发动机转速1)
                "E60031500L",  # query9 (左中间级压力)
                "E60041500L",  # query10 (左总管压力)
                "E60091515L",  # query13 (高压级活门位置-右)
            ]

            para_list_r = [  # 第一个刨面
                "n22",  # query8 (发动机转速2)
                "E60031500R",  # query9 (右中间级压力)
                "E60041500R",  # query10 (右总管压力)
                "E60091515R",  # query13 (高压级活门位置-右)
            ]

            # 时间范围
            time_start = '2023-01-01'
            time_end = '2025-04-30'

            data_set = FlightDataset_prsov(
                self.session,
                tail_list,
                para_list_l,
                para_list_r,
                time_start,
                time_end,
                args,
                Tag='train_normal'
            )
            train_mean = np.load(f"cache/{self.dataset_name}_{scaler_Tag}_mean.npy")
            train_std = np.load(f"cache/{self.dataset_name}_{scaler_Tag}_std.npy")
            print(f"Using dataset level scaling: {scaler_Tag}")

        """获取单个飞机的数据并进行序列处理"""
        single_flightdata = pd.DataFrame()
        for param_index, paramNAME in enumerate(self.para_list_r):
            query = f"""
            SELECT value
            FROM {paramNAME}
            WHERE "aircraft/tail" = '{self.tail_num}'
            AND TIME >= {self.time_start} AND TIME < {self.time_end}
            """
            # 执行查询并转换时间
            records = self.session.execute_query_statement(query).todf()
            # if paramNAME == 'IVVR':
            #     # 降采样为原来的1/4
            #     records = records.iloc[::4].reset_index(drop=True)
            records = records.rename(columns={'value': paramNAME})
            single_flightdata = pd.concat([single_flightdata, records], axis=1)

        query_time = f"""
        SELECT time
        FROM {self.para_list_l[0]}
        WHERE "aircraft/tail" = '{self.tail_num}'
        AND TIME >= {self.time_start} AND TIME < {self.time_end}
        """
        # 执行查询并转换时间
        self.records_time = self.session.execute_query_statement(query_time).todf()

        # 添加过滤条件
        def dataset_pipline(df):
            index = data_pipline_prsov(df)
            data_set_ = df.to_numpy()
            # 构建输入特征向量
            data = np.hstack((
                data_set_[:, 0].reshape(-1, 1),  # 索引0 → 转为列向量
                data_set_[:, 1].reshape(-1, 1),  # 索引3-5 → 转为列向量
                data_set_[:, 2].reshape(-1, 1),
            )).astype(np.float32)
            data = (data - train_mean) / train_std
            return data,index
        self.data, self.index = dataset_pipline(single_flightdata)

    def __getitem__(self, idx):
        """获取单个航班的数据并处理为样本块"""
        # 获取prsov调节
        chang_index = self.index[idx]
        start = chang_index - 33
        end = chang_index + 30

        data = self.data[start:end]

        def sliding_window(data, window_size, step=1):
            """
            处理Pandas DataFrame的滑动窗口
            """
            if len(data) < window_size:
                return pd.DataFrame()

            # 创建窗口索引
            indices = [
                (i, i + window_size)
                for i in range(0, len(data) - window_size + 1, step)
            ]

            # 使用索引获取窗口
            windows = [data[start:end] for start, end in indices]
            return np.array(windows)

        # 对data在第一个维度上做一个seq_len窗口的滑窗
        data = sliding_window(data, window_size=self.args.seq_len, step=1)
        # 对data进行标准化处理
        data_tensors = torch.tensor(data).float()
        return data_tensors,  torch.tensor(self.records_time.iloc[chang_index])

    def __len__(self):
        """返回总航班数量"""
        return len(self.index)

if __name__ == '__main__':
    # 飞机尾号列表
    tail_list = [
        "B-2008", "B-2007", "B-2029", "B-2009", "B-20DM",
        "B-226D", "B-226C", "B-20C5", "B-2080", "B-2081",
        "B-2042", "B-223N", "B-2041", "B-209Y", "B-2048",
        "B-2026", "B-2028", "B-2027", "B-2049", "B-20CK",
        "B-20EM", "B-222W", "B-20EN", "B-223G", "B-7185",
        "B-2071", "B-7183", "B-2073", "B-2072", "B-2075",
        "B-2099", "B-2010", "B-20AC", "B-7588"
    ]

    # 参数列表
    para_list_l = [  # 第一个刨面
        "FLIGHT_PHASE",  # query1
        "ALT_STD",  # query2
        "IAS",  # query3 (校准空速)
        "IVVR",  # query4 (记录垂直速度)
        "VRTG2",  # query5 (垂直加速度)
        "CABPRS",  # query6 (客舱压力数值)
        "PNEUMODE",  # query7 (引气控制模式)
        "n21",  # query8 (发动机转速1)
        "E60031500L",  # query9 (左中间级压力)
        "E60041500L",  # query10 (左总管压力)
        "E60071500L",  # query11 (左预冷器出口温度)
        "E60051500L",  # query12 (左管路流量)
        "E60091515L",  # query13 (高压级活门位置-左)
        "BLD_VLV1",  # query14 (PRSOV活门位置-左)
        "E60101515L",  # query15 (左FAMV活门位置)
        "E60121515_L",  # query16 (气源隔离活门位置-左)
        "E60111515L",  # query17 (引气隔离活门位置-左)
    ]

    para_list_r = [  # 第一个刨面
        "FLIGHT_PHASE",  # query1
        "ALT_STD",  # query2
        "IAS",  # query3 (校准空速)
        "IVVR",  # query4 (记录垂直速度)
        "VRTG2",  # query5 (垂直加速度)
        "CABPRS",  # query6 (客舱压力数值)
        "PNEUMODE",  # query7 (引气控制模式)
        "n22",  # query8 (发动机转速2)
        "E60031500R",  # query9 (右中间级压力)
        "E60041500R",  # query10 (右总管压力)
        "E60071500R",  # query11 (右预冷器出口温度)
        "E60051500R",  # query12 (右管路流量)
        "E60091515R",  # query13 (高压级活门位置-右)
        "BLD_VLV2",  # query14 (PRSOV活门位置-右)
        "E60101515R",  # query15 (右FAMV活门位置)
        "E60121515_R",  # query16 (气源隔离活门位置-右)
        "E60111515R",  # query17 (引气隔离活门位置-右)
    ]
    # 简化飞机列表
    # 参数列表
    para_list_l = [  # 第一个刨面
        "n21",  # query8 (发动机转速1)
        "E60031500L",  # query9 (左中间级压力)
        "E60041500L",  # query10 (左总管压力)
        "E60091515L",  # query13 (高压级活门位置-右)
    ]

    para_list_r = [  # 第一个刨面
        "n22",  # query8 (发动机转速2)
        "E60031500R",  # query9 (右中间级压力)
        "E60041500R",  # query10 (右总管压力)
        "E60091515R",  # query13 (高压级活门位置-右)
    ]

    # 时间范围
    time_start = '2023-01-01'
    time_end = '2025-04-30'

    shuffle_flag = False
    batch_size = 512

    config = TableSessionConfig(
        node_urls=["127.0.0.1:6667"],
        username="root",
        password="root",
        time_zone="UTC+8",
        # enable_compression=True
    )
    session = TableSession(config)
    session.execute_non_query_statement("USE b777")

    data_set = FlightDataset_prsov_singleac(
        session,
        tail_list,
        para_list_l,
        para_list_r,
        time_start,
        time_end,
        args,
        tail_num=tail_list[0]
    )
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag)
    for i, (data, time_index) in enumerate(data_loader):
        print(data.shape)
        print(time_index.shape)
        break
