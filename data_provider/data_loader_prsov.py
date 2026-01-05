import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
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
        # 返回为1的索引位置

    return df_data.index[df_data['close_prsov'] == 1].to_numpy()

def get_flight_times(tail_num,date1, date2,session=None):
    if session is not None:
        tail_num = tail_num
        time_start = date1
        time_end = date2

        # 构造查询语句
        query1 = f"""
        SELECT time as flight_start_time, 
        endtime as flight_end_time, 
        "flight/flightnumber" as flightnumber,
        "flight/duration/airborneduration" as flight_duration,
        "flight/departureairport/iata" as dep_airport,
        "flight/arrivalairport/iata" as arr_airport,
        "flight/datetime/startrecordingdatetime" as startrecordingdatetime
        FROM flight_metadata  
        where "aircraft/tail" = '{tail_num}'
        AND TIME >= {time_start} AND TIME < {time_end}
        """

        query0 = f"""
        SELECT *
            FROM flight_metadata
            where "aircraft/tail" = '{tail_num}'
            AND TIME >= {time_start} AND TIME < {time_end}
        """

        records0 = session.execute_query_statement(query1).todf()
        return records0
    
    
class FlightDataset_prsov(Dataset):
    def __init__(self, session, tail_list, para_list_l, para_list_r, time_start, time_end, args, Tag):
        self.session = session
        self.tail_list = tail_list
        self.para_list_l = para_list_l
        self.para_list_r = para_list_r
        self.time_start = time_start
        self.time_end = time_end
        self.args = args
        self.bins = args.bins
        self.scaler = StandardScaler()
        self.Tag = Tag
        self.dataset_name = '%d_planes' % len(self.tail_list) + '_' + '_'.join(self.para_list_l) + '_'.join(self.para_list_r) + '_' + str(self.time_start) + '_' + str(self.time_end)

        """获取参数表"""
        data_path = f"cache/{self.dataset_name}_{Tag}_data.npy"
        index_path = f"cache/{self.dataset_name}_{Tag}_index.npy"
        flight_index = f"cache/{self.dataset_name}_{Tag}_flight_index.npy"

        cache = True
        if cache and os.path.exists(data_path) and os.path.exists(index_path) and os.path.exists(flight_index):
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print(f"Loading cached data from {data_path}")
            self.data = np.load(data_path)
            self.index = np.load(index_path)
            self.flight_index = np.load(flight_index)

        else:
            """获取故障表，用于筛选航班元信息"""
            # 获取报文故障表，全部飞机系统报警信息，用于获取绝对正常的数据。
            self.fault_table_cfd = self.session.execute_query_statement("SELECT * FROM ata36abnormal").todf()
            # 获取真实故障表，经过确认的真实故障信息，用于确认真实的故障信息，此信息用于测试集
            self.fault_table_turefault_l = self.session.execute_query_statement("SELECT * FROM prsovl").todf()
            self.fault_table_turefault_r = self.session.execute_query_statement("SELECT * FROM prsovr").todf()

            """获取航班元信息，用于获取参数表中的数据"""
            # 获取绝对正常的航班元数据，并划分训练数据和测试数据，训练正常数据用于训练正常基线，测试正常数据用于测试模型误报率。
            self.flight_metadata_train_normal, self.flight_metadata_test_normal, self.flight_metadata_val_normal = self._flight_metadata_normal()
            # 获取故障征兆的航班元数据
            self.flight_metadata_abnormal_l = self._flight_metadata_abnormal(self.fault_table_turefault_l)
            self.flight_metadata_abnormal_r = self._flight_metadata_abnormal(self.fault_table_turefault_r)
            if Tag == 'train_normal_l':
                self.data, self.index, self.flight_index = self._flight_data(self.flight_metadata_train_normal, self.para_list_l)
            elif Tag == 'train_normal_r':
                self.data, self.index, self.flight_index = self._flight_data(self.flight_metadata_train_normal, self.para_list_r)

            elif Tag == 'train_normal':
                data1, index1, flight_index1 = self._flight_data(self.flight_metadata_train_normal, self.para_list_l)
                data2, index2, flight_index2 = self._flight_data(self.flight_metadata_train_normal, self.para_list_r)
                self.data = np.concatenate((data1, data2))
                self.index = np.concatenate((index1, index2+len(data1)))
                self.flight_index = np.concatenate((flight_index1, flight_index2+len(set(flight_index1))))

            elif Tag == 'val_normal_l':
                self.data, self.index, self.flight_index = self._flight_data(self.flight_metadata_val_normal, self.para_list_l)
            elif Tag == 'val_normal_r':
                self.data, self.index, self.flight_index = self._flight_data(self.flight_metadata_val_normal, self.para_list_r)
            elif Tag == 'val_normal':
                data1, index1, flight_index1 = self._flight_data(self.flight_metadata_val_normal, self.para_list_l)
                data2, index2, flight_index2 = self._flight_data(self.flight_metadata_val_normal, self.para_list_r)
                self.data = np.concatenate((data1, data2))
                self.index = np.concatenate((index1, index2+len(data1)))
                self.flight_index = np.concatenate((flight_index1, flight_index2+len(set(flight_index1))))

            elif Tag == 'test_normal_l':
                self.data, self.index, self.flight_index =  self._flight_data(self.flight_metadata_test_normal, self.para_list_l)
            elif Tag == 'test_normal_r':
                self.data, self.index, self.flight_index = self._flight_data(self.flight_metadata_test_normal, self.para_list_r)
            elif Tag == 'test_normal':
                data1, index1, flight_index1 = self._flight_data(self.flight_metadata_test_normal, self.para_list_l)
                data2, index2, flight_index2 = self._flight_data(self.flight_metadata_test_normal, self.para_list_r)
                self.data = np.concatenate((data1, data2))
                self.index = np.concatenate((index1, index2+len(data1)))
                self.flight_index = np.concatenate((flight_index1, flight_index2+len(set(flight_index1))))

            elif Tag == 'test_abnormal_l':
                self.data, self.index, self.flight_index = self._flight_data(self.flight_metadata_abnormal_l, self.para_list_l)
            elif Tag == 'test_abnormal_r':
                self.data, self.index, self.flight_index = self._flight_data(self.flight_metadata_abnormal_r, self.para_list_r)
            elif Tag == 'test_abnormal':
                data1, index1, flight_index1 = self._flight_data(self.flight_metadata_abnormal_l, self.para_list_l)
                data2, index2, flight_index2 = self._flight_data(self.flight_metadata_abnormal_r, self.para_list_r)
                self.data = np.concatenate((data1, data2))
                self.index = np.concatenate((index1, index2+len(data1)))
                self.flight_index = np.concatenate((flight_index1, flight_index2+len(set(flight_index1))))
            
            os.makedirs('cache', exist_ok=True)
            scaler_Tag = 'train_normal'
            if Tag == 'train_normal':
                self.scaler.fit(self.data)
                train_mean = self.scaler.mean_         # 形状 [n_features,]
                train_std = self.scaler.scale_         # 注意：scale_ 是标准差（非方差）
                np.save(f"cache/{self.dataset_name}_{scaler_Tag}_mean.npy", train_mean)
                np.save(f"cache/{self.dataset_name}_{scaler_Tag}_std.npy", train_std)

            # 使用train数据做归一化
            if self.args.dataset_scale and os.path.exists(f"cache/{self.dataset_name}_{scaler_Tag}_mean.npy"):
                train_mean = np.load(f"cache/{self.dataset_name}_{scaler_Tag}_mean.npy")
                train_std = np.load(f"cache/{self.dataset_name}_{scaler_Tag}_std.npy")
                print(f"Using dataset level scaling: {scaler_Tag}")
                # 执行归一化操作
                self.data = (self.data - train_mean) / train_std
            else:
                # 此处有一个风险是，当'train_normal'不存在的时候，会使用self-normalization，避免了重复加载'train_normal'
                print(f"Using self-normalization")
                self.scaler.fit(self.data)
                self.data = self.scaler.transform(self.data)

            
            np.save(data_path, self.data)
            np.save(index_path, self.index)
            np.save(flight_index, self.flight_index)
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print(f"Data saved to cache/{self.dataset_name}_{Tag}_data.npy, "
                      f"index saved to cache/{self.dataset_name}_{Tag}_index.npy, "
                      f"flight_index saved to cache/{self.dataset_name}_{Tag}_index.npy")

    def _flight_data(self, flight_metadata_list, para_list):
        """获取单个航班的数据并进行序列处理"""
        all_flightdata = []
        flight_info_index = 0
        for flight_info in tqdm(flight_metadata_list, desc='Fetching flight metadata...'):
            flight_start_time = flight_info['flight_start_time'].min()
            flight_end_time = flight_info['flight_end_time'].max()
            tail_num = flight_info['aircraft/tail'].iloc[0]
            single_flightdata = pd.DataFrame()
            for param_index, paramNAME in enumerate(para_list):
                query = f"""
                SELECT value
                FROM {paramNAME}
                WHERE "aircraft/tail" = '{tail_num}'
                AND TIME >= {flight_start_time} AND TIME < {flight_end_time}
                """
                # 执行查询并转换时间
                records = self.session.execute_query_statement(query).todf()
                # if paramNAME == 'IVVR':
                #     # 降采样为原来的1/4
                #     records = records.iloc[::4].reset_index(drop=True)
                records = records.rename(columns={'value': paramNAME})
                single_flightdata = pd.concat([single_flightdata, records], axis=1)
            single_flightdata['flight_group_index'] = flight_info_index
            flight_info_index += 1
            all_flightdata.append(single_flightdata)
        all_flightdata = pd.concat(all_flightdata, ignore_index=True, axis=0)

        # 添加过滤条件
        def dataset_pipline(df):
            index = data_pipline_prsov(df)
            data_set_ = df.to_numpy()

            # 构建输入特征向量
            data = np.hstack((
                data_set_[:, 0].reshape(-1, 1),  # 索引0 → 转为列向量
                data_set_[:, 1].reshape(-1, 1),  # 索引3-5 → 转为列向量
                data_set_[:, 2].reshape(-1, 1),
                data_set_[:, 3].reshape(-1, 1),
                data_set_[:, 4].reshape(-1, 1),
            )).astype(np.float32)
            flight_info_index = df['flight_group_index'].to_numpy()
            return data, index, flight_info_index
        return dataset_pipline(all_flightdata)


    def _flight_metadata_abnormal(self,fault_table_turefault):
        """收集所有飞机左系统故障的航班信息"""
        flight_metadata = pd.DataFrame()
        for tail_num in self.tail_list:
            try:
                flighttime_start_end = get_flight_times(tail_num, self.time_start, self.time_end, self.session)
                flighttime_start_end['aircraft/tail'] = tail_num
                flighttime_start_end['select_abnormal_r'] = 0
                fault_data_r = fault_table_turefault[fault_table_turefault['aircraft/tail'] == tail_num]
                # 获取经过人工确认的真实发生的故障数据
                fault_data_r = fault_data_r[fault_data_r['truefault'] == 1 ]

                for idx, row in fault_data_r.iterrows():
                    # 时间窗口为180天，认为故障发生前180的时间是故障征兆的数据
                    duration_forward = self.args.before_failure_days * 24 * 3600 * 1000_000_000
                    # 过滤第一个已经发生的故障报文，即获取故障航班的上一个航班，
                    # row['time']获取的是故障发生的航班起飞时间，

                    start_ns = row['time'] - duration_forward
                    end_ns = row['time']
                    time_mask = (flighttime_start_end['flight_start_time'] >= start_ns) & (flighttime_start_end['flight_start_time'] <= end_ns)
                    # 将ture的位置记录下来，做最后的更新
                    flighttime_start_end.loc[time_mask, 'select_abnormal_r'] += 1

                # 将时间段中发生故障的数据剔除
                flight_metadata_abnormal = flighttime_start_end[flighttime_start_end['select_abnormal_r'] != 0]
                if not flight_metadata_abnormal.empty:
                    mask_fault_indications  = ~flight_metadata_abnormal['flight_start_time'].isin(fault_data_r['time'])
                    flight_metadata_abnormal = flight_metadata_abnormal[mask_fault_indications]
                    flight_metadata = pd.concat([flight_metadata, flight_metadata_abnormal])
            except Exception as e:
                print(f"获取航班元数据时出错 (尾号: {tail_num}): {e}")

        # 计算时间差并确定分组边界
        time_diff = flight_metadata.groupby('aircraft/tail')['flight_start_time'].diff()
        # 使用60天作为分组边界，将相同飞机的不同故障数据分成不同的组
        time_threshold = 60 * 24 * 3600 * 1000_000_000
        new_cluster = ((time_diff > time_threshold) | time_diff.isna())
        flight_metadata['cluster_id'] = new_cluster.cumsum()
        # 按分组ID进行聚类
        clustered_dfs = []
        for (tail, cluster_id), group in flight_metadata.groupby(['aircraft/tail', 'cluster_id']):
            clustered_dfs.append(group)
        return clustered_dfs

    def _flight_metadata_normal(self):
        """收集所有飞机的航班时间段信息"""
        flight_metadata = pd.DataFrame()

        for tail_num in self.tail_list:
            try:
                flight_metadata_all = get_flight_times(tail_num, self.time_start, self.time_end, self.session)
                flight_metadata_all['aircraft/tail'] = tail_num
                flight_metadata_all['select_normal'] = 0 # 0表示正常，1表示故障,获取0所代表的航班

                # 获取此飞机发生故障时候的日期
                fault_data = self.fault_table_cfd[self.fault_table_cfd['aircraft/tail'] == tail_num]

                for idx, row in fault_data.iterrows():

                    # 时间窗口为三个月，认为故障发生前三个月之前的时间是正常的数据（指故障征兆时间）
                    duration_forward = 90 * 24 * 3600 * 1000_000_000
                    # 时间窗口为10天，认为故障发生后10天之后的时间是正常的数据（指故障修复时间）
                    duration_reword = 10 * 24 * 3600 * 1000_000_000
                    
                    # 如果涉及prsov的故障，则时间窗口调整为360天，完全避免故障征兆数据引入训练集
                    if 'PRSOV' in row['fde_desc']:
                        duration_forward = 360 * 24 * 3600 * 1000_000_000
                    
                    # 补充当故障不明确的时候，认为是虚假故障
                    if 'nan' in row['fde_desc']:
                        duration_forward = 0
                        duration_reword = 0

                    start_ns = row['time'] - duration_forward
                    end_ns = row['time'] + duration_reword
                    time_mask = (flight_metadata_all['flight_start_time'] >= start_ns) & (flight_metadata_all['flight_start_time'] <= end_ns)
                    # 将ture的位置记录下来，做最后的更新
                    flight_metadata_all.loc[time_mask, 'select_normal'] = 1

                # 将时间段中发生故障的数据剔除
                flight_metadata_normal = flight_metadata_all[flight_metadata_all['select_normal'] == 0]
                if not flight_metadata_normal.empty:
                    flight_metadata = pd.concat([flight_metadata, flight_metadata_normal])
            except Exception as e:
                print(f"获取航班元数据时出错 (尾号: {tail_num}): {e}")

        # 计算时间差并确定分组边界，对航班数据进行分组
        time_diff = flight_metadata.groupby('aircraft/tail')['flight_start_time'].diff()
        # 使用3天作为分组边界，将相同飞机的不同正常数据分成不同的组
        time_threshold = 3 * 24 * 3600 * 1000_000_000
        new_cluster = ((time_diff > time_threshold) | time_diff.isna())
        flight_metadata['cluster_id'] = new_cluster.cumsum()
        # 按分组ID进行聚类
        clustered_dfs = []
        for (tail, cluster_id), group in flight_metadata.groupby(['aircraft/tail', 'cluster_id']):
            clustered_dfs.append(group)

        # 初始化三个空列表
        train_list = []
        test_list = []

        # 遍历所有DataFrame并分类
        for df in clustered_dfs:
            length = len(df)
            if 100 <= length:
                train_list.append(df)
            elif 30 <= length <= 100:  # length > 50
                test_list.append(df)
        return train_list, test_list, test_list

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

        # 对data在第一个维度上做一个24窗口的滑窗
        data = sliding_window(data, window_size=self.args.seq_len, step=1)
        # 对data进行标准化处理

        data_tensors = torch.tensor(data).float()
        return data_tensors, torch.tensor([self.flight_index[chang_index]]*data_tensors.shape[0])

    def __len__(self):
        """返回总航班数量"""
        return len(self.index)
    
    def inverse_transform(self, data):
        """反归一化（对数据集归一化操作的反向操作）"""
        return self.scaler.inverse_transform(data)
