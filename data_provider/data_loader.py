import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
import os
import pandas as pd


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

class FlightDataset(Dataset):
    def __init__(self, session, tail_list, para_list, time_start, time_end, args, Tag):
        self.session = session
        self.tail_list = tail_list
        self.para_list = para_list
        self.time_start = time_start
        self.time_end = time_end
        self.args = args
        self.bins = args.bins
        self.scaler = StandardScaler()
        self.Tag = Tag
        self.dataset_name = '%d_planes' % len(self.tail_list) + '_' + '_'.join(self.para_list) + '_' + str(self.time_start) + '_' + str(self.time_end)

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
        
        """获取参数表"""
        data_path = f"cache/{self.dataset_name}_{Tag}_data.npy"
        label_path = f"cache/{self.dataset_name}_{Tag}_label.npy"
        index_path = f"cache/{self.dataset_name}_{Tag}_index.npy"

        cache = True
        if os.path.exists(data_path) and os.path.exists(label_path) and os.path.exists(index_path) and cache:
            print(f"Loading cached data from {data_path} and {label_path}")
            self.data = np.load(data_path)
            self.label = np.load(label_path)
            self.index = np.load(index_path)
        else:
            if Tag == 'train_l':
                self.data,self.label, self.index = self._flight_data(self.flight_metadata_train_normal, self.para_list)
            elif Tag == 'train_r':
                self.data,self.label, self.index = self._flight_data(self.flight_metadata_train_normal, self.para_list)
            elif Tag == 'train_a':
                self.data, self.label, self.index = self._flight_data(self.flight_metadata_train_normal, self.para_list)
            elif Tag == 'val_l':
                self.data,self.label, self.index = self._flight_data(self.flight_metadata_val_normal, self.para_list)
            elif Tag == 'val_r':
                self.data,self.label, self.index = self._flight_data(self.flight_metadata_val_normal, self.para_list)
            elif Tag == 'val_a':
                self.data, self.label, self.index = self._flight_data(self.flight_metadata_val_normal, self.para_list)

            elif Tag == 'test_normal_l':
                self.data,self.label, self.index =  self._flight_data(self.flight_metadata_test_normal, self.para_list)
            elif Tag == 'test_normal_r':
                self.data,self.label, self.index = self._flight_data(self.flight_metadata_test_normal, self.para_list)
            elif Tag == 'test_abnormal_l':
                self.data,self.label, self.index = self._flight_data(self.flight_metadata_abnormal_l, self.para_list)
            elif Tag == 'test_abnormal_r':
                self.data,self.label, self.index = self._flight_data(self.flight_metadata_abnormal_r, self.para_list)

            os.makedirs('cache', exist_ok=True)
            np.save(data_path, self.data)
            np.save(label_path, self.label)
            np.save(index_path, self.index)
            print(f"Data saved to cache/{self.dataset_name}_{Tag}_data.npy, label saved to cache/{self.dataset_name}_{Tag}_label.npy, index saved to cache/{self.dataset_name}_{Tag}_index.npy")



    def _flight_data(self, flight_metadata_list, para_list):

        loader_simple = True
        if loader_simple:
            print('loader_simple')
            return self._flight_data_simple(flight_metadata_list, para_list)
        """获取单个航班的数据并进行序列处理"""
        all_flightdata = []

        # 避免加载全量数据
        if 'train' in self.Tag:
            flight_metadata_list = flight_metadata_list[0:15]

        # 遍历航班元数据self.flight_metadata_train_normal
        for flight_group_index, flight_info in enumerate(tqdm(flight_metadata_list)):
            # 重置flight_info的行索引
            flight_info.reset_index(inplace=True, drop=True)
            for flightdata_index, row in flight_info.iterrows():
                tail_num = row['aircraft/tail']
                flight_start_time = row['flight_start_time']
                flight_end_time = row['flight_end_time']

                single_flightdata = pd.DataFrame()
                for param_index, paramNAME in enumerate(para_list):
                    query = f"""
                    SELECT time, value
                    FROM {paramNAME}
                    WHERE "aircraft/tail" = '{tail_num}'
                    AND TIME >= {flight_start_time} AND TIME < {flight_end_time}
                    """

                    # 执行查询并转换时间
                    records = self.session.execute_query_statement(query).todf()
                    records['time'] = pd.to_datetime(records['time'], unit='ns')

                    if param_index == 0:
                        records = records.rename(columns={'value': paramNAME, 'time': 'time_1'})
                        single_flightdata = records[['time_1', paramNAME]].copy()
                    else:
                        records = records.rename(columns={'value': paramNAME, 'time': f'{paramNAME}_timediff'})
                        single_flightdata = pd.merge_asof(
                            single_flightdata,
                            records,
                            left_on='time_1',
                            right_on=f'{paramNAME}_timediff',
                            direction='nearest'
                        )
                        # 移除时间差列以节省内存
                        single_flightdata.drop(columns=[f'{paramNAME}_timediff'], inplace=True)
                single_flightdata['flight_group_index'] = flight_group_index
                single_flightdata['flightdata_index'] = flightdata_index
                all_flightdata.append(single_flightdata)
        all_flightdata = pd.concat(all_flightdata, ignore_index=True, axis=0)
        # 添加过滤条件
        def dataset_pipline(df):

            scaler = StandardScaler()
            data_set_ = df.to_numpy()

            # 排除飞行阶段为6的数据
            mask = np.isin(data_set_[:, 1], [8, 9, 10])
            data_set_ = data_set_[mask]

            # 获取标签和索引
            index = data_set_[:,18:20].astype(int)
            label = data_set_[:,13].astype(int)

            # 构建输入特征向量
            data = np.hstack((
                data_set_[:, 8].reshape(-1, 1),  # 索引0 → 转为列向量
                data_set_[:, 9].reshape(-1, 1),  # 索引3-5 → 转为列向量
                data_set_[:, 10].reshape(-1, 1),
                data_set_[:, 12].reshape(-1, 1),
            )).astype(np.float32)

            scaler.fit(data)
            data = scaler.transform(data)

            # from verlize import main
                # main(df)
            return data, label, index

        return dataset_pipline(all_flightdata)

    def _flight_data_simple(self, flight_metadata_list, para_list):
        """获取单个航班的数据并进行序列处理"""
        all_flightdata = []
        flight_info_index = 0
        # 遍历航班元数据self.flight_metadata_train_normal
        # 避免加载全量数据

        if 'train' in self.Tag:
            flight_metadata_list = flight_metadata_list[0:-1]
            for flight_info in tqdm(flight_metadata_list):
                # 重置flight_info的行索引
                # flight_info.reset_index(inplace=True, drop=True)
                # 获取flight_info flight_start_time列的最小值和flight_end_time的最大值
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
                    if paramNAME == 'IVVR':
                        # 降采样为原来的1/4
                        records = records.iloc[::4].reset_index(drop=True)
                    records = records.rename(columns={'value': paramNAME})
                    single_flightdata = pd.concat([single_flightdata, records], axis=1)
                single_flightdata['flight_group_index'] = flight_info_index
                flight_info_index += 1
                single_flightdata['flightdata_index'] = 0
                all_flightdata.append(single_flightdata)
            all_flightdata = pd.concat(all_flightdata, ignore_index=True, axis=0)

            # 添加过滤条件
            def dataset_pipline(df):

                scaler = StandardScaler()
                data_set_ = df.to_numpy()

                # 排除飞行阶段为6的数据
                mask = np.isin(data_set_[:, 0], [8, 9, 10])
                data_set_ = data_set_[mask]

                # 获取标签和索引
                index = data_set_[:, 6:8].astype(int)
                label = data_set_[:, 5].astype(int)

                # label中不等于1的数据变为0
                label = np.where(label == 1, 1, 0)

                # 构建输入特征向量
                data = np.hstack((
                    data_set_[:, 0].reshape(-1, 1),  # 索引0 → 转为列向量
                    data_set_[:, 1].reshape(-1, 1),  # 索引3-5 → 转为列向量
                    data_set_[:, 2].reshape(-1, 1),
                    data_set_[:, 3].reshape(-1, 1),
                    data_set_[:, 4].reshape(-1, 1),
                )).astype(np.float32)

                scaler.fit(data)
                data = scaler.transform(data)

                # from verlize import main
                # main(df)
                return data, label, index
            return dataset_pipline(all_flightdata)

        if 'test_normal' in self.Tag:
            flight_metadata_list = flight_metadata_list[0:-1]

        for flight_group_index, flight_info in enumerate(flight_metadata_list):
            # 重置flight_info的行索引
            flight_info.reset_index(inplace=True, drop=True)
            for flightdata_index, row in flight_info.iterrows():
                tail_num = row['aircraft/tail']
                flight_start_time = row['flight_start_time']
                flight_end_time = row['flight_end_time']
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
                    if paramNAME == 'IVVR':
                        # 降采样为原来的1/4
                        records = records.iloc[::4].reset_index(drop=True)
                    records = records.rename(columns={'value': paramNAME})
                    single_flightdata = pd.concat([single_flightdata, records], axis=1)
                single_flightdata['flight_group_index'] = flight_group_index
                single_flightdata['flightdata_index'] = flightdata_index
                all_flightdata.append(single_flightdata)
        all_flightdata = pd.concat(all_flightdata, ignore_index=True, axis=0)
        # 添加过滤条件
        def dataset_pipline(df):

            scaler = StandardScaler()
            data_set_ = df.to_numpy()

            # 排除飞行阶段为6的数据
            mask = np.isin(data_set_[:, 0], [8, 9, 10])
            data_set_ = data_set_[mask]

            # 获取标签和索引
            index = data_set_[:,6:8].astype(int)
            label = data_set_[:,5].astype(int)

            # 构建输入特征向量
            data = np.hstack((
                data_set_[:, 0].reshape(-1, 1),  # 索引0 → 转为列向量
                data_set_[:, 1].reshape(-1, 1),  # 索引0 → 转为列向量
                data_set_[:, 2].reshape(-1, 1),  # 索引3-5 → 转为列向量
                data_set_[:, 3].reshape(-1, 1),
                data_set_[:, 4].reshape(-1, 1),
            )).astype(np.float32)

            scaler.fit(data)
            data = scaler.transform(data)

            # from verlize import main
                # main(df)
            return data, label, index

        return dataset_pipline(all_flightdata)


    def _flight_metadata_abnormal(self,fault_table_turefault):
        """收集所有飞机左系统故障的航班信息"""
        flight_metadata = pd.DataFrame()
        for tail_num in self.tail_list:
            try:
                flighttime_start_end = get_flight_times(tail_num, self.time_start, self.time_end, self.session)
                flighttime_start_end['aircraft/tail'] = tail_num
                flighttime_start_end['select_abnormal_r'] = 0
                # 获取此飞机左系统发生故障时候的日期
                fault_data_r = fault_table_turefault[fault_table_turefault['aircraft/tail'] == tail_num]
                # 获取经过人工确认的真实发生的故障数据
                fault_data_r = fault_data_r[fault_data_r['truefault'] == 1 ]

                for idx, row in fault_data_r.iterrows():

                    # 时间窗口为30天，认为故障发生前30的时间是故障征兆的数据
                    duration_forward = 30 * 24 * 3600 * 1000_000_000
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
        # 使用5天作为分组边界，将相同飞机的不同故障数据分成不同的组
        time_threshold = 5 * 24 * 3600 * 1000_000_000
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
                    # 时间窗口为一周，认为故障发生后一周之后的时间是正常的数据（指故障修复时间）
                    duration_reword = 7 * 24 * 3600 * 1000_000_000

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
        return train_list, test_list,test_list[int(-len(train_list)/2):]

    def __getitem__(self, idx):
        """获取单个航班的数据并处理为样本块"""
        start = idx
        end = start + self.args.seq_len

        data = self.data[start:end]
        label = self.label[start:end]

        # 训练阶段不存在index
        if self.index is None:
            index = None
        else:
            index = self.index[start:end]
        data_tensors = torch.tensor(data).float()
        label_tensors = torch.tensor(label).long()
        return data_tensors, label_tensors, index

    def __len__(self):
        """返回总航班数量"""
        return len(self.data) - self.args.seq_len
