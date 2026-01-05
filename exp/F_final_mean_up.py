import h5py
import numpy as np
import pandas as pd
import os
import yaml

class H5ParameterReader:
    """
    读取HDF5文件中记录参数的工具类
    """
    
    def __init__(self, file_path):
        """
        初始化参数读取器
        
        Args:
            file_path: HDF5文件路径
        """
        self.file_path = file_path
        self.parameters = {}  # 缓存已加载的参数
    
    def _load_parameter(self, param_name):
        """加载特定参数的数据"""
        if param_name in self.parameters:
            return  # 如果已经加载过，直接返回
        
        try:
            with h5py.File(self.file_path, 'r') as h5:
                if 'Recorded Parameters' in h5 and param_name in h5['Recorded Parameters']:
                    para_data = h5['Recorded Parameters'][param_name]
                    para_array = np.asarray(para_data)

                    if para_array.ndim == 1:
                        # 对于一维数组，直接使用整个数组作为值
                        self.parameters[param_name] = para_array
                    else:
                        # 对于二维数组，取第二列为值
                        self.parameters[param_name] = para_array[:, 1]
                else:
                    raise KeyError(f"参数 '{param_name}' 在文件中不存在")
        except Exception as e:
            raise KeyError(f"无法加载参数 '{param_name}': {str(e)}")
    
    def __getitem__(self, key):
        """
        通过 context["parameter_name"] 直接返回value数组
        """
        self._load_parameter(key)  # 按需加载参数
        return self.parameters[key]
    
    def get_parameter_names(self):
        """获取文件中所有可用的参数名称（不加载数据）"""
        try:
            with h5py.File(self.file_path, 'r') as h5:
                if 'Recorded Parameters' in h5:
                    return list(h5['Recorded Parameters'].keys())
                else:
                    return []
        except Exception as e:
            print(f"获取参数列表时出错: {e}")
            return []
import numpy as np


# 配置字典
configs = {
    "length": 6,
    "end_point_idx": 100,
    "window_size": 35,
}

def calculate_mean_up(df):
    """
    计算mean_up特征
    逻辑：对mean_diff进行判断，找到连续10个或以上>=0的序列，
    从序列第一个点往前找100个位置，在这个范围内找到mean列的最小值作为分母
    """
    # 使用全局配置参数
    length_threshold = configs["length"]
    end_point_idx = configs["end_point_idx"]
    
    # 标识连续非负diff的序列
    df['non_negative'] = df['mean_diff'] >= 0
    df['sequence_group'] = (df['non_negative'] != df['non_negative'].shift(1)).cumsum()
    
    # 计算每个序列的长度
    sequence_lengths = df.groupby('sequence_group').size()
    
    # 处理每个序列
    for group_id, length in sequence_lengths.items():
        if length >= length_threshold and df[df['sequence_group'] == group_id]['non_negative'].iloc[0]:
            # 找到连续递增序列
            sequence_indices = df[df['sequence_group'] == group_id].index
            
            if len(sequence_indices) > 0:
                # 获取序列的第一个点的位置
                first_point_idx = sequence_indices[0]
                
                # 从第一个点往前找end_point_idx个位置（确保不超出数据范围）
                start_idx = max(0, first_point_idx - end_point_idx)
                search_range_indices = list(range(start_idx, first_point_idx + 1))
                
                # 在这个范围内找到mean列的最小值
                if len(search_range_indices) > 0:
                    # 获取搜索范围内的mean值
                    search_means = df.loc[search_range_indices, 'mean']
                    
                    # 找到最小值及其索引
                    if not search_means.empty:
                        min_mean = search_means.min()
                        min_mean_idx = search_means.idxmin()
                        
                        # 使用这个最小值作为基准
                        for idx in sequence_indices:
                            current_mean = df.loc[idx, 'mean']
                            if min_mean != 0:  # 避免除零错误
                                ratio = current_mean / min_mean
                                df.loc[idx, 'mean_up'] = ratio
                            else:
                                df.loc[idx, 'mean_up'] = 1.0  # 如果最小均值为0，设为1
    
    # 清理临时列
    df.drop(columns=['non_negative', 'sequence_group','mean_diff'], inplace=True)
    
    return df

def find_warning_condition(df_max):
    
    df_max['mean'] = 0.0  # 新增均值列
    
    if 'loss' in df_max.columns:
        
        # 设置滑动窗口大小
        window_size = configs["window_size"]
        
        # 初始化列
        df_max['mean'] = 0.0
        df_max['mean_diff'] = np.nan  # 新增的均值差分列，初始化为NaN
        df_max['mean_up'] = 0.0  # 新增的mean_up列，初始化为0

        # 计算初始窗口的均值
        if len(df_max) >= window_size:
            df_max.loc[window_size-1:, 'mean'] = np.nan  # 后续将通过滑动窗口更新
        df_max.loc[:min(window_size-2, len(df_max)-1), 'mean'] = np.mean(df_max['loss'].iloc[:window_size-1])

        # 滑动窗口处理
        for i in range(len(df_max)):
            if i >= window_size - 1:
                # 计算当前窗口的均值
                window_data = df_max.iloc[i - window_size + 1:i + 1]
                mean_value = np.mean(window_data['loss'])
                df_max.loc[df_max.index[i], 'mean'] = mean_value
        
        
        # 计算均值的一阶差分（导数）
        df_max['mean_diff'] = df_max['mean'].diff()
        
        # 计算mean_up特征
        df_max = calculate_mean_up(df_max)
        
    return df_max
def flight_loss(context):
    

    try:
        from run import main
        # main输出loss,特征值
        loss = main(context)
        return loss
    except Exception as e:
        return None
    
def flight_warning(context_, single_flight_loss=False):
    ###########  写入本地HDF5文件地址 #########################
    
    # 返回单个航班的loss值
    if single_flight_loss:
        if os.path.exists(f"loss_{context_['ac']}.csv"):
            old_loss = pd.read_csv(f"loss_{context_['ac']}.csv")
            return float(old_loss['loss'].iloc[-1])

    # 获取单个航班的loss值
    loss = flight_loss(context_)

    # 跨航班运算逻辑
    # 获取所有航班的loss值
    if os.path.exists(f"loss_{context_['ac']}.csv"):
        old_loss = pd.read_csv(f"loss_{context_['ac']}.csv")
    else:
        old_loss = pd.DataFrame()

    if loss is not None:
        # TODO：预留返回单个航班LOSS值的接口，在这个地方直接返回单个航班的loss值
        if single_flight_loss:  # 如果是单个航班loss值，则直接返回
            return float(loss['loss'].max())
        
        # 如果loss大于1行，则仅保留loss最大值所在的行
        if len(loss) > 1:
            loss = loss.loc[[loss['loss'].idxmax()]]

        # 拼接loss和旧loss
        new_loss = pd.concat([old_loss, loss], ignore_index=True)

        # 更新最新的new_loss到csv文件中
        new_loss.tail(300).to_csv(f"loss_{context_['ac']}.csv", index=False)
        # 执行报警逻辑计算
        if len(new_loss) > 40:  # 确保至少有40数据点
            final_loss = find_warning_condition(new_loss)
            return final_loss['mean_up'].iloc[-1]
            
        else:
            return None

    else:
        return None
    
if __name__ == "__main__":
    context = H5ParameterReader(r"export-4213291-B-226C-2025-10-30-ZYHB-MMSM.hdf5")
    from collections import defaultdict
    context_ = defaultdict(dict)
    context_["N21"]["Value"] = context["N21"]["Value"].tolist()
    context_["E60031500L"]["Value"] = context["E60031500L"]["Value"].tolist()
    context_["E60041500L"]["Value"] = context["E60041500L"]["Value"].tolist()
    context_["ac"]="B-226C"
    context_["time"] = int(pd.to_datetime("2025-10-30").timestamp())
    mean_up = flight_warning(context_,single_flight_loss=True)
    print(mean_up)
    if mean_up > 0.5:  # 假设0.5为阈值
        print("天瞳事件已发生！")




        

