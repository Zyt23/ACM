from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import  adjust_learning_rate, EarlyStopping
import torch
import pandas as pd
import numpy as np
import os
import time
import torch.nn as nn
from iotdb.table_session import TableSession, TableSessionConfig
from tqdm import tqdm

class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)

        config = TableSessionConfig(
            node_urls=["127.0.0.1:6667"],
            username="root",
            password="root",
            time_zone="UTC+8"
            # enable_compression=True,
        )
        session = TableSession(config)
        session.execute_non_query_statement("USE b777")
        self.session = session

    def _build_model(self):
        if self.args.ddp:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            # for methods that do not use ddp (e.g. finetuning-based LLM4TS models)
            self.device = self.args.gpu

        model = self.model_dict[self.args.model].Model(self.args)

        if self.args.ddp:
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        elif self.args.dp:
            model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            self.device = self.args.gpu
            model = model.to(self.device)

        if self.args.adaptation:
            model.load_state_dict(torch.load(self.args.pretrain_model_path))
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, self.session)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):

        early_stopping = EarlyStopping(args=self.args, verbose=True)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        model_optim = self._select_optimizer()
        self.model.train()
        data_set, data_loader = self._get_data(flag=f'train_{self.args.train_mode}')
        valdata_set, valdata_loader = self._get_data(flag=f'val_{self.args.train_mode}')
        # 为每个epoch创建新的DataLoader以确保随机性
        for epoch in range(self.args.train_epochs):
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            epoch_time = time.time()
            # 训练循环
            total_batches = len(data_loader)
            for batch_idx, (batch_x, batch_y, index) in enumerate(data_loader):
                model_optim.zero_grad()
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                output = self.model(batch_x)
                loss = nn.CrossEntropyLoss()(output, batch_y)
                loss.backward()
                model_optim.step()
                # 记录损失
                with open(os.path.join(path, 'loss.txt'), 'a') as f:
                    f.write(f"Epoch {epoch + 1}: Loss {loss.item()}\n")
                # 打印进度
                if batch_idx % 1000 == 0 or batch_idx == total_batches - 1:
                    print(f"Epoch: {epoch + 1}/{self.args.train_epochs}, "
                          f"Batch: {batch_idx}/{total_batches}, "
                          f"Loss: {loss.item():.6f}, "
                          f"Time: {time.time() - epoch_time:.2f}s")

                # 定期保存检查点xh
                if batch_idx % total_batches == 0:
                    min_ratio = self.test_combined(setting,test=0)
                    # vali_loss = self.vali(valdata_loader)
                    # print(f"Epoch: {epoch + 1}/{self.args.train_epochs}, {vali_loss}")
            early_stopping(min_ratio, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # 保存最终模型
        torch.save(self.model.state_dict(), os.path.join(path, "final_model.pt"))
        return self.model

    def vali(self, vali_loader):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, index) in enumerate(vali_loader):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                output = self.model(batch_x)

                pred = output.detach().cpu()
                true = batch_y.detach().cpu()

                loss = nn.CrossEntropyLoss()(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test_combined(self, setting, test=1):
        # 加载模型
        if test:
            # 加载模型，多进程模式下，需要去掉"module."前缀
            if not self.args.ddp and not self.args.dp:
                best_model_path = self.args.test_file_name
                model_path = os.path.join(self.args.checkpoints, setting, best_model_path)
                state_dict = torch.load(model_path)
                # 如果保存的模型是在DataParallel模式下，状态字典的键会带有"module."前缀，需要去掉
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k  # 去掉"module."前缀
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
            else:
                best_model_path = self.args.test_file_name
                model_path = os.path.join(self.args.checkpoints, setting, best_model_path)
                state_dict = torch.load(model_path)
                self.model.load_state_dict(state_dict)

        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(os.path.join(self.args.checkpoints, setting), exist_ok=True)

        # 加载正常和异常数据
        def process_data(flag):
            dataset, dataloader = self._get_data(flag=flag)
            all_losses = torch.tensor([], device=self.device)
            all_losses_index = torch.tensor([], device='cpu')  # 索引放在CPU上

            self.model.eval()

            with torch.no_grad():
                for batch_idx, (batch_x, batch_y, index) in enumerate(dataloader):
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    output = self.model(batch_x)
                    loss_ = nn.CrossEntropyLoss(reduction='none')(output, batch_y.long()).flatten()

                    all_losses = torch.cat((all_losses, loss_))
                    all_losses_index = torch.cat((all_losses_index, index.reshape(-1, 2).cpu()))

            # 构建DataFrame并分组处理
            pd_loss_index = pd.DataFrame(all_losses_index.numpy(), columns=['flight_group_id', 'flight_id'])
            pd_loss = pd.DataFrame(all_losses.cpu().numpy(), columns=['loss'])
            combined_df = pd.concat([pd_loss_index, pd_loss], axis=1)

            avg_group_loss = []
            for group_id in np.sort(combined_df['flight_group_id'].unique()):
                group_df = combined_df[combined_df['flight_group_id'] == group_id]

                # 定义一个函数计算最大前10%的分位数（即90%分位数）
                def compute_stat(series, mode='quantile'):
                    if mode == 'quantile-9':
                        return series.quantile(q=0.9)
                    elif mode == 'mean':
                        return series.mean()
                    elif mode == 'quantile-8':
                        return series.mean()
                    else:
                        raise ValueError("Unsupported mode")
                # 使用groupby应用自定义统计函数
                flight_avg = [compute_stat(g['loss'],mode='quantile-9') for _, g in group_df.groupby('flight_id')]
                avg_group_loss.append(flight_avg)

            return avg_group_loss, combined_df

        avg_group_loss_normal,sec_normal_loss = process_data('test_normal_r')
        avg_group_loss_abnormal,sec_abnormal_loss = process_data('test_abnormal_r')

        # 秒级粒度可视化，一个秒一个点
        verlize_flight = False
        if verlize_flight:
            print("Processed verlize data successfully.")
            # 列名称改为 normal_loss
            sec_normal_loss.rename(columns={'loss': 'normal_loss'}, inplace=True)
            df1 = sec_normal_loss['normal_loss']
            sec_abnormal_loss.rename(columns={'loss': 'abnormal_loss'}, inplace=True)
            df2 = sec_abnormal_loss['abnormal_loss']
            df = pd.concat([df1, df2], axis=1)
            # nan 替换为 0
            df.fillna(value=0, inplace=True)
            # # 新建一个长度和df1相同的时间列,时间要是pd 的datatime格式
            df['time_1'] = pd.date_range(start='2000-04-01 00:00:00', periods=len(df), freq='s')
            from verlize import main
            main(df)

        # 航班级可视化，一个航班一个点
        verlize_flight_group = False
        if verlize_flight_group:
            print("Processed verlize data successfully.")
            df1 = pd.DataFrame(avg_group_loss_normal).T
            df1.columns = [f'normal{i}' for i in range(len(avg_group_loss_normal))]
            df2 = pd.DataFrame(avg_group_loss_abnormal).T
            df2.columns = [f'abnormal{i}' for i in range(len(avg_group_loss_abnormal))]
            df = pd.concat([df1, df2], axis=1)
            # nan 替换为 0
            df.fillna(value=0, inplace=True)
            # # 新建一个长度和df1相同的时间列,时间要是pd 的datatime格式
            df['time_1'] = pd.date_range(start='2000-04-01 00:00:00', periods=len(df), freq='s')
            from verlize import main
            main(df)

        # 预计算特征矩阵函数
        def precompute_features(data):
            """向量化计算所有特征矩阵"""
            n_groups = len(data)
            feat_matrix = np.zeros((n_groups, 26))

            for i, group_losses in enumerate(data):
                arr = np.array(group_losses)
                valid_count = len(arr) > 0

                # f1-f16: 基于损失值分布的区间统计
                bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                        0.10, 0.11, 0.12, 0.13, 0.14, 0.15, np.inf]
                if valid_count:
                    hist = np.histogram(arr, bins=bins)[0] / len(arr)
                    feat_matrix[i, :16] = hist
                else:
                    feat_matrix[i, :16] = 0

                # f17-f26: 基于相邻损失值差值的区间统计
                if len(arr) > 1:
                    diffs = np.abs(np.diff(arr))
                    # f17-f25: 差值在不同区间的比例
                    diff_bins = [0,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, np.inf]
                    diff_hist = np.histogram(diffs, bins=diff_bins)[0] / len(diffs)
                    # 存储f17-f25（共9个特征）
                    feat_matrix[i, 16:26] = diff_hist[:10]
                else:
                    feat_matrix[i, 16:26] = 0
            return feat_matrix

        # Store results
        normal_feat_matrix = precompute_features(avg_group_loss_normal)
        abnormal_feat_matrix = precompute_features(avg_group_loss_abnormal)

        # 优化参数
        min_ratio = float('inf')
        best_params = None
        best_avg_HF, best_avg_FF = None, None
        max_iterations = 5000000  # 最大迭代次数

        # 进行随机搜索优化
        for i in range(max_iterations):
            # 随机采样一组权重和阈值T2
            weights = np.random.uniform(0.1, 1.0, 26)
            T2 = np.random.uniform(0.01, 0.8)

            # 使用矩阵运算计算所有组的得分
            # HF：正常数据组的得分
            normal_scores = normal_feat_matrix @ weights
            # FF：异常数据组的得分
            abnormal_scores = abnormal_feat_matrix @ weights

            # 计算HF和FF指标
            HF = np.mean(normal_scores > T2)
            FF = np.mean(abnormal_scores > T2)

            # 只在满足约束条件时更新最优解
            if 0.001 < HF < 0.5 and FF > 0.4:
                ratio = HF / FF
                if ratio < min_ratio:
                    min_ratio = ratio
                    best_params = {'T2': T2, 'weights': weights}
                    best_avg_HF, best_avg_FF = HF, FF

        # 输出最终结果
        if best_params is not None:
            print(f"*****************************test-right****************")
            print(f"avg_HF: {best_avg_HF:.4f}")
            print(f"avg_FF: {best_avg_FF:.4f}")
            print(f"min_Ratio: {min_ratio:.4f}")
            print(f"*****************************test-right****************")
            # 保存结果到文件
            result_str = (f"avg_HF: {best_avg_HF}\n"
                          f"avg_FF: {best_avg_FF}\n"
                          f"Weights: {', '.join([f'{w:.2f}' for w in best_params['weights']])}\n"
                          f"min_Ratio: {min_ratio}\n")
            with open(os.path.join(folder_path, f"combined_result_{setting}.txt"), "a") as f:
                f.write(result_str)
        else:
            print("No valid parameters found")

        self.test_combinged_diff(setting, test=0, T2=best_params['T2'], weights=best_params['weights'])

        self.model.train()
        return min_ratio

    def test_combinged_diff(self, setting, test=1, T2=None, weights=None):
        # 加载模型
        if test:
            # 加载模型，多进程模式下，需要去掉"module."前缀
            if not self.args.ddp:
                best_model_path = self.args.test_file_name
                model_path = os.path.join(self.args.checkpoints, setting, best_model_path)
                state_dict = torch.load(model_path)
                # 如果保存的模型是在DataParallel模式下，状态字典的键会带有"module."前缀，需要去掉
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k  # 去掉"module."前缀
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
            else:
                best_model_path = self.args.test_file_name
                model_path = os.path.join(self.args.checkpoints, setting, best_model_path)
                state_dict = torch.load(model_path)
                self.model.load_state_dict(state_dict)

        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(os.path.join(self.args.checkpoints, setting), exist_ok=True)

        # 加载正常和异常数据
        def process_data(flag):
            dataset, dataloader = self._get_data(flag=flag)
            all_losses = torch.tensor([], device=self.device)
            all_losses_index = torch.tensor([], device='cpu')  # 索引放在CPU上

            self.model.eval()

            with torch.no_grad():
                for batch_idx, (batch_x, batch_y, index) in enumerate(dataloader):
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    output = self.model(batch_x)
                    loss_ = nn.CrossEntropyLoss(reduction='none')(output, batch_y.long()).flatten()

                    all_losses = torch.cat((all_losses, loss_))
                    all_losses_index = torch.cat((all_losses_index, index.reshape(-1, 2).cpu()))

            # 构建DataFrame并分组处理
            pd_loss_index = pd.DataFrame(all_losses_index.numpy(), columns=['flight_group_id', 'flight_id'])
            pd_loss = pd.DataFrame(all_losses.cpu().numpy(), columns=['loss'])
            combined_df = pd.concat([pd_loss_index, pd_loss], axis=1)

            avg_group_loss = []
            for group_id in np.sort(combined_df['flight_group_id'].unique()):
                group_df = combined_df[combined_df['flight_group_id'] == group_id]

                # 定义一个函数计算最大前10%的分位数（即90%分位数）
                def compute_stat(series, mode='quantile'):
                    if mode == 'quantile-9':
                        return series.quantile(q=0.9)
                    elif mode == 'mean':
                        return series.mean()
                    elif mode == 'quantile-8':
                        return series.mean()
                    else:
                        raise ValueError("Unsupported mode")

                # 使用groupby应用自定义统计函数
                flight_avg = [compute_stat(g['loss'], mode='quantile-9') for _, g in group_df.groupby('flight_id')]
                avg_group_loss.append(flight_avg)

            return avg_group_loss, combined_df

        avg_group_loss_normal, sec_normal_loss = process_data('test_normal_l')
        avg_group_loss_abnormal, sec_abnormal_loss = process_data('test_abnormal_l')

        # 秒级粒度可视化，一个秒一个点
        verlize_flight = False
        if verlize_flight:
            print("Processed verlize data successfully.")
            # 列名称改为 normal_loss
            sec_normal_loss.rename(columns={'loss': 'normal_loss'}, inplace=True)
            df1 = sec_normal_loss['normal_loss']
            sec_abnormal_loss.rename(columns={'loss': 'abnormal_loss'}, inplace=True)
            df2 = sec_abnormal_loss['abnormal_loss']
            df = pd.concat([df1, df2], axis=1)
            # nan 替换为 0
            df.fillna(value=0, inplace=True)
            # # 新建一个长度和df1相同的时间列,时间要是pd 的datatime格式
            df['time_1'] = pd.date_range(start='2000-04-01 00:00:00', periods=len(df), freq='s')
            from verlize import main
            main(df)

        # 航班级可视化，一个航班一个点
        verlize_flight_group = False
        if verlize_flight_group:
            print("Processed verlize data successfully.")
            df1 = pd.DataFrame(avg_group_loss_normal).T
            df1.columns = [f'normal{i}' for i in range(len(avg_group_loss_normal))]
            df2 = pd.DataFrame(avg_group_loss_abnormal).T
            df2.columns = [f'abnormal{i}' for i in range(len(avg_group_loss_abnormal))]
            df = pd.concat([df1, df2], axis=1)
            # nan 替换为 0
            df.fillna(value=0, inplace=True)
            # # 新建一个长度和df1相同的时间列,时间要是pd 的datatime格式
            df['time_1'] = pd.date_range(start='2000-04-01 00:00:00', periods=len(df), freq='s')
            from verlize import main
            main(df)

        # 预计算特征矩阵函数
        def precompute_features(data):
            """向量化计算所有特征矩阵"""
            n_groups = len(data)
            feat_matrix = np.zeros((n_groups, 26))

            for i, group_losses in enumerate(data):
                arr = np.array(group_losses)
                valid_count = len(arr) > 0

                # f1-f16: 基于损失值分布的区间统计
                bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                        0.10, 0.11, 0.12, 0.13, 0.14, 0.15, np.inf]
                if valid_count:
                    hist = np.histogram(arr, bins=bins)[0] / len(arr)
                    feat_matrix[i, :16] = hist
                else:
                    feat_matrix[i, :16] = 0

                # f17-f26: 基于相邻损失值差值的区间统计
                if len(arr) > 1:
                    diffs = np.abs(np.diff(arr))
                    # f17-f25: 差值在不同区间的比例
                    diff_bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, np.inf]
                    diff_hist = np.histogram(diffs, bins=diff_bins)[0] / len(diffs)
                    # 存储f17-f25（共9个特征）
                    feat_matrix[i, 16:26] = diff_hist[:10]
                else:
                    feat_matrix[i, 16:26] = 0
            return feat_matrix

        # Store results
        normal_feat_matrix = precompute_features(avg_group_loss_normal)
        abnormal_feat_matrix = precompute_features(avg_group_loss_abnormal)

        # 使用矩阵运算计算所有组的得分
        # HF：正常数据组的得分
        normal_scores = normal_feat_matrix @ weights
        # FF：异常数据组的得分
        abnormal_scores = abnormal_feat_matrix @ weights

        # 计算HF和FF指标
        HF = np.mean(normal_scores > T2)
        FF = np.mean(abnormal_scores > T2)
        ratio = 100
        # 只在满足约束条件时更新最优解
        if 0.001 < HF < 0.5 and FF > 0.4:
            ratio = HF / FF
        # 保存结果到文件
        print(f"*****************************test-left****************")
        print(f"HF: {HF:.4f}")
        print(f"FF: {FF:.4f}")
        print(f"Ratio: {ratio:.4f}")
        print(f"*****************************test-left****************")
        result_str = (f"test_diff_HF: {HF}\n"
                      f"test_diff_FF: {FF}\n"
                      f"test_diff_Ratio: {ratio}\n")
        with open(os.path.join(folder_path, f"combined_result_{setting}.txt"), "a") as f:
            f.write(result_str)

        self.model.train()
        return ratio

    def test_combined_single(self, setting, test=1):
        # 加载模型
        if test:
            print('loading model')
            best_model_path = self.args.test_file_name
            model_path = os.path.join(self.args.checkpoints, setting, best_model_path)
            print(f"loading model from {model_path}")
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()

        # 创建结果目录
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(os.path.join(self.args.checkpoints, setting), exist_ok=True)

        # 加载正常和异常数据
        def process_data(flag):
            dataset, dataloader = self._get_data(flag=flag)
            all_losses = torch.tensor([], device=self.device)
            all_losses_index = torch.tensor([], device='cpu')  # 索引放在CPU上

            with torch.no_grad():
                for batch_idx, (batch_x, batch_y, index) in enumerate(dataloader):
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    output = self.model(batch_x)
                    loss_ = nn.CrossEntropyLoss(reduction='none')(output, batch_y.long()).flatten()

                    all_losses = torch.cat((all_losses, loss_))
                    all_losses_index = torch.cat((all_losses_index, index.reshape(-1, 2).cpu()))

            # 构建DataFrame并分组处理
            pd_loss_index = pd.DataFrame(all_losses_index.numpy(), columns=['flight_group_id', 'flight_id'])
            pd_loss = pd.DataFrame(all_losses.cpu().numpy(), columns=['loss'])
            combined_df = pd.concat([pd_loss_index, pd_loss], axis=1)

            avg_group_loss = []
            for group_id in np.sort(combined_df['flight_group_id'].unique()):
                group_df = combined_df[combined_df['flight_group_id'] == group_id]

                # 定义一个函数计算最大前10%的分位数（即90%分位数）
                def compute_stat(series, mode='quantile'):
                    if mode == 'quantile-9':
                        return series.quantile(q=0.9)
                    elif mode == 'mean':
                        return series.mean()
                    elif mode == 'quantile-8':
                        return series.mean()
                    else:
                        raise ValueError("Unsupported mode")
                # 使用groupby应用自定义统计函数
                flight_avg = [compute_stat(g['loss'],mode='mean') for _, g in group_df.groupby('flight_id')]
                avg_group_loss.append(flight_avg)

            return avg_group_loss, combined_df

        print("Processing normal data...")
        avg_group_loss_normal,sec_normal_loss = process_data('test_normal_r')
        print("Processing abnormal data...")
        avg_group_loss_abnormal,sec_abnormal_loss = process_data('test_abnormal_r')

        # 秒级粒度可视化，一个秒一个点
        verlize_flight = False
        if verlize_flight:
            print("Processed verlize data successfully.")
            # 列名称改为 normal_loss
            sec_normal_loss.rename(columns={'loss': 'normal_loss'}, inplace=True)
            df1 = sec_normal_loss['normal_loss']
            sec_abnormal_loss.rename(columns={'loss': 'abnormal_loss'}, inplace=True)
            df2 = sec_abnormal_loss['abnormal_loss']
            df = pd.concat([df1, df2], axis=1)
            # nan 替换为 0
            df.fillna(value=0, inplace=True)
            # # 新建一个长度和df1相同的时间列,时间要是pd 的datatime格式
            df['time_1'] = pd.date_range(start='2000-04-01 00:00:00', periods=len(df), freq='s')
            from verlize import main
            main(df)

        # 航班级可视化，一个航班一个点
        verlize_flight_group = True
        if verlize_flight_group:
            print("Processed verlize data successfully.")
            df1 = pd.DataFrame(avg_group_loss_normal).T
            df1.columns = [f'normal{i}' for i in range(len(avg_group_loss_normal))]
            df2 = pd.DataFrame(avg_group_loss_abnormal).T
            df2.columns = [f'abnormal{i}' for i in range(len(avg_group_loss_abnormal))]
            df = pd.concat([df1, df2], axis=1)
            # nan 替换为 0
            df.fillna(value=0, inplace=True)
            # # 新建一个长度和df1相同的时间列,时间要是pd 的datatime格式
            df['time_1'] = pd.date_range(start='2000-04-01 00:00:00', periods=len(df), freq='s')
            from verlize import main
            main(df)

        # 定义阈值搜索范围
        T1_range = np.arange(0.01, 0.201, 0.01)
        T2_range = np.arange(0.01, 0.81, 0.05)

        # 存储结果
        results = []
        min_ratio = float('inf')
        best_T1, best_T2 = None, None
        best_avg_HF, best_avg_FF = None, None

        # 搜索最优阈值
        for T2 in T2_range:
            for T1 in T1_range:
                # 计算正常数据误报率 (HF)
                HF_list = []
                for group_losses in avg_group_loss_normal:
                    exceed_rate = np.mean(np.array(group_losses) > T1)
                    HF_list.append(1 if exceed_rate > T2 else 0)
                avg_HF = np.mean(HF_list)

                # 计算异常数据检出率 (FF)
                FF_list = []
                for group_losses in avg_group_loss_abnormal:
                    exceed_rate = np.mean(np.array(group_losses) > T1)
                    FF_list.append(1 if exceed_rate > T2 else 0)
                avg_FF = np.mean(FF_list)

                # 记录所有结果
                results.append({
                    'T1': T1,
                    'T2': T2,
                    'avg_HF': avg_HF,
                    'avg_FF': avg_FF
                })

                # 检查约束条件并更新最优解
                if 0.001 < avg_HF and avg_FF > 0.4:
                    ratio = avg_HF / avg_FF
                    if ratio < min_ratio:
                        min_ratio = ratio
                        best_T1, best_T2 = T1, T2
                        best_avg_HF, best_avg_FF = avg_HF, avg_FF

        # 输出结果
        if best_T1 is not None:
            print(f"\n\n=== 最优解 ===")
            print(f"T1 = {best_T1:.2f}, T2 = {best_T2:.2f}")
            print(f"avg_HF = {best_avg_HF:.4f} (0.001 < ), avg_FF = {best_avg_FF:.4f} (>0.4)")
            print(f"最小比例值: {min_ratio:.4f}")

            # 保存结果到文件
            result_str = (f"Best T1: {best_T1}\nBest T2: {best_T2}\n"
                          f"avg_HF: {best_avg_HF}\navg_FF: {best_avg_FF}\nRatio: {min_ratio}")
            with open(os.path.join(folder_path, f"combined_result_{setting}.txt"), "w") as f:
                f.write(result_str)
        else:
            print("未找到满足条件的阈值组合：avg_HF < 0.4 and avg_FF > 0.4")

        # 返回最优阈值
        return best_T1, best_T2



    def test_HF(self, setting, test=1):
        # 测试误报率
        if test:
            print('loading model')
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, best_model_path)))

        self.model.eval()

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 确保结果路径存在
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        data_set, dataloader = self._get_data(flag='test_normal_r')

        # 新建一个空tensor记录所有的loss
        all_losses = torch.zeros(0).to(self.device)
        all_losses_index = torch.zeros(0)
        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, (batch_x, batch_y, index) in enumerate(dataloader):

                # 将数据传输到设备
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 前向传播
                output = self.model(batch_x)

                # 计算损失（使用与训练相同的损失函数）
                loss = nn.CrossEntropyLoss()(output, batch_y)
                # 原始计算损失
                loss_= nn.CrossEntropyLoss(reduction='none')(output, batch_y.long()).flatten()
                all_losses = torch.cat((all_losses, loss_), dim=0)
                all_losses_index = torch.cat((all_losses_index, index.reshape(-1,2)), dim=0)

            # 获取所有航班组的loss
            pd_loss_index = pd.DataFrame(all_losses_index.cpu().numpy(), columns=['flight_group_id', 'flight_id'])
            pd_loss = pd.DataFrame(all_losses.cpu().numpy(), columns=['loss'])

            # 使用groupby直接创建二维列表
            results = []
            # 平均loss
            avg_group_loss = []
            # 1. 将两个DataFrame合并为一个
            combined_df = pd.concat([pd_loss_index, pd_loss], axis=1)

            # 获取所有唯一的flight_group_id
            group_ids = combined_df['flight_group_id'].unique()
            group_ids.sort()

            for group_id in group_ids:
                group_df = combined_df[combined_df['flight_group_id'] == group_id]

                # 在该group内按flight_id分组
                flight_losses = []
                flight_avg_loss = []
                for flight_id, flight_group in group_df.groupby('flight_id'):
                    flight_losses.append(flight_group['loss'].tolist())
                    flight_avg_loss.append(flight_group['loss'].mean())
                results.append(flight_losses)
                avg_group_loss.append(flight_avg_loss)

            """计算误报率指标"""
            # 对avg_group_loss中大于1的个数进行统计，并除以总长度
            # 定义阈值范围和步长
            T1_range = np.arange(0.04, 0.201, 0.01)  # 0.04到0.2，每次增加0.01
            T2_range = np.arange(0.1, 0.81, 0.05)  # 0.1到0.8，每次增加0.05

            # 存储所有结果的列表
            results = []

            # 外层循环控制0.50（T2）的变化
            for T2 in T2_range:
                # 内层循环控制0.1（T1）的变化
                for T1 in T1_range:
                    HF_list = []

                    for i in range(len(avg_group_loss)):
                        # 计算每组中损失超过T1阈值的比例
                        HF = np.sum(np.array(avg_group_loss[i]) > T1) / len(avg_group_loss[i])

                        # 根据T2阈值确定是否标记为异常
                        HF_num = 1 if HF > T2 else 0
                        HF_list.append(HF_num)

                    # 计算平均误报率
                    avg_HF = np.mean(HF_list)

                    # 存储当前阈值组合的结果
                    results.append({
                        'T1': T1,
                        'T2': T2,
                        'avg_HF': avg_HF,
                        'num_flights': len(avg_group_loss),
                        'HF_list': HF_list.copy()
                    })

                    # 打印当前阈值组合的结果
                    print(f"=== 阈值组合 T1={T1:.2f}, T2={T2:.2f} ===")
                    print(f"平均HF值: {avg_HF:.4f} ({len(avg_group_loss)}个航班)")

            # 可选：找到平均HF值最低的阈值组合
            best_result = min(results, key=lambda x: x['avg_HF'])
            print(f"\n\n最佳阈值组合: T1={best_result['T1']:.2f}, T2={best_result['T2']:.2f}")
            print(f"最低平均HF值: {best_result['avg_HF']:.4f}")


    def test_FF(self, setting, test=1):
        # 测试准确率
        if test:
            print('loading model')
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, best_model_path)))

        self.model.eval()

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 确保结果路径存在
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        data_set, dataloader = self._get_data(flag='test_abnormal_r')

        # 新建一个空tensor记录所有的loss
        all_losses = torch.zeros(0).to(self.device)
        all_losses_index = torch.zeros(0)
        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, (batch_x, batch_y, index) in enumerate(dataloader):

                # 将数据传输到设备
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # 前向传播
                output = self.model(batch_x)

                # 计算损失（使用与训练相同的损失函数）
                loss = nn.CrossEntropyLoss()(output, batch_y)
                # 原始计算损失
                loss_= nn.CrossEntropyLoss(reduction='none')(output, batch_y.long()).flatten()
                # 计算模型损失
                # outputs_ = torch.nn.functional.softmax(output, dim=1)
                # # 交换以下第3维度和第二维度的位置
                # outputs_1 = torch.transpose(outputs_.detach().cpu(), 1, 2).reshape(-1, 2).numpy()
                # # outputs = torch.argmax(outputs_, dim=1)
                #
                # df1 = pd.DataFrame(batch_y.cpu().numpy().flatten(), columns=['batch_y'])
                # df2 = pd.DataFrame(loss_.cpu().numpy().flatten(), columns=['loss'])
                # df3 = pd.DataFrame(outputs_1, columns=['chanal1','chanal2'])
                #
                #
                # df = pd.concat([df1, df2, df3], axis=1)
                # # 新建一个长度和df1相同的时间列,时间要是pd 的datatime格式
                # df['time_1'] = pd.date_range(start='2023-04-01 00:00:00', periods=len(df), freq='T')

                from verlize import main
                # main(df)
                all_losses = torch.cat((all_losses, loss_), dim=0)

                all_losses_index = torch.cat((all_losses_index, index.reshape(-1,2)), dim=0)

            # 获取所有航班组的loss
            pd_loss_index = pd.DataFrame(all_losses_index.cpu().numpy(), columns=['flight_group_id', 'flight_id'])
            pd_loss = pd.DataFrame(all_losses.cpu().numpy(), columns=['loss'])

            # 使用groupby直接创建二维列表
            results = []
            # 平均loss
            avg_group_loss = []
            # 1. 将两个DataFrame合并为一个
            combined_df = pd.concat([pd_loss_index, pd_loss], axis=1)

            # 获取所有唯一的flight_group_id
            group_ids = combined_df['flight_group_id'].unique()
            group_ids.sort()

            for group_id in group_ids:
                group_df = combined_df[combined_df['flight_group_id'] == group_id]

                # 在该group内按flight_id分组
                flight_losses = []
                flight_avg_loss = []
                for flight_id, flight_group in group_df.groupby('flight_id'):
                    flight_losses.append(flight_group['loss'].tolist())
                    flight_avg_loss.append(flight_group['loss'].mean())
                results.append(flight_losses)
                avg_group_loss.append(flight_avg_loss)

            # 对avg_group_loss中大于1的个数进行统计，并除以总长度
            # 定义阈值范围和步长
            T1_range = np.arange(0.04, 0.201, 0.01)  # 0.04到0.2，每次增加0.01
            T2_range = np.arange(0.1, 0.81, 0.05)  # 0.1到0.8，每次增加0.05

            # 存储所有结果的列表
            results = []

            # 外层循环控制0.50（T2）的变化
            for T2 in T2_range:
                # 内层循环控制0.1（T1）的变化
                for T1 in T1_range:
                    FF_list = []

                    for i in range(len(avg_group_loss)):
                        # 计算每组中损失超过T1阈值的比例
                        FF = np.sum(np.array(avg_group_loss[i]) > T1) / len(avg_group_loss[i])

                        # 根据T2阈值确定是否标记为异常
                        FF_num = 1 if FF > T2 else 0
                        FF_list.append(FF_num)

                    # 计算平均误报率
                    avg_FF = np.mean(FF_list)

                    # 存储当前阈值组合的结果
                    results.append({
                        'T1': T1,
                        'T2': T2,
                        'avg_FF': avg_FF,
                    })
                    # 将结果写入文件
                    with open(os.path.join(folder_path, f"FF_{setting}.txt"), "a") as file:
                        file.write(f"T1={T1:.2f}, T2={T2:.2f}: {avg_FF:.4f}\n")

                    # 打印当前阈值组合的结果
                    print(f"=== 阈值组合 T1={T1:.2f}, T2={T2:.2f} ===")
                    print(f"平均FF值: {avg_FF:.4f} ({len(avg_group_loss)}个航班)")

            # 可选：找到平均FF值最低的阈值组合
            best_result = min(results, key=lambda x: x['avg_FF'])
            print(f"\n\n最佳阈值组合: T1={best_result['T1']:.2f}, T2={best_result['T2']:.2f}")
            print(f"最低平均FF值: {best_result['avg_FF']:.4f}")
