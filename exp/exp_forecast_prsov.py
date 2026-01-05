from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import  adjust_learning_rate, EarlyStopping
import torch
import numpy as np
import os
import time
import torch.nn as nn
from iotdb.table_session import TableSession, TableSessionConfig
import pandas as pd
from transformers import AutoModelForCausalLM, AutoConfig

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_provider.data_loader_ac_test import FlightDataset_prsov_singleac
from torch.utils.data import DataLoader

class Exp_Forecast_prsov(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast_prsov, self).__init__(args)
        
        try:
            config = TableSessionConfig(
                node_urls=["10.254.43.34:6667"],
                # node_urls=["10.254.43.34:6667"],
                username="root",
                password="root",
                time_zone="UTC+8"
                # enable_compression=True,
            )
            session = TableSession(config)
        except Exception as e:
            config = TableSessionConfig(
                # node_urls=["10.254.43.34:6667"],
                node_urls=["10.254.43.34:6667"],
                username="root",
                password="root",
                time_zone="UTC+8"
                # enable_compression=True,
            )
            session = TableSession(config)
        session.execute_non_query_statement("USE b777")
        self.session = session
        # self.session = None

    def _build_model(self):
        if self.args.ddp:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            # for methods that do not use ddp (e.g. finetuning-based LLM4TS models)
            self.device = self.args.gpu

        model = self.model_dict[self.args.model].Model(self.args)
        if self.args.adaptation:
            model.load_state_dict(torch.load(self.args.pretrain_model_path))

        if(self.args.adapter in self.adapter_dict):
            model = self.adapter_dict[self.args.adapter].Model(self.args, model)
            print("loading adapter successfully:", self.args.adapter)

        model = self._move_model_to_device(model)
        
        return model
        
    def _move_model_to_device(self, model):
        if self.args.ddp:
            model = DDP(model.to(self.device), device_ids=[self.args.local_rank], find_unused_parameters=True)
        elif self.args.dp:
            model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            model = model.to(self.device)
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
        print("start training function")
        early_stopping = EarlyStopping(args=self.args, verbose=True)
        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            if not os.path.exists(path):
                os.makedirs(path)
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        
        data_set, data_loader = self._get_data(flag='train_normal')
        valdata_set, valdata_loader = self._get_data(flag='val_normal')
        # testdata_set, testdata_loader = self._get_data(flag="test_normal")
        # 为每个epoch创建新的DataLoader以确保随机性
        for epoch in range(self.args.train_epochs):
            # adjust_learning_rate(model_optim, epoch + 1, self.args)
            self.model.train()
            epoch_time = time.time()
            # 训练循环
            total_batches = len(data_loader)
            for batch_idx, (batch_x, flight_index) in enumerate(data_loader):
                model_optim.zero_grad()
                N, S, L, C = batch_x.shape
                batch_x = batch_x.to(self.device).reshape(-1, L, C)
                
                output = self.model(batch_x[:,:,0:self.args.input_channel])
                diff = torch.diff(batch_x[:, :, -1], dim=1)
                trend_labels = torch.full_like(diff, 0, dtype=torch.long)
                trend_labels[diff > 0] = 1
                # print("trend_labels shape:", trend_labels.shape, "output now shape:", output.shape)
                # trend_labels[diff < 0] = 2
                # 移除第2个维度
                # if(batch_idx<=5):
                #     print(output.shape, trend_labels.shape)
                # loss = nn.CrossEntropyLoss()(output[:,:,:-1], trend_labels)
                A, B = batch_x[:,:,-1].shape
                loss = nn.MSELoss()(output, batch_x[:,:,-1].reshape(A, B, 1)).cpu()
                loss.backward()
                model_optim.step()
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    # 记录损失
                    with open(os.path.join(path, 'loss.txt'), 'a') as f:
                        f.write(f"Epoch {epoch + 1}: Loss {loss.item()}\n")
                    # 打印进度
                    if batch_idx % 100 == 0 or batch_idx == total_batches - 1:
                        print(f"Epoch: {epoch + 1}/{self.args.train_epochs}, "
                              f"Batch: {batch_idx}/{total_batches}, "
                              f"Loss: {loss.item():.6f}, "
                              f"Time: {time.time() - epoch_time:.2f}s")

                # 定期保存检查点xh
                # if batch_idx % 1000 == 0:
                #     vali_loss = self.vali(valdata_loader)
                #     self.test_combined_single(setting, test=0)
                #     print(f"Epoch: {epoch + 1}/{self.args.train_epochs}, vali_loss: {vali_loss}")
                
            # 保存每个 Epoch 的检查点
            torch.save(self.model.state_dict(), os.path.join(path, f"checkpoint-EP{epoch}.pth"))
            vali_start = time.time()
            vali_loss_mse = self.vali(valdata_loader)
            
            # vali_start = time.time()
            # test_loss = self.vali(testdata_loader)
            # print(f"Validating Testset Time: {time.time() - vali_start:.2f}s")
            
            early_stopping(vali_loss_mse, self.model, path)
            
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print(f"Epoch: {epoch + 1}/{self.args.train_epochs}, vali_loss_mse: {vali_loss_mse}")
                print(f"Validating Validset Time: {time.time() - vali_start:.2f}s")
                
            # self.test_combined_single(setting, test=0)
            if early_stopping.early_stop:
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("Early stopping")
                break
            if self.args.cosine:# 使用 cosine 学习率优化策略
                scheduler.step()
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch+1, self.args)
            if self.args.ddp:
                data_loader.sampler.set_epoch(epoch+1)
        # 保存最终模型
        torch.save(self.model.state_dict(), os.path.join(path, "final_model.pth"))
        return self.model

    def vali(self, vali_loader):
        total_loss_mse = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (batch_x, flight_index) in enumerate(vali_loader):
                N, S, L, C = batch_x.shape
                batch_x = batch_x.to(self.device).reshape(-1, L, C)

                output = self.model(batch_x[:,:,0:self.args.input_channel])
                
                A, B = batch_x[:,:,-1].shape
                loss_mse = nn.MSELoss()(output, batch_x[:,:,-1].reshape(A, B, 1)).cpu()
                
                diff_x = torch.diff(batch_x[:, :, -1], dim=1)
                trend_labels_x = torch.full_like(diff_x, 0, dtype=torch.double)
                trend_labels_x[diff_x > 0] = 1

                diff_y = torch.diff(output, dim=1)
                trend_labels_y = torch.full_like(diff_y, 0, dtype=torch.double)
                trend_labels_y[diff_y > 0] = 1
                # trend_labels[diff < 0] = 2
                # 移除第2个维度
                total_loss_mse.append(loss_mse)
        total_loss_mse = np.average([t.cpu().numpy() for t in total_loss_mse])
        self.model.train()
        return total_loss_mse

    def test_single_ac(self, setting, test=1, adapter=''):

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
            node_urls=["10.254.43.34:6667"],
            username="root",
            password="root",
            time_zone="UTC+8",
            # enable_compression=True
        )
        session = TableSession(config)
        session.execute_non_query_statement("USE b777")
        df_result_all = pd.DataFrame()
        for tail_num in tail_list:
            data_set = FlightDataset_prsov_singleac(
                session,
                tail_list,
                para_list_l,
                para_list_r,
                time_start,
                time_end,
                self.args,
                tail_num=tail_num
            )
            data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag)

            if test:
                best_model_path = self.args.test_file_name
                model_path = os.path.join(self.args.checkpoints, setting, best_model_path)
                print(f"loading model from {model_path}")
                # device =(
                #     torch.device("cuda:{}".format(self.device))
                #     if torch.cuda.is_available()
                #     else torch.device("cpu")
                # )
                self.model.load_state_dict(torch.load(model_path))

            self.model.eval()
            total_loss = []
            total_time_index = []
            with torch.no_grad():
                for batch_idx, (batch_x, time_index) in enumerate(data_loader):
                    N, S, L, C = batch_x.shape
                    batch_x = batch_x.to(self.device).reshape(-1, L, C)
                    output = self.model(batch_x[:, :, 0:self.args.input_channel])
                    diff = torch.diff(batch_x[:, :, -1], dim=1)
                    trend_labels = torch.full_like(diff, 0, dtype=torch.long)
                    trend_labels[diff > 0] = 1
                    # loss = nn.CrossEntropyLoss(reduction='none')(output[:, :, :-1], trend_labels).cpu()
                    # total_loss.append(loss.reshape(N, S, (L-1), 1))
                    A, B = batch_x[:,:,-1].shape
                    loss = nn.MSELoss(reduction='none')(output[:,:,:-1], batch_x[:,:,-1].reshape(A, B, 1)).cpu()
                    total_loss.append(loss.reshape(N, S, L, 1))
                    total_time_index.append(time_index.flatten().detach().cpu())

            result = torch.cat(total_loss, dim=0)
            time_index_all = torch.cat(total_time_index, dim=0)

            # 动态计算reshape尺寸
            avg_loss_np1 = torch.mean(result, dim=2).numpy()
            # TODO 替换最优T1阈值
            exceed_normal = (avg_loss_np1 > self.args.threshold_T1).mean(axis=1).flatten()

            """1-模型预警结果"""
            df1 = pd.DataFrame({
                'time': time_index_all,
                'average_loss': exceed_normal
            })
            # 将df1的time列转换为datetime类型
            df1['time'] = pd.to_datetime(df1['time'])

            # 添加报警逻辑
            # 添加warning列并初始化为0
            df1['warning'] = 0
            df1['aircraft/tail'] = tail_num
            # 计算累积和用于高效窗口计算
            cumsum = np.concatenate(([0], np.cumsum(df1['average_loss'])))
            # 设置窗口参数，88次作动为一个报警窗口
            window_size = 88
            step_size = 1
            # 获取有效的窗口起始索引
            start_indices = np.arange(0, len(df1) - window_size + 1, step_size)
            # 遍历每个窗口
            for start in start_indices:
                end = start + window_size
                # 计算窗口内值的总和
                window_sum = cumsum[end] - cumsum[start]
                # 计算平均值并判断条件
                # TODO 替换最优T2阈值
                if window_sum / 88 > self.args.threshold_T2:
                    # 设置窗口末尾行的warning为1
                    warning_index = start + window_size - 1
                    df1.loc[warning_index, 'warning'] = 1

            # 获取warning为1的行
            warning_rows = df1[df1['warning'] == 1]

            """2-飞机故障结果"""
            df2 = data_set.fault_table_cfd
            # 将df2的time列转换为datetime类型
            df2['time'] = pd.to_datetime(df2['time'])

            """2-prsov维修记录"""
            df3 = data_set.fault_table_turefault_r
            df3['time'] = pd.to_datetime(df3['time'])

            # 按行拼接waring,df2,df3
            df_result = pd.concat([warning_rows, df2, df3], ignore_index=True)
            # 对时间进行排序
            df_result.sort_values(by='time', inplace=True)
            df_result['endtime'] = pd.to_datetime(df_result['endtime'])
            df_result_all = pd.concat([df_result_all, df_result], ignore_index=True)
        df_result_all.to_csv(f'resultr.csv', index=False)

    def test_combined_single(self, setting, test=1):
        # 加载模型
        if test:
            best_model_path = self.args.test_file_name
            model_path = os.path.join(self.args.checkpoints, setting, best_model_path)
            print(f"loading model from {model_path}")
            # device =(
            #     torch.device("cuda:{}".format(self.device))
            #     if torch.cuda.is_available()
            #     else torch.device("cpu")
            # )
            self.model.load_state_dict(torch.load(model_path))

        self.model.eval()
        
        test_start = time.time()
        
        # 创建结果目录
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(os.path.join(self.args.checkpoints, setting), exist_ok=True)
        
        # 加载正常和异常数据
        def process_data(flag):
            dataset, dataloader = self._get_data(flag=flag)
            total_loss = []
            flight_index_all = []
            
            with torch.no_grad():
                for batch_idx, (batch_x, flight_index) in enumerate(dataloader):
                    N, S, L, C = batch_x.shape # N是作动的个数，S是滑窗采样个数，L是样本长度，C是特征数
                    # flight_index = flight_index[:, 0:1] # 每个作动的S个采样，属于同一个组
                    batch_x = batch_x.to(self.device).reshape(-1, self.args.seq_len, 3)  # peizhongyi. 这里batch中的每个样本，是一个作动的若干个采样；在测试中，它们应该一起计算一个损失函数，刻画样本级作动的异常程度

                    output = self.model(batch_x[:, :, 0:self.args.input_channel])
                    diff = torch.diff(batch_x[:, :, -1], dim=1)
                    trend_labels = torch.full_like(diff, 0, dtype=torch.long)
                    trend_labels[diff > 0] = 1
                    # trend_labels[diff < 0] = 2

                    # loss = nn.CrossEntropyLoss(reduction='none')(output[:, :, :-1], trend_labels).cpu()
                    A, B = batch_x[:,:,-1].shape
                    loss = nn.MSELoss(reduction='none')(output[:,:,:-1], batch_x[:,:,-1].reshape(A, B, 1)).cpu()
                    # loss = loss.reshape(N, S, L-1, 1)  # 恢复batch维度
                    loss = loss.reshape(N, S, L, 1)
                    total_loss.append(loss)
                    flight_index_all.append(flight_index.flatten().detach().cpu()) # 一个batch的数据，被分到一个组

            # 按行拼接all_losses中的所有损失
            result = torch.cat(total_loss, dim=0)
            flight_index_all = torch.cat(flight_index_all, dim=0)
            return result, flight_index_all

        test_normal_data_path = f"cache/{setting}_test_normal_data.npy"
        test_abnormal_data_path = f"cache/{setting}_test_abnormal_data.npy"
        test_normal_index_path = f"cache/{setting}_test_normal_index.npy"
        test_abnormal_index_path = f"cache/{setting}_test_abnormal_index.npy"
        if os.path.exists(test_normal_data_path) and os.path.exists(test_abnormal_data_path):
            avg_operation_loss_normal = torch.from_numpy(np.load(test_normal_data_path))
            avg_operation_loss_abnormal = torch.from_numpy(np.load(test_abnormal_data_path))
            index_normal = torch.from_numpy(np.load(test_normal_index_path))
            index_abnormal = torch.from_numpy(np.load(test_abnormal_index_path))
        else:
            avg_operation_loss_normal, index_normal = process_data('test_normal')
            avg_operation_loss_abnormal, index_abnormal = process_data('test_abnormal')
            # 保存索引到文件
            # np.save(test_normal_index_path, index_normal.numpy())
            # np.save(test_abnormal_index_path, index_abnormal.numpy())
            # # 保存结果到文件
            # np.save(test_normal_data_path, avg_operation_loss_normal.numpy())
            # np.save(test_abnormal_data_path, avg_operation_loss_abnormal.numpy())

        # 采用均值计算
        def sample_level_loss_mean(loss_tensor):
            S, L, _ = loss_tensor.shape
            loss_tensor = torch.mean(loss_tensor, dim=1, keepdim=True).flatten()
            return loss_tensor

        # 动态计算reshape尺寸
        # avg_loss_np1 = torch.mean(avg_operation_loss_normal, dim=1).numpy()
        avg_loss_np1 = torch.stack([sample_level_loss_mean(g) for g in avg_operation_loss_normal]).flatten().numpy()
        # print("avg_loss_np1:", avg_loss_np1.shape)
        # 转换索引为 NumPy 数组
        index_np1 = index_normal.numpy()
        # 创建 DataFrame
        df1 = pd.DataFrame({
            'flight_group_id': index_np1,
            'average_loss': avg_loss_np1
        })
        # avg_loss_np2 = torch.mean(avg_operation_loss_abnormal, dim=1).numpy()
        avg_loss_np2 = torch.stack([sample_level_loss_mean(g) for g in avg_operation_loss_abnormal]).flatten().numpy()
        # print("avg_loss_np2:", avg_loss_np2.shape)
        index_np2 = index_abnormal.numpy()
        df2 = pd.DataFrame({
            'flight_group_id': index_np2,
            'average_loss': avg_loss_np2})

        def vectorized_sliding_window(arr, window_size, step):
            # todo: Use torch.unfold
            """使用NumPy向量化实现滑动窗口"""
            if len(arr) < window_size:
                return np.empty((0, window_size))
            num_windows = (len(arr) - window_size) // step + 1
            idx = np.arange(window_size) + np.arange(num_windows)[:, None] * (step)
            return arr[idx]

        # 预计算所有窗口数据
        def precompute_windows(df, window_size, step):
            group_windows_dict = {}
            for group_id in np.sort(df['flight_group_id'].unique()):
                group_data = df.loc[df['flight_group_id'] == group_id, 'average_loss'].values.reshape(-1, 24) # 24对应的是sql_len = 40,代表单次作动的总窗口数量。
                windows = vectorized_sliding_window(group_data, window_size, step)
                if windows.size > 0:
                    group_windows_dict[group_id] = windows
            return group_windows_dict

        # 使用您提供的统一检测函数
        def detect_one_group(windows, T1, T2):
            """一个windows代表：
            飞机在一段时间内的正常飞行的数据，或者故障前6个月的数据
            第一个维度是：检测次数，对应预警系统的检测次数，
            第二个维度是：单次检测的窗口长度，也就是判断是否发出预警的窗口
            第三维度是一次作动所有滑窗的的loss，一个值对应的是一个滑窗的loss均值，t1值就是在第三维度上对单次作动超过t1值的比例"""
            if len(windows) == 0:
                return False
            # 每k个窗口检测一次，有一次超过阈值就报警
            # k = 5
            # for i in range(0, len(windows), k):
            #     if i + k <= len(windows):
            #         group_windows = windows[i:i + k]
            #     else:
            #         group_windows = windows[i:]
            exceed_rates = np.mean(windows > T1, axis=2).mean(axis=1)
            # 检测当前组的所有窗口是否超过阈值
            exceed_mask = exceed_rates > T2
            # 获取为true的个数
            num_exceed = exceed_mask.sum()
            return num_exceed

        # 预计算正常/异常数据的窗口字典
        window_size = 88
        step = 1
        normal_windows_dict = precompute_windows(df1, window_size=window_size, step=step)
        abnormal_windows_dict = precompute_windows(df2, window_size=window_size, step=step)

        # 定义阈值搜索范围
        T1_range = np.arange(0.01, 1, 0.01)
        T2_range = np.arange(0.01, 1, 0.01)

        min_ratio = float('inf')
        best_params = (None, None, None, None)

        # 向量化搜索最优阈值
        for T1 in T1_range:
            # 计算整个T2范围的指标
            for T2 in T2_range:
                # 预计算正常分组数据（每5个窗口一组）
                total_normal_groups = 0
                total_abnormal_groups = 0
                normal_group_detections = 0
                # 处理正常数据：按5个窗口分组
                for group_id, windows in normal_windows_dict.items():
                    if len(windows) > 0:  # 只要有窗口就处理
                        # 使用统一的检测函数
                        detected = detect_one_group(windows, T1, T2)
                        normal_group_detections += detected
                        total_normal_groups += windows.shape[0]

                    # 处理异常数据：每组已经是最后5个窗口
                abnormal_group_detections = 0
                for group_id, windows in abnormal_windows_dict.items():
                    if len(windows) > 0:
                        # 使用统一的检测函数
                        detected = detect_one_group(windows, T1, T2)
                        abnormal_group_detections += detected
                        total_abnormal_groups += windows.shape[0]

                # 计算HF_rate（正常数据误报率）
                false_alarms = normal_group_detections
                HF_rate = false_alarms / total_normal_groups if total_normal_groups > 0 else 0
                # print(total_normal_groups, HF_rate)

                # 计算FF_rate（异常数据检出率）
                true_detections = abnormal_group_detections
                FF_rate = true_detections / total_abnormal_groups if total_abnormal_groups > 0 else 0
                # print(len(abnormal_group_detections), FF_rate)

                # 查找满足约束的最优解
                if HF_rate > 0.0000001 and FF_rate > 0.4:
                    ratio = HF_rate / FF_rate
                    if ratio < min_ratio:
                        min_ratio = ratio
                        best_params = (T1, T2, HF_rate, FF_rate)
                print(f"T1 = {T1:.3f}, T2 = {T2:.3f}, avg_HF = {HF_rate:.4f}, avg_FF = {FF_rate:.4f}")

        # 输出结果
        best_T1, best_T2, best_avg_HF, best_avg_FF = best_params
        if best_T1 is not None:
            print(f"\n\n=== 最优解 ===")
            print(f"T1 = {best_T1:.3f}, T2 = {best_T2:.3f}")
            print(f"avg_HF = {best_avg_HF:.4f} (0.001 < ), avg_FF = {best_avg_FF:.4f} (>0.4)")
            print(f"最小比例值: {min_ratio:.4f}")

            # 保存结果到文件
            result_str = (f"Best T1: {best_T1}\nBest T2: {best_T2}\n"
                          f"avg_HF: {best_avg_HF}\navg_FF: {best_avg_FF}\nRatio: {min_ratio}")
            result_str += f"\nDetection Strategy: Hierarchical (Last 3 windows + Full sequence)"
            result_str += f"\nNormal Groups: {len(normal_windows_dict)}, Normal 5-window Groups: {total_normal_groups}"
            result_str += f"\nAbnormal Groups: {len(abnormal_windows_dict)}"

            sample_results = []
            for i, (group_id, windows) in enumerate(abnormal_windows_dict.items()):
                if i < 5:  # 保存前5个组的检测结果
                    detected = detect_one_group(windows, best_T1, best_T2)
                    sample_results.append(f"Group {group_id}: Windows={len(windows)}, Detected={detected}")

            result_str += "\nSample Detection:\n" + "\n".join(sample_results)

            with open(os.path.join(folder_path, f"combined_result_{setting}.txt"), "w") as f:
                f.write(result_str)
        else:
            print("未找到满足条件的阈值组合：0.001 < avg_HF and avg_FF > 0.4")

        return best_T1, best_T2