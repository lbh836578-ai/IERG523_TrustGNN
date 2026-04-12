"""
数据模拟器
生成带有故障注入的模拟传感器数据
"""
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from torch.utils.data import Dataset, DataLoader
import random

from config import SystemConfig, SensorMeta, SensorType


class FaultType(Enum):
    """故障类型"""
    NONE = 0
    STUCK_AT = 1          # 卡死在某个值
    DRIFT = 2             # 漂移
    NOISE = 3             # 噪声增大
    SPIKE = 4             # 尖峰
    BIAS = 5              # 偏置
    MISSING = 6           # 数据丢失（NaN）
    RANDOM = 7            # 随机值


@dataclass
class FaultInfo:
    """故障信息"""
    sensor_id: str
    sensor_idx: int
    fault_type: FaultType
    start_time: int
    end_time: int
    parameters: Dict


@dataclass
class SensorDataTensor:
    """传感器数据张量"""
    X: torch.Tensor           # (N, T, F)
    timestamps: np.ndarray    # (T,)
    sensor_ids: List[str]


@dataclass
class GroundTruth:
    """训练标签"""
    clean_data: torch.Tensor           # (N, T, F) 干净数据
    fusion_target: torch.Tensor        # (T, output_F) 融合目标
    fault_mask: torch.Tensor           # (N, T) 故障掩码 1=故障
    fault_types: torch.Tensor          # (N, T) 故障类型编码
    credibility_target: torch.Tensor   # (N, T) 可信度目标 1=可信


class AgriculturalDataSimulator:
    """农业数据模拟器"""
    
    def __init__(self, config: SystemConfig, seed: int = 42):
        self.config = config
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        self.sensor_ids = list(config.sensors.keys())
        self.num_sensors = len(self.sensor_ids)
        
    def generate_clean_data(
        self, 
        num_samples: int = 1,
        window_size: int = None
    ) -> List[SensorDataTensor]:
        """生成干净的传感器数据"""
        if window_size is None:
            window_size = self.config.window_size
            
        samples = []
        
        for _ in range(num_samples):
            # 生成基础环境参数（真实值）
            base_temp = np.random.uniform(18, 30)          # 基础温度
            base_humidity = np.random.uniform(40, 75)      # 基础湿度
            base_soil = np.random.uniform(25, 55)          # 基础土壤湿度
            base_light = np.random.uniform(5000, 40000)    # 基础光照
            
            # 时间序列
            t = np.arange(window_size)
            
            # 生成时间变化模式（日变化）
            hour_cycle = 2 * np.pi * t / window_size
            
            # 温度：日变化 + 噪声
            temp_variation = 3 * np.sin(hour_cycle) + 0.5 * np.random.randn(window_size)
            
            # 湿度：与温度负相关
            humidity_variation = -2 * np.sin(hour_cycle) + 1.0 * np.random.randn(window_size)
            
            # 土壤湿度：缓慢变化
            soil_variation = 0.5 * np.cumsum(np.random.randn(window_size) * 0.1)
            soil_variation = soil_variation - soil_variation.mean()
            
            # 光照：日变化明显
            light_variation = 15000 * np.maximum(0, np.sin(hour_cycle)) + 500 * np.random.randn(window_size)
            
            # 为每个传感器生成数据
            data = np.zeros((self.num_sensors, window_size, 1))
            
            for i, sensor_id in enumerate(self.sensor_ids):
                sensor = self.config.sensors[sensor_id]
                
                if sensor.sensor_type == SensorType.TEMPERATURE:
                    # 温度传感器：基础值 + 时变 + 传感器特定噪声
                    base = base_temp + temp_variation
                    noise = np.random.randn(window_size) * sensor.noise_std
                    # 添加空间差异
                    spatial_offset = np.random.uniform(-1, 1)
                    data[i, :, 0] = base + noise + spatial_offset
                    
                elif sensor.sensor_type == SensorType.HUMIDITY:
                    base = base_humidity + humidity_variation
                    noise = np.random.randn(window_size) * sensor.noise_std
                    spatial_offset = np.random.uniform(-2, 2)
                    data[i, :, 0] = np.clip(base + noise + spatial_offset, 0, 100)
                    
                elif sensor.sensor_type == SensorType.SOIL_MOISTURE:
                    base = base_soil + soil_variation
                    noise = np.random.randn(window_size) * sensor.noise_std
                    spatial_offset = np.random.uniform(-5, 5)
                    data[i, :, 0] = np.clip(base + noise + spatial_offset, 0, 100)
                    
                elif sensor.sensor_type == SensorType.LIGHT:
                    base = base_light + light_variation
                    noise = np.random.randn(window_size) * sensor.noise_std
                    data[i, :, 0] = np.clip(base + noise, 0, 65535)
            
            # 创建时间戳
            timestamps = np.arange(window_size)
            
            samples.append(SensorDataTensor(
                X=torch.FloatTensor(data),
                timestamps=timestamps,
                sensor_ids=self.sensor_ids.copy()
            ))
            
        return samples
    
    def inject_faults(
        self,
        data: SensorDataTensor,
        fault_ratio: float = 0.3,
        fault_duration_ratio: float = 0.5
    ) -> Tuple[SensorDataTensor, List[FaultInfo], torch.Tensor]:
        """
        注入故障
        
        Args:
            data: 干净数据
            fault_ratio: 故障传感器比例
            fault_duration_ratio: 故障持续时间比例
            
        Returns:
            faulty_data: 注入故障后的数据
            fault_infos: 故障信息列表
            fault_mask: 故障掩码 (N, T)
        """
        X = data.X.clone()
        N, T, F = X.shape
        
        fault_mask = torch.zeros(N, T)
        fault_infos = []
        
        # 随机选择故障传感器
        num_faulty = max(1, int(N * fault_ratio))
        faulty_indices = random.sample(range(N), num_faulty)
        
        # 可用的故障类型
        fault_types = [
            FaultType.STUCK_AT,
            FaultType.DRIFT,
            FaultType.NOISE,
            FaultType.SPIKE,
            FaultType.BIAS,
            FaultType.RANDOM
        ]
        
        for idx in faulty_indices:
            sensor_id = self.sensor_ids[idx]
            sensor = self.config.sensors[sensor_id]
            
            # 随机选择故障类型
            fault_type = random.choice(fault_types)
            
            # 随机选择故障时间范围
            fault_duration = int(T * fault_duration_ratio * random.uniform(0.5, 1.0))
            start_time = random.randint(0, T - fault_duration)
            end_time = start_time + fault_duration
            
            # 注入故障
            params = {}
            
            if fault_type == FaultType.STUCK_AT:
                # 卡死在某个值
                stuck_value = X[idx, start_time, 0].item()
                X[idx, start_time:end_time, 0] = stuck_value
                params['stuck_value'] = stuck_value
                
            elif fault_type == FaultType.DRIFT:
                # 漂移
                drift_rate = random.uniform(0.1, 0.5) * random.choice([-1, 1])
                drift = torch.arange(end_time - start_time).float() * drift_rate
                X[idx, start_time:end_time, 0] += drift
                params['drift_rate'] = drift_rate
                
            elif fault_type == FaultType.NOISE:
                # 噪声增大
                noise_multiplier = random.uniform(3, 10)
                extra_noise = torch.randn(end_time - start_time) * sensor.noise_std * noise_multiplier
                X[idx, start_time:end_time, 0] += extra_noise
                params['noise_multiplier'] = noise_multiplier
                
            elif fault_type == FaultType.SPIKE:
                # 尖峰
                num_spikes = random.randint(3, 10)
                spike_times = random.sample(range(start_time, end_time), min(num_spikes, end_time - start_time))
                for st in spike_times:
                    spike_magnitude = random.uniform(5, 20) * random.choice([-1, 1])
                    X[idx, st, 0] += spike_magnitude
                params['num_spikes'] = num_spikes
                
            elif fault_type == FaultType.BIAS:
                # 偏置
                bias = random.uniform(5, 15) * random.choice([-1, 1])
                X[idx, start_time:end_time, 0] += bias
                params['bias'] = bias
                
            elif fault_type == FaultType.RANDOM:
                # 随机值
                random_values = torch.rand(end_time - start_time) * (sensor.max_value - sensor.min_value) + sensor.min_value
                X[idx, start_time:end_time, 0] = random_values
                
            # 记录故障掩码
            fault_mask[idx, start_time:end_time] = 1.0
            
            # 记录故障信息
            fault_infos.append(FaultInfo(
                sensor_id=sensor_id,
                sensor_idx=idx,
                fault_type=fault_type,
                start_time=start_time,
                end_time=end_time,
                parameters=params
            ))
        
        faulty_data = SensorDataTensor(
            X=X,
            timestamps=data.timestamps,
            sensor_ids=data.sensor_ids
        )
        
        return faulty_data, fault_infos, fault_mask
    
    def compute_ground_truth(
        self,
        clean_data: SensorDataTensor,
        fault_mask: torch.Tensor
    ) -> GroundTruth:
        """计算真实标签"""
        N, T, F = clean_data.X.shape
        
        # 融合目标：按传感器类型分组平均
        fusion_target = torch.zeros(T, 4)  # 4个输出通道
        
        for i, sensor_id in enumerate(self.sensor_ids):
            sensor = self.config.sensors[sensor_id]
            group = sensor.fusion_group
            
            # 加入到对应通道
            fusion_target[:, group] += clean_data.X[i, :, 0]
        
        # 归一化（按每个类型的传感器数量）
        group_counts = [0, 0, 0, 0]
        for sensor_id in self.sensor_ids:
            sensor = self.config.sensors[sensor_id]
            group_counts[sensor.fusion_group] += 1
            
        for g in range(4):
            if group_counts[g] > 0:
                fusion_target[:, g] /= group_counts[g]
        
        # 可信度目标：无故障=1，有故障=0
        credibility_target = 1.0 - fault_mask
        
        return GroundTruth(
            clean_data=clean_data.X,
            fusion_target=fusion_target,
            fault_mask=fault_mask,
            fault_types=fault_mask,
            credibility_target=credibility_target
        )
    
    def generate_dataset(
        self,
        num_samples: int = 100,
        inject_faults: bool = True,
        fault_ratio: float = 0.3
    ) -> Tuple[List[SensorDataTensor], List[GroundTruth], List[List[FaultInfo]]]:
        """
        生成完整数据集
        
        Returns:
            data_list: 传感器数据列表（可能有故障）
            gt_list: 真实标签列表
            fault_list: 故障信息列表
        """
        # 生成干净数据
        clean_samples = self.generate_clean_data(num_samples)
        
        data_list = []
        gt_list = []
        fault_list = []
        
        for clean_data in clean_samples:
            if inject_faults and random.random() < 0.8:  # 80%的样本有故障
                faulty_data, faults, fault_mask = self.inject_faults(
                    clean_data, fault_ratio=fault_ratio
                )
                gt = self.compute_ground_truth(clean_data, fault_mask)
                data_list.append(faulty_data)
                fault_list.append(faults)
            else:
                fault_mask = torch.zeros(clean_data.X.shape[0], clean_data.X.shape[1])
                gt = self.compute_ground_truth(clean_data, fault_mask)
                data_list.append(clean_data)
                fault_list.append([])
                
            gt_list.append(gt)
            
        return data_list, gt_list, fault_list
    
    def create_dataloader(
        self,
        data_list: List[SensorDataTensor],
        gt_list: List[GroundTruth],
        batch_size: int = 32,
        shuffle: bool = True
    ) -> DataLoader:
        """创建 DataLoader"""
        dataset = SensorDataset(data_list, gt_list)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )
    
    @staticmethod
    def collate_fn(batch):
        """批处理函数"""
        X = torch.stack([item[0] for item in batch])
        fusion_target = torch.stack([item[1] for item in batch])
        fault_mask = torch.stack([item[2] for item in batch])
        credibility_target = torch.stack([item[3] for item in batch])
        
        return X, fusion_target, fault_mask, credibility_target


class SensorDataset(Dataset):
    """传感器数据集"""
    
    def __init__(
        self, 
        data_list: List[SensorDataTensor], 
        gt_list: List[GroundTruth]
    ):
        self.data_list = data_list
        self.gt_list = gt_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        gt = self.gt_list[idx]
        
        return (
            data.X,                    # (N, T, F)
            gt.fusion_target,          # (T, output_F)
            gt.fault_mask,             # (N, T)
            gt.credibility_target      # (N, T)
        )